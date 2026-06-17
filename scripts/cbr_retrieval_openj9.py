"""CBR de RECUPERACIÓN (Case-Based Reasoning real) sobre OpenJ9.

Reemplaza el CBR-clasificador de TriagerX (cabeza softmax sobre devs) por un
recuperador basado en casos: embebe el bug, recupera los k bugs pasados más
similares y **vota a sus resolvedores** (`owner`), ponderando por similitud.

A diferencia del IBR (que usa el grafo de interacciones tipadas con un encoder
CONGELADO), aquí:
  - el voto es por el RESOLVEDOR del caso similar (la etiqueta), no por interacciones;
  - el encoder se puede AFINAR con pérdida contrastiva para que bugs del mismo dev
    queden cerca (--finetune; por defecto zero-shot para el baseline).

Scoring: score(dev) = Σ_{j ∈ topk, owner(j)=dev} sim(q, j)^τ · decay(Δt)
  - sim = coseno (embeddings normalizados); negativos recortados a 0.
  - τ agudiza la contribución de los vecinos más cercanos.
  - decay = exp(-λ·Δaños) si hay meta temporal (opcional, --decay).

Etiquetas = sorted(owners) (idéntico a train_openj9_cbr / eval_openj9_ibr → fusionable).

Uso:
    # baseline zero-shot con barrido (k, τ):
    uv run python scripts/cbr_retrieval_openj9.py \
        --train-csv artifacts/openj9/openj9_train_50.csv \
        --test-csv  artifacts/openj9/openj9_test_50.csv --sweep
    # punto único:
    uv run python scripts/cbr_retrieval_openj9.py ... --top-k 20 --tau 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from triager_omega.config import settings
from triager_omega.modules.aggregator import rank_metrics


def label_encoder(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    owners = sorted(set(train["owner"].dropna()) | set(test["owner"].dropna()))
    return {o: i for i, o in enumerate(owners)}


def vote_scores(sims: np.ndarray, train_cls: np.ndarray, num_classes: int,
                top_k: int, tau: float, decay: np.ndarray | None = None,
                vote: str = "sum", dev_freq: np.ndarray | None = None) -> np.ndarray:
    """Matriz de scores [Nte, C] por voto kNN ponderado por similitud^τ (· decay).

    `vote` controla cómo se agrega el peso de los vecinos de un mismo dev (mejora 1,
    ataca el sesgo de cola larga: un dev prolífico aparece en muchos vecinos):
      - sum  : Σ w           (baseline, favorece a devs frecuentes)
      - mean : Σ w / nº de vecinos de ese dev en el top-k
      - max  : max w         (vale el vecino más cercano, ignora cantidad)
      - idf  : Σ w · log(Ntr / freq(dev))   (penaliza devs globalmente frecuentes)
      - rank : Σ 1/(rango+1) en vez de sim^τ (peso por posición, no por magnitud)
    """
    n = sims.shape[0]
    scores = np.zeros((n, num_classes), dtype=np.float32)
    counts = np.zeros((n, num_classes), dtype=np.float32)
    w = np.clip(sims, 0.0, None) ** tau
    if decay is not None:
        w = w * decay[None, :]
    top_k = min(top_k, sims.shape[1])
    # top-k vecinos por fila, ORDENADOS (para el modo rank)
    part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
    for i in range(n):
        cols = part[i]
        order = cols[np.argsort(-sims[i, cols])]  # vecinos ordenados por similitud desc
        wi = w[i, order]
        if vote == "rank":
            wi = 1.0 / (np.arange(len(order)) + 1.0)
        cls = train_cls[order]
        if vote == "max":
            np.maximum.at(scores[i], cls, wi)
        else:
            np.add.at(scores[i], cls, wi)
            np.add.at(counts[i], cls, 1.0)
    if vote == "mean":
        scores = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)
    elif vote == "idf" and dev_freq is not None:
        idf = np.log(dev_freq.sum() / np.clip(dev_freq, 1, None)).astype(np.float32)
        scores = scores * idf[None, :]
    return scores


def bm25_sims(train_texts: list[str], test_texts: list[str]) -> np.ndarray:
    """Matriz BM25 [Nte, Ntr] (similitud léxica exacta; complementa la semántica).

    Tokeniza conservando identificadores técnicos (com.ibm.j9, códigos de error,
    nombres de clase) que la similitud densa difumina y que importan en la cola larga.
    """
    import re

    from rank_bm25 import BM25Okapi

    tok = lambda s: re.findall(r"[A-Za-z0-9_.:#]+", s.lower())
    corpus = [tok(t) for t in train_texts]
    bm25 = BM25Okapi(corpus)
    out = np.zeros((len(test_texts), len(train_texts)), dtype=np.float32)
    for i, q in enumerate(test_texts):
        out[i] = bm25.get_scores(tok(q))
    return out


def rrf_fuse(dense: np.ndarray, lexical: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion de dos matrices de similitud [Nte, Ntr].

    RRF(doc) = 1/(k+rango_denso) + 1/(k+rango_léxico). Robusto a escalas distintas
    (BM25 vs coseno): solo usa el orden, no la magnitud.
    """
    fused = np.zeros_like(dense, dtype=np.float32)
    for mat in (dense, lexical):
        ranks = np.empty_like(mat, dtype=np.int64)
        order = np.argsort(-mat, axis=1)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(mat.shape[1])[None, :]
        fused += 1.0 / (k + ranks + 1.0)
    return fused


def rerank_cross_encoder(sims: np.ndarray, train_texts: list[str], test_texts: list[str],
                         model_name: str, top_n: int, device: str, batch: int = 32) -> np.ndarray:
    """Re-ordena los top-N candidatos densos con un cross-encoder (mejora 2).

    El bi-encoder (MPNet) da recall; el cross-encoder mira query+candidato JUNTOS y
    da precisión fina en el top → sube el Top-1 sin entrenar nada. Devuelve una nueva
    matriz [Nte, Ntr] con sigmoid(score CE) en los candidatos y 0 en el resto.

    El modelo es chico (MiniLM-L6); en MPS la gestión de memoria revienta con muchos
    pares → se corre en CPU (rápido igual). En CUDA usa la GPU.
    """
    from sentence_transformers import CrossEncoder

    ce_device = "cpu" if device == "mps" else device  # MPS OOM con muchos pares
    ce = CrossEncoder(model_name, device=ce_device, max_length=256)
    # Los bug reports traen logs/stack traces enormes; el tokenizer procesa el string
    # COMPLETO antes de truncar a 256 tokens → thrashing. Recortamos a ~2000 chars
    # (cubre de sobra los 256 tokens útiles) antes de armar los pares.
    cap = 2000
    tr_cap = [t[:cap] for t in train_texts]
    te_cap = [t[:cap] for t in test_texts]
    top_n = min(top_n, sims.shape[1])
    cand = np.argpartition(-sims, kth=top_n - 1, axis=1)[:, :top_n]
    out = np.zeros_like(sims, dtype=np.float32)
    pairs, locs = [], []
    for i in range(sims.shape[0]):
        for j in cand[i]:
            pairs.append([te_cap[i], tr_cap[j]])
            locs.append((i, j))
    logger.info("Cross-encoder re-rank ({}): {} pares ({} queries × top-{})...",
                ce_device, len(pairs), sims.shape[0], top_n)
    raw = ce.predict(pairs, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)
    ssig = 1.0 / (1.0 + np.exp(-raw))  # logits → [0,1]
    for (i, j), s in zip(locs, ssig):
        out[i, j] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--decay", action="store_true", help="pondera por exp(-λ·Δaños) usando meta temporal")
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--meta", default=None, help="parquet de created_at (para --decay)")
    p.add_argument("--finetune", action="store_true", help="afina el encoder con pérdida contrastiva (paso 2)")
    p.add_argument("--loss", choices=["mnrl", "triplet"], default="mnrl",
                   help="mnrl=MultipleNegativesRanking (pares mismo-dev, estable); triplet=BatchAllTriplet (colapsa en datos chicos)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ft-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5, help="LR del fine-tuning (suave para no olvidar)")
    p.add_argument("--max-seq-len", type=int, default=256,
                   help="cap de tokens del encoder (memoria∝seq²; baja el thrashing en 8GB)")
    p.add_argument("--sweep", action="store_true", help="barre (k, τ) sobre los mismos embeddings")
    p.add_argument("--vote", choices=["sum", "mean", "max", "idf", "rank"], default="sum",
                   help="agregación del voto por dev (mejora 1, anti-cola-larga)")
    p.add_argument("--vote-sweep", action="store_true", help="barre los 5 modos de --vote a (k, τ) fijos")
    p.add_argument("--hybrid", action="store_true",
                   help="fusiona MPNet + BM25 por RRF (mejora 3, recupera por tokens técnicos exactos)")
    p.add_argument("--rrf-k", type=int, default=60, help="constante de RRF")
    p.add_argument("--rerank", action="store_true",
                   help="re-ordena top-N con cross-encoder (mejora 2, precisión fina del top)")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--rerank-n", type=int, default=50, help="nº de candidatos densos a re-ordenar")
    p.add_argument("--rerank-batch", type=int, default=32, help="batch del cross-encoder (baja si OOM)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = settings
    from pathlib import Path
    train = pd.read_csv(Path(args.train_csv) if args.train_csv else cfg.openj9_train_csv).drop_duplicates("issue_number")
    test = pd.read_csv(Path(args.test_csv) if args.test_csv else cfg.openj9_test_csv).drop_duplicates("issue_number")
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)
    le = label_encoder(train, test)
    C = len(le)
    train_cls = train["owner"].map(le).to_numpy().astype(int)
    true_idx = test["owner"].map(le).to_numpy().astype(int)
    dev_freq = np.bincount(train_cls, minlength=C).astype(np.float32)  # para --vote idf
    logger.info("CBR-recuperación OpenJ9 | train={} test={} clases={}", len(train), len(test), C)

    device = "cpu" if args.cpu else cfg.torch_device
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_len  # cap consistente para fine-tuning e inferencia

    if args.finetune:
        _finetune(model, train, train_cls, C, args, device)

    enc = lambda txts: model.encode(txts, batch_size=64, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
    logger.info("Embebiendo train + test con {}...", args.model)
    emb_tr = enc(train["text"].tolist())
    emb_te = enc(test["text"].tolist())
    sims = emb_te @ emb_tr.T  # coseno [Nte, Ntr]

    tr_txt, te_txt = train["text"].tolist(), test["text"].tolist()
    if args.hybrid:  # mejora 3: MPNet + BM25 por RRF
        logger.info("Híbrido BM25+MPNet (RRF k={})...", args.rrf_k)
        sims = rrf_fuse(sims, bm25_sims(tr_txt, te_txt), k=args.rrf_k)
    if args.rerank:  # mejora 2: cross-encoder sobre los top-N candidatos
        import gc
        model.to("cpu"); del model  # libera la GPU para el cross-encoder (8GB)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sims = rerank_cross_encoder(sims, tr_txt, te_txt, args.rerank_model, args.rerank_n,
                                    device, args.rerank_batch)

    decay = None
    if args.decay:
        meta = (pd.read_parquet(Path(args.meta) if args.meta else cfg.openj9_issue_meta_path)
                .drop_duplicates("issue_number").set_index("issue_number")["created_at"])
        ages = pd.to_datetime(train["issue_number"].map(meta), utc=True, errors="coerce")
        ref = ages.max()
        dyr = (ref - ages).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
        decay = np.exp(-args.lam * np.nan_to_num(dyr, nan=0.0)).astype(np.float32)

    def fmt(m):
        return f"Hit@1={m['hit@1']:.4f} Hit@5={m['hit@5']:.4f} Hit@10={m['hit@10']:.4f} MRR={m['mrr']:.4f}"

    def vs(k, tau, vote):
        return rank_metrics(vote_scores(sims, train_cls, C, k, tau, decay, vote, dev_freq), true_idx)

    if args.sweep:
        logger.success("== Barrido CBR-recuperación (hybrid={} rerank={} vote={}) ==",
                       args.hybrid, args.rerank, args.vote)
        for k in (5, 10, 15, 20, 30, 50):
            for tau in (1.0, 2.0, 4.0):
                logger.info("  k={:>2} τ={:.0f} : {}", k, tau, fmt(vs(k, tau, args.vote)))
    elif args.vote_sweep:
        logger.success("== Barrido de --vote (k={} τ={} hybrid={} rerank={}) ==",
                       args.top_k, args.tau, args.hybrid, args.rerank)
        for vote in ("sum", "mean", "max", "idf", "rank"):
            logger.info("  vote={:>4} : {}", vote, fmt(vs(args.top_k, args.tau, vote)))
    else:
        m = vs(args.top_k, args.tau, args.vote)
        logger.success("== CBR-recuperación | k={} τ={} vote={} hybrid={} rerank={} ==",
                       args.top_k, args.tau, args.vote, args.hybrid, args.rerank)
        logger.info("  {}", fmt(m))
        print(json.dumps(m, indent=2))


def _finetune(model: SentenceTransformer, train: pd.DataFrame, train_cls: np.ndarray,
              C: int, args, device: str) -> None:
    """Afina el encoder para acercar bugs del mismo dev (API moderna ST 5.x, CUDA/fp16).

    --loss mnrl (default): MultipleNegativesRankingLoss sobre pares (anchor, positive)
      = dos bugs DISTINTOS del mismo dev; negativos in-batch (sampler NO_DUPLICATES).
      Estándar para retrieval, estable y rápido.
    --loss triplet: BatchAllTripletLoss con `owner` como etiqueta (GROUP_BY_LABEL);
      colapsa el espacio en datasets chicos (verificado: 0.27→0.16), no recomendado."""
    from collections import defaultdict

    from datasets import Dataset
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    common = dict(
        output_dir=str(settings.openj9_dir / "cbr_retrieval_ft_trainer"),
        num_train_epochs=args.epochs, per_device_train_batch_size=args.ft_batch,
        learning_rate=args.lr, warmup_ratio=0.1, fp16=not args.cpu,
        dataloader_drop_last=True, logging_steps=20, save_strategy="no", report_to="none",
    )

    if args.loss == "triplet":
        logger.info("Fine-tuning TRIPLET (CUDA/fp16): {} ép, batch={}, lr={}...",
                    args.epochs, args.ft_batch, args.lr)
        ds = Dataset.from_dict({"sentence": train["text"].tolist(),
                                "label": [int(c) for c in train_cls]})
        loss = losses.BatchAllTripletLoss(model=model)
        targs = SentenceTransformerTrainingArguments(
            batch_sampler=BatchSamplers.GROUP_BY_LABEL, **common)
    else:  # mnrl
        texts = train["text"].tolist()
        by_dev: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(train_cls):
            by_dev[int(c)].append(i)
        rng = np.random.default_rng(settings.seed)
        anchors, positives = [], []
        for i, c in enumerate(train_cls):
            others = [j for j in by_dev[int(c)] if j != i]
            if not others:
                continue  # devs con 1 solo bug: no forman par (siguen en el índice al inferir)
            j = others[int(rng.integers(len(others)))]
            anchors.append(texts[i]); positives.append(texts[j])
        logger.info("Fine-tuning MNRL (CUDA/fp16): {} pares mismo-dev, {} ép, batch={}, lr={}...",
                    len(anchors), args.epochs, args.ft_batch, args.lr)
        ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
        loss = losses.MultipleNegativesRankingLoss(model=model)
        targs = SentenceTransformerTrainingArguments(
            batch_sampler=BatchSamplers.NO_DUPLICATES, **common)

    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    trainer.train()
    logger.success("Fine-tuning terminado.")


if __name__ == "__main__":
    main()
