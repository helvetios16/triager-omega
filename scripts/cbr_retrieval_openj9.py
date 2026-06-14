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
                top_k: int, tau: float, decay: np.ndarray | None = None) -> np.ndarray:
    """Matriz de scores [Nte, C] por voto kNN ponderado por similitud^τ (· decay)."""
    n = sims.shape[0]
    scores = np.zeros((n, num_classes), dtype=np.float32)
    w = np.clip(sims, 0.0, None) ** tau
    if decay is not None:
        w = w * decay[None, :]
    # top-k vecinos por fila
    idx = np.argpartition(-sims, kth=min(top_k, sims.shape[1] - 1), axis=1)[:, :top_k]
    for i in range(n):
        cols = idx[i]
        np.add.at(scores[i], train_cls[cols], w[i, cols])
    return scores


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
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--ft-batch", type=int, default=32)
    p.add_argument("--sweep", action="store_true", help="barre (k, τ) sobre los mismos embeddings")
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
    logger.info("CBR-recuperación OpenJ9 | train={} test={} clases={}", len(train), len(test), C)

    device = "cpu" if args.cpu else cfg.torch_device
    model = SentenceTransformer(args.model, device=device)

    if args.finetune:
        _finetune(model, train, train_cls, C, args, device)

    enc = lambda txts: model.encode(txts, batch_size=64, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
    logger.info("Embebiendo train + test con {}...", args.model)
    emb_tr = enc(train["text"].tolist())
    emb_te = enc(test["text"].tolist())
    sims = emb_te @ emb_tr.T  # coseno [Nte, Ntr]

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

    if args.sweep:
        logger.success("== Barrido CBR-recuperación (zero-shot={}) ==", not args.finetune)
        for k in (5, 10, 15, 20, 30, 50):
            for tau in (1.0, 2.0, 4.0):
                m = rank_metrics(vote_scores(sims, train_cls, C, k, tau, decay), true_idx)
                logger.info("  k={:>2} τ={:.0f} : {}", k, tau, fmt(m))
    else:
        m = rank_metrics(vote_scores(sims, train_cls, C, args.top_k, args.tau, decay), true_idx)
        logger.success("== CBR-recuperación | k={} τ={} decay={} ==", args.top_k, args.tau, args.decay)
        logger.info("  {}", fmt(m))
        print(json.dumps(m, indent=2))


def _finetune(model: SentenceTransformer, train: pd.DataFrame, train_cls: np.ndarray,
              C: int, args, device: str) -> None:
    """Afina el encoder con BatchAllTripletLoss usando `owner` como etiqueta:
    acerca bugs del mismo dev, aleja los de devs distintos.

    API moderna de ST 5.x (SentenceTransformerTrainer): usa CUDA + fp16 y agrupa el
    batch por etiqueta (GROUP_BY_LABEL) para que se formen triplets válidos. El
    model.fit() legacy corría en CPU (~69 s/it), por eso se evita."""
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    logger.info("Fine-tuning contrastivo (CUDA/fp16): {} épocas, batch={}...",
                args.epochs, args.ft_batch)
    ds = Dataset.from_dict({"sentence": train["text"].tolist(),
                            "label": [int(c) for c in train_cls]})
    loss = losses.BatchAllTripletLoss(model=model)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(settings.openj9_dir / "cbr_retrieval_ft_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.ft_batch,
        warmup_ratio=0.1,
        fp16=not args.cpu,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        dataloader_drop_last=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    trainer.train()
    logger.success("Fine-tuning terminado.")


if __name__ == "__main__":
    main()
