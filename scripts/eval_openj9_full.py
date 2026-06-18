"""Sistema completo en OpenJ9: FS = NPS(CBR) + W_f·NIS(IBR). Validación end-to-end.

Carga el CBR entrenado (`scripts/train_openj9_cbr.py`) y el IBR (interacciones de la
GitHub API) y los fusiona con la fórmula de TriagerX, alineando ambos por el mismo
`label_encoder` (sorted owners). Reporta CBR-solo (W_f=0), IBR-solo y el full barriendo
W_f. Flags `--ip-c/--ip-a/--ip-d` para ver si `contribution` ayuda END-TO-END.

OpenJ9 no tiene split de validación (solo train/test, time-sliced de TriagerX), así que
se reporta la CURVA de W_f en test y se destaca W_f=0.7 (default de TriagerX) como elección
principista (no sintonizada en test).

Uso (tras entrenar el CBR y minar el IBR):
    uv run python scripts/eval_openj9_full.py                 # contribution ON (TriagerX)
    uv run python scripts/eval_openj9_full.py --ip-c 0        # contribution OFF
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from triager_omega.cbr.predict import normalize_rows
from triager_omega.config import Settings
from triager_omega.modules.aggregator import rank_metrics
from triager_omega.modules.ibr import InteractionRecommender


def _interaction_table(inter_path, issue_ids: set[int]) -> dict:
    inter = pd.read_parquet(inter_path)
    inter = inter[inter["issue_number"].isin(issue_ids)].dropna(subset=["timestamp"])
    table: dict[int, list] = defaultdict(list)
    for n, dev, kind, ts in inter[["issue_number", "dev", "kind", "timestamp"]].itertuples(
        index=False, name=None
    ):
        table[int(n)].append((dev, kind, ts))
    return dict(table)


def _rrf_rank_term(mat: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank por elemento: 1/(k + rango), rango por fila desc de `mat`."""
    ranks = np.empty_like(mat, dtype=np.float64)
    order = np.argsort(-mat, axis=1)
    rows = np.arange(mat.shape[0])[:, None]
    ranks[rows, order] = np.arange(mat.shape[1])[None, :]
    return 1.0 / (k + ranks + 1.0)


def _gate_mask(nps: np.ndarray, gate: float | None) -> np.ndarray:
    """Máscara [N,1] (mejora 3): 1 si el CBR está INSEGURO (margen top1−top2 < gate),
    si no 0. Con None → todo 1 (sin gating). El NPS llega minmax por fila (top1=1)."""
    if gate is None:
        return np.ones((nps.shape[0], 1), dtype=np.float32)
    part = np.partition(-nps, 1, axis=1)
    top1, top2 = -part[:, 0], -part[:, 1]
    uncertain = (top1 - top2) < gate  # CBR dudoso → deja entrar al IBR
    return uncertain.astype(np.float32)[:, None]


def _make_fuser(nps: np.ndarray, nis: np.ndarray, mode: str, rrf_k: int,
                gate: float | None = None):
    """Devuelve `fuse(wf) -> [N, C]` según el modo de fusión (mejoras 2 y 3).

    - linear: FS = NPS + wf·NIS (suma de scores; amplifica el orden si están
      correlacionados → el ruido del IBR arrastra el Top-1).
    - rrf: FS = rr(NPS) + wf·rr(NIS), donde rr = 1/(k+rango). Solo usa el ORDEN
      (robusto a escalas) y el término del IBR se enmascara a los devs con señal
      (NIS>0), así un candidato sin interacciones no recibe crédito espurio.
    - gate (mejora 3): el término del IBR se aplica SOLO en las consultas donde el
      CBR está inseguro (margen top1−top2 < gate); donde el CBR es confiable, FS=NPS
      puro → preserva los Top-1 correctos y deja que el IBR solo desempate dudas.
    """
    gmask = _gate_mask(nps, gate)
    if mode == "rrf":
        rr_nps = _rrf_rank_term(nps, rrf_k)
        rr_nis = _rrf_rank_term(nis, rrf_k) * (nis > 0)  # IBR habla solo con señal
        return lambda wf: rr_nps + wf * rr_nis * gmask
    return lambda wf: nps + wf * nis * gmask


def run(args: argparse.Namespace) -> None:
    cfg = Settings(
        ibr_top_k_retrieve=args.top_k, ibr_tau=args.tau, ibr_lambda=args.lam,
        ip_contribution=args.ip_c, ip_assignment=args.ip_a, ip_discussion=args.ip_d,
    )
    device = "cpu" if args.cpu else cfg.torch_device

    from pathlib import Path
    train_csv = Path(args.train_csv) if args.train_csv else cfg.openj9_train_csv
    test_csv = Path(args.test_csv) if args.test_csv else cfg.openj9_test_csv
    inter_path = Path(args.interactions) if args.interactions else cfg.openj9_interactions_path
    meta_path = Path(args.meta) if args.meta else cfg.openj9_issue_meta_path
    train = pd.read_csv(train_csv).drop_duplicates("issue_number")
    test = pd.read_csv(test_csv).drop_duplicates("issue_number")
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    # ---------- CBR → NPS [N, C] ----------
    if args.cbr_mode == "retrieval":
        # Case-Based Reasoning real: kNN sobre bugs pasados → voto al `owner` (resolvedor).
        from cbr_retrieval_openj9 import label_encoder as _le_fn, vote_scores
        from sentence_transformers import SentenceTransformer
        le = _le_fn(train, test)
        num_classes = len(le)
        true_idx = test["owner"].map(le).to_numpy().astype(int)
        train_cls = train["owner"].map(le).to_numpy().astype(int)
        logger.info("CBR-recuperación: {} (k={}, τ={}, maxlen={})...",
                    args.cbr_model, args.cbr_topk, args.cbr_tau, args.cbr_maxlen)
        enc = SentenceTransformer(args.cbr_model, device=device)
        enc.max_seq_length = args.cbr_maxlen
        emb_tr = enc.encode(train["text"].tolist(), batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        emb_te = enc.encode(test["text"].tolist(), batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        sims = emb_te @ emb_tr.T
        nps = normalize_rows(
            vote_scores(sims, train_cls, num_classes, args.cbr_topk, args.cbr_tau), how="minmax")
    else:
        cbr_dir = cfg.openj9_dir / args.cbr_name
        if not (cbr_dir / "label_encoder.json").exists():
            raise SystemExit(f"No existe el CBR en {cbr_dir}. Corre antes scripts/train_openj9_cbr.py.")
        le: dict[str, int] = json.loads((cbr_dir / "label_encoder.json").read_text())
        num_classes = len(le)
        true_idx = test["owner"].map(le).to_numpy().astype(int)
        logger.info("CBR-clasificador: cargando {} y prediciendo NPS...", cbr_dir)
        tok = AutoTokenizer.from_pretrained(str(cbr_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(cbr_dir)).to(device).eval()
        logits = []
        with torch.no_grad():
            texts = test["text"].tolist()
            for i in range(0, len(texts), 32):
                e = tok(texts[i : i + 32], truncation=True, max_length=args.max_length,
                        padding=True, return_tensors="pt").to(device)
                logits.append(model(**e).logits.float().cpu().numpy())
        nps = normalize_rows(np.concatenate(logits, 0), how="minmax")

    # ---------- IBR → NIS [N, C] ----------
    logger.info("IBR (canal={}): scoring NIS (IP C/A/D={}/{}/{})...",
                args.ibr_channel, args.ip_c, args.ip_a, args.ip_d)
    ibr = InteractionRecommender(cfg=cfg)
    ibr._active = set(le)
    ibr._train_bug_ids = train["issue_number"].to_numpy()
    ibr._interactions = _interaction_table(inter_path, set(int(n) for n in ibr._train_bug_ids))
    meta = (pd.read_parquet(meta_path)
            .drop_duplicates("issue_number").set_index("issue_number")["created_at"])

    def _t_now(issue_number) -> "pd.Timestamp | None":
        t = meta.get(int(issue_number), None)
        return None if pd.isna(t) else t

    nis = np.zeros((len(test), num_classes), dtype=np.float32)
    if args.ibr_channel == "lexical":
        # Mejora 1 (decorrelar el IBR): los vecinos se recuperan por BM25 LÉXICO
        # (tokens técnicos exactos) en vez del MPNet semántico → el canal IBR deja
        # de mirar los mismos vecinos que el CBR-recuperación, así que aporta señal
        # ortogonal en vez de redundante. La agregación por interacciones tipadas
        # (IP · decay · anti-fuga temporal) es idéntica al canal semántico.
        from cbr_retrieval_openj9 import bm25_sims
        logger.info("IBR léxico: BM25 sobre {} bugs de train (k={})...",
                    len(train), cfg.ibr_top_k_retrieve)
        lex = bm25_sims(train["text"].tolist(), test["text"].tolist())  # [Nte, Ntr]
        train_ids = ibr._train_bug_ids
        k = min(cfg.ibr_top_k_retrieve, lex.shape[1])
        for i, row in enumerate(test.itertuples()):
            t_now = _t_now(row.issue_number)
            srow = lex[i]
            top = np.argpartition(-srow, kth=k - 1)[:k]
            isc: dict = {}
            for j in top:
                s = float(srow[j])
                if s <= 0.0:  # BM25 no negativo; 0 = sin solape léxico
                    continue
                bug_id = int(train_ids[j])
                for dev, kind, ts in ibr._interactions.get(bug_id, ()):  # noqa: B007
                    if dev not in ibr._active:
                        continue
                    if t_now is not None and ts >= t_now:  # anti-fuga temporal
                        continue
                    isc[dev] = isc.get(dev, 0.0) + s * ibr._ip(kind) * ibr._decay(ts, t_now)
            d = ibr._normalize_nis(isc)
            for owner, idx in le.items():
                nis[i, idx] = d.get(owner, 0.0)
    else:  # semantic (baseline): MPNet, MISMO encoder que el CBR (correlacionado)
        emb = ibr._sbert().encode(train["text"].tolist(), batch_size=64,
                                  convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        ibr._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
        q = ibr._sbert().encode(test["text"].tolist(), batch_size=64,
                                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        q = torch.as_tensor(q, dtype=torch.float32)
        for i, row in enumerate(test.itertuples()):
            d = ibr._score(q[i : i + 1], _t_now(row.issue_number))
            for owner, idx in le.items():
                nis[i, idx] = d.get(owner, 0.0)

    # ---------- fusión ----------
    def fmt(m):
        return f"Hit@1={m['hit@1']:.4f} Hit@5={m['hit@5']:.4f} Hit@10={m['hit@10']:.4f} MRR={m['mrr']:.4f}"

    wfs = [round(x, 1) for x in np.arange(0.1, 1.0001, 0.1)]
    logger.success("== OpenJ9 sistema completo | fusion={} | canal={} | IP C/A/D={}/{}/{} ==",
                   args.fusion, args.ibr_channel, args.ip_c, args.ip_a, args.ip_d)
    logger.info("CBR-solo        : {}", fmt(rank_metrics(nps, true_idx)))
    logger.info("IBR-solo        : {}", fmt(rank_metrics(nis, true_idx)))

    if args.gate_sweep:
        # Mejora 3: barre el umbral de gate (reusa NPS/NIS). Por gate reporta el W_f
        # que maximiza Top-1 y el que maximiza MRR (curva en test, mismo caveat).
        for gate in [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            fz = _make_fuser(nps, nis, args.fusion, args.rrf_k, gate)
            rows = [(wf, rank_metrics(fz(wf), true_idx)) for wf in wfs]
            bh = max(rows, key=lambda r: r[1]["hit@1"])
            bm = max(rows, key=lambda r: r[1]["mrr"])
            cov = "off " if gate is None else f"{float((_gate_mask(nps, gate)).mean()):.2f}"
            logger.info("gate={:>4} (frac IBR={}): best-Top1 wf={:.1f} {} | best-MRR wf={:.1f} {}",
                        "off" if gate is None else f"{gate:.1f}", cov,
                        bh[0], fmt(bh[1]), bm[0], fmt(bm[1]))
        return

    fused = _make_fuser(nps, nis, args.fusion, args.rrf_k, args.gate)
    for wf in wfs:
        tag = "  <- TriagerX W_f=0.7" if abs(wf - 0.7) < 1e-9 else ""
        logger.info("FS W_f={:.1f}     : {}{}", wf, fmt(rank_metrics(fused(wf), true_idx)), tag)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--ip-c", type=float, default=1.5)
    p.add_argument("--ip-a", type=float, default=0.5)
    p.add_argument("--ip-d", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    # IBR: cómo recupera los vecinos para el NIS (mejora 1 = decorrelar del CBR).
    p.add_argument("--ibr-channel", choices=["semantic", "lexical"], default="semantic",
                   help="semantic=MPNet (mismo encoder que el CBR, correlacionado); "
                        "lexical=BM25 (tokens exactos, decorrelado del CBR)")
    # Fusión CBR↔IBR (mejora 2): suma lineal o Reciprocal Rank Fusion.
    p.add_argument("--fusion", choices=["linear", "rrf"], default="linear",
                   help="linear=NPS+wf·NIS; rrf=fusión por rango (solo orden, IBR enmascarado a NIS>0)")
    p.add_argument("--rrf-k", type=int, default=60, help="constante de RRF (mejora 2)")
    # Fusión condicional (mejora 3): aplica el IBR solo si el CBR está inseguro.
    p.add_argument("--gate", type=float, default=None,
                   help="umbral de margen top1-top2 del CBR; el IBR solo entra si margen<gate")
    p.add_argument("--gate-sweep", action="store_true",
                   help="barre umbrales de gate (mejora 3) reusando NPS/NIS")
    p.add_argument("--cpu", action="store_true")
    # CBR: clasificador (DeBERTa entrenado) o recuperación (kNN sobre casos, sin entrenar).
    p.add_argument("--cbr-mode", choices=["classifier", "retrieval"], default="classifier")
    p.add_argument("--cbr-model", default="sentence-transformers/all-mpnet-base-v2",
                   help="encoder para cbr-mode=retrieval (modelo ST o carpeta afinada)")
    p.add_argument("--cbr-topk", type=int, default=50)
    p.add_argument("--cbr-tau", type=float, default=2.0)
    p.add_argument("--cbr-maxlen", type=int, default=384)
    # Set alternativo (50 clases): CBR/CSVs/interacciones propios sin pisar el 17-set.
    p.add_argument("--cbr-name", default="cbr_model", help="subcarpeta del CBR-clasificador en artifacts/openj9/")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--interactions", default=None, help="parquet de interacciones (default: config)")
    p.add_argument("--meta", default=None, help="parquet de meta/created_at (default: config)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
