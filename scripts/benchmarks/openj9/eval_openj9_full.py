"""Sistema completo OpenJ9: FS = NPS(CBR) + W_f·NIS(IBR)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from triager_omega.config import Settings
from triager_omega.modules.aggregator import (
    gate_mask, make_fuser, normalize_rows, rank_metrics, rrf_rank_term,
)
from triager_omega.modules.ibr import InteractionIndex, InteractionRecommender, interaction_table_from_rows
from triager_omega.retrieval import CbrConfig, CbrRetriever
from triager_omega.retrieval.bm25 import bm25_sims


@dataclass
class EvaluationConfig:
    """Configuración explícita de evaluación; conserva los defaults OpenJ9 full."""
    top_k: int = 15
    tau: float = 0.6
    lam: float = 0.01
    ip_c: float = 1.5
    ip_a: float = 0.5
    ip_d: float = 0.1
    max_length: int = 256
    ibr_channel: str = "semantic"
    fusion: str = "linear"
    rrf_k: int = 60
    gate: float | None = None
    gate_sweep: bool = False
    cpu: bool = False
    cbr_mode: str = "retrieval"
    cbr_model: str = "sentence-transformers/all-mpnet-base-v2"
    cbr_topk: int = 50
    cbr_tau: float = 2.0
    cbr_vote: str = "idf"
    cbr_maxlen: int = 384
    cbr_name: str = "cbr_model"
    train_csv: str | None = None
    test_csv: str | None = None
    interactions: str | None = None
    meta: str | None = None


CONFIG = EvaluationConfig()


def _interaction_table(inter_path, issue_ids: set[int]) -> dict:
    inter = pd.read_parquet(inter_path)
    inter = inter[inter["issue_number"].isin(issue_ids)].dropna(subset=["timestamp"])
    rows = ((n, dev, kind, ts) for n, dev, kind, ts in inter[["issue_number", "dev", "kind", "timestamp"]].itertuples(index=False, name=None))
    return interaction_table_from_rows(rows, issue_ids)


# Compatibility aliases for older benchmark tests/callers; implementation is shared.
_rrf_rank_term = rrf_rank_term
_gate_mask = gate_mask
_make_fuser = make_fuser


def _format_metrics(metrics: dict) -> str:
    return f"Hit@1={metrics['hit@1']:.4f} Hit@5={metrics['hit@5']:.4f} Hit@10={metrics['hit@10']:.4f} MRR={metrics['mrr']:.4f}"


def evaluate(
    config: EvaluationConfig = CONFIG, *, train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None, train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None, nis: np.ndarray | None = None,
    interactions_df: pd.DataFrame | None = None, meta_df: pd.DataFrame | None = None,
    fusion_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Calcula NPS/NIS y la curva de fusión, devolviendo matrices y métricas.

    DataFrames/embeddings/NIS inyectables permiten pruebas offline. La ruta normal
    conserva la carga y fórmulas IBR históricas.
    """
    cfg = Settings(ibr_top_k_retrieve=config.top_k, ibr_tau=config.tau, ibr_lambda=config.lam,
                   ip_contribution=config.ip_c, ip_assignment=config.ip_a, ip_discussion=config.ip_d)
    device = "cpu" if config.cpu else cfg.torch_device
    if train_df is None:
        train_df = pd.read_csv(Path(config.train_csv) if config.train_csv else cfg.openj9_train_csv)
    if test_df is None:
        test_df = pd.read_csv(Path(config.test_csv) if config.test_csv else cfg.openj9_test_csv)
    train, test = train_df.drop_duplicates("issue_number").copy(), test_df.drop_duplicates("issue_number").copy()
    for frame in (train, test):
        frame["text"] = frame["text"].fillna("").astype(str)

    if config.cbr_mode == "retrieval":
        retriever = CbrRetriever(CbrConfig(encoder_name=config.cbr_model, device=device,
            max_seq_length=config.cbr_maxlen, top_k=config.cbr_topk, tau=config.cbr_tau,
            vote_strategy=config.cbr_vote))
        retriever.fit(train["text"].tolist(), train["owner"].tolist(), test["owner"].tolist(),
                      precomputed_embeddings=train_embeddings)
        nps = normalize_rows(retriever.score(test["text"].tolist(), test_embeddings=test_embeddings), how="minmax")
        le = retriever.label_encoder
    elif config.cbr_mode == "classifier":
        # Compatibilidad histórica: dependencias pesadas solo se cargan en esta rama.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cbr_dir = cfg.openj9_dir / config.cbr_name
        if not (cbr_dir / "label_encoder.json").exists():
            raise SystemExit(f"No existe el CBR en {cbr_dir}. Corre antes scripts/train_openj9_cbr.py.")
        le = json.loads((cbr_dir / "label_encoder.json").read_text())
        model = AutoModelForSequenceClassification.from_pretrained(str(cbr_dir)).to(device).eval()
        tok = AutoTokenizer.from_pretrained(str(cbr_dir))
        logits = []
        with torch.no_grad():
            texts = test["text"].tolist()
            for i in range(0, len(texts), 32):
                batch = tok(texts[i:i + 32], truncation=True, max_length=config.max_length,
                            padding=True, return_tensors="pt").to(device)
                logits.append(model(**batch).logits.float().cpu().numpy())
        nps = normalize_rows(np.concatenate(logits, 0), how="minmax")
    else:
        raise ValueError("cbr_mode debe ser 'retrieval' o 'classifier'")
    true_idx = test["owner"].map(le).to_numpy().astype(int)
    num_classes = len(le)

    if nis is None:
        logger.info("IBR (canal={}): scoring NIS (IP C/A/D={}/{}/{})...", config.ibr_channel, config.ip_c, config.ip_a, config.ip_d)
        ibr = InteractionRecommender(cfg=cfg)
        train_ids = train["issue_number"].to_numpy()
        if interactions_df is None:
            path = Path(config.interactions) if config.interactions else cfg.openj9_interactions_path
            inter = pd.read_parquet(path)
        else:
            inter = interactions_df.copy()
        inter = inter[inter["issue_number"].isin(set(train_ids))].dropna(subset=["timestamp"])
        rows = ((n, dev, kind, ts) for n, dev, kind, ts in inter[["issue_number", "dev", "kind", "timestamp"]].itertuples(index=False, name=None))
        ibr.set_index(InteractionIndex(train_ids, set(le), interaction_table_from_rows(rows, set(train_ids))))
        if meta_df is None:
            path = Path(config.meta) if config.meta else cfg.openj9_issue_meta_path
            meta_df = pd.read_parquet(path)
        meta = meta_df.drop_duplicates("issue_number").set_index("issue_number")["created_at"]
        def t_now(issue_number):
            t = meta.get(int(issue_number), None)
            return None if pd.isna(t) else t
        nis = np.zeros((len(test), num_classes), dtype=np.float32)
        if config.ibr_channel == "lexical":
            sims = bm25_sims(train["text"].tolist(), test["text"].tolist())
            for i, row in enumerate(test.itertuples()):
                now = t_now(row.issue_number)
                d = ibr.score_similarities(sims[i], now, policy="lexical")
                for owner, idx in le.items(): nis[i, idx] = d.get(owner, 0.)
        elif config.ibr_channel == "semantic":
            # Los embeddings inyectados pertenecen al CBR, cuyo modelo puede ser
            # distinto del configurado para IBR. Preservar su encoder independiente.
            emb = torch.as_tensor(ibr.encode_texts(train["text"].tolist(), show_progress_bar=False), dtype=torch.float32)
            q = torch.as_tensor(ibr.encode_texts(test["text"].tolist(), show_progress_bar=False), dtype=torch.float32)
            for i, row in enumerate(test.itertuples()):
                from sentence_transformers import util
                sims = util.cos_sim(q[i:i + 1], emb)[0]
                d = ibr.score_similarities(sims.cpu().numpy(), t_now(row.issue_number))
                for owner, idx in le.items(): nis[i, idx] = d.get(owner, 0.)
        else: raise ValueError("ibr_channel debe ser 'semantic' o 'lexical'")
    if np.asarray(nis).shape != (len(test), num_classes):
        raise ValueError(f"nis debe tener forma {(len(test), num_classes)}")
    wfs = ([float(weight) for weight in fusion_weights] if fusion_weights is not None
           else [round(x, 1) for x in np.arange(0.1, 1.0001, 0.1)])
    if not wfs or any(not np.isfinite(weight) or weight < 0 for weight in wfs):
        raise ValueError("fusion_weights must be a nonempty list of finite nonnegative weights")
    curves = []
    if config.gate_sweep:
        for gate in [None, *[i / 10 for i in range(1, 10)]]:
            fuser = _make_fuser(nps, nis, config.fusion, config.rrf_k, gate)
            rows = [(wf, rank_metrics(fuser(wf), true_idx)) for wf in wfs]
            curves.append({"gate": gate, "coverage": float(_gate_mask(nps, gate).mean()),
                           "best_hit1": max(rows, key=lambda r: r[1]["hit@1"]),
                           "best_mrr": max(rows, key=lambda r: r[1]["mrr"])})
    else:
        fuser = _make_fuser(nps, nis, config.fusion, config.rrf_k, config.gate)
        curves = [{"gate": config.gate, "weight": wf, "metrics": rank_metrics(fuser(wf), true_idx)} for wf in wfs]
    return {"nps": nps, "nis": nis, "true_indices": true_idx, "label_encoder": dict(le),
            "cbr_only": rank_metrics(nps, true_idx), "ibr_only": rank_metrics(nis, true_idx),
            "curves": curves, "gate_sweep": config.gate_sweep}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Wrapper legacy CLI; defaults derivados de CONFIG."""
    result = evaluate(EvaluationConfig(**vars(args)))
    fmt = _format_metrics
    logger.success("== OpenJ9 sistema completo | fusion={} | canal={} ==", args.fusion, args.ibr_channel)
    logger.info("CBR-solo: {}", fmt(result["cbr_only"]))
    logger.info("IBR-solo: {}", fmt(result["ibr_only"]))
    for row in result["curves"]:
        if args.gate_sweep:
            logger.info("gate={} best-Top1 wf={} {} | best-MRR wf={} {}", row["gate"], row["best_hit1"][0], fmt(row["best_hit1"][1]), row["best_mrr"][0], fmt(row["best_mrr"][1]))
        else:
            logger.info("FS W_f={:.1f} : {}{}", row["weight"], fmt(row["metrics"]), "  <- TriagerX W_f=0.7" if row["weight"] == 0.7 else "")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=CONFIG.top_k)
    p.add_argument("--tau", type=float, default=CONFIG.tau)
    p.add_argument("--lam", type=float, default=CONFIG.lam)
    p.add_argument("--ip-c", type=float, default=CONFIG.ip_c)
    p.add_argument("--ip-a", type=float, default=CONFIG.ip_a)
    p.add_argument("--ip-d", type=float, default=CONFIG.ip_d)
    p.add_argument("--max-length", type=int, default=CONFIG.max_length)
    p.add_argument("--ibr-channel", choices=["semantic", "lexical"], default=CONFIG.ibr_channel)
    p.add_argument("--fusion", choices=["linear", "rrf"], default=CONFIG.fusion)
    p.add_argument("--rrf-k", type=int, default=CONFIG.rrf_k)
    p.add_argument("--gate", type=float, default=CONFIG.gate)
    p.add_argument("--gate-sweep", action="store_true", default=CONFIG.gate_sweep)
    p.add_argument("--cpu", action="store_true", default=CONFIG.cpu)
    p.add_argument("--cbr-mode", choices=["classifier", "retrieval"], default=CONFIG.cbr_mode)
    p.add_argument("--cbr-model", default=CONFIG.cbr_model)
    p.add_argument("--cbr-topk", type=int, default=CONFIG.cbr_topk)
    p.add_argument("--cbr-tau", type=float, default=CONFIG.cbr_tau)
    p.add_argument("--cbr-vote", choices=["sum", "mean", "max", "idf", "rank"], default=CONFIG.cbr_vote)
    p.add_argument("--cbr-maxlen", type=int, default=CONFIG.cbr_maxlen)
    p.add_argument("--cbr-name", default=CONFIG.cbr_name)
    p.add_argument("--train-csv", default=CONFIG.train_csv)
    p.add_argument("--test-csv", default=CONFIG.test_csv)
    p.add_argument("--interactions", default=CONFIG.interactions)
    p.add_argument("--meta", default=CONFIG.meta)
    run(p.parse_args())


if __name__ == "__main__": main()
