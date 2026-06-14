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


def run(args: argparse.Namespace) -> None:
    cfg = Settings(
        ibr_top_k_retrieve=args.top_k, ibr_tau=args.tau, ibr_lambda=args.lam,
        ip_contribution=args.ip_c, ip_assignment=args.ip_a, ip_discussion=args.ip_d,
    )
    cbr_dir = cfg.openj9_dir / args.cbr_name
    if not (cbr_dir / "label_encoder.json").exists():
        raise SystemExit(f"No existe el CBR en {cbr_dir}. Corre antes scripts/train_openj9_cbr.py.")
    le: dict[str, int] = json.loads((cbr_dir / "label_encoder.json").read_text())
    num_classes = len(le)
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
    true_idx = test["owner"].map(le).to_numpy().astype(int)

    # ---------- CBR → NPS [N, C] ----------
    logger.info("CBR: cargando {} y prediciendo NPS...", cbr_dir)
    tok = AutoTokenizer.from_pretrained(str(cbr_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(cbr_dir)).to(device).eval()
    logits = []
    with torch.no_grad():
        texts = test["text"].tolist()
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt").to(device)
            logits.append(model(**enc).logits.float().cpu().numpy())
    nps = normalize_rows(np.concatenate(logits, 0), how="minmax")

    # ---------- IBR → NIS [N, C] ----------
    logger.info("IBR: embebiendo train + scoring NIS (IP C/A/D={}/{}/{})...",
                args.ip_c, args.ip_a, args.ip_d)
    ibr = InteractionRecommender(cfg=cfg)
    ibr._active = set(le)
    ibr._train_bug_ids = train["issue_number"].to_numpy()
    emb = ibr._sbert().encode(train["text"].tolist(), batch_size=64,
                              convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    ibr._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
    ibr._interactions = _interaction_table(inter_path, set(int(n) for n in ibr._train_bug_ids))
    meta = (pd.read_parquet(meta_path)
            .drop_duplicates("issue_number").set_index("issue_number")["created_at"])
    q = ibr._sbert().encode(test["text"].tolist(), batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    q = torch.as_tensor(q, dtype=torch.float32)
    nis = np.zeros((len(test), num_classes), dtype=np.float32)
    for i, row in enumerate(test.itertuples()):
        t_now = meta.get(int(row.issue_number), None)
        if pd.isna(t_now):
            t_now = None
        d = ibr._score(q[i : i + 1], t_now)
        for owner, idx in le.items():
            nis[i, idx] = d.get(owner, 0.0)

    # ---------- fusión ----------
    def fmt(m):
        return f"Hit@1={m['hit@1']:.4f} Hit@5={m['hit@5']:.4f} Hit@10={m['hit@10']:.4f} MRR={m['mrr']:.4f}"

    logger.success("== OpenJ9 sistema completo | IP C/A/D={}/{}/{} ==", args.ip_c, args.ip_a, args.ip_d)
    logger.info("CBR-solo        : {}", fmt(rank_metrics(nps, true_idx)))
    logger.info("IBR-solo        : {}", fmt(rank_metrics(nis, true_idx)))
    for wf in [round(x, 1) for x in np.arange(0.1, 1.0001, 0.1)]:
        tag = "  <- TriagerX W_f=0.7" if abs(wf - 0.7) < 1e-9 else ""
        logger.info("FS W_f={:.1f}     : {}{}", wf, fmt(rank_metrics(nps + wf * nis, true_idx)), tag)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--ip-c", type=float, default=1.5)
    p.add_argument("--ip-a", type=float, default=0.5)
    p.add_argument("--ip-d", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    # Set alternativo (50 clases): CBR/CSVs/interacciones propios sin pisar el 17-set.
    p.add_argument("--cbr-name", default="cbr_model", help="subcarpeta del CBR en artifacts/openj9/")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--interactions", default=None, help="parquet de interacciones (default: config)")
    p.add_argument("--meta", default=None, help="parquet de meta/created_at (default: config)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
