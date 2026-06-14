"""Evalúa el IBR sobre OpenJ9 (validación vs TriagerX, régimen donde `contribution` vale).

Reutiliza el MOTOR de scoring de `triager_omega.modules.ibr.InteractionRecommender`
(`_score`/`_ip`/`_decay`/`_normalize_nis`) inyectándole los datos de OpenJ9 en vez
de los del piloto Mozilla — es el "adaptador" mínimo, sin tocar el módulo de producción.

Datos:
  - texto + etiqueta: `assets/openj9_{train,test}_17.csv` (columna `text`, etiqueta `owner`).
  - interacciones: `artifacts/openj9/openj9_interactions.parquet` (minería de la GitHub API).
  - t_now de cada query: `artifacts/openj9/openj9_issue_meta.parquet` (created_at).

Índice = issues de TRAIN; queries = issues de TEST (split temporal de TriagerX).
Identidad trivial: `dev` y `owner` son ambos logins de GitHub.

Config por defecto = la de TriagerX para OpenJ9 (`triagerx_config.yaml`):
  top_k=15, τ=0.6, λ=0.01, IP contribution/assignment/discussion = 1.5/0.5/0.1.
Flags para ablar cada pieza (p.ej. `--ip-c 0` para apagar contribution).

Uso (tras correr scripts/mine_openj9_timelines.py):
    uv run python scripts/eval_openj9_ibr.py
    uv run python scripts/eval_openj9_ibr.py --ip-c 0          # ablación: sin contribution
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger

from triager_omega.config import Settings
from triager_omega.modules.ibr import InteractionRecommender


def _label_encoder(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    owners = sorted(set(train["owner"].dropna()) | set(test["owner"].dropna()))
    return {o: i for i, o in enumerate(owners)}


def _interaction_table(cfg: Settings, issue_ids: set[int]) -> dict:
    inter = pd.read_parquet(cfg.openj9_interactions_path)
    inter = inter[inter["issue_number"].isin(issue_ids)].dropna(subset=["timestamp"])
    table: dict[int, list] = defaultdict(list)
    for n, dev, kind, ts in inter[["issue_number", "dev", "kind", "timestamp"]].itertuples(
        index=False, name=None
    ):
        table[int(n)].append((dev, kind, ts))  # dev = login (str), igual que `owner`
    return dict(table)


def run(args: argparse.Namespace) -> dict:
    # Config OpenJ9 (overridea los defaults Mozilla de config.py).
    cfg = Settings(
        ibr_top_k_retrieve=args.top_k, ibr_tau=args.tau, ibr_lambda=args.lam,
        ip_contribution=args.ip_c, ip_assignment=args.ip_a, ip_discussion=args.ip_d,
    )
    if not cfg.openj9_interactions_path.exists():
        raise SystemExit(
            f"No existe {cfg.openj9_interactions_path}. Corre primero "
            "`uv run python scripts/mine_openj9_timelines.py` (con GITHUB_TOKEN)."
        )

    train = pd.read_csv(cfg.openj9_train_csv).drop_duplicates("issue_number")
    test = pd.read_csv(cfg.openj9_test_csv).drop_duplicates("issue_number")
    le = _label_encoder(train, test)
    logger.info("OpenJ9 | train={} test={} devs(owner)={}", len(train), len(test), len(le))

    ibr = InteractionRecommender(cfg=cfg)
    ibr._active = set(le)

    # Índice = train. Embebe el texto (mismo SBERT que TriagerX: all-mpnet-base-v2).
    ibr._train_bug_ids = train["issue_number"].to_numpy()
    emb = ibr._sbert().encode(
        train["text"].fillna("").tolist(), batch_size=64,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    ibr._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
    ibr._interactions = _interaction_table(cfg, set(int(n) for n in ibr._train_bug_ids))

    meta = (
        pd.read_parquet(cfg.openj9_issue_meta_path)
        .drop_duplicates("issue_number").set_index("issue_number")["created_at"]
    )

    # Queries = test.
    q_emb = ibr._sbert().encode(
        test["text"].fillna("").tolist(), batch_size=64,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    q_emb = torch.as_tensor(q_emb, dtype=torch.float32)

    ks = (1, 3, 5, 10)
    ranks, signaled = [], []
    for i, row in enumerate(test.itertuples()):
        t_now = meta.get(int(row.issue_number), None)
        if pd.isna(t_now):
            t_now = None
        nis = ibr._score(q_emb[i : i + 1], t_now)
        true = row.owner
        order = sorted(nis.items(), key=lambda kv: (-kv[1], kv[0]))
        rank = next((r for r, (d, _) in enumerate(order) if d == true), len(order))
        ranks.append(rank)
        signaled.append(nis.get(true, 0.0) > 0.0)

    ranks_arr = np.asarray(ranks)
    metrics = {f"hit@{k}": float((ranks_arr < k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / (ranks_arr + 1)).mean())
    metrics["coverage"] = float(np.mean(signaled))

    logger.success("== IBR-solo OpenJ9 | top_k={} τ={} λ={} | IP C/A/D={}/{}/{} ==",
                   args.top_k, args.tau, args.lam, args.ip_c, args.ip_a, args.ip_d)
    for k, v in metrics.items():
        logger.info("  {} = {:.4f}", k, v)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # defaults = triagerx_config.yaml (OpenJ9)
    p.add_argument("--top-k", type=int, default=15, help="maximum_similar_issues (TriagerX: 15)")
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--ip-c", type=float, default=1.5, help="ip_contribution (commit/PR)")
    p.add_argument("--ip-a", type=float, default=0.5, help="ip_assignment")
    p.add_argument("--ip-d", type=float, default=0.1, help="ip_discussion")
    run(p.parse_args())


if __name__ == "__main__":
    main()
