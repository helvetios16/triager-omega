"""Reconstruye el set de ~50 clases de OpenJ9 para una comparación JUSTA con TriagerX.

El repo de TriagerX solo trae el subset preprocesado de 17 devs
(`openj9_{train,test}_17.csv`), pero el paper reporta sobre **50 clases**. Sus
absolutos (CBR 0.270, full 0.328) no son comparables con nuestros 17 (azar 1/17
vs 1/50). Este script recupera un set de ~50 clases desde el dataset crudo para
poder comparar DeBERTa-solo vs DeBERTa-solo en igualdad de condiciones.

Reconstrucción (ver docs/comparacion-triagerx-openj9.md §4):
  - Clases: owners con >= MIN_ISSUES issues (≥20 → 51 ≈ las 50 de TriagerX).
  - Texto: formato exacto de TriagerX `"Bug Title: {título}\\nBug Description: {cuerpo}"`.
  - Split temporal: por `issue_number` (proxy de tiempo; en el 17-set el corte
    train/test es limpio por issue_number, sin solapamiento). train < CUT / test >= CUT.

Salida: artifacts/openj9/openj9_{train,test}_50.csv (columnas compatibles con
train_openj9_cbr.py: incluye `text` y `owner`).

Uso:
    uv run python scripts/build_openj9_50.py
    uv run python scripts/build_openj9_50.py --min-issues 20 --cut 17695
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from triager_omega.config import settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-issues", type=int, default=20,
                   help="umbral de issues por owner para entrar como clase (≥20 → 51)")
    p.add_argument("--cut", type=int, default=17695,
                   help="corte de issue_number: train < cut, test >= cut (igual que el 17-set)")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="si >0, carvea un split de validación de la COLA temporal del train "
                        "(los issue_number más recientes antes de --cut) y escribe "
                        "openj9_{traininner,val}_50.csv para sintonizar hiperparámetros sin tocar test")
    args = p.parse_args()

    cfg = settings
    src = cfg.triagerx_repo / "assets" / "openj9_22112024.csv"
    out = cfg.openj9_dir
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(src).drop_duplicates("issue_number")
    vc = full["owner"].value_counts()
    devs = set(vc[vc >= args.min_issues].index)
    logger.info("Dataset completo: {} issues | clases (owners ≥{} issues): {}",
                len(full), args.min_issues, len(devs))

    d = full[full["owner"].isin(devs)].copy()
    d["text"] = ("Bug Title: " + d["issue_title"].fillna("").astype(str)
                 + "\nBug Description: " + d["issue_body"].fillna("").astype(str))

    train = d[d.issue_number < args.cut].copy()
    test = d[d.issue_number >= args.cut].copy()
    logger.info("train={} test={} | azar 1/clases={:.4f}", len(train), len(test), 1 / len(devs))

    cols = ["issue_number", "issue_title", "issue_body", "owner", "component", "text"]
    train[cols].to_csv(out / "openj9_train_50.csv", index=False)
    test[cols].to_csv(out / "openj9_test_50.csv", index=False)
    logger.success("Escrito artifacts/openj9/openj9_{{train,test}}_50.csv")

    # ---- split de validación temporal (cola del train) para sintonizar HPs ----
    if args.val_frac > 0:
        # val_cut = cuantil temporal del train; los issue_number >= val_cut (los más
        # recientes, justo antes de test) son validación. El índice de tuning usa solo
        # traininner (< val_cut) → val no recupera casos futuros (mismo protocolo que test).
        val_cut = int(train["issue_number"].quantile(1 - args.val_frac))
        train_inner = train[train.issue_number < val_cut].copy()
        val = train[train.issue_number >= val_cut].copy()
        train_inner[cols].to_csv(out / "openj9_traininner_50.csv", index=False)
        val[cols].to_csv(out / "openj9_val_50.csv", index=False)
        logger.success(
            "Validación temporal: val_cut={} | traininner={} val={} ({:.0%}) | "
            "test queda intacto ({}). Tuning: --train-csv traininner --test-csv val; "
            "reporte: --train-csv train_50 --test-csv test_50.",
            val_cut, len(train_inner), len(val), len(val) / len(train), len(test))


if __name__ == "__main__":
    main()
