"""Construye el subconjunto PILOTO del CBR (escala TriagerX).

Toma el `splits.parquet` del Módulo 1 (450 devs) y lo recorta a:
  - los `pilot_n_devs` desarrolladores más activos en train (TriagerX usó 17-41),
  - con un tope de `pilot_cap` bugs/dev en train (los MÁS RECIENTES, para respetar
    la temporalidad y aplanar la cabeza).
Los bugs de val/test de esos devs se conservan completos (evaluación realista).

Salida (en artifacts/pilot/):
  - splits.parquet        (Bug Id, contributor_id, creation_time, split)
  - label_encoder.json    (Contributor Id str -> class_idx, sobre los devs piloto)

Ejecutar:  uv run python -m triager_omega.cbr.pilot
"""

from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from triager_omega.config import Settings, settings


def build_pilot(cfg: Settings = settings) -> dict:
    cfg.pilot_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.read_parquet(cfg.splits_path)
    train = splits[splits["split"] == "train"]

    # top-N devs por frecuencia en train.
    top_devs = train["contributor_id"].value_counts().head(cfg.pilot_n_devs).index.tolist()
    logger.info("Top {} devs seleccionados (de {} totales).",
                len(top_devs), train["contributor_id"].nunique())

    sub = splits[splits["contributor_id"].isin(top_devs)].copy()

    # Cap por split (los bugs MÁS RECIENTES por dev): train a `pilot_cap`,
    # val/test a `pilot_eval_cap` (acota el coste de destilación sin perder
    # poder de evaluación; cientos de bugs de test ya dan Hit@K estable).
    def cap_split(name: str, cap: int) -> pd.DataFrame:
        part = sub[sub["split"] == name].sort_values("creation_time")
        return part.groupby("contributor_id", group_keys=False).tail(cap)

    tr_capped = cap_split("train", cfg.pilot_cap)
    pilot = pd.concat(
        [tr_capped, cap_split("val", cfg.pilot_eval_cap), cap_split("test", cfg.pilot_eval_cap)]
    ).sort_values("creation_time").reset_index(drop=True)

    # label encoder sobre los devs piloto (ordenado para estabilidad).
    devs = sorted(int(d) for d in top_devs)
    label_encoder = {str(d): i for i, d in enumerate(devs)}

    # persistencia.
    pilot.to_parquet(cfg.pilot_splits_path, index=False)
    cfg.pilot_label_encoder_path.write_text(json.dumps(label_encoder, indent=2), encoding="utf-8")

    counts = pilot["split"].value_counts().to_dict()
    logger.success("Piloto escrito en {}", cfg.pilot_dir)
    logger.info("  splits: train={} val={} test={}",
                counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))
    logger.info("  total bugs a destilar: {}", len(pilot))
    logger.info("  bugs/dev en train (cap {}): min={} max={}",
                cfg.pilot_cap,
                int(tr_capped["contributor_id"].value_counts().min()),
                int(tr_capped["contributor_id"].value_counts().max()))
    return {"n_devs": len(devs), "n_bugs": len(pilot), **counts}


if __name__ == "__main__":
    build_pilot()
