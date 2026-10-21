"""Módulo 1 — Preprocesamiento y balanceo.

Produce los artefactos base que consumen los módulos posteriores:

- ``active_candidates.json``   directorio de desarrolladores elegibles (filtro de candidatos).
- ``label_encoder.json``       mapeo Contributor Id <-> índice de clase para DeBERTa.
- ``splits.parquet``           bugs filtrados con su split temporal (train/val/test) y peso.
- ``sample_weights_train.npy`` pesos de muestreo (1/frecuencia) alineados al split de train.

Ejecutar:  uv run python -m triager_omega.data.preprocessor
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from loguru import logger

from triager_omega.config import Settings, settings
from triager_omega.data import loader

_URL_RE = re.compile(r"https?://\S+")


# --------------------------------------------------------------------------- #
# Identidad
# --------------------------------------------------------------------------- #
def normalize_identity(
    bugs: pd.DataFrame, email_to_id: pd.Series, cfg: Settings = settings
) -> pd.DataFrame:
    """Añade la columna canónica ``contributor_id`` (Int64) mapeando el email
    de ``Assigned To`` mediante el puente de comentarios.

    Marca también ``is_automated`` para asignados a usuarios automáticos
    (p. ej. nobody@mozilla.org).
    """
    out = bugs.copy()
    at = out["Assigned To"].astype("string")

    markers = "|".join(re.escape(m) for m in cfg.automated_user_markers)
    out["is_automated"] = at.str.contains(markers, case=False, na=False)

    out["contributor_id"] = at.map(email_to_id).astype("Int64")
    return out


# --------------------------------------------------------------------------- #
# Directorio activo de candidatos
# --------------------------------------------------------------------------- #
def build_active_directory(
    contributors: pd.DataFrame, cfg: Settings = settings, reference_time: pd.Timestamp | None = None
) -> set[int]:
    """Conjunto de ``Contributor Id`` elegibles como recomendación.

    Criterio: ``Assigned To and Fixed >= active_threshold``. Opcionalmente,
    actividad reciente dentro de ``last_activity_horizon_months`` respecto a
    ``reference_time`` (último bug del train).
    """
    df = contributors
    mask = df["Assigned To and Fixed"] >= cfg.active_threshold

    if cfg.last_activity_horizon_months is not None and reference_time is not None:
        last = pd.to_datetime(df["Last Activity"], errors="coerce", utc=True)
        horizon = reference_time - pd.DateOffset(months=cfg.last_activity_horizon_months)
        mask &= last >= horizon

    active = set(df.loc[mask, "Contributor Id"].astype("int64").tolist())
    logger.info("Directorio activo: {} devs (umbral≥{})", len(active), cfg.active_threshold)
    return active


# --------------------------------------------------------------------------- #
# Filtrado de bugs
# --------------------------------------------------------------------------- #
def filter_bugs(
    bugs: pd.DataFrame, active_ids: set[int], cfg: Settings = settings
) -> pd.DataFrame:
    """Aplica las reglas de descarte del Módulo 1 y registra cuántos caen por cada motivo."""
    n0 = len(bugs)
    summary = bugs["Summary"].astype("string")

    drop_summary = summary.isna() | (summary.str.strip() == "")
    drop_automated = bugs["is_automated"]
    drop_unmapped = bugs["contributor_id"].isna()
    drop_inactive = ~bugs["contributor_id"].isin(active_ids)

    keep = ~(drop_summary | drop_automated | drop_unmapped | drop_inactive)

    logger.info("Filtrado de bugs ({} iniciales):", n0)
    logger.info("  - sin summary:            {}", int(drop_summary.sum()))
    logger.info("  - asignado automático:    {}", int(drop_automated.sum()))
    logger.info("  - email sin id mapeado:   {}", int(drop_unmapped.sum()))
    logger.info("  - dev fuera de directorio:{}", int((drop_inactive & ~drop_unmapped).sum()))
    logger.info("  => conservados:           {} ({:.1f}%)", int(keep.sum()), keep.mean() * 100)

    out = bugs.loc[keep].copy()
    out["contributor_id"] = out["contributor_id"].astype("int64")
    return out


# --------------------------------------------------------------------------- #
# Split temporal
# --------------------------------------------------------------------------- #
def temporal_split(bugs: pd.DataFrame, cfg: Settings = settings) -> pd.DataFrame:
    """Ordena por ``Creation Time`` y asigna split train/val/test cronológico.

    Evita la fuga temporal: el triaje real es prospectivo.
    """
    out = bugs.copy()
    out["creation_time"] = pd.to_datetime(out["Creation Time"], errors="coerce", utc=True)
    out = out.sort_values("creation_time", kind="stable").reset_index(drop=True)

    n = len(out)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_val] = "val"
    split[n_train + n_val :] = "test"
    out["split"] = split

    logger.info(
        "Split temporal: train={} val={} test={}",
        int((split == "train").sum()),
        int((split == "val").sum()),
        int((split == "test").sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# Pesos de muestreo (balanceo de cola larga)
# --------------------------------------------------------------------------- #
def compute_sample_weights(train: pd.DataFrame) -> np.ndarray:
    """w(b) = 1 / freq_train(dev). Devuelve un array alineado a las filas de `train`."""
    freq = train["contributor_id"].value_counts()
    weights = (1.0 / train["contributor_id"].map(freq)).to_numpy(dtype=np.float64)
    return weights


# --------------------------------------------------------------------------- #
# Codificador de etiquetas
# --------------------------------------------------------------------------- #
def build_label_encoder(train: pd.DataFrame) -> dict[str, int]:
    """Mapeo ``Contributor Id (str) -> class_idx`` sobre los devs presentes en train.

    Se usan claves str para serializar a JSON sin ambigüedad.
    """
    devs = sorted(train["contributor_id"].unique().tolist())
    return {str(dev): idx for idx, dev in enumerate(devs)}


# --------------------------------------------------------------------------- #
# Limpieza de texto (mínima; reutilizable por módulos posteriores)
# --------------------------------------------------------------------------- #
def clean_text(text: str | None, cfg: Settings = settings) -> str:
    """Normaliza URLs y trunca stacktraces largos. Conserva el contenido semántico."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = _URL_RE.sub("<URL>", s)
    lines = s.splitlines()
    if len(lines) > cfg.max_stacktrace_lines:
        lines = lines[: cfg.max_stacktrace_lines]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def run(cfg: Settings = settings) -> dict:
    """Ejecuta el Módulo 1 completo y materializa los artefactos."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("== Módulo 1: preprocesamiento y balanceo ==")
    bugs = loader.load_bugs(cfg=cfg)
    contributors = loader.load_contributors(cfg=cfg)
    email_to_id = loader.build_email_to_id(cfg=cfg)

    bugs = normalize_identity(bugs, email_to_id, cfg=cfg)
    active_ids = build_active_directory(contributors, cfg=cfg)
    bugs = filter_bugs(bugs, active_ids, cfg=cfg)
    bugs = temporal_split(bugs, cfg=cfg)

    train = bugs[bugs["split"] == "train"]
    weights = compute_sample_weights(train)
    label_encoder = build_label_encoder(train)

    # --- cobertura del directorio en cada split (techo de Hit@K) ---
    classes = set(int(k) for k in label_encoder)
    for s in ("train", "val", "test"):
        sub = bugs[bugs["split"] == s]
        cov = sub["contributor_id"].isin(classes).mean() * 100 if len(sub) else 0.0
        logger.info("Cobertura clases en {}: {:.1f}% ({} bugs)", s, cov, len(sub))

    # --- persistencia ---
    cfg.active_candidates_path.write_text(
        json.dumps(sorted(active_ids), indent=2), encoding="utf-8"
    )
    cfg.label_encoder_path.write_text(
        json.dumps(label_encoder, indent=2), encoding="utf-8"
    )
    bugs_out = bugs[["Bug Id", "contributor_id", "creation_time", "split"]].copy()
    bugs_out.to_parquet(cfg.splits_path, index=False)
    np.save(cfg.sample_weights_path, weights)

    logger.success("Artefactos escritos en {}", cfg.artifacts_dir)
    logger.info("  - {}", cfg.active_candidates_path.name)
    logger.info("  - {}", cfg.label_encoder_path.name)
    logger.info("  - {} ({} filas)", cfg.splits_path.name, len(bugs_out))
    logger.info("  - {} ({} pesos)", cfg.sample_weights_path.name, len(weights))

    return {
        "n_active": len(active_ids),
        "n_classes": len(label_encoder),
        "n_bugs": len(bugs_out),
        "n_train": len(train),
    }


if __name__ == "__main__":
    run()
