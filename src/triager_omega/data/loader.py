"""Carga perezosa por columnas de los tres parquets de BugsRepo.

Los nombres de archivo y rutas vienen de `config.settings`. Todas las funciones
permiten seleccionar columnas para no cargar tablas enteras en memoria.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from triager_omega.config import Settings, settings

# Columnas relevantes por defecto (subconjunto del esquema completo).
BUG_COLUMNS = [
    "Bug Id",
    "Summary",
    "Product",
    "Component",
    "Priority",
    "Severity",
    "Assigned To",
    "Creation Time",
]

CONTRIBUTOR_COLUMNS = [
    "Contributor Id",
    "User Name",
    "Assigned To and Fixed",
    "Last Activity",
]

COMMENT_COLUMNS = [
    "Bug Id",
    "Author Id",
    "Creator",
    "Time",
    "Bug Report",
]


def load_bugs(columns: list[str] | None = BUG_COLUMNS, cfg: Settings = settings) -> pd.DataFrame:
    """Carga la tabla principal de bugs (PK: ``Bug Id``)."""
    logger.debug("Cargando bugs desde {}", cfg.bugs_path)
    return pd.read_parquet(cfg.bugs_path, columns=columns)


def load_contributors(
    columns: list[str] | None = CONTRIBUTOR_COLUMNS, cfg: Settings = settings
) -> pd.DataFrame:
    """Carga el perfil de contribuidores (PK: ``Contributor Id``)."""
    logger.debug("Cargando contributors desde {}", cfg.contributors_path)
    return pd.read_parquet(cfg.contributors_path, columns=columns)


def load_comments(
    columns: list[str] | None = COMMENT_COLUMNS, cfg: Settings = settings
) -> pd.DataFrame:
    """Carga los comentarios (PK: ``Comment Id``; FK: ``Bug Id``, ``Author Id``)."""
    logger.debug("Cargando comments desde {}", cfg.comments_path)
    return pd.read_parquet(cfg.comments_path, columns=columns)


def build_email_to_id(comments: pd.DataFrame | None = None, cfg: Settings = settings) -> pd.Series:
    """Construye el puente ``email -> Contributor Id``.

    El label ``Assigned To`` de los bugs son emails, pero la clave canónica del
    sistema es el ``Contributor Id`` entero (usado en comments.Author Id y en la
    PK de contributors). La tabla de comentarios contiene ambos: ``Creator``
    (email) y ``Author Id`` (entero), por lo que sirve de puente.

    Devuelve una Series indexada por email cuyo valor es el ``Contributor Id``.
    """
    if comments is None:
        comments = load_comments(columns=["Author Id", "Creator"], cfg=cfg)

    bridge = comments.dropna(subset=["Author Id", "Creator"]).copy()
    bridge["Author Id"] = bridge["Author Id"].astype("int64")
    bridge["Creator"] = bridge["Creator"].astype("string")
    bridge = bridge.drop_duplicates(subset="Creator")
    mapping = bridge.set_index("Creator")["Author Id"]
    logger.debug("Puente email->id construido con {} emails únicos", mapping.shape[0])
    return mapping
