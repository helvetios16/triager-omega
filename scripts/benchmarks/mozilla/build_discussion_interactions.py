"""Construye la señal `discussion` del IBR a partir de bug_comments (PLAN §7.2).

Limpia los comentarios de Bugzilla y los normaliza al MISMO esquema tipado que
`repo_interactions.parquet` (commit/review de gecko-dev), para que luego la
Interaction Table se arme con un simple concat:

    bug_id | Contributor Id | kind | timestamp

Dos limpiezas (descubiertas inspeccionando los datos):

1. BOTS — los comentarios de cuentas automáticas no son interacción humana y
   contaminarían la señal. En este dataset los bots viven en cuatro dominios con
   TLD anonimizado falso, cada uno con solo 4-5 direcciones, TODAS automatización:
   @bmo.tld (pulsebot, phab-bot, github-automation, update-bot, automation),
   @mozilla.bugs (wptsync, intermittent-bug-filer, pulgasaur, telemetry-probes),
   @bots.tld (orangefactor, ccadb2onercl, error-propagation, jira-integration),
   @mozilla.tld (release-mgmt-account-bot, autonag-nomail-bot).
   Filtrar por DOMINIO es preciso: NO toca humanos como botond@mozilla.com ni los
   @orange.fr (un filtro por substring 'bot'/'orange' sí los borraría por error).

2. DUPLICADOS — el parquet crudo trae cada comentario repetido (~58% de filas son
   duplicados exactos). Se eliminan con drop_duplicates.

Además, para el IBR se colapsa a UNA interacción discussion por (bug, dev) usando
el comentario MÁS ANTIGUO como timestamp: que un dev comente 10 veces en un bug no
debe multiplicar por 10 su crédito (sesgaría hacia bugs charlatanes).

Salida: artifacts/discussion_interactions.parquet

Ejecutar:
    uv run python scripts/build_discussion_interactions.py
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from triager_omega.config import settings

# Dominios cuyo 100% de direcciones son bots/automatización (ver docstring).
BOT_DOMAINS = frozenset({"bmo.tld", "mozilla.bugs", "bots.tld", "mozilla.tld"})


def main() -> None:
    cols = ["Bug Id", "Author Id", "Creator", "Time"]
    bc = pd.read_parquet(settings.comments_path, columns=cols)
    n0 = len(bc)
    logger.info("Comentarios crudos: {}", n0)

    # 1) Dedup exacto (el parquet trae cada comentario repetido).
    bc = bc.drop_duplicates()
    logger.info("Tras dedup exacto: {} (-{})", len(bc), n0 - len(bc))

    # 2) Filtrar bots por dominio del email.
    creator = bc["Creator"].astype("string").fillna("")
    domain = creator.str.rsplit("@", n=1).str[-1]
    is_bot = domain.isin(BOT_DOMAINS)
    bc = bc[~is_bot]
    logger.info("Tras filtrar bots ({} dominios): {} (-{} comentarios bot)",
                len(BOT_DOMAINS), len(bc), int(is_bot.sum()))

    # 3) Necesitamos Author Id (= Contributor Id) y Time para atribuir + decaer.
    bc = bc.dropna(subset=["Author Id", "Bug Id"]).copy()
    bc["Contributor Id"] = bc["Author Id"].astype("int64")
    bc["bug_id"] = bc["Bug Id"].astype("int64")
    bc["timestamp"] = pd.to_datetime(bc["Time"], utc=True, errors="coerce")
    bc = bc.dropna(subset=["timestamp"])

    # 4) Colapsar a UNA discussion por (bug, dev) con el comentario más antiguo.
    disc = (
        bc.groupby(["bug_id", "Contributor Id"], as_index=False)["timestamp"].min()
    )
    disc["kind"] = "discussion"
    disc = disc[["bug_id", "Contributor Id", "kind", "timestamp"]]

    out_path = settings.artifacts_dir / "discussion_interactions.parquet"
    disc.to_parquet(out_path, index=False)

    logger.info("== Resumen ==")
    logger.info("Interacciones discussion (bug,dev) únicas: {}", len(disc))
    logger.info("Bugs con discusión: {}", disc["bug_id"].nunique())
    logger.info("Devs con discusión: {}", disc["Contributor Id"].nunique())
    logger.info("Escrito -> {}", out_path)


if __name__ == "__main__":
    main()
