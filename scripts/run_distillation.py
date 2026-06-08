"""Destila el subconjunto piloto (Módulo 2) con caché incremental y reanudable.

Carga los bugs del piloto (artifacts/pilot/splits.parquet), su texto crudo y el
primer comentario, los destila con el backend configurado y guarda en
artifacts/pilot/distillations.parquet (idempotente: omite los ya destilados).

Ejecutar:
    uv run python scripts/run_distillation.py --smoke 5         # prueba 5 bugs, imprime
    uv run python scripts/run_distillation.py                   # batch completo (reanudable)
    uv run python scripts/run_distillation.py --backend google  # forzar backend
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from loguru import logger

from triager_omega.cbr import distillation as D
from triager_omega.config import settings
from triager_omega.data import loader


def load_first_comments(bug_ids: set[int]) -> dict[int, str]:
    """Primer comentario (preferentemente el reporte inicial) por bug."""
    bc = pd.read_parquet(
        settings.comments_path, columns=["Bug Id", "Time", "Bug Report", "Text"]
    )
    bc = bc[bc["Bug Id"].isin(bug_ids)].dropna(subset=["Text"]).drop_duplicates()
    bc["Time"] = pd.to_datetime(bc["Time"], utc=True, errors="coerce")
    # prioriza Bug Report==True; dentro de eso, el más antiguo.
    bc = bc.sort_values(["Bug Report", "Time"], ascending=[False, True])
    first = bc.groupby("Bug Id").first()
    return first["Text"].to_dict()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", type=int, default=0, help="solo N bugs, imprime resultados")
    p.add_argument("--backend", choices=["ollama", "lmstudio", "google"], default=None)
    p.add_argument("--flush-every", type=int, default=50, help="guardar cache cada N bugs")
    args = p.parse_args()
    if args.backend:
        settings.distill_backend = args.backend

    settings.pilot_dir.mkdir(parents=True, exist_ok=True)

    pilot = pd.read_parquet(settings.pilot_splits_path)
    pilot = pilot.drop_duplicates(subset="Bug Id")  # por si splits trae bug_ids repetidos
    bug_ids = set(pilot["Bug Id"].tolist())
    bugs = loader.load_bugs(
        columns=["Bug Id", "Summary", "Product", "Component", "Severity", "Priority"]
    )
    # bug_metadata trae filas duplicadas por Bug Id → deduplicar antes de indexar.
    bugs = bugs[bugs["Bug Id"].isin(bug_ids)].drop_duplicates(subset="Bug Id").set_index("Bug Id")

    logger.info("Cargando primeros comentarios de {} bugs...", len(bug_ids))
    comments = load_first_comments(bug_ids)

    # cache existente (reanudable).
    done: dict[int, dict] = {}
    if settings.distillations_path.exists():
        cached = pd.read_parquet(settings.distillations_path)
        done = {int(r["Bug Id"]): r.to_dict() for _, r in cached.iterrows()}
        logger.info("Cache: {} bugs ya destilados.", len(done))

    todo = [b for b in pilot["Bug Id"].tolist() if b not in done]
    if args.smoke:
        todo = todo[: args.smoke]
    logger.info("Backend={} | a destilar: {} bugs", settings.distill_backend, len(todo))

    client = D.make_client()
    rows = list(done.values())
    n_fallback = 0

    for i, bug_id in enumerate(todo, 1):
        bug = bugs.loc[bug_id]
        user_input = D.build_input(bug, comments.get(bug_id))
        d = D.distill_one(client, user_input, bug)
        text = D.to_distilled_text(d)
        is_fb = bool(d.get("_fallback"))
        n_fallback += is_fb
        rows.append({
            "Bug Id": int(bug_id),
            "distilled_json": json.dumps(d, ensure_ascii=False),
            "distilled_text": text,
            "fallback": is_fb,
        })

        if args.smoke:
            print(f"\n----- bug {bug_id} -----")
            print("INPUT:\n", user_input[:400])
            print("DESTILADO:", text)
        if not args.smoke and (i % args.flush_every == 0 or i == len(todo)):
            pd.DataFrame(rows).to_parquet(settings.distillations_path, index=False)
            logger.info("  {}/{} destilados (fallbacks: {})", i, len(todo), n_fallback)

    if not args.smoke:
        pd.DataFrame(rows).to_parquet(settings.distillations_path, index=False)
        logger.success("Destilación completa: {} bugs, {} fallbacks -> {}",
                       len(rows), n_fallback, settings.distillations_path)


if __name__ == "__main__":
    main()
