"""Inspecciona un bug y muestra su relación entre las 3 fuentes del IBR.

Para un Bug Id dado, vuelca lado a lado:
  FUENTE 1 — BugsRepo (bug_metadata.parquet): campos clave + asignado.
  FUENTE 2 — gecko-dev (repo_interactions.parquet): commits / reviews.
  FUENTE 3 — discussion_interactions.parquet: comentaristas (limpios, sin bots).
Y una verificación de consistencia: ¿el asignado de Bugzilla es quien commiteó el
fix en gecko-dev? ¿quién aparece en más de una fuente?

Sirve para auditar a mano que el puente bug_id ↔ Contributor Id está bien.

Ejecutar:
    uv run python scripts/inspect_bug.py 1703327
    uv run python scripts/inspect_bug.py              # usa el bug por defecto
"""

from __future__ import annotations

import argparse
import ast

import pandas as pd
from loguru import logger

from triager_omega.config import settings

# Pesos por tipo (de config.py) para anotar la fuerza de cada interacción.
# commit y review comparten ip_contribution (fusión estilo TriagerX).
IP = {
    "commit": settings.ip_contribution,
    "review": settings.ip_contribution,
    "assignment": settings.ip_assignment,
    "discussion": settings.ip_discussion,
}

# Campos de bug_metadata que se muestran (los relevantes para el triaje).
META_FIELDS = [
    "Bug Id", "Summary", "Product", "Component", "Status", "Resolution",
    "Assigned To", "Creator", "Type", "Severity", "Priority",
    "Comment Count", "Creation Time", "Keywords",
]


def name_lookup() -> dict[int, str]:
    """Contributor Id -> User Name (sin duplicados)."""
    co = pd.read_parquet(
        settings.contributors_path, columns=["Contributor Id", "User Name"]
    ).drop_duplicates("Contributor Id")
    return dict(zip(co["Contributor Id"], co["User Name"]))


def assignee_id(detail: object) -> int | None:
    """Saca el Contributor Id del campo 'Assigned To Detail' (dict serializado)."""
    try:
        d = detail if isinstance(detail, dict) else ast.literal_eval(str(detail))
        return int(d.get("id")) if d.get("id") is not None else None
    except (ValueError, SyntaxError, TypeError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bug_id", type=int, nargs="?", default=1703327)
    args = p.parse_args()
    bug = args.bug_id
    name = name_lookup()

    # ---- FUENTE 1: BugsRepo ----
    bm = pd.read_parquet(settings.bugs_path, columns=None)
    row = bm[bm["Bug Id"] == bug]
    if row.empty:
        logger.error("Bug {} no está en bug_metadata.parquet", bug)
        raise SystemExit(1)
    s = row.iloc[0]
    assignee = assignee_id(s.get("Assigned To Detail"))

    print("#" * 74)
    print(f"#  FUENTE 1 — BugsRepo (bug_metadata): bug {bug}")
    print("#" * 74)
    for c in META_FIELDS:
        if c in bm.columns:
            v = str(s[c])
            v = v[:160] + " ..." if len(v) > 160 else v
            print(f"{c:>18} : {v}")
    print(f"{'-> Assigned To id':>18} : {assignee}  ({name.get(assignee, '??')})")

    # ---- FUENTE 2: gecko-dev ----
    print("\n" + "#" * 74)
    print("#  FUENTE 2 — gecko-dev (repo_interactions): commits / reviews")
    print("#" * 74)
    ri = pd.read_parquet(settings.artifacts_dir / "repo_interactions.parquet")
    gk = ri[ri["bug_id"] == bug].sort_values("timestamp").drop_duplicates()
    if gk.empty:
        print("  (sin commits/reviews para este bug)")
    else:
        gk = gk.copy()
        gk["quien"] = gk["Contributor Id"].map(name)
        gk["IP"] = gk["kind"].map(IP)
        print(gk[["kind", "IP", "raw_actor", "Contributor Id", "quien",
                  "commit_hash", "timestamp"]].to_string(index=False))

    # ---- FUENTE 3: discussion ----
    print("\n" + "#" * 74)
    print("#  FUENTE 3 — discussion_interactions: comentaristas (limpios)")
    print("#" * 74)
    di = pd.read_parquet(settings.artifacts_dir / "discussion_interactions.parquet")
    dd = di[di["bug_id"] == bug].sort_values("timestamp").copy()
    if dd.empty:
        print("  (este bug no tiene comentarios en el dump, o todos eran bots)")
    else:
        dd["quien"] = dd["Contributor Id"].map(name)
        dd["IP"] = IP["discussion"]
        print(dd[["Contributor Id", "quien", "IP", "timestamp"]].to_string(index=False))

    # ---- Verificación de consistencia ----
    print("\n" + "#" * 74)
    print("#  VERIFICACIÓN DE RELACIÓN")
    print("#" * 74)
    committers = set(gk[gk["kind"] == "commit"]["Contributor Id"]) if not gk.empty else set()
    reviewers = set(gk[gk["kind"] == "review"]["Contributor Id"]) if not gk.empty else set()
    discussers = set(dd["Contributor Id"]) if not dd.empty else set()

    print(f"  ¿el asignado ({name.get(assignee, assignee)}) commiteó el fix?  ",
          "SÍ ✅" if assignee in committers else "no")
    print(f"  ¿el asignado participó en la discusión?                   ",
          "SÍ ✅" if assignee in discussers else "no")

    # Personas en más de una fuente (consistencia cruzada).
    multi = {}
    for dev in committers | reviewers | discussers:
        fuentes = []
        if dev in committers:
            fuentes.append("commit")
        if dev in reviewers:
            fuentes.append("review")
        if dev in discussers:
            fuentes.append("discussion")
        if len(fuentes) > 1:
            multi[dev] = fuentes
    if multi:
        print("  personas en MÚLTIPLES fuentes (señal robusta):")
        for dev, fuentes in multi.items():
            print(f"     - {name.get(dev, dev)}: {', '.join(fuentes)}")
    else:
        print("  (nadie aparece en más de una fuente)")


if __name__ == "__main__":
    main()
