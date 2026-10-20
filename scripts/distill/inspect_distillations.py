"""Inspecciona el resultado de la destilación (Módulo 2).

Lee artifacts/pilot/distillations.parquet y muestra: cobertura, tasa de fallback,
estadísticas de longitud y una muestra de la tabla (crudo -> destilado).

Ejecutar:
    uv run python scripts/inspect_distillations.py            # resumen + 10 muestras
    uv run python scripts/inspect_distillations.py --n 20     # 20 muestras
    uv run python scripts/inspect_distillations.py --fallbacks # solo muestra fallbacks
    uv run python scripts/inspect_distillations.py --full      # texto destilado completo
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from triager_omega.config import settings
from triager_omega.data import loader


def _trunc(s: str, n: int) -> str:
    s = " ".join(str(s).split())  # colapsa saltos de línea/espacios
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10, help="número de muestras a mostrar")
    p.add_argument("--fallbacks", action="store_true", help="mostrar solo bugs con fallback")
    p.add_argument("--full", action="store_true", help="no truncar el texto destilado")
    args = p.parse_args()

    path = settings.distillations_path
    if not path.exists():
        raise SystemExit(f"No existe {path}. Corre primero scripts/run_distillation.py")

    df = pd.read_parquet(path)
    n_total = len(df)
    n_unique = df["Bug Id"].nunique()
    n_fb = int(df["fallback"].sum())

    # cobertura respecto al piloto.
    pilot = pd.read_parquet(settings.pilot_splits_path).drop_duplicates(subset="Bug Id")
    n_pilot = pilot["Bug Id"].nunique()
    pendientes = n_pilot - n_unique

    # longitudes del texto destilado.
    lens = df["distilled_text"].str.len()

    print("=" * 70)
    print(f"ARCHIVO: {path}")
    print("=" * 70)
    print(f"Bugs destilados      : {n_total} ({n_unique} únicos)")
    print(f"Piloto total         : {n_pilot}")
    print(f"Cobertura            : {n_unique}/{n_pilot} ({100 * n_unique / n_pilot:.1f}%)"
          + (f"  | PENDIENTES: {pendientes}" if pendientes else "  | COMPLETO ✓"))
    print(f"Fallbacks            : {n_fb} ({100 * n_fb / n_total:.1f}%)  "
          f"[LLM OK: {n_total - n_fb}]")
    print(f"Texto destilado len  : min={lens.min()}  media={lens.mean():.0f}  "
          f"máx={lens.max()}")

    # estadísticas de campos del JSON.
    n_sym, n_cap, n_sub = [], [], []
    for js in df["distilled_json"]:
        try:
            d = json.loads(js)
        except (json.JSONDecodeError, TypeError):
            continue
        n_sym.append(len(d.get("symptoms") or []))
        n_cap.append(len(d.get("capabilities") or []))
        n_sub.append(1 if (d.get("fault_location", {}) or {}).get("subsystem") else 0)
    if n_sym:
        s, c = pd.Series(n_sym), pd.Series(n_cap)
        print(f"Síntomas por bug     : media={s.mean():.2f}  (0 síntomas: {int((s == 0).sum())})")
        print(f"Capabilities por bug : media={c.mean():.2f}  (0 caps: {int((c == 0).sum())})")
        print(f"Subsystem no vacío   : {sum(n_sub)}/{len(n_sub)} "
              f"({100 * sum(n_sub) / len(n_sub):.1f}%)")
    print("=" * 70)

    # muestra de la tabla, con el Summary crudo para comparar.
    sample = df[df["fallback"]] if args.fallbacks else df
    if sample.empty:
        print("\n(no hay filas que mostrar con ese filtro)")
        return
    sample = sample.head(args.n)

    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])
    bugs = bugs.drop_duplicates(subset="Bug Id").set_index("Bug Id")

    title = "MUESTRA (solo fallbacks)" if args.fallbacks else "MUESTRA"
    print(f"\n{title} — {len(sample)} bugs (crudo → destilado):\n")
    for _, r in sample.iterrows():
        bid = int(r["Bug Id"])
        summary = bugs.loc[bid]["Summary"] if bid in bugs.index else "(sin metadata)"
        flag = " [FALLBACK]" if r["fallback"] else ""
        text = r["distilled_text"] if args.full else _trunc(r["distilled_text"], 160)
        print(f"#{bid}{flag}")
        print(f"  CRUDO : {_trunc(summary, 120)}")
        print(f"  DEST. : {text}")
        print()


if __name__ == "__main__":
    main()
