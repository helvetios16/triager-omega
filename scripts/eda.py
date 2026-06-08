"""EDA del dataset BugsRepo

Estadísticas clave para validar los supuestos del Módulo 1:
distribución de `Assigned To`, usuarios automáticos, puente de identidad,
cobertura del directorio activo a distintos umbrales, splits temporales y
disponibilidad de comentarios.
"""

from __future__ import annotations

import pandas as pd

from triager_omega.config import settings
from triager_omega.data import loader
from triager_omega.data.preprocessor import normalize_identity


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    cfg = settings
    print(f"DATA_DIR = {cfg.data_dir}")

    bugs = loader.load_bugs(cfg=cfg)
    contributors = loader.load_contributors(cfg=cfg)
    email_to_id = loader.build_email_to_id(cfg=cfg)

    # ---------------------------------------------------------------- #
    hr("1. Volúmenes")
    print(f"  bugs:         {len(bugs):>10,}")
    print(f"  contributors: {len(contributors):>10,}")
    print(f"  emails puente (comments): {email_to_id.shape[0]:>10,}")

    # ---------------------------------------------------------------- #
    hr("2. Label 'Assigned To'")
    at = bugs["Assigned To"].astype("string")
    nobody = at.str.contains("nobody", case=False, na=False)
    print(f"  nulos:            {int(at.isna().sum()):>10,}")
    print(f"  vacíos:           {int((at.fillna('') == '').sum()):>10,}")
    print(f"  'nobody':         {int(nobody.sum()):>10,}  ({nobody.mean() * 100:.1f}%)")
    valid = at[~at.isna() & (at != '') & ~nobody]
    print(f"  con asignado real:{len(valid):>10,}  ({len(valid) / len(bugs) * 100:.1f}%)")

    # ---------------------------------------------------------------- #
    hr("3. Cola larga (devs con asignado real)")
    vc = valid.value_counts()
    print(f"  devs únicos:      {len(vc):>10,}")
    print(f"  máx bugs/dev:     {int(vc.iloc[0]):>10,}")
    print(f"  mediana bugs/dev: {vc.median():>10.0f}")
    print(f"  p90 bugs/dev:     {vc.quantile(0.9):>10.0f}")
    top1 = vc.head(max(1, len(vc) // 100)).sum()
    print(f"  bugs cubiertos por top-1% devs: {top1 / vc.sum() * 100:.1f}%")
    for thr in (10, 20, 50):
        print(f"  devs con ≥{thr:>2} bugs: {int((vc >= thr).sum()):>6,}")

    # ---------------------------------------------------------------- #
    hr("4. Identidad email -> Contributor Id")
    print(f"  Assigned To con '@': {at.str.contains('@', na=False).mean() * 100:.1f}%")
    mapped = valid.map(email_to_id)
    print(f"  asignados reales mapeables a id: {mapped.notna().mean() * 100:.1f}%")

    # ---------------------------------------------------------------- #
    hr("5. Directorio activo (cobertura sobre bugs con asignado real)")
    bugs_id = normalize_identity(bugs, email_to_id, cfg=cfg)
    real = ~bugs_id["is_automated"] & bugs_id["contributor_id"].notna()
    n_real = int(real.sum())
    print(f"  bugs con asignado real e id: {n_real:,}")
    for thr in (10, 20, 50):
        active = set(
            contributors.loc[contributors["Assigned To and Fixed"] >= thr, "Contributor Id"]
            .astype("int64")
            .tolist()
        )
        cov = bugs_id.loc[real, "contributor_id"].isin(active)
        flag = "  <- elegido" if thr == cfg.active_threshold else ""
        print(
            f"  umbral≥{thr:>2}: {len(active):>4,} devs | "
            f"cobertura={cov.sum() / n_real * 100:5.1f}% ({int(cov.sum()):,}/{n_real:,}){flag}"
        )

    # ---------------------------------------------------------------- #
    hr("6. Fechas y split temporal")
    ct = pd.to_datetime(bugs["Creation Time"], errors="coerce", utc=True)
    print(f"  Creation Time parseables: {ct.notna().mean() * 100:.1f}%")
    print(f"  rango: {ct.min()}  ->  {ct.max()}")

    # ---------------------------------------------------------------- #
    hr("7. Comentarios")
    comments = loader.load_comments(cfg=cfg)
    print(f"  comentarios totales:        {len(comments):>12,}")
    print(f"  reportes iniciales (Bug Report=True): {int(comments['Bug Report'].sum()):>12,}")
    auth = comments["Author Id"].dropna().astype("int64")
    cid = set(contributors["Contributor Id"].astype("int64"))
    print(f"  Author Id en contributors PK: {auth.isin(cid).mean() * 100:.1f}%")
    bugs_con_comentarios = comments["Bug Id"].nunique()
    print(f"  bugs con ≥1 comentario: {bugs_con_comentarios:,} ({bugs_con_comentarios / len(bugs) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("  EDA completado")
    print("=" * 70)


if __name__ == "__main__":
    main()
