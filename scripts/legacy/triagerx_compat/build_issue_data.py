"""Convierte nuestro caché crudo de timelines (artifacts/openj9/raw/{n}.json con
claves issue_number/created_at/assignees/timeline) al formato que el IBR de TriagerX
espera (`{n}.json` con `assignees` + `timeline_data`), para poder correr su
`triagerx/system/triagerx.py` sobre EXACTAMENTE los mismos datos minados.

El timeline crudo ya viene en el formato del GitHub Timeline API (event/actor/
created_at/source/commit_url), que es justo lo que `_get_contribution_data` parsea;
solo hay que renombrar `timeline` -> `timeline_data`.

Uso (en omen):
  python build_issue_data.py --raw <...>\\triager-omega\\artifacts\\openj9\\raw \\
                             --out <...>\\triagerX\\omega_split\\issue_data
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, help="Dir con los {n}.json crudos")
    p.add_argument("--out", required=True, help="Dir destino issue_data/")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, empty = 0, 0
    for f in glob.glob(os.path.join(args.raw, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        # TriagerX espera assignees = [{"login": ...}]; nuestro raw los guarda como [str].
        raw_assignees = d.get("assignees", []) or []
        assignees = [
            a if isinstance(a, dict) else {"login": a} for a in raw_assignees
        ]
        rec = {
            "assignees": assignees,
            "timeline_data": d.get("timeline", []) or [],
        }
        if not rec["timeline_data"] and not rec["assignees"]:
            empty += 1
        with open(os.path.join(args.out, os.path.basename(f)), "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        n += 1
    print(f"convertidos={n} | sin_assignees_ni_timeline={empty}")


if __name__ == "__main__":
    main()
