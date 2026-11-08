"""Align alternate OpenJ9 text views to the original temporal split CSVs by issue ID."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {"issue_number", "owner", "text"}


def align_view(base_path: Path, view_path: Path) -> pd.DataFrame:
    base = pd.read_csv(base_path)
    view = pd.read_csv(view_path)
    for label, frame in (("base", base), ("view", view)):
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} CSV missing required columns: {sorted(missing)}")
        if frame.issue_number.isna().any() or frame.issue_number.duplicated().any():
            raise ValueError(f"{label} CSV has missing or duplicate issue_number values")
        if frame.owner.isna().any():
            raise ValueError(f"{label} CSV has missing owner values")
    base_ids = base.issue_number.tolist()
    view_ids = view.issue_number.tolist()
    missing = sorted(set(base_ids) - set(view_ids))
    if missing:
        raise ValueError(f"view does not cover all base issue IDs; missing={missing[:10]}")
    owner_by_id = view.set_index("issue_number").owner
    mismatches = [issue_id for issue_id, owner in zip(base_ids, base.owner)
                  if owner_by_id.loc[issue_id] != owner]
    if mismatches:
        raise ValueError(f"view owner mismatch for issue IDs: {mismatches[:10]}")
    return view.set_index("issue_number").loc[base_ids].reset_index()


def build_splits(pairs: list[tuple[str, Path, Path]], output_dir: Path, name: str) -> list[Path]:
    outputs = []
    for split, base_path, view_path in pairs:
        output = output_dir / f"{split}_{name}.csv"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite view CSV: {output}")
        outputs.append((output, align_view(base_path, view_path)))
    for path, frame in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return [path for path, _ in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traininner-base-csv", required=True)
    parser.add_argument("--val-base-csv", required=True)
    parser.add_argument("--traininner-view-csv", required=True)
    parser.add_argument("--val-view-csv", required=True)
    parser.add_argument("--train-base-csv", required=True)
    parser.add_argument("--test-base-csv", required=True)
    parser.add_argument("--train-view-csv", required=True)
    parser.add_argument("--test-view-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", choices=["distilled", "both"], required=True)
    args = parser.parse_args()
    build_splits([
        ("traininner", Path(args.traininner_base_csv), Path(args.traininner_view_csv)),
        ("val", Path(args.val_base_csv), Path(args.val_view_csv)),
        ("train", Path(args.train_base_csv), Path(args.train_view_csv)),
        ("test", Path(args.test_base_csv), Path(args.test_view_csv)),
    ], Path(args.output_dir), args.name)


if __name__ == "__main__":
    main()
