"""Fuse raw OpenJ9 CBR vote-sum scores with exported original TriagerX IBR NIS.

Val selects one fixed linear-fusion weight; test is evaluated only at that
weight. The caller supplies the exact temporal split's raw IBR export.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_FULL_SPEC = importlib.util.spec_from_file_location("eval_openj9_full", SCRIPT_DIR / "eval_openj9_full.py")
_FULL = importlib.util.module_from_spec(_FULL_SPEC)
sys.modules[_FULL_SPEC.name] = _FULL
assert _FULL_SPEC.loader is not None
_FULL_SPEC.loader.exec_module(_FULL)

DEFAULT_WEIGHTS = [round(value, 1) for value in np.arange(0.1, 1.0001, 0.1)]


def _load_nis(path: Path, queries: pd.DataFrame, label_encoder: dict[str, int]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "triagerx-original-ibr-nis-v1":
        raise ValueError(f"Unsupported NIS export schema: {payload.get('schema')!r}")
    ids = [int(value) for value in payload.get("issue_ids", [])]
    query_ids = [int(value) for value in queries.issue_number]
    if ids != query_ids:
        raise ValueError("NIS issue IDs/order do not exactly match the requested split query CSV")
    developers = payload.get("developer_logins")
    scores = np.asarray(payload.get("scores_nis"), dtype=float)
    if not isinstance(developers, list) or scores.shape != (len(query_ids), len(developers)):
        raise ValueError("NIS export labels and score matrix are not aligned")
    normalized = [str(name).lower() for name in developers]
    normalized_encoder = {str(name).lower(): index for name, index in label_encoder.items()}
    if len(normalized_encoder) != len(label_encoder) or len(set(normalized)) != len(normalized):
        raise ValueError("Developer logins collide after case normalization")
    if any(name not in normalized_encoder for name in normalized):
        raise ValueError("NIS export includes developer(s) absent from CBR label encoder")
    aligned = np.zeros((len(query_ids), len(label_encoder)), dtype=np.float32)
    for source_idx, developer in enumerate(normalized):
        aligned[:, normalized_encoder[developer]] = scores[:, source_idx]
    return aligned


def _validate_export_source(payload: dict, train_path: Path, query_path: Path) -> str:
    data = payload.get("data")
    if not isinstance(data, dict) or not data.get("train_csv") or not data.get("test_csv"):
        raise ValueError("NIS export must record its source train_csv and test_csv")
    if Path(data["train_csv"]).expanduser().resolve() != train_path.expanduser().resolve():
        raise ValueError("NIS export train_csv does not match the requested split training CSV")
    if Path(data["test_csv"]).expanduser().resolve() != query_path.expanduser().resolve():
        raise ValueError("NIS export test_csv does not match the requested split query CSV")
    policy = payload.get("temporal_policy")
    if policy not in {"strict", "original"}:
        raise ValueError(f"Unknown NIS temporal policy: {policy!r}")
    return policy


def _evaluate_split(train_path: Path, query_path: Path, nis_path: Path, args, weights: list[float]):
    train = pd.read_csv(train_path).drop_duplicates("issue_number").reset_index(drop=True)
    query = pd.read_csv(query_path).drop_duplicates("issue_number").reset_index(drop=True)
    if not {"issue_number", "owner", "text"}.issubset(train) or not {"issue_number", "owner", "text"}.issubset(query):
        raise ValueError("split CSVs require issue_number, owner, and text")
    payload = json.loads(nis_path.read_text(encoding="utf-8"))
    policy = _validate_export_source(payload, train_path, query_path)
    if policy == "original" and not args.allow_original_temporal:
        raise ValueError("Refusing temporally leaky original-policy IBR export; explicitly pass --allow-original-temporal")
    # Build exactly the same deterministic candidate IDs as the full evaluator;
    # NIS columns missing test-only labels are explicitly zero-filled by login.
    from triager_omega.retrieval import build_label_encoder
    labels = build_label_encoder(train.owner, query.owner)
    cfg = _FULL.EvaluationConfig(
        cpu=args.cpu, cbr_mode="retrieval", cbr_model=args.cbr_model,
        cbr_topk=50, cbr_tau=2.0, cbr_maxlen=384, cbr_vote="sum",
        fusion="linear", gate=None,
    )
    nis = _load_nis(nis_path, query, labels)
    return _FULL.evaluate(cfg, train_df=train, test_df=query, nis=nis, fusion_weights=weights)


def run(args: argparse.Namespace) -> dict:
    for name in ("traininner_csv", "val_csv", "train_csv", "test_csv", "val_ibr", "test_ibr"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fusion result: {output}")
    if args.cbr_model.startswith(("sentence-transformers/", "all-")):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    weights = [float(value) for value in args.weights.split(",")]
    if not weights or any(not np.isfinite(value) or value < 0 for value in weights):
        raise ValueError("--weights must be comma-separated finite nonnegative numbers")

    val_train, val_query = Path(args.traininner_csv), Path(args.val_csv)
    test_train, test_query = Path(args.train_csv), Path(args.test_csv)
    val_export = json.loads(Path(args.val_ibr).read_text(encoding="utf-8"))
    test_export = json.loads(Path(args.test_ibr).read_text(encoding="utf-8"))
    val_policy = _validate_export_source(val_export, val_train, val_query)
    test_policy = _validate_export_source(test_export, test_train, test_query)
    if val_policy != test_policy:
        raise ValueError("Validation and test IBR exports must use the same temporal policy")
    if val_policy == "original" and not args.allow_original_temporal:
        raise ValueError("Refusing temporally leaky original-policy IBR exports; explicitly pass --allow-original-temporal")

    val = _evaluate_split(val_train, val_query, Path(args.val_ibr), args, weights)
    # Deterministic fixed protocol: maximize validation MRR, then Hit@1, then
    # choose the smaller weight on exact ties. No test metric participates.
    selected = max(val["curves"], key=lambda row: (row["metrics"]["mrr"], row["metrics"]["hit@1"], -row["weight"]))
    selected_weight = float(selected["weight"])
    test = _evaluate_split(test_train, test_query, Path(args.test_ibr), args, [selected_weight])
    result = {
        "protocol": "traininner->val selects fixed W_f by MRR, then Hit@1, then smaller weight; train->test evaluated only at selected weight",
        "cbr": {"vote": "sum", "top_k": 50, "tau": 2.0, "max_length": 384, "model": args.cbr_model},
        "ibr_temporal_policies": {"val": json.loads(Path(args.val_ibr).read_text())["temporal_policy"],
                                  "test": json.loads(Path(args.test_ibr).read_text())["temporal_policy"]},
        "weights_tested_on_validation": weights,
        "validation": {"cbr_only": val["cbr_only"], "ibr_only": val["ibr_only"],
                       "curve": val["curves"], "selected_weight": selected_weight},
        "test": {"cbr_only": test["cbr_only"], "ibr_only": test["ibr_only"],
                 "selected_weight": selected_weight, "metrics": test["curves"][0]["metrics"]},
        "inputs": {key: str(Path(getattr(args, key))) for key in
                   ("traininner_csv", "val_csv", "train_csv", "test_csv", "val_ibr", "test_ibr")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traininner-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--val-ibr", required=True, help="original IBR per-query export for traininner->val")
    parser.add_argument("--test-ibr", required=True, help="original IBR per-query export for train->test")
    parser.add_argument("--cbr-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--weights", default=",".join(map(str, DEFAULT_WEIGHTS)))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-original-temporal", action="store_true",
                        help="explicitly allow original IBR temporal policy (not prospectively valid)")
    parser.add_argument("--output", required=True, help="new JSON path; existing files are never overwritten")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
