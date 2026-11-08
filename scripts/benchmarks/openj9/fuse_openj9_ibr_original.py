"""Fuse an explicitly selected OpenJ9 CBR text view with exported original TriagerX IBR NIS.

IBR source CSVs remain fixed; separate CBR view CSVs must match their issue-ID
order and owners. Validation selects a fixed fusion weight; test sees only that
weight. Dense and CbrRetriever hybrid/RRF modes use sum voting.
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

DEFAULT_WEIGHTS = [round(value, 1) for value in np.arange(0.0, 1.0001, 0.1)]


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


def _read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"issue_number", "owner", "text"}.issubset(frame):
        raise ValueError(f"split CSV requires issue_number, owner, and text: {path}")
    if frame.issue_number.isna().any() or frame.issue_number.duplicated().any():
        raise ValueError(f"split CSV has missing or duplicate issue_number values: {path}")
    if frame.owner.isna().any():
        raise ValueError(f"split CSV has missing owner values: {path}")
    frame["text"] = frame.text.fillna("").astype(str)
    return frame.reset_index(drop=True)


def _validate_cbr_view(ibr_frame: pd.DataFrame, cbr_frame: pd.DataFrame, split: str) -> None:
    if ibr_frame.issue_number.tolist() != cbr_frame.issue_number.tolist():
        raise ValueError(f"{split} CBR view issue IDs/order do not exactly match IBR source CSV")
    if ibr_frame.owner.tolist() != cbr_frame.owner.tolist():
        raise ValueError(f"{split} CBR view owner labels/order do not exactly match IBR source CSV")


def _select_weight(curves: list[dict]) -> float:
    selected = max(curves, key=lambda row: (row["metrics"]["hit@1"], row["metrics"]["mrr"], -row["weight"]))
    return float(selected["weight"])


def _evaluate_split(train_path: Path, query_path: Path, nis_path: Path, args, weights: list[float],
                    cbr_train_path: Path | None = None, cbr_query_path: Path | None = None):
    train = _read_split(train_path)
    query = _read_split(query_path)
    cbr_train = _read_split(cbr_train_path or train_path)
    cbr_query = _read_split(cbr_query_path or query_path)
    _validate_cbr_view(train, cbr_train, "train")
    _validate_cbr_view(query, cbr_query, "query")
    payload = json.loads(nis_path.read_text(encoding="utf-8"))
    policy = _validate_export_source(payload, train_path, query_path)
    if policy == "original" and not args.allow_original_temporal:
        raise ValueError("Refusing temporally leaky original-policy IBR export; explicitly pass --allow-original-temporal")
    # Build exactly the same deterministic candidate IDs as the full evaluator;
    # NIS columns missing test-only labels are explicitly zero-filled by login.
    from triager_omega.retrieval import build_label_encoder
    labels = build_label_encoder(train.owner, query.owner)
    from triager_omega.modules.aggregator import normalize_rows, rank_metrics, make_fuser
    from triager_omega.retrieval import CbrConfig, CbrRetriever, rrf_fuse

    nis = _load_nis(nis_path, query, labels)
    retriever = CbrRetriever(CbrConfig(
        encoder_name=args.cbr_model, device="cpu" if args.cpu else _FULL.Settings().torch_device,
        max_seq_length=args.cbr_maxlen, top_k=args.cbr_topk, tau=args.cbr_tau,
        vote_strategy="sum", rrf_k=args.rrf_k,
    ))
    retriever.fit(cbr_train.text.tolist(), cbr_train.owner.tolist(), query.owner.tolist())
    dense = retriever.compute_dense_sims(cbr_query.text.tolist())
    similarities = dense if args.cbr_mode == "dense" else rrf_fuse(
        dense, retriever.compute_lexical_sims(cbr_query.text.tolist()), k=args.rrf_k)
    nps = normalize_rows(retriever.score_sims(similarities, top_k=args.cbr_topk,
                                             tau=args.cbr_tau, vote_strategy="sum"), how="minmax")
    true_idx = query.owner.map(labels).to_numpy().astype(int)
    fuser = make_fuser(nps, nis, "linear", 60, None)
    curves = [{"gate": None, "weight": weight, "metrics": rank_metrics(fuser(weight), true_idx)}
              for weight in weights]
    return {"curves": curves, "cbr_only": rank_metrics(nps, true_idx),
            "ibr_only": rank_metrics(nis, true_idx)}


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
    cbr_paths = {}
    for name in ("cbr_traininner_csv", "cbr_val_csv", "cbr_train_csv", "cbr_test_csv"):
        value = getattr(args, name, None) or getattr(args, name.removeprefix("cbr_"))
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        cbr_paths[name] = path

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

    val = _evaluate_split(val_train, val_query, Path(args.val_ibr), args, weights,
                          cbr_paths["cbr_traininner_csv"], cbr_paths["cbr_val_csv"])
    # Select only on validation: maximize Hit@1, then MRR, then prefer lower weight.
    selected_weight = _select_weight(val["curves"])
    test = _evaluate_split(test_train, test_query, Path(args.test_ibr), args, [selected_weight],
                           cbr_paths["cbr_train_csv"], cbr_paths["cbr_test_csv"])
    result = {
        "protocol": "traininner->val selects fixed W_f by Hit@1, then MRR, then smaller weight; train->test evaluated only at selected weight",
        "cbr": {"mode": args.cbr_mode, "vote": "sum", "top_k": args.cbr_topk,
                "tau": args.cbr_tau, "max_length": args.cbr_maxlen,
                "rrf_k": args.rrf_k if args.cbr_mode == "hybrid" else None,
                "model": args.cbr_model},
        "ibr_temporal_policies": {"val": json.loads(Path(args.val_ibr).read_text())["temporal_policy"],
                                  "test": json.loads(Path(args.test_ibr).read_text())["temporal_policy"]},
        "weights_tested_on_validation": weights,
        "validation": {"cbr_only": val["cbr_only"], "ibr_only": val["ibr_only"],
                       "curve": val["curves"], "selected_weight": selected_weight},
        "test": {"cbr_only": test["cbr_only"], "ibr_only": test["ibr_only"],
                 "selected_weight": selected_weight, "metrics": test["curves"][0]["metrics"]},
        "inputs": {key: str(Path(getattr(args, key))) for key in
                   ("traininner_csv", "val_csv", "train_csv", "test_csv", "val_ibr", "test_ibr")},
        "cbr_view_inputs": {key: str(value) for key, value in cbr_paths.items()},
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
    parser.add_argument("--cbr-mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--cbr-topk", type=int, default=50)
    parser.add_argument("--cbr-tau", type=float, default=2.0)
    parser.add_argument("--cbr-maxlen", type=int, default=384)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--cbr-traininner-csv", help="CBR view CSV; defaults to IBR source CSV")
    parser.add_argument("--cbr-val-csv", help="CBR view CSV; defaults to IBR source CSV")
    parser.add_argument("--cbr-train-csv", help="CBR view CSV; defaults to IBR source CSV")
    parser.add_argument("--cbr-test-csv", help="CBR view CSV; defaults to IBR source CSV")
    parser.add_argument("--weights", default=",".join(map(str, DEFAULT_WEIGHTS)))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-original-temporal", action="store_true",
                        help="explicitly allow original IBR temporal policy (not prospectively valid)")
    parser.add_argument("--output", required=True, help="new JSON path; existing files are never overwritten")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
