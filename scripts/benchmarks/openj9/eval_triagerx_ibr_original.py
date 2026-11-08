"""Evaluate the original TriagerX IBR on the local OpenJ9 50-class split.

This imports TriagerX from an external checkout; it does not copy or reimplement
its scoring. Raw issue JSON is adapted into an isolated temporary directory by
renaming ``timeline`` to ``timeline_data`` and wrapping assignee logins in the
objects expected by TriagerX.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULT_PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_TRIAGERX = Path("/Users/sebastian/Documents/Python/triagerX")


def adapt_raw_issues(
    raw_dir: Path, output_dir: Path, issue_ids: set[int], *, cutoff=None, stats: dict | None = None
) -> tuple[int, int]:
    """Adapt raw issues; optionally retain only verifiably pre-query timeline events."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for issue_id in sorted(issue_ids):
        source = raw_dir / f"{issue_id}.json"
        if not source.is_file():
            raise ValueError(f"Missing raw issue {issue_id}: {source}")
        try:
            issue = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid raw JSON for issue {issue_id}: {source}: {exc}") from exc
        if not isinstance(issue, dict) or not isinstance(issue.get("assignees"), list):
            raise ValueError(f"Issue {issue_id} must be an object with an assignees list")
        if "timeline_data" in issue:
            raise ValueError(f"Issue {issue_id} already has timeline_data; refusing ambiguous adaptation")
        timeline = issue.get("timeline")
        if not isinstance(timeline, list) or any(not isinstance(event, dict) for event in timeline):
            raise ValueError(f"Issue {issue_id} must have a timeline list of event objects")
        assignees = []
        for assignee in issue["assignees"] if cutoff is None else []:
            if isinstance(assignee, str) and assignee.strip():
                assignees.append({"login": assignee})
            elif isinstance(assignee, dict) and isinstance(assignee.get("login"), str) and assignee["login"].strip():
                assignees.append(assignee)
            else:
                raise ValueError(f"Issue {issue_id} has an invalid assignee: {assignee!r}")
        if cutoff is not None:
            retained = []
            for event in timeline:
                parsed = pd.to_datetime(event.get("created_at"), utc=True, errors="coerce")
                if pd.isna(parsed):
                    if stats is not None:
                        stats["excluded_no_valid_date"] = stats.get("excluded_no_valid_date", 0) + 1
                elif parsed < cutoff:
                    retained.append(event)
                elif stats is not None:
                    stats["excluded_future_or_equal"] = stats.get("excluded_future_or_equal", 0) + 1
            timeline = retained
        adapted = dict(issue)
        adapted["assignees"] = assignees
        del adapted["timeline"]
        adapted["timeline_data"] = timeline
        (output_dir / source.name).write_text(json.dumps(adapted), encoding="utf-8")
        copied += 1
    return copied, len(issue_ids)


def _load_triagerx(source: Path):
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TriagerX source not found: {source}; pass --triagerx-source")
    package_root = next((p for p in source.parents if p.name == "triagerx"), None)
    if package_root is None or not (package_root / "__init__.py").exists():
        raise ValueError(f"Expected source inside a TriagerX package: {source}")
    repo_root = package_root.parent
    sys.path.insert(0, str(repo_root))
    try:
        from triagerx.system.triagerx import TriagerX
    except Exception:
        sys.path.remove(str(repo_root))
        raise
    return TriagerX


def _eligible_train_positions(train_timestamps: pd.Series, query_timestamp) -> np.ndarray:
    """Return only train rows with a verified timestamp strictly before the query."""
    return np.flatnonzero(np.asarray(train_timestamps < query_timestamp))


def _date(value) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Missing/invalid created_at for query: {value!r}")
    return datetime.strptime(parsed.strftime("%Y-%m-%d"), "%Y-%m-%d")


def run(args: argparse.Namespace) -> dict:
    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    meta_path = Path(args.meta)
    raw_dir = Path(args.raw_dir)
    for path in (train_path, test_path, meta_path, raw_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    train = pd.read_csv(train_path).drop_duplicates("issue_number").reset_index(drop=True)
    test = pd.read_csv(test_path).drop_duplicates("issue_number").reset_index(drop=True)
    for name, frame in (("train", train), ("test", test)):
        missing = {"issue_number", "owner", "text"} - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f"{name} CSV needs nonempty issue_number/owner/text columns; missing={sorted(missing)}")
    meta_frame = pd.read_parquet(meta_path)
    if not {"issue_number", "created_at"}.issubset(meta_frame.columns):
        raise ValueError("meta parquet needs issue_number and created_at columns")
    meta = meta_frame.drop_duplicates("issue_number").set_index("issue_number")["created_at"]

    train_ids = set(map(int, train.issue_number))
    available_raw = {int(path.stem) for path in raw_dir.glob("*.json") if path.stem.isdigit()}
    missing_raw = sorted(train_ids - available_raw)
    print(f"Raw coverage: {len(train_ids) - len(missing_raw)}/{len(train_ids)} train issues present; required 100%")
    if missing_raw:
        raise ValueError(f"Raw issue coverage incomplete ({len(missing_raw)} missing); examples: {missing_raw[:10]}")
    query_ids = set(map(int, test.issue_number))
    meta_by_query = {int(issue_id): value for issue_id, value in meta.items()}
    missing_meta = sorted(query_ids - set(meta_by_query))
    invalid_dates = []
    checkpoint_dates = {}
    query_timestamps = {}
    for issue_id in sorted(query_ids - set(missing_meta)):
        try:
            query_timestamps[issue_id] = pd.to_datetime(meta_by_query[issue_id], utc=True, errors="raise")
            checkpoint_dates[issue_id] = _date(meta_by_query[issue_id])
        except ValueError:
            invalid_dates.append(issue_id)
    valid_dates = len(query_ids) - len(missing_meta) - len(invalid_dates)
    print(f"Valid query-date coverage: {valid_dates}/{len(query_ids)} test issues; required 100%")
    if missing_meta:
        raise ValueError(f"Query metadata coverage incomplete ({len(missing_meta)} missing); examples: {missing_meta[:10]}")
    if invalid_dates:
        raise ValueError(f"Query metadata has missing/invalid created_at dates ({len(invalid_dates)}); examples: {invalid_dates[:10]}")

    policy = getattr(args, "temporal_policy", "strict")
    if policy not in {"strict", "original"}:
        raise ValueError(f"Unknown temporal policy: {policy}")
    triager_class = _load_triagerx(Path(args.triagerx_source))
    with tempfile.TemporaryDirectory(prefix="triagerx-ibr-adapted-") as temporary:
        issues_dir = Path(temporary)
        copied, requested = adapt_raw_issues(raw_dir, issues_dir, train_ids)
        print(f"Adapted raw coverage: {copied}/{requested} train issues (100%)")

        from sentence_transformers import SentenceTransformer
        model_path = args.model
        try:
            similarity_model = SentenceTransformer(model_path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot load cached model {model_path!r} offline; provide --model with a local cache") from exc
        train_embeddings = np.asarray(similarity_model.encode(train.text.fillna("").astype(str).tolist(), show_progress_bar=False))
        embedding_path = issues_dir / "train_embeddings.npy"
        np.save(embedding_path, train_embeddings)

        expected = set(train.owner.dropna().astype(str).str.lower())
        dev_map = {dev: index for index, dev in enumerate(sorted(expected))}
        component_map = {"unknown": 0}
        dummy = nn.Identity()
        evaluator = triager_class(
            component_prediction_model=dummy,
            developer_prediction_model=nn.Identity(),
            similarity_model=similarity_model,
            train_data=train,
            train_embeddings=str(embedding_path),
            issues_path=str(issues_dir),
            developer_id_map=dev_map,
            component_id_map=component_map,
            expected_developers=expected,
            device="cpu",
            similarity_prediction_weight=0.0,
            time_decay_factor=args.decay,
            direct_assignment_score=args.assignment,
            contribution_score=args.contribution,
            discussion_score=args.discussion,
            train_checkpoint_date=datetime(1970, 1, 1),
        )
        ranks = []
        temporal_stats = {"excluded_no_valid_date": 0, "excluded_future_or_equal": 0}
        excluded_current_assignees = 0
        train_timestamps = pd.to_datetime(train.issue_number.map(meta), utc=True, errors="coerce")
        unknown_train_dates = int(train_timestamps.isna().sum())
        temporal_stats["train_issues_without_valid_date"] = unknown_train_dates
        original_train_data = evaluator._train_data
        original_embeddings = evaluator._all_embeddings
        for row in test.itertuples(index=False):
            issue_id = int(row.issue_number)
            if issue_id not in meta.index:
                raise ValueError(f"No meta created_at for query issue {issue_id}")
            evaluator._train_checkpoint_date = checkpoint_dates[issue_id]
            if policy == "strict":
                eligible_positions = _eligible_train_positions(train_timestamps, query_timestamps[issue_id])
                evaluator._train_data = original_train_data.iloc[eligible_positions].reset_index(drop=True)
                evaluator._all_embeddings = original_embeddings[eligible_positions]
                similar = evaluator._get_top_k_similar_issues(
                    str(row.text), min(args.top_k, len(eligible_positions)), args.threshold
                ) if len(eligible_positions) else []
                query_dir = issues_dir / f"query-{issue_id}"
                neighbor_ids = {int(evaluator._train_data.iloc[int(index)].issue_number) for index, _score in similar}
                adapt_raw_issues(raw_dir, query_dir, neighbor_ids, cutoff=query_timestamps[issue_id], stats=temporal_stats)
                excluded_current_assignees += len(neighbor_ids)
                evaluator._issues_path = str(query_dir)
                evaluator._all_issues = [f"{neighbor_id}.json" for neighbor_id in neighbor_ids]
            else:
                similar = evaluator._get_top_k_similar_issues(str(row.text), args.top_k, args.threshold)
            contributors = evaluator._get_historical_contributors(similar)
            ordered = [dev for dev, _score in contributors]
            owner = str(row.owner).lower()
            ranks.append(next((rank for rank, dev in enumerate(ordered) if dev == owner), len(ordered)))

    rank_array = np.asarray(ranks)
    metrics = {f"hit@{k}": float(np.mean(rank_array < k)) for k in (1, 3, 5, 10)}
    metrics["queries"] = len(ranks)
    metrics["train_issues"] = len(train)
    metrics["coverage_raw_train"] = copied / requested if requested else 0.0
    result = {
        "metrics": metrics,
        "parameters": {"top_k": args.top_k, "threshold": args.threshold, "decay": args.decay,
                       "contribution": args.contribution, "assignment": args.assignment,
                       "discussion": args.discussion, "model": args.model},
        "data": {"train_csv": str(train_path), "test_csv": str(test_path), "meta": str(meta_path),
                 "raw_dir": str(raw_dir), "triagerx_source": str(Path(args.triagerx_source).resolve())},
        "temporal_policy": policy,
        "temporal_exclusions": {**temporal_stats, "current_assignee_lists_excluded_for_neighbor_issues": excluded_current_assignees},
        "temporal_note": (
            "TriagerX scoring retained. Query checkpoints remain truncated to day as in the original grid. "
            "Strict mode filters neighbor timeline events to valid UTC created_at strictly before the full query timestamp; "
            "all current assignee lists are excluded because historical assignment times are unavailable. TriagerX infers "
            "last_assignment heuristically from retained PR/commit events; this is not observed assignment history."
            if policy == "strict" else
            "Original behavior retained: checkpoints are created_at truncated to day and full timelines/current assignees "
            "are used, so future-event leakage is possible."
        ),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite result: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triagerx-source", default=str(DEFAULT_TRIAGERX / "triagerx/system/triagerx.py"))
    parser.add_argument("--train-csv", default=str(DEFAULT_PROJECT / "artifacts/openj9/openj9_train_50.csv"))
    parser.add_argument("--test-csv", default=str(DEFAULT_PROJECT / "artifacts/openj9/openj9_test_50.csv"))
    parser.add_argument("--meta", default=str(DEFAULT_PROJECT / "artifacts/openj9/openj9_issue_meta_50.parquet"))
    parser.add_argument("--raw-dir", default=str(DEFAULT_PROJECT / "artifacts/openj9/raw"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="local cached model path/name; loading is offline-only")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--contribution", type=float, default=1.5)
    parser.add_argument("--assignment", type=float, default=0.5)
    parser.add_argument("--discussion", type=float, default=0.1)
    parser.add_argument("--temporal-policy", choices=("strict", "original"), default="strict",
                        help="strict excludes unverifiable/future neighbor evidence; original reproduces prior behavior")
    parser.add_argument("--output", help="optional new result JSON path; existing paths are never overwritten")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
