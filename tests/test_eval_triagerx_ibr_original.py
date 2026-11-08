import importlib.util
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from torch import nn

SCRIPT = Path(__file__).parents[1] / "scripts/benchmarks/openj9/eval_triagerx_ibr_original.py"
spec = importlib.util.spec_from_file_location("eval_triagerx_ibr_original", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class AdaptRawIssuesTests(unittest.TestCase):
    def test_preserves_full_event_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, adapted_dir = root / "raw", root / "adapted"
            raw.mkdir()
            event = {"event": "commented", "actor": {"login": "dev"}, "extra": {"nested": [1, 2]}}
            original = {"assignees": [{"login": "dev"}], "timeline": [event], "other": "untouched"}
            source = raw / "12.json"
            source.write_text(json.dumps(original), encoding="utf-8")

            self.assertEqual(module.adapt_raw_issues(raw, adapted_dir, {12}), (1, 1))
            adapted = json.loads((adapted_dir / "12.json").read_text(encoding="utf-8"))
            self.assertEqual(adapted["timeline_data"], [event])
            self.assertNotIn("timeline", adapted)
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), original)

    def test_converts_assignee_logins_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            original = {"assignees": ["alice", {"login": "bob", "id": 42}], "timeline": []}
            source = raw / "1.json"
            source.write_text(json.dumps(original), encoding="utf-8")
            module.adapt_raw_issues(raw, root / "adapted", {1})
            converted = json.loads((root / "adapted/1.json").read_text(encoding="utf-8"))
            self.assertEqual(converted["assignees"], [{"login": "alice"}, {"login": "bob", "id": 42}])
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), original)

    def test_rejects_invalid_assignee(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "1.json").write_text(json.dumps({"assignees": [None], "timeline": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid assignee"):
                module.adapt_raw_issues(raw, root / "adapted", {1})

    def test_rejects_incompatible_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "3.json").write_text(json.dumps({"assignees": [], "timeline": "bad"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timeline list"):
                module.adapt_raw_issues(raw, root / "out", {3})


class EvaluationTests(unittest.TestCase):
    def _fixtures(self, root):
        train = pd.DataFrame({"issue_number": [1, 2], "owner": ["dev-a", "dev-b"], "text": ["one", "two"]})
        test = pd.DataFrame({"issue_number": [3], "owner": ["dev-b"], "text": ["query"]})
        train_path, test_path = root / "train.csv", root / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        meta_path = root / "meta.parquet"
        pd.DataFrame({"issue_number": [3], "created_at": ["2021-02-03T10:00:00Z"]}).to_parquet(meta_path)
        raw_dir = root / "raw"
        raw_dir.mkdir()
        for issue_id in (1, 2):
            (raw_dir / f"{issue_id}.json").write_text(
                json.dumps({"assignees": [], "timeline": []}), encoding="utf-8"
            )
        return train_path, test_path, meta_path, raw_dir

    def test_run_calls_original_ibr_methods_and_reports_hit_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path, test_path, meta_path, raw_dir = self._fixtures(Path(directory))
            instances = []

            class FakeIBR:
                def __init__(self, **kwargs):
                    self.assert_train = kwargs["train_data"].issue_number.tolist()
                    self.initial_checkpoint = kwargs["train_checkpoint_date"]
                    self.checkpoints = []

                def _get_top_k_similar_issues(self, issue, k, threshold):
                    self.checkpoints.append(self._train_checkpoint_date)
                    return [(1, 0.9), (0, 0.8)]

                def _get_historical_contributors(self, similar):
                    self.assert_similar = similar
                    return [("dev-b", 1.0), ("dev-a", 0.5)]

            def factory(**kwargs):
                instance = FakeIBR(**kwargs)
                instances.append(instance)
                return instance

            class FakeSentenceTransformer:
                def __init__(self, path, *, local_files_only):
                    self.assert_local_only = (path, local_files_only)

                def encode(self, texts, **kwargs):
                    self.assert_texts = list(texts)
                    return [[1.0, 0.0], [0.0, 1.0]]

            args = Namespace(
                train_csv=train_path, test_csv=test_path, meta=meta_path, raw_dir=raw_dir,
                triagerx_source="external.py", model="offline-model", top_k=15, threshold=0.6,
                decay=0.01, contribution=1.5, assignment=0.5, discussion=0.1, output=None,
            )
            with patch.object(module, "_load_triagerx", return_value=factory), patch(
                "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
            ), redirect_stdout(io.StringIO()):
                result = module.run(args)

            self.assertEqual(instances[0].assert_train, [1, 2])
            self.assertEqual(instances[0].initial_checkpoint, datetime(1970, 1, 1))
            self.assertEqual(instances[0].checkpoints, [datetime(2021, 2, 3)])
            self.assertEqual(instances[0].assert_similar, [(1, 0.9), (0, 0.8)])
            self.assertEqual(result["metrics"]["hit@1"], 1.0)
            self.assertEqual(result["metrics"]["hit@3"], 1.0)
            self.assertEqual(result["metrics"]["coverage_raw_train"], 1.0)
            self.assertIn("future-event leakage", result["temporal_note"])

    def test_invalid_query_date_fails_before_loading_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path, test_path, meta_path, raw_dir = self._fixtures(Path(directory))
            pd.DataFrame({"issue_number": [3], "created_at": [None]}).to_parquet(meta_path)
            args = Namespace(train_csv=train_path, test_csv=test_path, meta=meta_path, raw_dir=raw_dir,
                             triagerx_source="external.py", model="offline-model", top_k=15, threshold=0.6,
                             decay=0.01, contribution=1.5, assignment=0.5, discussion=0.1, output=None)
            with patch.object(module, "_load_triagerx") as load_triagerx, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "invalid created_at"):
                    module.run(args)
            load_triagerx.assert_not_called()


class OriginalTriagerXIntegrationTests(unittest.TestCase):
    def test_original_similarity_and_contributor_methods_with_local_fixture(self):
        source = module.DEFAULT_TRIAGERX / "triagerx/system/triagerx.py"
        if not source.is_file():
            self.skipTest(f"external TriagerX source unavailable: {source}")
        triager_class = module._load_triagerx(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            issue = {
                "assignees": [{"login": "dev-a"}],
                "timeline_data": [{
                    "event": "referenced", "actor": {"login": "dev-a"},
                    "created_at": "2021-02-02T00:00:00Z", "commit_url": "https://example/commit/1",
                }],
            }
            (root / "1.json").write_text(json.dumps(issue), encoding="utf-8")
            embeddings_path = root / "train.npy"
            np.save(embeddings_path, np.asarray([[1.0, 0.0]], dtype=np.float32))

            class FakeEncoder:
                def encode(self, text):
                    return np.asarray([[1.0, 0.0]], dtype=np.float32)

            evaluator = triager_class(
                component_prediction_model=nn.Identity(), developer_prediction_model=nn.Identity(),
                similarity_model=FakeEncoder(), train_data=pd.DataFrame({"issue_number": [1]}),
                train_embeddings=str(embeddings_path), issues_path=str(root),
                developer_id_map={"dev-a": 0}, component_id_map={"unknown": 0},
                expected_developers={"dev-a"}, device="cpu", similarity_prediction_weight=0.0,
                time_decay_factor=0.01, direct_assignment_score=0.5, contribution_score=1.5,
                discussion_score=0.1, train_checkpoint_date=datetime(2021, 2, 3),
            )
            similar = evaluator._get_top_k_similar_issues("query", 1, 0.6)
            contributors = evaluator._get_historical_contributors(similar)
            self.assertEqual([index for index, _score in similar], [0])
            self.assertEqual(contributors[0][0], "dev-a")
            self.assertGreater(contributors[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
