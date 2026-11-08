from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module

fuse = load_script("adapted_fuse", ROOT / "scripts/benchmarks/openj9/fuse_openj9_ibr_original.py")
views = load_script("openj9_views", ROOT / "scripts/benchmarks/openj9/build_openj9_view_splits.py")


class ViewAlignmentTests(unittest.TestCase):
    def test_reorders_by_ids_and_reproduces_same_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.csv"
            view = root / "view.csv"
            pd.DataFrame({"issue_number": [2, 1], "owner": ["b", "a"], "text": ["raw2", "raw1"]}).to_csv(base, index=False)
            pd.DataFrame({"issue_number": [1, 2, 3], "owner": ["a", "b", "c"], "text": ["dist1", "dist2", "extra"]}).to_csv(view, index=False)
            expected = views.align_view(base, view)
            pd.testing.assert_frame_equal(expected, views.align_view(base, view))
            self.assertEqual(expected.issue_number.tolist(), [2, 1])
            self.assertEqual(expected.text.tolist(), ["dist2", "dist1"])

    def test_rejects_missing_ids_duplicate_ids_and_owner_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, view = root / "base.csv", root / "view.csv"
            pd.DataFrame({"issue_number": [2, 1], "owner": ["b", "a"], "text": ["x", "y"]}).to_csv(base, index=False)
            for ids, owners, message in (([1], ["a"], "cover"), ([1, 1], ["a", "a"], "duplicate"),
                                         ([1, 2], ["a", "wrong"], "owner mismatch")):
                pd.DataFrame({"issue_number": ids, "owner": owners, "text": ["x"] * len(ids)}).to_csv(view, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    views.align_view(base, view)

    def test_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("base.csv", "view.csv"):
                pd.DataFrame({"issue_number": [1], "owner": ["a"], "text": [name]}).to_csv(root / name, index=False)
            out = root / "out"
            out.mkdir()
            (out / "traininner_distilled.csv").write_text("existing")
            pairs = [("traininner", root / "base.csv", root / "view.csv")]
            with self.assertRaises(FileExistsError):
                views.build_splits(pairs, out, "distilled")


class CbrViewFusionTests(unittest.TestCase):
    def test_weight_selection_uses_top1_then_mrr_then_smallest_weight(self):
        curves = [
            {"weight": 0.0, "metrics": {"hit@1": .4, "mrr": .5}},
            {"weight": .2, "metrics": {"hit@1": .5, "mrr": .4}},
            {"weight": .4, "metrics": {"hit@1": .5, "mrr": .7}},
            {"weight": .6, "metrics": {"hit@1": .5, "mrr": .7}},
        ]
        self.assertEqual(fuse._select_weight(curves), .4)

    def test_requires_exact_ibr_view_issue_order_and_owner(self):
        source = pd.DataFrame({"issue_number": [1, 2], "owner": ["a", "b"], "text": ["x", "y"]})
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            fuse._validate_cbr_view(source, source.iloc[::-1], "validation")
        wrong_owner = source.copy()
        wrong_owner.loc[1, "owner"] = "c"
        with self.assertRaisesRegex(ValueError, "owner labels/order"):
            fuse._validate_cbr_view(source, wrong_owner, "validation")

    def test_hybrid_uses_real_retriever_dense_lexical_and_rrf(self):
        class MockRetriever:
            def __init__(self, config):
                self.config = config
                self.label_encoder = {"dev": 0}
            def fit(self, *args, **kwargs):
                return self
            def compute_dense_sims(self, texts):
                return np.array([[0.8, 0.1]])
            def compute_lexical_sims(self, texts):
                return np.array([[0.2, 0.9]])
            def score_sims(self, sims, **kwargs):
                np.testing.assert_allclose(sims, [[1 / 61 + 1 / 62, 1 / 62 + 1 / 61]])
                self.score_kwargs = kwargs
                return np.array([[1., 0.]])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = pd.DataFrame({"issue_number": [1, 2], "owner": ["dev", "other"], "text": ["a", "b"]})
            query = pd.DataFrame({"issue_number": [3], "owner": ["dev"], "text": ["c"]})
            for name, frame in (("train.csv", train), ("query.csv", query)):
                frame.to_csv(root / name, index=False)
            payload = {"schema": "triagerx-original-ibr-nis-v1", "issue_ids": [3],
                       "developer_logins": ["dev", "other"], "scores_nis": [[0.5, 0.0]], "temporal_policy": "strict",
                       "data": {"train_csv": str(root / "train.csv"), "test_csv": str(root / "query.csv")}}
            nis_path = root / "nis.json"
            nis_path.write_text(json.dumps(payload))
            args = type("Args", (), {"cpu": True, "cbr_model": "mock", "cbr_maxlen": 384,
                                      "cbr_topk": 30, "cbr_tau": 2., "rrf_k": 60, "cbr_mode": "hybrid"})()
            with patch("triager_omega.retrieval.CbrRetriever", MockRetriever):
                result = fuse._evaluate_split(root / "train.csv", root / "query.csv", nis_path,
                                              args, [0.], root / "train.csv", root / "query.csv")
            self.assertIn("cbr_only", result)
            self.assertEqual(result["curves"][0]["weight"], 0.)


if __name__ == "__main__":
    unittest.main()
