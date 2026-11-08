import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import unittest
from argparse import Namespace

SCRIPT = Path(__file__).parents[1] / "scripts/benchmarks/openj9/fuse_openj9_ibr_original.py"
spec = importlib.util.spec_from_file_location("fuse_openj9_ibr_original", SCRIPT)
fuse = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fuse
spec.loader.exec_module(fuse)


class NISAlignmentTests(unittest.TestCase):
    def test_issue_order_and_login_not_column_position(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            payload = {
        "schema": "triagerx-original-ibr-nis-v1",
        "issue_ids": [8, 9],
        "developer_logins": ["bob", "ALICE"],
        "scores_nis": [[1, 0], [0, 1]],
    }
            path = tmp_path / "scores.json"
            path.write_text(json.dumps(payload))
            frame = pd.DataFrame({"issue_number": [8, 9]})
            aligned = fuse._load_nis(path, frame, {"alice": 0, "bob": 1})
            np.testing.assert_array_equal(aligned, [[0, 1], [1, 0]])
            with self.assertRaisesRegex(ValueError, "issue IDs/order"):
                fuse._load_nis(path, frame.iloc[::-1], {"alice": 0, "bob": 1})

    def test_rejects_case_collisions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            frame = pd.DataFrame({"issue_number": [8]})
            path.write_text(json.dumps({"schema": "triagerx-original-ibr-nis-v1", "issue_ids": [8],
                                        "developer_logins": ["dev", "DEV"], "scores_nis": [[0, 1]]}))
            with self.assertRaisesRegex(ValueError, "collide"):
                fuse._load_nis(path, frame, {"dev": 0})


class FusionProtocolTests(unittest.TestCase):
    def test_run_selects_on_validation_and_evaluates_one_test_weight(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / f"{name}.csv" for name in
                     ("traininner", "val", "train", "test")}
            for path in paths.values():
                path.write_text("issue_number,owner,text\\n1,dev,query\\n")
            exports = {"val": root / "val-ibr.json", "test": root / "test-ibr.json"}
            for split, train_key, query_key in (("val", "traininner", "val"), ("test", "train", "test")):
                exports[split].write_text(json.dumps({
                    "temporal_policy": "strict",
                    "data": {"train_csv": str(paths[train_key].resolve()),
                             "test_csv": str(paths[query_key].resolve())},
                }))
            args = Namespace(
                traininner_csv=paths["traininner"], val_csv=paths["val"],
                train_csv=paths["train"], test_csv=paths["test"],
                val_ibr=exports["val"], test_ibr=exports["test"],
                cbr_model="local-mock-model", cpu=True, weights="0.1,0.2",
                allow_original_temporal=False, output=root / "fusion.json",
            )
            fake_result = {"curves": [
                {"weight": .1, "metrics": {"mrr": .2, "hit@1": .1}},
                {"weight": .2, "metrics": {"mrr": .8, "hit@1": .4}},
            ], "cbr_only": {"hit@1": .1}, "ibr_only": {"hit@1": .2}}
            with patch.object(fuse, "_evaluate_split", side_effect=[fake_result, {
                    "curves": [{"weight": .2, "metrics": {"mrr": .5, "hit@1": .3}}],
                    "cbr_only": {"hit@1": .2}, "ibr_only": {"hit@1": .3}}]) as evaluate:
                result = fuse.run(args)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args_list[0].args[4], [.1, .2])
            self.assertEqual(evaluate.call_args_list[1].args[4], [.2])
            self.assertEqual(result["validation"]["selected_weight"], .2)
            self.assertEqual(result["test"]["selected_weight"], .2)
            self.assertTrue(args.output.is_file())

    def test_run_rejects_mixed_temporal_policies(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / f"{name}.csv" for name in
                     ("traininner", "val", "train", "test")}
            for path in paths.values():
                path.write_text("issue_number,owner,text\\n1,dev,query\\n")
            exports = {"val": root / "val-ibr.json", "test": root / "test-ibr.json"}
            for split, train_key, query_key, policy in (
                ("val", "traininner", "val", "strict"), ("test", "train", "test", "original")):
                exports[split].write_text(json.dumps({
                    "temporal_policy": policy,
                    "data": {"train_csv": str(paths[train_key].resolve()),
                             "test_csv": str(paths[query_key].resolve())},
                }))
            args = Namespace(
                traininner_csv=paths["traininner"], val_csv=paths["val"],
                train_csv=paths["train"], test_csv=paths["test"],
                val_ibr=exports["val"], test_ibr=exports["test"],
                cbr_model="local-mock-model", cpu=True, weights="0.1,0.2",
                allow_original_temporal=True, output=root / "fusion.json",
            )
            with patch.object(fuse, "_evaluate_split") as evaluate:
                with self.assertRaisesRegex(ValueError, "same temporal policy"):
                    fuse.run(args)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
