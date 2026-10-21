import importlib.util
import sys
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/benchmarks/openj9/eval_openj9_ibr.py"
spec = importlib.util.spec_from_file_location("eval_openj9_ibr_integration", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_run_uses_real_ibr_scoring_and_encodes_train_and_test_once(tmp_path, monkeypatch):
    train = pd.DataFrame({"issue_number": [1, 2], "owner": ["z", "a"], "text": ["train one", "train two"]})
    test = pd.DataFrame({"issue_number": [3], "owner": ["z"], "text": ["query"]})
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False); test.to_csv(test_path, index=False)
    interactions_path, meta_path = tmp_path / "interactions.parquet", tmp_path / "meta.parquet"
    pd.DataFrame({"issue_number": [1, 2], "dev": ["z", "a"], "kind": ["commit", "assignment"],
                  "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"])}).to_parquet(interactions_path)
    pd.DataFrame({"issue_number": [3], "created_at": pd.to_datetime(["2021-01-01"])}).to_parquet(meta_path)
    encoded = []
    vectors = {"train one": [3., 0.], "train two": [0., 2.], "query": [1., 1.]}

    class Encoder:
        def encode(self, texts, **kwargs):
            encoded.append(list(texts))
            return np.asarray([vectors[t] for t in texts])
    monkeypatch.setattr(module.InteractionRecommender, "_sbert", lambda self: Encoder())
    defaults = dict(top_k=15, tau=.6, lam=.01, ip_c=1.5, ip_a=.5, ip_d=.1,
                    train_csv=str(train_path), test_csv=str(test_path),
                    interactions=str(interactions_path), meta=str(meta_path))
    result = module.run(Namespace(**defaults))
    assert encoded == [["train one", "train two"], ["query"]]

    # Independent util.cos_sim + top-k/IP/decay/div-max reference.
    from sentence_transformers import util
    sims = util.cos_sim(torch.tensor([[1., 1.]]), torch.tensor([[3., 0.], [0., 2.]]))[0].numpy()
    assert sims == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])
    raw = {"z": sims[0] * 1.5 * np.exp(-.01 * 366),
           "a": sims[1] * .5 * np.exp(-.01 * 366)}
    maximum = max(raw.values())
    expected = np.array([[raw["a"] / maximum, raw["z"] / maximum]], dtype=np.float32)  # sorted owners: a,z
    # The public run returns rank metrics rather than internal NIS; derive the rank independently.
    rank = int(np.flatnonzero(np.argsort(-expected[0], kind="stable") == 1)[0])
    expected_metrics = {f"hit@{k}": float(rank < k) for k in (1, 3, 5, 10)}
    expected_metrics.update(mrr=1. / (rank + 1), coverage=1.)
    assert result == expected_metrics
