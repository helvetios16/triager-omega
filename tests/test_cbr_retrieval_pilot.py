import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks/mozilla"))
import cbr_retrieval_pilot as benchmark

from triager_omega.modules.aggregator import normalize_rows, rank_metrics
from triager_omega.retrieval import vote_scores


LABELS = {"zeta": 0, "alpha": 1, "ghost": 2}  # pilot order is intentionally not lexical
DF = pd.DataFrame({
    "Bug Id": [1, 2, 2, 3, 4, 5],
    "split": ["train", "train", "train", "train", "test", "val"],
    "contributor_id": ["zeta", "alpha", "alpha", "zeta", "alpha", "ghost"],
    "raw": ["r1", "r2", "duplicate", "r3", "r4", "r5"],
    "distilled_text": ["d1", "d2", "duplicate-d", "d3", "d4", "d5"],
    "creation_time": pd.to_datetime(["2020-01-01"] * 6, utc=True),
})
TRAIN_EMB = np.array([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32)
TEST_EMB = np.array([[.2, .8]], dtype=np.float32)


def _eval(mode="both", *, fuse_ibr=False, nis=None):
    return benchmark.evaluate(
        benchmark.BenchmarkConfig(text_mode=mode, top_k=2, tau=2, fuse_ibr=fuse_ibr), df=DF,
        label_encoder=LABELS, train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB, nis=nis,
    )


@pytest.mark.parametrize("mode", ["raw", "distilled", "both"])
def test_evaluate_matches_legacy_idf_votes_and_preserves_pilot_candidate_order(mode, monkeypatch):
    captured = {}
    original_fit = benchmark.CbrRetriever.fit
    original_dense = benchmark.CbrRetriever.compute_dense_sims

    def fit(self, texts, *args, **kwargs):
        captured["train"] = texts
        return original_fit(self, texts, *args, **kwargs)

    def dense(self, texts, *args, **kwargs):
        captured["test"] = texts
        return original_dense(self, texts, *args, **kwargs)

    monkeypatch.setattr(benchmark.CbrRetriever, "fit", fit)
    monkeypatch.setattr(benchmark.CbrRetriever, "compute_dense_sims", dense)
    result = _eval(mode)
    expected_train = {"raw": ["r1", "r2", "r3"], "distilled": ["d1", "d2", "d3"],
                      "both": ["r1 d1", "r2 d2", "r3 d3"]}
    expected_test = {"raw": ["r4"], "distilled": ["d4"], "both": ["r4 d4"]}
    assert captured == {"train": expected_train[mode], "test": expected_test[mode]}
    # First training row for duplicate Bug Id=2 is retained: 3 training cases.
    np.testing.assert_array_equal(result["train"]["Bug Id"], [1, 2, 3])
    np.testing.assert_array_equal(result["test"]["Bug Id"], [4])
    assert result["label_encoder"] == LABELS
    assert result["train_classes"].tolist() == [0, 1, 0]
    sims = TEST_EMB @ TRAIN_EMB.T
    np.testing.assert_allclose(result["similarities"], sims)
    expected = vote_scores(sims, np.array([0, 1, 0]), 3, 2, 2.)
    # Primitive default is the existing IDF weighting (not sum); columns are pilot order.
    np.testing.assert_array_equal(result["evaluations"][0]["scores"], expected)
    assert expected[0, 2] == 0.0  # candidate with no train/test bugs stays in the matrix
    assert result["true_indices"].tolist() == [1]


def test_sweep_uses_same_similarity_matrix_and_matches_legacy_primitives(monkeypatch):
    calls = 0
    original = benchmark.CbrRetriever.compute_dense_sims

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(benchmark.CbrRetriever, "compute_dense_sims", counted)
    result = benchmark.evaluate(
        benchmark.BenchmarkConfig(sweep=True, top_k=50), df=DF, label_encoder=LABELS,
        train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB,
    )
    assert calls == 1
    assert len(result["evaluations"]) == 21
    classes = np.array([0, 1, 0])
    sims = TEST_EMB @ TRAIN_EMB.T
    for actual, (k, tau) in zip(result["evaluations"],
                                ((k, t) for k in (5, 10, 20, 30, 50, 75, 100)
                                 for t in (1., 2., 4.))):
        expected = vote_scores(sims, classes, len(LABELS), k, tau)
        np.testing.assert_array_equal(actual["scores"], expected)
        assert actual["metrics"] == rank_metrics(expected, np.array([1]))


def test_sweep_precedes_fusion_and_does_not_require_ibr():
    result = benchmark.evaluate(
        benchmark.BenchmarkConfig(sweep=True, fuse_ibr=True), df=DF,
        label_encoder=LABELS, train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB,
    )
    assert "fusion" not in result
    assert len(result["evaluations"]) == 21


def test_wf_fusion_uses_normalized_cbr_and_candidate_aligned_injected_nis():
    nis = np.array([[0.1, 0.8, 0.0]], dtype=np.float32)
    result = _eval(fuse_ibr=True, nis=nis)
    raw = vote_scores(TEST_EMB @ TRAIN_EMB.T, np.array([0, 1, 0]), 3, 2, 2.)
    nps = normalize_rows(raw, how="minmax")
    np.testing.assert_array_equal(result["nps"], nps)
    np.testing.assert_array_equal(result["nis"], nis)
    assert [item["w_f"] for item in result["fusion"]] == [round(x / 10, 1) for x in range(1, 11)]
    for item in result["fusion"]:
        expected = nps + item["w_f"] * nis
        np.testing.assert_array_equal(item["scores"], expected)
        assert item["metrics"] == rank_metrics(expected, np.array([1]))


def test_owner_outside_pilot_candidates_fails_instead_of_expanding_candidates():
    bad = DF.copy()
    bad.loc[bad["Bug Id"] == 1, "contributor_id"] = "outside"
    with pytest.raises(ValueError, match="fuera del label_encoder"):
        benchmark.evaluate(benchmark.BenchmarkConfig(), df=bad, label_encoder=LABELS,
                           train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB)
