import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/benchmarks/openj9/eval_openj9_full.py"
spec = importlib.util.spec_from_file_location("eval_openj9_full", SCRIPT)
full = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full
spec.loader.exec_module(full)


def _frames():
    train = pd.DataFrame({"issue_number": [1, 2, 3], "owner": ["a", "b", "a"],
                          "text": ["alpha", "beta", "gamma"]})
    test = pd.DataFrame({"issue_number": [4, 5], "owner": ["a", "b"],
                         "text": ["query", "other"]})
    return train, test


def test_config_defaults_and_default_retrieval_offline():
    train, test = _frames()
    te = np.array([[1., 0.], [0., 1.]])
    tr = np.array([[1., 0.], [0., 1.], [.8, .2]])
    nis = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    cfg = full.EvaluationConfig(cpu=True)
    assert cfg.cbr_mode == "retrieval"
    assert (cfg.cbr_topk, cfg.cbr_tau, cfg.cbr_maxlen) == (50, 2.0, 384)
    assert (cfg.top_k, cfg.tau, cfg.lam, cfg.ip_c, cfg.ip_a, cfg.ip_d) == (15, .6, .01, 1.5, .5, .1)
    result = full.evaluate(cfg, train_df=train, test_df=test, train_embeddings=tr,
                           test_embeddings=te, nis=nis)
    # Explicit legacy IDF vote: sum(max(sim,0)^tau) * log(N/frequency), then minmax.
    sims = te @ tr.T
    raw = np.array([[1. + .8 ** 2, 0.], [0., 1.]]) * np.array([np.log(3 / 2), np.log(3)])
    mn, mx = raw.min(1, keepdims=True), raw.max(1, keepdims=True)
    expected = (raw - mn) / np.where(mx == mn, 1, mx - mn)
    np.testing.assert_allclose(result["nps"], expected)
    assert result["label_encoder"] == {"a": 0, "b": 1}
    assert len(result["curves"]) == 10


def test_fusion_matches_fixed_legacy_linear_rrf_gate_and_sweep():
    nps = np.array([[1., .4, 0.], [.7, .6, 0.]])
    nis = np.array([[0., 1., .3], [1., 0., 0.]])
    linear = full._make_fuser(nps, nis, "linear", 60, .2)
    expected_linear = nps + .5 * nis * np.array([[0], [1]])
    np.testing.assert_allclose(linear(.5), expected_linear)
    rrf = full._make_fuser(nps, nis, "rrf", 60, None)
    expected = full._rrf_rank_term(nps, 60) + .7 * full._rrf_rank_term(nis, 60) * (nis > 0)
    np.testing.assert_allclose(rrf(.7), expected)
    assert full._gate_mask(np.array([[1., .95, 0.]]), .1)[0, 0] == 1
    assert full._gate_mask(np.array([[1., .5, 0.]]), .1)[0, 0] == 0
    train, test = _frames()
    out = full.evaluate(full.EvaluationConfig(cpu=True, gate_sweep=True), train_df=train,
                         test_df=test, train_embeddings=np.eye(2)[[0, 1, 0]],
                         test_embeddings=np.eye(2), nis=np.zeros((2, 2)))
    assert [r["gate"] for r in out["curves"]] == [None, .1, .2, .3, .4, .5, .6, .7, .8, .9]


def test_full_candidate_alignment_and_curves_match_legacy_primitives():
    from triager_omega.modules.aggregator import normalize_rows, rank_metrics
    from triager_omega.retrieval import vote_scores

    train, test = _frames()
    test.loc[1, "owner"] = "c"  # candidato presente únicamente en test
    tr = np.array([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32)
    te = np.eye(2, dtype=np.float32)
    nis = np.array([[0., 1., .2], [.3, 0., 1.]], dtype=np.float32)
    expected_nps = normalize_rows(vote_scores(
        te @ tr.T, np.array([0, 1, 0]), 3, 50, 2.0,
    ))
    truth = np.array([0, 2])

    def rank_terms(values):
        # Referencia explícita por fila, sin usar helpers de fusión productivos.
        terms = np.empty(values.shape, dtype=np.float64)
        for i, row in enumerate(values):
            for rank, column in enumerate(sorted(range(len(row)), key=lambda j: -row[j])):
                terms[i, column] = 1.0 / (61 + rank)
        return terms

    for mode in ("linear", "rrf"):
        for gate in (None, .5):
            result = full.evaluate(
                full.EvaluationConfig(cpu=True, fusion=mode, gate=gate),
                train_df=train, test_df=test, train_embeddings=tr,
                test_embeddings=te, nis=nis,
            )
            assert result["label_encoder"] == {"a": 0, "b": 1, "c": 2}
            np.testing.assert_array_equal(result["true_indices"], truth)
            np.testing.assert_array_equal(result["nps"], expected_nps)
            sorted_scores = np.sort(expected_nps, axis=1)
            mask = np.ones((2, 1)) if gate is None else (
                sorted_scores[:, -1] - sorted_scores[:, -2] < gate
            )[:, None]
            for row in result["curves"]:
                weight = row["weight"]
                if mode == "linear":
                    expected = expected_nps + weight * nis * mask
                else:
                    expected = rank_terms(expected_nps) + weight * rank_terms(nis) * (nis > 0) * mask
                assert row["metrics"] == rank_metrics(expected, truth)


def test_actual_ibr_dispatch_semantic_and_lexical(monkeypatch):
    train, test = _frames()
    train["issue_number"] = [1, 2, 3]
    test["issue_number"] = [4, 5]
    interactions = pd.DataFrame({"issue_number": [1, 2], "dev": ["a", "b"],
        "kind": ["commit", "assignment"], "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"])})
    meta = pd.DataFrame({"issue_number": [4, 5], "created_at": pd.to_datetime(["2021-01-01", "2021-01-01"])})
    embeddings = np.array([[1., 0.], [0., 1.], [.8, .2]])
    queries = np.array([[1., 0.], [0., 1.]])

    encoded_batches = []

    class Encoder:
        def encode(self, texts, **kwargs):
            encoded_batches.append(list(texts))
            return queries if len(texts) == 2 else embeddings
    monkeypatch.setattr(full.InteractionRecommender, "_sbert", lambda self: Encoder())
    semantic = full.evaluate(full.EvaluationConfig(cpu=True), train_df=train, test_df=test,
        train_embeddings=embeddings, test_embeddings=queries, interactions_df=interactions, meta_df=meta)
    assert encoded_batches == [train["text"].tolist(), test["text"].tolist()]
    assert semantic["nis"].shape == (2, 2)
    np.testing.assert_array_equal(semantic["nis"], np.eye(2, dtype=np.float32))

    monkeypatch.setattr(full, "bm25_sims", lambda docs, qs: np.array([[2., 0., 1.], [0., 3., 0.]]))
    lexical = full.evaluate(full.EvaluationConfig(cpu=True, ibr_channel="lexical"), train_df=train,
        test_df=test, train_embeddings=embeddings, test_embeddings=queries,
        interactions_df=interactions, meta_df=meta)
    assert lexical["nis"].shape == (2, 2)
    np.testing.assert_array_equal(lexical["nis"], np.eye(2, dtype=np.float32))
