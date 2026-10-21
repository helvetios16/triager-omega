import math

import numpy as np
import pandas as pd
import pytest
import torch

from triager_omega.modules.ibr import IbrConfig, IbrScorer, InteractionIndex


def _legacy_reference(similarities, case_ids, candidates, interactions, config, t_now, policy="semantic"):
    """Frozen pre-engine algorithm: torch.topk/cosine scores and ordered sums."""
    sims = np.asarray(similarities)
    k = min(config.top_k, len(sims))
    if policy == "semantic":
        vals, indices = torch.topk(torch.as_tensor(sims), k)
        selected = [(v, i) for v, i in zip(vals.tolist(), indices.tolist()) if v >= config.tau]
    else:
        indices = np.argpartition(-sims, kth=k - 1)[:k]
        selected = [(sims[i], i) for i in indices if sims[i] > 0]
    raw = {dev: 0.0 for dev in candidates}
    for sim, idx in selected:
        for dev, kind, ts in interactions.get(case_ids[idx], ()):
            if dev not in raw or (t_now is not None and ts >= t_now):
                continue
            point = config.ip_contribution if kind in {"commit", "review", "pull_request", "commits"} else (config.ip_assignment if kind == "assignment" else config.ip_discussion)
            days = 0 if t_now is None else (t_now - ts).days
            raw[dev] += float(sim) * point * (1.0 if t_now is None else math.exp(-config.decay_lambda * days))
    maximum = max(raw.values(), default=0.0)
    return {dev: (value / maximum if maximum > 0 else 0.0) for dev, value in raw.items()}


def test_engine_semantic_and_lexical_retrieval_keep_distinct_selection_policies():
    now = pd.Timestamp("2024-01-10", tz="UTC")
    index = InteractionIndex(
        case_ids=np.array([10, 20, 30]), active_candidates={7, "owner"},
        interactions={
            10: [(7, "assignment", now - pd.Timedelta(days=1))],
            20: [("owner", "commit", now - pd.Timedelta(days=2))],
            30: [(7, "discussion", now - pd.Timedelta(days=1))],
        },
    )
    scorer = IbrScorer(IbrConfig(top_k=2, tau=0.8, decay_lambda=0.0), index)
    assert scorer.score([0.9, 0.75, 0.1], now, policy="semantic") == {7: 1.0, "owner": 0.0}
    assert scorer.score([0.9, 0.75, 0.1], now, policy="lexical") == pytest.approx({7: 0.4, "owner": 1.0})


def test_engine_temporal_ip_normalization_and_missing_time_behavior():
    now = pd.Timestamp("2024-01-10", tz="UTC")
    index = InteractionIndex(np.array([1, 2, 3, 4]), {1, 2, 3}, {
        1: [(1, "commit", now - pd.Timedelta(days=10))],
        2: [(2, "assignment", now - pd.Timedelta(days=1))],
        3: [(3, "discussion", now)], 4: [(1, "assignment", now + pd.Timedelta(days=1))],
    })
    scorer = IbrScorer(IbrConfig(top_k=4, tau=0.0, decay_lambda=0.01), index)
    result = scorer.score([1., 1., 1., 1.], now)
    assert result[1] == 1.0
    assert result[2] == pytest.approx(0.5 * math.exp(-0.01) / (1.5 * math.exp(-0.1)))
    assert result[3] == 0.0
    assert scorer.score([1., 1., 1., 1.], None)[1] == 1.0
    assert scorer.score([0., 0., 0., 0.], now) == {1: 0., 2: 0., 3: 0.}


def test_engine_preserves_nat_propagation():
    now = pd.Timestamp("2024-01-10", tz="UTC")
    index = InteractionIndex(np.array([1]), {1}, {1: [(1, "assignment", pd.NaT)]})
    scorer = IbrScorer(IbrConfig(top_k=1, tau=0.0), index)
    assert math.isnan(scorer.score([1.], now)[1])


def test_engine_matches_frozen_legacy_reference_for_semantic_and_lexical():
    now = pd.Timestamp("2024-03-01", tz="UTC")
    ids = ["case-a", 11, "case-c", 12]
    candidates = ["alice", 7, "bob"]
    interactions = {
        "case-a": [("alice", "commit", now - pd.Timedelta(days=3)), ("alice", "assignment", now - pd.Timedelta(days=3))],
        11: [(7, "discussion", now - pd.Timedelta(days=1))],
        "case-c": [("bob", "review", now), ("bob", "assignment", pd.NaT)],
        12: [("unknown", "assignment", now - pd.Timedelta(days=2))],
    }
    config = IbrConfig(top_k=3, tau=0.6, decay_lambda=0.02)
    scorer = IbrScorer(config, InteractionIndex(np.asarray(ids, dtype=object), candidates, interactions))
    sims = np.asarray([0.9, 0.6, 0.6, 0.2], dtype=np.float32)
    for policy in ("semantic", "lexical"):
        expected = _legacy_reference(sims, ids, candidates, interactions, config, now, policy)
        actual = scorer.score(sims, now, policy=policy)
        assert list(actual) == candidates
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=0, atol=0, equal_nan=True)


def test_engine_rejects_similarity_shape_or_alignment_mismatch():
    scorer = IbrScorer(IbrConfig(top_k=2), InteractionIndex(np.array([1, 2]), ["x"], {}))
    with pytest.raises(ValueError, match="1D"):
        scorer.score([[0.8, 0.7]])
    with pytest.raises(ValueError, match="igual longitud"):
        scorer.score([0.8])


def test_ibr_cosine_similarity_uses_sentence_transformers_api_exactly():
    from sentence_transformers import util
    from triager_omega.modules.ibr import InteractionRecommender

    ibr = InteractionRecommender()
    ibr._train_embeddings = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    actual = ibr.cosine_similarities(query)
    expected = util.cos_sim(query, ibr._train_embeddings)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, torch.mv(ibr._train_embeddings, query[0]))


def test_ibr_config_defaults_match_canonical_settings():
    from triager_omega.config import Settings

    assert IbrConfig().top_k == Settings().ibr_top_k_retrieve == 20
    assert IbrConfig().tau == Settings().ibr_tau == 0.6
    assert IbrConfig(top_k=9, tau=0.3).top_k == 9
    assert IbrConfig(top_k=9, tau=0.3).tau == 0.3
