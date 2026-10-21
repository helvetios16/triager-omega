from __future__ import annotations

import numpy as np
import pytest

from triager_omega.retrieval.retriever import CbrConfig, CbrRetriever
from triager_omega.retrieval.voting import vote_scores


TRAIN_TEXTS = ["a", "b", "c"]
TRAIN_OWNERS = ["alice", "bob", "alice"]
TRAIN_EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)


def make_retriever(**config):
    retriever = CbrRetriever(CbrConfig(device="cpu", top_k=2, vote_strategy="sum", **config))
    retriever.fit(TRAIN_TEXTS, TRAIN_OWNERS, precomputed_embeddings=TRAIN_EMBEDDINGS)
    return retriever


def test_default_device_is_resolved_when_config_is_created(monkeypatch):
    from types import SimpleNamespace
    from triager_omega.retrieval import retriever as module

    monkeypatch.setattr(module, "Settings", lambda: SimpleNamespace(torch_device="cpu"))
    assert CbrConfig().device == "cpu"
    monkeypatch.setattr(module, "Settings", lambda: SimpleNamespace(torch_device="cuda"))
    assert CbrConfig().device == "cuda"
    assert CbrConfig(device="custom").device == "custom"


def test_score_delegates_to_score_sims_and_matches_vote_scores():
    retriever = make_retriever()
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = query_embeddings @ TRAIN_EMBEDDINGS.T
    decay = np.array([1.0, 0.5, 0.25], dtype=np.float32)

    score = retriever.score(["query"], query_embeddings, temporal_decay=decay)
    from_sims = retriever.score_sims(sims, temporal_decay=decay)
    direct = vote_scores(
        sims, retriever.train_classes, len(retriever.label_encoder), top_k=2,
        tau=1.0, decay=decay, vote="sum", dev_freq=retriever.dev_frequencies,
    )
    np.testing.assert_allclose(score, from_sims)
    np.testing.assert_allclose(score, direct)


def test_explicit_zero_neighbors_is_consistent_between_scoring_entry_points():
    retriever = make_retriever()
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = query @ TRAIN_EMBEDDINGS.T

    scores = retriever.score(["query"], query, top_k=0)
    np.testing.assert_array_equal(scores, retriever.score_sims(sims, top_k=0))
    np.testing.assert_array_equal(scores, np.zeros((1, 2), dtype=np.float32))


def test_score_uses_configured_and_overridden_rrf_k(monkeypatch):
    retriever = make_retriever(rrf_k=7)
    dense = np.array([[0.9, 0.2, 0.1]], dtype=np.float32)
    lexical = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
    monkeypatch.setattr(retriever, "compute_dense_sims", lambda texts, embeddings: dense)
    monkeypatch.setattr(retriever, "compute_lexical_sims", lambda texts: lexical)
    from triager_omega.retrieval.fusion import rrf_fuse

    expected_config = retriever.score(["q"], mode="hybrid")
    expected_override = retriever.score(["q"], mode="hybrid", rrf_k=2)
    assert not np.allclose(expected_config, expected_override)
    np.testing.assert_allclose(
        retriever.score(["q"], mode="hybrid"),
        retriever.score_sims(rrf_fuse(dense, lexical, k=7)),
    )
    np.testing.assert_allclose(
        retriever.score(["q"], mode="hybrid", rrf_k=2),
        retriever.score_sims(rrf_fuse(dense, lexical, k=2)),
    )


def test_rank_preserves_class_order_for_tied_scores(monkeypatch):
    retriever = make_retriever()
    monkeypatch.setattr(retriever, "score", lambda *args, **kwargs: np.array([[1.0, 1.0]]))
    assert retriever.rank(["q"], top_n=2) == [["alice", "bob"]]


def test_scoring_requires_fit_even_for_precomputed_sims():
    retriever = CbrRetriever(CbrConfig(device="cpu"))
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        retriever.score_sims(np.ones((1, 0)))
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        retriever.score(["q"], mode="lexical")


def test_fit_validates_training_alignment_and_embedding_shape():
    retriever = CbrRetriever(CbrConfig(device="cpu"))
    with pytest.raises(ValueError, match="misma longitud"):
        retriever.fit(["a"], ["alice", "bob"], precomputed_embeddings=np.ones((1, 2)))
    with pytest.raises(ValueError, match="2D"):
        retriever.fit(TRAIN_TEXTS, TRAIN_OWNERS, precomputed_embeddings=np.ones(3))
    with pytest.raises(ValueError, match="fila por texto"):
        retriever.fit(TRAIN_TEXTS, TRAIN_OWNERS, precomputed_embeddings=np.ones((2, 2)))


def test_query_and_similarity_inputs_validate_alignment():
    retriever = make_retriever()
    with pytest.raises(ValueError, match="2D"):
        retriever.compute_dense_sims(["q"], np.ones(2))
    with pytest.raises(ValueError, match="fila por texto"):
        retriever.compute_dense_sims(["q"], np.ones((2, 2)))
    with pytest.raises(ValueError, match="dimensiones"):
        retriever.compute_dense_sims(["q"], np.ones((1, 3)))
    with pytest.raises(ValueError, match="2D"):
        retriever.score_sims(np.ones(3))
    with pytest.raises(ValueError, match="columna por caso"):
        retriever.score_sims(np.ones((1, 2)))
    with pytest.raises(ValueError, match="temporal_decay"):
        retriever.score_sims(np.ones((1, 3)), temporal_decay=np.ones(2))


def test_dense_sweep_does_not_reinvoke_encoder():
    retriever = make_retriever()
    class NoEncoder:
        def encode(self, *args, **kwargs):
            raise AssertionError("encoder should not be invoked in score_sims sweep")

    retriever._encoder = NoEncoder()
    precomputed_sims = np.array([[0.8, 0.1, 0.6]], dtype=np.float32)
    for k in (1, 2, 3):
        retriever.score_sims(precomputed_sims, top_k=k)
