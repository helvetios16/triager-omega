import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks/openj9"))
import cbr_retrieval_openj9 as benchmark

from triager_omega.modules.aggregator import rank_metrics
from triager_omega.retrieval import build_label_encoder, rrf_fuse, vote_scores


TRAIN = pd.DataFrame({
    "issue_number": [1, 2, 3, 4],
    "owner": ["alice", "bob", "alice", "carol"],
    "text": ["fix alpha", "bug beta", "alpha crash", "beta stack"],
})
TEST = pd.DataFrame({
    "issue_number": [5, 6],
    "owner": ["bob", "dora"],
    "text": ["alpha bug", "beta crash"],
})
TRAIN_EMB = np.array([[1, 0], [0, 1], [0.8, 0.6], [0.1, 0.9]], dtype=np.float32)
TEST_EMB = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)


def _legacy_reference(train, test, dense, config, temporal_decay=None):
    encoder = build_label_encoder(train.owner, test.owner)
    classes = train.owner.map(encoder).to_numpy().astype(int)
    true = test.owner.map(encoder).to_numpy().astype(int)
    freq = np.bincount(classes, minlength=len(encoder)).astype(np.float32)
    sims = dense
    if config.hybrid:
        from triager_omega.retrieval import bm25_sims
        sims = rrf_fuse(dense, bm25_sims(train.text.tolist(), test.text.tolist()), config.rrf_k)
    if config.rerank:
        sims = benchmark.rerank_cross_encoder(
            sims, train.text.tolist(), test.text.tolist(), config.rerank_model,
            config.rerank_n, "cpu", config.rerank_batch,
        )
    if config.sweep:
        points = [(k, tau, config.vote) for k in (5, 10, 15, 20, 30, 50)
                  for tau in (1., 2., 4.)]
    elif config.vote_sweep:
        points = [(config.top_k, config.tau, v) for v in ("sum", "mean", "max", "idf", "rank")]
    else:
        points = [(config.top_k, config.tau, config.vote)]
    outputs = []
    for k, tau, vote in points:
        scores = vote_scores(sims, classes, len(encoder), k, tau, temporal_decay, vote, freq)
        outputs.append((scores, rank_metrics(scores, true)))
    return sims, outputs


def test_dense_and_hybrid_sweeps_equal_legacy_primitives_and_encode_once(monkeypatch):
    calls = {"dense": 0, "lexical": 0}
    original_dense = benchmark.CbrRetriever.compute_dense_sims
    original_lexical = benchmark.CbrRetriever.compute_lexical_sims

    def dense(self, texts, embeddings=None):
        calls["dense"] += 1
        return original_dense(self, texts, embeddings)

    def lexical(self, texts):
        calls["lexical"] += 1
        return original_lexical(self, texts)

    monkeypatch.setattr(benchmark.CbrRetriever, "compute_dense_sims", dense)
    monkeypatch.setattr(benchmark.CbrRetriever, "compute_lexical_sims", lexical)
    for hybrid in (False, True):
        config = benchmark.BenchmarkConfig(hybrid=hybrid, sweep=True)
        result = benchmark.evaluate(config, train_df=TRAIN, test_df=TEST,
                                    train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB)
        expected_sims, expected = _legacy_reference(TRAIN, TEST, TEST_EMB @ TRAIN_EMB.T, config)
        np.testing.assert_array_equal(result["similarities"], expected_sims)
        assert len(result["evaluations"]) == len(expected) == 18
        for actual, (scores, metrics) in zip(result["evaluations"], expected):
            np.testing.assert_array_equal(actual["scores"], scores)
            assert actual["metrics"] == metrics
    assert calls == {"dense": 2, "lexical": 1}


def test_vote_sweep_test_only_candidate_and_decay_match_legacy(monkeypatch):
    config = benchmark.BenchmarkConfig(vote_sweep=True, decay=True, lam=0.3)
    meta = pd.DataFrame({"issue_number": [1, 2, 3, 4],
                         "created_at": pd.to_datetime(["2020-01-01", "2021-01-01",
                                                        "2022-01-01", "2023-01-01"], utc=True)})
    result = benchmark.evaluate(config, train_df=TRAIN, test_df=TEST,
                                train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB,
                                meta_df=meta)
    decay = benchmark._temporal_decay(TRAIN, config, meta)
    expected_sims, expected = _legacy_reference(TRAIN, TEST, TEST_EMB @ TRAIN_EMB.T,
                                                  config, decay)
    np.testing.assert_array_equal(result["similarities"], expected_sims)
    assert result["label_encoder"] == {"alice": 0, "bob": 1, "carol": 2, "dora": 3}
    np.testing.assert_array_equal(result["temporal_decay"], decay)
    for actual, (scores, metrics) in zip(result["evaluations"], expected):
        np.testing.assert_array_equal(actual["scores"], scores)
        assert actual["metrics"] == metrics


def test_rerank_releases_encoder_before_shared_voting(monkeypatch):
    order = []

    class FakeEncoder:
        def to(self, device):
            assert device == "cpu"
            order.append("offload")

    original_init = benchmark.CbrRetriever.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._encoder = FakeEncoder()

    monkeypatch.setattr(benchmark.CbrRetriever, "__init__", init)

    def rerank(sims, *args):
        order.append("rerank")
        return sims[:, ::-1].copy()
    original_score = benchmark.CbrRetriever.score_sims
    def score(self, *args, **kwargs):
        order.append("vote")
        return original_score(self, *args, **kwargs)
    monkeypatch.setattr(benchmark, "rerank_cross_encoder", rerank)
    monkeypatch.setattr(benchmark.CbrRetriever, "score_sims", score)
    result = benchmark.evaluate(benchmark.BenchmarkConfig(rerank=True), train_df=TRAIN,
                                test_df=TEST, train_embeddings=TRAIN_EMB,
                                test_embeddings=TEST_EMB)
    assert order == ["offload", "rerank", "vote"]
    np.testing.assert_array_equal(result["similarities"], (TEST_EMB @ TRAIN_EMB.T)[:, ::-1])


def test_finetune_uses_shared_encoder_before_embedding_queries(monkeypatch):
    events = []
    shared_encoder = object()
    monkeypatch.setattr(benchmark.CbrRetriever, "encoder", property(lambda self: shared_encoder))
    def finetune(model, *args):
        events.append(("finetune", model))
    monkeypatch.setattr(benchmark, "_finetune", finetune)
    original_dense = benchmark.CbrRetriever.compute_dense_sims
    def dense(self, *args, **kwargs):
        events.append(("encode", self.encoder))
        return original_dense(self, *args, **kwargs)
    monkeypatch.setattr(benchmark.CbrRetriever, "compute_dense_sims", dense)
    benchmark.evaluate(benchmark.BenchmarkConfig(finetune=True), train_df=TRAIN, test_df=TEST,
                       train_embeddings=TRAIN_EMB, test_embeddings=TEST_EMB)
    assert [event[0] for event in events] == ["finetune", "encode"]
    assert events[0][1] is events[1][1]
