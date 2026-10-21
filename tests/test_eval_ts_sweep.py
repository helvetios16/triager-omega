import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks/typescript"))
import eval_ts_sweep as benchmark
from triager_omega.modules.aggregator import rank_metrics
from triager_omega.retrieval import bm25_sims, build_label_encoder, rrf_fuse, vote_scores


TRAIN = pd.DataFrame({"issue_number": [1, 2, 3, 4], "owner": ["a", "b", "a", "c"],
                      "text": ["alpha fix", "beta bug", "alpha crash", "beta stack"]})
TEST = pd.DataFrame({"issue_number": [5, 6], "owner": ["b", "d"],
                     "text": ["alpha bug", "beta crash"]})
TRAIN_EMB = np.array([[1, 0], [0, 1], [.8, .6], [.1, .9]], dtype=np.float32)
TEST_EMB = np.array([[.9, .1], [.2, .8]], dtype=np.float32)


def reference(modality, config):
    labels = build_label_encoder(TRAIN.owner, TEST.owner)
    cls = TRAIN.owner.map(labels).to_numpy(dtype=np.int64)
    true = TEST.owner.map(labels).to_numpy(dtype=np.int64)
    dense = TEST_EMB @ TRAIN_EMB.T
    modes = [("dense", dense)]
    if modality != "both":
        modes.append(("hybrid", rrf_fuse(dense, bm25_sims(TRAIN.text.tolist(), TEST.text.tolist()), k=60)))
    expected = []
    for mode, sims in modes:
        for vote in config.votes:
            for k in config.ks:
                for tau in config.taus:
                    scores = vote_scores(sims, cls, len(labels), k, tau, vote=vote,
                                         dev_freq=np.bincount(cls, minlength=len(labels)))
                    expected.append((mode, scores, rank_metrics(scores, true, ks=(1, 5, 10))))
    return labels, modes, expected


def test_evaluate_views_match_previous_primitives_and_reuse_sims(monkeypatch):
    calls = {"dense": 0, "lexical": 0}
    dense_original = benchmark.CbrRetriever.compute_dense_sims
    lexical_original = benchmark.CbrRetriever.compute_lexical_sims

    def dense(self, *args, **kwargs):
        calls["dense"] += 1
        return dense_original(self, *args, **kwargs)

    def lexical(self, *args, **kwargs):
        calls["lexical"] += 1
        return lexical_original(self, *args, **kwargs)

    monkeypatch.setattr(benchmark.CbrRetriever, "compute_dense_sims", dense)
    monkeypatch.setattr(benchmark.CbrRetriever, "compute_lexical_sims", lexical)
    monkeypatch.setattr(benchmark.CbrRetriever, "encoder",
                        property(lambda self: (_ for _ in ()).throw(AssertionError("unexpected encode"))))
    config = benchmark.BenchmarkConfig(ks=(1, 3), taus=(0.5, 2.0))
    frames = {name: (TRAIN, TEST) for name in ("raw", "distilled", "both")}
    embeddings = {name: (TRAIN_EMB, TEST_EMB) for name in frames}
    result = benchmark.evaluate(config, dataframes=frames, embeddings=embeddings)
    assert calls == {"dense": 3, "lexical": 2}
    for modality in frames:
        labels, modes, expected = reference(modality, config)
        details = result["views"][modality]
        assert details["label_encoder"] == labels
        for mode, sims in modes:
            np.testing.assert_array_equal(details["similarities"][mode], sims)
        actual = [row for row in result["results"] if row["modality"] == modality]
        assert len(actual) == len(expected)
        for row, (mode, scores, metrics), actual_scores in zip(
                actual, expected, details["score_matrices"]):
            assert row["type"] == mode
            assert {key: row[key] for key in metrics} == metrics
            np.testing.assert_array_equal(actual_scores, scores)
            assert row["k"] in config.ks


def test_bm25_cache_hit_miss_and_misaligned_cache_recomputed(tmp_path, monkeypatch):
    config = benchmark.BenchmarkConfig(ks=(1,), taus=(1.0,), votes=("sum",))
    calls = []
    original = benchmark.CbrRetriever.compute_lexical_sims

    def lexical(self, texts):
        calls.append(1)
        return original(self, texts)

    monkeypatch.setattr(benchmark.CbrRetriever, "compute_lexical_sims", lexical)
    embeddings = {"raw": (TRAIN_EMB, TEST_EMB)}
    cache = tmp_path / "cache"

    benchmark.evaluate(config, dataframes={"raw": (TRAIN, TEST)}, embeddings=embeddings,
                       cache_dir=cache)
    cached_path = cache / "bm25_raw.npy"
    assert cached_path.exists()
    assert np.load(cached_path).shape == (len(TEST), len(TRAIN))
    assert len(calls) == 1

    benchmark.evaluate(config, dataframes={"raw": (TRAIN, TEST)}, embeddings=embeddings,
                       cache_dir=cache)
    assert len(calls) == 1

    np.save(cached_path, np.zeros((1, 1)))
    benchmark.evaluate(config, dataframes={"raw": (TRAIN, TEST)}, embeddings=embeddings,
                       cache_dir=cache)
    assert len(calls) == 2
    assert np.load(cached_path).shape == (len(TEST), len(TRAIN))


def test_main_writes_json_and_markdown_to_configured_paths(tmp_path, monkeypatch):
    json_path = tmp_path / "nested" / "results.json"
    report_path = tmp_path / "nested" / "report.md"
    config = benchmark.BenchmarkConfig(results_path=json_path, report_path=report_path,
                                       embeddings_dir=tmp_path / "embeddings")
    monkeypatch.setattr(benchmark, "CONFIG", config)
    views = {"raw": (TRAIN, TEST)}
    monkeypatch.setattr(benchmark, "_load_views", lambda received: views)
    best = {"k": 10, "tau": 1.0, "vote": "sum", "hit@1": 0.5,
            "hit@5": 0.75, "hit@10": 1.0, "mrr": 0.7}
    monkeypatch.setattr(benchmark, "evaluate", lambda *args, **kwargs: {
        "results": [{"modality": "raw", "type": "dense", **best}],
        "summary_bests": [("raw_dense", best)], "views": {},
    })

    benchmark.main()

    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {"modality": "raw", "type": "dense", **best}
    ]
    markdown = report_path.read_text(encoding="utf-8")
    assert "\\n" not in markdown
    assert "# Resultados de Evaluación" in markdown
    assert "\n\n## Tabla Comparativa Head-to-Head\n\n" in markdown
    assert "| **raw_dense** | **0.5000** | 0.7500 | 1.0000 | 0.7000 | k=10, τ=1.0, vote=sum |\n" in markdown


def test_defaults_preserve_historical_sweep_and_view_modes():
    config = benchmark.CONFIG
    assert config.model == "sentence-transformers/all-mpnet-base-v2"
    assert config.max_seq_len == 384
    assert (config.ks, config.taus, config.votes) == ((10, 20, 30, 50), (0.5, 1.0, 2.0, 4.0), ("sum", "idf"))
    assert config.rrf_k == 60
    assert len(config.ks) * len(config.taus) * len(config.votes) == 32
