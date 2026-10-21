from io import StringIO

import pandas as pd
from loguru import logger

from triager_omega.config import Settings


def test_mozilla_and_pilot_properties_are_canonical_independent_of_existence(tmp_path):
    artifacts = tmp_path / "artifacts"
    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=artifacts)
    assert cfg.mozilla_dir == artifacts / "mozilla"
    assert cfg.pilot_dir == artifacts / "mozilla" / "pilot"
    (artifacts / "mozilla").mkdir(parents=True)
    (artifacts / "pilot").mkdir()
    assert cfg.mozilla_dir == artifacts / "mozilla"
    assert cfg.pilot_dir == artifacts / "mozilla" / "pilot"


def test_mozilla_read_prefers_canonical_when_both_exist(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    canonical = cfg.mozilla_dir / "repo_interactions.parquet"
    legacy = cfg.artifacts_dir / "repo_interactions.parquet"
    canonical.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    canonical.touch()
    legacy.touch()
    assert cfg.mozilla_read_path("repo_interactions.parquet") == canonical


def test_mozilla_read_uses_legacy_with_warning_only_if_canonical_missing(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    legacy = cfg.artifacts_dir / "discussion_interactions.parquet"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    messages = StringIO()
    sink_id = logger.add(messages, format="{message}", level="WARNING")
    try:
        selected = cfg.mozilla_read_path("discussion_interactions.parquet")
    finally:
        logger.remove(sink_id)
    assert selected == legacy
    assert "[LEGACY_FALLBACK]" in messages.getvalue()
    assert str(legacy) in messages.getvalue()


def test_missing_legacy_and_canonical_reads_return_canonical(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    assert cfg.mozilla_read_path("repo_interactions.parquet") == cfg.repo_interactions_path
    assert cfg.pilot_read_path("splits.parquet") == cfg.pilot_splits_path


def test_repo_producer_writes_canonical_path_without_mining(tmp_path, monkeypatch):
    from triager_omega.data import repo_miner

    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    expected = pd.DataFrame({"bug_id": [123], "Contributor Id": [101]})
    monkeypatch.setattr(repo_miner, "mine_raw_commits", lambda **kwargs: expected)
    monkeypatch.setattr(repo_miner, "extract_interactions", lambda frame: frame)
    monkeypatch.setattr(repo_miner, "bridge_to_contributor_id", lambda frame, **kwargs: frame)
    result = repo_miner.build_interaction_table(cfg=cfg)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(pd.read_parquet(cfg.repo_interactions_path), expected)
    assert not (cfg.artifacts_dir / "repo_interactions.parquet").exists()


def test_pilot_legacy_precedence_and_write_destinations(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    legacy = cfg.artifacts_dir / "pilot" / "label_encoder.json"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert cfg.pilot_read_path("label_encoder.json") == legacy
    assert cfg.pilot_splits_path == cfg.artifacts_dir / "mozilla" / "pilot" / "splits.parquet"
    assert cfg.repo_interactions_path == cfg.artifacts_dir / "mozilla" / "repo_interactions.parquet"
    assert cfg.discussion_interactions_path == cfg.artifacts_dir / "mozilla" / "discussion_interactions.parquet"


def test_mozilla_ibr_matrix_real_scoring_preserves_encoder_columns(monkeypatch):
    import numpy as np
    import pandas as pd
    import torch
    from triager_omega.config import Settings
    from triager_omega.modules.ibr import InteractionIndex, InteractionRecommender
    import importlib.util
    import sys
    from pathlib import Path
    script = Path(__file__).parents[1] / "scripts/benchmarks/mozilla/cbr_retrieval_pilot.py"
    spec = importlib.util.spec_from_file_location("mozilla_pilot_test", script)
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    _ibr_matrix = pilot._ibr_matrix

    now = pd.Timestamp("2021-01-01")
    encoder = {"10": 0, "2": 1}  # intentionally not lexical numeric order
    index = InteractionIndex(
        case_ids=np.array([11, 12]), active_candidates={10, 2},
        interactions={
            11: [(2, "commit", pd.Timestamp("2020-01-01"))],
            12: [(10, "assignment", pd.Timestamp("2020-01-01"))],
        },
    )
    ibr = InteractionRecommender(cfg=Settings(ip_contribution=1.5, ip_assignment=.5, ip_discussion=.1))
    ibr.text_mode = "raw"
    ibr.set_index(index)
    ibr._train_embeddings = torch.tensor([[3., 0.], [0., 2.]])
    encoded = []
    class Encoder:
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.array([[1., .4]])
    monkeypatch.setattr(ibr, "_sbert", lambda: Encoder())
    frame = pd.DataFrame({"raw": ["query"], "creation_time": [now]})
    actual = _ibr_matrix(frame, encoder, ibr)

    from sentence_transformers import util
    similarities = util.cos_sim(torch.tensor([[1., .4]]), ibr._train_embeddings)[0].numpy()
    # cosine below tau excludes the assignment case; commit case 2 is the sole signal.
    expected = np.array([[0., 1.]], dtype=np.float32)
    assert similarities[1] < .6 < similarities[0]
    np.testing.assert_allclose(actual, expected)
    assert encoded == ["query"]
