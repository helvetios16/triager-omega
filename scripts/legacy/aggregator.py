"""Historical Mozilla pilot aggregator; not exported by the active package.

The former CLI is intentionally not retained: its ``_build`` path always failed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from loguru import logger

from triager_omega.config import Settings, settings
from triager_omega.modules.aggregator import linear_fuse, normalize_rows, rank_metrics
from triager_omega.modules.ibr import InteractionRecommender, combine_text, load_pilot_text


class TriagerAggregator:
    """Archived pilot wrapper combining CBR NPS with IBR NIS."""

    def __init__(self, cbr: Any, ibr: InteractionRecommender,
                 w_f: float | None = None, nps_norm: str = "minmax",
                 cfg: Settings = settings):
        self.cbr, self.ibr, self.cfg = cbr, ibr, cfg
        self.w_f = cfg.ibr_w_f if w_f is None else w_f
        self.nps_norm = nps_norm
        self.label_encoder = json.loads(
            cfg.pilot_read_path("label_encoder.json").read_text(encoding="utf-8")
        )
        self.num_classes = len(self.label_encoder)

    def _nis_matrix(self, texts: list[str], t_nows) -> np.ndarray:
        q_emb = torch.as_tensor(
            self.ibr.encode_texts(texts, batch_size=64, show_progress_bar=True),
            dtype=torch.float32,
        )
        result = np.zeros((len(texts), self.num_classes), dtype=np.float32)
        for i, now in enumerate(t_nows):
            scores = self.ibr.score_embedding(q_emb[i:i + 1], now)
            for dev, idx in self.label_encoder.items():
                result[i, idx] = scores.get(int(dev), 0.0)
        return result

    def _matrices(self, split: str, limit: int | None = None):
        df, _ = load_pilot_text(self.cbr.text_mode, self.cfg)
        part = df[df["split"] == split].drop_duplicates("Bug Id").reset_index(drop=True)
        if limit:
            part = part.head(limit)
        cbr_texts = combine_text(part, self.cbr.text_mode).tolist()
        ibr_texts = combine_text(part, self.ibr.text_mode).tolist()
        nps = self.cbr.nps_matrix(cbr_texts, how=self.nps_norm)
        nis = self._nis_matrix(ibr_texts, list(part["creation_time"]))
        truth = part["contributor_id"].astype(str).map(self.label_encoder).to_numpy(dtype=int)
        return nps, nis, truth

    def evaluate(self, split: str = "test", ks=(1, 3, 5, 10),
                 limit: int | None = None) -> dict:
        nps, nis, truth = self._matrices(split, limit)
        metrics = rank_metrics(linear_fuse(nps, nis, self.w_f), truth, ks)
        metrics["w_f"] = self.w_f
        metrics["_cbr_only"] = rank_metrics(nps, truth, ks)
        metrics["_ibr_only"] = rank_metrics(nis, truth, ks)
        return metrics

    def grid_search_wf(self, split: str = "val", ks=(1, 3, 5, 10),
                       select="hit@5", grid=None, limit: int | None = None) -> dict:
        if grid is None:
            grid = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
        nps, nis, truth = self._matrices(split, limit)
        table = []
        for weight in grid:
            row = rank_metrics(linear_fuse(nps, nis, weight), truth, ks)
            row["w_f"] = weight
            table.append(row)
        best = max(table, key=lambda row: row[select])
        return {"select": select, "best": best, "grid": table}
