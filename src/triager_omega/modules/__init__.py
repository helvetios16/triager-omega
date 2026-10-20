"""Módulos del sistema completo (post-CBR).

  - ibr.py        Módulo 4 — Interaction-Based Recommender (réplica de TriagerX).
  - aggregator.py Módulo 5 — fusión FS = NPS + W_f·NIS.
"""

from triager_omega.modules.aggregator import TriagerAggregator, normalize_rows, rank_metrics
from triager_omega.modules.ibr import InteractionRecommender

__all__ = [
    "InteractionRecommender",
    "TriagerAggregator",
    "normalize_rows",
    "rank_metrics",
]
