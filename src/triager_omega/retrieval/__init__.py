"""Módulo de Case-Based Reasoning (CBR) por Recuperación Semántica Zero-Shot y Fusión Híbrida."""

from triager_omega.retrieval.bm25 import bm25_sims
from triager_omega.retrieval.encoder import build_label_encoder
from triager_omega.retrieval.fusion import rerank_cross_encoder, rrf_fuse
from triager_omega.retrieval.retriever import CbrConfig, CbrRetriever
from triager_omega.retrieval.voting import vote_scores

__all__ = [
    "CbrRetriever",
    "CbrConfig",
    "vote_scores",
    "bm25_sims",
    "rrf_fuse",
    "rerank_cross_encoder",
    "build_label_encoder",
]
