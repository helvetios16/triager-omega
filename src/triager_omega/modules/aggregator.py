"""Pure NumPy fusion and ranking helpers shared by benchmark consumers."""

from __future__ import annotations

import numpy as np


def normalize_rows(logits: np.ndarray, how: str = "minmax") -> np.ndarray:
    """Normaliza cada fila a [0,1] (minmax, TriagerX) o a probabilidad (softmax)."""
    if how == "softmax":
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    mn = logits.min(axis=1, keepdims=True)
    mx = logits.max(axis=1, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, mx - mn)
    return (logits - mn) / rng


def rank_metrics(scores: np.ndarray, true_idx: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    """Hit@K y MRR; en empates estables gana la clase de menor índice."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = (order == true_idx[:, None]).argmax(axis=1)
    metrics = {f"hit@{k}": float((ranks < k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
    return metrics


def gate_mask(nps: np.ndarray, gate: float | None) -> np.ndarray:
    """Máscara [N,1] que habilita NIS cuando el margen NPS top1-top2 < gate."""
    if gate is None:
        return np.ones((nps.shape[0], 1), dtype=np.float32)
    part = np.partition(-nps, 1, axis=1)
    top1, top2 = -part[:, 0], -part[:, 1]
    return ((top1 - top2) < gate).astype(np.float32)[:, None]


def rrf_rank_term(mat: np.ndarray, k: int) -> np.ndarray:
    """RRF rank term con el argsort histórico (incluyendo su desempate actual)."""
    ranks = np.empty_like(mat, dtype=np.float64)
    order = np.argsort(-mat, axis=1)
    rows = np.arange(mat.shape[0])[:, None]
    ranks[rows, order] = np.arange(mat.shape[1])[None, :]
    return 1.0 / (k + ranks + 1.0)


def linear_fuse(nps: np.ndarray, nis: np.ndarray, weight: float,
                gate: float | None = None) -> np.ndarray:
    """Fusión lineal compartida; requiere matrices alineadas [N,C], no normaliza."""
    _validate_alignment(nps, nis)
    return nps + weight * nis * gate_mask(nps, gate)


def rrf_fuse(nps: np.ndarray, nis: np.ndarray, weight: float, k: int = 60,
             gate: float | None = None) -> np.ndarray:
    """Fusión RRF de rankings de desarrolladores, conservando máscara NIS>0."""
    _validate_alignment(nps, nis)
    return rrf_rank_term(nps, k) + weight * rrf_rank_term(nis, k) * (nis > 0) * gate_mask(nps, gate)


def make_fuser(nps: np.ndarray, nis: np.ndarray, mode: str, rrf_k: int,
               gate: float | None = None):
    """Precalcula términos para barridos de pesos."""
    _validate_alignment(nps, nis)
    gmask = gate_mask(nps, gate)
    if mode == "rrf":
        rr_nps = rrf_rank_term(nps, rrf_k)
        rr_nis = rrf_rank_term(nis, rrf_k) * (nis > 0)
        return lambda weight: rr_nps + weight * rr_nis * gmask
    if mode == "linear":
        return lambda weight: nps + weight * nis * gmask
    raise ValueError("mode debe ser 'linear' o 'rrf'")


def _validate_alignment(nps: np.ndarray, nis: np.ndarray) -> None:
    if nps.ndim != 2 or nis.ndim != 2 or nps.shape != nis.shape:
        raise ValueError(f"NPS/NIS deben ser matrices alineadas [N,C]; recibido {nps.shape} y {nis.shape}.")
