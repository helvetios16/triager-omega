from __future__ import annotations

from typing import Sequence
import numpy as np


def rrf_fuse(dense: np.ndarray, lexical: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion (RRF) para combinar similitudes densas y léxicas.

    RRF(doc) = 1 / (k + rank_denso) + 1 / (k + rank_léxico).
    Opera sobre el orden posicional en lugar de la magnitud absoluta, siendo robusto
    a diferencias de escala entre similitud coseno [-1..1] y scores BM25 [0..∞).
    """
    fused = np.zeros_like(dense, dtype=np.float32)
    for mat in (dense, lexical):
        ranks = np.empty_like(mat, dtype=np.int64)
        order = np.argsort(-mat, axis=1)
        rows = np.arange(mat.shape[0])[:, None]
        ranks[rows, order] = np.arange(mat.shape[1])[None, :]
        fused += 1.0 / (k + ranks + 1.0)
    return fused


def rerank_cross_encoder(
    sims: np.ndarray,
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 50,
    device: str = "cpu",
    batch_size: int = 32,
    char_cap: int = 2000,
) -> np.ndarray:
    """Reordena los Top-N candidatos densos usando un Cross-Encoder para máxima precisión.

    El Bi-Encoder (ej. MPNet) provee alta cobertura/recall; el Cross-Encoder analiza el par
    (query, candidato) conjuntamente generando un score de interacción atenta directa.
    """
    from sentence_transformers import CrossEncoder

    ce_device = "cpu" if device == "mps" else device
    ce = CrossEncoder(model_name, device=ce_device, max_length=256)

    tr_cap = [str(t)[:char_cap] for t in train_texts]
    te_cap = [str(t)[:char_cap] for t in test_texts]

    top_n = min(top_n, sims.shape[1])
    cand = np.argpartition(-sims, kth=top_n - 1, axis=1)[:, :top_n]
    out = np.zeros_like(sims, dtype=np.float32)

    pairs: list[tuple[str, str]] = []
    locs: list[tuple[int, int]] = []

    for i in range(sims.shape[0]):
        for j in cand[i]:
            pairs.append((te_cap[i], tr_cap[j]))
            locs.append((i, j))

    if pairs:
        ce_scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        sig = 1.0 / (1.0 + np.exp(-ce_scores))
        for (i, j), s in zip(locs, sig):
            out[i, j] = s

    return out
