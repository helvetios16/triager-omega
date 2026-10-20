from __future__ import annotations

import re
from typing import Sequence
import numpy as np


def bm25_sims(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    token_pattern: str = r"[A-Za-z0-9_.:#]+",
) -> np.ndarray:
    """Calcula la matriz de similitud léxica BM25 [N_test, N_train].

    Conserva identificadores técnicos (paquetes, códigos de error, stack traces)
    que la similitud densa semántica difumina y que resultan clave en la cola larga.
    """
    from rank_bm25 import BM25Okapi

    regex = re.compile(token_pattern)
    tokenize = lambda s: regex.findall(str(s).lower())

    corpus = [tokenize(t) for t in train_texts]
    bm25 = BM25Okapi(corpus)

    out = np.zeros((len(test_texts), len(train_texts)), dtype=np.float32)
    for i, q in enumerate(test_texts):
        out[i] = bm25.get_scores(tokenize(q))

    return out
