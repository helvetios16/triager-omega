from __future__ import annotations

import numpy as np


def vote_scores(
    sims: np.ndarray,
    train_cls: np.ndarray,
    num_classes: int,
    top_k: int,
    tau: float = 1.0,
    decay: np.ndarray | None = None,
    vote: str = "idf",
    dev_freq: np.ndarray | None = None,
) -> np.ndarray:
    """Calcula la matriz de scores [N_test, C] por agregación kNN ponderada por similitud^τ (· decay).

    Parámetros:
      sims: matriz de similitudes [N_test, N_train].
      train_cls: array [N_train] con el índice de clase del desarrollador que resolvió cada bug.
      num_classes: C total de desarrolladores candidatos.
      top_k: cantidad de vecinos más cercanos a considerar.
      tau: exponente de agudeza para la similitud (sim^tau).
      decay: factor opcional de decaimiento temporal [N_train].
      vote: estrategia de agregación para mitigar el sesgo de desarrolladores hiperfrecuentes:
        - "sum"  : suma directa Σ w (favorece a desarrolladores con muchos bugs históricos).
        - "mean" : promedio Σ w / n_vecinos_del_dev en top_k.
        - "max"  : toma la similitud del vecino más cercano ignorando la frecuencia.
        - "idf"  : ponderación inversa por frecuencia global Σ w · log(N_train / freq(dev)).
        - "rank" : peso recíproco 1 / (rango + 1) en lugar de sim^tau.
      dev_freq: array opcional [C] con las frecuencias globales por clase (para vote="idf").
    """
    n_test = sims.shape[0]
    scores = np.zeros((n_test, num_classes), dtype=np.float32)
    counts = np.zeros((n_test, num_classes), dtype=np.float32)

    weights = np.clip(sims, 0.0, None) ** tau
    if decay is not None:
        weights = weights * decay[None, :]

    k = min(top_k, sims.shape[1])
    # Partición de los top-k vecinos por fila
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

    for i in range(n_test):
        cols = part[i]
        order = cols[np.argsort(-sims[i, cols])]
        w_i = weights[i, order]

        if vote == "rank":
            w_i = 1.0 / (np.arange(len(order)) + 1.0)

        classes = train_cls[order]
        if vote == "max":
            np.maximum.at(scores[i], classes, w_i)
        else:
            np.add.at(scores[i], classes, w_i)
            np.add.at(counts[i], classes, 1.0)

    if vote == "mean":
        scores = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)
    elif vote == "idf":
        if dev_freq is None:
            dev_freq = np.bincount(train_cls, minlength=num_classes)
        idf = np.log(dev_freq.sum() / np.clip(dev_freq, 1, None)).astype(np.float32)
        scores = scores * idf[None, :]

    return scores
