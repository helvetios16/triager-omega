"""Pruebas de caracterización para triager-omega.

Fijan el comportamiento matemático exacto y actual de las primitivas:
1. Normalización de scores (normalize_rows: minmax y softmax).
2. Métricas de ranking y desempates deterministas (rank_metrics).
3. Agregación de votos kNN (vote_scores: sum, rank, max, mean, idf, decay).
4. Fusión por rangos recíprocos (rrf_fuse) y similitud léxica (bm25_sims).
5. Lógica de decaimiento temporal y causalidad anti-fuga en IBR.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from triager_omega.modules.aggregator import normalize_rows, rank_metrics
from triager_omega.retrieval.bm25 import bm25_sims
from triager_omega.retrieval.fusion import rrf_fuse
from triager_omega.retrieval.voting import vote_scores


# ============================================================================
# 1. Normalización de filas (normalize_rows)
# ============================================================================
def test_normalize_rows_minmax_standard():
    logits = np.array([
        [10.0, 20.0, 30.0],
        [0.0, 5.0, 10.0],
    ], dtype=np.float32)

    norm = normalize_rows(logits, how="minmax")

    assert norm.shape == logits.shape
    assert np.allclose(norm[0], [0.0, 0.5, 1.0])
    assert np.allclose(norm[1], [0.0, 0.5, 1.0])


def test_normalize_rows_minmax_zero_range():
    # Cuando todos los elementos de la fila son idénticos, mx - mn == 0
    # La implementación actual usa: rng = np.where((mx - mn) == 0, 1.0, mx - mn)
    # Por lo que (logits - mn) / 1.0 da exactamente 0.0
    logits = np.array([
        [5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    norm = normalize_rows(logits, how="minmax")
    assert np.allclose(norm, 0.0)


def test_normalize_rows_minmax_fractional_range():
    # Verificación crítica: cuando mx - mn < 1.0 (p.ej. 0.5),
    # el divisor DEBE ser 0.5 y NO 1.0
    logits = np.array([
        [0.1, 0.35, 0.6],  # min=0.1, max=0.6, diff=0.5
    ], dtype=np.float32)

    norm = normalize_rows(logits, how="minmax")
    assert np.allclose(norm[0], [0.0, 0.5, 1.0])


def test_normalize_rows_softmax():
    logits = np.array([
        [1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0],  # prueba de estabilidad numérica
    ], dtype=np.float32)

    probs = normalize_rows(logits, how="softmax")
    assert np.allclose(probs.sum(axis=1), [1.0, 1.0])
    # Ambas filas tienen las mismas diferencias relativas, luego deben tener idénticas probs
    assert np.allclose(probs[0], probs[1])


# ============================================================================
# 2. Métricas de ranking y estabilidad de desempates (rank_metrics)
# ============================================================================
def test_rank_metrics_exact_values():
    # 3 consultas, 4 clases (0, 1, 2, 3)
    scores = np.array([
        [0.9, 0.1, 0.0, 0.0],  # orden: [0, 1, 2, 3] -> clase 0 en rango 0
        [0.1, 0.8, 0.0, 0.0],  # orden: [1, 0, 2, 3] -> clase 0 en rango 1
        [0.1, 0.2, 0.3, 0.9],  # orden: [3, 2, 1, 0] -> clase 3 en rango 0
    ], dtype=np.float32)

    true_idx = np.array([0, 0, 3], dtype=np.int64)
    # Rangos (0-based): q0 -> 0; q1 -> 1; q2 -> 0.
    # Hit@1: 2/3 = 0.6667
    # Hit@3: 3/3 = 1.0000
    # MRR: (1/1 + 1/2 + 1/1) / 3 = 2.5 / 3 = 0.8333...

    m = rank_metrics(scores, true_idx, ks=(1, 3))
    assert pytest.approx(m["hit@1"], 1e-4) == 2.0 / 3.0
    assert pytest.approx(m["hit@3"], 1e-4) == 1.0
    assert pytest.approx(m["mrr"], 1e-4) == 2.5 / 3.0


def test_rank_metrics_stable_tie_breaking():
    # Cuando hay empate exacto de scores, kind="stable" en argsort garantiza
    # que se preserve el orden de columnas original: índice 0 gana a 1, 1 a 2.
    scores = np.array([
        [0.5, 0.5, 0.1],  # clase 0 y 1 empatadas a 0.5
    ], dtype=np.float32)

    # Si true_idx es 0, gana Top-1 (rango 0)
    m0 = rank_metrics(scores, np.array([0]), ks=(1,))
    assert m0["hit@1"] == 1.0

    # Si true_idx es 1, queda en rango 1 (no es Top-1)
    m1 = rank_metrics(scores, np.array([1]), ks=(1,))
    assert m1["hit@1"] == 0.0
    assert m1["mrr"] == 0.5  # 1 / (1 + 1)


# ============================================================================
# 3. Agregación de votos kNN (vote_scores)
# ============================================================================
def test_vote_scores_sum_strategy():
    # 1 consulta, 4 bugs de train pertenecientes a 3 clases
    # train_cls: bug 0 -> dev 0; bug 1 -> dev 1; bug 2 -> dev 0; bug 3 -> dev 2
    sims = np.array([[0.8, 0.6, 0.4, 0.2]], dtype=np.float32)
    train_cls = np.array([0, 1, 0, 2], dtype=np.int64)
    num_classes = 3

    # top_k=3 -> toma bugs 0 (sim=0.8), 1 (sim=0.6), 2 (sim=0.4)
    # dev 0 recibe 0.8^1 + 0.4^1 = 1.2
    # dev 1 recibe 0.6^1 = 0.6
    # dev 2 no entra en top_k=3 -> 0.0
    scores = vote_scores(sims, train_cls, num_classes=num_classes, top_k=3, tau=1.0, vote="sum")
    assert np.allclose(scores[0], [1.2, 0.6, 0.0])


def test_vote_scores_tau_exponent():
    sims = np.array([[0.8, 0.6]], dtype=np.float32)
    train_cls = np.array([0, 1], dtype=np.int64)
    scores = vote_scores(sims, train_cls, num_classes=2, top_k=2, tau=2.0, vote="sum")
    assert np.allclose(scores[0], [0.8**2, 0.6**2])


def test_vote_scores_rank_strategy():
    sims = np.array([[0.9, 0.7, 0.5]], dtype=np.float32)
    train_cls = np.array([0, 1, 2], dtype=np.int64)
    # vote="rank": ignora sim y asigna 1/(rango + 1)
    # orden: rango 0 -> 1/1 = 1.0; rango 1 -> 1/2 = 0.5; rango 2 -> 1/3
    scores = vote_scores(sims, train_cls, num_classes=3, top_k=3, vote="rank")
    assert np.allclose(scores[0], [1.0, 0.5, 1.0 / 3.0])


def test_vote_scores_max_strategy():
    sims = np.array([[0.5, 0.9]], dtype=np.float32)
    train_cls = np.array([0, 0], dtype=np.int64)  # ambos votos van a dev 0
    scores = vote_scores(sims, train_cls, num_classes=1, top_k=2, tau=1.0, vote="max")
    assert np.allclose(scores[0, 0], 0.9)


def test_vote_scores_mean_strategy():
    sims = np.array([[0.8, 0.4]], dtype=np.float32)
    train_cls = np.array([0, 0], dtype=np.int64)  # ambos votos van a dev 0
    scores = vote_scores(sims, train_cls, num_classes=1, top_k=2, tau=1.0, vote="mean")
    assert np.allclose(scores[0, 0], (0.8 + 0.4) / 2.0)


def test_vote_scores_temporal_decay():
    sims = np.array([[0.8, 0.6]], dtype=np.float32)
    train_cls = np.array([0, 1], dtype=np.int64)
    decay = np.array([0.25, 1.0], dtype=np.float32)

    scores = vote_scores(sims, train_cls, num_classes=2, top_k=2, tau=1.0,
                         decay=decay, vote="sum")
    assert np.allclose(scores[0], [0.8 * 0.25, 0.6])


def test_vote_scores_idf_strategy():
    sims = np.array([[1.0, 1.0]], dtype=np.float32)
    train_cls = np.array([0, 1], dtype=np.int64)
    # dev 0 tiene freq=9; dev 1 tiene freq=1; total=10
    dev_freq = np.array([9, 1], dtype=np.int64)
    scores = vote_scores(sims, train_cls, num_classes=2, top_k=2, tau=1.0, vote="idf", dev_freq=dev_freq)

    idf_0 = np.log(10.0 / 9.0)
    idf_1 = np.log(10.0 / 1.0)
    assert np.allclose(scores[0], [idf_0, idf_1])


# ============================================================================
# 4. Fusión recíproca (rrf_fuse) y similitud léxica (bm25_sims)
# ============================================================================
def test_rrf_fuse_exact():
    # 1 consulta, 2 documentos
    # denso: doc 0 (0.9), doc 1 (0.1) -> rangos [0, 1]
    # léxico: doc 0 (1.0), doc 1 (10.0) -> rangos [1, 0]
    dense = np.array([[0.9, 0.1]], dtype=np.float32)
    lexical = np.array([[1.0, 10.0]], dtype=np.float32)
    k = 60

    fused = rrf_fuse(dense, lexical, k=k)

    # doc 0: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
    # doc 1: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
    assert pytest.approx(fused[0, 0], 1e-6) == (1.0 / 61.0 + 1.0 / 62.0)
    assert pytest.approx(fused[0, 1], 1e-6) == (1.0 / 61.0 + 1.0 / 62.0)


def test_bm25_sims_token_retention():
    # Comprueba con recuperación real que los identificadores se conservan como
    # tokens completos, incluidos los símbolos técnicos permitidos por el patrón.
    train_texts = [
        "Exception in Foo$Bar class",
        "Crash in java.util.Map",
        "Failure at module#method:42",
        "Memory leak in unrelated subsystem",
        "Foo Bar java util Map module method 42",
    ]
    test_texts = ["Foo$Bar", "java.util.Map", "module#method:42"]

    sims = bm25_sims(train_texts, test_texts)
    assert sims.shape == (3, 5)
    # Los fragmentos separados no deben coincidir con identificadores completos.
    # Esta aserción falla si se eliminan los símbolos del patrón de tokenización.
    np.testing.assert_array_equal(sims[:, 4], np.zeros(3))
    assert sims[0, 0] > 0 and sims[0, 0] == sims[0].max()
    assert sims[1, 1] > 0 and sims[1, 1] == sims[1].max()
    assert sims[2, 2] > 0 and sims[2, 2] == sims[2].max()


# ============================================================================
# 6. Caracterización de CbrRetriever con embeddings precomputados
# ============================================================================
def test_cbr_retriever_fit_and_score_precomputed():
    from triager_omega.retrieval.retriever import CbrConfig, CbrRetriever

    # Instanciamos con device="cpu" para evitar dependencias de hardware
    cfg = CbrConfig(device="cpu", top_k=2, tau=1.0, vote_strategy="sum")
    retriever = CbrRetriever(config=cfg)

    # 3 bugs de entrenamiento, 2 desarrolladores:
    # doc 0 (alice) -> ortogonal x: [1.0, 0.0]
    # doc 1 (bob)   -> ortogonal y: [0.0, 1.0]
    # doc 2 (alice) -> ortogonal x: [1.0, 0.0]
    train_texts = ["compiler crash", "memory leak", "syntax parser error"]
    train_owners = ["alice", "bob", "alice"]
    train_embs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float32)

    retriever.fit(
        train_texts=train_texts,
        train_owners=train_owners,
        precomputed_embeddings=train_embs,
    )

    # Comprobar label encoder ordenado lexicográficamente
    assert retriever.label_encoder == {"alice": 0, "bob": 1}
    assert np.array_equal(retriever.train_classes, [0, 1, 0])
    assert np.array_equal(retriever.dev_frequencies, [2, 1])

    # Consulta alineada con x: debe dar similitud [1.0, 0.0, 1.0]
    test_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = retriever.score(
        test_texts=["new compiler crash"],
        test_embeddings=test_embs,
        mode="dense",
        top_k=3,
        tau=1.0,
        vote_strategy="sum",
    )

    # alice (clase 0) recibe 1.0 + 1.0 = 2.0; bob (clase 1) recibe 0.0
    assert scores.shape == (1, 2)
    assert np.allclose(scores[0], [2.0, 0.0])

def _synthetic_ibr():
    import torch

    from triager_omega.config import Settings
    from triager_omega.modules.ibr import InteractionRecommender

    cfg = Settings(ibr_tau=0.5, ibr_top_k_retrieve=4, ibr_lambda=0.01,
                   ip_contribution=1.5, ip_assignment=0.5, ip_discussion=0.1)
    ibr = InteractionRecommender(cfg=cfg)
    ibr._active = {101, 102, 103}
    ibr._train_bug_ids = np.array([1, 2, 3, 4])
    # La query y los primeros tres casos tienen coseno 1; el cuarto es ortogonal.
    ibr._train_embeddings = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    return ibr, torch.tensor([[1., 0.]])


def test_ibr_score_uses_real_decay_causality_and_active_candidates():
    ibr, query = _synthetic_ibr()
    t_now = pd.Timestamp("2023-01-15 00:00:00", tz="UTC")
    past_10d = pd.Timestamp("2023-01-05 00:00:00", tz="UTC")
    past_1d = pd.Timestamp("2023-01-14 00:00:00", tz="UTC")
    future = pd.Timestamp("2023-01-20 00:00:00", tz="UTC")
    ibr._interactions = {
        1: [(101, "assignment", past_10d)],
        2: [(102, "assignment", past_1d)],
        3: [(103, "assignment", t_now), (103, "assignment", future),
            (999, "assignment", past_1d)],
        4: [(101, "assignment", past_1d)],  # no alcanza el umbral de similitud
    }

    # _score filtra ts >= t_now (incluida igualdad), candidatos no activos,
    # casos bajo umbral y decae usando días enteros transcurridos.
    assert ibr._decay(past_10d, t_now) == pytest.approx(math.exp(-0.1), abs=1e-7)
    assert ibr._decay(t_now, t_now) == 1.0
    assert ibr._decay(future, t_now) == 0.0
    nis = ibr._score(query, t_now)
    # 101 aporta 0.5*exp(-0.1); 102 aporta 0.5*exp(-0.01),
    # y la normalización divide ambas señales por el máximo.
    expected_ratio = math.exp(-0.1) / math.exp(-0.01)
    assert nis == pytest.approx({101: expected_ratio, 102: 1.0, 103: 0.0}, abs=1e-6)


def test_ibr_score_with_none_t_now_uses_undecayed_interactions():
    ibr, query = _synthetic_ibr()
    past = pd.Timestamp("2023-01-05 00:00:00", tz="UTC")
    future = pd.Timestamp("2023-01-20 00:00:00", tz="UTC")
    ibr._interactions = {
        1: [(101, "assignment", past)],
        2: [(102, "assignment", future)],
    }

    # Con None no hay corte temporal ni decaimiento: ambas señales valen 0.5,
    # por tanto ambas se normalizan a uno. No se fuerza NIS cero.
    assert ibr._decay(past, None) == 1.0
    assert ibr._score(query, None) == pytest.approx({101: 1.0, 102: 1.0, 103: 0.0})


def test_ibr_nat_timestamp_behavior_is_nan():
    ibr, query = _synthetic_ibr()
    nat = pd.NaT
    ibr._interactions = {1: [(101, "assignment", nat)]}

    # El comportamiento actual de pandas/IBR no trata NaT como ausencia limpia:
    # la resta produce NaN y el score/normalización propagan NaN.
    score = ibr._decay(nat, pd.Timestamp("2023-01-15", tz="UTC"))
    nis = ibr._score(query, pd.Timestamp("2023-01-15", tz="UTC"))
    assert math.isnan(score)
    assert math.isnan(nis[101])
    assert nis[102] == 0.0 and nis[103] == 0.0


def test_ibr_nat_query_time_behavior_is_nan():
    ibr, query = _synthetic_ibr()
    past = pd.Timestamp("2023-01-05", tz="UTC")
    ibr._interactions = {1: [(101, "assignment", past)]}

    # NaT como fecha de consulta tampoco activa la rama especial de None.
    # Caracterizamos la propagación actual, sin introducir la política futura.
    assert math.isnan(ibr._decay(past, pd.NaT))
    nis = ibr._score(query, pd.NaT)
    assert math.isnan(nis[101])
    assert nis[102] == 0.0 and nis[103] == 0.0


def test_ibr_normalize_nis_all_positive_and_no_signal():
    ibr, _ = _synthetic_ibr()
    normalized = ibr._normalize_nis({101: 2.0, 102: 0.5, 103: 1.0})
    assert normalized == pytest.approx({101: 1.0, 102: 0.25, 103: 0.5})
    assert ibr._normalize_nis({}) == {101: 0.0, 102: 0.0, 103: 0.0}
