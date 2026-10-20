from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from triager_omega.retrieval.bm25 import bm25_sims
from triager_omega.retrieval.encoder import build_label_encoder
from triager_omega.retrieval.fusion import rerank_cross_encoder, rrf_fuse
from triager_omega.retrieval.voting import vote_scores


@dataclass
class CbrConfig:
    encoder_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "mps"
    max_seq_length: int = 384
    batch_size: int = 64
    top_k: int = 20
    tau: float = 1.0
    vote_strategy: str = "idf"  # "sum", "mean", "max", "idf", "rank"


class CbrRetriever:
    """Case-Based Reasoning (CBR) real basado en recuperación semántica zero-shot y voto kNN.

    Implementa el pipeline que superó al estado del arte (TriagerX) en los benchmarks de
    OpenJ9 y TypeScript:
      1. Indexación densa con Bi-Encoder (MPNet) o léxica (BM25) de bugs históricos resueltos.
      2. Recuperación de los Top-K bugs análogos para cada nuevo reporte.
      3. Fusión híbrida opcional (Dense + BM25 mediante Reciprocal Rank Fusion).
      4. Ponderación y agregación de votos por desarrollador mitigando el sesgo de cola larga.
    """

    def __init__(self, config: CbrConfig | None = None):
        self.config = config or CbrConfig()
        self._encoder: SentenceTransformer | None = None
        self.label_encoder: dict[str, int] = {}
        self.inv_label_encoder: dict[int, str] = {}
        self.train_texts: list[str] = []
        self.train_classes: np.ndarray = np.array([])
        self.train_embeddings: np.ndarray | None = None
        self.dev_frequencies: np.ndarray | None = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("Cargando SentenceTransformer '{}' en dispositivo '{}'...",
                        self.config.encoder_name, self.config.device)
            self._encoder = SentenceTransformer(
                self.config.encoder_name,
                device=self.config.device,
            )
            self._encoder.max_seq_length = self.config.max_seq_length
        return self._encoder

    def fit(
        self,
        train_texts: Sequence[str],
        train_owners: Sequence[str],
        test_owners: Sequence[str] | None = None,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> CbrRetriever:
        """Indexa la base de casos (bugs históricos resueltos)."""
        self.train_texts = [str(t) for t in train_texts]
        self.label_encoder = build_label_encoder(train_owners, test_owners)
        self.inv_label_encoder = {idx: name for name, idx in self.label_encoder.items()}

        self.train_classes = np.array(
            [self.label_encoder[o] for o in train_owners],
            dtype=np.int64,
        )
        self.dev_frequencies = np.bincount(
            self.train_classes,
            minlength=len(self.label_encoder),
        )

        if precomputed_embeddings is not None:
            self.train_embeddings = precomputed_embeddings
        else:
            logger.info("Calculando embeddings densos para {} bugs de entrenamiento...", len(self.train_texts))
            self.train_embeddings = self.encoder.encode(
                self.train_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return self

    def compute_dense_sims(
        self,
        test_texts: Sequence[str],
        test_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calcula la matriz de similitud coseno densa [N_test, N_train]."""
        if self.train_embeddings is None:
            raise RuntimeError("El recuperador no ha sido ajustado (fit) previamente.")

        if test_embeddings is None:
            test_embeddings = self.encoder.encode(
                [str(t) for t in test_texts],
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return test_embeddings @ self.train_embeddings.T

    def compute_lexical_sims(self, test_texts: Sequence[str]) -> np.ndarray:
        """Calcula la matriz de similitud léxica BM25 [N_test, N_train]."""
        return bm25_sims(self.train_texts, [str(t) for t in test_texts])

    def score(
        self,
        test_texts: Sequence[str],
        test_embeddings: np.ndarray | None = None,
        mode: str = "dense",  # "dense", "lexical", "hybrid"
        top_k: int | None = None,
        tau: float | None = None,
        vote_strategy: str | None = None,
        temporal_decay: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calcula la matriz de recomendación NPS [N_test, C] por agregación de votos."""
        k = top_k or self.config.top_k
        t = tau if tau is not None else self.config.tau
        v = vote_strategy or self.config.vote_strategy

        if mode == "dense":
            sims = self.compute_dense_sims(test_texts, test_embeddings)
        elif mode == "lexical":
            sims = self.compute_lexical_sims(test_texts)
        elif mode == "hybrid":
            dense_sims = self.compute_dense_sims(test_texts, test_embeddings)
            lexical_sims = self.compute_lexical_sims(test_texts)
            sims = rrf_fuse(dense_sims, lexical_sims)
        else:
            raise ValueError(f"Modo desconocido: '{mode}'. Opciones: 'dense', 'lexical', 'hybrid'.")

        return vote_scores(
            sims=sims,
            train_cls=self.train_classes,
            num_classes=len(self.label_encoder),
            top_k=k,
            tau=t,
            decay=temporal_decay,
            vote=v,
            dev_freq=self.dev_frequencies,
        )

    def rank(
        self,
        test_texts: Sequence[str],
        top_n: int = 5,
        **score_kwargs,
    ) -> list[list[str]]:
        """Devuelve la lista ordenada de Top-N desarrolladores recomendados por reporte."""
        scores = self.score(test_texts, **score_kwargs)
        ranked_indices = np.argsort(-scores, axis=1)[:, :top_n]

        recommendations = []
        for row in ranked_indices:
            recommendations.append([self.inv_label_encoder[idx] for idx in row])
        return recommendations
