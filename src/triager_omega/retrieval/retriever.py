from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from triager_omega.config import Settings

from triager_omega.retrieval.bm25 import bm25_sims
from triager_omega.retrieval.encoder import build_label_encoder
from triager_omega.retrieval.fusion import rrf_fuse
from triager_omega.retrieval.voting import vote_scores


@dataclass
class CbrConfig:
    encoder_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = field(default_factory=lambda: Settings().torch_device)
    max_seq_length: int = 384
    batch_size: int = 64
    top_k: int = 20
    tau: float = 1.0
    vote_strategy: str = "idf"  # "sum", "mean", "max", "idf", "rank"
    rrf_k: int = 60


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
        self._is_fitted = False

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
        """Indexa la base de casos (bugs históricos resueltos).

        Los embeddings precalculados deben estar ya normalizados en L2 para que
        el producto punto de ``compute_dense_sims`` represente similitud coseno.
        No se normalizan ni alteran automáticamente.
        """
        if len(train_texts) != len(train_owners):
            raise ValueError(
                "train_texts y train_owners deben tener la misma longitud "
                f"(recibidos {len(train_texts)} y {len(train_owners)})."
            )
        self._is_fitted = False
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
            embeddings = np.asarray(precomputed_embeddings)
            if embeddings.ndim != 2:
                raise ValueError("precomputed_embeddings debe ser una matriz 2D.")
            if embeddings.shape[0] != len(self.train_texts):
                raise ValueError(
                    "precomputed_embeddings debe tener una fila por texto de entrenamiento "
                    f"(recibidas {embeddings.shape[0]} filas para {len(self.train_texts)} textos)."
                )
            self.train_embeddings = embeddings
        else:
            logger.info("Calculando embeddings densos para {} bugs de entrenamiento...", len(self.train_texts))
            self.train_embeddings = self.encoder.encode(
                self.train_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        self.train_embeddings = np.asarray(self.train_embeddings)
        if self.train_embeddings.ndim != 2:
            raise ValueError("Los embeddings de entrenamiento deben ser una matriz 2D.")
        if self.train_embeddings.shape[0] != len(self.train_texts):
            raise ValueError(
                "Los embeddings de entrenamiento deben tener una fila por texto "
                f"(recibidas {self.train_embeddings.shape[0]} filas para {len(self.train_texts)} textos)."
            )
        self._is_fitted = True
        return self

    def compute_dense_sims(
        self,
        test_texts: Sequence[str],
        test_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calcula la matriz de similitud coseno densa [N_test, N_train]."""
        self._require_fitted()

        if test_embeddings is None:
            test_embeddings = self.encoder.encode(
                [str(t) for t in test_texts],
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            test_embeddings = np.asarray(test_embeddings)

        if test_embeddings.ndim != 2:
            raise ValueError("test_embeddings debe ser una matriz 2D.")
        if test_embeddings.shape[0] != len(test_texts):
            raise ValueError(
                "test_embeddings debe tener una fila por texto de consulta "
                f"(recibidas {test_embeddings.shape[0]} filas para {len(test_texts)} textos)."
            )
        if test_embeddings.shape[1] != self.train_embeddings.shape[1]:
            raise ValueError(
                "Las dimensiones de embeddings train/query deben coincidir "
                f"({self.train_embeddings.shape[1]} != {test_embeddings.shape[1]})."
            )

        return test_embeddings @ self.train_embeddings.T

    def compute_lexical_sims(self, test_texts: Sequence[str]) -> np.ndarray:
        """Calcula la matriz de similitud léxica BM25 [N_test, N_train]."""
        self._require_fitted()
        return bm25_sims(self.train_texts, [str(t) for t in test_texts])

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("El recuperador debe ajustarse con fit() antes de puntuar.")

    def score(
        self,
        test_texts: Sequence[str],
        test_embeddings: np.ndarray | None = None,
        mode: str = "dense",  # "dense", "lexical", "hybrid"
        top_k: int | None = None,
        tau: float | None = None,
        vote_strategy: str | None = None,
        temporal_decay: np.ndarray | None = None,
        rrf_k: int | None = None,
    ) -> np.ndarray:
        """Calcula la matriz de recomendación NPS [N_test, C] por agregación de votos."""
        self._require_fitted()
        if mode == "dense":
            sims = self.compute_dense_sims(test_texts, test_embeddings)
        elif mode == "lexical":
            sims = self.compute_lexical_sims(test_texts)
        elif mode == "hybrid":
            dense_sims = self.compute_dense_sims(test_texts, test_embeddings)
            lexical_sims = self.compute_lexical_sims(test_texts)
            sims = rrf_fuse(dense_sims, lexical_sims, k=self.config.rrf_k if rrf_k is None else rrf_k)
        else:
            raise ValueError(f"Modo desconocido: '{mode}'. Opciones: 'dense', 'lexical', 'hybrid'.")

        return self.score_sims(
            sims,
            top_k=top_k,
            tau=tau,
            vote_strategy=vote_strategy,
            temporal_decay=temporal_decay,
        )

    def score_sims(
        self,
        sims: np.ndarray,
        top_k: int | None = None,
        tau: float | None = None,
        vote_strategy: str | None = None,
        temporal_decay: np.ndarray | None = None,
    ) -> np.ndarray:
        """Agrega votos sobre similitudes [N_query, N_train] ya calculadas.

        No calcula embeddings ni similitudes léxicas. Similitudes y decaimiento
        deben alinearse con los casos indexados durante ``fit``.
        """
        self._require_fitted()
        sims = np.asarray(sims)
        if sims.ndim != 2:
            raise ValueError("sims debe ser una matriz 2D [N_query, N_train].")
        if sims.shape[1] != len(self.train_texts):
            raise ValueError(
                "sims debe tener una columna por caso de entrenamiento "
                f"(recibidas {sims.shape[1]} columnas para {len(self.train_texts)} casos)."
            )
        if temporal_decay is not None and np.asarray(temporal_decay).shape != (len(self.train_texts),):
            raise ValueError(
                "temporal_decay debe tener un valor por caso de entrenamiento "
                f"(esperada forma ({len(self.train_texts)},))."
            )
        return vote_scores(
            sims=sims,
            train_cls=self.train_classes,
            num_classes=len(self.label_encoder),
            top_k=self.config.top_k if top_k is None else top_k,
            tau=self.config.tau if tau is None else tau,
            decay=temporal_decay,
            vote=self.config.vote_strategy if vote_strategy is None else vote_strategy,
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
        ranked_indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_n]

        recommendations = []
        for row in ranked_indices:
            recommendations.append([self.inv_label_encoder[idx] for idx in row])
        return recommendations
