"""Fase CBR (Módulos 2-3): subconjunto piloto, destilación y Case-Based Reasoning.

NOTA DE ARQUITECTURA:
  El entrenamiento del clasificador supervisado DeBERTa (train.py / predict.py)
  ha sido descartado en favor de la arquitectura de Case-Based Reasoning real
  (CBR de Recuperación Zero-Shot e Híbrido denso+léxico).
  Para la nueva arquitectura, utilizar `triager_omega.retrieval`.
"""

from triager_omega.retrieval import CbrRetriever, CbrConfig, vote_scores, bm25_sims
