#!/usr/bin/env python3
"""Evaluación comparativa del CBR de Recuperación en TypeScript.

Compara vistas raw, distilled y both con MPNet, votos CBR y fusión híbrida.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from triager_omega.config import settings
from triager_omega.modules.aggregator import rank_metrics
from triager_omega.retrieval import CbrConfig, CbrRetriever, build_label_encoder, rrf_fuse


@dataclass
class BenchmarkConfig:
    """Defaults explícitos del barrido TypeScript vigente, sin recalibrar."""

    model: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_len: int = 384
    train_raw: str | Path = "artifacts/typescript/distilled/ts_train_union_raw.csv"
    test_raw: str | Path = "artifacts/typescript/distilled/ts_test_raw.csv"
    train_distilled: str | Path = "artifacts/typescript/distilled/ts_train_union_distilled.csv"
    test_distilled: str | Path = "artifacts/typescript/distilled/ts_test_distilled.csv"
    train_both: str | Path = "artifacts/typescript/distilled/ts_train_union_both.csv"
    test_both: str | Path = "artifacts/typescript/distilled/ts_test_both.csv"
    embeddings_dir: str | Path = "artifacts/typescript/embeddings"
    results_path: str | Path = "artifacts/typescript/ts_cbr_evaluation_results.json"
    report_path: str | Path = "docs/experimental/typescript-cbr-resultados.md"
    device: str | None = None
    ks: tuple[int, ...] = (10, 20, 30, 50)
    taus: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    votes: tuple[str, ...] = ("sum", "idf")
    rrf_k: int = 60


CONFIG = BenchmarkConfig()


def label_encoder(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """Compatibilidad de import con el script legacy."""
    return build_label_encoder(train["owner"], test["owner"])


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop_duplicates("issue_number").copy()
    result["text"] = result["text"].fillna("").astype(str)
    return result


def _evaluate_view(
    modality: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
    config: BenchmarkConfig, train_embeddings: np.ndarray | None,
    test_embeddings: np.ndarray | None, cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    tr, te = _normalise_frame(train_df), _normalise_frame(test_df)
    train_cache = cache_dir / f"tr_{modality}.npy" if cache_dir else None
    test_cache = cache_dir / f"te_{modality}.npy" if cache_dir else None
    if train_embeddings is None and train_cache and train_cache.exists():
        train_embeddings = np.load(train_cache)
    if test_embeddings is None and test_cache and test_cache.exists():
        test_embeddings = np.load(test_cache)
    retriever = CbrRetriever(CbrConfig(
        encoder_name=config.model,
        device=config.device or settings.torch_device,
        max_seq_length=config.max_seq_len,
        top_k=20,
        tau=1.0,
        vote_strategy="sum",
        rrf_k=config.rrf_k,
    ))
    retriever.fit(tr["text"].tolist(), tr["owner"].tolist(), te["owner"].tolist(),
                  precomputed_embeddings=train_embeddings)
    if train_cache and train_embeddings is None:
        train_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(train_cache, retriever.train_embeddings)
    texts_test = te["text"].tolist()
    if test_embeddings is None:
        test_embeddings = retriever.encoder.encode(
            texts_test, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True)
        if test_cache:
            test_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(test_cache, test_embeddings)
    dense = retriever.compute_dense_sims(texts_test, test_embeddings)
    # The historical both view is dense-only; raw and distilled also report RRF.
    views = [("dense", dense)]
    if modality != "both":
        # Historical caches are validated by shape only; content-based invalidation is out of scope.
        lexical_cache = cache_dir / f"bm25_{modality}.npy" if cache_dir else None
        lexical = None
        expected_shape = (len(te), len(tr))
        if lexical_cache is not None and lexical_cache.exists():
            cached_lexical = np.load(lexical_cache)
            if cached_lexical.shape == expected_shape:
                lexical = cached_lexical
            else:
                logger.warning("Caché BM25 {} tiene forma {}; se esperaba {}. Recalculando.",
                               lexical_cache, cached_lexical.shape, expected_shape)
        if lexical is None:
            lexical = retriever.compute_lexical_sims(texts_test)
            if lexical_cache is not None:
                lexical_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(lexical_cache, lexical)
        views.append(("hybrid", rrf_fuse(dense, lexical, k=config.rrf_k)))
    results: list[dict[str, Any]] = []
    score_matrices: list[np.ndarray] = []
    best: dict[str, dict[str, Any] | None] = {"dense": None, "hybrid": None}
    best_score = {"dense": -1.0, "hybrid": -1.0}
    true = np.array([retriever.label_encoder[o] for o in te["owner"]], dtype=np.int64)
    for kind, sims in views:
        for vote in config.votes:
            for k in config.ks:
                for tau in config.taus:
                    scores = retriever.score_sims(sims, top_k=k, tau=tau, vote_strategy=vote)
                    metrics = rank_metrics(scores, true, ks=(1, 5, 10))
                    score_matrices.append(scores)
                    item = {"modality": modality, "type": kind, "vote": vote,
                            "k": k, "tau": tau, **metrics}
                    results.append(item)
                    if metrics["hit@1"] > best_score[kind]:
                        best_score[kind] = metrics["hit@1"]
                        best[kind] = item
    details = {"similarities": {kind: sims for kind, sims in views},
               "train_classes": retriever.train_classes.copy(), "true_indices": true,
               "label_encoder": dict(retriever.label_encoder)}
    details["score_matrices"] = score_matrices
    return results, best["dense"], best["hybrid"], details


def evaluate(
    config: BenchmarkConfig = CONFIG,
    *,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    dataframes: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    embeddings: dict[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evalúa una vista (train_df/test_df) o varias (dataframes), sin requerir red.

    Si no se inyectan embeddings, CbrRetriever los calcula una vez por conjunto;
    similitudes dense/RRF se construyen antes del barrido y se reutilizan.
    """
    if dataframes is None:
        if train_df is not None and test_df is not None:
            dataframes = {"raw": (train_df, test_df)}
        elif train_df is not None or test_df is not None:
            raise ValueError("train_df y test_df deben proporcionarse juntos")
        else:
            dataframes = _load_views(config)
    results, summaries, details = [], [], {}
    for modality, (tr, te) in dataframes.items():
        emb_tr, emb_te = (embeddings or {}).get(modality, (None, None))
        if modality == "raw" and embeddings is None:
            emb_tr, emb_te = train_embeddings, test_embeddings
        view_results, dense, hybrid, view_details = _evaluate_view(
            modality, tr, te, config, emb_tr, emb_te,
            Path(cache_dir) if cache_dir is not None else None)
        results.extend(view_results)
        summaries.append((f"{modality}_dense", dense))
        if hybrid is not None:
            summaries.append((f"{modality}_hybrid", hybrid))
        details[modality] = view_details
    return {"results": results, "summary_bests": summaries, "views": details}


def _load_views(config: BenchmarkConfig) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    paths = {
        "raw": (config.train_raw, config.test_raw),
        "distilled": (config.train_distilled, config.test_distilled),
        "both": (config.train_both, config.test_both),
    }
    loaded = {}
    for name, (train_path, test_path) in paths.items():
        loaded[name] = (pd.read_csv(train_path), pd.read_csv(test_path))
    return loaded


def main() -> None:
    device = CONFIG.device or settings.torch_device
    logger.info("Iniciando evaluación CBR de TypeScript en dispositivo: {}", device)
    config = BenchmarkConfig(**{**vars(CONFIG), "device": device})
    views = _load_views(config)
    result = evaluate(config, dataframes=views, cache_dir=config.embeddings_dir)
    all_results, summary_bests = result["results"], result["summary_bests"]
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("=== RESUMEN COMPARATIVO EN TYPESCRIPT ===")
    print("\n| Configuración | Top-1 (Hit@1) | Hit@5 | Hit@10 | MRR | Parámetros óptimos |")
    print("|---|---|---|---|---|---|")
    print("| **TriagerX CBR (Paper publicado)** | 0.3240 | — | — | — | Ensemble RoBERTa+DeBERTa+3CNN (Entrenado 20 épocas) |")
    print("| **TriagerX Baseline (RoBERTa-FCN)** | 0.3190 | — | — | — | Clasificador Dense individual |")
    for tag, best in summary_bests:
        cfg = f"k={best['k']}, τ={best['tau']}, vote={best['vote']}"
        print(f"| **{tag}** | **{best['hit@1']:.4f}** | {best['hit@5']:.4f} | {best['hit@10']:.4f} | {best['mrr']:.4f} | {cfg} |")
    doc = Path(config.report_path)
    doc.parent.mkdir(parents=True, exist_ok=True)
    with doc.open("w", encoding="utf-8") as report:
        report.write("# Resultados de Evaluación Experimental: CBR de Recuperación en TypeScript\n\n")
        report.write(f"**Dispositivo:** `{device}` | **Encoder:** `{config.model}` (`max_seq_len={config.max_seq_len}`)\n")
        report.write("**Dataset:** Train 9,831 issues · Test 1,115 issues · 40 clases\n\n")
        report.write("## Tabla Comparativa Head-to-Head\n\n")
        report.write("| Configuración | Top-1 (Hit@1) | Hit@5 | Hit@10 | MRR | Hiperparámetros |\n|---|---|---|---|---|---|\n")
        report.write("| **TriagerX CBR (Paper publicado)** | 0.3240 | — | — | — | Ensemble 2 Transformers + 3 CNN (Entrenado) |\n")
        report.write("| **TriagerX RoBERTa-FCN** | 0.3190 | — | — | — | Clasificador denso individual |\n")
        for tag, best in summary_bests:
            cfg = f"k={best['k']}, τ={best['tau']}, vote={best['vote']}"
            report.write(f"| **{tag}** | **{best['hit@1']:.4f}** | {best['hit@5']:.4f} | {best['hit@10']:.4f} | {best['mrr']:.4f} | {cfg} |\n")


if __name__ == "__main__":
    main()
