"""CBR de recuperación OpenJ9 mediante el ``CbrRetriever`` compartido.

La interfaz primaria es ``evaluate(BenchmarkConfig(...))``. Los defaults
reproducen el script legacy (MPNet, maxlen=256, k=20, tau=2, voto sum).
Fine-tuning y reranking se conservan como modos opcionales de compatibilidad.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from triager_omega.config import settings
from triager_omega.modules.aggregator import rank_metrics
from triager_omega.retrieval import (
    CbrConfig,
    CbrRetriever,
    build_label_encoder,
    rerank_cross_encoder,
    vote_scores,
)

# Import compatibility: eval_openj9_full.py imports both names from this module.
def label_encoder(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    return build_label_encoder(train["owner"], test["owner"])


@dataclass
class BenchmarkConfig:
    """Parámetros explícitos del benchmark OpenJ9; defaults legacy sin recalibrar."""

    model: str = "sentence-transformers/all-mpnet-base-v2"
    train_csv: str | Path | None = None
    test_csv: str | Path | None = None
    top_k: int = 20
    tau: float = 2.0
    decay: bool = False
    lam: float = 0.3
    meta: str | Path | None = None
    finetune: bool = False
    loss: str = "mnrl"
    epochs: int = 1
    ft_batch: int = 16
    lr: float = 2e-5
    max_seq_len: int = 256
    sweep: bool = False
    vote: str = "sum"
    vote_sweep: bool = False
    hybrid: bool = False
    rrf_k: int = 60
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_n: int = 50
    rerank_batch: int = 32
    cpu: bool = False


# Editar esta instancia configura la ejecución normal y los defaults del wrapper legacy.
CONFIG = BenchmarkConfig()


def _load_data(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(Path(config.train_csv) if config.train_csv else settings.openj9_train_csv)
    test = pd.read_csv(Path(config.test_csv) if config.test_csv else settings.openj9_test_csv)
    train = train.drop_duplicates("issue_number")
    test = test.drop_duplicates("issue_number")
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)
    return train, test


def _temporal_decay(train: pd.DataFrame, config: BenchmarkConfig,
                    meta_df: pd.DataFrame | None = None) -> np.ndarray:
    if meta_df is None:
        path = Path(config.meta) if config.meta else settings.openj9_issue_meta_path
        meta_df = pd.read_parquet(path)
    meta = meta_df.drop_duplicates("issue_number").set_index("issue_number")["created_at"]
    ages = pd.to_datetime(train["issue_number"].map(meta), utc=True, errors="coerce")
    reference = ages.max()
    years = (reference - ages).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    return np.exp(-config.lam * np.nan_to_num(years, nan=0.0)).astype(np.float32)


def evaluate(
    config: BenchmarkConfig,
    *,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    meta_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Ejecuta el benchmark y devuelve métricas y parámetros por punto evaluado.

    Embeddings opcionales permiten evaluar datos estáticos sin cargar modelos; si
    faltan, el encoder compartido los calcula una sola vez antes de todo barrido.
    """
    if train_df is None or test_df is None:
        loaded_train, loaded_test = _load_data(config)
        train_df = loaded_train if train_df is None else train_df.copy()
        test_df = loaded_test if test_df is None else test_df.copy()
    else:
        train_df, test_df = train_df.copy(), test_df.copy()
    for frame in (train_df, test_df):
        frame["text"] = frame["text"].fillna("").astype(str)

    retriever = CbrRetriever(CbrConfig(
        encoder_name=config.model,
        device="cpu" if config.cpu else settings.torch_device,
        max_seq_length=config.max_seq_len,
        top_k=config.top_k,
        tau=config.tau,
        vote_strategy=config.vote,
        rrf_k=config.rrf_k,
    ))
    if config.finetune:
        # Legacy ordering: train the encoder before computing train/query embeddings.
        labels = build_label_encoder(train_df["owner"], test_df["owner"])
        train_classes = np.array([labels[o] for o in train_df["owner"]], dtype=np.int64)
        _finetune(retriever.encoder, train_df, train_classes, len(labels), config,
                  "cpu" if config.cpu else settings.torch_device)
    retriever.fit(
        train_df["text"].tolist(), train_df["owner"].tolist(),
        test_df["owner"].tolist(), precomputed_embeddings=train_embeddings,
    )

    train_texts, test_texts = retriever.train_texts, test_df["text"].tolist()
    dense_sims = retriever.compute_dense_sims(test_texts, test_embeddings)
    sims = dense_sims
    if config.hybrid:
        logger.info("Híbrido BM25+MPNet (RRF k={})...", config.rrf_k)
        lexical_sims = retriever.compute_lexical_sims(test_texts)
        from triager_omega.retrieval import rrf_fuse
        sims = rrf_fuse(dense_sims, lexical_sims, k=config.rrf_k)
    if config.rerank:
        # Conserva la liberación de GPU legacy antes de cargar el cross-encoder.
        # No acceder a .encoder: cargaría un modelo innecesario con embeddings precalculados.
        import gc
        import torch

        if retriever._encoder is not None:
            retriever._encoder.to("cpu")
            retriever._encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = "cpu" if config.cpu else settings.torch_device
        sims = rerank_cross_encoder(
            sims, train_texts, test_texts, config.rerank_model,
            config.rerank_n, device, config.rerank_batch,
        )

    decay = _temporal_decay(train_df, config, meta_df) if config.decay else None
    true_idx = np.array([retriever.label_encoder[o] for o in test_df["owner"]], dtype=int)
    if config.sweep:
        points = [(k, tau, config.vote) for k in (5, 10, 15, 20, 30, 50)
                  for tau in (1.0, 2.0, 4.0)]
    elif config.vote_sweep:
        points = [(config.top_k, config.tau, vote)
                  for vote in ("sum", "mean", "max", "idf", "rank")]
    else:
        points = [(config.top_k, config.tau, config.vote)]

    evaluations = []
    for k, tau, vote in points:
        scores = retriever.score_sims(sims, top_k=k, tau=tau,
                                      vote_strategy=vote, temporal_decay=decay)
        metrics = rank_metrics(scores, true_idx)
        evaluations.append({"top_k": k, "tau": tau, "vote": vote,
                            "metrics": metrics, "scores": scores})
    logger.info("CBR-recuperación OpenJ9 | train={} test={} clases={}",
                len(train_df), len(test_df), len(retriever.label_encoder))
    return {"evaluations": evaluations, "label_encoder": dict(retriever.label_encoder),
            "similarities": sims, "train_classes": retriever.train_classes.copy(),
            "true_indices": true_idx, "temporal_decay": decay}


def main() -> None:
    """Wrapper CLI legacy; la configuración Python es la interfaz primaria."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--loss", choices=["mnrl", "triplet"], default="mnrl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--ft-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--vote", choices=["sum", "mean", "max", "idf", "rank"], default="sum")
    parser.add_argument("--vote-sweep", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--rerank-n", type=int, default=50)
    parser.add_argument("--rerank-batch", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(**vars(CONFIG))
    config = BenchmarkConfig(**vars(parser.parse_args()))
    result = evaluate(config)
    for item in result["evaluations"]:
        metrics = item["metrics"]
        logger.info("k={} tau={} vote={} Hit@1={:.4f} Hit@5={:.4f} Hit@10={:.4f} MRR={:.4f}",
                    item["top_k"], item["tau"], item["vote"], metrics["hit@1"],
                    metrics["hit@5"], metrics["hit@10"], metrics["mrr"])
    print(json.dumps([item["metrics"] for item in result["evaluations"]], indent=2))


def _finetune(model, train: pd.DataFrame, train_cls: np.ndarray,
              C: int, config: BenchmarkConfig, device: str) -> None:
    """Fine-tuning legacy sin cambios metodológicos; excepcional a la ruta compartida."""
    from collections import defaultdict
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    common = dict(
        output_dir=str(settings.openj9_dir / "cbr_retrieval_ft_trainer"),
        num_train_epochs=config.epochs, per_device_train_batch_size=config.ft_batch,
        learning_rate=config.lr, warmup_ratio=0.1, fp16=not config.cpu,
        dataloader_drop_last=True, logging_steps=20, save_strategy="no", report_to="none",
    )
    if config.loss == "triplet":
        logger.info("Fine-tuning TRIPLET (CUDA/fp16): {} ép, batch={}, lr={}...",
                    config.epochs, config.ft_batch, config.lr)
        ds = Dataset.from_dict({"sentence": train["text"].tolist(),
                                "label": [int(c) for c in train_cls]})
        loss = losses.BatchAllTripletLoss(model=model)
        targs = SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.GROUP_BY_LABEL, **common)
    else:
        texts = train["text"].tolist()
        by_dev: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(train_cls):
            by_dev[int(c)].append(i)
        rng = np.random.default_rng(settings.seed)
        anchors, positives = [], []
        for i, c in enumerate(train_cls):
            others = [j for j in by_dev[int(c)] if j != i]
            if others:
                anchors.append(texts[i])
                positives.append(texts[others[int(rng.integers(len(others)))]])
        logger.info("Fine-tuning MNRL (CUDA/fp16): {} pares mismo-dev, {} ép, batch={}, lr={}...",
                    len(anchors), config.epochs, config.ft_batch, config.lr)
        ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
        loss = losses.MultipleNegativesRankingLoss(model=model)
        targs = SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.NO_DUPLICATES, **common)
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    logger.success("Fine-tuning terminado.")


if __name__ == "__main__":
    main()
