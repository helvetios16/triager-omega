"""CBR de recuperación sobre el piloto Mozilla.

La interfaz Python primaria es ``evaluate(CONFIG, ...)``; el CLI conserva las
opciones del benchmark y usa ``CONFIG`` como fuente de sus defaults.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from triager_omega.config import settings
from triager_omega.modules.aggregator import linear_fuse, normalize_rows, rank_metrics
from triager_omega.modules.ibr import InteractionRecommender, combine_text, load_pilot_text
from triager_omega.retrieval import CbrConfig, CbrRetriever


@dataclass
class BenchmarkConfig:
    """Parámetros explícitos para el CBR Mozilla de recuperación."""

    model: str = "sentence-transformers/all-mpnet-base-v2"
    text_mode: str = "both"
    ibr_mode: str = "distilled"
    split: str = "test"
    top_k: int = 50
    tau: float = 2.0
    max_seq_len: int = 384
    sweep: bool = False
    fuse_ibr: bool = False
    cpu: bool = False
    nps_norm: str = "minmax"


# Editar esta instancia configura la ejecución normal y los defaults del CLI.
CONFIG = BenchmarkConfig()


def _ibr_matrix(test: pd.DataFrame, label_encoder: dict[str, int], ibr) -> np.ndarray:
    """Replica la ruta IBR del piloto y alinea NIS al label_encoder original."""
    q = ibr.encode_texts(combine_text(test, ibr.text_mode).tolist(), show_progress_bar=False)
    q = torch.as_tensor(q, dtype=torch.float32)
    nis = np.zeros((len(test), len(label_encoder)), dtype=np.float32)
    for i, t_now in enumerate(test["creation_time"]):
        scores = ibr.score_embedding(torch.as_tensor(q[i:i + 1]), t_now)
        for dev_str, idx in label_encoder.items():
            nis[i, idx] = scores.get(int(dev_str), 0.0)
    return nis


def evaluate(
    config: BenchmarkConfig = CONFIG,
    *,
    df: pd.DataFrame | None = None,
    label_encoder: dict[str, int] | None = None,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    nis: np.ndarray | None = None,
    ibr=None,
) -> dict[str, Any]:
    """Evalúa el CBR y opcionalmente su fusión IBR sin modificar artefactos.

    ``df``, ``label_encoder`` y embeddings permiten pruebas/datasets sintéticos.
    ``nis`` inyecta una matriz IBR ya calculada; alternativamente, ``ibr`` puede
    suministrar el recomendador con su interfaz privada actual del piloto.
    """
    if df is None or label_encoder is None:
        loaded_df, loaded_encoder = load_pilot_text(config.text_mode, settings)
        df = loaded_df if df is None else df.copy()
        label_encoder = loaded_encoder if label_encoder is None else label_encoder
    else:
        df = df.copy()
    label_encoder = dict(label_encoder)
    ordered_owners = [owner for owner, _ in sorted(label_encoder.items(), key=lambda item: item[1])]
    if sorted(label_encoder.values()) != list(range(len(label_encoder))):
        raise ValueError("label_encoder debe usar índices contiguos desde cero.")

    train = df[df["split"] == "train"].drop_duplicates("Bug Id").reset_index(drop=True)
    test = df[df["split"] == config.split].drop_duplicates("Bug Id").reset_index(drop=True)
    train_owners = train["contributor_id"].astype(str)
    test_owners = test["contributor_id"].astype(str)
    unknown = sorted((set(train_owners) | set(test_owners)) - set(label_encoder))
    if unknown:
        raise ValueError(f"Owners fuera del label_encoder del piloto: {unknown}")
    train_cls = train_owners.map(label_encoder).to_numpy(dtype=int)
    true_idx = test_owners.map(label_encoder).to_numpy(dtype=int)
    logger.info("CBR-recuperación Mozilla | text={} | train={} {}={} clases={}",
                config.text_mode, len(train), config.split, len(test), len(label_encoder))

    retriever = CbrRetriever(CbrConfig(
        encoder_name=config.model,
        device="cpu" if config.cpu else settings.torch_device,
        max_seq_length=config.max_seq_len,
        top_k=config.top_k,
        tau=config.tau,
        vote_strategy="idf",
    ))
    train_texts = combine_text(train, config.text_mode).tolist()
    test_texts = combine_text(test, config.text_mode).tolist()
    # Keep the pilot's complete candidate set and index mapping, including owners
    # absent from both splits. CbrRetriever's own encoder is alphabetical, so
    # its score columns are explicitly remapped after vote aggregation.
    retriever.fit(train_texts, train_owners.tolist(), test_owners=ordered_owners,
                  precomputed_embeddings=train_embeddings)
    sims = retriever.compute_dense_sims(test_texts, test_embeddings)
    column_order = [retriever.label_encoder[owner] for owner in ordered_owners]

    def score(k: int, tau: float) -> np.ndarray:
        internal = retriever.score_sims(sims, top_k=k, tau=tau, vote_strategy="idf")
        return internal[:, column_order]

    if config.sweep:
        points = [(k, tau) for k in (5, 10, 20, 30, 50, 75, 100)
                  for tau in (1.0, 2.0, 4.0)]
    else:
        points = [(config.top_k, config.tau)]
    evaluations = []
    for k, tau in points:
        scores = score(k, tau)
        evaluations.append({"top_k": k, "tau": tau, "scores": scores,
                            "metrics": rank_metrics(scores, true_idx)})
    result: dict[str, Any] = {
        "evaluations": evaluations,
        "label_encoder": label_encoder,
        "similarities": sims,
        "train_classes": train_cls,
        "true_indices": true_idx,
        "train": train,
        "test": test,
    }
    # The historical --sweep path returns before any IBR loading/fusion.
    if config.sweep or not config.fuse_ibr:
        return result

    nps_raw = evaluations[0]["scores"]
    nps = normalize_rows(nps_raw, how=config.nps_norm)
    if nis is None:
        if ibr is None:
            ibr = InteractionRecommender(text_mode=config.ibr_mode).load()
        nis = _ibr_matrix(test, label_encoder, ibr)
    nis = np.asarray(nis)
    if nis.shape != (len(test), len(label_encoder)):
        raise ValueError(f"nis debe tener forma {(len(test), len(label_encoder))}, recibida {nis.shape}.")
    wf_values = [round(x, 1) for x in np.arange(0.1, 1.0001, 0.1)]
    fusion = []
    for wf in wf_values:
        scores = linear_fuse(nps, nis, wf)
        fusion.append({"w_f": wf, "scores": scores, "metrics": rank_metrics(scores, true_idx)})
    result.update({"nps": nps, "nis": nis, "cbr_metrics": rank_metrics(nps, true_idx),
                   "ibr_metrics": rank_metrics(nis, true_idx), "fusion": fusion})
    return result


def _fmt(metrics: dict) -> str:
    return (f"Hit@1={metrics['hit@1']:.4f} Hit@3={metrics['hit@3']:.4f} "
            f"Hit@5={metrics['hit@5']:.4f} Hit@10={metrics['hit@10']:.4f} MRR={metrics['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CONFIG.model)
    parser.add_argument("--text-mode", choices=["raw", "distilled", "both"], default=CONFIG.text_mode,
                        help="vista del texto para el CBR (igual que el clasificador del piloto)")
    parser.add_argument("--ibr-mode", choices=["raw", "distilled", "both"], default=CONFIG.ibr_mode)
    parser.add_argument("--split", choices=["val", "test"], default=CONFIG.split)
    parser.add_argument("--top-k", type=int, default=CONFIG.top_k)
    parser.add_argument("--tau", type=float, default=CONFIG.tau)
    parser.add_argument("--max-seq-len", type=int, default=CONFIG.max_seq_len)
    parser.add_argument("--sweep", action="store_true", default=CONFIG.sweep,
                        help="barre (k, τ) sobre los mismos embeddings")
    parser.add_argument("--fuse-ibr", action="store_true", default=CONFIG.fuse_ibr,
                        help="fusiona con el IBR y barre W_f")
    parser.add_argument("--cpu", action="store_true", default=CONFIG.cpu)
    args = parser.parse_args()
    config = BenchmarkConfig(**{**vars(CONFIG), **vars(args)})
    result = evaluate(config)
    for item in result["evaluations"]:
        logger.info("k={} tau={} : {}", item["top_k"], item["tau"], _fmt(item["metrics"]))
    if "fusion" in result:
        logger.success("== Sistema completo CBR-recuperación + IBR | split={} ==", config.split)
        logger.info("  CBR-solo : {}", _fmt(result["cbr_metrics"]))
        logger.info("  IBR-solo : {}", _fmt(result["ibr_metrics"]))
        for item in result["fusion"]:
            logger.info("  FS W_f={:.1f}: {}", item["w_f"], _fmt(item["metrics"]))
    print(json.dumps([item["metrics"] for item in result["evaluations"]], indent=2))


if __name__ == "__main__":
    main()
