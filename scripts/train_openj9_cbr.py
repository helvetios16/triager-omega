"""Entrena un CBR (DeBERTa) sobre OpenJ9 — para la validación del sistema completo.

Análogo a `triager_omega.cbr.train` pero sobre los CSV de TriagerX (OpenJ9) en vez
del piloto Mozilla. Reusa la maquinaria de entrenamiento (BugDataset, métricas Hit@K/MRR,
WeightedRandomSampler, adam_eps=1e-4 para el nan de DeBERTa-v3 en MPS).

  - Texto: columna `text` (título+cuerpo preprocesado de TriagerX; OpenJ9 no tiene destilado).
  - Etiqueta: `owner` (login) → class_idx vía label_encoder = sorted(owners), MISMO orden
    que `scripts/eval_openj9_ibr.py` para que CBR e IBR se puedan sumar por columna.
  - Split: train CSV vs test CSV (time-sliced de TriagerX; no hay val).

Salida: artifacts/openj9/cbr_model/ (pesos + tokenizer + label_encoder.json + metrics.json).

Uso:
    uv run python scripts/train_openj9_cbr.py
    uv run python scripts/train_openj9_cbr.py --epochs 5 --model microsoft/deberta-v3-base
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from triager_omega.cbr.train import BugDataset, WeightedTrainer, make_compute_metrics
from triager_omega.config import settings


def label_encoder(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """sorted(owners) — idéntico a eval_openj9_ibr._label_encoder (CBR e IBR alineados)."""
    owners = sorted(set(train["owner"].dropna()) | set(test["owner"].dropna()))
    return {o: i for i, o in enumerate(owners)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--adam-eps", type=float, default=1e-4, help="1e-4 evita nan de DeBERTa-v3 en MPS")
    p.add_argument("--no-weighted", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=settings.seed)
    # Permite apuntar a CSVs alternativos (p.ej. el set de 50 clases) sin pisar
    # el modelo de 17: --train-csv/--test-csv + --out-name.
    p.add_argument("--train-csv", default=None, help="CSV de train (default: config 17-set)")
    p.add_argument("--test-csv", default=None, help="CSV de test (default: config 17-set)")
    p.add_argument("--out-name", default="cbr_model", help="subcarpeta de salida en artifacts/openj9/")
    args = p.parse_args()

    import torch
    torch.manual_seed(args.seed)

    cfg = settings
    train = pd.read_csv(args.train_csv or cfg.openj9_train_csv).drop_duplicates("issue_number")
    test = pd.read_csv(args.test_csv or cfg.openj9_test_csv).drop_duplicates("issue_number")
    le = label_encoder(train, test)
    num_labels = len(le)
    logger.info("OpenJ9 CBR | train={} test={} clases(owner)={}", len(train), len(test), num_labels)

    for df in (train, test):
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["owner"].map(le).astype(int)

    out_dir = cfg.openj9_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    train_ds = BugDataset(train["text"].tolist(), train["label"].tolist(), tokenizer, args.max_length)
    test_ds = BugDataset(test["text"].tolist(), test["label"].tolist(), tokenizer, args.max_length)

    sample_weights = None
    if not args.no_weighted:
        freq = train["label"].value_counts()
        sample_weights = (1.0 / train["label"].map(freq)).to_numpy(dtype=np.float64)

    training_args = TrainingArguments(
        output_dir=str(cfg.openj9_dir / "cbr_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        adam_epsilon=args.adam_eps,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        use_cpu=args.cpu,
    )

    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=make_compute_metrics(), sample_weights=sample_weights,
    )
    logger.info("== Entrenando CBR OpenJ9 ==")
    trainer.train()

    logger.info("== Evaluación en test ==")
    metrics = {k: v for k, v in trainer.evaluate(test_ds, metric_key_prefix="test").items()
               if k.startswith("test_")}
    for k, v in metrics.items():
        logger.info("  {} = {:.4f}", k, v)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_encoder.json").write_text(json.dumps(le, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"num_labels": num_labels, **metrics}, indent=2), encoding="utf-8")
    logger.success("CBR OpenJ9 guardado en {}", out_dir)


if __name__ == "__main__":
    main()
