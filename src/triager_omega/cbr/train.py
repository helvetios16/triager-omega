"""Módulo 3 — Entrenamiento del CBR (DeBERTa) sobre el piloto.

Basado en `scripts/train_cbr_quick.py` (misma config que corre en Mac/MPS:
adam_epsilon=1e-4 para evitar el nan de DeBERTa-v3 en MPS, WeightedRandomSampler
para la cola larga, métricas Hit@K/MRR). Diferencias:

  - Lee los artefactos del PILOTO: `artifacts/pilot/{splits.parquet, label_encoder.json}`.
  - Une `artifacts/pilot/distillations.parquet` y entrena sobre el texto de DOS VISTAS
    (§5.8): crudo `Summary [SEP] Product Component` + destilado `[FL] ... [SY] ... [CP] ...`.
  - Al terminar GUARDA el modelo entrenado en `artifacts/pilot/cbr_model/`
    (pesos + config + tokenizer) y las métricas en `metrics.json`.

Flujo de archivos:  distillations.parquet ─► [este módulo] ─► cbr_model/

Ejecutar (tras correr la destilación):
    uv run python -m triager_omega.cbr.train
    uv run python -m triager_omega.cbr.train --text-mode raw   # ablación §11.2.4
    uv run python -m triager_omega.cbr.train --eval-only       # solo val/test, sin reentrenar
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from triager_omega.config import settings
from triager_omega.data import loader


# --------------------------------------------------------------------------- #
# Datos: une splits piloto + texto crudo + texto destilado (dos vistas, §5.8)
# --------------------------------------------------------------------------- #
def build_dataframe(text_mode: str = "both") -> tuple[pd.DataFrame, dict[str, int]]:
    """Construye el df de entrenamiento del piloto con el texto según `text_mode`:
    'raw' (solo crudo), 'distilled' (solo destilado) o 'both' (concatenado)."""
    label_encoder: dict[str, int] = json.loads(
        settings.pilot_label_encoder_path.read_text(encoding="utf-8")
    )
    splits = pd.read_parquet(settings.pilot_splits_path).drop_duplicates(subset="Bug Id")

    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])
    bugs = bugs.drop_duplicates(subset="Bug Id")
    df = splits.merge(bugs, on="Bug Id", how="left")

    # texto crudo (una vista).
    for col in ("Summary", "Product", "Component"):
        df[col] = df[col].fillna("").astype(str)
    df["raw"] = df["Summary"] + " [SEP] " + df["Product"] + " " + df["Component"]

    # texto destilado (otra vista) desde el parquet de la etapa 2.
    if settings.distillations_path.exists():
        dist = pd.read_parquet(settings.distillations_path)[["Bug Id", "distilled_text"]]
        df = df.merge(dist.drop_duplicates("Bug Id"), on="Bug Id", how="left")
    else:
        df["distilled_text"] = ""
    df["distilled_text"] = df["distilled_text"].fillna("")

    n_missing = int((df["distilled_text"].str.strip() == "").sum())
    if n_missing:
        logger.warning("{} bugs sin destilado (usan solo crudo).", n_missing)

    # combinación según el modo (ablación §11.2.4).
    if text_mode == "raw":
        df["text"] = df["raw"]
    elif text_mode == "distilled":
        # si falta el destilado, cae al crudo para no perder la fila.
        df["text"] = df["distilled_text"].where(df["distilled_text"].str.strip() != "", df["raw"])
    else:  # both
        df["text"] = df["raw"] + " " + df["distilled_text"]

    df["label"] = df["contributor_id"].astype(str).map(label_encoder)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df, label_encoder


class BugDataset(Dataset):
    """Tokeniza perezosamente cada bug (idéntico a train_cbr_quick)."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_compute_metrics(ks=(1, 3, 5, 10)):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        order = np.argsort(-logits, axis=1)
        ranks = (order == labels[:, None]).argmax(axis=1)
        metrics = {f"hit@{k}": float((ranks < k).mean()) for k in ks}
        metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
        return metrics

    return compute_metrics


class WeightedTrainer(Trainer):
    """Sampler ponderado 1/freq para la cola larga (idéntico a train_cbr_quick)."""

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        if self._sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        w = torch.as_tensor(self._sample_weights, dtype=torch.double)
        return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


# --------------------------------------------------------------------------- #
# Dispositivo y precisión (autodetección)
# --------------------------------------------------------------------------- #
def detect_device() -> str:
    """Detecta el acelerador disponible: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_runtime(args: argparse.Namespace) -> dict:
    """Resuelve dispositivo, precisión, acumulación y eps de AdamW.

    Los valores 'auto' (None en la CLI) se ajustan al dispositivo:
      - CUDA: bf16 ON (si la GPU lo soporta) + batch físico chico con
        gradient accumulation para caber en GPUs de poca VRAM (p.ej. 8 GB);
        adam_eps=1e-8.
      - MPS: sin bf16; adam_eps=1e-4 (evita el nan de DeBERTa-v3 en MPS).
      - CPU: sin bf16; adam_eps=1e-8.
    Cualquier flag pasado explícitamente tiene prioridad sobre el 'auto'.
    """
    device = "cpu" if args.cpu else detect_device()

    if args.bf16 == "on":
        use_bf16 = True
    elif args.bf16 == "off":
        use_bf16 = False
    else:  # auto
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    adam_eps = args.adam_eps
    if adam_eps is None:
        adam_eps = 1e-4 if device == "mps" else 1e-8

    batch_size = args.batch_size if args.batch_size is not None else (8 if device == "cuda" else 16)
    grad_accum = args.grad_accum if args.grad_accum is not None else (2 if device == "cuda" else 1)

    rt = {
        "device": device,
        "bf16": use_bf16,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "adam_eps": adam_eps,
        "eff_batch": batch_size * grad_accum,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "-",
    }
    return rt


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    rt = resolve_runtime(args)
    logger.info(
        "Runtime | device={} gpu={} bf16={} batch={} grad_accum={} (eff_batch={}) adam_eps={}",
        rt["device"], rt["gpu"], rt["bf16"], rt["batch_size"],
        rt["grad_accum"], rt["eff_batch"], rt["adam_eps"],
    )

    df, label_encoder = build_dataframe(args.text_mode)
    num_labels = len(label_encoder)
    logger.info("Piloto | clases (devs): {} | bugs etiquetados: {} | text_mode={}",
                num_labels, len(df), args.text_mode)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    logger.info("train={} val={} test={}", len(train_df), len(val_df), len(test_df))

    # carpeta por modo (ablación §11.2.4): cbr_model_both / _raw / _distilled.
    out_dir = settings.pilot_dir / f"cbr_model_{args.text_mode}"

    # --eval-only: carga los pesos ya afinados desde out_dir (no el base de HF)
    # y se salta el entrenamiento; sirve para sacar val/test sin reentrenar.
    if args.eval_only:
        if not out_dir.exists():
            raise FileNotFoundError(
                f"--eval-only pero no existe el modelo entrenado en {out_dir}. "
                "Corre primero sin --eval-only para generarlo."
            )
        model_src = str(out_dir)
        logger.info("eval-only: cargando modelo afinado desde {}", out_dir)
    else:
        model_src = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_src)
    model = AutoModelForSequenceClassification.from_pretrained(model_src, num_labels=num_labels)

    train_ds = BugDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length)
    val_ds = BugDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_length)
    test_ds = BugDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, args.max_length)

    sample_weights = None
    if not args.no_weighted:
        freq = train_df["label"].value_counts()
        sample_weights = (1.0 / train_df["label"].map(freq)).to_numpy(dtype=np.float64)

    training_args = TrainingArguments(
        output_dir=str(settings.pilot_dir / "cbr_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=rt["batch_size"],
        per_device_eval_batch_size=rt["batch_size"],
        gradient_accumulation_steps=rt["grad_accum"],
        learning_rate=args.lr,
        adam_epsilon=rt["adam_eps"],  # 1e-4 en MPS (nan DeBERTa-v3), 1e-8 en CUDA/CPU
        bf16=rt["bf16"],
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
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(), sample_weights=sample_weights,
    )

    if args.eval_only:
        logger.info("== eval-only: se omite el entrenamiento ==")
    else:
        logger.info("== Entrenando ==")
        trainer.train()

    logger.info("== Evaluación final en val ==")
    metrics = {k: v for k, v in trainer.evaluate().items() if k.startswith("eval_")}
    for k, v in metrics.items():
        logger.info("  {} = {:.4f}", k, v)

    # Evaluación sobre el conjunto de TEST (número de reporte, no usado en training).
    logger.info("== Evaluación final en test ==")
    test_metrics = {
        k: v
        for k, v in trainer.evaluate(test_ds, metric_key_prefix="test").items()
        if k.startswith("test_")
    }
    for k, v in test_metrics.items():
        logger.info("  {} = {:.4f}", k, v)
    metrics.update(test_metrics)

    # --- Guardar el modelo entrenado (etapa 3 del flujo) ---
    # En eval-only no se re-guardan los pesos (ya existen); solo se actualizan
    # las métricas para no pisar el modelo con una copia idéntica.
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps({"text_mode": args.text_mode, "num_labels": num_labels, **metrics}, indent=2),
        encoding="utf-8",
    )
    if args.eval_only:
        logger.success("Métricas actualizadas en {}", out_dir / "metrics.json")
    else:
        logger.success("Modelo CBR guardado en {}", out_dir)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--text-mode", choices=["raw", "distilled", "both"], default="both",
                   help="qué texto usar (ablación §11.2.4): crudo, destilado o ambos")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=None,
                   help="batch físico por dispositivo; auto = 8 en CUDA, 16 en MPS/CPU")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="pasos de gradient accumulation; auto = 2 en CUDA, 1 en MPS/CPU")
    p.add_argument("--bf16", choices=["auto", "on", "off"], default="auto",
                   help="precisión bf16; auto = ON en CUDA compatible, OFF en MPS/CPU")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--no-weighted", action="store_true", help="desactiva el sampler ponderado")
    p.add_argument("--eval-only", action="store_true",
                   help="carga el modelo afinado de cbr_model_<modo> y solo evalúa val/test (sin reentrenar)")
    p.add_argument("--cpu", action="store_true", help="fuerza CPU")
    # eps de AdamW: auto = 1e-4 en MPS (nan de DeBERTa-v3) y 1e-8 en CUDA/CPU.
    p.add_argument("--adam-eps", type=float, default=None,
                   help="epsilon de AdamW; auto = 1e-4 en MPS (evita el nan de DeBERTa-v3), 1e-8 en CUDA/CPU")
    p.add_argument("--seed", type=int, default=settings.seed)
    run(p.parse_args())


if __name__ == "__main__":
    main()
