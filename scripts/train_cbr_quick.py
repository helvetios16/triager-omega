# Prueba minima de fine-tuning de DeBERTa (Modulo 3 / CBR) — version exploratoria.
# Objetivo: aprender el ciclo de entrenamiento de un clasificador transformer de
# punta a punta, SIN destilacion (Modulo 2). Usa texto crudo:
#     "<Summary> [SEP] <Product> <Component>"
# como entrada y el `contributor_id` (via label_encoder) como etiqueta.
# NO es el modulo de produccion del PLAN (eso sera src/.../modules/cbr.py).
# Es un script de juguete para experimentar con pocos datos y ver metricas Hit@K.
# Ejecutar (ejemplo rapido, ~unos minutos en M-series con MPS):
#     uv run python scripts/train_cbr_quick.py --max-train 3000 --max-eval 1000 --epochs 2
# Argumentos utiles:
#     --model        modelo base HuggingFace (default: microsoft/deberta-v3-base)
#     --max-train N  n max de bugs de train a usar (subconjunto para ir rapido)
#     --max-eval N   n max de bugs de val a usar
#     --epochs       epocas
#     --batch-size   tamano de batch
#     --lr           learning rate
#     --max-length   longitud maxima de tokens
#     --no-weighted  desactiva el WeightedRandomSampler (balanceo de cola larga)

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
# Datos
# --------------------------------------------------------------------------- #
def build_dataframe() -> tuple[pd.DataFrame, dict[str, int]]:
    # Une splits.parquet con el texto de los bugs y mapea la etiqueta a class_idx.

    # Carga el mapeo contributor_id -> class_idx generado en pasos previos.
    label_encoder: dict[str, int] = json.loads(
        settings.label_encoder_path.read_text(encoding="utf-8")
    )

    # Lee la asignacion de cada bug a un split (Bug Id, contributor_id, split).
    splits = pd.read_parquet(settings.splits_path)
    # Carga solo las columnas de texto necesarias de los bugs.
    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])

    # Cruza splits con el texto de cada bug por Bug Id.
    df = splits.merge(bugs, on="Bug Id", how="left")

    # Construye el texto crudo de entrada (sin destilacion).
    # Rellena nulos y fuerza str en cada columna de texto antes de concatenar.
    for col in ("Summary", "Product", "Component"):
        df[col] = df[col].fillna("").astype(str)
    # Concatena Summary + Product + Component separados por [SEP].
    df["text"] = df["Summary"] + " [SEP] " + df["Product"] + " " + df["Component"]

    # Etiqueta: contributor_id -> class_idx via label_encoder.
    df["label"] = df["contributor_id"].astype(str).map(label_encoder)
    # Filtra los devs que no estan presentes en el encoder (label NaN).
    df = df.dropna(subset=["label"]).copy()
    # Convierte la etiqueta a entero para el clasificador.
    df["label"] = df["label"].astype(int)

    return df, label_encoder


class BugDataset(Dataset):
    # Tokeniza perezosamente cada bug. Devuelve input_ids/attention_mask/labels.

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Tokeniza el texto del bug con truncado y padding a max_length.
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Devuelve los tensores aplanados (sin la dimension de batch) + la etiqueta.
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Metricas Hit@K / MRR
# --------------------------------------------------------------------------- #
def make_compute_metrics(ks=(1, 5, 10)):
    def compute_metrics(eval_pred):
        # Desempaqueta logits y etiquetas reales del batch de evaluacion.
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        # Ordena las clases de forma descendente por logit (ranking de devs).
        order = np.argsort(-logits, axis=1)
        # Posicion 0-based del dev real dentro del ranking.
        ranks = (order == labels[:, None]).argmax(axis=1)
        # Hit@K: fraccion de bugs cuyo dev real cae en el top-K.
        metrics = {f"hit@{k}": float((ranks < k).mean()) for k in ks}
        # MRR: media del reciproco del rango (1 = perfecto).
        metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
        return metrics

    return compute_metrics


# --------------------------------------------------------------------------- #
# Trainer con muestreo ponderado (balanceo de cola larga)
# --------------------------------------------------------------------------- #
class WeightedTrainer(Trainer):
    # Sustituye el sampler de train por un WeightedRandomSampler (w = 1/freq_clase).

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        # Sin pesos: cae al sampler por defecto de HuggingFace.
        if self._sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        # Con pesos: muestrea con reemplazo proporcional a 1/freq_clase.
        w = torch.as_tensor(self._sample_weights, dtype=torch.double)
        return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # Define los argumentos de linea de comandos del experimento.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--max-train", type=int, default=3000)
    p.add_argument("--max-eval", type=int, default=1000)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--no-weighted", action="store_true", help="desactiva el sampler ponderado")
    p.add_argument("--cpu", action="store_true", help="fuerza CPU (evita el nan de DeBERTa en MPS)")
    # Ajuste de parametro forzado por el chip M (Apple Silicon / backend MPS).
    # Epsilon de AdamW: en MPS DeBERTa-v3 da nan con el default 1e-8.
    # El 2 momento de Adam queda ~0 en MPS y al dividir por sqrt(v)+eps explota a nan.
    # Subir eps a 1e-4 estabiliza el entrenamiento en el chip M. Usa 1e-8 en CPU/CUDA.
    p.add_argument(
        "--adam-eps",
        type=float,
        default=1e-4,
        help="epsilon de AdamW. En MPS DeBERTa-v3 da nan con el default 1e-8; "
        "1e-4 lo estabiliza (el 2 momento de Adam queda ~0 en MPS). Usa 1e-8 en CPU/CUDA.",
    )
    p.add_argument("--seed", type=int, default=settings.seed)
    args = p.parse_args()

    # Fija la semilla global de torch para reproducibilidad.
    torch.manual_seed(args.seed)

    # Construye el dataframe etiquetado y obtiene el n de clases (devs).
    df, label_encoder = build_dataframe()
    num_labels = len(label_encoder)
    logger.info("Clases (devs): {} | bugs etiquetados: {}", num_labels, len(df))

    # Separa los bugs por split de train y validacion.
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    # Submuestrea train/val para experimentar rapido (muestreo reproducible).
    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed)
    if args.max_eval and len(val_df) > args.max_eval:
        val_df = val_df.sample(n=args.max_eval, random_state=args.seed)
    logger.info("Usando train={} val={}", len(train_df), len(val_df))

    # Carga el tokenizer y el modelo base con cabeza de clasificacion.
    logger.info("Cargando tokenizer y modelo base '{}'...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # Envuelve los textos/etiquetas en datasets que tokenizan al vuelo.
    train_ds = BugDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length
    )
    val_ds = BugDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_length
    )

    # Pesos de muestreo = 1 / frecuencia de la clase EN el subconjunto de train.
    sample_weights = None
    if not args.no_weighted:
        # Frecuencia de cada clase dentro del train submuestreado.
        freq = train_df["label"].value_counts()
        # Peso inverso a la frecuencia para favorecer clases raras (cola larga).
        sample_weights = (1.0 / train_df["label"].map(freq)).to_numpy(dtype=np.float64)

    # Configura los hiperparametros y la estrategia de train/eval.
    training_args = TrainingArguments(
        output_dir="artifacts/cbr_quick",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        # Ajuste por el chip M: eps de AdamW elevado para evitar el nan en MPS.
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

    # Instancia el trainer con sampler ponderado y metricas Hit@K/MRR.
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(),
        sample_weights=sample_weights,
    )

    # Lanza el ciclo de entrenamiento.
    logger.info("== Entrenando ==")
    trainer.train()

    # Evalua el modelo final en el split de validacion.
    logger.info("== Evaluacion final en val ==")
    metrics = trainer.evaluate()
    # Imprime solo las metricas de evaluacion (prefijo eval_).
    for k, v in metrics.items():
        if k.startswith("eval_"):
            logger.info("  {} = {:.4f}", k, v)


if __name__ == "__main__":
    main()
