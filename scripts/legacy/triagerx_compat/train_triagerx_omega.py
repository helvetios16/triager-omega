"""Entrena el CBR ensemble de TriagerX (DeBERTa-base + RoBERTa-base + 3 CNN)
sobre NUESTRO split exacto de OpenJ9 (51 clases), para una comparación
apples-to-apples contra el CBR-recuperación de triager-omega.

Diferencias vs training/developer/developer_training_openj9.py del repo TriagerX:
  - NO hace train_test_split interno: consume directamente nuestros CSV ya
    divididos (--train-csv / --test-csv), preservando el split temporal de
    build_openj9_50.py.
  - Conserva NUESTRAS 51 clases: --threshold 1 mantiene todos los owners de
    train (su default 20 los filtraría sobre el split-train y cambiaría el set).
  - Sintetiza las columnas que su TextProcessor exige (issue_url, labels) a
    partir de nuestro esquema (issue_number, issue_title, issue_body, owner,
    component), sin tocar el código de TriagerX.
  - Desactiva wandb (log-manager stub) para correr headless en omen.

Todo lo demás (modelo, CombinedLoss, AdamW eps=1e-8, scheduler, ModelTrainer,
ModelEvaluator, topk) queda idéntico al original.

Uso (en omen, con el venv de triagerX):
  triagerX\\.venv\\Scripts\\python.exe train_triagerx_omega.py \\
    --triagerx-root C:\\Users\\OMEN\\Documents\\Programacion\\Python\\triagerX \\
    --config <triagerx-root>\\training\\training_config\\openj9\\developer\\triagerx.yaml \\
    --train-csv <triagerx-root>\\omega_split\\openj9_train_50.csv \\
    --test-csv  <triagerx-root>\\omega_split\\openj9_test_50.csv \\
    --out-dir   <triagerx-root>\\omega_split\\runs --threshold 1 --dry-run
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

tqdm.pandas()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triagerx-root", required=True, help="Raíz del repo TriagerX clonado")
    p.add_argument("--config", required=True, help="YAML de config (p.ej. triagerx.yaml)")
    p.add_argument("--train-csv", required=True, help="Nuestro openj9_train_50.csv")
    p.add_argument("--test-csv", required=True, help="Nuestro openj9_test_50.csv")
    p.add_argument(
        "--val-csv",
        default=None,
        help="Nuestro openj9_val_50.csv. Si se da, la selección de checkpoint se hace "
        "sobre val y se evalúa en test (comparación honesta). Si NO se da, se replica "
        "el método de TriagerX: validación = test (selección sobre test = sesgo optimista).",
    )
    p.add_argument("--out-dir", required=True, help="Carpeta de salida (pesos + reporte)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Min issues/owner en train para conservar la clase (1 = preservar nuestras 51)",
    )
    p.add_argument("--dry-run", action="store_true", help="Solo validar el data path; no entrena")
    # --- knobs para caber en la RTX 5060 8GB (no cambian la metodología, solo memoria) ---
    p.add_argument("--batch-size", type=int, default=None, help="Override del batch_size del YAML (8GB: 4-6)")
    p.add_argument("--epochs", type=int, default=None, help="Override de epochs del YAML")
    p.add_argument("--max-tokens", type=int, default=None, help="Override de max_tokens del YAML")
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Activa gradient checkpointing en los transformers base (ahorra VRAM a costa de ~20%% tiempo)",
    )
    p.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Si >0, recorta train/val/test a N filas (humo de memoria/estabilidad, no reportable)",
    )
    return p.parse_args()


class NoOpLog:
    """Reemplaza EpochLogManager para correr sin wandb."""

    def log_epoch(self, epoch_num, total_epochs, metrics):
        msg = " | ".join(f"{k}: {v}" for k, v in metrics.items())
        logger.info(f"Epochs: {epoch_num + 1}/{total_epochs} | {msg}")

    def finish(self):
        pass


def load_split(path, text_processor, use_description):
    """Lee nuestro CSV y lo deja en el formato que espera el pipeline TriagerX."""
    df = pd.read_csv(path)
    # Columnas que TextProcessor.prepare_dataframe(is_openj9=True) exige:
    if "issue_url" not in df.columns:
        df["issue_url"] = df["issue_number"].apply(
            lambda n: f"https://github.com/eclipse-openj9/openj9/issues/{n}"
        )
    if "labels" not in df.columns:
        df["labels"] = df["component"] if "component" in df.columns else ""
    # El original renombra assignees->owner (ya tenemos owner) e issue_body->description.
    df = df.rename(columns={"issue_body": "description"})
    df = text_processor.prepare_dataframe(
        df,
        use_special_tokens=False,
        use_summary=False,
        use_description=use_description,
        component_training=False,
    )
    df = df.sort_values(by="issue_number")
    df = df[df["owner"].notna()]
    return df


def main():
    args = parse_args()
    sys.path.insert(0, args.triagerx_root)

    from triagerx.dataset.text_processor import TextProcessor
    from triagerx.loss.loss_functions import CombinedLoss
    from triagerx.model.module_factory import DatasetFactory, ModelFactory
    from triagerx.trainer.model_evaluator import ModelEvaluator
    from triagerx.trainer.model_trainer import ModelTrainer
    from triagerx.trainer.train_config import TrainConfig
    from util.epoch_log_manager import EpochLogManager

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    use_description = config.get("use_description")
    base_transformer_models = config.get("base_transformer_models")
    unfrozen_layers = config.get("unfrozen_layers")
    num_classifiers = config.get("num_classifiers")
    dropout = config.get("dropout")
    max_tokens = args.max_tokens or config.get("max_tokens")
    model_key = config.get("model_key")
    learning_rate = config.get("learning_rate")
    epochs = args.epochs or config.get("epochs")
    batch_size = args.batch_size or config.get("batch_size")
    early_stopping_patience = config.get("early_stopping_patience")
    topk_indices = config.get("topk_indices")
    logger.info(f"batch_size={batch_size} epochs={epochs} max_tokens={max_tokens} grad_ckpt={args.grad_checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = f"openj9_triagerx_omega_seed{args.seed}_th{args.threshold}"
    weights_save_location = os.path.join(args.out_dir, f"{run_name}.pt")
    test_report_location = os.path.join(args.out_dir, f"classification_report_{run_name}.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed=args.seed)
    logger.info(f"Device: {device} | run: {run_name}")

    # --- NUESTRO split (sin train_test_split interno) ---
    df_train = load_split(args.train_csv, TextProcessor, use_description)
    df_test = load_split(args.test_csv, TextProcessor, use_description)
    df_val = load_split(args.val_csv, TextProcessor, use_description) if args.val_csv else None

    dev_counts = df_train["owner"].value_counts()
    keep = dev_counts.index[dev_counts >= args.threshold]
    df_train = df_train[df_train["owner"].isin(keep)]

    train_owners = sorted(set(df_train["owner"]))
    owner_set = set(train_owners)
    df_test = df_test[df_test["owner"].isin(owner_set)]
    if df_val is not None:
        df_val = df_val[df_val["owner"].isin(owner_set)]

    lbl2idx = {dev: idx for idx, dev in enumerate(train_owners)}
    idx2lbl = {idx: dev for dev, idx in lbl2idx.items()}
    df_train["owner_id"] = df_train["owner"].map(lbl2idx)
    df_test["owner_id"] = df_test["owner"].map(lbl2idx)
    if df_val is not None:
        df_val["owner_id"] = df_val["owner"].map(lbl2idx)

    if args.smoke > 0:
        logger.warning(f"SMOKE: recortando a {args.smoke} filas por split (no reportable)")
        df_train = df_train.head(args.smoke)
        df_test = df_test.head(args.smoke)
        if df_val is not None:
            df_val = df_val.head(args.smoke)

    # Selección de checkpoint: en val si se dio (honesto), si no en test (réplica de TriagerX).
    df_select = df_val if df_val is not None else df_test
    select_src = "val (held-out, honesto)" if df_val is not None else "test (réplica TriagerX, sesgo)"
    n_val = len(df_val) if df_val is not None else 0
    logger.info(
        f"Train: {len(df_train)} | Val: {n_val} | Test: {len(df_test)} | clases: {len(train_owners)}"
    )
    logger.info(f"Selección de checkpoint sobre: {select_src}")
    logger.info(f"Ejemplo de text:\n{df_train['text'].iloc[0][:300]}")

    if args.dry_run:
        logger.info("DRY-RUN: data path validado. No se entrena.")
        return

    class_counts = np.bincount(df_train["owner_id"])
    num_samples = int(sum(class_counts))
    labels = df_train["owner_id"].to_list()
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(num_samples)]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

    model = ModelFactory.get_model(
        model_key=model_key,
        output_size=len(train_owners),
        unfrozen_layers=unfrozen_layers,
        num_classifiers=num_classifiers,
        base_models=base_transformer_models,
        dropout=dropout,
        max_tokens=max_tokens,
        label_map=idx2lbl,
    )

    if args.grad_checkpoint:
        # Ahorro de VRAM: recomputa activaciones en el backward en vez de guardarlas.
        # use_reentrant=False evita el error con capas congeladas (sin grad).
        for bm in model.base_models:
            bm.config.use_cache = False
            bm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing activado en los transformers base.")

    criterion = CrossEntropyLoss() if model_key == "fcn-transformer" else CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.001)

    train_ds = DatasetFactory.get_dataset(df_train, model, "text", "owner_id", max_length=max_tokens)
    select_ds = DatasetFactory.get_dataset(df_select, model, "text", "owner_id", max_length=max_tokens)
    test_ds = DatasetFactory.get_dataset(df_test, model, "text", "owner_id", max_length=max_tokens)

    train_dataloader = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=False, sampler=sampler
    )
    # Para la selección de checkpoint durante el entrenamiento (val si se dio, si no test).
    select_dataloader = DataLoader(select_ds, batch_size=batch_size)
    # Para la evaluación final reportable (SIEMPRE test held-out).
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_config = TrainConfig(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=select_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        output_path=weights_save_location,
        device=device,
        topk_indices=topk_indices,
        log_manager=EpochLogManager(None),  # None => sin wandb (no login), pasa la validación pydantic
        early_stopping_patience=early_stopping_patience,
        scheduler=scheduler,
    )

    logger.info("Starting training...")
    ModelTrainer(train_config).train()
    logger.info("Finished training. Evaluating...")

    model.load_state_dict(torch.load(weights_save_location))
    ModelEvaluator().evaluate(
        model=model,
        dataloader=test_dataloader,
        device=device,
        run_name=run_name,
        topk_indices=topk_indices,
        weights_save_location=weights_save_location,
        test_report_location=test_report_location,
        combined_loss=False if model_key == "fcn-transformer" else True,
    )
    logger.info(f"Listo. Reporte: {test_report_location}")


if __name__ == "__main__":
    main()
