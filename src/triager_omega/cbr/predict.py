"""Inferencia del CBR (Módulo 3) → NPS para el agregador (Módulo 5).

Carga el DeBERTa afinado en `artifacts/pilot/cbr_model_<text_mode>/` y devuelve,
para cada bug, la matriz de *logits* sobre las clases (devs del piloto) y/o el
NPS normalizado. El orden de columnas es el del `label_encoder` (class_idx),
igual que el IBR, para que el agregador pueda sumar `FS = NPS + W_f·NIS` columna
a columna.

Normalización del NPS:
  - 'minmax'  : (x − min)/(max − min) por fila — **réplica de TriagerX**
    (`_normalize_tensor` sobre la salida del modelo). Misma escala [0,1] que el NIS.
  - 'softmax' : distribución de probabilidad (variante del PLAN §8.2).
"""

from __future__ import annotations

import json

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from triager_omega.config import Settings, settings


def normalize_rows(logits: np.ndarray, how: str = "minmax") -> np.ndarray:
    """Normaliza cada fila a [0,1] (minmax, TriagerX) o a probabilidad (softmax)."""
    if how == "softmax":
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    # minmax por fila (réplica de _normalize_tensor de TriagerX).
    mn = logits.min(axis=1, keepdims=True)
    mx = logits.max(axis=1, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, mx - mn)
    return (logits - mn) / rng


class CbrPredictor:
    """Envuelve el DeBERTa afinado del piloto para inferencia batched."""

    def __init__(self, text_mode: str = "both", cfg: Settings = settings, device: str | None = None):
        self.cfg = cfg
        self.text_mode = text_mode
        self.device = device or cfg.torch_device
        out_dir = cfg.pilot_dir / f"cbr_model_{text_mode}"
        if not out_dir.exists():
            raise FileNotFoundError(
                f"No existe el modelo CBR en {out_dir}. Entrénalo con "
                f"`python -m triager_omega.cbr.train --text-mode {text_mode}`."
            )
        logger.info("Cargando CBR afinado desde {} (device={})", out_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(out_dir))
        self.model.to(self.device).eval()

        self.label_encoder: dict[str, int] = json.loads(
            cfg.pilot_label_encoder_path.read_text(encoding="utf-8")
        )
        # columna class_idx -> Contributor Id (para mapear con el IBR).
        self.idx2dev = {idx: int(dev) for dev, idx in self.label_encoder.items()}
        self.num_classes = len(self.label_encoder)

    @torch.no_grad()
    def logits(self, texts: list[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
        """Devuelve la matriz de logits `[N, num_classes]` (orden = class_idx)."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits  # [B, C]
            out.append(logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def nps_matrix(self, texts: list[str], how: str = "minmax", **kw) -> np.ndarray:
        """Matriz NPS `[N, num_classes]` normalizada por fila."""
        return normalize_rows(self.logits(texts, **kw), how=how)
