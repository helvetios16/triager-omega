"""DEPRECATED: El clasificador supervisado DeBERTa ha sido descartado.

La arquitectura activa de Case-Based Reasoning utiliza recuperación semántica
zero-shot e híbrida (ver `triager_omega.retrieval.CbrRetriever`).

El código de entrenamiento supervisado de DeBERTa fue reubicado en:
    scripts/legacy/train_deberta_supervised.py
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

warnings.warn(
    "triager_omega.cbr.train está deprecado. La arquitectura vigente utiliza "
    "triager_omega.retrieval. El entrenamiento de DeBERTa reside en scripts/legacy/train_deberta_supervised.py.",
    DeprecationWarning,
    stacklevel=2,
)

# Permitir importación retrocompatible de utilidades históricas
_LEGACY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "legacy"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

try:
    from train_deberta_supervised import BugDataset, WeightedTrainer, make_compute_metrics, main
except ImportError:
    pass

if __name__ == "__main__":
    main()
