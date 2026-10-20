from __future__ import annotations

from typing import Iterable, Sequence
import pandas as pd


def build_label_encoder(
    train_owners: Sequence[str] | pd.Series,
    test_owners: Sequence[str] | pd.Series | None = None,
) -> dict[str, int]:
    """Construye un mapeo determinista de propietarios/desarrolladores a índices enteros [0..C-1]."""
    owners = set(pd.Series(train_owners).dropna().unique())
    if test_owners is not None:
        owners |= set(pd.Series(test_owners).dropna().unique())
    return {owner: idx for idx, owner in enumerate(sorted(owners))}
