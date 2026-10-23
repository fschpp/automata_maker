"""Registro de familias de mecanismos y ajuste automático entre ellas."""

from __future__ import annotations

import numpy as np

from mechchar.optim.fitcore import Family, FitResult, ProgressFn, fit
from mechchar.optim.fivebar_fit import GEARED_FIVE_BAR
from mechchar.optim.fourbar_fit import FOUR_BAR

ALL_FAMILIES: tuple[Family, ...] = (FOUR_BAR, GEARED_FIVE_BAR)


def family_by_name(name: str) -> Family:
    for fam in ALL_FAMILIES:
        if fam.name == name:
            return fam
    raise KeyError(f"familia desconocida: {name!r}")


def fit_best(
    target: np.ndarray,
    families: tuple[Family, ...] = ALL_FAMILIES,
    *,
    on_progress: ProgressFn | None = None,
    **kwargs: object,
) -> FitResult | None:
    """Ajusta todas las familias y devuelve la de menor error (ver ``fitcore.fit``)."""
    return fit(target, families, on_progress=on_progress, **kwargs)  # type: ignore[arg-type]
