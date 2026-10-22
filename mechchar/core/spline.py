"""Spline cerrada que pasa por sus puntos de control, y selección de puntos. Sin GTK."""

from __future__ import annotations

import numpy as np

from mechchar.core.viewport import Viewport


def closed_catmull_rom(points: np.ndarray, per_segment: int = 16) -> np.ndarray:
    """Muestrea una spline Catmull-Rom cerrada que interpola ``points`` (n >= 3).

    Devuelve ``n * per_segment`` puntos; el primero de cada tramo es un punto de control.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n < 3:
        raise ValueError("se necesitan al menos 3 puntos de control")
    p0, p1 = np.roll(p, 1, axis=0), p
    p2, p3 = np.roll(p, -1, axis=0), np.roll(p, -2, axis=0)
    u = (np.arange(per_segment) / per_segment)[None, :, None]  # (1, k, 1)
    a = (2.0 * p1)[:, None, :]
    b = (-p0 + p2)[:, None, :]
    c = (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[:, None, :]
    d = (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[:, None, :]
    seg = 0.5 * (a + b * u + c * u**2 + d * u**3)  # (n, k, 2)
    return seg.reshape(n * per_segment, 2)


def pick_point(
    points: list[tuple[float, float]], vp: Viewport, sx: float, sy: float, radius_px: float
) -> int | None:
    """Índice del punto de control (en mundo) más cercano a (sx, sy) en píxeles, o None."""
    best: int | None = None
    best_d2 = radius_px**2
    for i, (x, y) in enumerate(points):
        px, py = vp.world_to_screen(x, y)
        d2 = (px - sx) ** 2 + (py - sy) ** 2
        if d2 <= best_d2:
            best, best_d2 = i, d2
    return best
