"""Transformación mundo <-> pantalla, zoom y pan. Sin dependencias de GTK.

Convención: el mundo tiene y hacia arriba; la pantalla tiene y hacia abajo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_SCALE = 1e-3  # px por unidad de mundo
MAX_SCALE = 1e5


@dataclass
class Viewport:
    scale: float = 50.0  # px por unidad de mundo
    ox: float = 0.0  # posición en pantalla (px) del origen del mundo
    oy: float = 0.0

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x * self.scale, self.oy - y * self.scale

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.ox) / self.scale, (self.oy - sy) / self.scale

    def pan_by(self, dx: float, dy: float) -> None:
        """Desplaza la vista dx, dy píxeles de pantalla."""
        self.ox += dx
        self.oy += dy

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Escala por `factor` manteniendo fijo el punto de mundo bajo (sx, sy)."""
        new_scale = min(max(self.scale * factor, MIN_SCALE), MAX_SCALE)
        wx, wy = self.screen_to_world(sx, sy)
        self.scale = new_scale
        self.ox = sx - wx * new_scale
        self.oy = sy + wy * new_scale

    def center_origin(self, width: float, height: float) -> None:
        self.ox, self.oy = width / 2, height / 2


def nice_step(target_world: float) -> float:
    """Menor paso de la secuencia 1-2-5 x 10^k que es >= target_world."""
    if target_world <= 0:
        raise ValueError("target_world debe ser positivo")
    exp = math.floor(math.log10(target_world))
    base = 10.0**exp
    for m in (1, 2, 5, 10):
        if m * base >= target_world * (1 - 1e-12):
            return m * base
    raise AssertionError("inalcanzable")


def grid_lines(
    vp: Viewport, width: float, height: float, target_px: float = 50.0
) -> tuple[float, list[float], list[float]]:
    """Devuelve (paso, xs, ys): posiciones de mundo de las líneas verticales y horizontales."""
    step = nice_step(target_px / vp.scale)
    x0, y1 = vp.screen_to_world(0, 0)
    x1, y0 = vp.screen_to_world(width, height)
    xs = [i * step for i in range(math.ceil(x0 / step), math.floor(x1 / step) + 1)]
    ys = [j * step for j in range(math.ceil(y0 / step), math.floor(y1 / step) + 1)]
    return step, xs, ys
