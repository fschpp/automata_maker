"""Spline cerrada editable: puntos de control arrastrables dibujados sobre el canvas."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from mechchar.core.spline import closed_catmull_rom, pick_point
from mechchar.core.viewport import Viewport

HIT_RADIUS_PX = 12.0
CURVE_COLOR = (0.1, 0.1, 0.1)
POINT_COLOR = (0.15, 0.45, 0.85)


def default_points(n: int = 8, a: float = 3.0, b: float = 1.5) -> list[tuple[float, float]]:
    """Puntos de control iniciales sobre una elipse."""
    return [
        (a * math.cos(2 * math.pi * k / n), b * math.sin(2 * math.pi * k / n)) for k in range(n)
    ]


class SplineEditor:
    """Implementa el protocolo ``DragTarget`` del canvas: reclama el arrastre si empieza sobre un
    punto de control y lo mueve; si no, el canvas hace pan."""

    def __init__(self, viewport: Viewport, points: list[tuple[float, float]] | None = None) -> None:
        self.vp = viewport
        self.points = list(points) if points is not None else default_points()
        self.on_change: Callable[[], None] | None = None
        self._active: int | None = None

    def curve(self, per_segment: int = 16) -> np.ndarray:
        return closed_catmull_rom(np.array(self.points), per_segment)

    # --- DragTarget -----------------------------------------------------
    def drag_begin(self, sx: float, sy: float) -> bool:
        self._active = pick_point(self.points, self.vp, sx, sy, HIT_RADIUS_PX)
        return self._active is not None

    def drag_update(self, sx: float, sy: float) -> None:
        if self._active is None:
            return
        self.points[self._active] = self.vp.screen_to_world(sx, sy)
        if self.on_change is not None:
            self.on_change()

    def drag_end(self) -> None:
        self._active = None

    # --- dibujo ---------------------------------------------------------
    def draw(self, cr: Any, vp: Viewport) -> None:
        curve = self.curve()
        cr.set_source_rgb(*CURVE_COLOR)
        cr.set_line_width(2.0)
        cr.move_to(*vp.world_to_screen(*curve[0]))
        for x, y in curve[1:]:
            cr.line_to(*vp.world_to_screen(x, y))
        cr.close_path()
        cr.stroke()
        for i, (x, y) in enumerate(self.points):
            sx, sy = vp.world_to_screen(x, y)
            cr.arc(sx, sy, 6.0 if i != self._active else 8.0, 0, 2 * math.pi)
            cr.set_source_rgb(1, 1, 1)
            cr.fill_preserve()
            cr.set_source_rgb(*POINT_COLOR)
            cr.set_line_width(2.0)
            cr.stroke()
