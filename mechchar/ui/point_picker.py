"""Elegir un punto sobre una pieza del personaje con un clic (punto de actuación o marcador)."""

from __future__ import annotations

from collections.abc import Callable

from mechchar.core.actuation import pick_point
from mechchar.core.character import Character, PointRef
from mechchar.core.viewport import Viewport

PICK_TOL_PX = 10.0  # un clic hasta esta distancia de una pieza la elige


class PointPicker:
    """Implementa ``DragTarget``: con un modo activo (``"attach"`` o ``"marker"``), el clic
    elige un punto de una pieza de ``character``. Reclama el arrastre para que no haga pan."""

    def __init__(self, viewport: Viewport) -> None:
        self.vp = viewport
        self.character: Character | None = None
        self.mode: str = "attach"
        self.on_pick: Callable[[PointRef, str], None] | None = None
        self.on_miss: Callable[[], None] | None = None

    def drag_begin(self, sx: float, sy: float) -> bool:
        if self.character is None:
            return False
        world = self.vp.screen_to_world(sx, sy)
        ref = pick_point(self.character, world, PICK_TOL_PX / self.vp.scale)
        if ref is None:
            if self.on_miss is not None:
                self.on_miss()
        elif self.on_pick is not None:
            self.on_pick(ref, self.mode)
        return True

    def drag_update(self, sx: float, sy: float) -> None:
        pass

    def drag_end(self) -> None:
        pass
