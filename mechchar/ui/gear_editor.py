"""Interacción con el mouse del editor de la red de engranajes (``DragTarget`` del ``Canvas``).

No importa GTK: recibe el ``Viewport`` y el ``GearNetModel`` y se prueba llamando a
``drag_begin`` / ``drag_update`` / ``drag_end`` como lo hace el ``Canvas``.

* Apretar sobre un eje reclama el arrastre: si el mouse se mueve más de ``DRAG_PX`` lo mueve; si
  se suelta casi donde se apretó es un clic y marca o desmarca el eje (dos marcados se pueden
  conectar).
* Con el modo «agregar» un clic en un lugar vacío agrega un eje ahí (redondeado a ``SNAP``).
* En cualquier otro caso no reclama el arrastre y el ``Canvas`` hace el desplazamiento.
"""

from __future__ import annotations

from collections.abc import Callable

from mechchar.core.gearedit import GearNetModel
from mechchar.core.spline import pick_point
from mechchar.core.viewport import Viewport

HIT_PX = 14.0  # radio para agarrar un eje
DRAG_PX = 4.0  # cuánto hay que mover el mouse para que sea arrastre y no clic
SNAP = 0.1  # los ejes que se agregan o se arrastran caen en múltiplos de esto (mundo)


def snap(v: float, step: float = SNAP) -> float:
    return round(v / step) * step


class GearEditor:
    def __init__(
        self, model: GearNetModel, viewport: Viewport, on_change: Callable[[], None]
    ) -> None:
        self.model = model
        self.viewport = viewport
        self.on_change = on_change
        self.add_mode = False
        self._axis: int | None = None
        self._adding = False
        self._press = (0.0, 0.0)
        self._moved = False

    def _hit(self, sx: float, sy: float) -> int | None:
        pts = [(a.x, a.y) for a in self.model.axes]
        return pick_point(pts, self.viewport, sx, sy, HIT_PX)

    # --- DragTarget ----------------------------------------------------------------------------
    def drag_begin(self, sx: float, sy: float) -> bool:
        self._press = (sx, sy)
        self._moved = False
        self._axis = self._hit(sx, sy)
        self._adding = self._axis is None and self.add_mode
        return self._axis is not None or self._adding

    def drag_update(self, sx: float, sy: float) -> None:
        if self._axis is None:
            return
        if not self._moved:
            if (sx - self._press[0]) ** 2 + (sy - self._press[1]) ** 2 < DRAG_PX**2:
                return
            self._moved = True
        x, y = self.viewport.screen_to_world(sx, sy)
        self.model.move_axis(self._axis, snap(x), snap(y))
        self.on_change()

    def drag_end(self) -> None:
        if self._axis is not None and not self._moved:
            self.model.toggle_select(self._axis)
            self.on_change()
        elif self._adding:
            x, y = self.viewport.screen_to_world(*self._press)
            self.model.add_axis(snap(x), snap(y))
            self.on_change()
        self._axis = None
        self._adding = False
        self._moved = False
