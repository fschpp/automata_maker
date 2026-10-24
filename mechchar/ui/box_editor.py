"""Editor de la caja de engranajes: se dibuja arrastrando un rectángulo sobre el lienzo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mechchar.core.viewport import Viewport
from mechchar.optim.gearbox import Bounds, GearBox

BOX_COLOR = (0.85, 0.45, 0.05)
MIN_SIZE_PX = 8.0  # un arrastre más corto que esto se toma como un clic y no crea una caja


class BoxEditor:
    """Implementa ``DragTarget``: mientras está activo, todo arrastre dibuja una caja nueva."""

    def __init__(self, viewport: Viewport) -> None:
        self.vp = viewport
        self.box: GearBox | None = None
        self.on_change: Callable[[], None] | None = None  # al terminar de dibujar o quitar
        self.on_preview: Callable[[], None] | None = None  # mientras se arrastra
        self._anchor: tuple[float, float] | None = None
        self._anchor_px: tuple[float, float] = (0.0, 0.0)
        self._last_px: tuple[float, float] = (0.0, 0.0)
        self._preview: GearBox | None = None

    # --- DragTarget ------------------------------------------------------------------------
    def drag_begin(self, sx: float, sy: float) -> bool:
        self._anchor = self.vp.screen_to_world(sx, sy)
        self._anchor_px = (sx, sy)
        return True

    def drag_update(self, sx: float, sy: float) -> None:
        if self._anchor is None:
            return
        self._preview = GearBox.from_corners(self._anchor, self.vp.screen_to_world(sx, sy))
        self._last_px = (sx, sy)
        if self.on_preview is not None:
            self.on_preview()

    def drag_end(self) -> None:
        preview, self._preview, self._anchor = self._preview, None, None
        if preview is None:
            return
        dx = abs(self._last_px[0] - self._anchor_px[0])
        dy = abs(self._last_px[1] - self._anchor_px[1])
        if dx < MIN_SIZE_PX or dy < MIN_SIZE_PX:
            return
        self.box = preview
        if self.on_change is not None:
            self.on_change()

    def clear(self) -> None:
        self.box = None
        if self.on_change is not None:
            self.on_change()

    # --- dibujo ----------------------------------------------------------------------------
    def draw(self, cr: Any, vp: Viewport) -> None:
        shown = self._preview or self.box
        if shown is not None:
            draw_box(cr, vp, shown, dashed=self._preview is not None)


def draw_box(cr: Any, vp: Viewport, box: GearBox, dashed: bool = False) -> None:
    x0, y1 = vp.world_to_screen(box.x0, box.y0)
    x1, y0 = vp.world_to_screen(box.x1, box.y1)
    cr.rectangle(x0, y0, x1 - x0, y1 - y0)
    cr.set_source_rgba(*BOX_COLOR, 0.07)
    cr.fill_preserve()
    cr.set_source_rgba(*BOX_COLOR, 0.95)
    cr.set_line_width(2.0)
    cr.set_dash([8.0, 5.0] if dashed else [])
    cr.stroke()
    cr.set_dash([])


def draw_bounds(
    cr: Any, vp: Viewport, bounds: Bounds, color: tuple[float, float, float], ok: bool = True
) -> None:
    """Rectángulo que ocupa un mecanismo en todo su ciclo; punteado y más fino si se sale."""
    x0, y1 = vp.world_to_screen(bounds[0], bounds[1])
    x1, y0 = vp.world_to_screen(bounds[2], bounds[3])
    cr.rectangle(x0, y0, x1 - x0, y1 - y0)
    cr.set_source_rgba(*color, 0.85)
    cr.set_line_width(1.5)
    cr.set_dash([] if ok else [3.0, 4.0])
    cr.stroke()
    cr.set_dash([])
