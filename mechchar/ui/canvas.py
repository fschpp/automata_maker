"""Canvas Cairo con grilla, zoom (rueda) y pan (arrastrar). Solo presentación."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from mechchar.core.viewport import Viewport, grid_lines  # noqa: E402

ZOOM_STEP = 1.1


class Canvas(Gtk.DrawingArea):
    def __init__(self) -> None:
        super().__init__()
        self.viewport = Viewport()
        self._centered = False
        # Fracción (x, y) de la ventana donde se coloca el origen al abrir.
        self.origin_frac = (0.5, 0.5)
        # Callback opcional ``overlay(cr, viewport)`` que dibuja la escena sobre la grilla.
        self.overlay: Callable[[Any, Viewport], None] | None = None
        self._pointer = (0.0, 0.0)
        self._drag_start = (0.0, 0.0)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_draw_func(self._draw)

        drag = Gtk.GestureDrag()
        drag.set_button(0)  # cualquier botón
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)

        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        self.add_controller(motion)

    # --- eventos -------------------------------------------------------
    def _on_motion(self, _ctrl: Gtk.EventControllerMotion, x: float, y: float) -> None:
        self._pointer = (x, y)

    def _on_drag_begin(self, _g: Gtk.GestureDrag, _x: float, _y: float) -> None:
        self._drag_start = (self.viewport.ox, self.viewport.oy)

    def _on_drag_update(self, _g: Gtk.GestureDrag, dx: float, dy: float) -> None:
        self.viewport.ox = self._drag_start[0] + dx
        self.viewport.oy = self._drag_start[1] + dy
        self.queue_draw()

    def _on_scroll(self, _c: Gtk.EventControllerScroll, _dx: float, dy: float) -> bool:
        factor = ZOOM_STEP ** (-dy)  # dy < 0: rueda hacia arriba = acercar
        self.viewport.zoom_at(*self._pointer, factor)
        self.queue_draw()
        return True

    # --- dibujo --------------------------------------------------------
    def _draw(self, _area: Gtk.DrawingArea, cr, width: int, height: int) -> None:
        vp = self.viewport
        if not self._centered:
            vp.ox, vp.oy = width * self.origin_frac[0], height * self.origin_frac[1]
            self._centered = True

        cr.set_source_rgb(0.97, 0.97, 0.97)
        cr.paint()

        _step, xs, ys = grid_lines(vp, width, height)
        cr.set_line_width(1.0)
        cr.set_source_rgb(0.82, 0.82, 0.82)
        for x in xs:
            sx, _ = vp.world_to_screen(x, 0)
            cr.move_to(round(sx) + 0.5, 0)
            cr.line_to(round(sx) + 0.5, height)
        for y in ys:
            _, sy = vp.world_to_screen(0, y)
            cr.move_to(0, round(sy) + 0.5)
            cr.line_to(width, round(sy) + 0.5)
        cr.stroke()

        # ejes
        ox, oy = vp.world_to_screen(0, 0)
        cr.set_line_width(1.5)
        cr.set_source_rgb(0.85, 0.25, 0.25)  # eje x
        cr.move_to(0, oy)
        cr.line_to(width, oy)
        cr.stroke()
        cr.set_source_rgb(0.25, 0.65, 0.25)  # eje y
        cr.move_to(ox, 0)
        cr.line_to(ox, height)
        cr.stroke()

        if self.overlay is not None:
            self.overlay(cr, vp)
