"""Visor GTK de la red de engranajes (M5): canvas con zoom/pan, slider de giro y botón Girar.

    python scripts/view_gear_train.py

No toca ``window.py``: sirve para ver ``draw_gear_train`` en la ventana real antes de conectarlo
a una escena. Los ejes de ``AXES`` se pueden editar; el tercero gira al revés que la entrada.
"""

from __future__ import annotations

import math
import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from mechchar.core.gearnet import Axis, plan_train  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.scene import draw_gear_train  # noqa: E402

AXES = [Axis("entrada", 0, 0), Axis("b", 9, 1), Axis("c", 4, 8, same_direction=False)]
SECONDS_PER_TURN = 8.0


class Window(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(
            application=app, title="Red de engranajes", default_width=900, default_height=700
        )
        self.train = plan_train(AXES, 0)
        self.theta = 0.0
        self.selected: set[int] = set()
        self.playing = False
        self._last = 0.0

        self.canvas = Canvas()
        self.canvas.viewport.scale = 30.0
        self.canvas.origin_frac = (0.3, 0.65)
        self.canvas.overlay = lambda cr, vp: draw_gear_train(
            cr, vp, self.train, self.theta, self.selected
        )

        self.scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 360.0, 1.0)
        self.scale.set_hexpand(True)
        self.scale.connect("value-changed", self._on_scale)
        self.play = Gtk.ToggleButton(label="Girar")
        self.play.connect("toggled", self._on_play)
        self.status = Gtk.Label(
            label="ok" if self.train.ok else "problemas: " + "; ".join(self.train.problems),
            xalign=0.0,
        )
        sel = Gtk.Box(spacing=6)
        for i, a in enumerate(self.train.axes):
            b = Gtk.CheckButton(label=f"Resaltar {a.name}")
            b.connect("toggled", self._on_select, i)
            sel.append(b)

        bar = Gtk.Box(spacing=8, margin_start=8, margin_end=8, margin_top=4, margin_bottom=4)
        bar.append(self.play)
        bar.append(self.scale)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.append(self.canvas)
        box.append(bar)
        box.append(sel)
        box.append(self.status)
        self.set_child(box)
        self.add_tick_callback(self._tick)

    def _on_scale(self, scale: Gtk.Scale) -> None:
        self.theta = math.radians(scale.get_value())
        self.canvas.queue_draw()

    def _on_play(self, button: Gtk.ToggleButton) -> None:
        self.playing = button.get_active()
        self._last = 0.0

    def _on_select(self, button: Gtk.CheckButton, i: int) -> None:
        (self.selected.add if button.get_active() else self.selected.discard)(i)
        self.canvas.queue_draw()

    def _tick(self, _widget: Gtk.Widget, clock) -> bool:
        now = clock.get_frame_time() / 1e6
        if self.playing and self._last:
            self.theta = (self.theta + 2 * math.pi * (now - self._last) / SECONDS_PER_TURN) % (
                2 * math.pi
            )
            self.scale.handler_block_by_func(self._on_scale)
            self.scale.set_value(math.degrees(self.theta))
            self.scale.handler_unblock_by_func(self._on_scale)
            self.canvas.queue_draw()
        self._last = now
        return GLib.SOURCE_CONTINUE


class App(Gtk.Application):
    def do_activate(self) -> None:
        (self.props.active_window or Window(self)).present()


if __name__ == "__main__":
    sys.exit(App(application_id="io.github.mechchar.geartrain").run(sys.argv))
