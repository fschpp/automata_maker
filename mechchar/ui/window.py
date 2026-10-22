from __future__ import annotations

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from mechchar.core.mechanisms import four_bar  # noqa: E402
from mechchar.core.playback import Playback  # noqa: E402
from mechchar.core.viewport import Viewport  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.scene import draw_four_bar, draw_path  # noqa: E402

PERIOD_S = 4.0  # segundos por ciclo


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="Mechanical Characters Designer")
        self.set_default_size(1000, 700)

        # Mecanismo de demostración: cuadrilátero manivela-balancín.
        self.mech = four_bar(d=4.0, r=1.0, l=3.5, rr=3.0)
        cycle = self.mech.asm.trace_cycle(self.mech.state0, n_steps=180)
        self.path = [self.mech.tracer_world(s) for s in cycle.states] if cycle.ok else []
        self.playback = Playback(self.mech.asm, self.mech.state0, period=PERIOD_S)
        self._last_time: int | None = None
        self._updating_scale = False

        self.canvas = Canvas()
        self.canvas.origin_frac = (0.3, 0.5)
        self.canvas.overlay = self._draw_scene

        self.play_button = Gtk.Button(label="Pausa")
        self.play_button.connect("clicked", self._on_play_clicked)
        self.scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.001)
        self.scale.set_hexpand(True)
        self.scale.set_draw_value(False)
        self.scale.connect("value-changed", self._on_scale_changed)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        controls.set_margin_start(8)
        controls.set_margin_end(8)
        controls.set_margin_top(6)
        controls.set_margin_bottom(6)
        controls.append(self.play_button)
        controls.append(self.scale)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.append(self.canvas)
        root.append(controls)
        self.set_child(root)

        self.add_tick_callback(self._on_tick)

    # --- dibujo --------------------------------------------------------
    def _draw_scene(self, cr: Any, vp: Viewport) -> None:
        draw_path(cr, vp, self.path)
        draw_four_bar(cr, vp, self.mech, self.playback.state)

    # --- animación -----------------------------------------------------
    def _on_tick(self, _widget: Gtk.Widget, clock: Any) -> bool:
        now = clock.get_frame_time()  # microsegundos
        if self._last_time is not None and self.playback.playing:
            self.playback.advance((now - self._last_time) / 1e6)
            self._set_scale(self.playback.phase)
            self.canvas.queue_draw()
        self._last_time = now
        return True  # seguir llamando en cada cuadro

    def _set_scale(self, value: float) -> None:
        self._updating_scale = True
        self.scale.set_value(value)
        self._updating_scale = False

    # --- controles -----------------------------------------------------
    def _on_play_clicked(self, _button: Gtk.Button) -> None:
        self.playback.playing = not self.playback.playing
        self.play_button.set_label("Pausa" if self.playback.playing else "Reproducir")

    def _on_scale_changed(self, scale: Gtk.Scale) -> None:
        if self._updating_scale:
            return  # cambio causado por la animación, no por el usuario
        self.playback.playing = False
        self.play_button.set_label("Reproducir")
        self.playback.seek_phase(scale.get_value())
        self.canvas.queue_draw()
