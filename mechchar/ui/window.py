from __future__ import annotations

from typing import Any

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from mechchar.core.mechanisms import FourBar  # noqa: E402
from mechchar.core.playback import Playback  # noqa: E402
from mechchar.core.viewport import Viewport  # noqa: E402
from mechchar.optim.fourbar_fit import FitResult, fit_four_bar, to_four_bar  # noqa: E402
from mechchar.optim.job import LatestJobRunner  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.scene import draw_four_bar, draw_path  # noqa: E402
from mechchar.ui.spline_editor import SplineEditor  # noqa: E402

PERIOD_S = 4.0  # segundos por ciclo


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="Mechanical Characters Designer")
        self.set_default_size(1000, 700)

        # Mecanismo ajustado (None hasta el primer ajuste).
        self.mech: FourBar | None = None
        self.path: list[Any] = []
        self.playback: Playback | None = None
        self._last_time: int | None = None
        self._updating_scale = False

        self.canvas = Canvas()
        self.canvas.overlay = self._draw_scene
        self.editor = SplineEditor(self.canvas.viewport)
        self.editor.on_change = self._on_spline_changed
        self.canvas.drag_target = self.editor

        self.runner = LatestJobRunner(
            lambda target: fit_four_bar(target),
            lambda _item, result, error: GLib.idle_add(self._on_fit_done, result, error),
        )

        self.fit_button = Gtk.Button(label="Ajustar mecanismo")
        self.fit_button.connect("clicked", self._on_fit_clicked)
        self.play_button = Gtk.Button(label="Pausa")
        self.play_button.connect("clicked", self._on_play_clicked)
        self.play_button.set_sensitive(False)
        self.scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.001)
        self.scale.set_hexpand(True)
        self.scale.set_draw_value(False)
        self.scale.set_sensitive(False)
        self.scale.connect("value-changed", self._on_scale_changed)
        self.status = Gtk.Label(
            label="Arrastrá los puntos azules para dibujar la curva y presioná «Ajustar»."
        )
        self.status.set_xalign(0.0)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        for widget in (self.fit_button, self.play_button):
            controls.append(widget)
        controls.append(self.scale)
        for margin in ("start", "end", "top", "bottom"):
            getattr(controls, f"set_margin_{margin}")(6)
        self.status.set_margin_start(8)
        self.status.set_margin_bottom(6)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.append(self.canvas)
        root.append(controls)
        root.append(self.status)
        self.set_child(root)

        self.add_tick_callback(self._on_tick)

    # --- dibujo --------------------------------------------------------
    def _draw_scene(self, cr: Any, vp: Viewport) -> None:
        self.editor.draw(cr, vp)
        if self.mech is not None and self.playback is not None:
            draw_path(cr, vp, self.path)
            draw_four_bar(cr, vp, self.mech, self.playback.state)

    # --- ajuste --------------------------------------------------------
    def _on_fit_clicked(self, _button: Gtk.Button) -> None:
        if self.runner.busy:
            return
        self.fit_button.set_sensitive(False)
        self.status.set_text("Ajustando mecanismo… (unos segundos)")
        self.runner.submit(self.editor.curve(per_segment=24))

    def _on_fit_done(self, result: FitResult | None, error: BaseException | None) -> bool:
        self.fit_button.set_sensitive(True)
        if error is not None:
            self.status.set_text(f"Error al ajustar: {error!r}")
            return False
        fb = to_four_bar(result.params, result.branch) if result is not None else None
        if result is None or fb is None:
            self.status.set_text("No se encontró un mecanismo válido para esta curva.")
            return False
        self.mech, self.path = fb, list(result.curve)
        self.playback = Playback(fb.asm, fb.state0, period=PERIOD_S)
        self.play_button.set_sensitive(True)
        self.play_button.set_label("Pausa")
        self.scale.set_sensitive(True)
        self._last_time = None
        pct = result.rms_rel * 100
        self.status.set_text(f"Ajuste listo: error {pct:.1f} % del tamaño de la curva.")
        self._zoom_to_fit(fb)
        self.canvas.queue_draw()
        return False  # GLib.SOURCE_REMOVE

    def _zoom_to_fit(self, fb: FourBar) -> None:
        """Encuadra la spline y el mecanismo en todo su ciclo."""
        cycle = fb.asm.trace_cycle(fb.state0, n_steps=24, max_jump=1.0)
        pts = [self.editor.curve()] + [fb.joints(s) for s in cycle.states]
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        self.canvas.viewport.fit_bounds(
            lo[0], lo[1], hi[0], hi[1], self.canvas.get_width(), self.canvas.get_height()
        )

    def _on_spline_changed(self) -> None:
        if self.mech is not None:
            self.status.set_text("Curva modificada: presioná «Ajustar» para recalcular.")

    # --- animación -----------------------------------------------------
    def _on_tick(self, _widget: Gtk.Widget, clock: Any) -> bool:
        now = clock.get_frame_time()  # microsegundos
        pb = self.playback
        if pb is not None and self._last_time is not None and pb.playing:
            pb.advance((now - self._last_time) / 1e6)
            self._set_scale(pb.phase)
            self.canvas.queue_draw()
        self._last_time = now
        return True  # seguir llamando en cada cuadro

    def _set_scale(self, value: float) -> None:
        self._updating_scale = True
        self.scale.set_value(value)
        self._updating_scale = False

    # --- controles -----------------------------------------------------
    def _on_play_clicked(self, _button: Gtk.Button) -> None:
        if self.playback is None:
            return
        self.playback.playing = not self.playback.playing
        self.play_button.set_label("Pausa" if self.playback.playing else "Reproducir")

    def _on_scale_changed(self, scale: Gtk.Scale) -> None:
        if self._updating_scale or self.playback is None:
            return  # cambio causado por la animación, no por el usuario
        self.playback.playing = False
        self.play_button.set_label("Reproducir")
        self.playback.seek_phase(scale.get_value())
        self.canvas.queue_draw()
