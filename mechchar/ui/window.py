from __future__ import annotations

import threading
from typing import Any

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from mechchar.core.character import (  # noqa: E402
    LEG_ATTACH,
    LEG_MARKER,
    biped_character,
    leg_character,
    leg_refs,
)
from mechchar.core.playback import Playback  # noqa: E402
from mechchar.core.rig import MultiRig, SubSpec  # noqa: E402
from mechchar.core.viewport import Viewport  # noqa: E402
from mechchar.db.motion_db import (  # noqa: E402
    MotionDB,
    Suggestion,
    load_or_build,
    make_candidate_source,
    suggest,
)
from mechchar.optim.character_fit import CharacterFamily  # noqa: E402
from mechchar.optim.families import ALL_FAMILIES, family_by_name, fit_best  # noqa: E402
from mechchar.optim.fitcore import FitResult, Throttle, refine_from  # noqa: E402
from mechchar.optim.job import LatestJobRunner  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.scene import (  # noqa: E402
    CANDIDATE_COLORS,
    draw_candidates,
    draw_character,
    draw_mechanism,
    draw_path,
    draw_progress_curve,
)
from mechchar.ui.spline_editor import SplineEditor, default_points  # noqa: E402

PERIOD_S = 4.0  # segundos por ciclo
FAMILY_LABELS = {"four_bar": "cuadrilátero", "geared_five_bar": "5-bar con engranajes"}


SCENES = ("Curva libre", "Pierna de prueba", "Bípedo")
LEG_PREFIXES = ("l_", "r_")
GAIT_PHASES = (0.0, 0.5)  # marcha alternada: una pierna medio ciclo detrás de la otra


def family_label(name: str) -> str:
    """Nombre legible de una familia; ``base+personaje`` se muestra como «base, con personaje»."""
    base, _, extra = name.partition("+")
    label = FAMILY_LABELS.get(base, base)
    return f"{label}, con personaje" if extra else label


def describe(result: FitResult) -> str:
    label = family_label(result.family)
    if result.variant.ratio:
        label += f" (relación {result.variant.ratio:+.0f})"
    return label


COLOR_NAMES = ("rojo", "verde", "azul")


def describe_suggestion(s: Suggestion) -> str:
    label = family_label(s.family)
    if s.variant.ratio:
        label += f" (relación {s.variant.ratio:+.0f})"
    return label


def _hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in color)


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, dbs: list[MotionDB] | None = None) -> None:
        super().__init__(application=app, title="Mechanical Characters Designer")
        self.set_default_size(1000, 700)

        # Mecanismo ajustado (None hasta el primer ajuste).
        self.mech: Any = None
        self.leg_families: tuple[CharacterFamily, ...] = ()  # vacío = escena de curva libre
        self.progress: np.ndarray | None = None  # curva en evaluación mientras se ajusta
        self.path: list[Any] = []
        self.playback: Playback | None = None
        self._last_time: int | None = None
        self._updating_scale = False
        self.dbs: list[MotionDB] = list(dbs) if dbs else []  # una base por familia
        self.suggestions: list[Suggestion] = []  # los mejores candidatos de la base
        self.biped = False  # escena «Bípedo»: el ajuste de una pierna se usa en las dos
        self._updating_phase = False

        self.canvas = Canvas()
        self.canvas.overlay = self._draw_scene
        self.editor = SplineEditor(self.canvas.viewport)
        self.editor.on_change = self._on_spline_changed
        self.canvas.drag_target = self.editor

        self.runner = LatestJobRunner(
            self._fit_work,
            lambda _item, result, error: GLib.idle_add(self._on_fit_done, result, error),
        )
        # Consulta a la base mientras se arrastra: solo importa el último pedido.
        self.query_runner = LatestJobRunner(
            lambda target: suggest(self.dbs, target),
            lambda _item, result, error: GLib.idle_add(self._on_suggestions, result, error),
        )

        self.fit_button = Gtk.Button(label="Ajustar mecanismo")
        self.fit_button.connect("clicked", self._on_fit_clicked)
        self.scene_dropdown = Gtk.DropDown.new_from_strings(list(SCENES))
        self.scene_dropdown.connect("notify::selected", self._on_scene_changed)
        self.candidate_buttons: list[Gtk.Button] = []
        for i, color in enumerate(CANDIDATE_COLORS):
            button = Gtk.Button()
            dot = Gtk.Label()
            dot.set_markup(f'<span foreground="{_hex(color)}" size="large">●</span> {i}')
            button.set_child(dot)
            button.set_sensitive(False)
            button.connect("clicked", self._on_candidate_clicked, i)
            self.candidate_buttons.append(button)
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
        row = (self.scene_dropdown, self.fit_button, *self.candidate_buttons, self.play_button)
        for widget in row:
            controls.append(widget)
        controls.append(self.scale)
        for margin in ("start", "end", "top", "bottom"):
            getattr(controls, f"set_margin_{margin}")(6)
        self.status.set_margin_start(8)
        self.status.set_margin_bottom(6)

        # Control de desfase (un slider por pierna); solo se ve en la escena «Bípedo».
        self.phase_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.phase_scales: list[Gtk.Scale] = []
        for k, prefix in enumerate(LEG_PREFIXES):
            label = Gtk.Label(label=f"Desfase {'izq.' if prefix == 'l_' else 'der.'}")
            scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.005)
            scale.set_hexpand(True)
            scale.set_draw_value(True)
            scale.set_digits(2)
            scale.add_mark(0.5, Gtk.PositionType.BOTTOM, None)
            scale.set_sensitive(False)
            scale.connect("value-changed", self._on_phase_changed, k)
            self.phase_scales.append(scale)
            self.phase_row.append(label)
            self.phase_row.append(scale)
        self.phase_row.set_margin_start(6)
        self.phase_row.set_margin_end(6)
        self.phase_row.set_visible(False)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.append(self.canvas)
        root.append(controls)
        root.append(self.phase_row)
        root.append(self.status)
        self.set_child(root)

        self.add_tick_callback(self._on_tick)

        if self.dbs:
            GLib.idle_add(self._request_suggestions)
        else:
            self.status.set_text("Preparando la base de movimientos (solo la primera vez)…")
            threading.Thread(target=self._load_dbs, daemon=True).start()

    # --- base de movimientos -------------------------------------------
    def _load_dbs(self) -> None:
        """Hilo de trabajo: carga las bases del caché o las construye si faltan."""
        try:
            dbs = [load_or_build(fam.name, workers=1) for fam in ALL_FAMILIES]
        except Exception as exc:  # no impedir el uso de la app si falla la base
            GLib.idle_add(self.status.set_text, f"No se pudo preparar la base: {exc!r}")
            return
        GLib.idle_add(self._on_dbs_ready, dbs)

    def _on_dbs_ready(self, dbs: list[MotionDB]) -> bool:
        self.dbs = dbs
        self._request_suggestions()
        return False

    def _family(self, name: str) -> Any:
        for fam in self.leg_families:
            if fam.name == name:
                return fam
        return family_by_name(name)

    def _request_suggestions(self) -> bool:
        if self.dbs and not self.leg_families:
            self.query_runner.submit(self.editor.curve(per_segment=24))
        return False

    def _on_suggestions(self, result: list[Suggestion] | None, error: BaseException | None) -> bool:
        if error is not None or result is None:
            return False
        self.suggestions = result
        for i, button in enumerate(self.candidate_buttons):
            if i < len(result):
                s = result[i]
                button.set_sensitive(not self.runner.busy)
                tip = f"{describe_suggestion(s)}, error {s.rel_error * 100:.1f} %"
                button.set_tooltip_text(tip)
            else:
                button.set_sensitive(False)
        if result and not self.runner.busy:
            parts = [
                f"{i} {COLOR_NAMES[i]}: {describe_suggestion(s)} ({s.rel_error * 100:.1f} %)"
                for i, s in enumerate(result)
            ]
            self.status.set_text("Candidatos de la base · " + " · ".join(parts))
        self.canvas.queue_draw()
        return False

    def _on_candidate_clicked(self, _button: Gtk.Button, index: int) -> None:
        if self.runner.busy or index >= len(self.suggestions):
            return
        self._start_fit(self.suggestions[index])

    # --- dibujo --------------------------------------------------------
    def _draw_scene(self, cr: Any, vp: Viewport) -> None:
        if self.leg_families and self.mech is None:
            character = self.leg_families[0].character
            rest = character.build()[1]
            draw_character(cr, vp, character, rest, [LEG_ATTACH, LEG_MARKER])
            if self.biped:  # el ajuste se hace sobre una pierna; el bípedo se arma al terminar
                draw_character(cr, vp, biped_character(), biped_character().build()[1])
        self.editor.draw(cr, vp)
        if self.suggestions and self.progress is None:
            draw_candidates(cr, vp, [s.curve for s in self.suggestions])
        if self.mech is not None and self.playback is not None:
            draw_path(cr, vp, self.path)
            draw_mechanism(cr, vp, self.mech, self.playback.state)
        if self.progress is not None:
            draw_progress_curve(cr, vp, self.progress)

    # --- ajuste --------------------------------------------------------
    def _fit_work(self, item: tuple[np.ndarray, Suggestion | None]) -> FitResult | None:
        """Hilo de trabajo: ajuste completo, o refinamiento del candidato elegido."""
        target, chosen = item
        progress = Throttle(self._post_progress)
        if chosen is not None:
            return refine_from(
                family_by_name(chosen.family), chosen.p0, chosen.variant, target,
                on_progress=progress,
            )
        if self.leg_families:  # el personaje entra en la simulación; no usa la base de movimientos
            return fit_best(target, self.leg_families, on_progress=progress)  # type: ignore[arg-type]
        source = make_candidate_source(self.dbs) if self.dbs else None
        return fit_best(target, on_progress=progress, candidate_source=source)

    def _on_fit_clicked(self, _button: Gtk.Button) -> None:
        if not self.runner.busy:
            self._start_fit(None)

    def _start_fit(self, chosen: Suggestion | None) -> None:
        self.fit_button.set_sensitive(False)
        for button in self.candidate_buttons:
            button.set_sensitive(False)
        what = "candidato elegido" if chosen is not None else "mecanismo"
        self.status.set_text(f"Ajustando {what}… (unos segundos)")
        self.progress = None
        self.runner.submit((self.editor.curve(per_segment=24), chosen))

    def _post_progress(self, label: str, curve: np.ndarray) -> None:
        """Llamado desde el hilo de trabajo: pasa la curva en evaluación al hilo principal."""
        GLib.idle_add(self._on_progress, label, curve.copy())

    def _on_progress(self, label: str, curve: np.ndarray) -> bool:
        if self.runner.busy:  # ignora avisos que llegan después de terminar
            self.progress = curve
            for name, nice in FAMILY_LABELS.items():
                label = label.replace(name, nice)
            self.status.set_text(f"Ajustando mecanismo… {label}")
            self.canvas.queue_draw()
        return False

    def _on_fit_done(self, result: FitResult | None, error: BaseException | None) -> bool:
        self.fit_button.set_sensitive(True)
        self.progress = None
        for i, button in enumerate(self.candidate_buttons):
            button.set_sensitive(i < len(self.suggestions))
        if error is not None:
            self.status.set_text(f"Error al ajustar: {error!r}")
            return False
        fb = (
            self._family(result.family).to_mechanism(result.params, result.variant)
            if result is not None
            else None
        )
        if fb is not None and self.biped and result is not None:
            fb = self._build_biped(result)
        if result is None or fb is None:
            self.status.set_text("No se encontró un mecanismo válido para esta curva.")
            return False
        self.mech, self.path = fb, list(result.curve)
        self.suggestions = []  # ya se eligió: se despeja el lienzo hasta la próxima edición
        for button in self.candidate_buttons:
            button.set_sensitive(False)
        self.playback = Playback(fb.asm, fb.state0, period=PERIOD_S)
        if isinstance(fb, MultiRig):
            self._apply_gait(fb)
        self.play_button.set_sensitive(True)
        self.play_button.set_label("Pausa")
        self.scale.set_sensitive(True)
        self._last_time = None
        pct = result.rms_rel * 100
        self.status.set_text(
            f"Ajuste listo: {describe(result)}, error {pct:.1f} % del tamaño de la curva."
        )
        self._zoom_to_fit(fb)
        self.canvas.queue_draw()
        return False  # GLib.SOURCE_REMOVE

    def _build_biped(self, result: FitResult) -> MultiRig | None:
        """Usa el mecanismo ajustado a una pierna en las dos del bípedo (cada una con su propio
        desfase). Los mecanismos son iguales, así que la unión es factible en reposo."""
        base = self._family(result.family).base
        specs = []
        for prefix in LEG_PREFIXES:
            built = base.build(result.params, result.variant)
            mech = base.to_mechanism(result.params, result.variant)
            if built is None or mech is None:
                return None
            attach, marker = leg_refs(prefix)
            specs.append(SubSpec(prefix + "leg", built, mech, attach, marker))
        try:
            return MultiRig(biped_character(), specs)
        except ValueError as exc:
            self.status.set_text(f"No se pudo armar el bípedo: {exc}")
            return None

    def _apply_gait(self, rig: MultiRig) -> None:
        """Marcha alternada de partida: aplica los desfases por defecto en los sliders."""
        assert self.playback is not None
        for k, phase in enumerate(GAIT_PHASES):
            state, ok = rig.set_phase(k, phase, self.playback.state, self.playback.t)
            if ok:
                self.playback.state = state
        self._updating_phase = True
        for k, scale in enumerate(self.phase_scales):
            scale.set_value(rig.phases[k])
            scale.set_sensitive(True)
        self._updating_phase = False

    def _on_phase_changed(self, scale: Gtk.Scale, k: int) -> None:
        pb, rig = self.playback, self.mech
        if self._updating_phase or pb is None or not isinstance(rig, MultiRig):
            return
        state, ok = rig.set_phase(k, scale.get_value(), pb.state, pb.t)
        if ok:
            pb.state = state
        else:
            self.status.set_text("Ese desfase no se puede alcanzar sin saltar de configuración.")
            self._updating_phase = True
            scale.set_value(rig.phases[k])
            self._updating_phase = False
        self.canvas.queue_draw()

    def _zoom_to_fit(self, fb: Any) -> None:
        """Encuadra la spline y el mecanismo en todo su ciclo."""
        cycle = fb.asm.trace_cycle(fb.state0, n_steps=24, max_jump=1.0)
        pts = [self.editor.curve()] + [fb.outline_points(s) for s in cycle.states]
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        self.canvas.viewport.fit_bounds(
            lo[0], lo[1], hi[0], hi[1], self.canvas.get_width(), self.canvas.get_height()
        )

    # --- escenas ---------------------------------------------------------
    def _on_scene_changed(self, dropdown: Gtk.DropDown, _param: Any) -> None:
        if self.runner.busy:
            return
        self.set_scene(dropdown.get_selected())

    def set_scene(self, index: int) -> None:
        """0 = curva libre; 1 = pierna de prueba con el marcador en el pie; 2 = bípedo (el ajuste de
        una pierna se usa en las dos, con un desfase por pierna)."""
        self.mech, self.playback, self.progress = None, None, None
        self.path, self.suggestions = [], []
        self.biped = index == 2
        self.phase_row.set_visible(self.biped)
        for scale in self.phase_scales:
            scale.set_sensitive(False)
        self.play_button.set_sensitive(False)
        self.scale.set_sensitive(False)
        for button in self.candidate_buttons:
            button.set_sensitive(False)
        vp = self.canvas.viewport
        if index in (1, 2):
            leg = leg_character()
            self.leg_families = tuple(
                CharacterFamily(base, leg, LEG_ATTACH, LEG_MARKER)
                for base in ALL_FAMILIES
            )
            self.editor.points = default_points(8, 0.55, 0.25, (0.9, -2.4))
            vp.fit_bounds(-1.5, -3.2, 2.5, 0.6, self.canvas.get_width(), self.canvas.get_height())
            self.status.set_text(
                "Pierna de prueba: arrastrá la curva que debe dibujar la punta del pie "
                "(punto rojo); el naranja es donde se une el mecanismo. Presioná «Ajustar»."
                + (" Después, un slider de desfase por pierna." if index == 2 else "")
            )
        else:
            self.leg_families = ()
            self.editor.points = default_points()
            w, h = self.canvas.get_width(), self.canvas.get_height()
            vp.scale, vp.ox, vp.oy = 50.0, w * 0.3, h / 2
            self.status.set_text("Arrastrá los puntos azules y presioná «Ajustar».")
            self._request_suggestions()
        self.canvas.queue_draw()

    def _on_spline_changed(self) -> None:
        if self.leg_families:
            return
        if self.dbs and not self.runner.busy:
            self.query_runner.submit(self.editor.curve(per_segment=24))
        elif self.mech is not None:
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
