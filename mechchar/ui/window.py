from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from mechchar.core.actuation import (  # noqa: E402
    ActuationAdvice,
    advise,
    point_mobility,
    suggest_attach_points,
)
from mechchar.core.character import (  # noqa: E402
    LEG_SPEC,
    PUSHING_MAN_SPEC,
    CharacterSpec,
    PartOverride,
    PointRef,
    apply_overrides,
    biped_character,
    leg_refs,
    merge_override,
    prefix_pose,
)
from mechchar.core.gearnet import Axis, describe_train, input_axis, plan_train  # noqa: E402
from mechchar.core.kinematics import ReachReport, reach_report  # noqa: E402
from mechchar.core.playback import Playback  # noqa: E402
from mechchar.core.project import (  # noqa: E402
    FitRecord,
    History,
    PointPick,
    ProjectState,
    load_project,
    save_project,
)
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
from mechchar.optim.curves import resample_closed  # noqa: E402
from mechchar.optim.families import ALL_FAMILIES, family_by_name, fit_best  # noqa: E402
from mechchar.optim.fitcore import (  # noqa: E402
    FitResult,
    Throttle,
    Variant,
    refine_from,
    trace_curve_generic,
)
from mechchar.optim.gearbox import GearBox, box_families, mechanism_bounds  # noqa: E402
from mechchar.optim.job import LatestJobRunner  # noqa: E402
from mechchar.ui.box_editor import BoxEditor, draw_bounds  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.point_picker import PointPicker  # noqa: E402
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


SCENES = ("Curva libre", "Pierna de prueba", "Bípedo", "Pushing Man")
SCENE_SPECS = {1: LEG_SPEC, 2: LEG_SPEC, 3: PUSHING_MAN_SPEC}  # personaje de cada escena
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
        self.box: GearBox | None = None  # caja de engranajes (None = sin límite)
        self._fit_box: GearBox | None = None
        self.fit_result: FitResult | None = None  # el ajuste que se está mostrando
        self._restoring = False  # mientras se restaura una foto no se registran cambios
        self.current_file: str | None = None
        self.mech_bounds: tuple[float, float, float, float] | None = None
        # Puntos elegidos sobre el personaje (escenas con pierna) y su diagnóstico.
        self.spec: CharacterSpec = LEG_SPEC  # el personaje de la escena actual
        self.attach_ref: PointRef = LEG_SPEC.attach
        self.marker_ref: PointRef = LEG_SPEC.marker
        self.advice: ActuationAdvice | None = None
        # Cambios del usuario sobre las piezas (z y fijas) de la pierna sola; el bípedo los hereda.
        self.part_overrides: tuple[PartOverride, ...] = ()
        self._updating_part_controls = False
        self._suggest_i = 0
        self._updating_mode = False
        self.active_leg: int | None = None  # mecanismo cuyo slider de desfase se movió último

        self.canvas = Canvas()
        self.canvas.overlay = self._draw_scene
        self.editor = SplineEditor(self.canvas.viewport)
        self.editor.on_change = self._on_spline_changed
        self.editor.on_commit = lambda: self._commit()  # _commit también revisa el alcance
        self.canvas.drag_target = self.editor
        self.box_editor = BoxEditor(self.canvas.viewport)
        self.box_editor.on_preview = lambda: self.canvas.queue_draw()
        self.box_editor.on_change = self._on_box_changed
        self.picker = PointPicker(self.canvas.viewport)
        self.picker.on_pick = self._on_point_picked
        self.picker.on_miss = self._on_pick_miss

        self.runner = LatestJobRunner(
            self._fit_work,
            lambda _item, result, error: GLib.idle_add(self._on_fit_done, result, error),
        )
        # Consulta a la base mientras se arrastra: solo importa el último pedido.
        self.query_runner = LatestJobRunner(
            lambda item: suggest(self.dbs, item[0], box=item[1]),
            lambda _item, result, error: GLib.idle_add(self._on_suggestions, result, error),
        )

        # Alcance de la curva con la pierna (cinemática inversa, ~1 s): en un hilo, solo el último.
        self.reach_ok = True
        self.reach_runner = LatestJobRunner(
            lambda item: reach_report(*item),
            lambda _item, result, error: GLib.idle_add(self._on_reach, result, error),
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
        self.box_toggle = Gtk.ToggleButton(label="Caja")
        self.box_toggle.set_tooltip_text(
            "Caja de engranajes: activala y arrastrá en el lienzo para dibujar el rectángulo "
            "donde tiene que caber el mecanismo. Después, arrastrá desde adentro para moverla "
            "y desde un borde o una esquina para cambiarle el tamaño."
        )
        self.box_toggle.connect("toggled", self._on_mode_toggled, "box")
        self.attach_toggle = Gtk.ToggleButton(label="Actuación")
        self.attach_toggle.set_tooltip_text(
            "Elegir el punto de actuación: activalo y hacé clic sobre una pieza del personaje "
            "(ahí se une el mecanismo)."
        )
        self.marker_toggle = Gtk.ToggleButton(label="Marcador")
        self.marker_toggle.set_tooltip_text(
            "Elegir el marcador: activalo y hacé clic sobre el punto del personaje cuya curva "
            "querés controlar."
        )
        self.suggest_button = Gtk.Button(label="Sugerir")
        self.suggest_button.set_tooltip_text(
            "Sugerir un punto de actuación para el marcador actual (repetí para ver el siguiente)"
        )
        self.attach_toggle.connect("toggled", self._on_mode_toggled, "attach")
        self.marker_toggle.connect("toggled", self._on_mode_toggled, "marker")
        self.suggest_button.connect("clicked", self._on_suggest_clicked)
        self.pick_controls = (self.attach_toggle, self.marker_toggle, self.suggest_button)
        for widget in self.pick_controls:
            widget.set_sensitive(False)
        self.box_clear = Gtk.Button(label="✕")
        self.box_clear.set_tooltip_text("Quitar la caja de engranajes")
        self.box_clear.set_sensitive(False)
        self.box_clear.connect("clicked", lambda _b: self.box_editor.clear())
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
        row = (
            self.scene_dropdown, self.fit_button, self.box_toggle, self.box_clear,
            *self.pick_controls, *self.candidate_buttons, self.play_button,
        )
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

        # Panel de propiedades (solo en las escenas con personaje).
        self.props = Gtk.Label()
        self.props.set_xalign(0.0)
        self.props.set_yalign(0.0)
        self.props.set_wrap(True)
        self.props.set_selectable(True)
        self.props.set_width_chars(30)
        self.props.set_max_width_chars(34)
        for margin in ("start", "end", "top", "bottom"):
            getattr(self.props, f"set_margin_{margin}")(8)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_min_content_width(280)
        scroller.set_child(self.props)
        scroller.set_vexpand(True)

        # Edición de una pieza: plano de profundidad z y si es fija (chasis).
        self.part_names = [part.name for part in LEG_SPEC.factory(()).parts]
        self.part_dropdown = Gtk.DropDown.new_from_strings(self.part_names)
        self.part_z = Gtk.SpinButton.new_with_range(-10.0, 10.0, 0.5)
        self.part_z.set_digits(1)
        self.part_z.set_tooltip_text(
            "Plano de profundidad z de la pieza (la simulación sigue siendo 2D por plano)"
        )
        self.part_fixed = Gtk.CheckButton(label="Fija (chasis)")
        self.part_fixed.set_tooltip_text(
            "Una pieza fija no se mueve: el mecanismo no puede accionarla"
        )
        self.part_dropdown.connect("notify::selected", lambda *_a: self._sync_part_controls())
        self.part_z.connect("value-changed", self._on_part_z_changed)
        self.part_fixed.connect("toggled", self._on_part_fixed_toggled)
        self.part_controls = (self.part_dropdown, self.part_z, self.part_fixed)
        for widget in self.part_controls:
            widget.set_sensitive(False)
        editor_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        editor_row.append(Gtk.Label(label="Pieza"))
        editor_row.append(self.part_dropdown)
        editor_row.append(Gtk.Label(label="z"))
        editor_row.append(self.part_z)
        part_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        part_box.append(editor_row)
        part_box.append(self.part_fixed)
        for margin in ("start", "end", "top", "bottom"):
            getattr(part_box, f"set_margin_{margin}")(8)
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        side.append(part_box)
        side.append(scroller)
        self.props_frame = Gtk.Frame(label="Propiedades")
        self.props_frame.set_child(side)
        self.props_frame.set_visible(False)
        middle = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        middle.append(self.canvas)
        middle.append(self.props_frame)
        middle.set_vexpand(True)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.append(middle)
        root.append(controls)
        root.append(self.phase_row)
        root.append(self.status)
        self.set_child(root)

        # Barra de título con proyecto (abrir / guardar) y deshacer / rehacer.
        header = Gtk.HeaderBar()
        self.open_button = Gtk.Button(label="Abrir")
        self.open_button.set_tooltip_text("Abrir un proyecto (Ctrl+O)")
        self.open_button.connect("clicked", lambda _b: self._on_open())
        self.save_button = Gtk.Button(label="Guardar")
        self.save_button.set_tooltip_text("Guardar el proyecto (Ctrl+S)")
        self.save_button.connect("clicked", lambda _b: self._on_save())
        self.undo_button = Gtk.Button(label="↶")
        self.undo_button.set_tooltip_text("Deshacer (Ctrl+Z)")
        self.undo_button.connect("clicked", lambda _b: self._undo())
        self.redo_button = Gtk.Button(label="↷")
        self.redo_button.set_tooltip_text("Rehacer (Ctrl+Mayús+Z o Ctrl+Y)")
        self.redo_button.connect("clicked", lambda _b: self._redo())
        for widget in (self.open_button, self.save_button, self.undo_button, self.redo_button):
            header.pack_start(widget)
        self.set_titlebar(header)
        shortcuts = Gtk.ShortcutController()
        shortcuts.set_scope(Gtk.ShortcutScope.GLOBAL)
        for keys, action in (
            ("<Control>z", self._undo), ("<Control><Shift>z", self._redo),
            ("<Control>y", self._redo), ("<Control>s", self._on_save),
            ("<Control>o", self._on_open),
        ):
            shortcuts.add_shortcut(
                Gtk.Shortcut.new(
                    Gtk.ShortcutTrigger.parse_string(keys),
                    Gtk.CallbackAction.new(lambda _w, _a, fn=action: bool(fn() or True)),
                )
            )
        self.add_controller(shortcuts)
        self.history: History[ProjectState] = History(self._snapshot())
        self._update_history_buttons()

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
            self.query_runner.submit((self.editor.curve(per_segment=24), self.box))
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
            draw_character(cr, vp, character, rest, [self.attach_ref, self.marker_ref])
            if self.biped:  # el ajuste se hace sobre una pierna; el bípedo se arma al terminar
                biped = biped_character(self.part_overrides)
                draw_character(cr, vp, biped, biped.build()[1])
        self.editor.draw(cr, vp)
        self.box_editor.draw(cr, vp)
        if self.suggestions and self.progress is None:
            draw_candidates(cr, vp, [s.curve for s in self.suggestions])
            for i, s in enumerate(self.suggestions[: len(CANDIDATE_COLORS)]):
                if s.bounds is not None:  # el rectángulo de cada candidato, como en el video
                    draw_bounds(cr, vp, s.bounds, CANDIDATE_COLORS[i])
        if self.mech is not None and self.playback is not None:
            draw_path(cr, vp, self.path)
            draw_mechanism(cr, vp, self.mech, self.playback.state, self.active_leg)
            if self.mech_bounds is not None:
                inside = self.box is None or self.box.contains(self.mech_bounds, self.box.slack)
                draw_bounds(cr, vp, self.mech_bounds, (0.1, 0.55, 0.2), inside)
        if self.progress is not None:
            draw_progress_curve(cr, vp, self.progress)

    # --- ajuste --------------------------------------------------------
    def _fit_work(
        self, item: tuple[np.ndarray, Suggestion | None, GearBox | None]
    ) -> FitResult | None:
        """Hilo de trabajo: ajuste completo, o refinamiento del candidato elegido."""
        target, chosen, box = item
        progress = Throttle(self._post_progress)
        if chosen is not None:
            fam = box_families((family_by_name(chosen.family),), box)[0]
            return refine_from(fam, chosen.p0, chosen.variant, target, on_progress=progress)
        if self.leg_families:  # el personaje entra en la simulación; no usa la base de movimientos
            legs = box_families(self.leg_families, box)
            return fit_best(target, legs, on_progress=progress)  # type: ignore[arg-type]
        source = make_candidate_source(self.dbs) if self.dbs else None
        families = box_families(ALL_FAMILIES, box)
        return fit_best(target, families, on_progress=progress, candidate_source=source)

    def _on_fit_clicked(self, _button: Gtk.Button) -> None:
        if not self.runner.busy:
            self._start_fit(None)

    def _start_fit(self, chosen: Suggestion | None) -> None:
        if self.leg_families and self.advice is not None and not self.advice.ok:
            self.status.set_text("No se puede ajustar: " + self.advice.text)
            return
        self.fit_button.set_sensitive(False)
        self._update_pick_controls(busy=True)
        for button in self.candidate_buttons:
            button.set_sensitive(False)
        what = "candidato elegido" if chosen is not None else "mecanismo"
        self.status.set_text(f"Ajustando {what}… (unos segundos)")
        self.progress = None
        self._fit_box = self.box  # la caja con la que se pidió; dibujarla de nuevo no la cambia
        self.runner.submit((self.editor.curve(per_segment=24), chosen, self.box))

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
        self._update_pick_controls(busy=False)
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
            if self._fit_box is not None:
                self.status.set_text(
                    "No se encontró un mecanismo que quepa en la caja de engranajes: "
                    "agrandala o achicá la curva."
                )
            else:
                self.status.set_text("No se encontró un mecanismo válido para esta curva.")
            return False
        self._show_result(result, fb)
        pct = result.rms_rel * 100
        note = ""
        box = self._fit_box
        if box is not None and self.mech_bounds is not None:
            out = box.worst_overshoot(self.mech_bounds)
            note = (
                " Cabe en la caja." if out <= box.slack
                else f" Se sale de la caja por {out:.2f}: probá con una caja más grande."
            )
        self.status.set_text(
            f"Ajuste listo: {describe(result)}, error {pct:.1f} % del tamaño de la curva.{note}"
        )
        self._zoom_to_fit(fb)
        self._refresh_properties()
        self._commit()
        self.canvas.queue_draw()
        return False  # GLib.SOURCE_REMOVE

    def _show_result(
        self, result: FitResult, fb: Any, phases: Sequence[float] = GAIT_PHASES
    ) -> None:
        """Muestra un mecanismo ajustado y deja la reproducción lista."""
        self.mech, self.path, self.fit_result = fb, list(result.curve), result
        self.suggestions = []  # ya se eligió: se despeja el lienzo hasta la próxima edición
        for button in self.candidate_buttons:
            button.set_sensitive(False)
        self.playback = Playback(fb.asm, fb.state0, period=PERIOD_S)
        if isinstance(fb, MultiRig):
            self._apply_gait(fb, phases)
        self.play_button.set_sensitive(True)
        self.play_button.set_label("Pausa")
        self.scale.set_sensitive(True)
        self._last_time = None
        fam = self._family(result.family)
        self.mech_bounds = mechanism_bounds(fam, result.params, result.variant)

    def _build_biped(self, result: FitResult) -> MultiRig | None:
        """Usa el mecanismo ajustado a una pierna en las dos del bípedo (cada una con su propio
        desfase). Los mecanismos son iguales, así que la unión es factible en reposo."""
        fam = self._family(result.family)
        base = fam.base
        # la pose de cada pierna en t = 0 sale de la IK: sin ella el solver podría caer en otra
        # rama del personaje sin avisar (las piernas trazarían otra curva que la ajustada)
        pose = fam.pose_for(result.params, result.variant)
        specs = []
        for prefix in LEG_PREFIXES:
            built = base.build(result.params, result.variant)
            mech = base.to_mechanism(result.params, result.variant)
            if built is None or mech is None:
                return None
            attach, marker = leg_refs(prefix, self.attach_ref, self.marker_ref)
            leg_pose = prefix_pose(pose, prefix) if pose is not None else None
            specs.append(SubSpec(prefix + "leg", built, mech, attach, marker, leg_pose))
        try:
            return MultiRig(biped_character(self.part_overrides), specs)
        except ValueError as exc:
            self.status.set_text(f"No se pudo armar el bípedo: {exc}")
            return None

    def _apply_gait(self, rig: MultiRig, phases: Sequence[float] = GAIT_PHASES) -> None:
        """Aplica los desfases (por defecto, la marcha alternada) y los pone en los sliders."""
        assert self.playback is not None
        for k, phase in enumerate(phases[: len(rig.subs)]):
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
        self.active_leg = k  # el resto se atenúa para ver qué pierna controla el slider
        self._refresh_properties()
        self._commit(key=f"fase{k}")
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
        if self.runner.busy or self._restoring:
            return
        self.set_scene(dropdown.get_selected())
        self._commit()

    def _clear_result(self) -> None:
        """Quita el mecanismo mostrado (y su reproducción) sin tocar la curva ni la caja."""
        self.mech, self.playback, self.progress, self.fit_result = None, None, None, None
        self.active_leg = None
        self.path, self.suggestions = [], []
        self.mech_bounds = None
        for scale in self.phase_scales:
            scale.set_sensitive(False)
        self.play_button.set_sensitive(False)
        self.scale.set_sensitive(False)
        for button in self.candidate_buttons:
            button.set_sensitive(False)

    def set_scene(self, index: int) -> None:
        """0 = curva libre; 1 = pierna de prueba con el marcador en el pie; 2 = bípedo (el ajuste de
        una pierna se usa en las dos, con un desfase por pierna); 3 = Pushing Man (el marcador en
        la mano)."""
        self._clear_result()
        self.biped = index == 2
        self.phase_row.set_visible(self.biped)
        vp = self.canvas.viewport
        self.spec = SCENE_SPECS.get(index, LEG_SPEC)
        self.attach_ref, self.marker_ref, self._suggest_i = self.spec.attach, self.spec.marker, 0
        self.part_overrides = ()
        if index in SCENE_SPECS:
            self._rebuild_leg_families()
            cx, cy, ea, eb = self.spec.ellipse
            self.editor.points = default_points(8, ea, eb, (cx, cy))
            x0, y0, x1, y1 = self.spec.view
            vp.fit_bounds(x0, y0, x1, y1, self.canvas.get_width(), self.canvas.get_height())
            self.status.set_text(
                self.spec.intro
                + (" Después, un slider de desfase por pierna." if index == 2 else "")
            )
        else:
            self.leg_families = ()
            self.picker.character, self.advice = None, None
            self.editor.points = default_points()
            w, h = self.canvas.get_width(), self.canvas.get_height()
            vp.scale, vp.ox, vp.oy = 50.0, w * 0.3, h / 2
            self.status.set_text("Arrastrá los puntos azules y presioná «Ajustar».")
            self._request_suggestions()
        self._update_pick_controls(busy=False)
        self._refresh_properties()
        self.canvas.queue_draw()
        self._check_reach()

    def _on_spline_changed(self) -> None:
        if self.leg_families:
            return
        if self.dbs and not self.runner.busy:
            self.query_runner.submit((self.editor.curve(per_segment=24), self.box))
        elif self.mech is not None:
            self.status.set_text("Curva modificada: presioná «Ajustar» para recalcular.")

    # --- proyecto e historial --------------------------------------------
    def _snapshot(self) -> ProjectState:
        """Foto de todo lo que el usuario edita."""
        fit = None
        if self.mech is not None and self.fit_result is not None:
            r = self.fit_result
            fit = FitRecord(
                r.family, tuple(float(v) for v in r.params), r.variant.branch, r.variant.ratio,
                float(r.rms_rel),
            )
        phases = tuple(self.mech.phases) if isinstance(self.mech, MultiRig) else ()
        b = self.box
        attach = marker = None
        if self.leg_families:
            attach = PointPick(self.attach_ref.part, *self.attach_ref.point)
            marker = PointPick(self.marker_ref.part, *self.marker_ref.point)
        return ProjectState(
            int(self.scene_dropdown.get_selected()),
            tuple((float(x), float(y)) for x, y in self.editor.points),
            None if b is None else (b.x0, b.y0, b.x1, b.y1),
            fit,
            phases,
            attach,
            marker,
            self.part_overrides if self.leg_families else (),
        )

    def _commit(self, key: str | None = None) -> None:
        if self._restoring:
            return
        self.history.commit(self._snapshot(), key)
        self._update_history_buttons()
        self._check_reach()

    def _check_reach(self) -> None:
        """Pide en segundo plano cuánto de la curva objetivo alcanza el marcador de la pierna."""
        if not self.leg_families or self._restoring:
            return
        curve = resample_closed(self.editor.curve(per_segment=24), 32)
        self.reach_runner.submit((self.leg_families[0].character, self.marker_ref, curve))

    def _on_reach(self, report: ReachReport | None, error: BaseException | None) -> bool:
        if error is not None or report is None or not self.leg_families or self.runner.busy:
            return False
        if not report.ok:
            self.status.set_text(
                f"Atención: {report.n_outside} de {report.n_points} puntos de la curva quedan "
                f"fuera del alcance de {self.spec.who} (al peor le faltan {report.worst:.2f}). El "
                f"ajuste no podrá recorrerlos: acercá la curva a {self.spec.reach_zone}."
            )
        elif not self.reach_ok:
            self.status.set_text(
                f"La curva está al alcance de {self.spec.who}. Presioná «Ajustar»."
            )
        self.reach_ok = report.ok
        return False

    def _update_history_buttons(self) -> None:
        self.undo_button.set_sensitive(self.history.can_undo)
        self.redo_button.set_sensitive(self.history.can_redo)

    def _undo(self) -> None:
        self._step_history(self.history.undo)

    def _redo(self) -> None:
        self._step_history(self.history.redo)

    def _step_history(self, step: Any) -> None:
        if self.runner.busy:
            self.status.set_text("Esperá a que termine el ajuste para deshacer o rehacer.")
            return
        state = step()
        if state is not None:
            self._restore(state)
        self._update_history_buttons()

    def _restore(self, state: ProjectState, zoom: bool = False) -> None:
        """Lleva la ventana a la foto ``state`` (sin registrar un cambio nuevo)."""
        self._restoring = True
        try:
            if state.scene != self.scene_dropdown.get_selected():
                self.scene_dropdown.set_selected(state.scene)
                self.set_scene(state.scene)
            else:
                self._clear_result()
            if self.leg_families:
                self.part_overrides = self._valid_parts(state.parts)
                self._apply_picks(state.attach, state.marker)
            self.editor.points = list(state.spline)
            b = state.box
            self.box_editor.box = None if b is None else GearBox(*b)
            self.box = self.box_editor.box
            self.box_clear.set_sensitive(self.box is not None)
            if state.fit is None:
                self.status.set_text("Estado restaurado.")
                self._request_suggestions()
            else:
                self._restore_fit(state, zoom)
        finally:
            self._restoring = False
        self._refresh_properties()
        self.canvas.queue_draw()
        self._check_reach()

    def _restore_fit(self, state: ProjectState, zoom: bool) -> None:
        """Vuelve a armar el mecanismo guardado desde sus parámetros."""
        rec = state.fit
        assert rec is not None
        try:
            fam = self._family(rec.family)
        except KeyError:
            self.status.set_text(f"El proyecto usa una familia desconocida: {rec.family}.")
            return
        variant = Variant(rec.branch, rec.ratio)
        params = np.array(rec.params, dtype=float)
        if len(params) != fam.n_params:
            self.status.set_text("Los parámetros guardados no corresponden a esa familia.")
            return
        if self.leg_families:  # el banco de poses de la IK sale de la curva objetivo guardada
            fam.prepare(resample_closed(self.editor.curve(per_segment=24), 48))
        curve = trace_curve_generic(lambda p: fam.build(p, variant), params, 180)
        fb = fam.to_mechanism(params, variant)
        if curve is None or fb is None:
            self.status.set_text("No se pudo reconstruir el mecanismo guardado.")
            return
        result = FitResult(rec.family, params, variant, 0.0, rec.rms_rel, curve, rec.rms_rel)
        if self.biped:
            fb = self._build_biped(result)
            if fb is None:
                return
        self._show_result(result, fb, state.phases or GAIT_PHASES)
        self.status.set_text(
            f"Estado restaurado: {describe(result)}, error {rec.rms_rel * 100:.1f} %."
        )
        if zoom:
            self._zoom_to_fit(fb)

    def _on_save(self) -> None:
        dialog = Gtk.FileDialog()
        dialog.set_title("Guardar proyecto")
        dialog.set_initial_name(
            os.path.basename(self.current_file) if self.current_file else "proyecto.json"
        )
        dialog.save(self, None, self._on_save_chosen)

    def _on_save_chosen(self, dialog: Gtk.FileDialog, result: Any) -> None:
        try:
            path = dialog.save_finish(result).get_path()
        except GLib.Error:
            return  # canceló
        self.save_to(path)

    def save_to(self, path: str) -> bool:
        try:
            save_project(path, self._snapshot())
        except OSError as exc:
            self.status.set_text(f"No se pudo guardar: {exc.strerror or exc}")
            return False
        self.current_file = path
        self.set_title(f"Mechanical Characters Designer — {os.path.basename(path)}")
        self.status.set_text(f"Proyecto guardado en {path}.")
        return True

    def _on_open(self) -> None:
        if self.runner.busy:
            self.status.set_text("Esperá a que termine el ajuste para abrir un proyecto.")
            return
        dialog = Gtk.FileDialog()
        dialog.set_title("Abrir proyecto")
        dialog.open(self, None, self._on_open_chosen)

    def _on_open_chosen(self, dialog: Gtk.FileDialog, result: Any) -> None:
        try:
            path = dialog.open_finish(result).get_path()
        except GLib.Error:
            return  # canceló
        self.open_from(path)

    def open_from(self, path: str) -> bool:
        """Abre un proyecto; si el archivo no sirve deja todo como estaba y lo avisa."""
        try:
            state = load_project(path)
        except (OSError, ValueError) as exc:
            self.status.set_text(f"No se pudo abrir el proyecto: {exc}")
            return False
        self._restore(state, zoom=True)
        self.history.reset(self._snapshot())
        self._update_history_buttons()
        self.current_file = path
        self.set_title(f"Mechanical Characters Designer — {os.path.basename(path)}")
        return True

    def _on_mode_toggled(self, button: Gtk.ToggleButton, mode: str) -> None:
        """Modo del lienzo: con «Caja» arrastrar dibuja la caja; con «Actuación» o «Marcador» un
        clic elige un punto del personaje; sin ninguno, arrastrar edita la curva (excluyentes)."""
        if self._updating_mode:
            return
        toggles = {
            "box": self.box_toggle, "attach": self.attach_toggle, "marker": self.marker_toggle
        }
        if button.get_active():
            self._updating_mode = True
            for other, toggle in toggles.items():
                if other != mode:
                    toggle.set_active(False)
            self._updating_mode = False
            if mode == "box":
                self.canvas.drag_target = self.box_editor
            else:
                self.picker.mode = mode
                self.canvas.drag_target = self.picker
        elif not any(t.get_active() for t in toggles.values()):
            self.canvas.drag_target = self.editor
        self.box_editor.active = self.box_toggle.get_active()  # con la caja activa van las manijas
        self.canvas.queue_draw()

    # --- puntos de actuación y marcador --------------------------------------
    def _update_pick_controls(self, busy: bool) -> None:
        enabled = bool(self.leg_families) and not busy
        for widget in (*self.pick_controls, *self.part_controls):
            widget.set_sensitive(enabled)
        if not self.leg_families:  # sin personaje no hay nada que elegir
            for toggle in (self.attach_toggle, self.marker_toggle):
                toggle.set_active(False)

    def _leg(self) -> Any:
        """El personaje de la escena con los cambios del usuario sobre sus piezas."""
        return self.spec.factory(self.part_overrides)

    def _rebuild_leg_families(self) -> None:
        """Arma las familias con personaje para los puntos elegidos y rehace el diagnóstico."""
        leg = self._leg()
        self.leg_families = tuple(
            CharacterFamily(base, leg, self.attach_ref, self.marker_ref) for base in ALL_FAMILIES
        )
        self.picker.character = leg
        self._set_part_names([part.name for part in leg.parts])
        self.advice = advise(leg, self.attach_ref, self.marker_ref)
        self._refresh_properties()

    def set_points(self, attach: PointRef | None = None, marker: PointRef | None = None) -> None:
        """Cambia el punto de actuación y/o el marcador. El mecanismo ajustado deja de valer y se
        quita; si se mueve el marcador, la curva objetivo lo acompaña (se traslada igual)."""
        if not self.leg_families:
            return
        new_attach, new_marker = attach or self.attach_ref, marker or self.marker_ref
        if new_attach == self.attach_ref and new_marker == self.marker_ref:
            return
        if new_marker != self.marker_ref:
            dx = new_marker.point[0] - self.marker_ref.point[0]
            dy = new_marker.point[1] - self.marker_ref.point[1]
            self.editor.points = [(x + dx, y + dy) for x, y in self.editor.points]
            self._suggest_i = 0
        self.attach_ref, self.marker_ref = new_attach, new_marker
        self._clear_result()
        self._rebuild_leg_families()
        assert self.advice is not None
        self.status.set_text(
            f"Actuación en «{new_attach.part}», marcador en «{new_marker.part}». "
            + self.advice.text
        )
        self._commit()
        self.canvas.queue_draw()

    def _on_pick_miss(self) -> None:
        self.status.set_text("Hacé clic sobre una pieza del personaje (o cerca de ella).")

    def _on_point_picked(self, ref: PointRef, mode: str) -> None:
        if self.runner.busy:
            self.status.set_text("Esperá a que termine el ajuste para cambiar los puntos.")
            return
        if mode == "attach":
            self.set_points(attach=ref)
        else:
            self.set_points(marker=ref)

    def _on_suggest_clicked(self, _button: Gtk.Button) -> None:
        if self.runner.busy or self.picker.character is None:
            return
        found = suggest_attach_points(self.picker.character, self.marker_ref, limit=3)
        if not found:
            self.status.set_text("No encontré un punto de actuación que se mueva en el plano.")
            return
        ref, score = found[self._suggest_i % len(found)]
        self._suggest_i += 1
        self.set_points(attach=ref)
        self.status.set_text(
            f"Sugerido {self._suggest_i}: actuación en «{ref.part}» (alcance {score:.2f}; es una "
            "pista de primer orden, no un pronóstico del error). «Sugerir» de nuevo muestra otro."
        )

    def _set_part_names(self, names: list[str]) -> None:
        """Pone en el selector las piezas del personaje de la escena (cambian entre escenas)."""
        if names == self.part_names:
            return
        self.part_names = names
        self._updating_part_controls = True  # el modelo nuevo no debe disparar cambios
        try:
            self.part_dropdown.set_model(Gtk.StringList.new(names))
            self.part_dropdown.set_selected(0)
        finally:
            self._updating_part_controls = False

    def _apply_picks(self, attach: PointPick | None, marker: PointPick | None) -> None:
        """Pone los puntos de una foto (``None`` = los de la escena); si una pieza ya no existe,
        usa el punto por defecto y lo avisa."""
        leg = self._leg()
        refs: list[PointRef] = []
        for pick, default in ((attach, self.spec.attach), (marker, self.spec.marker)):
            if pick is None:
                refs.append(default)
                continue
            try:
                leg.index(pick.part)
            except KeyError:
                self.status.set_text(f"El proyecto usa una pieza desconocida: {pick.part}.")
                refs.append(default)
                continue
            refs.append(PointRef(pick.part, (pick.x, pick.y)))
        self.attach_ref, self.marker_ref = refs
        self._rebuild_leg_families()

    def _valid_parts(self, parts: tuple[PartOverride, ...]) -> tuple[PartOverride, ...]:
        """Los cambios de una foto, o ninguno (con aviso) si nombran una pieza que no existe."""
        try:
            apply_overrides(self.spec.factory(()), parts)
        except ValueError as exc:
            self.status.set_text(f"Se ignoraron los cambios de piezas del proyecto: {exc}.")
            return ()
        return parts

    def _selected_part(self) -> str:
        return self.part_names[self.part_dropdown.get_selected()]

    def _sync_part_controls(self) -> None:
        """Muestra en los controles los valores de la pieza elegida (sin disparar cambios)."""
        if not self.leg_families:
            return
        leg = self.picker.character
        if leg is None:
            return
        part = leg.parts[leg.index(self._selected_part())]
        self._updating_part_controls = True
        try:
            self.part_z.set_value(part.z)
            self.part_fixed.set_active(part.fixed)
        finally:
            self._updating_part_controls = False

    def _on_part_z_changed(self, spin: Gtk.SpinButton) -> None:
        if not self._updating_part_controls:
            self.set_part_props(self._selected_part(), z=float(spin.get_value()))

    def _on_part_fixed_toggled(self, check: Gtk.CheckButton) -> None:
        if not self._updating_part_controls:
            self.set_part_props(self._selected_part(), fixed=bool(check.get_active()))

    def set_part_props(
        self, name: str, z: float | None = None, fixed: bool | None = None
    ) -> None:
        """Cambia el plano z y/o si una pieza es fija. El mecanismo ajustado deja de valer (el
        personaje cambió) y se quita; el diagnóstico de grados de libertad se rehace."""
        leg = self.picker.character
        if not self.leg_families or leg is None:
            return
        if self.runner.busy:
            self.status.set_text("Esperá a que termine el ajuste para editar las piezas.")
            self._sync_part_controls()
            return
        current = leg.parts[leg.index(name)]
        if (z is None or z == current.z) and (fixed is None or fixed == current.fixed):
            return
        self.part_overrides = merge_override(self.part_overrides, name, z=z, fixed=fixed)
        self._clear_result()
        self._rebuild_leg_families()
        assert self.advice is not None
        new_leg = self.picker.character
        assert new_leg is not None
        edited = new_leg.parts[new_leg.index(name)]
        self.status.set_text(
            f"«{name}»: z = {edited.z:g}, {'fija' if edited.fixed else 'móvil'}. "
            + self.advice.text
        )
        self._commit(f"part:{name}")
        self.canvas.queue_draw()

    def _refresh_properties(self) -> None:
        """Panel de propiedades: personaje, puntos elegidos, diagnóstico y estado del ajuste."""
        self.props_frame.set_visible(bool(self.leg_families))
        leg = self.picker.character
        if not self.leg_families or leg is None or self.advice is None:
            self.props.set_text("")
            return
        self._sync_part_controls()
        esc = GLib.markup_escape_text
        n_fixed = sum(1 for part in leg.parts if part.fixed)
        lines = [
            f"<b>{esc(leg.name.capitalize())}</b>",
            f"{len(leg.parts)} piezas ({n_fixed} fija)" if n_fixed == 1
            else f"{len(leg.parts)} piezas ({n_fixed} fijas)",
            f"Grados de libertad solo: {self.advice.dof}",
        ]
        if self.part_overrides:
            edits = ", ".join(
                f"{esc(ov.name)}"
                + (f" z={ov.z:g}" if ov.z is not None else "")
                + ("" if ov.fixed is None else (" fija" if ov.fixed else " móvil"))
                for ov in self.part_overrides
            )
            lines.append(f"Cambios: {edits}")
        lines.append("")
        points = (
            ("Punto de actuación", self.attach_ref, self.advice.attach_condition),
            ("Marcador", self.marker_ref, self.advice.marker_condition),
        )
        for title, ref, cond in points:
            part = leg.parts[leg.index(ref.part)]
            mob = point_mobility(leg, ref)
            lines.append(f"<b>{title}</b>")
            lines.append(
                f"«{esc(ref.part)}» en ({ref.point[0]:.2f}, {ref.point[1]:.2f})"
                f" · plano z = {part.z:g}" + (" · fija" if part.fixed else "")
            )
            lines.append(
                f"Se mueve en {mob} dirección(es) independiente(s)"
                + (f" · condición {cond:.2f}" if mob == 2 and self.advice.ok else "")
            )
            lines.append("")
        if not self.advice.ok:
            color = "#b3261e"  # no se puede ajustar
        elif self.advice.warnings:
            color = "#b26a00"  # se puede, con reparos
        else:
            color = "#1a7f37"
        lines.append(f'<span foreground="{color}">{esc(self.advice.text)}</span>')
        if self.fit_result is not None:
            lines += ["", "<b>Ajuste</b>", esc(describe(self.fit_result)),
                      f"Error {self.fit_result.rms_rel * 100:.1f} % del tamaño de la curva"]
        if isinstance(self.mech, MultiRig):
            lines += ["", "<b>Desfases</b>"]
            for sub, phase in zip(self.mech.subs, self.mech.phases, strict=True):
                lines.append(f"{esc(sub.name)}: {phase:.2f} ciclo")
            lines += ["", "<b>Tren de engranajes</b>"]
            text, train_ok = self._train_summary(self.mech)
            color = "#1a7f37" if train_ok else "#b3261e"
            lines.append(f'<span foreground="{color}">{esc(text)}</span>')
        self.props.set_markup("\n".join(lines))

    @staticmethod
    def _train_summary(rig: MultiRig) -> tuple[str, bool]:
        """Estado del tren de engranajes que une los ejes de entrada de los mecanismos del rig."""
        axes = []
        for sub in rig.subs:
            xy = input_axis(rig.asm, rig.state0, sub.drive)
            if xy is None:
                return f"No se pudo ubicar el eje de entrada de «{sub.name}».", False
            axes.append(Axis(sub.name, xy[0], xy[1]))
        return describe_train(plan_train(axes))

    def _on_box_changed(self) -> None:
        self.box = self.box_editor.box
        self.box_clear.set_sensitive(self.box is not None)
        if self.box is None:
            self.status.set_text("Sin caja de engranajes.")
        else:
            self.status.set_text(
                f"Caja de engranajes de {self.box.width:.1f} × {self.box.height:.1f}: "
                "los candidatos y el ajuste tienen que caber adentro."
            )
        if self.dbs and not self.leg_families and not self.runner.busy:
            self._request_suggestions()
        self._commit()
        self.canvas.queue_draw()

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
