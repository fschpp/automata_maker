"""Ventana «Red de engranajes» (M5): ejes que se arrastran, pares que se eligen con clics,
intermedios, animación y vista lateral con la profundidad de cada eje cilíndrico.

La lógica está en ``core/gearedit.py`` (modelo), ``ui/gear_editor.py`` (mouse) y
``ui/gear_draw.py`` (dibujo); acá solo se arma la ventana y se conectan.
"""

from __future__ import annotations

import math
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from mechchar.core.gearedit import GearNetModel, PlanOutcome  # noqa: E402
from mechchar.core.gearnet import axles  # noqa: E402
from mechchar.ui.canvas import Canvas  # noqa: E402
from mechchar.ui.gear_draw import draw_editor, draw_side_view  # noqa: E402
from mechchar.ui.gear_editor import GearEditor  # noqa: E402

SECONDS_PER_TURN = 8.0
MAX_INTERMEDIATES = 8
OK_COLOR = "#1a7f37"
BAD_COLOR = "#b3261e"
HELP = (
    "Clic en un eje: lo selecciona (con dos, «Conectar» los une). Arrastrá un eje para "
    "moverlo. Con «Agregar ejes», un clic en un lugar vacío crea uno. Aro naranja: la entrada; "
    "↺: gira al revés que la entrada. Línea llena: par elegido; punteada: se completó sola."
)


def _clear(box: Gtk.Box) -> None:
    child = box.get_first_child()
    while child is not None:
        nxt = child.get_next_sibling()
        box.remove(child)
        child = nxt


def _header(text: str) -> Gtk.Label:
    label = Gtk.Label(xalign=0.0, margin_top=8)
    label.set_markup(f"<b>{GLib.markup_escape_text(text)}</b>")
    return label


class GearWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, model: GearNetModel | None = None) -> None:
        super().__init__(
            application=app, title="Red de engranajes", default_width=1200, default_height=780
        )
        self.model = model if model is not None else GearNetModel.example()
        self.theta = 0.0
        self.playing = False
        self._last: float | None = None
        self._building = False
        self._notice = ""
        self._outcome: PlanOutcome = self.model.plan()
        self.axis_rows: list[dict[str, Any]] = []
        self.link_rows: list[dict[str, Any]] = []

        # --- lienzo, vista lateral y controles --------------------------------------------
        self.canvas = Canvas()
        self.canvas.viewport.scale = 28.0
        self.canvas.origin_frac = (0.12, 0.8)
        self.canvas.set_vexpand(True)
        self.canvas.overlay = lambda cr, vp: draw_editor(
            cr, vp, self.model, self._outcome, self.theta
        )
        self.editor = GearEditor(self.model, self.canvas.viewport, self._on_editor_change)
        self.canvas.drag_target = self.editor

        self.side = Gtk.DrawingArea()
        self.side.set_content_height(170)
        self.side.set_draw_func(self._draw_side)

        self.play = Gtk.ToggleButton(label="Girar")
        self.play.connect("toggled", self._on_play)
        self.scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 360.0, 1.0)
        self.scale.set_hexpand(True)
        self.scale.connect("value-changed", self._on_scale)
        bar = Gtk.Box(spacing=8, margin_start=8, margin_end=8, margin_top=4, margin_bottom=4)
        bar.append(self.play)
        bar.append(self.scale)

        self.status = Gtk.Label(xalign=0.0, wrap=True, margin_start=8, margin_end=8)
        self.status.set_selectable(True)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        for w in (self.canvas, self.side, bar, self.status):
            left.append(w)

        # --- panel de la derecha ----------------------------------------------------------
        self.add_toggle = Gtk.ToggleButton(label="Agregar ejes")
        self.add_toggle.set_tooltip_text("Un clic en un lugar vacío del lienzo crea un eje")
        self.add_toggle.connect("toggled", self._on_add_toggled)
        example = Gtk.Button(label="Ejemplo")
        example.connect("clicked", lambda _b: self._replace(GearNetModel.example()))
        empty = Gtk.Button(label="Vaciar")
        empty.connect("clicked", lambda _b: self._replace(GearNetModel()))
        tools = Gtk.Box(spacing=6)
        for w in (self.add_toggle, example, empty):
            tools.append(w)

        self.axes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.links_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        self.auto_check = Gtk.CheckButton(label="Completar los pares que falten solo")
        self.auto_check.set_active(self.model.auto)
        self.auto_check.connect("toggled", self._on_auto_toggled)
        self.connect_button = Gtk.Button(label="Conectar seleccionados")
        self.connect_button.connect("clicked", self._on_connect)
        self.clear_button = Gtk.Button(label="Quitar los pares elegidos")
        self.clear_button.connect("clicked", self._on_clear_links)

        self.thickness = Gtk.SpinButton.new_with_range(0.2, 10.0, 0.1)
        self.thickness.set_value(1.0)
        self.protrusion = Gtk.SpinButton.new_with_range(0.0, 20.0, 0.5)
        self.protrusion.set_value(2.0)
        for spin in (self.thickness, self.protrusion):
            spin.connect("value-changed", lambda _s: self._refresh(rebuild=False))
        depth = Gtk.Grid(column_spacing=8, row_spacing=4)
        depth.attach(Gtk.Label(label="Espesor del engranaje", xalign=0.0), 0, 0, 1, 1)
        depth.attach(self.thickness, 1, 0, 1, 1)
        depth.attach(Gtk.Label(label="Sobresale el eje de entrada", xalign=0.0), 0, 1, 1, 1)
        depth.attach(self.protrusion, 1, 1, 1, 1)
        self.axles_label = Gtk.Label(xalign=0.0, wrap=True)
        self.axles_label.add_css_class("monospace")
        help_label = Gtk.Label(label=HELP, xalign=0.0, wrap=True, margin_top=8)
        help_label.add_css_class("dim-label")

        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        panel.set_margin_start(8)
        panel.set_margin_end(8)
        panel.set_margin_top(8)
        panel.set_margin_bottom(8)
        for w in (
            tools,
            _header("Ejes"),
            self.axes_box,
            _header("Pares"),
            self.links_box,
            self.auto_check,
            self.connect_button,
            self.clear_button,
            _header("Profundidad de los ejes"),
            depth,
            self.axles_label,
            help_label,
        ):
            panel.append(w)
        scroller = Gtk.ScrolledWindow()
        scroller.set_min_content_width(380)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(panel)

        root = Gtk.Box()
        root.append(left)
        root.append(scroller)
        self.set_child(root)
        self.add_tick_callback(self._tick)
        self._refresh(rebuild=True)

    # --- estado y refresco ----------------------------------------------------------------------
    def _replace(self, new: GearNetModel) -> None:
        """Pone el contenido de ``new`` en el modelo que usan el editor y el lienzo."""
        m = self.model
        m.axes, m.root, m.links = new.axes, new.root, new.links
        m.auto, m.forced, m.selected = new.auto, new.forced, new.selected
        self._notice = ""
        self.auto_check.set_active(m.auto)
        self._refresh(rebuild=True)

    def _on_editor_change(self) -> None:
        self._refresh(rebuild=False)

    def _rows_stale(self) -> bool:
        """¿Las filas del panel ya no corresponden al modelo? (se agregó un eje con el mouse o,
        al mover ejes, cambiaron los pares que se completan solos)."""
        chosen = {frozenset(p) for p in self.model.links}
        wanted = [(i, j, frozenset((i, j)) not in chosen) for i, j in self._outcome.links]
        shown = [(r["i"], r["j"], r["auto"]) for r in self.link_rows]
        return len(self.axis_rows) != len(self.model.axes) or shown != wanted

    def _refresh(self, rebuild: bool) -> None:
        if self._building:
            return
        self._outcome = self.model.plan()
        if rebuild or self._rows_stale():
            self._rebuild_rows()
        self._update_rows()
        text, ok = self.model.describe(self._outcome)
        color = OK_COLOR if ok else BAD_COLOR
        markup = f'<span foreground="{color}">{GLib.markup_escape_text(text)}</span>'
        if self._notice:
            note = GLib.markup_escape_text(self._notice)
            markup += f'\n<span foreground="{BAD_COLOR}">{note}</span>'
        self.status.set_markup(markup)
        self.connect_button.set_sensitive(self.model.can_connect_selected())
        self.clear_button.set_sensitive(bool(self.model.links))
        train = self._outcome.train
        if train is not None and train.connections:
            lines = [
                f"{a.name}: z {a.z0:.1f}–{a.z1:.1f} (largo {a.length:.1f}"
                f"{', sobresale' if a.input else ''})"
                for a in axles(train, self.thickness.get_value(), self.protrusion.get_value())
            ]
            self.axles_label.set_text("\n".join(lines))
        else:
            self.axles_label.set_text("Sin engranajes todavía.")
        self.canvas.queue_draw()
        self.side.queue_draw()

    def _pair_text(self, i: int, j: int, auto: bool) -> str:
        a, b = self.model.axes[i].name, self.model.axes[j].name
        return f"{a} — {b}" + ("  (auto)" if auto else "")

    def _rebuild_rows(self) -> None:
        self._building = True
        try:
            _clear(self.axes_box)
            _clear(self.links_box)
            self.axis_rows, self.link_rows = [], []
            first_radio: Gtk.CheckButton | None = None
            for i, axis in enumerate(self.model.axes):
                radio = Gtk.CheckButton()
                radio.set_tooltip_text("Eje de entrada (el que mueve el motor o la manivela)")
                if first_radio is None:
                    first_radio = radio
                else:
                    radio.set_group(first_radio)
                radio.set_active(i == self.model.root)
                radio.connect("toggled", self._on_root_toggled, i)
                entry = Gtk.Entry(text=axis.name, hexpand=True)
                entry.set_width_chars(10)
                entry.connect("changed", self._on_name_changed, i)
                reverse = Gtk.CheckButton(label="↺")
                reverse.set_tooltip_text("Gira al revés que el eje de entrada")
                reverse.set_active(not axis.same_direction)
                reverse.set_sensitive(i != self.model.root)
                reverse.connect("toggled", self._on_reverse_toggled, i)
                remove = Gtk.Button(label="✕")
                remove.set_tooltip_text("Quitar el eje")
                remove.connect("clicked", self._on_remove_axis, i)
                row = Gtk.Box(spacing=6)
                for w in (radio, entry, reverse, remove):
                    row.append(w)
                self.axes_box.append(row)
                self.axis_rows.append(
                    {"radio": radio, "entry": entry, "reverse": reverse, "remove": remove}
                )
            chosen = {frozenset(p) for p in self.model.links}
            for i, j in self._outcome.links:
                auto = frozenset((i, j)) not in chosen
                label = Gtk.Label(xalign=0.0, hexpand=True)
                info = Gtk.Label(xalign=0.0)
                info.add_css_class("dim-label")
                choice = Gtk.DropDown.new_from_strings(
                    ["Auto"] + [str(k) for k in range(MAX_INTERMEDIATES + 1)]
                )
                choice.set_tooltip_text(
                    "Intermedios: impar = mismo sentido, par = sentido contrario"
                )
                forced = self.model.forced.get(frozenset((i, j)))
                choice.set_selected(0 if forced is None else min(forced, MAX_INTERMEDIATES) + 1)
                choice.connect("notify::selected", self._on_intermediates, i, j)
                button = Gtk.Button(label="Fijar" if auto else "✕")
                button.set_tooltip_text("Dejar este par como elegido" if auto else "Quitar el par")
                button.connect("clicked", self._on_pair_button, i, j, auto)
                top = Gtk.Box(spacing=6)
                for w in (label, choice, button):
                    top.append(w)
                cell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                cell.append(top)
                cell.append(info)
                self.links_box.append(cell)
                self.link_rows.append(
                    {
                        "i": i,
                        "j": j,
                        "auto": auto,
                        "label": label,
                        "info": info,
                        "choice": choice,
                        "button": button,
                    }
                )
        finally:
            self._building = False

    def _update_rows(self) -> None:
        """Textos que cambian sin reconstruir las filas (para no perder el foco al escribir)."""
        train = self._outcome.train
        rep = self.model.representatives()
        for row in self.link_rows:
            i, j = row["i"], row["j"]
            row["label"].set_text(self._pair_text(i, j, row["auto"]))
            info = ""
            if train is not None:
                for c in train.connections:
                    if {c.a, c.b} == {rep[i], rep[j]}:
                        info = (
                            f"{c.intermediates} intermedio(s) · extremos de {c.gears[0].teeth} "
                            f"dientes · capa {c.layer + 1}"
                        )
            row["info"].set_text(info)

    # --- eventos del panel ----------------------------------------------------------------------
    def _guard(self, action: Any, rebuild: bool = True) -> None:
        """Ejecuta ``action``; un ``ValueError`` se muestra en el estado en vez de romper."""
        if self._building:
            return
        self._notice = ""
        try:
            action()
        except ValueError as exc:
            self._notice = str(exc)
        self._refresh(rebuild=rebuild)

    def _on_root_toggled(self, button: Gtk.CheckButton, i: int) -> None:
        if button.get_active():
            self._guard(lambda: self.model.set_root(i))

    def _on_name_changed(self, entry: Gtk.Entry, i: int) -> None:
        text = entry.get_text()
        if self._building or not text.strip():
            return
        self._guard(lambda: self.model.rename_axis(i, text), rebuild=False)

    def _on_reverse_toggled(self, button: Gtk.CheckButton, i: int) -> None:
        self._guard(lambda: self.model.set_same_direction(i, not button.get_active()))

    def _on_remove_axis(self, _b: Gtk.Button, i: int) -> None:
        self._guard(lambda: self.model.remove_axis(i))

    def _on_pair_button(self, _b: Gtk.Button, i: int, j: int, auto: bool) -> None:
        self._guard(lambda: self.model.connect(i, j) if auto else self.model.unlink(i, j))

    def _on_intermediates(self, dropdown: Gtk.DropDown, _p: Any, i: int, j: int) -> None:
        sel = int(dropdown.get_selected())
        self._guard(
            lambda: self.model.set_intermediates(i, j, None if sel == 0 else sel - 1),
            rebuild=False,
        )

    def _on_connect(self, _b: Gtk.Button) -> None:
        self._guard(self.model.connect_selected)

    def _on_clear_links(self, _b: Gtk.Button) -> None:
        self._guard(self.model.clear_links)

    def _on_auto_toggled(self, button: Gtk.CheckButton) -> None:
        def apply() -> None:
            self.model.auto = button.get_active()

        self._guard(apply)

    def _on_add_toggled(self, button: Gtk.ToggleButton) -> None:
        self.editor.add_mode = button.get_active()

    # --- animación y dibujo ---------------------------------------------------------------------
    def _on_scale(self, scale: Gtk.Scale) -> None:
        self.theta = math.radians(scale.get_value())
        self.canvas.queue_draw()

    def _on_play(self, button: Gtk.ToggleButton) -> None:
        self.playing = button.get_active()
        self._last = None

    def _tick(self, _w: Gtk.Widget, clock: Any) -> bool:
        now = clock.get_frame_time() / 1e6
        if self.playing and self._last is not None:
            self.theta = (self.theta + 2 * math.pi * (now - self._last) / SECONDS_PER_TURN) % (
                2 * math.pi
            )
            self.scale.handler_block_by_func(self._on_scale)
            self.scale.set_value(math.degrees(self.theta))
            self.scale.handler_unblock_by_func(self._on_scale)
            self.canvas.queue_draw()
        self._last = now
        return GLib.SOURCE_CONTINUE

    def _draw_side(self, _area: Gtk.DrawingArea, cr: Any, width: int, height: int) -> None:
        draw_side_view(
            cr,
            float(width),
            float(height),
            self._outcome.train,
            self.thickness.get_value(),
            self.protrusion.get_value(),
        )
