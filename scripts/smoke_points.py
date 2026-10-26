"""Prueba de humo de la elección de puntos de actuación y marcador (M4) sin pantalla: modos del
lienzo, clics sobre piezas, diagnóstico, «Sugerir», panel de propiedades, deshacer, guardar/abrir
y un ajuste del bípedo con el punto elegido. Dibuja la escena a /tmp/shot_points_*.png.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_points.py
"""

import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.core.character import LEG_ATTACH, LEG_MARKER
from mechchar.core.project import load_project
from mechchar.ui.window import MainWindow

fail: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def click(win, x: float, y: float) -> None:
    """Un clic del mouse sobre el punto (x, y) del mundo, como lo recibe el lienzo."""
    sx, sy = win.canvas.viewport.world_to_screen(x, y)
    claimed = win.canvas.drag_target.drag_begin(sx, sy)
    if claimed:
        win.canvas.drag_target.drag_end()


def render(win, path: str) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 900, 600)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    win.canvas.viewport.fit_bounds(-1.5, -3.2, 2.5, 0.6, 900, 600)
    win._draw_scene(cr, win.canvas.viewport)
    surface.write_to_png(path)
    print("escena dibujada en", path)


def script(win, tmp: Path):
    wait = lambda cond: ("until", cond)  # noqa: E731
    check(not win.props_frame.get_visible(), "en la curva libre no hay panel de propiedades")
    check(not any(w.get_sensitive() for w in win.pick_controls), "ni botones para elegir puntos")
    win.scene_dropdown.set_selected(1)
    yield ("sleep", 300)
    check(win.props_frame.get_visible() and all(w.get_sensitive() for w in win.pick_controls),
          "la pierna de prueba muestra el panel y habilita los botones")
    check("shank" in win.props.get_label() and "foot" in win.props.get_label(),
          "el panel nombra la actuación y el marcador por defecto")
    check(win.attach_ref == LEG_ATTACH and win.marker_ref == LEG_MARKER, "puntos por defecto")
    label = win.props.get_label()
    check("condición" in label and "casi singular" in label and "b26a00" in label,
          "el panel muestra la condición y avisa (en naranja) que el pie está en una pose casi singular")
    render(win, "/tmp/shot_points_0_inicio.png")

    # 1. modos del lienzo, excluyentes
    win.attach_toggle.set_active(True)
    check(win.canvas.drag_target is win.picker and win.picker.mode == "attach", "Actuación activa")
    win.marker_toggle.set_active(True)
    check(not win.attach_toggle.get_active() and win.picker.mode == "marker", "Marcador reemplaza")
    win.box_toggle.set_active(True)
    check(win.canvas.drag_target is win.box_editor and not win.marker_toggle.get_active(),
          "Caja reemplaza a Marcador")
    win.box_toggle.set_active(False)
    check(win.canvas.drag_target is win.editor, "sin modos, arrastrar edita la curva")

    # 2. elegir un punto que un pasador sobrerrestringiría (el muslo)
    win.attach_toggle.set_active(True)
    click(win, 0.45, -0.6)
    check(win.attach_ref.part == "thigh", "el clic elige el muslo")
    check(win.advice is not None and not win.advice.ok, "el diagnóstico lo rechaza")
    check("sobrerrestringe" in win.status.get_text(), "y el estado lo explica")
    check("sobrerrestringe" in win.props.get_label(), "y el panel también")
    win._start_fit(None)
    check(not win.runner.busy and "No se puede ajustar" in win.status.get_text(),
          "ajustar con ese punto se rechaza sin lanzar el trabajo")

    # 3. un clic en el vacío no cambia nada
    before = (win.attach_ref, win.marker_ref)
    click(win, 3.0, 3.0)
    check((win.attach_ref, win.marker_ref) == before, "un clic en el vacío no cambia los puntos")
    check("Hacé clic sobre una pieza" in win.status.get_text(), "y avisa")

    # 4. «Sugerir»
    win.attach_toggle.set_active(False)
    win.suggest_button.emit("clicked")
    first = win.attach_ref
    check(first.part != "thigh" and win.advice.ok, "Sugerir propone un punto que sirve")
    win.suggest_button.emit("clicked")
    second = win.attach_ref
    check(second != first and win.advice.ok, "y otra vez propone el siguiente")
    check(win.mech is None, "cambiar el punto deja la escena sin mecanismo viejo")

    # 5. mover el marcador traslada la curva objetivo con él
    spline0 = list(win.editor.points)
    win.marker_toggle.set_active(True)
    click(win, 0.45, -2.45)  # otro punto del pie
    dx = win.marker_ref.point[0] - LEG_MARKER.point[0]
    dy = win.marker_ref.point[1] - LEG_MARKER.point[1]
    check(win.marker_ref.part == "foot" and abs(dx + 0.45) < 0.2, "el marcador pasa a otro punto del pie")
    check(all(abs(a[0] - (b[0] + dx)) < 1e-9 and abs(a[1] - (b[1] + dy)) < 1e-9
              for a, b in zip(win.editor.points, spline0, strict=True)),
          "la curva objetivo se trasladó igual")
    win.marker_toggle.set_active(False)
    render(win, "/tmp/shot_points_1_elegidos.png")

    # 6. deshacer / rehacer / guardar / abrir
    chosen = win._snapshot()
    check(chosen.attach is not None and chosen.marker is not None, "los puntos están en la foto")
    win._undo()
    check(win.marker_ref == LEG_MARKER and win.attach_ref == second,
          "deshacer devuelve el marcador y deja la actuación como estaba")
    win._redo()
    check(win._snapshot() == chosen, "rehacer lo vuelve a poner")
    path = tmp / "puntos.json"
    check(win.save_to(str(path)) and load_project(path) == chosen, "guardar escribe los puntos")
    while win.history.can_undo:
        win._undo()
    check(win.open_from(str(path)), "abrir el proyecto")
    check(win._snapshot() == chosen, "abrir restaura los mismos puntos")
    check(win.props_frame.get_visible() and win.marker_ref.point == (chosen.marker.x, chosen.marker.y),
          "y el panel los muestra")

    # 7. bípedo: el ajuste usa el punto elegido (cada pierna, con su prefijo)
    win.scene_dropdown.set_selected(2)
    check(win.attach_ref == LEG_ATTACH, "al cambiar de escena vuelven los puntos por defecto")
    win.suggest_button.emit("clicked")
    picked = win.attach_ref
    win._start_fit(None)
    check(win.runner.busy and not any(w.get_sensitive() for w in win.pick_controls),
          "durante el ajuste no se pueden cambiar los puntos")
    yield ("sleep", 500)
    yield wait(lambda: not win.runner.busy)
    check(win.mech is not None and len(win.mech.subs) == 2, "el bípedo se arma con el punto elegido")
    check([s.attach.part for s in win.mech.subs] == ["l_" + picked.part, "r_" + picked.part],
          "cada pierna se une en su copia del punto elegido")
    check(all(w.get_sensitive() for w in win.pick_controls), "terminado el ajuste se habilitan de nuevo")
    check(win.mech.analyze().dof == 0 and win.mech.analyze().redundant == 0,
          "el conjunto queda con 0 grados de libertad y sin redundancia")
    check("Desfases" in win.props.get_label(), "el panel muestra los desfases")
    check(win.active_leg is None, "sin tocar sliders no hay pierna resaltada")
    win.phase_scales[0].set_value(0.1)
    check(win.active_leg == 0, "mover un slider resalta su pierna (la otra se atenúa)")
    render(win, "/tmp/shot_points_2_bipedo.png")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokepoints")

    def on_activate(app):
        win = MainWindow(app, dbs=[])
        win.present()
        tmp = Path(tempfile.mkdtemp())
        gen = script(win, tmp)
        state = {"wait": None}

        def finish():
            print("FALLAS:", fail if fail else "ninguna")
            app.quit()
            sys.exit(1 if fail else 0)

        def pump():
            try:
                if state["wait"] is not None:
                    kind, arg = state["wait"]
                    if kind == "until" and not arg():
                        return True
                    state["wait"] = None
                step = next(gen)
            except StopIteration:
                return finish()
            except Exception:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                fail.append("excepción")
                return finish()
            if step[0] == "sleep":
                GLib.timeout_add(step[1], lambda: (state.update(wait=None), pump_once())[1])
                state["wait"] = ("until", lambda: False)
                return False
            state["wait"] = step
            return True

        def pump_once():
            GLib.timeout_add(100, pump)
            return False

        GLib.timeout_add(1000, pump)

    app.connect("activate", on_activate)
    app.run([])


main()
