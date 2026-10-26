"""Prueba de humo de la escena «Pushing Man» en la ventana real: el panel y el selector de piezas
cambian con el personaje, el aviso de alcance habla de la figura, editar una pieza rehace el
diagnóstico, el ajuste de la curva por defecto baja de 3 %, y guardar, cambiar a la pierna y
volver a abrir restaura el personaje y la misma pose. Dibuja la escena a /tmp/shot_pm_*.png.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_pushing_man.py
"""

import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
import numpy as np
from gi.repository import GLib, Gtk

from mechchar.core.character import PUSHING_MAN_SPEC
from mechchar.ui.spline_editor import default_points
from mechchar.ui.window import MainWindow

fail: list[str] = []
PM_PARTS = ["base", "torso", "arm", "forearm", "strut"]


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def render(win, path: str) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 900, 600)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    x0, y0, x1, y1 = win.spec.view
    win.canvas.viewport.fit_bounds(x0, y0, x1, y1, 900, 600)
    win._draw_scene(cr, win.canvas.viewport)
    surface.write_to_png(path)
    print("escena dibujada en", path)


def click(win, x: float, y: float) -> None:
    sx, sy = win.canvas.viewport.world_to_screen(x, y)
    if win.canvas.drag_target.drag_begin(sx, sy):
        win.canvas.drag_target.drag_end()


def script(win, tmp: Path):
    until = lambda cond: ("until", cond)  # noqa: E731
    settled = lambda: not win.reach_runner.busy  # noqa: E731

    win.scene_dropdown.set_selected(3)
    yield ("sleep", 500)
    yield until(settled)
    yield ("sleep", 300)
    check(win.spec is PUSHING_MAN_SPEC and win.picker.character.name == "pushing_man",
          "la escena 3 trabaja con el Pushing Man")
    check(win.part_names == PM_PARTS, f"el selector de piezas lista las del personaje {win.part_names}")
    check(win.part_dropdown.get_model().get_n_items() == len(PM_PARTS), "y el modelo del selector cambió")
    check(win.props_frame.get_visible() and "pushing_man" in win.props.get_label().lower(),
          "el panel de propiedades muestra el personaje")
    check("Pushing Man" in win.status.get_text() or "mano" in win.status.get_text(),
          "el texto de ayuda es el de la figura")
    check(win.reach_ok, "la curva por defecto está al alcance")
    check(not win.phase_row.get_visible(), "no hay sliders de desfase (no es el bípedo)")
    render(win, "/tmp/shot_pm_0_inicio.png")

    # aviso de alcance con las palabras de la figura
    win.editor.points = default_points(8, 0.4, 0.3, (2.4, 1.8))
    win._commit()
    yield ("sleep", 200)
    yield until(settled)
    yield ("sleep", 300)
    text = win.status.get_text()
    check(not win.reach_ok and "fuera del alcance de la figura" in text and "la zona que alcanza la mano" in text,
          "una curva fuera de alcance avisa con las palabras de la figura")
    cx, cy, a, b = PUSHING_MAN_SPEC.ellipse
    win.editor.points = default_points(8, a, b, (cx, cy))
    win._commit()
    yield ("sleep", 200)
    yield until(settled)
    yield ("sleep", 300)
    check(win.reach_ok and "al alcance de la figura" in win.status.get_text(), "volver a una alcanzable lo dice")

    # elegir el punto de actuación con un clic sobre el antebrazo
    win.attach_toggle.set_active(True)
    click(win, 1.1, 0.55)
    check(win.attach_ref.part == "forearm", "un clic sobre el antebrazo elige el punto de actuación")
    click(win, 0.2, 1.0)  # solo sobre el torso, que gira alrededor de la cadera fija
    check(win.attach_ref.part == "torso" and win.advice is not None and not win.advice.ok
          and "sobrerrestringe" in win.status.get_text(),
          "un punto del torso se rechaza porque un pasador lo sobrerrestringiría")
    win.attach_toggle.set_active(False)
    win.set_points(attach=PUSHING_MAN_SPEC.attach)  # volver al punto por defecto

    # editar una pieza: fijar la barra pasiva cambia los grados de libertad
    dof0 = win.advice.dof
    win.part_dropdown.set_selected(PM_PARTS.index("strut"))
    check(win._selected_part() == "strut", "se elige la barra en el selector")
    win.part_fixed.set_active(True)
    check(win.advice.dof != dof0, f"fijar la barra cambia los grados de libertad ({dof0} -> {win.advice.dof})")
    win.part_fixed.set_active(False)
    check(win.advice.dof == dof0 and win.advice.ok, "y soltarla los devuelve")

    # ajuste real de la curva por defecto
    win._start_fit(None)
    yield ("sleep", 500)
    yield until(lambda: not win.runner.busy)
    yield ("sleep", 300)
    r = win.fit_result
    check(r is not None and r.rms_rel < 0.03,
          f"el ajuste de la curva por defecto baja de 3 % ({0 if r is None else r.rms_rel * 100:.2f} %)")
    check(win.mech is not None and win.playback is not None, "queda un mecanismo animado")
    render(win, "/tmp/shot_pm_1_ajustado.png")
    before = np.array(win.mech.state0)

    # guardar, pasar a la pierna (otro personaje) y volver a abrir
    path = tmp / "pm.json"
    check(win.save_to(str(path)), "guardar el proyecto")
    win.scene_dropdown.set_selected(1)
    yield ("sleep", 600)
    yield until(settled)
    check(win.spec.key == "leg" and win.part_names == ["pelvis", "thigh", "shank", "foot", "rod"],
          f"al cambiar a la pierna el selector vuelve a sus piezas {win.part_names}")
    win.part_dropdown.set_selected(2)
    check(win._selected_part() == "shank", "y se puede elegir una pieza de la pierna")
    check(win.open_from(str(path)), "abrir el proyecto del Pushing Man")
    yield ("sleep", 800)
    yield until(lambda: not win.runner.busy)
    check(win.spec is PUSHING_MAN_SPEC and win.part_names == PM_PARTS,
          "se restaura el personaje y su selector de piezas")
    check(win.mech is not None, "se reconstruye el mecanismo")
    if win.mech is not None:
        d = np.abs(np.array(win.mech.state0) - before).max()
        check(d < 1e-5, f"el personaje queda en la misma pose (diferencia {d:.1e})")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokepm")

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
