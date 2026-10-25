"""Prueba de humo del alcance de la pierna en la ventana real: la curva por defecto está al alcance,
arrastrar la curva a la zona que el pie no alcanza avisa, volver a una alcanzable lo dice, el ajuste
de una curva alcanzable baja de 3 % (criterio histórico de M2) y, al guardar y volver a abrir, el
personaje queda en la misma pose (misma rama de ensamblado) que el ajuste.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_reach.py
"""

import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
import numpy as np
from gi.repository import GLib, Gtk

from mechchar.core.character import LEG_DEFAULT_ELLIPSE
from mechchar.ui.spline_editor import default_points
from mechchar.ui.window import MainWindow

fail: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def script(win, tmp: Path):
    until = lambda cond: ("until", cond)  # noqa: E731
    settled = lambda: not win.reach_runner.busy  # noqa: E731

    def default_curve():
        cx, cy, a, b = LEG_DEFAULT_ELLIPSE
        return default_points(8, a, b, (cx, cy))

    win.scene_dropdown.set_selected(1)
    yield ("sleep", 400)
    yield until(settled)
    yield ("sleep", 300)
    check(win.reach_ok, "la curva por defecto de la pierna está al alcance")
    check("Atención" not in win.status.get_text(), "y no hay aviso")

    # una elipse que el pie no alcanza (la de las primeras versiones de la escena)
    win.editor.points = default_points(8, 0.55, 0.25, (0.9, -2.4))
    win._commit()
    yield ("sleep", 200)
    yield until(settled)
    yield ("sleep", 300)
    check(not win.reach_ok, "una curva fuera de la franja que alcanza el pie se detecta")
    text = win.status.get_text()
    check("fuera del alcance" in text and "Atención" in text, "y se avisa en la barra de estado")

    # volver a una alcanzable lo dice
    win.editor.points = default_curve()
    win._commit()
    yield ("sleep", 200)
    yield until(settled)
    yield ("sleep", 300)
    check(win.reach_ok and "al alcance de la pierna" in win.status.get_text(),
          "volver a una curva alcanzable lo avisa")

    # ajuste real con la curva por defecto
    win._start_fit(None)
    yield ("sleep", 500)
    yield until(lambda: not win.runner.busy)
    yield ("sleep", 300)
    r = win.fit_result
    check(r is not None and r.rms_rel < 0.03, f"el ajuste de la curva por defecto baja de 3 % ({0 if r is None else r.rms_rel * 100:.2f} %)")
    check(win.mech is not None and win.playback is not None, "queda un mecanismo animado")
    before = np.array(win.mech.state0)
    marker0 = np.array(win.mech.marker_world(win.mech.state0))

    # guardar, cambiar de escena y volver a abrir: mismo estado inicial del personaje
    path = tmp / "alcance.json"
    check(win.save_to(str(path)), "guardar el proyecto")
    win.scene_dropdown.set_selected(0)
    yield ("sleep", 300)
    check(win.open_from(str(path)), "abrir el proyecto")
    yield ("sleep", 800)
    yield until(lambda: not win.runner.busy)
    check(win.mech is not None, "se reconstruye el mecanismo guardado")
    if win.mech is not None:
        after = np.array(win.mech.state0)
        check(after.shape == before.shape and np.abs(after - before).max() < 1e-5,
              f"el personaje queda en la misma pose (diferencia {np.abs(after - before).max():.1e})")
        check(np.abs(np.array(win.mech.marker_world(win.mech.state0)) - marker0).max() < 1e-5,
              "y el marcador empieza en el mismo punto")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokereach")

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
