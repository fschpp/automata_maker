"""Prueba de humo de guardar/abrir y deshacer/rehacer sin pantalla.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_project.py
"""

import json
import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.core.project import load_project
from mechchar.ui.window import MainWindow

fail: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def drag_point(win, index: int, dx: float, dy: float) -> None:
    """Mueve un punto de control de la spline como lo haría el mouse."""
    vp = win.canvas.viewport
    x, y = win.editor.points[index]
    sx, sy = vp.world_to_screen(x, y)
    assert win.editor.drag_begin(sx, sy)
    win.editor.drag_update(sx + dx, sy + dy)
    win.editor.drag_end()


def script(win, tmp: Path):
    wait = lambda cond: ("until", cond)  # noqa: E731
    yield wait(lambda: bool(win.dbs))
    yield ("sleep", 1500)
    check(not win.history.can_undo, "al abrir no hay nada que deshacer")
    original = win._snapshot()

    # 1. editar la spline
    drag_point(win, 0, 40, -30)
    moved = win._snapshot()
    check(moved.spline != original.spline and win.history.can_undo, "mover un punto es un paso")
    # 2. ajustar
    win._start_fit(None)
    yield ("sleep", 500)
    yield wait(lambda: not win.runner.busy)
    check(win.mech is not None and win._snapshot().fit is not None, "el ajuste queda en la foto")
    fitted = win._snapshot()
    bounds = win.mech_bounds
    # 3. deshacer / rehacer
    win._undo()
    check(win.mech is None and win._snapshot().spline == moved.spline, "deshacer quita el ajuste")
    win._undo()
    check(win._snapshot() == original and not win.history.can_undo, "deshacer vuelve al inicio")
    win._redo()
    win._redo()
    check(win.mech is not None and win._snapshot() == fitted, "rehacer reconstruye el mecanismo")
    check(
        all(abs(a - b) < 1e-9 for a, b in zip(win.mech_bounds, bounds, strict=True)),
        "el mecanismo rehecho es el mismo (mismos límites)",
    )
    check(not win.history.can_redo, "no queda nada que rehacer")
    # 4. guardar y abrir
    path = tmp / "p.json"
    check(win.save_to(str(path)) and path.exists(), "guardar escribe el archivo")
    check(load_project(path) == fitted, "lo guardado es la foto actual")
    win._undo()
    win._undo()
    check(win._snapshot() == original, "volvió al inicio antes de abrir")
    check(win.open_from(str(path)), "abrir el proyecto funciona")
    check(win._snapshot() == fitted and win.mech is not None, "abrir restaura el ajuste")
    check(not win.history.can_undo, "abrir empieza un historial nuevo")
    # 5. archivos que no sirven no tocan nada
    bad = tmp / "malo.json"
    bad.write_text("{no es json", encoding="utf-8")
    before = win._snapshot()
    check(not win.open_from(str(bad)), "un archivo roto no se abre")
    check("No se pudo abrir" in win.status.get_text(), "y se avisa en el estado")
    check(not win.open_from(str(tmp / "nada.json")), "un archivo inexistente tampoco")
    d = json.loads(path.read_text())
    d["fit"]["family"] = "no_existe"
    (tmp / "familia.json").write_text(json.dumps(d), encoding="utf-8")
    win.open_from(str(tmp / "familia.json"))
    check(win.mech is None, "una familia desconocida deja la escena sin mecanismo")
    check("desconocida" in win.status.get_text(), "y lo avisa")
    win.open_from(str(path))
    check(win._snapshot() == before, "se puede volver a abrir el bueno")

    # 6. bípedo con desfases
    win.scene_dropdown.set_selected(2)
    check(win.biped and win.history.can_undo, "cambiar de escena es un paso")
    win._start_fit(None)
    yield ("sleep", 500)
    yield wait(lambda: not win.runner.busy)
    check(win.mech is not None and win.mech.phases == [0.0, 0.5], "el bípedo arranca con marcha")
    win.phase_scales[1].set_value(0.8)
    win.phase_scales[1].set_value(0.85)  # dos cambios seguidos: un solo paso
    check(win.mech.phases[1] == 0.85, "el slider mueve el desfase")
    bip = win._snapshot()
    check(bip.phases == (0.0, 0.85), "el desfase está en la foto")
    bpath = tmp / "bipedo.json"
    win.save_to(str(bpath))
    win._undo()
    check(win.mech.phases == [0.0, 0.5], "un solo deshacer vuelve el slider entero a 0,5")
    check(abs(win.phase_scales[1].get_value() - 0.5) < 1e-9, "y el slider de la UI también")
    win._redo()
    check(win.mech.phases[1] == 0.85, "rehacer vuelve a 0,85")
    win._undo()  # sin desfase movido
    win._undo()  # sin ajuste, pero todavía en el bípedo
    check(win.biped and win.mech is None, "dos pasos atrás: bípedo sin ajuste")
    win._undo()  # sale del bípedo y recupera el ajuste de la escena 0
    check(not win.biped and win.mech is not None and win._snapshot() == before,
          "un paso más: vuelve a la escena 0 con su ajuste")
    check(win.open_from(str(bpath)), "abrir el proyecto del bípedo")
    check(win.biped and win.mech is not None and win.mech.phases == [0.0, 0.85],
          "se restauran las dos piernas y sus desfases")
    check(abs(win.phase_scales[1].get_value() - 0.85) < 1e-9, "y los sliders")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokeproject")

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
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fail.append(repr(exc))
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
