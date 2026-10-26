"""Prueba de humo de la edición de piezas desde el panel (M4) sin pantalla: elegir una pieza,
cambiar su z y marcarla fija, el diagnóstico de grados de libertad, que el ajuste anterior se
quite, deshacer/rehacer, guardar/abrir y que el bípedo herede los cambios.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_parts.py
"""

import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.core.character import PartOverride
from mechchar.core.project import load_project, save_project
from mechchar.ui.window import MainWindow

fail: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def pick(win, name: str) -> None:
    win.part_dropdown.set_selected(win.part_names.index(name))


def script(win, tmp: Path):
    check(not win.props_frame.get_visible(), "en la curva libre no hay panel de propiedades")
    check(not any(w.get_sensitive() for w in win.part_controls), "ni controles de pieza")
    win.scene_dropdown.set_selected(1)
    yield ("sleep", 300)
    check(all(w.get_sensitive() for w in win.part_controls), "la pierna habilita los controles")
    check(win.part_overrides == (), "sin cambios al empezar")

    # 1. z de una pieza desde el control
    pick(win, "shank")
    check(win.part_z.get_value() == 0.0 and not win.part_fixed.get_active(), "shank: z 0, móvil")
    win.part_z.set_value(2.0)
    check(win.part_overrides == (PartOverride("shank", z=2.0),), "el spin guarda el z")
    check("z=2" in win.props.get_label() and "shank" in win.props.get_label(), "el panel lo lista")
    check(win.advice is not None and win.advice.dof == 2, "z no cambia los grados de libertad")

    # 2. marcar fija una pieza cambia el diagnóstico y quita el ajuste
    pick(win, "thigh")
    win.part_fixed.set_active(True)
    check(
        win.advice is not None and win.advice.dof < 2, "fijar el muslo baja los grados de libertad"
    )
    check(win.mech is None and win.fit_result is None, "no queda un mecanismo viejo")

    # 3. un punto sobre una pieza fija se rechaza al ajustar
    pick(win, "shank")
    win.part_fixed.set_active(True)
    win.fit_button.emit("clicked")
    check(
        not win.runner.busy and "fija" in win.status.get_text(),
        "Ajustar se niega con la actuación fija",
    )
    win.part_fixed.set_active(False)
    pick(win, "thigh")
    win.part_fixed.set_active(False)
    check(win.advice is not None and win.advice.ok, "al soltar las piezas vuelve a poder ajustar")

    # 4. deshacer / rehacer
    pick(win, "foot")
    win.part_z.set_value(1.5)
    after = win.part_overrides
    win._undo()
    check(win.part_overrides != after, "deshacer revierte el último cambio de pieza")
    win._redo()
    check(win.part_overrides == after, "rehacer lo vuelve a aplicar")

    # 5. guardar y abrir
    path = tmp / "partes.json"
    save_project(path, win._snapshot())
    check(load_project(path).parts == win.part_overrides, "el archivo guarda los cambios de piezas")

    # 6. el bípedo hereda los cambios
    win.scene_dropdown.set_selected(2)
    yield ("sleep", 300)
    check(win.part_overrides == (), "cambiar de escena parte de las piezas originales")
    pick(win, "foot")
    win.part_z.set_value(3.0)
    check(win.part_overrides == (PartOverride("foot", z=3.0),), "z del pie en el bípedo")
    check(win.picker.character is not None, "hay personaje en la escena del bípedo")

    yield ("done", None)


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokeparts")

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
