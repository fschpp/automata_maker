"""Prueba de humo de mover y redimensionar la caja de engranajes en la ventana real: dibujarla,
moverla arrastrando desde adentro, cambiarle el tamaño desde un borde y una esquina, deshacer y
rehacer, guardar y abrir el proyecto, y dibujar una nueva fuera de la caja. Dibuja la escena a
/tmp/shot_boxedit_*.png.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_box_edit.py
"""

import sys
import tempfile
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.optim.gearbox import GearBox
from mechchar.ui.window import MainWindow

fail: list[str] = []


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
    win.canvas.viewport.fit_bounds(-6.0, -3.5, 6.0, 3.5, 900, 600)
    win._draw_scene(cr, win.canvas.viewport)
    surface.write_to_png(path)
    print("escena dibujada en", path)


def drag(win, start, end) -> None:
    """Un arrastre del mouse entre dos puntos del mundo, como lo recibe el lienzo."""
    vp = win.canvas.viewport
    a, b = vp.world_to_screen(*start), vp.world_to_screen(*end)
    target = win.canvas.drag_target
    if target.drag_begin(*a):
        target.drag_update((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        target.drag_update(*b)
        target.drag_end()


def same(a: GearBox | None, b: GearBox, tol: float = 1e-3) -> bool:
    if a is None:
        return False
    pairs = zip((a.x0, a.y0, a.x1, a.y1), (b.x0, b.y0, b.x1, b.y1), strict=True)
    return all(abs(u - v) < tol for u, v in pairs)


def script(win, tmp: Path):
    yield ("sleep", 400)
    win.canvas.viewport.fit_bounds(-6.0, -3.5, 6.0, 3.5, 900, 600)
    win.box_toggle.set_active(True)
    check(win.canvas.drag_target is win.box_editor and win.box_editor.active,
          "con «Caja» activa el lienzo edita la caja y se muestran las manijas")

    drag(win, (-2.0, -1.0), (2.0, 1.0))
    check(same(win.box, GearBox(-2.0, -1.0, 2.0, 1.0)), "se dibuja la caja arrastrando")
    render(win, "/tmp/shot_boxedit_0_dibujada.png")

    drag(win, (0.0, 0.0), (1.5, -0.5))
    check(same(win.box, GearBox(-0.5, -1.5, 3.5, 0.5)),
          "arrastrar desde adentro mueve la caja sin cambiarle el tamaño")
    check("4.0 × 2.0" in win.status.get_text(), "y la barra de estado informa el tamaño")

    drag(win, (3.5, 0.0), (4.5, 0.0))
    check(same(win.box, GearBox(-0.5, -1.5, 4.5, 0.5)), "arrastrar el borde derecho la estira")
    drag(win, (-0.5, -1.5), (0.5, -2.0))
    check(same(win.box, GearBox(0.5, -2.0, 4.5, 0.5)), "arrastrar una esquina cambia dos lados")
    render(win, "/tmp/shot_boxedit_1_editada.png")
    edited = win.box
    assert edited is not None

    win._undo()
    check(same(win.box, GearBox(-0.5, -1.5, 4.5, 0.5)), "deshacer vuelve a la caja anterior")
    win._undo()
    win._undo()
    check(same(win.box, GearBox(-2.0, -1.0, 2.0, 1.0)), "deshacer tres veces vuelve a la dibujada")
    win._redo()
    win._redo()
    win._redo()
    check(same(win.box, edited), "rehacer reconstruye la caja editada")

    path = tmp / "caja.json"
    check(win.save_to(str(path)), "guardar el proyecto")
    win.box_editor.clear()
    check(win.box is None, "quitar la caja")
    check(win.open_from(str(path)) and same(win.box, edited, 1e-6),
          "abrir el proyecto restaura la caja editada")

    drag(win, (6.0, 3.0), (7.0, 3.4))
    check(same(win.box, GearBox(6.0, 3.0, 7.0, 3.4)), "arrastrar fuera de la caja dibuja una nueva")

    win.box_toggle.set_active(False)
    check(not win.box_editor.active and win.canvas.drag_target is win.editor,
          "al desactivar «Caja» se ocultan las manijas y se vuelve a editar la curva")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokeboxedit")

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
