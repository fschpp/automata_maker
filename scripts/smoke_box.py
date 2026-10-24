"""Prueba de humo de la caja de engranajes sin pantalla: dibuja la caja arrastrando, comprueba que
las sugerencias y el ajuste caben, y guarda /tmp/shot_box_*.png (dibujado con cairo).

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_box.py
"""

import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.optim.curves import bbox_diag
from mechchar.ui.window import MainWindow


def render(win, path):
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 900, 600)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    win.canvas.viewport.fit_bounds(-14, -14, 14, 10, 900, 600)
    win._draw_scene(cr, win.canvas.viewport)
    surface.write_to_png(path)
    print("escena dibujada en", path)


def on_activate(app):
    win = MainWindow(app, dbs=[])
    win.present()
    fail = []
    state = {"step": 0}

    def tick():
        if not win.dbs:
            return True  # esperando la base de movimientos
        if state["step"] == 0:
            state["step"] = 1
            curve = win.editor.curve(per_segment=24)
            state["size"] = bbox_diag(curve)
            win.suggestions = []
            win._request_suggestions()
            GLib.timeout_add(1500, tick)
            return False
        if state["step"] == 1:
            free = list(win.suggestions)
            print("sin caja:", [None if s.bounds is None else [round(v, 1) for v in s.bounds] for s in free])
            if not free or any(s.bounds is None for s in free):
                fail.append("sin caja no hay sugerencias con rectángulo")
            render(win, "/tmp/shot_box_0_sin_caja.png")
            # caja dibujada con el mouse: arrastre de esquina a esquina en pantalla
            vp = win.canvas.viewport
            pad = 1.0 * state["size"]
            xs, ys = vp.world_to_screen(-3 - pad, 1.5 + pad), vp.world_to_screen(3 + pad, -1.5 - pad)
            win.box_toggle.set_active(True)
            assert win.canvas.drag_target is win.box_editor
            win.box_editor.drag_begin(*xs)
            win.box_editor.drag_update(*ys)
            win.box_editor.drag_end()
            print("caja:", win.box, "|", win.status.get_text())
            if win.box is None:
                fail.append("la caja no se creó")
                return finish()
            state["step"] = 2
            GLib.timeout_add(1500, tick)
            return False
        if state["step"] == 2:
            print("con caja:", [None if s.bounds is None else [round(v, 1) for v in s.bounds] for s in win.suggestions])
            for s in win.suggestions:
                if not win.box.contains(s.bounds):
                    fail.append("una sugerencia se sale de la caja")
            win.box_toggle.set_active(False)
            win._start_fit(None)
            state["step"] = 3
            GLib.timeout_add(500, tick)
            return False
        if state["step"] == 3:
            if win.runner.busy:
                return True
            print("estado:", win.status.get_text())
            if win.mech is None or win.mech_bounds is None:
                fail.append("no hubo ajuste dentro de la caja")
            elif win.box.worst_overshoot(win.mech_bounds) > win.box.slack:
                fail.append(f"el mecanismo se sale de la caja: {win.box.overshoot(win.mech_bounds)}")
            render(win, "/tmp/shot_box_1_con_caja.png")
            return finish()
        return False

    def finish():
        print("FALLAS:", fail if fail else "ninguna")
        app.quit()
        sys.exit(1 if fail else 0)

    GLib.timeout_add(1000, tick)


app = Gtk.Application(application_id="ar.mechchar.smokebox")
app.connect("activate", on_activate)
app.run([])
