"""Prueba de humo de la UI sin pantalla: arrastra un punto de control, ajusta un mecanismo (se
guarda una captura a mitad del ajuste con la curva en evaluación) y anima el resultado; guarda
capturas en /tmp/shot_*.png.

Uso (necesita xvfb y GTK4):
    GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a /usr/bin/python3 scripts/smoke_ui.py
"""

import sys
import time

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.ui.window import MainWindow


def snap(win, path):
    w, h = win.get_width(), win.get_height()
    paintable = Gtk.WidgetPaintable.new(win)
    snapshot = Gtk.Snapshot()
    paintable.snapshot(snapshot, w, h)
    tex = win.get_native().get_renderer().render_texture(snapshot.to_node(), None)
    tex.save_to_png(path)


def on_activate(app):
    win = MainWindow(app)
    win.present()
    state = {"t0": None}

    def drag_point():
        vp, ed = win.canvas.viewport, win.editor
        sx, sy = vp.world_to_screen(*ed.points[2])  # punto de arriba
        assert ed.drag_begin(sx, sy), "no se pudo agarrar el punto de control"
        ed.drag_update(sx + 20, sy - 30)
        ed.drag_end()
        print("punto 2 ->", tuple(round(v, 2) for v in ed.points[2]))
        win.canvas.queue_draw()
        GLib.timeout_add(400, start_fit)  # dejar que la ventana redibuje antes de la captura
        return False

    def start_fit():
        snap(win, "/tmp/shot_1_antes.png")
        win.fit_button.emit("clicked")
        state["t0"] = time.time()
        GLib.timeout_add(500, poll)
        return False

    def poll():
        if win.runner.busy and time.time() - state["t0"] < 90:
            if win.progress is not None and not state.get("shot_progress"):
                state["shot_progress"] = True
                print("progreso visible:", len(win.progress), "puntos |", win.status.get_text())
                snap(win, "/tmp/shot_1b_progreso.png")
            return True
        print("curva de progreso mostrada durante el ajuste:", bool(state.get("shot_progress")))
        print("estado:", win.status.get_text())
        print("mecanismo:", win.mech is not None, "| tiempo de ajuste:", round(time.time() - state["t0"], 1), "s")
        GLib.timeout_add(1200, finish)
        return False

    def finish():
        pb = win.playback
        print("fase animada:", None if pb is None else round(pb.t, 3), "ok =", None if pb is None else pb.ok)
        snap(win, "/tmp/shot_2_despues.png")
        app.quit()
        return False

    GLib.timeout_add(1500, drag_point)


app = Gtk.Application(application_id="io.github.mechchar.smoke")
app.connect("activate", on_activate)
sys.exit(app.run([]))
