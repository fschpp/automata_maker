"""Prueba de humo de la UI sin pantalla: muestra los candidatos de la base, arrastra un punto de
control (los candidatos se actualizan), refina el candidato 0 y luego prueba «Ajustar». Guarda
capturas en /tmp/shot_*.png.

Uso (necesita xvfb y GTK4):
    GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a /usr/bin/python3 scripts/smoke_ui.py
"""

import sys
import time

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.db.motion_db import build_database
from mechchar.ui.window import MainWindow


def snap(win, path):
    w, h = win.get_width(), win.get_height()
    paintable = Gtk.WidgetPaintable.new(win)
    snapshot = Gtk.Snapshot()
    paintable.snapshot(snapshot, w, h)
    tex = win.get_native().get_renderer().render_texture(snapshot.to_node(), None)
    tex.save_to_png(path)


def on_activate(app):
    t0 = time.time()
    dbs = [  # bases chicas: la prueba no necesita las de 3000 entradas
        build_database("four_bar", n_target=400, n_samples=12_000, workers=1),
        build_database("geared_five_bar", n_target=300, n_samples=20_000, workers=1),
    ]
    print(f"bases de prueba: {[len(d) for d in dbs]} en {time.time() - t0:.1f} s")
    win = MainWindow(app, dbs=dbs)
    win.present()
    st = {}

    def show_initial():
        print("candidatos iniciales:", [(s.family, round(s.rel_error * 100, 1)) for s in win.suggestions])
        assert len(win.suggestions) == 3, "no llegaron los 3 candidatos"
        print("botones activos:", [b.get_sensitive() for b in win.candidate_buttons])
        snap(win, "/tmp/shot_0_candidatos.png")
        st["before"] = [s.curve.copy() for s in win.suggestions]
        vp, ed = win.canvas.viewport, win.editor
        sx, sy = vp.world_to_screen(*ed.points[2])
        assert ed.drag_begin(sx, sy), "no se pudo agarrar el punto de control"
        ed.drag_update(sx + 20, sy - 30)
        ed.drag_end()
        GLib.timeout_add(900, after_drag)
        return False

    def after_drag():
        moved = any(a.shape != b.shape or abs(a - b).max() > 1e-6 for a, b in
                    zip(st["before"], [s.curve for s in win.suggestions], strict=False))
        print("los candidatos se actualizaron al arrastrar:", moved)
        print("estado:", win.status.get_text()[:120])
        snap(win, "/tmp/shot_1_tras_arrastrar.png")
        win.candidate_buttons[0].emit("clicked")
        st["t0"] = time.time()
        GLib.timeout_add(300, poll_refine)
        return False

    def wait_idle(next_step, label):
        if win.runner.busy and time.time() - st["t0"] < 90:
            return True
        print(f"{label}: {win.status.get_text()}  ({time.time() - st['t0']:.1f} s)")
        GLib.timeout_add(900, next_step)
        return False

    def poll_refine():
        return wait_idle(snap_and_fit, "refinar candidato 0")

    def snap_and_fit():
        print("mecanismo mostrado:", win.mech is not None, "| animando:", win.playback is not None)
        snap(win, "/tmp/shot_2_refinado.png")
        win.fit_button.emit("clicked")
        st["t0"] = time.time()
        GLib.timeout_add(300, poll_fit)
        return False

    def poll_fit():
        return wait_idle(finish, "botón Ajustar")

    def finish():
        pb = win.playback
        print("animación ok:", pb is not None and pb.ok, "| fase:", round(pb.t, 3) if pb else None)
        snap(win, "/tmp/shot_3_ajustar.png")
        app.quit()
        return False

    GLib.timeout_add(1500, show_initial)


app = Gtk.Application(application_id="io.github.mechchar.smoke")
app.connect("activate", on_activate)
sys.exit(app.run([]))
