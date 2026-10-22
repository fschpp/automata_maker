"""Prueba de humo de la UI sin pantalla: abre la ventana, deja correr la animación, mueve el
deslizador y guarda capturas en /tmp/shot1.png y /tmp/shot2.png.

Uso (necesita xvfb y GTK4):
    GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a /usr/bin/python3 scripts/smoke_ui.py
"""

import sys
import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk
from mechchar.ui.window import MainWindow

shots = []

def snap(win, path):
    w, h = win.get_width(), win.get_height()
    paintable = Gtk.WidgetPaintable.new(win)
    snapshot = Gtk.Snapshot()
    paintable.snapshot(snapshot, w, h)
    node = snapshot.to_node()
    tex = win.get_native().get_renderer().render_texture(node, None)
    tex.save_to_png(path)

def on_activate(app):
    win = MainWindow(app)
    win.present()
    def step1():
        snap(win, "/tmp/shot1.png"); print("t1 =", round(win.playback.t, 3), "ok =", win.playback.ok); return False
    def step2():
        win.scale.set_value(0.5)   # simula al usuario moviendo el deslizador
        return False
    def step3():
        snap(win, "/tmp/shot2.png")
        print("playing =", win.playback.playing, "phase =", round(win.playback.phase, 3), "ok =", win.playback.ok)
        app.quit(); return False
    GLib.timeout_add(1500, step1)
    GLib.timeout_add(1800, step2)
    GLib.timeout_add(2400, step3)

app = Gtk.Application(application_id="io.github.mechchar.smoke")
app.connect("activate", on_activate)
sys.exit(app.run([]))
