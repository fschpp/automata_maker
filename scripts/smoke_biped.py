"""Prueba de humo de la escena «Bípedo» sin pantalla: ajusta una pierna, arma el bípedo, mueve el
slider de desfase y guarda capturas en /tmp/shot_biped_*.png.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_biped.py
"""

import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.core.rig import MultiRig
from mechchar.ui.window import MainWindow


def snap(win, path):
    w, h = win.get_width(), win.get_height()
    paintable = Gtk.WidgetPaintable.new(win)
    snapshot = Gtk.Snapshot()
    paintable.snapshot(snapshot, w, h)
    node = snapshot.to_node()
    if node is None:  # ventana aún sin contenido pintable: se omite la captura
        print("sin captura:", path, w, h)
        return
    win.get_native().get_renderer().render_texture(node, None).save_to_png(path)


def render(win, path):
    """Dibuja la escena directamente en un PNG con cairo (no depende de la ventana)."""
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 900, 600)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    win.canvas.viewport.fit_bounds(-1.5, -3.2, 2.5, 0.6, 900, 600)
    win._draw_scene(cr, win.canvas.viewport)
    surface.write_to_png(path)
    print("escena dibujada en", path)


def on_activate(app):
    win = MainWindow(app, dbs=[])
    win.present()
    fail = []

    def start():
        win.scene_dropdown.set_selected(2)
        assert win.biped and win.phase_row.get_visible()
        render(win, "/tmp/shot_biped_0_reposo.png")
        win._start_fit(None)
        GLib.timeout_add(500, wait_fit)
        return False

    def wait_fit():
        if win.runner.busy:
            return True
        rig = win.mech
        if not isinstance(rig, MultiRig):
            fail.append(f"no se armó el bípedo: {win.status.get_text()}")
            return finish()
        print("estado:", win.status.get_text())
        print("desfases iniciales:", rig.phases, [round(s.get_value(), 2) for s in win.phase_scales])
        assert rig.phases == [0.0, 0.5]
        before = rig.marker_world(win.playback.state, 0).copy()
        win.playback.playing = False
        win.phase_scales[1].set_value(0.8)  # el usuario mueve el slider derecho
        print("desfases tras mover el slider:", rig.phases)
        if abs(rig.phases[1] - 0.8) > 1e-9:
            fail.append("el slider no cambió el desfase")
        if abs(rig.marker_world(win.playback.state, 0) - before).max() > 1e-6:
            fail.append("la pierna izquierda se movió con el slider derecho")
        res = rig.asm.residual(win.playback.state, win.playback.t)
        if abs(res).max() > 1e-6:
            fail.append(f"residuo alto tras mover el slider: {abs(res).max():.2e}")
        win.canvas.queue_draw()
        render(win, "/tmp/shot_biped_1_desfase.png")
        GLib.timeout_add(100, finish)
        return False

    def finish():
        print("FALLAS:", fail if fail else "ninguna")
        app.quit()
        sys.exit(1 if fail else 0)

    GLib.timeout_add(2000, start)


app = Gtk.Application(application_id="ar.mechchar.smokebiped")
app.connect("activate", on_activate)
app.run([])
