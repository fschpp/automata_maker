"""Prueba de humo de la ventana «Red de engranajes» (M5) sin pantalla: clics y arrastres sobre los
ejes, conectar, intermedios, cambio de entrada, sentido de giro, agregar y quitar ejes, la
animación y la vista lateral. Dibuja la escena a /tmp/shot_gears_*.png.

Uso: GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_gears.py
"""

import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from mechchar.core.gearedit import GearNetModel
from mechchar.ui.gear_scene import GearWindow

fail: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FALLA"), msg)
    if not cond:
        fail.append(msg)


def click_axis(win, i: int) -> None:
    ed = win.editor
    sx, sy = win.canvas.viewport.world_to_screen(win.model.axes[i].x, win.model.axes[i].y)
    check(ed.drag_begin(sx, sy), f"el eje {i} reclama el arrastre")
    ed.drag_end()


def drag_axis(win, i: int, dx_px: float, dy_px: float) -> None:
    ed = win.editor
    sx, sy = win.canvas.viewport.world_to_screen(win.model.axes[i].x, win.model.axes[i].y)
    ed.drag_begin(sx, sy)
    ed.drag_update(sx + dx_px, sy + dy_px)
    ed.drag_end()


def render(win, path: str) -> None:
    import cairo

    w, h = 900, 560
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h + 170)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    win.canvas.overlay(cr, win.canvas.viewport)
    cr.save()
    cr.translate(0, h)
    win._draw_side(win.side, cr, w, 170)
    cr.restore()
    surface.write_to_png(path)
    print("escena dibujada en", path)


def script(win: GearWindow):
    m = win.model
    check(len(m.axes) == 4 and win.status.get_text() != "", "abre con el ejemplo y un estado")
    check(len(win.axis_rows) == 4 and len(win.link_rows) == 3, "4 filas de ejes y 3 pares (auto)")
    check(all(r["auto"] for r in win.link_rows), "los pares iniciales se completaron solos")
    check(not win.connect_button.get_sensitive(), "«Conectar» deshabilitado sin selección")
    check(not win.clear_button.get_sensitive(), "«Quitar los pares» deshabilitado sin pares")
    render(win, "/tmp/shot_gears_0.png")

    # elegir un par con dos clics y conectarlo
    click_axis(win, 0)
    click_axis(win, 3)
    check(m.selected == [0, 3], "dos clics seleccionan dos ejes")
    check(win.connect_button.get_sensitive(), "«Conectar» se habilita con dos ejes")
    win.connect_button.emit("clicked")
    check(m.links == [(0, 3)] and m.selected == [], "«Conectar» agrega el par y limpia la selección")
    check(any(not r["auto"] for r in win.link_rows), "el par elegido aparece como no automático")
    check(win.clear_button.get_sensitive(), "«Quitar los pares» se habilita")
    render(win, "/tmp/shot_gears_1.png")

    # conectar de nuevo el mismo par: un aviso, no una excepción
    click_axis(win, 3)
    click_axis(win, 0)
    win.connect_button.emit("clicked")
    check("ya están conectados" in win.status.get_label(), "un par repetido se avisa en el estado")
    m.clear_selection()
    win._refresh(rebuild=False)

    # arrastrar un eje
    x0, y0 = m.axes[3].x, m.axes[3].y
    drag_axis(win, 3, 60.0, -30.0)
    check((m.axes[3].x, m.axes[3].y) != (x0, y0), "arrastrar mueve el eje")
    check(win.status.get_label() != "", "el estado se actualiza al mover")

    # intermedios forzados
    row = next(r for r in win.link_rows if not r["auto"])
    row["choice"].set_selected(4)  # 3 intermedios
    yield ("sleep", 100)
    pair = frozenset((row["i"], row["j"]))
    check(m.forced.get(pair) == 3, "el selector fuerza 3 intermedios")
    row["choice"].set_selected(3)  # 2 intermedios: no cierra la paridad si giran igual
    yield ("sleep", 100)
    print("estado con paridad dudosa:", win.status.get_text())
    row["choice"].set_selected(0)
    yield ("sleep", 100)
    check(pair not in m.forced, "«Auto» vuelve a dejar los intermedios libres")

    # cambiar la entrada y el sentido de giro
    win.axis_rows[2]["radio"].set_active(True)
    yield ("sleep", 100)
    check(m.root == 2 and m.axes[2].same_direction, "elegir otra entrada la deja como referencia")
    check(not win.axis_rows[2]["reverse"].get_sensitive(), "la entrada no puede ir al revés")
    win.axis_rows[1]["reverse"].set_active(not win.axis_rows[1]["reverse"].get_active())
    yield ("sleep", 100)
    check(len(win.axis_rows) == 4, "las filas se reconstruyen sin perder ejes")

    # renombrar sin reconstruir (conserva el foco)
    entry = win.axis_rows[0]["entry"]
    entry.set_text("motor")
    yield ("sleep", 100)
    check(m.axes[0].name == "motor" and win.axis_rows[0]["entry"] is entry,
          "renombrar actualiza el modelo sin reconstruir la fila")

    # agregar un eje con el mouse
    win.add_toggle.set_active(True)
    n = len(m.axes)
    sx, sy = win.canvas.viewport.world_to_screen(16.0, 3.0)
    check(win.editor.drag_begin(sx, sy), "en modo agregar un clic en vacío reclama")
    win.editor.drag_end()
    yield ("sleep", 100)
    check(len(m.axes) == n + 1 and len(win.axis_rows) == n + 1, "se agregó un eje y su fila")
    win.add_toggle.set_active(False)
    render(win, "/tmp/shot_gears_2.png")

    # quitar un eje desde el panel
    win.axis_rows[-1]["remove"].emit("clicked")
    yield ("sleep", 100)
    check(len(m.axes) == n and len(win.axis_rows) == n, "«✕» quita el eje y su fila")

    # profundidad de los ejes
    win.protrusion.set_value(5.0)
    yield ("sleep", 100)
    check("sobresale" in win.axles_label.get_text(), "el eje de entrada figura como sobresaliente")

    # animación
    win.play.set_active(True)
    t0 = win.theta
    yield ("sleep", 800)
    check(win.theta != t0, "«Girar» avanza el ángulo")
    win.play.set_active(False)
    win.scale.set_value(90.0)
    check(abs(win.theta - 1.5707963) < 1e-3, "el control deslizante fija el ángulo")

    # automático apagado: los pares tienen que unir todo
    win.auto_check.set_active(False)
    yield ("sleep", 100)
    m.clear_links()
    win._refresh(rebuild=True)
    check(not win._outcome.ok and "faltan" in win.status.get_label(),
          "sin automático y sin pares el estado explica qué falta")
    win.auto_check.set_active(True)

    # ejemplo y vaciar
    win._replace(GearNetModel())
    yield ("sleep", 100)
    check(len(win.axis_rows) == 0 and "Agregá ejes" in win.status.get_label(),
          "una escena vacía lo dice")
    win._replace(GearNetModel.example())
    yield ("sleep", 100)
    check(len(win.axis_rows) == 4, "«Ejemplo» vuelve a cargar los 4 ejes")
    render(win, "/tmp/shot_gears_3.png")


def main() -> None:
    app = Gtk.Application(application_id="ar.mechchar.smokegears")

    def on_activate(app):
        win = GearWindow(app)
        win.present()
        gen = script(win)

        def finish():
            print("FALLAS:", fail if fail else "ninguna")
            app.quit()
            sys.exit(1 if fail else 0)

        def pump():
            try:
                step = next(gen)
            except StopIteration:
                return finish()
            except Exception:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                fail.append("excepción")
                return finish()
            GLib.timeout_add(step[1], pump)
            return False

        GLib.timeout_add(800, pump)

    app.connect("activate", on_activate)
    app.run([])


main()
