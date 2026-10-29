"""La ventana «Red de engranajes» ejecutada contra un GTK de mentira (M5, bundle 6).

No prueba cómo se ve (eso es de ``scripts/smoke_gears.py`` con GTK de verdad): prueba que el
código de la ventana se construye, que cada control hace lo que dice y que las filas del panel
siguen al modelo."""

import types

import pytest

from tests import fakegtk
from tests.test_geardraw import Recorder


@pytest.fixture
def win():
    with fakegtk.installed():
        from mechchar.ui import gear_scene

        yield gear_scene.GearWindow(None)


def click_axis(win, i):
    ed = win.editor
    sx, sy = win.canvas.viewport.world_to_screen(win.model.axes[i].x, win.model.axes[i].y)
    assert ed.drag_begin(sx, sy)
    ed.drag_end()


def pair_rows(win):
    return [(r["i"], r["j"], r["auto"]) for r in win.link_rows]


# --- construcción ------------------------------------------------------------------------------


def test_opens_with_the_example_and_consistent_rows(win):
    assert len(win.model.axes) == 4 and len(win.axis_rows) == 4
    assert len(win.link_rows) == 3 and all(r["auto"] for r in win.link_rows)
    assert win.status.get_text()
    assert not win.connect_button.get_sensitive() and not win.clear_button.get_sensitive()
    assert win.canvas.drag_target is win.editor
    assert (
        win.axis_rows[0]["radio"].get_active() and not win.axis_rows[0]["reverse"].get_sensitive()
    )
    assert win.axis_rows[2]["reverse"].get_active()  # «pierna der.» gira al revés


def test_the_canvas_overlay_and_the_side_view_draw_without_errors(win):
    cr = Recorder()
    win.canvas.overlay(cr, win.canvas.viewport)
    assert cr.count("close_path") == len(win._outcome.train.gears())
    side = Recorder()
    win._draw_side(None, side, 600, 170)
    assert side.count("rectangle") > 0


# --- elegir y conectar -------------------------------------------------------------------------


def test_two_clicks_and_connect_adds_a_chosen_pair(win):
    click_axis(win, 0)
    click_axis(win, 3)
    assert win.model.selected == [0, 3] and win.connect_button.get_sensitive()
    win.connect_button.emit("clicked")
    assert win.model.links == [(0, 3)] and win.model.selected == []
    assert pair_rows(win)[0] == (0, 3, False) and win.clear_button.get_sensitive()
    assert not win.connect_button.get_sensitive()


def test_a_repeated_pair_is_a_message_not_an_exception(win):
    win.model.connect(0, 3)
    win._refresh(rebuild=True)
    click_axis(win, 3)
    click_axis(win, 0)
    win.connect_button.emit("clicked")
    assert "ya están conectados" in win.status.get_label()
    assert win.model.links == [(0, 3)]


def test_fix_an_auto_pair_and_remove_a_chosen_one(win):
    i, j, auto = pair_rows(win)[0]
    assert auto
    win.link_rows[0]["button"].emit("clicked")
    assert frozenset((i, j)) in {frozenset(p) for p in win.model.links}
    row = next(r for r in win.link_rows if not r["auto"])
    row["button"].emit("clicked")
    assert win.model.links == []


def test_clear_links_button(win):
    win.model.connect(0, 3)
    win._refresh(rebuild=True)
    win.clear_button.emit("clicked")
    assert win.model.links == [] and not win.clear_button.get_sensitive()


# --- mover -------------------------------------------------------------------------------------


def test_dragging_keeps_the_rows_in_step_with_the_automatic_pairs(win):
    ed = win.editor
    sx, sy = win.canvas.viewport.world_to_screen(win.model.axes[3].x, win.model.axes[3].y)
    ed.drag_begin(sx, sy)
    # lo llevo pegado a «entrada»: cambian los pares más cortos que se completan solos
    tx, ty = win.canvas.viewport.world_to_screen(1.0, 1.0)
    ed.drag_update(tx, ty)
    ed.drag_end()
    assert (win.model.axes[3].x, win.model.axes[3].y) == pytest.approx((1.0, 1.0))
    chosen = {frozenset(p) for p in win.model.links}
    assert pair_rows(win) == [
        (i, j, frozenset((i, j)) not in chosen) for i, j in win._outcome.links
    ]
    assert (0, 3) in win._outcome.links or (3, 0) in win._outcome.links


# --- intermedios -------------------------------------------------------------------------------


def test_the_intermediates_selector_forces_and_releases(win):
    win.model.connect(0, 1)
    win._refresh(rebuild=True)
    row = next(r for r in win.link_rows if not r["auto"])
    pair = frozenset((row["i"], row["j"]))
    row["choice"].set_selected(4)  # opción 4 = 3 intermedios
    assert win.model.forced[pair] == 3
    assert "intermedio" in row["info"].get_text()
    row["choice"].set_selected(0)
    assert pair not in win.model.forced


def test_a_wrong_parity_shows_the_reason_in_the_status(win):
    win.model.connect(0, 1)
    win._refresh(rebuild=True)
    row = next(r for r in win.link_rows if not r["auto"])
    row["choice"].set_selected(3)  # 2 intermedios entre ejes que giran igual
    assert not win._outcome.ok and "intermedios" in win.status.get_text()
    row["choice"].set_selected(0)
    assert win._outcome.ok


# --- ejes --------------------------------------------------------------------------------------


def test_changing_the_input_makes_it_the_reference(win):
    win.axis_rows[2]["radio"].set_active(True)
    assert win.model.root == 2 and win.model.axes[2].same_direction
    assert not win.axis_rows[2]["reverse"].get_sensitive()
    assert win.axis_rows[0]["reverse"].get_sensitive()
    assert sum(r["radio"].get_active() for r in win.axis_rows) == 1
    assert win._outcome.ok


def test_reverse_checkbox_and_the_input_guard(win):
    win.axis_rows[1]["reverse"].set_active(True)
    assert win.model.axes[1].same_direction is False
    assert win._outcome.ok
    win.axis_rows[1]["reverse"].set_active(False)
    assert win.model.axes[1].same_direction is True


def test_renaming_does_not_rebuild_the_row_and_updates_the_pair_labels(win):
    entry = win.axis_rows[0]["entry"]
    entry.set_text("motor")
    assert win.model.axes[0].name == "motor" and win.axis_rows[0]["entry"] is entry
    assert any("motor" in r["label"].get_text() for r in win.link_rows)
    entry.set_text("  ")  # un nombre vacío se ignora
    assert win.model.axes[0].name == "motor"


def test_add_mode_and_remove_axis_keep_rows_in_step(win):
    n = len(win.model.axes)
    win.add_toggle.set_active(True)
    sx, sy = win.canvas.viewport.world_to_screen(16.0, 3.0)
    assert win.editor.drag_begin(sx, sy)
    win.editor.drag_end()
    assert len(win.model.axes) == n + 1 and len(win.axis_rows) == n + 1
    win.add_toggle.set_active(False)
    assert not win.editor.add_mode
    win.axis_rows[-1]["remove"].emit("clicked")
    assert len(win.model.axes) == n and len(win.axis_rows) == n
    assert all(0 <= i < n and 0 <= j < n for i, j, _ in pair_rows(win))


def test_removing_an_axis_with_chosen_pairs_drops_them(win):
    win.model.connect(0, 3)
    win._refresh(rebuild=True)
    win.axis_rows[3]["remove"].emit("clicked")
    assert win.model.links == [] and len(win.axis_rows) == 3


# --- red, profundidad y animación --------------------------------------------------------------


def test_auto_off_without_pairs_explains_what_is_missing(win):
    win.auto_check.set_active(False)
    assert not win.model.auto and not win._outcome.ok
    assert "faltan" in win.status.get_text() and win.link_rows == []
    win.auto_check.set_active(True)
    assert win._outcome.ok and len(win.link_rows) == 3


def test_the_axle_list_follows_the_protrusion(win):
    assert "sobresale" in win.axles_label.get_text()
    win.protrusion.set_value(6.0)
    short = win.axles_label.get_text()
    win.protrusion.set_value(0.5)
    assert short != win.axles_label.get_text()
    win.model.clear_links()
    win.auto_check.set_active(False)
    assert win.axles_label.get_text() == "Sin engranajes todavía."


def test_play_advances_the_angle_and_the_slider_sets_it(win):
    frames = iter([0.0, 1.0, 2.0])

    def clock():
        return types.SimpleNamespace(get_frame_time=lambda: int(next(frames) * 1e6))

    win.play.set_active(True)
    win.tick(win, clock())  # el primer cuadro solo toma la hora
    t0 = win.theta
    win.tick(win, clock())
    assert win.theta > t0
    win.play.set_active(False)
    win.scale.set_value(90.0)
    assert win.theta == pytest.approx(1.5707963, abs=1e-3)


def test_replace_loads_a_whole_model_and_rebuilds(win):
    from mechchar.core.gearedit import GearNetModel

    win._replace(GearNetModel())
    assert win.axis_rows == [] and "Agregá ejes" in win.status.get_text()
    win._replace(GearNetModel.example())
    assert len(win.axis_rows) == 4 and win._outcome.ok
    assert win.editor.model is win.model  # el editor sigue mirando el mismo modelo


def test_without_play_the_frames_do_not_advance_the_angle(win):
    times = iter(range(1, 10))

    def clock():
        return types.SimpleNamespace(get_frame_time=lambda: int(next(times) * 1e6))

    for _ in range(4):
        win.tick(win, clock())
    assert win.theta == 0.0
