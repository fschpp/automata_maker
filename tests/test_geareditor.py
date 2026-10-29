"""Mouse del editor de la red de engranajes (M5, bundle 6)."""

import pytest

from mechchar.core.gearedit import GearNetModel
from mechchar.core.viewport import Viewport
from mechchar.ui.gear_editor import DRAG_PX, HIT_PX, GearEditor, snap

VP = Viewport(scale=30.0, ox=100.0, oy=300.0)


def make():
    m = GearNetModel.example()
    calls = []
    return m, GearEditor(m, VP, lambda: calls.append(1)), calls


def at(model, i):
    return VP.world_to_screen(model.axes[i].x, model.axes[i].y)


def test_pressing_an_axis_claims_the_drag_and_empty_space_does_not():
    m, ed, _ = make()
    assert ed.drag_begin(*at(m, 1)) is True
    ed.drag_end()
    assert ed.drag_begin(700.0, 20.0) is False


def test_the_hit_radius_is_in_pixels():
    m, ed, _ = make()
    x, y = at(m, 2)
    assert ed.drag_begin(x + HIT_PX - 1, y) is True
    ed.drag_end()
    assert ed.drag_begin(x + HIT_PX + 3, y) is False


def test_a_click_selects_and_clicking_again_deselects():
    m, ed, calls = make()
    x, y = at(m, 1)
    ed.drag_begin(x, y)
    ed.drag_update(x + 1.0, y - 1.0)  # una sacudida menor que DRAG_PX no es arrastre
    ed.drag_end()
    assert m.selected == [1] and (m.axes[1].x, m.axes[1].y) == (9.0, 1.0) and len(calls) == 1
    ed.drag_begin(x, y)
    ed.drag_end()
    assert m.selected == []


def test_dragging_moves_the_axis_in_world_units_and_does_not_select():
    m, ed, calls = make()
    x, y = at(m, 3)
    ed.drag_begin(x, y)
    ed.drag_update(
        x + 60.0, y - 30.0
    )  # 2 unidades a la derecha y 1 hacia arriba (y de pantalla baja)
    ed.drag_end()
    assert (m.axes[3].x, m.axes[3].y) == pytest.approx((15.0, 10.0))
    assert m.selected == [] and calls


def test_small_movements_below_the_threshold_do_not_move_it():
    m, ed, _ = make()
    x, y = at(m, 0)
    ed.drag_begin(x, y)
    ed.drag_update(x + DRAG_PX - 1.0, y)
    ed.drag_end()
    assert (m.axes[0].x, m.axes[0].y) == (0.0, 0.0)


def test_dragged_positions_snap_to_the_grid_step():
    m, ed, _ = make()
    x, y = at(m, 3)
    ed.drag_begin(x, y)
    ed.drag_update(x + 31.3, y + 17.9)
    ed.drag_end()
    assert m.axes[3].x == pytest.approx(round(m.axes[3].x, 1))
    assert m.axes[3].y == pytest.approx(round(m.axes[3].y, 1))
    assert snap(0.04) == 0.0 and snap(0.06) == pytest.approx(0.1)


def test_add_mode_adds_an_axis_on_a_click_in_empty_space_only():
    m, ed, calls = make()
    ed.add_mode = True
    n = len(m.axes)
    sx, sy = VP.world_to_screen(20.0, 5.0)
    assert ed.drag_begin(sx, sy) is True
    ed.drag_update(sx + 2.0, sy)  # no es arrastre: el clic cuenta igual
    ed.drag_end()
    assert len(m.axes) == n + 1 and (m.axes[-1].x, m.axes[-1].y) == pytest.approx((20.0, 5.0))
    assert calls
    # sobre un eje existente, en modo agregar, sigue siendo seleccionarlo
    ed.drag_begin(*at(m, 0))
    ed.drag_end()
    assert len(m.axes) == n + 1 and m.selected == [0]


def test_without_add_mode_a_click_in_empty_space_adds_nothing():
    m, ed, _ = make()
    n = len(m.axes)
    assert ed.drag_begin(700.0, 20.0) is False
    ed.drag_end()
    assert len(m.axes) == n


def test_two_clicks_then_connect_builds_a_link():
    m, ed, _ = make()
    for i in (0, 3):
        ed.drag_begin(*at(m, i))
        ed.drag_end()
    assert m.can_connect_selected()
    assert m.connect_selected() == (0, 3)


def test_once_dragging_the_axis_follows_the_mouse_even_back_near_where_it_started():
    m, ed, _ = make()
    x, y = at(m, 3)
    ed.drag_begin(x, y)
    ed.drag_update(x + 60.0, y)  # lejos: ya es arrastre
    ed.drag_update(x + 2.0, y)  # vuelve cerca del punto de partida, sin soltar
    ed.drag_end()
    assert m.axes[3].x == pytest.approx(13.0 + 2.0 / 30.0, abs=0.06)
    assert m.selected == []  # fue un arrastre, no un clic
