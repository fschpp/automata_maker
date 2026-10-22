import math

import pytest

from mechchar.core.viewport import MAX_SCALE, MIN_SCALE, Viewport, grid_lines, nice_step


def test_roundtrip():
    vp = Viewport(scale=37.0, ox=120.0, oy=80.0)
    for p in [(0.0, 0.0), (3.2, -1.5), (-10.0, 7.25)]:
        assert vp.screen_to_world(*vp.world_to_screen(*p)) == pytest.approx(p)


def test_y_axis_points_up():
    vp = Viewport(scale=10.0, ox=0.0, oy=100.0)
    _, sy = vp.world_to_screen(0.0, 5.0)
    assert sy < 100.0


def test_pan():
    vp = Viewport(scale=10.0, ox=0.0, oy=0.0)
    vp.pan_by(15, -4)
    assert vp.world_to_screen(0, 0) == (15, -4)


def test_zoom_keeps_point_under_cursor():
    vp = Viewport(scale=20.0, ox=300.0, oy=200.0)
    cursor = (410.0, 120.0)
    before = vp.screen_to_world(*cursor)
    vp.zoom_at(*cursor, 1.7)
    assert vp.scale == pytest.approx(34.0)
    assert vp.screen_to_world(*cursor) == pytest.approx(before)


def test_zoom_is_clamped():
    vp = Viewport(scale=1.0)
    vp.zoom_at(0, 0, 1e12)
    assert vp.scale == MAX_SCALE
    vp.zoom_at(0, 0, 1e-30)
    assert vp.scale == MIN_SCALE


@pytest.mark.parametrize(
    ("target", "expected"),
    [(0.9, 1), (1, 1), (1.1, 2), (3, 5), (7, 10), (0.03, 0.05), (140, 200)],
)
def test_nice_step(target, expected):
    assert nice_step(target) == pytest.approx(expected)


def test_nice_step_rejects_nonpositive():
    with pytest.raises(ValueError):
        nice_step(0)


def test_grid_lines_cover_view_and_include_origin():
    vp = Viewport(scale=50.0, ox=400.0, oy=300.0)
    step, xs, ys = grid_lines(vp, 800, 600)
    assert step == pytest.approx(1.0)
    assert 0.0 in xs and 0.0 in ys
    assert min(xs) >= -8.0 - 1e-9 and max(xs) <= 8.0 + 1e-9
    assert all(math.isclose(x / step, round(x / step)) for x in xs)


def test_grid_step_adapts_to_zoom():
    vp = Viewport(scale=500.0, ox=0.0, oy=0.0)
    step, _, _ = grid_lines(vp, 800, 600)
    assert step == pytest.approx(0.1)


def test_fit_bounds_centers_and_fits_the_rectangle():
    vp = Viewport()
    vp.fit_bounds(-2.0, -1.0, 6.0, 3.0, 800, 600, margin_px=50)
    assert vp.scale == pytest.approx(min(700 / 8.0, 500 / 4.0))
    sx, sy = vp.world_to_screen(2.0, 1.0)  # el centro del rectángulo cae en el centro de la vista
    assert (sx, sy) == pytest.approx((400.0, 300.0))
    for corner in [(-2.0, -1.0), (6.0, 3.0)]:
        x, y = vp.world_to_screen(*corner)
        assert 50 - 1e-9 <= x <= 750 + 1e-9 and 50 - 1e-9 <= y <= 550 + 1e-9
