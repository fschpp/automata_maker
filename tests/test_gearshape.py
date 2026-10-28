"""Dientes, fases de montaje y giro de la red de engranajes (M5, bundle 4)."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from mechchar.core.gearnet import Axis, plan_train, simulate_train
from mechchar.core.gearshape import (
    chain_phases,
    connection_rotations,
    gear_outline,
    gear_rotations,
    mesh_phase,
    root_radius,
    tip_radius,
)
from mechchar.core.geartrain import Gear
from mechchar.core.viewport import Viewport
from mechchar.ui.scene import draw_gear_train

AXES = [Axis("entrada", 0, 0), Axis("b", 9, 1), Axis("c", 4, 8, same_direction=False)]
THETAS = np.linspace(0, 2 * math.pi, 49)


@pytest.fixture(scope="module")
def train():
    t = plan_train(AXES, 0)
    assert t.ok and len(t.connections) == 2
    return t


# --- contorno ----------------------------------------------------------------------------------


def test_outline_has_four_vertices_per_tooth_between_root_and_tip():
    g = Gear(3.0, -2.0, 14, 0.5)
    pts = gear_outline(g)
    assert pts.shape == (4 * 14, 2)
    r = np.hypot(pts[:, 0] - g.x, pts[:, 1] - g.y)
    assert r.min() == pytest.approx(root_radius(g)) and r.max() == pytest.approx(tip_radius(g))
    assert root_radius(g) < g.radius < tip_radius(g)


def test_outline_is_a_valid_counterclockwise_polygon_with_one_tooth_at_the_rotation():
    g = Gear(0.0, 0.0, 12, 1.0)
    poly = Polygon(gear_outline(g, 0.3))
    assert poly.is_valid and poly.exterior.is_ccw
    # el centro del primer diente (cabeza) queda en el ángulo de rotación
    c = np.mean(gear_outline(g, 0.3)[1:3], axis=0)
    assert math.atan2(c[1], c[0]) == pytest.approx(0.3, abs=1e-9)


def test_outline_area_grows_with_teeth_at_the_same_module():
    small = Polygon(gear_outline(Gear(0, 0, 10, 1.0))).area
    big = Polygon(gear_outline(Gear(0, 0, 20, 1.0))).area
    assert 3.0 < big / small < 5.0  # ~ (z2 / z1)^2 = 4


# --- fases y giro ------------------------------------------------------------------------------


def test_mesh_phase_puts_a_gap_where_the_neighbour_has_a_tooth():
    a, b = Gear(0, 0, 16, 1.0), Gear(16.0, 0.0, 16, 1.0)
    rot_b = mesh_phase(a, 0.0, b)  # a tiene un diente justo hacia b (ángulo 0)
    # hacia a (ángulo π desde b) b tiene un hueco: su diente más cercano queda a medio paso
    off = (math.pi - rot_b) % (2 * math.pi / 16)
    assert off == pytest.approx(math.pi / 16, abs=1e-9)


def test_first_gear_of_each_chain_starts_at_zero(train):
    for c in train.connections:
        assert chain_phases(c.gears)[0] == 0.0


def test_neighbouring_gears_never_overlap_while_the_train_turns(train):
    for th in THETAS:
        for c, rots in zip(train.connections, gear_rotations(train, th), strict=True):
            polys = [Polygon(gear_outline(g, r)) for g, r in zip(c.gears, rots, strict=True)]
            for p, q in zip(polys, polys[1:], strict=False):
                assert (
                    p.intersection(q).area
                    < 0.01 * Polygon(gear_outline(c.gears[0])).area / c.gears[0].teeth
                )


def test_without_the_mesh_phase_neighbours_do_overlap(train):
    c = train.connections[0]
    polys = [Polygon(gear_outline(g, 0.0)) for g in c.gears]
    worst = max(p.intersection(q).area for p, q in zip(polys, polys[1:], strict=False))
    tooth = Polygon(gear_outline(c.gears[0])).area / c.gears[0].teeth
    assert worst > 0.05 * tooth  # el chequeo de arriba distingue una red mal montada


def test_rotations_match_the_simulated_axis_angles(train):
    sim = simulate_train(train, 12)
    step = 5
    theta = sim[step, train.root]
    for c, rots0, rots in zip(
        train.connections, gear_rotations(train, 0.0), gear_rotations(train, theta), strict=True
    ):
        turned_first = rots[0] - rots0[0]
        turned_last = rots[-1] - rots0[-1]
        assert turned_first == pytest.approx(sim[step, c.a] - sim[0, c.a], abs=1e-9)
        assert turned_last == pytest.approx(sim[step, c.b] - sim[0, c.b], abs=1e-9)


def test_intermediates_turn_the_opposite_way_and_by_the_teeth_ratio(train):
    c = max(train.connections, key=lambda c: len(c.gears))
    d = [
        x - y
        for x, y in zip(
            connection_rotations(train, c, 0.1), connection_rotations(train, c, 0.0), strict=True
        )
    ]
    for k, g in enumerate(c.gears):
        want = (-1.0) ** k * c.gears[0].teeth / g.teeth * d[0]
        assert d[k] == pytest.approx(want)
    assert d[0] * d[1] < 0


# --- dibujo ------------------------------------------------------------------------------------


class Recorder:
    """Un ``cr`` de mentira que anota las llamadas."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call

    def colors(self):
        return [a for n, a in self.calls if n == "set_source_rgba"]


def test_draw_gear_train_draws_every_gear_with_its_teeth(train):
    cr = Recorder()
    draw_gear_train(cr, Viewport(scale=30.0, ox=100.0, oy=300.0), train, 0.2)
    n_gears = len(train.gears())
    assert sum(1 for n, _ in cr.calls if n == "close_path") == n_gears
    # un move_to + 4·z líneas por engranaje
    lines = sum(1 for n, _ in cr.calls if n == "line_to")
    assert lines == sum(4 * g.teeth - 1 for g in train.gears())


def test_selected_axes_are_dark_blue_and_the_input_orange(train):
    from mechchar.ui.scene import INPUT_FILL, SELECTED_FILL

    plain = Recorder()
    draw_gear_train(plain, Viewport(), train)
    assert INPUT_FILL in plain.colors() and SELECTED_FILL not in plain.colors()
    sel = Recorder()
    draw_gear_train(sel, Viewport(), train, selected={2})
    assert SELECTED_FILL in sel.colors()


@pytest.mark.parametrize("z_a, z_b, angle", [(15, 15, 0.0), (13, 17, 0.7), (15, 11, -2.1)])
def test_odd_teeth_and_oblique_neighbours_mesh_without_overlap(z_a, z_b, angle):
    m = 1.0
    a = Gear(0.0, 0.0, z_a, m)
    d = m * (z_a + z_b) / 2.0
    b = Gear(d * math.cos(angle), d * math.sin(angle), z_b, m)
    rot_a = 0.37
    pa = Polygon(gear_outline(a, rot_a))
    pb = Polygon(gear_outline(b, mesh_phase(a, rot_a, b)))
    tooth = pa.area / z_a
    assert pa.intersection(pb).area < 0.01 * tooth
    wrong = Polygon(gear_outline(b, mesh_phase(a, rot_a, b) + math.pi / z_b))  # tooth on tooth
    assert pa.intersection(wrong).area > 0.05 * tooth
