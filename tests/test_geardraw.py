"""Dibujo de la escena «Red de engranajes» con un contexto Cairo de mentira (M5, bundle 6)."""

import pytest

from mechchar.core.gearedit import GearNetModel
from mechchar.core.gearnet import Axis, axles, plan_train
from mechchar.core.viewport import Viewport
from mechchar.ui.gear_draw import (
    AUTO_LINK,
    CHOSEN_LINK,
    INPUT_AXLE_FILL,
    draw_editor,
    draw_side_view,
    side_view_bounds,
)
from mechchar.ui.scene import INPUT_FILL, SELECTED_FILL


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)

    def args(self, name):
        return [a for n, a in self.calls if n == name]

    def texts(self):
        return [a[0] for a in self.args("show_text")]


VP = Viewport(scale=30.0, ox=100.0, oy=300.0)


def drawn(model, theta=0.0):
    cr = Recorder()
    draw_editor(cr, VP, model, model.plan(), theta)
    return cr


# --- editor ------------------------------------------------------------------------------------


def test_every_axis_is_drawn_with_its_name_and_reversed_ones_have_the_arrow():
    m = GearNetModel.example()
    texts = drawn(m).texts()
    assert sorted(texts) == sorted([a.name + ("" if a.same_direction else " ↺") for a in m.axes])
    assert any(t.endswith("↺") for t in texts)


def dashed(cr):
    """Cuántas veces se pidió un trazo punteado (``set_dash`` con un patrón no vacío)."""
    return sum(1 for pattern, _ in cr.args("set_dash") if pattern)


def test_chosen_links_are_solid_and_completed_ones_dashed():
    m = GearNetModel.example()
    m.connect(0, 3)
    cr = drawn(m)
    colors = cr.args("set_source_rgb")
    assert CHOSEN_LINK in colors and AUTO_LINK in colors
    assert dashed(cr) == len(m.auto_links()) == 2
    assert cr.args("set_dash")[-1] == ([], 0.0)  # lo que se dibuja después no queda punteado


def test_with_all_pairs_chosen_nothing_is_dashed():
    m = GearNetModel.example()
    for pair in ((0, 1), (0, 2), (1, 3)):
        m.connect(*pair)
    cr = drawn(m)
    assert dashed(cr) == 0 and AUTO_LINK not in cr.args("set_source_rgb")


def ring_colors(cr):
    return [a for a in cr.args("set_source_rgba") if a[:3] in (INPUT_FILL[:3], SELECTED_FILL[:3])]


def test_input_and_selected_axes_get_a_ring():
    m = GearNetModel.example()
    plain = drawn(m)
    m.toggle_select(2)
    sel = drawn(m)
    assert (
        len(ring_colors(sel)) > len(ring_colors(plain)) >= 1
    )  # el aro de la entrada y el nuevo azul


def test_a_failed_plan_still_draws_axes_and_links_but_no_gears():
    m = GearNetModel()
    m.add_axis(0, 0)
    m.add_axis(5, 0)
    m.auto = False  # sin pares no se puede armar
    cr = drawn(m)
    assert cr.count("close_path") == 0 and len(cr.texts()) == 2


def test_the_gears_are_drawn_when_the_train_exists():
    m = GearNetModel.example()
    out = m.plan()
    cr = drawn(m)
    assert cr.count("close_path") == len(out.train.gears())


# --- vista lateral -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train():
    t = plan_train(
        [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 18, 0.5)], 0, links=[(0, 1), (0, 2)]
    )
    assert t.ok
    return t


def side(train, **kw):
    cr = Recorder()
    draw_side_view(cr, 600.0, 200.0, train, **kw)
    return cr


def test_side_view_draws_a_rectangle_per_gear_and_per_axle(train):
    cr = side(train)
    assert cr.count("rectangle") == len(train.gears()) + len(axles(train))


def test_side_view_marks_the_input_axle_and_names_the_axes(train):
    cr = side(train)
    assert INPUT_AXLE_FILL in cr.args("set_source_rgb")
    assert {"a", "b", "c"} <= set(cr.texts())
    assert not any(t.startswith("intermedio") for t in cr.texts())


def test_side_view_fits_inside_the_area(train):
    cr = side(train, thickness=1.5, protrusion=3.0)
    for x, y, w, h in cr.args("rectangle"):
        assert -1e-6 <= x and x + w <= 600.0 + 1e-6
        assert -1e-6 <= y and y + h <= 200.0 + 1e-6


def test_longer_protrusion_makes_the_input_axle_taller_on_screen(train):
    def tallest(cr):
        return max(h for _, _, _, h in cr.args("rectangle"))

    assert tallest(side(train, protrusion=6.0)) > tallest(side(train, protrusion=0.0))


def test_side_view_without_gears_says_so():
    one = plan_train([Axis("a", 0, 0)], 0)
    for t in (None, one):
        cr = side(t)
        assert cr.count("rectangle") == 0 and "Sin engranajes" in cr.texts()[0]


def test_bounds_cover_the_gears_and_the_protruding_axle(train):
    x0, z0, x1, z1 = side_view_bounds(train, 1.0, 2.0)
    assert x0 < 0.0 and x1 > 18.0 and z0 == 0.0
    assert z1 == max(a.z1 for a in axles(train, 1.0, 2.0))
