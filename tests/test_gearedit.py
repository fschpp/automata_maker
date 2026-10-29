"""Modelo de edición de la escena «Red de engranajes» (M5, bundle 6)."""

import math

import pytest

from mechchar.core.gearedit import GearNetModel
from mechchar.core.gearnet import train_problems


def line(n=3, dx=9.0):
    m = GearNetModel()
    for k in range(n):
        m.add_axis(k * dx, 0.5 * (k % 2))
    return m


# --- ejes --------------------------------------------------------------------------------------


def test_axes_get_unique_default_names_and_validate_the_position():
    m = GearNetModel()
    assert [m.add_axis(0, 0), m.add_axis(5, 0)] == [0, 1]
    assert [a.name for a in m.axes] == ["eje 1", "eje 2"]
    m.rename_axis(0, "eje 3")
    assert m.add_axis(9, 0) == 2 and m.axes[2].name == "eje 4"  # no repite «eje 3»
    with pytest.raises(ValueError):
        m.add_axis(math.nan, 0)
    with pytest.raises(ValueError):
        m.move_axis(0, 0, math.inf)
    with pytest.raises(ValueError):
        m.rename_axis(0, "  ")
    with pytest.raises(IndexError):
        m.move_axis(9, 0, 0)


def test_move_axis_changes_only_its_position():
    m = line()
    m.move_axis(1, 2.5, -1.0)
    assert (m.axes[1].x, m.axes[1].y, m.axes[1].name) == (2.5, -1.0, "eje 2")
    assert m.axes[0].x == 0.0


def test_remove_axis_drops_its_links_and_shifts_the_rest():
    m = line(4)
    m.links = [(0, 1), (1, 2), (2, 3)]
    m.forced = {frozenset((2, 3)): 1, frozenset((0, 1)): 3}
    m.selected = [3, 1]
    m.root = 3
    m.remove_axis(1)
    assert [a.name for a in m.axes] == ["eje 1", "eje 3", "eje 4"]
    assert m.links == [(1, 2)] and m.forced == {frozenset((1, 2)): 1}
    assert m.selected == [2] and m.root == 2


def test_removing_the_root_makes_the_first_axis_the_new_reference():
    m = line(3)
    m.axes[0] = m.axes[0].__class__("eje 1", 0.0, 0.0, False)  # va al revés de la entrada
    m.axes[2] = m.axes[2].__class__("eje 3", 18.0, 0.0, True)
    m.set_root(1)  # la entrada ahora es el eje 2 (que gira igual que el 3 y al revés del 1)
    m.remove_axis(1)
    assert m.root == 0 and m.axes[0].same_direction is True
    assert m.axes[1].same_direction is False  # el eje 3 gira al revés del eje 1


def test_the_input_axis_is_the_reference_and_cannot_turn_against_itself():
    m = line(3)
    m.set_same_direction(1, False)
    assert m.axes[1].same_direction is False
    with pytest.raises(ValueError, match="referencia"):
        m.set_same_direction(0, False)


def test_changing_the_input_keeps_how_each_axis_turns():
    m = line(3)
    m.set_same_direction(1, False)  # el 1 gira al revés que el 0; el 2 igual que el 0
    out_before = m.plan()
    m.set_root(1)
    assert m.root == 1 and m.axes[1].same_direction is True
    # respecto de la nueva entrada: el 0 gira al revés y el 2 también
    assert [a.same_direction for a in m.axes] == [False, True, False]
    out_after = m.plan()
    assert out_before.ok and out_after.ok

    # el sentido relativo entre cada par de ejes no cambió
    def rel(out):
        return {frozenset((c.a, c.b)): c.same_direction for c in out.train.connections}

    for pair, same in rel(out_before).items():
        if pair in rel(out_after):
            assert rel(out_after)[pair] == same


# --- selección y pares -------------------------------------------------------------------------


def test_selection_toggles_and_keeps_the_last_two():
    m = line(4)
    m.toggle_select(0)
    m.toggle_select(1)
    assert m.selected == [0, 1] and m.can_connect_selected()
    m.toggle_select(2)
    assert m.selected == [1, 2]
    m.toggle_select(1)
    assert m.selected == [2] and not m.can_connect_selected()
    m.clear_selection()
    assert m.selected == []


def test_connect_selected_adds_the_pair_and_clears_the_selection():
    m = line(3)
    with pytest.raises(ValueError, match="dos ejes"):
        m.connect_selected()
    m.toggle_select(2)
    m.toggle_select(0)
    assert m.connect_selected() == (2, 0)
    assert m.links == [(2, 0)] and m.selected == []


def test_invalid_pairs_are_rejected_and_do_not_change_the_model():
    m = line(3)
    m.connect(0, 1)
    m.connect(1, 2)
    for bad, text in (((1, 0), "ya están conectados"), ((0, 2), "lazo"), ((1, 1), "consigo")):
        with pytest.raises(ValueError, match=text):
            m.connect(*bad)
    assert m.links == [(0, 1), (1, 2)]
    twins = GearNetModel()
    twins.add_axis(0, 0)
    twins.add_axis(5, 5)
    twins.add_axis(5, 5)
    with pytest.raises(ValueError, match="coinciden"):
        twins.connect(1, 2)


def test_unlink_and_clear_also_drop_the_forced_intermediates():
    m = line(3)
    m.connect(0, 1)
    m.connect(1, 2)
    m.set_intermediates(0, 1, 3)
    m.unlink(1, 0)
    assert m.links == [(1, 2)] and m.forced == {}
    with pytest.raises(ValueError, match="no estaba"):
        m.unlink(0, 1)
    m.set_intermediates(1, 2, 1)
    m.clear_links()
    assert m.links == [] and m.forced == {}


def test_forced_intermediates_validate_the_pair_and_the_count():
    m = line(3)
    m.connect(0, 1)
    m.set_intermediates(0, 1, 3)
    assert m.forced == {frozenset((0, 1)): 3}
    m.set_intermediates(1, 0, None)
    assert m.forced == {}
    with pytest.raises(ValueError, match="negativa"):
        m.set_intermediates(0, 1, -1)
    m.auto = False
    with pytest.raises(ValueError, match="no está conectado"):
        m.set_intermediates(1, 2, 1)  # sin auto, el 1-2 no existe todavía


# --- armar la red ------------------------------------------------------------------------------


def test_an_empty_or_single_axis_model_plans_without_raising():
    out = GearNetModel().plan()
    assert out.train is None and "Agregá ejes" in out.error and not out.ok
    one = GearNetModel()
    one.add_axis(0, 0)
    out = one.plan()
    assert out.ok and out.train.connections == () and "no hace falta" in one.describe(out)[0]


def test_auto_completes_the_missing_pairs_and_keeps_the_chosen_ones():
    m = line(4)
    m.connect(0, 3)
    out = m.plan()
    assert out.ok and out.links[0] == (0, 3) and len(out.links) == 3
    assert {frozenset(p) for p in out.links} >= {frozenset((0, 3))}
    assert len(m.auto_links()) == 2
    assert {frozenset((c.a, c.b)) for c in out.train.connections} == {
        frozenset(p) for p in out.links
    }
    assert train_problems(out.train) == []


def test_without_auto_the_chosen_pairs_must_join_everything():
    m = line(3)
    m.auto = False
    m.connect(0, 1)
    out = m.plan()
    assert not out.ok and out.train is None and "faltan" in out.error
    assert m.describe(out)[1] is False
    m.connect(1, 2)
    assert m.plan().ok


def test_forced_intermediates_reach_the_plan_and_a_wrong_parity_is_an_error_message():
    m = line(3)
    m.connect(0, 1)
    m.set_intermediates(0, 1, 3)  # mismo sentido pide impar: 3 sirve
    out = m.plan()
    assert out.ok
    assert next(c for c in out.train.connections if {c.a, c.b} == {0, 1}).intermediates == 3
    m.set_intermediates(0, 1, 2)  # par con mismo sentido: no cierra la paridad
    out = m.plan()
    assert not out.ok and out.train is None and "intermedios" in out.error


def test_a_failing_plan_never_raises_and_the_message_says_why():
    m = GearNetModel()
    m.add_axis(0, 0)
    m.add_axis(0.4, 0)
    assert m.plan().ok  # el módulo sale de la distancia y encaja
    out = m.plan(module=1.0)  # con un módulo grande los extremos no llegan a los dientes mínimos
    assert not out.ok and out.error is None and out.train.problems  # problemas del tren
    text, ok = m.describe(out)
    assert not ok and "demasiado grande" in text


def test_the_example_plans_and_has_a_reversed_axis():
    m = GearNetModel.example()
    out = m.plan()
    assert out.ok and len(out.train.connections) == 3
    assert [a.same_direction for a in m.axes] == [True, True, False, True]
    text, ok = m.describe(out)
    assert ok and "engranajes" in text


def test_axes_that_coincide_share_one_axis_in_the_plan():
    m = line(2)
    m.add_axis(9.0, 0.5)  # coincide con el eje 2 (índice 1)
    out = m.plan()
    assert out.ok and out.train.shared == ((1, 2),)
    assert m.representatives() == [0, 1, 1]
