"""Pares elegidos por el usuario, y ejes cilíndricos de la red (M5, bundle 5)."""

import pytest

from mechchar.core.gearnet import (
    Axis,
    axles,
    complete_links,
    oriented_tree,
    plan_train,
    simulate_train,
    spanning_tree,
    train_problems,
)

LINE = [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 18, 0.5)]


# --- orientar los pares ------------------------------------------------------------------------


def test_links_are_oriented_from_the_input_in_the_order_they_are_reached():
    assert oriented_tree(3, [(2, 1), (1, 0)], 0) == [(0, 1), (1, 2)]
    assert oriented_tree(3, [(0, 1), (0, 2)], 0) == [(0, 1), (0, 2)]
    assert oriented_tree(3, [(0, 1), (1, 2)], 2) == [(2, 1), (1, 0)]


@pytest.mark.parametrize(
    "links, text",
    [
        ([(0, 5)], "no existe"),
        ([(1, 1)], "consigo mismo"),
        ([(0, 1), (1, 0)], "repetido"),
        ([(0, 1)], "faltan 2"),
        ([(0, 1), (1, 2), (0, 2)], "lazo"),
    ],
)
def test_invalid_links_say_why(links, text):
    with pytest.raises(ValueError, match=text):
        oriented_tree(3, links, 0)


def test_a_disconnected_axis_is_reported_before_a_loop():
    with pytest.raises(ValueError, match="no unen todos"):
        oriented_tree(4, [(0, 1), (1, 2), (0, 2)], 0)  # el 3 queda afuera


# --- completar los que faltan ------------------------------------------------------------------


def test_with_no_choice_the_completion_is_the_shortest_tree():
    assert sorted(map(sorted, complete_links(LINE))) == sorted(map(sorted, spanning_tree(LINE)))


def test_chosen_links_are_kept_and_the_rest_are_the_shortest_that_join_everything():
    axes = [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 18, 0), Axis("d", 18, 9)]
    out = complete_links(axes, [(0, 2)])
    assert out[0] == (0, 2) and len(out) == 3
    assert oriented_tree(4, out, 0)  # une todo y sin lazos
    assert {frozenset(p) for p in out} == {frozenset((0, 2)), frozenset((0, 1)), frozenset((2, 3))}


def test_completion_rejects_loops_and_coincident_pairs():
    with pytest.raises(ValueError, match="lazo"):
        complete_links(LINE, [(0, 1), (1, 2), (0, 2)])
    twin = [Axis("a", 0, 0), Axis("b", 5, 5), Axis("c", 5, 5)]
    with pytest.raises(ValueError, match="coinciden"):
        complete_links(twin, [(1, 2)])
    with pytest.raises(ValueError, match="no existe"):
        complete_links(LINE, [(0, 9)])


def test_coincident_axes_are_not_joined_to_each_other_by_the_completion():
    twin = [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 9, 0)]
    out = complete_links(twin)
    assert len(out) == 1 and frozenset(out[0]) != frozenset((1, 2))


# --- plan_train con pares elegidos -------------------------------------------------------------


def pairs(train):
    return {frozenset((c.a, c.b)) for c in train.connections}


def test_the_connections_are_the_chosen_pairs():
    chain = plan_train(LINE, 0, links=[(0, 1), (1, 2)])
    star = plan_train(LINE, 0, links=[(0, 1), (0, 2)])
    assert chain.ok and star.ok
    assert pairs(chain) == {frozenset((0, 1)), frozenset((1, 2))}
    assert pairs(star) == {frozenset((0, 1)), frozenset((0, 2))}
    assert train_problems(star) == [] and train_problems(chain) == []


def test_each_connection_starts_at_the_end_closer_to_the_input_whatever_the_order_given():
    t = plan_train(LINE, 0, links=[(2, 1), (1, 0)])
    assert [(c.a, c.b) for c in t.connections] == [(0, 1), (1, 2)]


def test_no_links_keeps_the_automatic_tree():
    assert pairs(plan_train(LINE, 0)) == pairs(plan_train(LINE, 0, links=spanning_tree(LINE)))


def test_a_chosen_tree_turns_every_axis_the_way_it_was_asked():
    axes = [Axis("a", 0, 0), Axis("b", 9, 0, False), Axis("c", 18, 0.5)]
    t = plan_train(axes, 0, links=[(0, 1), (0, 2)])
    assert t.ok
    sim = simulate_train(t, 12)
    for i, want in enumerate((1.0, -1.0, 1.0)):
        assert sim[5, i] / sim[5, 0] == pytest.approx(want, abs=1e-6)


def test_the_module_follows_the_chosen_pairs():
    # unir a con c directo deja una distancia mayor que la del árbol corto: otro módulo
    far = plan_train(LINE, 0, links=[(0, 2), (2, 1)])
    near = plan_train(LINE, 0)
    assert far.module > near.module


def test_links_between_coincident_axes_or_missing_ones_are_errors():
    twin = [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 9, 0)]
    with pytest.raises(ValueError, match="coinciden"):
        plan_train(twin, 0, links=[(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="faltan"):
        plan_train(LINE, 0, links=[(0, 1)])
    with pytest.raises(ValueError, match="lazo"):
        plan_train(LINE, 0, links=[(0, 1), (1, 2), (0, 2)])


def test_links_to_a_coincident_axis_use_the_group():
    twin = [Axis("a", 0, 0), Axis("b", 9, 0), Axis("c", 9, 0)]
    t = plan_train(twin, 0, links=[(0, 2)])  # el 2 comparte eje con el 1
    assert t.ok and t.shared == ((1, 2),) and [(c.a, c.b) for c in t.connections] == [(0, 1)]


def test_forced_intermediates_still_apply_with_chosen_links():
    t = plan_train(LINE, 0, links=[(0, 1), (0, 2)], intermediates={(0, 1): 3})
    assert next(c for c in t.connections if c.b == 1).intermediates == 3


# --- ejes cilíndricos --------------------------------------------------------------------------


def test_one_axle_per_point_with_gears_and_one_per_intermediate():
    t = plan_train(LINE, 0, links=[(0, 1), (0, 2)])
    ax = axles(t)
    n_inter = sum(c.intermediates for c in t.connections)
    assert len(ax) == 3 + n_inter
    assert [a.name for a in ax[:3]] == ["a", "b", "c"]
    assert all(a.gears == 1 for a in ax[3:]) and all(not a.input for a in ax[3:])


def test_stacked_gears_share_one_axle_that_spans_their_layers():
    t = plan_train(LINE, 0, links=[(0, 1), (0, 2)])
    a = axles(t, thickness=1.0, protrusion=0.0)[0]
    layers = sorted(c.layer for c in t.connections)
    assert a.gears == 2 and layers == [0, 1]
    assert (a.z0, a.z1) == (0.0, 2.0) and a.length == 2.0


def test_only_the_input_axle_protrudes():
    t = plan_train(LINE, 0, links=[(0, 1), (0, 2)])
    flush = {a.name: a.z1 for a in axles(t, 1.0, 0.0)}
    out = {a.name: a.z1 for a in axles(t, 1.0, 3.0)}
    assert out["a"] == flush["a"] + 3.0
    for name in ("b", "c"):
        assert out[name] == flush[name]
    assert [a.name for a in axles(t) if a.input] == ["a"]


def test_the_input_axle_is_the_one_of_the_root_even_when_it_is_not_the_first():
    t = plan_train(LINE, 1)
    assert [a.name for a in axles(t) if a.input] == ["b"]


def test_thickness_scales_the_axle():
    t = plan_train(LINE, 0)
    thin = {a.name: a.length for a in axles(t, 1.0, 0.0)}
    thick = {a.name: a.length for a in axles(t, 2.5, 0.0)}
    assert thick["b"] > thin["b"] and thick["c"] == pytest.approx(2.5)


def test_no_gears_no_axles_and_bad_arguments():
    one = plan_train([Axis("a", 0, 0)], 0)
    assert axles(one) == []
    twin = plan_train([Axis("a", 0, 0), Axis("b", 0, 0)], 0)
    assert axles(twin) == []
    with pytest.raises(ValueError):
        axles(plan_train(LINE, 0), thickness=0.0)
    with pytest.raises(ValueError):
        axles(plan_train(LINE, 0), protrusion=-1.0)
