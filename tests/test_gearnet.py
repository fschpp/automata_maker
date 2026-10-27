"""Red de engranajes que une los ejes de entrada de los mecanismos (M5, bundle 2)."""

import math

import numpy as np
import pytest

from mechchar.core.gearnet import (
    END_FILL,
    Axis,
    module_for,
    plan_train,
    simulate_train,
    spanning_tree,
    train_problems,
)
from mechchar.core.geartrain import MIN_TEETH, chain_problems, direction


def _speeds(train, step: int = 5) -> np.ndarray:
    """Velocidad de cada eje relativa a la del eje de entrada, medida simulando la red."""
    angles = simulate_train(train, 12)
    return angles[step] / angles[step, train.root]


def _wanted(axes) -> np.ndarray:
    return np.array([1.0 if a.same_direction else -1.0 for a in axes])


def _assert_good(train) -> None:
    assert train.ok, train.problems
    assert train_problems(train) == []
    assert _speeds(train) == pytest.approx(_wanted(train.axes), abs=1e-6)


# --- árbol de expansión ------------------------------------------------------------------------


def test_spanning_tree_connects_every_axis_once_nearest_first() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 10, 0), Axis("c", 4, 0), Axis("d", 20, 0)]
    # desde a: c (4), luego b (6 desde c), luego d (10 desde b)
    assert spanning_tree(axes, 0) == [(0, 2), (2, 1), (1, 3)]


def test_spanning_tree_depends_on_the_root_and_ties_break_by_index() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 10, 0), Axis("c", 20, 0)]
    assert spanning_tree(axes, 0) == [(0, 1), (1, 2)]
    assert spanning_tree(axes, 2) == [(2, 1), (1, 0)]
    square = [Axis("a", 0, 0), Axis("b", 5, 0), Axis("c", 0, 5)]
    assert spanning_tree(square, 0) == [(0, 1), (0, 2)]  # empate: el índice menor primero


def test_spanning_tree_rejects_a_bad_root() -> None:
    with pytest.raises(ValueError, match="no existe"):
        spanning_tree([Axis("a", 0, 0), Axis("b", 5, 0)], 5)


# --- módulo ------------------------------------------------------------------------------------


def test_default_module_makes_the_end_gears_fill_80_percent_of_the_shortest_distance() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 10, 0), Axis("c", 10, 6)]
    edges = spanning_tree(axes)
    assert module_for(axes, edges) == pytest.approx(END_FILL * 6.0 / 16)


def test_all_gears_share_one_module_and_every_end_gear_has_the_same_teeth() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False)])
    gears = train.gears()
    assert {g.module for g in gears} == {train.module}
    for c in train.connections:
        assert c.gears[0].teeth == c.gears[-1].teeth == train.end_teeth


def test_a_given_module_is_used_and_too_big_a_module_is_reported() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 10, 0)]
    assert plan_train(axes, module=0.4).module == 0.4
    big = plan_train(axes, module=2.0)
    assert not big.ok and big.connections == ()
    assert "demasiado grande" in big.problems[0]
    with pytest.raises(ValueError, match="positivo"):
        plan_train(axes, module=-1.0)


# --- una conexión ------------------------------------------------------------------------------


@pytest.mark.parametrize("same", [True, False])
def test_two_axes_get_the_parity_that_the_direction_asks_for(same: bool) -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 10, 0, same)])
    assert len(train.connections) == 1
    c = train.connections[0]
    assert c.same_direction == same
    assert (direction(c.intermediates) == 1) == same
    assert chain_problems(c.gears) == []
    _assert_good(train)


def test_ends_do_not_overlap_and_every_neighbour_meshes() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 10, 0)])
    (c,) = train.connections
    assert math.dist((c.gears[0].x, c.gears[0].y), (c.gears[-1].x, c.gears[-1].y)) == 10.0
    for g, h in zip(c.gears, c.gears[1:], strict=False):
        assert math.dist((g.x, g.y), (h.x, h.y)) == pytest.approx(g.radius + h.radius, rel=1e-9)


# --- una red -----------------------------------------------------------------------------------


def test_mixed_directions_give_the_right_parity_on_every_branch() -> None:
    axes = [Axis("entrada", 0, 0), Axis("m1", 8, 1), Axis("m2", 3, 7, False),
            Axis("m3", 11, 8)]
    train = plan_train(axes)
    assert len(train.connections) == len(axes) - 1  # un árbol
    for c in train.connections:
        want_same = axes[c.a].same_direction == axes[c.b].same_direction
        assert (direction(c.intermediates) == 1) == want_same
    _assert_good(train)


def test_every_axis_ends_up_at_the_speed_of_the_input_axis() -> None:
    axes = [Axis(f"e{i}", x, y, i % 3 != 0 or i == 0)
            for i, (x, y) in enumerate([(0, 0), (9, 2), (4, 8), (12, 9), (1, 14)])]
    train = plan_train(axes)
    _assert_good(train)
    assert np.abs(_speeds(train)) == pytest.approx(np.ones(len(axes)))


def test_the_input_axis_turns_once_per_cycle_in_the_positive_direction() -> None:
    """Lo absoluto, no solo lo relativo: con todos los ejes a la misma velocidad, las velocidades
    relativas a la entrada no cambian si el motor está en otro eje, así que hay que medir el
    ángulo del propio eje de entrada. El último eje pide girar al revés a propósito."""
    axes = [Axis("entrada", 0, 0), Axis("m1", 8, 1), Axis("m2", 3, 7), Axis("m3", 11, 8, False)]
    train = plan_train(axes)
    angles = simulate_train(train, 24)
    assert angles[:, 0] == pytest.approx(2 * math.pi * np.arange(24) / 24)
    assert angles[:, 3] == pytest.approx(-angles[:, 0])  # el que gira al revés


def test_the_root_can_be_any_axis() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False)]
    for root in range(3):
        wants = [Axis(a.name, a.x, a.y, a.same_direction == axes[root].same_direction)
                 for a in axes]
        train = plan_train(wants, root=root)
        assert train.root == root
        _assert_good(train)


def test_a_single_axis_is_a_trivial_train_with_nothing_to_simulate() -> None:
    train = plan_train([Axis("solo", 1, 1)])
    assert train.ok and train.connections == () and train.gears() == []
    with pytest.raises(ValueError, match="nada que simular"):
        simulate_train(train)


# --- capas -------------------------------------------------------------------------------------


def test_connections_that_reach_the_same_axis_are_stacked_on_it() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 5, 0), Axis("c", 10, 0)]
    train = plan_train(axes)
    _assert_good(train)
    layers = {c.layer for c in train.connections}
    assert len(layers) == 2  # b lleva un engranaje por conexión, en capas distintas
    at_b = [g for g in train.gears() if (g.x, g.y) == (5.0, 0.0)]
    assert len(at_b) == 2 and len({g.z for g in at_b}) == 2


def test_the_layer_pitch_sets_the_height_of_each_layer() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 5, 0), Axis("c", 10, 0)], layer_pitch=2.5)
    for c in train.connections:
        assert {g.z for g in c.gears} == {c.layer * 2.5}


def test_gears_of_a_connection_share_one_layer() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False)])
    for c in train.connections:
        assert len({g.z for g in c.gears}) == 1


# --- forzar los intermedios --------------------------------------------------------------------


def test_the_number_of_intermediates_can_be_forced_in_either_axis_order() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 10, 0)]
    for key in ((0, 1), (1, 0)):
        train = plan_train(axes, intermediates={key: 3})
        assert train.ok and train.connections[0].intermediates == 3
        _assert_good(train)


def test_a_forced_count_with_the_wrong_parity_is_an_error() -> None:
    with pytest.raises(ValueError, match="no dan el sentido"):
        plan_train([Axis("a", 0, 0), Axis("b", 10, 0)], intermediates={(0, 1): 2})
    with pytest.raises(ValueError, match="no dan el sentido"):
        plan_train([Axis("a", 0, 0), Axis("b", 10, 0, False)], intermediates={(0, 1): 1})


def test_a_connection_that_cannot_be_made_is_reported_and_left_out() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 10, 0)], intermediates={(0, 1): 21})
    assert not train.ok and train.connections == ()
    assert "no se pudo unir" in train.problems[0] and "21" in train.problems[0]
    with pytest.raises(ValueError, match="problemas"):
        simulate_train(train)


# --- validación de entrada ---------------------------------------------------------------------


def test_bad_axes_are_rejected() -> None:
    with pytest.raises(ValueError, match="al menos un eje"):
        plan_train([])
    with pytest.raises(ValueError, match="coinciden"):
        plan_train([Axis("a", 0, 0), Axis("b", 0, 0)])
    with pytest.raises(ValueError, match="no finita"):
        plan_train([Axis("a", float("nan"), 0), Axis("b", 1, 0)])


def test_the_plan_is_deterministic() -> None:
    axes = [Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False), Axis("d", 11, 8)]
    assert plan_train(axes) == plan_train(axes)


# --- robustez ----------------------------------------------------------------------------------


def _random_axes(rng: np.random.Generator, n: int, spacing: float) -> list[Axis]:
    pts: list[np.ndarray] = []
    while len(pts) < n:
        p = rng.uniform(0, 10, 2)
        if all(float(np.hypot(*(p - q))) > spacing for q in pts):
            pts.append(p)
    return [Axis(f"e{i}", float(p[0]), float(p[1]), bool(rng.integers(0, 2)) or i == 0)
            for i, p in enumerate(pts)]


@pytest.mark.parametrize("seed", range(6))
def test_random_layouts_always_resolve_to_a_valid_train(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    for _ in range(8):
        axes = _random_axes(rng, int(rng.integers(2, 8)), 1.5)
        train = plan_train(axes)
        _assert_good(train)
        assert max(c.layer for c in train.connections) < 6  # no se dispara el uso de capas


def test_planning_is_fast_even_in_a_crowded_layout() -> None:
    """Antes de descartar de entrada las capas ya ocupadas en los extremos y las cantidades de
    intermedios imposibles, una red de 5 ejes tardaba hasta ~6 s."""
    import time

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(15):
        axes = _random_axes(rng, 5, 2.0)
        t0 = time.perf_counter()
        plan_train(axes)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 1.0, f"{worst:.2f} s"


def test_the_minimum_teeth_are_respected_by_every_gear() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False)])
    assert min(g.teeth for g in train.gears()) >= MIN_TEETH
