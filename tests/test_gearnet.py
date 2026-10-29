"""Red de engranajes que une los ejes de entrada de los mecanismos (M5, bundle 2)."""

import math

import numpy as np
import pytest

from mechchar.core.gearnet import (
    END_FILL,
    Axis,
    input_axis,
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


def test_a_single_axis_is_a_trivial_train_that_just_turns_with_the_input() -> None:
    train = plan_train([Axis("solo", 1, 1)])
    assert train.ok and train.connections == () and train.gears() == []
    angles = simulate_train(train, 8)
    assert angles[:, 0] == pytest.approx(2 * math.pi * np.arange(8) / 8)


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
    with pytest.raises(ValueError, match="no finita"):
        plan_train([Axis("a", float("nan"), 0), Axis("b", 1, 0)])
    with pytest.raises(ValueError, match="no existe"):
        plan_train([Axis("a", 0, 0), Axis("b", 5, 0)], root=7)


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


# --- ejes que coinciden ------------------------------------------------------------------------


def test_axes_in_the_same_place_share_a_shaft_and_need_no_gears() -> None:
    """Las dos piernas del bípedo usan el mismo ajuste: sus ejes de entrada coinciden."""
    train = plan_train([Axis("pierna izq", 0.1045, -0.9723), Axis("pierna der", 0.1045, -0.9723)])
    assert train.ok and train.connections == () and train.shared == ((0, 1),)
    angles = simulate_train(train, 8)
    assert angles[:, 0] == pytest.approx(angles[:, 1])


def test_coincident_axes_are_one_node_of_the_network() -> None:
    axes = [Axis("a", 0, 0), Axis("a2", 0, 0), Axis("b", 10, 0), Axis("c", 10, 0)]
    train = plan_train(axes)
    assert train.ok and train.shared == ((0, 1), (2, 3))
    assert len(train.connections) == 1  # entre los dos puntos; el resto comparte eje
    c = train.connections[0]
    assert (c.a, c.b) == (0, 2)  # indexada por el primer eje de cada grupo
    assert c.same_direction is True


def test_a_network_with_shared_shafts_turns_every_axis_at_the_right_speed() -> None:
    axes = [Axis("a", 0, 0), Axis("a2", 0, 0), Axis("b", 8, 1), Axis("b2", 8, 1),
            Axis("c", 3, 7, False), Axis("c2", 3, 7, False)]
    train = plan_train(axes)
    _assert_good(train)
    speeds = _speeds(train)
    assert speeds[0] == speeds[1] and speeds[2] == speeds[3] and speeds[4] == speeds[5]


def test_coincident_axes_that_ask_for_opposite_directions_are_a_problem() -> None:
    train = plan_train([Axis("a", 0, 0), Axis("a2", 0, 0, False), Axis("b", 9, 0)])
    assert not train.ok
    assert "sentidos opuestos" in train.problems[0] and "'a2'" in train.problems[0]
    with pytest.raises(ValueError, match="problemas"):
        simulate_train(train)


def test_what_counts_as_the_same_place_is_a_tolerance() -> None:
    near = plan_train([Axis("a", 0, 0), Axis("b", 1e-8, 0)])
    assert near.shared == ((0, 1),) and near.connections == ()
    apart = plan_train([Axis("a", 0, 0), Axis("b", 1e-3, 0)], module=1e-5)
    assert apart.shared == ()  # a 1e-3 son ejes distintos (y demasiado juntos para engranar)


def test_the_root_and_forced_counts_can_name_any_member_of_a_shared_shaft() -> None:
    axes = [Axis("a", 0, 0), Axis("a2", 0, 0), Axis("b", 10, 0)]
    train = plan_train(axes, root=1, intermediates={(1, 2): 3})
    assert train.ok and train.root == 1
    assert train.connections[0].intermediates == 3
    _assert_good(train)


# --- el eje de entrada de un mecanismo ---------------------------------------------------------


def _built_mechanisms():
    from mechchar.optim.families import ALL_FAMILIES

    t = np.linspace(0, 2 * math.pi, 60, endpoint=False)
    target = np.column_stack([3 + 1.2 * np.cos(t), 1 + 0.6 * np.sin(t)])
    for fam in ALL_FAMILIES:
        cand = fam.candidates(target, 40, 3000, 2, np.random.default_rng(0))
        assert cand, f"{fam.name} no dio candidatos"
        built = fam.build(cand[0].p0, cand[0].variant)
        assert built is not None
        yield fam.name, built


def test_the_input_axis_does_not_move_while_the_crank_turns() -> None:
    """Es el punto de la bancada alrededor del cual gira la pieza con el Input Driver, en las tres
    familias: se mide viendo que ese punto de la pieza no se mueve a lo largo del ciclo."""
    from mechchar.core.assembly import Drive, Pin, world_point

    seen = 0
    for name, built in _built_mechanisms():
        axis = input_axis(built.asm, built.state0)
        assert axis is not None, name
        drive = next(c for c in built.asm.constraints if isinstance(c, Drive))
        pin = next(c for c in built.asm.constraints if isinstance(c, Pin)
                   and drive.comp in (c.a, c.b))
        local, comp = (pin.pa, pin.a) if pin.a == drive.comp else (pin.pb, pin.b)
        result = built.asm.trace_cycle(built.state0, 24)
        assert result.ok, name
        for state in result.states:
            assert world_point(state, comp, local) == pytest.approx(axis, abs=1e-6), name
        seen += 1
    assert seen == 3


def test_input_axis_is_none_without_a_driver_or_a_fixed_pivot() -> None:
    from mechchar.core.assembly import Assembly, Drive, FixedState, Pin

    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    crank = asm.add_component("manivela", 1.0, 2.0, 0.0)
    asm.add(Pin(ground, (1.0, 2.0), crank, (0.0, 0.0)))
    assert input_axis(asm, asm.initial_state()) is None  # sin Input Driver
    asm.add(Drive(crank))
    assert input_axis(asm, asm.initial_state()) == pytest.approx((1.0, 2.0))
    floating = Assembly()
    floating.add_component("suelta", 0.0, 0.0, 0.0)
    floating.add(Drive(0))
    assert input_axis(floating, floating.initial_state()) is None  # no gira sobre un punto fijo


@pytest.mark.parametrize("ground_first", [True, False])
def test_input_axis_works_whichever_member_of_the_pin_is_the_ground(ground_first: bool) -> None:
    """El eje es el punto del pin en la pieza con el Input Driver, sea el primer o el segundo
    miembro del pin, y con el pivote fuera del origen de la pieza."""
    from mechchar.core.assembly import Assembly, Drive, FixedState, Pin

    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    crank = asm.add_component("manivela", 1.0, 2.0, 0.0)  # su origen no es el pivote
    pivot_on_crank = (0.5, 0.0)  # (1.5, 2.0) en el mundo
    if ground_first:
        asm.add(Pin(ground, (1.5, 2.0), crank, pivot_on_crank))
    else:
        asm.add(Pin(crank, pivot_on_crank, ground, (1.5, 2.0)))
    asm.add(Drive(crank))
    assert input_axis(asm, asm.initial_state()) == pytest.approx((1.5, 2.0))


def test_input_axis_can_pick_one_of_several_drivers_in_a_joint_assembly() -> None:
    from mechchar.core.assembly import Assembly, Drive, FixedState, Pin

    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    drives = []
    for x in (1.0, 5.0):
        crank = asm.add_component(f"manivela{x}", x, 0.0, 0.0)
        asm.add(Pin(ground, (x, 0.0), crank, (0.0, 0.0)))
        drive = Drive(crank)
        asm.add(drive)
        drives.append(drive)
    state = asm.initial_state()
    assert input_axis(asm, state) == pytest.approx((1.0, 0.0))  # por defecto, el primero
    assert input_axis(asm, state, drives[0]) == pytest.approx((1.0, 0.0))
    assert input_axis(asm, state, drives[1]) == pytest.approx((5.0, 0.0))


def test_describe_train_says_what_the_network_is() -> None:
    from mechchar.core.gearnet import describe_train

    one = describe_train(plan_train([Axis("solo", 0, 0)]))
    assert one == ("Hay un solo mecanismo: no hace falta ningún engranaje.", True)
    text, ok = describe_train(plan_train([Axis("a", 1.5, -2), Axis("b", 1.5, -2)]))
    assert ok and "comparten el eje de entrada en (1.50, -2.00)" in text
    assert "sin engranajes" in text
    train = plan_train([Axis("a", 0, 0), Axis("b", 8, 1), Axis("c", 3, 7, False)])
    text, ok = describe_train(train)
    layers = len({c.layer for c in train.connections})
    assert ok and text.startswith(f"{len(train.gears())} engranajes en {layers} capa")
    assert f"módulo {train.module:.3g}" in text and f"{train.end_teeth} dientes" in text
    pair = plan_train([Axis("a", 0, 0), Axis("a2", 0, 0), Axis("b", 8, 1)])
    shared_text, ok = describe_train(pair)
    assert ok and shared_text.endswith("1 eje compartido.")
    text, ok = describe_train(plan_train([Axis("a", 0, 0), Axis("b", 0, 0, False)]))
    assert not ok and "sentidos opuestos" in text
