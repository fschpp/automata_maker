"""Colisiones y asignación de capas (M6, bundle 1)."""

import math

import numpy as np
import pytest

pytest.importorskip("ortools")

from mechchar.core.layers import (
    Body,
    LayerPlan,
    Pin,
    assign_layers,
    collisions,
    layer_offsets,
    pin_conflicts,
)

STEPS = 12


def square(cx, cy, h=0.5):
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


def still(name, cx, cy, h=0.5, **kw):
    return Body(name, tuple(square(cx, cy, h) for _ in range(STEPS)), **kw)


def moving(name, path, h=0.5, **kw):
    return Body(name, tuple(square(x, y, h) for x, y in path), **kw)


def pair_names(bodies, pairs):
    return {tuple(sorted(bodies[i].name for i in p)) for p in pairs}


# --- colisiones --------------------------------------------------------------------------------


def test_overlapping_bodies_collide_and_distant_ones_do_not():
    bs = [still("a", 0, 0), still("b", 0.5, 0), still("c", 5, 0)]
    assert pair_names(bs, collisions(bs)) == {("a", "b")}


def test_touching_edges_are_not_a_collision_but_clearance_makes_it_one():
    bs = [still("a", 0, 0), still("b", 1.0, 0)]  # comparten el borde x = 0,5
    assert collisions(bs) == set()
    assert len(collisions(bs, clearance=0.1)) == 1
    far = [still("a", 0, 0), still("b", 1.5, 0)]  # a 0,5 de distancia
    assert collisions(far, clearance=0.2) == set()
    assert len(collisions(far, clearance=0.6)) == 1


def test_collisions_only_count_when_both_are_there_at_the_same_step():
    # a pasa por x = 0 en el paso 0 y b llega a x = 0 recién en el paso 6: nunca coinciden
    a = moving("a", [(0, 0)] * 6 + [(10, 0)] * 6)
    b = moving("b", [(10, 0)] * 6 + [(0, 0)] * 6)
    assert collisions([a, b]) == set()
    c = moving("c", [(0, 0)] * 12)
    assert len(collisions([a, c])) == 1  # con c sí coincide en los pasos 0 a 5


def test_a_sweeping_body_collides_with_one_it_passes_through():
    sweep = moving("sweep", [(-3 + 6 * k / (STEPS - 1), 0) for k in range(STEPS)])
    bs = [sweep, still("mid", 0, 0), still("off", 0, 4)]
    assert pair_names(bs, collisions(bs)) == {("mid", "sweep")}


def test_collision_input_validation():
    with pytest.raises(ValueError, match="mismo nombre"):
        collisions([still("a", 0, 0), still("a", 3, 0)])
    with pytest.raises(ValueError, match="distinta cantidad"):
        collisions([still("a", 0, 0), Body("b", (square(0, 0),))])
    with pytest.raises(ValueError, match="holgura"):
        collisions([still("a", 0, 0)], clearance=-1.0)


# --- capas -------------------------------------------------------------------------------------


def check_plan(bodies, plan, pins=(), clearance=0.0):
    """El plan respeta todas las reglas, medido de nuevo con colisiones y pasadores."""
    assert plan.ok, plan.problems
    for p in collisions(bodies, clearance):
        i, j = sorted(p)
        assert plan.layers[bodies[i].name] != plan.layers[bodies[j].name]
    for k, i in pin_conflicts(bodies, pins, clearance):
        ls = [plan.layers[n] for n in pins[k].members]
        assert not (min(ls) <= plan.layers[bodies[i].name] <= max(ls))


def test_no_collisions_needs_one_layer():
    bs = [still("a", 0, 0), still("b", 5, 0), still("c", 10, 0)]
    plan = assign_layers(bs)
    assert plan.n_layers == 1 and set(plan.layers.values()) == {0}


def test_a_chain_of_collisions_needs_two_layers_and_a_triangle_three():
    chain = [still("a", 0, 0), still("b", 0.8, 0), still("c", 1.6, 0), still("d", 2.4, 0)]
    plan = assign_layers(chain)
    assert plan.n_layers == 2
    check_plan(chain, plan)
    triple = [still("a", 0, 0), still("b", 0.3, 0), still("c", 0.1, 0.2)]
    plan = assign_layers(triple)
    assert plan.n_layers == 3
    check_plan(triple, plan)


def test_the_minimum_number_of_layers_is_found():
    # un grupo de 4 que se pisan entre sí (clique de 4) y otros que no molestan
    bs = [still(f"k{i}", 0.05 * i, 0.05 * i) for i in range(4)] + [still("x", 9, 9)]
    assert assign_layers(bs).n_layers == 4


def test_not_enough_layers_reports_a_problem_and_invents_nothing():
    bs = [still(f"k{i}", 0.05 * i, 0) for i in range(4)]
    plan = assign_layers(bs, max_layers=3)
    assert not plan.ok and plan.layers == {} and "3 capas" in plan.problems[0]


def test_preassigned_layers_are_respected_and_not_compacted():
    bs = [still("a", 0, 0, layer=2), still("b", 0.3, 0), still("c", 7, 0)]
    plan = assign_layers(bs)
    assert plan.layers["a"] == 2 and plan.layers["b"] != 2 and plan.n_layers >= 3
    check_plan(bs, plan)
    with pytest.raises(ValueError):
        Body("n", (square(0, 0),), layer=-1)


def test_preassigned_layers_that_collide_have_no_solution():
    bs = [still("a", 0, 0, layer=1), still("b", 0.3, 0, layer=1)]
    assert not assign_layers(bs).ok


# --- pasadores ---------------------------------------------------------------------------------


def pin_through(members, x, y, radius=0.1):
    return Pin(tuple(members), np.array([[x, y]] * STEPS), radius)


def test_a_body_on_the_pin_path_cannot_sit_between_the_pinned_bodies():
    # a y b no se pisan entre sí pero van unidos por un pasador en (0, 0); c pisa el pasador
    a, b = still("a", -3, 0, 0.5), still("b", 3, 0, 0.5)
    c = still("c", 0, 0, 0.3)
    pin = Pin(("a", "b"), np.array([[0.0, 0.0]] * STEPS), 0.1)
    bs = [a, b, c]
    assert pin_conflicts(bs, [pin]) == [(0, 2)]
    plan = assign_layers(bs, [pin])
    check_plan(bs, plan, [pin])
    la, lb, lc = (plan.layers[n] for n in "abc")
    assert la != lb and not (min(la, lb) < lc < max(la, lb))
    assert plan.n_layers == 3 and plan.pin_length == 1  # c afuera, a y b contiguas


def test_members_on_the_same_pin_may_share_the_span():
    # tres piezas en el mismo pasador: ninguna cuenta como «en medio» de las otras
    bs = [still("a", 0, 0), still("b", 0.1, 0), still("c", 0.2, 0)]
    pin = pin_through("abc", 0, 0)
    assert pin_conflicts(bs, [pin]) == []
    plan = assign_layers(bs, [pin])
    assert plan.n_layers == 3 and plan.pin_length == 2


def test_pin_length_is_minimised_with_the_minimum_layers():
    # a y b unidas; x y y no tocan nada de eso, pero se pisan con a y con b: caben afuera o
    # entre ellas sin molestar al pasador. El plan tiene que dejar a y b contiguas.
    a, b = still("a", -3, 0), still("b", 3, 0)
    x, y = still("x", -3, 0.2), still("y", 3, 0.2)
    pin = pin_through("ab", 0, 0)
    plan = assign_layers([a, b, x, y], [pin])
    assert plan.ok and plan.pin_length == 1 and plan.optimal


def test_pin_input_validation():
    with pytest.raises(ValueError, match="al menos dos"):
        Pin(("a", "a"), np.zeros((STEPS, 2)))
    with pytest.raises(ValueError, match="radio"):
        Pin(("a", "b"), np.zeros((STEPS, 2)), radius=0.0)
    bs = [still("a", 0, 0), still("b", 3, 0)]
    with pytest.raises(ValueError, match="no existe"):
        assign_layers(bs, [pin_through("az", 0, 0)])
    with pytest.raises(ValueError, match="pasos"):
        assign_layers(bs, [Pin(("a", "b"), np.zeros((3, 2)))])


# --- espesores ---------------------------------------------------------------------------------


def test_layer_thickness_is_the_thickest_body_and_z_accumulates():
    bs = [
        still("c", 9, 0, thickness=5.0),
        still("a", 0, 0, thickness=2.0),
        still("b", 0.3, 0, thickness=3.0),
        still("d", 0.2, 0.2, thickness=1.5),
    ]
    plan = LayerPlan({"c": 0, "a": 0, "b": 1, "d": 2}, 3, 0, True)
    t, z = layer_offsets(plan, bs)
    assert t == [5.0, 3.0, 1.5] and z == [0.0, 5.0, 8.0]
    with pytest.raises(ValueError):
        layer_offsets(LayerPlan({}, 0, 0, False, ("x",)), bs)


# --- escala ------------------------------------------------------------------------------------


def test_fifty_bodies_with_a_gear_like_motion_solve_fast():
    import time

    rng = np.random.default_rng(1)
    bodies = []
    for i in range(50):
        cx, cy = rng.uniform(0, 12, 2)
        r = rng.uniform(0.3, 0.8)
        ang = rng.uniform(0, 2 * math.pi)
        path = [
            (
                cx + 0.3 * math.cos(ang + 2 * math.pi * k / STEPS),
                cy + 0.3 * math.sin(ang + 2 * math.pi * k / STEPS),
            )
            for k in range(STEPS)
        ]
        bodies.append(moving(f"p{i}", path, r))
    pins = [
        pin_through((f"p{i}", f"p{i + 1}"), *bodies[i].shapes[0].mean(axis=0))
        for i in range(0, 48, 6)
    ]
    t0 = time.perf_counter()
    plan = assign_layers(bodies, pins, time_limit=5.0)
    elapsed = time.perf_counter() - t0
    check_plan(bodies, plan, pins)
    assert elapsed < 30.0


# --- contra fuerza bruta -----------------------------------------------------------------------


def _brute(bodies, pins, clearance=0.0):
    """Mínimo de capas y, con él, mínimo largo de pasadores, probando todas las asignaciones."""
    import itertools

    names = [b.name for b in bodies]
    pairs = [tuple(sorted(p)) for p in collisions(bodies, clearance)]
    conflicts = pin_conflicts(bodies, pins, clearance)
    n = len(bodies)
    for m in range(1, n + 1):
        best = None
        for combo in itertools.product(range(m), repeat=n):
            if any(combo[i] == combo[j] for i, j in pairs):
                continue
            lay = dict(zip(names, combo, strict=True))
            if any(len({lay[x] for x in pin.members}) < len(pin.members) for pin in pins):
                continue
            if any(
                min(lay[x] for x in pins[k].members)
                <= combo[i]
                <= max(lay[x] for x in pins[k].members)
                for k, i in conflicts
            ):
                continue
            length = sum(
                max(lay[x] for x in p.members) - min(lay[x] for x in p.members) for p in pins
            )
            best = length if best is None else min(best, length)
        if best is not None:
            return m, best
    raise AssertionError("sin solución")


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_on_small_random_instances(seed):
    rng = np.random.default_rng(seed)
    n = 5
    bodies = [still(f"p{i}", *rng.uniform(0, 2.5, 2), rng.uniform(0.3, 0.7)) for i in range(n)]
    pins = []
    for _ in range(int(rng.integers(0, 3))):
        a, b = rng.choice(n, 2, replace=False)
        pins.append(pin_through((f"p{a}", f"p{b}"), *rng.uniform(0, 2.5, 2), radius=0.15))
    plan = assign_layers(bodies, pins, time_limit=10.0)
    m, length = _brute(bodies, pins)
    check_plan(bodies, plan, pins)
    assert (plan.n_layers, plan.pin_length) == (m, length)
    assert plan.optimal
