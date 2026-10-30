"""Del rig y la red de engranajes al problema de capas (M6, bundle 2)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

pytest.importorskip("ortools")

from mechchar.core.assembly import Assembly, CycleResult, Drive, FixedState  # noqa: E402
from mechchar.core.character import Character, Part, Pivot, biped_character  # noqa: E402
from mechchar.core.gearlink import plan_rig_train  # noqa: E402
from mechchar.core.layerbuild import build_layer_problem  # noqa: E402
from mechchar.core.layers import assign_layers, collisions, pin_conflicts  # noqa: E402
from mechchar.core.mechanisms import four_bar, geared_five_bar  # noqa: E402
from mechchar.core.rig import MultiRig, SubAssembly  # noqa: E402
from tests.test_rig import leg_mechanism, make_specs  # noqa: E402,F401


def fake_rig(placements, character=None, five_bar=False):
    """Un rig sin personaje (o con el dado): un cuadrilátero —o un 5-bar con engranajes— por cada
    (dx, dy), con la bancada en ese punto."""
    character = character or Character("vacío")
    asm, s_char = character.build() if character.parts else (Assembly(), np.zeros(0))
    states, subs = [s_char], []
    for k, (dx, dy) in enumerate(placements):
        if five_bar:
            fb = geared_five_bar(
                d=6.0, r1=1.0, r2=1.0, l1=4.0, l2=4.0, ratio=1.0, psi=0.5, pose=(dx, dy, 0.0)
            )
            s = fb.state0.copy()
        else:
            fb = four_bar(d=6.0, r=1.2, l=5.0, rr=4.0)
            fb.asm.constraints[0] = FixedState(fb.ground, dx, dy, 0.0)
            s = fb.state0.copy()
            s[0::3] += dx
            s[1::3] += dy
        n_before = len(asm.constraints)
        first = asm.append(fb.asm)
        drive = next(c for c in asm.constraints[n_before:] if isinstance(c, Drive))
        subs.append(
            SubAssembly(f"m{k}", fb, None, None, first, len(fb.asm.components), drive,
                        drive.offset, 0, (0.0, 0.0))
        )  # fmt: skip
        states.append(s)
    return SimpleNamespace(
        character=character,
        n_char=len(character.parts),
        asm=asm,
        state0=np.concatenate(states),
        subs=subs,
    )


def chassis(thickness_outline=0.5):
    """Dos piezas fijas del chasis unidas por un pivote, que se pisan."""
    h = thickness_outline

    def box(cx):
        return [(cx - h, -h), (cx + h, -h), (cx + h, h), (cx - h, h)]

    return Character(
        "chasis",
        parts=[
            Part("base", 0.0, 0.0, 0.0, box(0.0), fixed=True),
            Part("tapa", 0.0, 0.0, 0.0, box(0.8), fixed=True),
        ],
        pivots=[Pivot("base", "tapa", (0.4, 0.0))],
    )


def inside(body, step, point, tol=1e-6):
    return Polygon(body.shapes[step]).buffer(tol).contains(Point(*point))


# --- eslabones y pasadores de un mecanismo -----------------------------------------------------


def test_a_four_bar_gives_a_body_per_moving_link_and_a_pin_per_joint():
    p = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=12)
    assert [b.name for b in p.bodies] == ["m0:crank", "m0:coupler", "m0:rocker"]
    assert [sorted(pin.members) for pin in p.pins] == [
        ["m0:coupler", "m0:crank"],
        ["m0:coupler", "m0:rocker"],
    ]  # los pasadores contra la bancada entran a la pared y no se modelan
    assert p.meshes == () and p.n_steps == 12
    assert all(len(b.shapes) == 12 for b in p.bodies)


def test_every_pin_stays_inside_both_of_its_bodies_at_every_step():
    p = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=12, link_width=0.2)
    by_name = {b.name: b for b in p.bodies}
    for pin in p.pins:
        assert pin.path.shape == (12, 2)
        for name in pin.members:
            for k in range(12):
                assert inside(by_name[name], k, pin.path[k]), (name, k)


def test_links_move_and_keep_their_size():
    p = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=12)
    coupler = next(b for b in p.bodies if b.name.endswith("coupler"))
    areas = [Polygon(s).area for s in coupler.shapes]
    assert max(areas) - min(areas) < 1e-6 * max(areas)  # rígido
    assert not np.allclose(coupler.shapes[0], coupler.shapes[6])  # y se mueve


def test_link_width_and_thicknesses_are_used():
    thin = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=8, link_width=0.1)
    wide = build_layer_problem(
        fake_rig([(0.0, 0.0)]), n_steps=8, link_width=0.6, link_thickness=2.5
    )
    a = Polygon(thin.bodies[0].shapes[0]).area
    b = Polygon(wide.bodies[0].shapes[0]).area
    assert b > 3.0 * a and wide.bodies[0].thickness == 2.5 and pin_radius_of(wide) == 0.08


def pin_radius_of(problem):
    return problem.pins[0].radius


def test_input_validation():
    rig = fake_rig([(0.0, 0.0)])
    for kw in ({"link_width": 0.0}, {"pin_radius": -1.0}, {"gear_thickness": math.nan}):
        with pytest.raises(ValueError, match="positivo"):
            build_layer_problem(rig, **kw)
    with pytest.raises(ValueError, match="2 pasos"):
        build_layer_problem(rig, n_steps=1)
    train, _ = plan_rig_train(fake_rig([(0.0, 0.0), (14.0, 2.0)]))
    with pytest.raises(ValueError, match="el tren y sus ejes"):
        build_layer_problem(rig, train=train)


def test_a_cycle_that_cannot_be_traced_is_an_error_that_says_why(monkeypatch):
    rig = fake_rig([(0.0, 0.0)])
    bad = CycleResult(np.empty((0, 3)), False, "salto de rama en t=0.2500")
    monkeypatch.setattr(rig.asm, "trace_cycle", lambda s0, n, **kw: bad)
    with pytest.raises(ValueError, match="salto de rama en t=0.2500"):
        build_layer_problem(rig, n_steps=8)


def test_few_steps_are_taken_from_the_fine_cycle_not_traced_coarsely():
    # con 4 pasos la manivela da un cuarto de vuelta por paso: trazado directo parecería un salto
    p = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=4)
    crank = p.bodies[0]
    assert len(crank.shapes) == 4 and not np.allclose(crank.shapes[0], crank.shapes[1])


def test_steps_are_evenly_spaced_in_the_cycle():
    p = build_layer_problem(fake_rig([(0.0, 0.0)]), n_steps=8)
    tip = [s.mean(axis=0) for s in p.bodies[0].shapes]  # el centro de la manivela
    d = [np.linalg.norm(tip[(k + 1) % 8] - tip[k]) for k in range(8)]
    assert (
        max(d) - min(d) < 1e-6
    )  # el paso es parejo (y el ciclo cierra: el último vuelve al primero)


# --- con la red de engranajes ------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_mechanisms():
    rig = fake_rig([(0.0, 0.0), (16.0, 3.0)])
    train, axes = plan_rig_train(rig)
    assert train.ok and len(train.connections) == 1
    return rig, train, axes


def test_the_train_adds_gears_with_teeth_meshes_and_a_shaft_per_axis(two_mechanisms):
    rig, train, axes = two_mechanisms
    p = build_layer_problem(rig, train, axes, n_steps=12)
    gears = [b for b in p.bodies if b.name.startswith("engranaje")]
    c = train.connections[0]
    assert len(gears) == len(c.gears)
    assert gears[0].shapes[0].shape[0] == 4 * c.gears[0].teeth  # con dientes
    assert p.meshes == tuple(
        zip([g.name for g in gears], [g.name for g in gears][1:], strict=False)
    )
    shafts = [pin for pin in p.pins if any(m.startswith("engranaje") for m in pin.members)]
    assert len(shafts) == 2  # el eje de cada mecanismo
    assert {len(s.members) for s in shafts} == {2}  # la manivela y el engranaje de ese eje
    assert {s.members[1 if s.members[0].startswith("m") else 0] for s in shafts} == {
        gears[0].name, gears[-1].name
    }  # fmt: skip
    for s in shafts:
        assert np.allclose(s.path, s.path[0])  # el eje no se mueve


def test_gears_turn_with_the_input_and_return_to_where_they_started(two_mechanisms):
    rig, train, axes = two_mechanisms
    p = build_layer_problem(rig, train, axes, n_steps=12)
    g = next(b for b in p.bodies if b.name == "engranaje 1.1")
    assert not np.allclose(g.shapes[0], g.shapes[1])
    # con z teeth el contorno se repite cada 1/z de vuelta: tras una vuelta entera, igual al inicio
    t = train.connections[0].gears[0].teeth
    assert Polygon(g.shapes[0]).area == pytest.approx(Polygon(g.shapes[5]).area)
    assert t > 3


def test_the_whole_pipeline_assigns_layers_that_respect_everything(two_mechanisms):
    rig, train, axes = two_mechanisms
    p = build_layer_problem(rig, train, axes, n_steps=12)
    plan = assign_layers(p.bodies, p.pins, meshes=p.meshes, time_limit=10.0)
    assert plan.ok and plan.n_layers >= 2
    layers = plan.layers
    for a, b in p.meshes:  # los que engranan, en la misma capa
        assert layers[a] == layers[b]
    ignore = {frozenset((i, j)) for i, bi in enumerate(p.bodies) for j, bj in enumerate(p.bodies)
              if (bi.name, bj.name) in set(p.meshes)}  # fmt: skip
    for pair in collisions(p.bodies, ignore=ignore):
        i, j = sorted(pair)
        assert layers[p.bodies[i].name] != layers[p.bodies[j].name]
    for k, i in pin_conflicts(p.bodies, p.pins):
        ls = [layers[n] for n in p.pins[k].members]
        assert not (min(ls) <= layers[p.bodies[i].name] <= max(ls))
    # la manivela y el engranaje de su eje no comparten capa (se pisan)
    for s in p.pins:
        if any(m.startswith("engranaje") for m in s.members):
            assert len({layers[m] for m in s.members}) == len(s.members)


# --- el bípedo real ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def biped_problem(leg_mechanism):  # noqa: F811
    rig = MultiRig(biped_character(), make_specs(leg_mechanism))
    return rig, build_layer_problem(rig, n_steps=12)


def test_the_biped_has_its_parts_and_the_links_of_both_legs(biped_problem):
    rig, p = biped_problem
    names = [b.name for b in p.bodies]
    assert len(names) == len(set(names))
    assert [b.name for b in p.bodies[: rig.n_char]] == [q.name for q in rig.character.parts]
    for leg in ("l_leg", "r_leg"):
        assert {f"{leg}:crank", f"{leg}:coupler", f"{leg}:rocker"} <= set(names)
    assert len(p.pins) >= 6  # cada pierna: manivela, acoplador-balancín, y la unión al personaje


def test_every_pin_of_the_biped_stays_inside_its_bodies(biped_problem):
    _, p = biped_problem
    by_name = {b.name: b for b in p.bodies}
    for pin in p.pins:
        for name in pin.members:
            for k in range(p.n_steps):
                assert inside(by_name[name], k, pin.path[k], tol=1e-5), (name, k)


def test_the_biped_gets_a_valid_layer_assignment(biped_problem):
    _, p = biped_problem
    plan = assign_layers(p.bodies, p.pins, time_limit=10.0)
    assert plan.ok and plan.n_layers >= 2
    for pair in collisions(p.bodies):
        i, j = sorted(pair)
        assert plan.layers[p.bodies[i].name] != plan.layers[p.bodies[j].name]
    for k, i in pin_conflicts(p.bodies, p.pins):
        ls = [plan.layers[n] for n in p.pins[k].members]
        assert not (min(ls) <= plan.layers[p.bodies[i].name] <= max(ls))


# --- engranajes del 5-bar, chasis, espesores y radios -------------------------------------------


def test_the_two_gears_of_a_five_bar_are_discs_that_mesh_with_each_other():
    p = build_layer_problem(fake_rig([(0.0, 0.0)], five_bar=True), n_steps=12)
    names = [b.name for b in p.bodies]
    assert {"m0:gear1", "m0:gear2", "m0:link1", "m0:link2"} <= set(names)
    assert p.meshes == (("m0:gear1", "m0:gear2"),)
    gear = next(b for b in p.bodies if b.name == "m0:gear1")
    assert Polygon(gear.shapes[0]).area > 0.9 * math.pi * 3.0**2  # el disco (R = d / 2 = 3)
    link = next(b for b in p.bodies if b.name == "m0:link2")
    assert Polygon(link.shapes[0]).area < 0.3 * Polygon(gear.shapes[0]).area
    plan = assign_layers(p.bodies, p.pins, meshes=p.meshes, time_limit=10.0)
    assert plan.ok and plan.layers["m0:gear1"] == plan.layers["m0:gear2"]
    assert plan.layers["m0:link1"] != plan.layers["m0:gear1"]  # la biela se pisa con el disco


def test_pins_between_fixed_chassis_parts_are_not_pins_of_the_layer_problem():
    p = build_layer_problem(fake_rig([(0.0, 0.0)], character=chassis()), n_steps=8)
    assert [b.name for b in p.bodies[:2]] == ["base", "tapa"]
    assert not any({"base", "tapa"} <= set(pin.members) for pin in p.pins)
    plan = assign_layers(p.bodies, p.pins, time_limit=5.0)
    assert plan.ok  # las dos piezas del chasis se pisan: van en capas distintas, sin pasador


def test_thicknesses_and_radii_given_are_the_ones_used(two_mechanisms):
    p = build_layer_problem(
        fake_rig([(0.0, 0.0)], character=chassis()),
        n_steps=8,
        part_thickness=2.5,
        link_thickness=3.5,
    )
    assert [b.thickness for b in p.bodies] == [2.5, 2.5, 3.5, 3.5, 3.5]
    rig, train, axes = two_mechanisms
    q = build_layer_problem(
        rig, train, axes, n_steps=8, gear_thickness=1.7, pin_radius=0.05, axle_radius=0.3
    )
    gears = [b for b in q.bodies if b.name.startswith("engranaje")]
    assert {g.thickness for g in gears} == {1.7}
    shafts = [pin for pin in q.pins if any(m.startswith("engranaje") for m in pin.members)]
    joints = [pin for pin in q.pins if pin not in shafts]
    assert {s.radius for s in shafts} == {0.3} and {j.radius for j in joints} == {0.05}
