"""Puente entre el rig y la red de engranajes (M5, bundle 3)."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from mechchar.core.assembly import Assembly, Drive, FixedState, Pin
from mechchar.core.character import biped_character
from mechchar.core.gearlink import plan_rig_train, rig_axes
from mechchar.core.gearnet import simulate_train, train_problems
from mechchar.core.rig import MultiRig
from tests.test_rig import leg_mechanism, make_specs  # noqa: F401  (fixture y ayuda)


@dataclass
class _Sub:
    name: str
    drive: Drive


def fake_rig(positions, ratios=None, grounded=True):
    """Un «rig» mínimo: una manivela por posición, clavada a una bancada fija (o suelta)."""
    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    state = [0.0, 0.0, 0.0]
    subs = []
    for i, (x, y) in enumerate(positions):
        crank = asm.add_component(f"manivela{i}", x, y, 0.0)
        if grounded:
            asm.add(Pin(ground, (x, y), crank, (0.0, 0.0)))
        drive = Drive(crank, ratio=(ratios or [1.0] * len(positions))[i])
        asm.add(drive)
        subs.append(_Sub(f"m{i}", drive))
        state += [x, y, 0.0]
    return SimpleNamespace(asm=asm, subs=subs, state0=np.array(state))


# --- ejes sacados de un rig real ---------------------------------------------------------------


def test_biped_legs_share_one_axis(leg_mechanism):  # noqa: F811
    rig = MultiRig(biped_character(), make_specs(leg_mechanism))
    ra = rig_axes(rig)
    assert [a.name for a in ra.axes] == ["l_leg", "r_leg"] and ra.members == ((0, 1),)
    # el eje es el origen de la manivela en el estado del rig, el mismo para las dos piernas
    d = rig.subs[0].drive
    for a in ra.axes:
        assert (a.x, a.y) == pytest.approx((rig.state0[3 * d.comp], rig.state0[3 * d.comp + 1]))
    assert ra.axis_of(0) == ra.axis_of(1) == 0


def test_coaxial_legs_need_no_gears(leg_mechanism):  # noqa: F811
    rig = MultiRig(biped_character(), make_specs(leg_mechanism))
    train, ra = plan_rig_train(rig)
    assert train.ok and train.connections == () and train_problems(train) == []
    assert train.shared == ((0, 1),) and len(ra.members) == 1


def test_coaxial_axes_that_turn_opposite_are_a_problem(leg_mechanism):  # noqa: F811
    rig = MultiRig(biped_character(), make_specs(leg_mechanism))
    rig.subs[1].drive.ratio = -1.0
    train, _ = plan_rig_train(rig)
    assert not train.ok and "sentidos opuestos" in train.problems[0]


# --- rig sencillo con ejes separados -----------------------------------------------------------


def test_axes_follow_the_state_given():
    rig = fake_rig([(0.0, 0.0), (10.0, 0.0)])
    moved = rig.state0.copy()
    moved[3 * 2], moved[3 * 2 + 1] = 12.0, 1.0
    ra = rig_axes(rig, moved)
    assert [(a.x, a.y) for a in ra.axes] == [(0.0, 0.0), (12.0, 1.0)]


def test_the_root_is_the_mechanism_asked_for_and_axes_keep_the_order_of_the_rig():
    rig = fake_rig([(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)])
    ra = rig_axes(rig, root=2)
    assert [a.name for a in ra.axes] == ["m0", "m1", "m2"] and ra.root == 2
    assert ra.members[0] == (2,)  # el grupo de la entrada va primero
    assert sorted(k for g in ra.members for k in g) == [0, 1, 2]
    train, _ = plan_rig_train(rig, root=2)
    assert train.ok and train.root == 2


def test_axes_that_coincide_share_one_even_when_the_root_is_the_second():
    rig = fake_rig([(0.0, 0.0), (9.0, 1.0), (9.0, 1.0)])
    ra = rig_axes(rig, root=2)
    assert ra.members == ((1, 2), (0,)) and ra.axis_of(2) == 1
    train, _ = plan_rig_train(rig, root=2)
    assert train.ok and train.shared == ((1, 2),) and len(train.connections) == 1
    assert train_problems(train) == []


def test_turning_the_other_way_sets_same_direction_false():
    rig = fake_rig([(0.0, 0.0), (10.0, 0.0)], ratios=[1.0, -1.0])
    ra = rig_axes(rig)
    assert [a.same_direction for a in ra.axes] == [True, False]


def test_a_different_speed_is_an_error():
    with pytest.raises(ValueError, match="misma velocidad"):
        rig_axes(fake_rig([(0.0, 0.0), (10.0, 0.0)], ratios=[1.0, 2.0]))


def test_a_crank_not_pinned_to_the_frame_is_an_error():
    with pytest.raises(ValueError, match="no está fijo a la bancada"):
        rig_axes(fake_rig([(0.0, 0.0), (10.0, 0.0)], grounded=False))


def test_unknown_root_is_an_error():
    with pytest.raises(ValueError, match="no existe"):
        rig_axes(fake_rig([(0.0, 0.0)]), root=3)


@pytest.mark.parametrize("ratios", [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
def test_the_train_turns_each_axis_the_way_its_mechanism_does(ratios):
    rig = fake_rig([(0.0, 0.0), (9.0, 1.0), (4.0, 8.0)], ratios=ratios)
    train, ra = plan_rig_train(rig)
    assert train.ok and train_problems(train) == [], train.problems
    angles = simulate_train(train, 12)
    step = 5
    for k, sub in enumerate(rig.subs):
        i = ra.axis_of(k)
        want = math.copysign(1.0, sub.drive.ratio) * math.copysign(1.0, ratios[0])
        assert angles[step, i] / angles[step, 0] == pytest.approx(want, abs=1e-6)


def test_kwargs_reach_plan_train():
    rig = fake_rig([(0.0, 0.0), (30.0, 0.0)])
    train, _ = plan_rig_train(rig, module=1.0)
    assert train.module == 1.0
    with pytest.raises(ValueError, match="paridad|sentido pedido"):
        plan_rig_train(rig, intermediates={(0, 1): 2})
