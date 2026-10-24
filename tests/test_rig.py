import math
from dataclasses import replace

import numpy as np
import pytest

from mechchar.core.assembly import SolveResult
from mechchar.core.character import (
    LEG_ATTACH,
    LEG_MARKER,
    PointRef,
    biped_character,
    leg_character,
    leg_refs,
)
from mechchar.core.playback import Playback
from mechchar.core.rig import MultiRig, SubSpec
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.families import FOUR_BAR

TARGET_T = np.linspace(0, 2 * math.pi, 120, endpoint=False)
N = 100  # pasos por ciclo; un desfase de 0,25 son 25 pasos


@pytest.fixture(scope="module")
def leg_mechanism():
    """Un mecanismo ajustado a una sola pierna (se procesa un sub-ensamblaje a la vez)."""
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    target = np.column_stack([0.9 + 0.55 * np.cos(TARGET_T), -2.4 + 0.25 * np.sin(TARGET_T)])
    cand = fam.candidates(target, 40, 16000, 1, np.random.default_rng(0))[0]
    return cand.p0, cand.variant


def make_specs(leg_mechanism):
    p, variant = leg_mechanism
    specs = []
    for prefix in ("l_", "r_"):
        attach, marker = leg_refs(prefix)
        specs.append(
            SubSpec(
                prefix + "leg", FOUR_BAR.build(p, variant), FOUR_BAR.to_mechanism(p, variant),
                attach, marker,
            )
        )
    return specs


@pytest.fixture
def rig(leg_mechanism):
    return MultiRig(biped_character(), make_specs(leg_mechanism))


# --- personaje de prueba -------------------------------------------------------------------------


def test_biped_has_two_legs_in_different_planes_and_four_dof():
    ch = biped_character()
    a = ch.analyze()
    assert (a.dof, a.redundant) == (4, 0)  # 2 por pierna
    assert {p.z for p in ch.parts if p.name.startswith("l_")} == {0.0}
    assert {p.z for p in ch.parts if p.name.startswith("r_")} == {1.0}
    assert [p.name for p in ch.parts if p.fixed] == ["pelvis"]  # una sola pelvis compartida


def test_leg_refs_follow_the_prefix():
    attach, marker = leg_refs("r_")
    assert attach == PointRef("r_shank", LEG_ATTACH.point)
    assert marker == PointRef("r_foot", LEG_MARKER.point)
    assert leg_refs() == (LEG_ATTACH, LEG_MARKER)


# --- varios mecanismos con un motor ---------------------------------------------------------------


def test_each_leg_gets_its_own_mechanism_and_the_whole_has_no_free_motion(rig):
    a = rig.analyze()
    assert (a.dof, a.redundant) == (0, 0)
    assert len(rig.subs) == 2 and rig.phases == [0.0, 0.0]
    assert np.max(np.abs(rig.asm.residual(rig.state0, 0.0))) < 1e-9
    # el estado se reparte: personaje + un tramo por mecanismo
    assert len(rig.state0) == 3 * (rig.n_char + sum(s.n_comp for s in rig.subs))
    assert len(rig.mech_state(rig.state0, 1)) == 3 * rig.subs[1].n_comp


def test_without_phase_both_feet_move_together(rig):
    cyc = rig.asm.trace_cycle(rig.state0, N, max_jump=1.0)
    assert cyc.ok
    left = np.array([rig.marker_world(s, 0) for s in cyc.states])
    right = np.array([rig.marker_world(s, 1) for s in cyc.states])
    assert np.allclose(left, right, atol=1e-7)
    assert np.ptp(left[:, 0]) > 0.3  # y el pie sí se mueve


@pytest.mark.parametrize("delta", [0.25, 0.5, -0.3])
def test_phase_shift_is_a_time_shift_of_that_mechanism(rig, delta):
    cyc = rig.asm.trace_cycle(rig.state0, N, max_jump=1.0)
    left = np.array([rig.marker_world(s, 0) for s in cyc.states])
    state, ok = rig.set_phase(1, delta, rig.state0, t=0.0)
    assert ok and rig.phases == [0.0, delta]
    # el pie derecho en t = 0 es el izquierdo en t = delta
    shift = round(delta * N) % N
    assert rig.marker_world(state, 1) == pytest.approx(left[shift], abs=1e-6)
    # y el izquierdo no se movió
    assert rig.marker_world(state, 0) == pytest.approx(left[0], abs=1e-7)
    assert np.max(np.abs(rig.asm.residual(state, 0.0))) < 1e-8


def test_shifted_rig_keeps_running_with_the_single_motor(rig):
    state, ok = rig.set_phase(1, 0.5, rig.state0)
    assert ok
    pb = Playback(rig.asm, state, period=2.0)
    left0, right0 = rig.marker_world(state, 0), rig.marker_world(state, 1)
    for _ in range(100):
        pb.advance(0.02)
        assert pb.ok
    # tras un ciclo completo ambos pies vuelven a su posición y el desfase se conserva
    assert pb.t == pytest.approx(1.0, abs=1e-9)
    assert rig.marker_world(pb.state, 0) == pytest.approx(left0, abs=1e-5)
    assert rig.marker_world(pb.state, 1) == pytest.approx(right0, abs=1e-5)


def test_sliding_the_phase_through_a_full_turn_comes_back_to_the_start(rig):
    state = rig.state0
    start = rig.marker_world(state, 1).copy()
    for i in range(1, 51):  # el slider avanza de a 0,02 hasta un ciclo entero
        state, ok = rig.set_phase(1, i / 50, state)
        assert ok, i
    assert rig.phases[1] == pytest.approx(1.0)
    assert rig.marker_world(state, 1) == pytest.approx(start, abs=1e-6)
    # la otra pierna ni se enteró
    assert rig.marker_world(state, 0) == pytest.approx(rig.marker_world(rig.state0, 0), abs=1e-7)


def test_a_big_slider_jump_is_split_into_small_steps(rig, monkeypatch):
    calls = []
    real = rig.asm.solve

    def spy(s0, t, tol=1e-9):
        calls.append(t)
        return real(s0, t, tol)

    monkeypatch.setattr(rig.asm, "solve", spy)
    _, ok = rig.set_phase(0, 0.5, rig.state0)
    assert ok and len(calls) == 25  # 0,5 / 0,02


def test_failed_phase_change_leaves_phase_and_pose_untouched(rig, monkeypatch):
    state, ok = rig.set_phase(1, 0.1, rig.state0)
    assert ok
    before = state.copy()
    monkeypatch.setattr(
        rig.asm, "solve", lambda s0, t, tol=1e-9: SolveResult(np.asarray(s0), 1.0, False)
    )
    new_state, ok = rig.set_phase(1, 0.6, state)
    assert not ok
    assert rig.phases[1] == pytest.approx(0.1)
    assert np.array_equal(new_state, before)
    assert rig.subs[1].drive.offset == pytest.approx(
        rig.subs[1].offset0 + 2 * math.pi * 0.1
    )


# --- validación ----------------------------------------------------------------------------------


def test_rig_validation(leg_mechanism):
    ch = biped_character()
    specs = make_specs(leg_mechanism)
    with pytest.raises(ValueError, match="al menos un"):
        MultiRig(ch, [])
    with pytest.raises(ValueError, match="mismo nombre"):
        MultiRig(ch, [specs[0], replace(specs[1], name=specs[0].name)])
    far = replace(specs[1], attach=PointRef("r_shank", (0.0, -1.2)))  # no coincide con el marcador
    with pytest.raises(ValueError, match="infactibles|factibles"):
        MultiRig(ch, [specs[0], far])
    with pytest.raises(KeyError):
        MultiRig(ch, [replace(specs[0], attach=PointRef("no_existe", (0.0, 0.0)))])
