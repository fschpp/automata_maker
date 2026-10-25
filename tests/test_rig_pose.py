"""Pose inicial por cinemática inversa en el bípedo (bundle 3 de la mejora de la pierna)."""

import math

import numpy as np
import pytest

from mechchar.core.character import (
    LEG_ATTACH,
    LEG_MARKER,
    biped_character,
    leg_character,
    leg_refs,
    prefix_pose,
)
from mechchar.core.rig import MultiRig, SubSpec
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.curves import bbox_diag, chamfer_rms, resample_closed
from mechchar.optim.families import FOUR_BAR
from mechchar.optim.fitcore import trace_curve_generic

N = 48
T = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
REACHABLE = np.column_stack([1.0 + 0.35 * np.cos(T), -2.0 + 0.15 * np.sin(T)])


@pytest.fixture(scope="module")
def fitted():
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    c = fam.candidates(REACHABLE, N, 16000, 1, np.random.default_rng(0))[0]
    single = trace_curve_generic(fam._builder(c.variant), c.p0, N)
    assert single is not None
    return fam, c, single


def _rig(c, pose):
    specs = []
    for prefix in ("l_", "r_"):
        attach, marker = leg_refs(prefix)
        specs.append(
            SubSpec(
                prefix + "leg",
                FOUR_BAR.build(c.p0, c.variant),
                FOUR_BAR.to_mechanism(c.p0, c.variant),
                attach,
                marker,
                prefix_pose(pose, prefix) if pose is not None else None,
            )
        )
    return MultiRig(biped_character(), specs)


def _leg_curves(rig):
    res = rig.asm.trace_cycle(rig.state0, n_steps=N, max_jump=1.0)
    assert res.ok, res.message
    return [np.array([rig.marker_world(s, k) for s in res.states]) for k in (0, 1)]


def test_pose_for_skips_the_fixed_pelvis_and_names_the_moving_parts(fitted) -> None:
    fam, c, _ = fitted
    pose = fam.pose_for(c.p0, c.variant)
    assert pose is not None and set(pose) == {"thigh", "shank", "foot", "rod"}
    assert set(prefix_pose(pose, "r_")) == {"r_thigh", "r_shank", "r_foot", "r_rod"}


def test_biped_legs_trace_the_curve_that_the_fit_measured(fitted) -> None:
    """Con la pose de la IK cada pierna del bípedo recorre lo mismo que la pierna ajustada."""
    fam, c, single = fitted
    rig = _rig(c, fam.pose_for(c.p0, c.variant))
    for curve in _leg_curves(rig):
        assert chamfer_rms(curve, single) / bbox_diag(single) < 1e-6


def test_without_the_pose_the_biped_can_silently_land_on_another_branch(fitted) -> None:
    """Por eso hace falta: el residuo en t = 0 es ~0 igual, pero la pierna traza otra curva."""
    _, c, single = fitted
    rig = _rig(c, None)
    assert np.abs(rig.asm.residual(rig.state0, 0.0)).max() < 1e-6
    for curve in _leg_curves(rig):
        assert chamfer_rms(curve, single) / bbox_diag(single) > 0.1


def test_reopening_a_project_recovers_the_same_pose_with_prepare(fitted) -> None:
    """Una familia nueva (como al abrir un archivo) reconstruye la pose desde la curva objetivo."""
    fam, c, _ = fitted
    expected = fam.pose_for(c.p0, c.variant)
    fresh = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    fresh.prepare(resample_closed(REACHABLE, N))
    got = fresh.pose_for(c.p0, c.variant)
    assert expected is not None and got is not None
    for name, xya in expected.items():
        assert got[name] == pytest.approx(xya, abs=1e-6)


def test_unknown_part_in_the_pose_is_rejected(fitted) -> None:
    fam, c, _ = fitted
    pose = {"no_existe": (0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="no_existe"):
        _rig(c, pose)


def test_pose_for_is_none_when_the_mechanism_cannot_reach_its_attach_point(fitted) -> None:
    fam, c, _ = fitted
    far = c.p0.copy()
    far[:2] += [5.0, -5.0]  # el mecanismo entero lejos del personaje
    fresh = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    assert fresh.pose_for(far, c.variant) is None
