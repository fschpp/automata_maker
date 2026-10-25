"""Candidatos de ``CharacterFamily`` por cinemática inversa (bundle 2 de la mejora de la pierna)."""

import math

import numpy as np
import pytest

from mechchar.core.assembly import world_point
from mechchar.core.character import LEG_ATTACH, LEG_MARKER, leg_character
from mechchar.core.kinematics import point_curve, track_marker
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.families import FOUR_BAR
from mechchar.optim.fitcore import (
    curve_and_jacobian_generic,
    fit,
    trace_curve_generic,
)

N = 48


def ellipse(cx: float, cy: float, a: float, b: float, n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


REACHABLE = ellipse(1.0, -2.0, 0.35, 0.15)  # el pie la recorre (ver test_kinematics)
UNREACHABLE = ellipse(0.9, -2.4, 0.55, 0.25)  # la elipse por defecto de la escena


def _family() -> CharacterFamily:
    return CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)


def test_candidates_for_a_reachable_curve_start_close_to_it() -> None:
    """Antes (curva del marcador trasladada) el mejor candidato quedaba a ~46 %; con la curva
    real del punto de actuación queda a ~3-5 %."""
    fam = _family()
    cands = fam.candidates(REACHABLE, N, 16000, 2, np.random.default_rng(0))
    assert cands and fam.reach_error < 1e-5
    assert cands[0].rel_error < 0.10
    assert [c.rel_error for c in cands] == sorted(c.rel_error for c in cands)


def test_candidate_marker_curve_has_the_target_size_not_another_branch() -> None:
    """Regresión: resolver la pose inicial solo por la posición del punto de actuación puede caer
    en otra rama del personaje (el marcador recorría una curva de ~4 veces el tamaño)."""
    fam = _family()
    c = fam.candidates(REACHABLE, N, 16000, 1, np.random.default_rng(0))[0]
    curve = trace_curve_generic(fam._builder(c.variant), c.p0, N)
    assert curve is not None
    assert bbox_diag(curve) == pytest.approx(bbox_diag(REACHABLE), rel=0.3)
    assert np.linalg.norm(curve.mean(0) - REACHABLE.mean(0)) < 0.1


def test_initial_pose_puts_the_attach_point_on_the_target_in_the_marker_branch() -> None:
    fam = _family()
    fam.prepare(REACHABLE)
    states, _ = track_marker(fam.character, LEG_MARKER, REACHABLE)
    attach = point_curve(fam.character, LEG_ATTACH, states)
    ia = fam.character.index(LEG_ATTACH.part)
    la = fam.character.local_point(LEG_ATTACH.part, LEG_ATTACH.point)
    for k in (0, 7, 23, 41):
        pose = fam.initial_pose(attach[k] + [0.01, -0.01])  # cerca, no exactamente sobre la curva
        assert pose is not None
        assert world_point(pose, ia, la) == pytest.approx(attach[k] + [0.01, -0.01], abs=1e-6)
        assert pose == pytest.approx(states[k], abs=0.1)  # misma rama, no otra pose


def test_initial_pose_without_a_bank_follows_the_rest_branch() -> None:
    fam = _family()
    rest = fam.character.build()[1]
    ia = fam.character.index(LEG_ATTACH.part)
    la = fam.character.local_point(LEG_ATTACH.part, LEG_ATTACH.point)
    target = world_point(rest, ia, la) + [0.2, 0.1]
    pose = fam.initial_pose(target)
    assert pose is not None
    assert world_point(pose, ia, la) == pytest.approx(target, abs=1e-6)
    assert fam.initial_pose(np.array([0.0, -6.0])) is None  # fuera del alcance


def test_prepare_resets_the_warm_start() -> None:
    fam = _family()
    fam.prepare(REACHABLE)
    assert fam.initial_pose(np.array([0.0, -1.7])) is not None
    assert fam._hint is not None
    fam.prepare(REACHABLE)
    assert fam._hint is None


def test_unreachable_curve_reports_the_reach_error_and_still_gives_candidates() -> None:
    """La escena por defecto no es alcanzable: se avisa (reach_error) y se sigue con el método
    anterior en vez de dejar al usuario sin candidatos."""
    fam = _family()
    cands = fam.candidates(UNREACHABLE, N, 16000, 2, np.random.default_rng(0))
    assert fam.reach_error > 0.1
    assert cands


def test_unposed_build_gives_the_same_jacobian() -> None:
    """Las construcciones perturbadas sin cinemática inversa no cambian el gradiente."""
    fam = _family()
    c = fam.candidates(REACHABLE, N, 16000, 1, np.random.default_rng(0))[0]
    build = fam._builder(c.variant)

    def unposed(q: np.ndarray):
        return fam.build_unposed(q, c.variant)

    x1, j1 = curve_and_jacobian_generic(build, c.p0, 16)  # type: ignore[misc]
    x2, j2 = curve_and_jacobian_generic(build, c.p0, 16, unposed)  # type: ignore[misc]
    assert x1 == pytest.approx(x2, abs=1e-12)
    assert j1 == pytest.approx(j2, abs=1e-9)


def test_fit_reaches_a_reachable_ellipse_below_the_acceptance_threshold() -> None:
    """Criterio de M2 en la pierna de prueba: antes ~41 %, ahora < 3 % (medido 0,55-0,63 %)."""
    fam = _family()
    res = fit(REACHABLE, [fam], n_refine=2)
    assert res is not None
    assert res.rms_rel < 0.03
    # el resultado es de verdad el marcador del personaje: se vuelve a medir desde cero
    curve = trace_curve_generic(fam._builder(res.variant), res.params, 180)
    assert curve is not None
    assert chamfer_rms(curve, REACHABLE) / bbox_diag(REACHABLE) < 0.03
