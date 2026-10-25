"""Familia manivela-corredera: armado, curva, gradiente, compacidad, ajuste y base."""

import math

import numpy as np
import pytest

from mechchar.core.mechanisms import SliderCrank
from mechchar.db.motion_db import build_database
from mechchar.optim.curves import bbox_diag
from mechchar.optim.families import ALL_FAMILIES, family_by_name, fit_best
from mechchar.optim.fitcore import (
    MAX_LINK_RATIO,
    Variant,
    curve_and_jacobian_finite_differences,
    curve_and_jacobian_generic,
    fit,
)
from mechchar.optim.slidercrank_fit import (
    N_PARAMS,
    SLIDER_CRANK,
    _compactness,
    _fast_local_curves,
    build,
    to_slider_crank,
    trace_curve,
)

# r, l, h, tx, ty = 0,5; 2,0; 0,3; 1,0; 0,4 con la bancada en la pose identidad y psi = 0
P = np.array([0.0, 0.0, 0.0, 0.5, 2.0, 0.3, 1.0, 0.4, 0.0])


def _pose(p: np.ndarray, ax: float, ay: float, phi: float) -> np.ndarray:
    q = p.copy()
    q[0], q[1], q[2], q[8] = ax, ay, phi, phi  # psi = phi: la manivela parte en 0° del marco local
    return q


@pytest.mark.parametrize("branch", (1, -1))
def test_build_is_feasible_and_the_slider_stays_on_its_guide(branch):
    b = build(_pose(P, 1.5, -0.7, 0.6), branch)
    assert b is not None
    assert np.abs(b.asm.residual(b.state0, 0.0)).max() < 1e-9
    mech = to_slider_crank(_pose(P, 1.5, -0.7, 0.6), branch)
    assert isinstance(mech, SliderCrank)
    # extremo de la biela en el marco de la bancada: y = h
    c, s = math.cos(0.6), math.sin(0.6)
    w = mech.joints(mech.state0)[2] - np.array([1.5, -0.7])
    assert (-s * w[0] + c * w[1]) == pytest.approx(P[5], abs=1e-9)


def test_build_rejects_a_rod_too_short_to_reach_the_guide():
    q = P.copy()
    q[4], q[5] = 0.2, 2.0  # l < |h - r sin psi|
    assert build(q, 1) is None
    q = P.copy()
    q[3] = -1.0
    assert build(q, 1) is None


@pytest.mark.parametrize("branch", (1, -1))
def test_fast_curve_matches_simulator(branch):
    n = 36
    arrays = (np.array([v]) for v in (0.5, 2.0, 0.3, 1.0, 0.4))
    local, valid = _fast_local_curves(*arrays, np.array([float(branch)]), n)
    assert valid[0]
    sim = trace_curve(P, branch, n)
    assert sim is not None
    assert np.max(np.abs(sim - local[0])) < 1e-6


def test_world_pose_only_moves_the_curve_rigidly():
    n = 24
    base = trace_curve(P, 1, n)
    moved = trace_curve(_pose(P, 1.5, -0.7, 0.6), 1, n)
    assert base is not None and moved is not None
    c, s = math.cos(0.6), math.sin(0.6)
    expect = base @ np.array([[c, s], [-s, c]]) + np.array([1.5, -0.7])
    assert moved == pytest.approx(expect, abs=1e-6)


def test_a_rod_shorter_than_the_guide_distance_is_not_a_valid_sample():
    local, valid = _fast_local_curves(
        *(np.array([v]) for v in (0.5, 0.6, 1.0, 0, 0)), np.array([1.0]), 24
    )
    assert not valid[0]  # |h - r sin| llega a 1,5 > l = 0,6


@pytest.mark.parametrize("branch", (1, -1))
def test_implicit_function_gradient_matches_finite_differences(branch):
    p = _pose(P, 0.4, 0.2, 0.3)

    def build_fn(q):
        return build(q, branch)

    n = 24
    fast = curve_and_jacobian_generic(build_fn, p, n)
    slow = curve_and_jacobian_finite_differences(build_fn, p, n)
    assert fast is not None and slow is not None
    assert np.max(np.abs(fast[1] - slow[1])) / max(1.0, np.max(np.abs(fast[1]))) < 1e-5
    assert fast[0] == pytest.approx(slow[0], abs=1e-9)


def test_compactness_jacobian_matches_finite_differences():
    p = P.copy()
    p[3], p[4], p[5], p[6], p[7] = 9.0, 7.0, -8.0, 5.0, 6.0  # todo se pasa del límite
    size = 2.0  # límite 1,5 · 2 = 3
    res, jac = _compactness(p, size)
    assert np.count_nonzero(res) == 4
    fd = np.zeros_like(jac)
    for j in range(N_PARAMS):
        h = 1e-7
        pp, pm = p.copy(), p.copy()
        pp[j] += h
        pm[j] -= h
        fd[:, j] = (_compactness(pp, size)[0] - _compactness(pm, size)[0]) / (2 * h)
    assert np.max(np.abs(jac - fd)) < 1e-5


def test_registry():
    assert family_by_name("slider_crank") is SLIDER_CRANK
    assert ALL_FAMILIES[-1] is SLIDER_CRANK
    assert SLIDER_CRANK.n_params == N_PARAMS


def test_mechanism_outline_includes_the_guide():
    mech = to_slider_crank(P, 1)
    assert mech is not None
    pts = mech.outline_points(mech.state0)
    assert pts.shape == (6, 2)
    reach = P[3] + P[4]
    assert pts[:, 0].min() <= -reach + 1e-9 and pts[:, 0].max() >= reach - 1e-9


def test_fit_recovers_a_curve_made_by_a_slider_crank():
    target = trace_curve(P, 1, 120)
    assert target is not None
    result = fit(target, [SLIDER_CRANK], seed=2)
    assert result is not None and result.family == "slider_crank"
    assert result.rms_rel < 0.03


def test_fitted_slider_crank_is_compact():
    t = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    target = np.column_stack([2.0 * np.cos(t), 1.0 * np.sin(t)])
    result = fit(target, [SLIDER_CRANK])
    assert result is not None
    p = result.params
    limit = MAX_LINK_RATIO * bbox_diag(target) * 1.01
    assert max(p[3], p[4], abs(p[5]), math.hypot(p[6], p[7])) <= limit


def test_fit_best_can_return_a_slider_crank_and_reports_it_in_progress():
    labels: list[str] = []
    target = trace_curve(P, 1, 120)
    result = fit_best(target, n_candidates=8000, on_progress=lambda lab, _c: labels.append(lab))
    assert result is not None and result.rms_rel < 0.03
    assert "slider_crank" in set(labels)


def test_the_database_can_sample_the_family():
    db = build_database("slider_crank", n_target=120, n_samples=8_000, workers=1)
    assert db.family == "slider_crank" and len(db.pts) > 20
    assert set(np.unique(db.branch)) <= {-1, 1}
    assert Variant(int(db.branch[0])).ratio == 0.0


def test_initial_candidates_already_look_like_the_target():
    """``assemble`` devuelve la pose y el desfase correctos: el mejor candidato, antes de
    refinar, ya queda cerca (si psi estuviera mal, el refinamiento lo escondería)."""
    target = trace_curve(P, 1, 48)
    assert target is not None
    best = SLIDER_CRANK.candidates(target, 48, 16000, 3, np.random.default_rng(0))[0]
    assert best.rel_error < 0.12
    sim = trace_curve(best.p0, best.variant.branch, 48)
    assert sim is not None
    assert np.abs(sim - best.curve).max() < 1e-6  # la curva alineada es la que arma p0
