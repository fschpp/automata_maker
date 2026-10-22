import math

import numpy as np
import pytest

from mechchar.optim.curves import (
    bbox_diag,
    best_correspondence,
    chamfer_rms,
    resample_closed,
)
from mechchar.optim.fourbar_fit import (
    N_PARAMS,
    _fast_local_curves,
    curve_and_jacobian,
    distance_residuals,
    fit_four_bar,
    trace_curve,
)


def ellipse(m: int = 200, a: float = 3.0, b: float = 1.5) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


def bean(m: int = 200) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return np.column_stack([2.0 * np.cos(t), 1.2 * np.sin(t) + 0.5 * np.cos(2 * t)])


# p = (ax, ay, phi, d, r, lc, rr, tx, ty, psi): manivela-balancín con pose arbitraria
P_TEST = np.array([0.4, -0.3, 0.35, 4.0, 1.0, 3.5, 3.0, 1.7, 0.9, 0.6])


# --- curvas -------------------------------------------------------------------------------------


def test_resample_closed_is_uniform_and_keeps_shape():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    out = resample_closed(sq, 8)
    assert out.shape == (8, 2)
    assert np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1) == pytest.approx(1.0)


def test_chamfer_rms_zero_for_same_set_regardless_of_order():
    c = ellipse(50)
    assert chamfer_rms(c, np.roll(c, 7, axis=0)) == pytest.approx(0.0, abs=1e-12)
    assert chamfer_rms(c, c + [0.0, 0.5]) > 0.1


def test_bbox_diag():
    assert bbox_diag(np.array([[0.0, 0.0], [3.0, 4.0]])) == pytest.approx(5.0)


def test_best_correspondence_recovers_shift_and_direction():
    y = resample_closed(ellipse(100), 40)
    shift, direction, cost = best_correspondence(np.roll(y, -5, axis=0), y)
    assert (shift, direction) == (5, 1)
    assert cost == pytest.approx(0.0, abs=1e-12)
    shift, direction, cost = best_correspondence(y[::-1].copy(), y)
    assert direction == -1 and cost == pytest.approx(0.0, abs=1e-12)


# --- modelo y gradiente -------------------------------------------------------------------------


def test_fast_curve_matches_simulator():
    n = 36
    r, lc, rr, tx, ty = 0.25, 0.875, 0.75, 0.4, 0.2  # = P_TEST / 4, d = 1
    local, valid = _fast_local_curves(*(np.array([v]) for v in (r, lc, rr, tx, ty)),
                                      np.array([1.0]), n)
    assert valid[0]
    p = np.array([0.0, 0.0, 0.0, 1.0, r, lc, rr, tx, ty, 0.0])
    sim = trace_curve(p, 1, n)
    assert sim is not None
    assert np.max(np.abs(sim - local[0])) < 1e-6


def test_implicit_function_gradient_matches_finite_differences():
    n = 24
    out = curve_and_jacobian(P_TEST, 1, n)
    assert out is not None
    x, dx = out
    jac_fd = np.zeros_like(dx)
    for j in range(N_PARAMS):
        h = 1e-5 * (1.0 + abs(P_TEST[j]))
        pp, pm = P_TEST.copy(), P_TEST.copy()
        pp[j] += h
        pm[j] -= h
        xp, xm = trace_curve(pp, 1, n), trace_curve(pm, 1, n)
        assert xp is not None and xm is not None
        jac_fd[:, :, j] = (xp - xm) / (2 * h)
    assert np.max(np.abs(dx - jac_fd)) / max(1.0, np.max(np.abs(dx))) < 1e-5
    assert x == pytest.approx(trace_curve(P_TEST, 1, n), abs=1e-9)


def test_unbuildable_parameters_return_none():
    bad = P_TEST.copy()
    bad[4] = 6.0  # manivela demasiado larga: el ciclo no se completa
    assert curve_and_jacobian(bad, 1, 24) is None
    assert trace_curve(bad, 1, 24) is None


def test_distance_residuals_ignore_timing_and_vanish_on_the_same_curve():
    y_dense = resample_closed(ellipse(), 400)
    y = resample_closed(ellipse(), 48)
    # misma curva recorrida con otro desfase y otro sentido: no debe cambiar el residuo
    x = np.roll(y, 11, axis=0)[::-1].copy()
    r, jac = distance_residuals(x, np.zeros((48, 2, N_PARAMS)), y, y_dense)
    assert r.shape == (96,) and jac.shape == (96, N_PARAMS)
    assert np.max(np.abs(r)) < 2e-3


# --- ajuste (criterio de aceptación de M2) ------------------------------------------------------


@pytest.mark.parametrize(("name", "target"), [("elipse", ellipse()), ("frijol", bean())])
def test_fit_reaches_three_percent_of_curve_size(name, target):
    result = fit_four_bar(target)
    assert result is not None, name
    assert result.rms_rel < 0.03, f"{name}: {result.rms_rel:.4f}"


def test_fit_recovers_exactly_a_curve_made_by_a_four_bar():
    target = trace_curve(P_TEST, 1, 120)
    assert target is not None
    result = fit_four_bar(target, seed=1)
    assert result is not None
    assert result.rms_rel < 0.03
