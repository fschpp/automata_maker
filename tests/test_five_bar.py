import math

import numpy as np
import pytest

from mechchar.core.mechanisms import circle_intersections, geared_five_bar
from mechchar.optim.curves import bbox_diag
from mechchar.optim.families import ALL_FAMILIES, family_by_name, fit_best
from mechchar.optim.fitcore import (
    MAX_LINK_RATIO,
    Variant,
    curve_and_jacobian_finite_differences,
    curve_and_jacobian_generic,
    fit,
    refine_from,
)
from mechchar.optim.fivebar_fit import (
    CRANK_FIT,
    GEARED_FIVE_BAR,
    N_PARAMS,
    _compactness,
    _fast_local_curves,
    _ratio_split,
    build,
    to_five_bar,
    trace_curve,
)
from mechchar.optim.fourbar_fit import FOUR_BAR
from mechchar.optim.fourbar_fit import trace_curve as trace_four_bar

# p = (ax, ay, phi, d, r1, r2, l1, l2, tx, ty, psi, delta)
P5 = np.array([0.2, -0.1, 0.3, 4.0, 0.9, 0.7, 3.4, 3.0, 1.7, 0.8, 0.0, 0.9])
P4 = np.array([0.4, -0.3, 0.35, 4.0, 1.0, 3.5, 3.0, 1.7, 0.9, 0.6])
RATIOS = (-2.0, -1.0, 1.0, 2.0)


def ellipse(m: int = 200, a: float = 3.0, b: float = 1.5) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


# --- mecanismo (núcleo) -------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
def test_five_bar_matches_analytic_positions(ratio):
    d, r1, r2, l1, l2 = 1.0, 0.25, 0.2, 0.9, 0.8
    delta, psi, pose = 0.7, 0.3, (0.4, -0.2, 0.5)
    m = geared_five_bar(
        d=d, r1=r1, r2=r2, l1=l1, l2=l2, ratio=ratio, tracer=(1.2, 0.3),
        branch=1, delta=delta, psi=psi, pose=pose,
    )
    n = 60
    cyc = m.asm.trace_cycle(m.state0, n_steps=n, max_jump=1.0)
    assert cyc.ok, cyc.reason
    c, s = math.cos(pose[2]), math.sin(pose[2])
    to_world = lambda q: np.array([pose[0], pose[1]]) + np.array([[c, -s], [s, c]]) @ q  # noqa: E731
    for k in range(0, n, 7):
        t = k / n
        th1 = psi + 2 * math.pi * t
        th2 = ratio * th1 + delta - ratio * 0.0  # θ₂ = ratio·θ₁ + δ (ángulos del marco local)
        # la fase del driver en el marco local es psi + 2πt; θ₂ local = ratio·(psi+2πt) + δ
        th2 = ratio * th1 + delta
        b1 = np.array([r1 * math.cos(th1), r1 * math.sin(th1)])
        b2 = np.array([d + r2 * math.cos(th2), r2 * math.sin(th2)])
        j = circle_intersections(tuple(b1), l1, tuple(b2), l2)[0]
        joints = m.joints(cyc.states[k])
        assert joints[4] == pytest.approx(to_world(j), abs=1e-7)
        assert joints[2] == pytest.approx(to_world(b1), abs=1e-7)
        assert joints[3] == pytest.approx(to_world(b2), abs=1e-7)


@pytest.mark.parametrize("ratio", RATIOS)
def test_five_bar_is_square_and_has_one_dof_per_phase(ratio):
    m = geared_five_bar(d=1.0, r1=0.25, r2=0.2, l1=0.9, l2=0.8, ratio=ratio)
    an = m.asm.analyze(m.state0, 0.0)
    assert (m.asm.n_vars, m.asm.n_eq) == (15, 15)
    assert an.dof == 0 and an.redundant == 0


def test_gear_radii_follow_the_ratio():
    m = geared_five_bar(d=3.0, r1=0.3, r2=0.3, l1=2.0, l2=2.0, ratio=-2.0)
    r1, r2 = m.gear_radii()
    assert r1 / r2 == pytest.approx(2.0) and r1 + r2 == pytest.approx(3.0)
    assert _ratio_split(2.0) == pytest.approx((2 / 3, 1 / 3))


def test_unassemblable_five_bar():
    with pytest.raises(ValueError):
        geared_five_bar(d=1.0, r1=0.2, r2=0.2, l1=0.1, l2=0.1, ratio=-1.0)  # bielas muy cortas
    with pytest.raises(ValueError):
        geared_five_bar(d=1.0, r1=0.2, r2=0.2, l1=1.0, l2=1.0, ratio=0.0)
    assert build(np.where(np.arange(N_PARAMS) == 3, -1.0, P5), Variant(1, -1.0)) is None
    assert to_five_bar(np.where(np.arange(N_PARAMS) == 6, 0.01, P5), Variant(1, -1.0)) is None


# --- evaluación rápida y gradiente --------------------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
def test_fast_curve_matches_simulator(ratio):
    n = 36
    # P5 / 4 con d = 1 y la bancada en la pose identidad
    r1, r2, l1, l2, tx, ty, delta = 0.225, 0.175, 0.85, 0.75, 0.425, 0.2, 0.9
    arrays = (np.array([v]) for v in (r1, r2, l1, l2, tx, ty, ratio, delta, 1.0))
    local, valid = _fast_local_curves(*arrays, n)
    assert valid[0]
    p = np.array([0.0, 0.0, 0.0, 1.0, r1, r2, l1, l2, tx, ty, 0.0, delta])
    sim = trace_curve(p, Variant(1, ratio), n)
    assert sim is not None
    assert np.max(np.abs(sim - local[0])) < 1e-6


@pytest.mark.parametrize("ratio", (-1.0, 2.0))
def test_implicit_function_gradient_matches_finite_differences(ratio):
    v = Variant(1, ratio)

    def build_fn(q):
        return build(q, v)

    n = 24
    fast = curve_and_jacobian_generic(build_fn, P5, n)
    slow = curve_and_jacobian_finite_differences(build_fn, P5, n)
    assert fast is not None and slow is not None
    assert np.max(np.abs(fast[1] - slow[1])) / max(1.0, np.max(np.abs(fast[1]))) < 1e-5
    assert fast[0] == pytest.approx(slow[0], abs=1e-9)


@pytest.mark.parametrize("ratio", (-2.0, 1.0))
def test_compactness_jacobian_matches_finite_differences(ratio):
    v = Variant(1, ratio)
    p = P5.copy()
    p[3], p[4], p[5], p[6] = 2.0, 1.9, 1.4, 9.0  # manivelas fuera de su engranaje y una biela larga
    size = 4.0  # límite de 1,5·4 = 6 -> la biela 1 se pasa
    res, jac = _compactness(p, size, v)
    assert np.count_nonzero(res) >= 3
    fd = np.zeros_like(jac)
    for j in range(N_PARAMS):
        h = 1e-7
        pp, pm = p.copy(), p.copy()
        pp[j] += h
        pm[j] -= h
        fd[:, j] = (_compactness(pp, size, v)[0] - _compactness(pm, size, v)[0]) / (2 * h)
    assert np.max(np.abs(jac - fd)) < 1e-5


# --- ajuste -------------------------------------------------------------------------------------


def test_registry():
    assert [f.name for f in ALL_FAMILIES] == ["four_bar", "geared_five_bar", "slider_crank"]
    assert family_by_name("geared_five_bar") is GEARED_FIVE_BAR
    with pytest.raises(KeyError):
        family_by_name("no existe")


def test_fit_recovers_a_curve_made_by_a_geared_five_bar():
    target = trace_curve(P5, Variant(1, -1.0), 120)
    assert target is not None
    result = fit(target, [GEARED_FIVE_BAR], seed=2)
    assert result is not None and result.family == "geared_five_bar"
    assert result.rms_rel < 0.03
    assert result.variant.ratio in RATIOS


def test_fitted_five_bar_is_compact_and_cranks_fit_in_their_gears():
    target = ellipse()
    result = fit(target, [GEARED_FIVE_BAR])
    assert result is not None
    p, v = result.params, result.variant
    limit = MAX_LINK_RATIO * bbox_diag(target) * 1.01
    assert max(p[3:8].max(), math.hypot(p[8], p[9])) <= limit
    f1, f2 = _ratio_split(v.ratio)
    assert p[4] <= CRANK_FIT * f1 * p[3] * 1.01
    assert p[5] <= CRANK_FIT * f2 * p[3] * 1.01


def test_fit_best_chooses_a_family_and_reports_progress():
    calls: list[tuple[str, tuple[int, ...]]] = []
    result = fit_best(
        ellipse(), n_candidates=8000, on_progress=lambda lab, c: calls.append((lab, c.shape))
    )
    assert result is not None
    assert result.family in {f.name for f in ALL_FAMILIES}
    assert result.rms_rel < 0.03
    labels = {lab for lab, _ in calls}
    assert any("candidato inicial" in lab for lab in labels)
    assert {"four_bar", "geared_five_bar"} <= labels  # evaluaciones del refinamiento de cada una
    assert all(shape[1] == 2 for _, shape in calls)


# --- robustez ante un inicial malo (el video: «even with relatively bad initial guesses») -------


def _bad_start(family, p_true, variant, target, rng, scale):
    """Perturba ``p_true`` hasta obtener un mecanismo que se ensambla pero con error alto."""
    for _ in range(50):
        p0 = p_true * (1.0 + scale * rng.standard_normal(p_true.size))
        if family.build(p0, variant) is None:
            continue
        r = refine_from(family, p0, variant, target, max_nfev=0 + 1)
        if r is not None and r.initial_rms_rel > 0.10:
            return p0
    raise AssertionError("no se encontró un inicial malo ensamblable")


@pytest.mark.parametrize(
    ("family", "p_true", "variant"),
    [(FOUR_BAR, P4, Variant(1)), (GEARED_FIVE_BAR, P5, Variant(1, -1.0))],
)
def test_refinement_converges_from_a_bad_initial_guess(family, p_true, variant):
    if family is FOUR_BAR:
        target = trace_four_bar(p_true, 1, 120)
    else:
        target = trace_curve(p_true, variant, 120)
    assert target is not None
    p0 = _bad_start(family, p_true, variant, target, np.random.default_rng(3), 0.2)
    result = refine_from(family, p0, variant, target)
    assert result is not None
    assert result.initial_rms_rel > 0.10  # el inicial era realmente malo
    assert result.rms_rel < 0.03
