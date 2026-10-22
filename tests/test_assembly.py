import math

import numpy as np
import pytest

from mechchar.core.assembly import Assembly, Drive, FixedState, Phase, Pin
from mechchar.core.mechanisms import circle_intersections, crank_slider, four_bar

D, R, L, RR = 4.0, 1.0, 3.5, 3.0  # Grashof: manivela-balancín


def analytic_coupler_point(t: float, prev: np.ndarray, fb_tracer: tuple[float, float]):
    """Punto trazador del acoplador por intersección de circunferencias (rama continua)."""
    th = 2.0 * math.pi * t
    crank_tip = (R * math.cos(th), R * math.sin(th))
    cands = circle_intersections(crank_tip, L, (D, 0.0), RR)
    p = min(cands, key=lambda q: float(np.linalg.norm(q - prev)))
    ang = math.atan2(p[1] - crank_tip[1], p[0] - crank_tip[0])
    c, s = math.cos(ang), math.sin(ang)
    tracer = np.array(crank_tip) + np.array([c * fb_tracer[0] - s * fb_tracer[1],
                                             s * fb_tracer[0] + c * fb_tracer[1]])
    return tracer, p


def fd_jacobian(asm: Assembly, s: np.ndarray, t: float, eps: float = 1e-6) -> np.ndarray:
    cols = []
    for j in range(s.size):
        sp, sm = s.copy(), s.copy()
        sp[j] += eps
        sm[j] -= eps
        cols.append((asm.residual(sp, t) - asm.residual(sm, t)) / (2 * eps))
    return np.array(cols).T


def test_four_bar_cycle_matches_analytic_curve():
    fb = four_bar(D, R, L, RR)
    res = fb.asm.trace_cycle(fb.state0, n_steps=360)
    assert res.ok, res.reason
    prev = fb.asm.world_point(fb.state0, fb.coupler, (L, 0.0))
    worst = 0.0
    for k, state in enumerate(res.states):
        expected, prev = analytic_coupler_point(k / 360, prev, fb.tracer)
        worst = max(worst, float(np.linalg.norm(fb.tracer_world(state) - expected)))
    assert worst < 1e-6


def test_four_bar_closure_and_branch_continuity():
    fb = four_bar(D, R, L, RR)
    res = fb.asm.trace_cycle(fb.state0, n_steps=360)
    assert res.ok
    assert res.max_jump < 0.1  # pasos suaves: no hubo salto de rama


def test_four_bar_other_branch_is_mirror_assembly():
    up, down = four_bar(D, R, L, RR, branch=1), four_bar(D, R, L, RR, branch=-1)
    assert up.state0[3 * up.coupler + 2] == pytest.approx(-down.state0[3 * down.coupler + 2])
    assert up.asm.trace_cycle(up.state0, 120).ok
    assert down.asm.trace_cycle(down.state0, 120).ok


def test_non_grashof_mechanism_is_rejected():
    fb = four_bar(4.0, 3.0, 2.0, 2.0)  # factible en t=0 pero la manivela no gira completa
    res = fb.asm.trace_cycle(fb.state0, n_steps=360)
    assert not res.ok
    assert res.reason


def test_crank_slider_matches_analytic_solution():
    cs = crank_slider(1.0, 3.0)
    res = cs.asm.trace_cycle(cs.state0, n_steps=360)
    assert res.ok, res.reason
    for k, state in enumerate(res.states):
        th = 2.0 * math.pi * k / 360
        expected = cs.r * math.cos(th) + math.sqrt(cs.l**2 - (cs.r * math.sin(th)) ** 2)
        assert cs.slider_x(state) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", ["four_bar", "crank_slider"])
def test_analytic_jacobian_matches_finite_differences(name):
    mech = four_bar(D, R, L, RR) if name == "four_bar" else crank_slider(1.0, 3.0)
    rng = np.random.default_rng(0)
    s = mech.state0 + 0.2 * rng.standard_normal(mech.state0.size)
    t = 0.3
    jac = mech.asm.jacobian(s, t)
    jac_fd = fd_jacobian(mech.asm, s, t)
    assert np.max(np.abs(jac - jac_fd)) / max(1.0, np.max(np.abs(jac))) < 1e-5


def test_phase_constraint_gears_and_jacobian():
    asm = Assembly()
    g = asm.add_component("ground")
    a = asm.add_component("gear_a", 0.0, 0.0)
    b = asm.add_component("gear_b", 2.0, 0.0)
    asm.add(FixedState(g))
    asm.add(Pin(g, (0.0, 0.0), a, (0.0, 0.0)))
    asm.add(Pin(g, (2.0, 0.0), b, (0.0, 0.0)))
    asm.add(Drive(a))
    asm.add(Phase(a, b, ratio=-2.0, offset=0.3))
    res = asm.solve(asm.initial_state(), 0.125)
    assert res.ok
    assert res.state[3 * b + 2] == pytest.approx(-2.0 * (2 * math.pi * 0.125) + 0.3)
    s = res.state + 0.1
    assert np.max(np.abs(asm.jacobian(s, 0.1) - fd_jacobian(asm, s, 0.1))) < 1e-6


def test_constraint_analysis_dof_and_redundancy():
    fb = four_bar(D, R, L, RR)
    base = fb.asm.analyze(fb.state0)
    assert (base.dof, base.redundant) == (0, 0)

    under = four_bar(D, R, L, RR)
    under.asm.constraints.pop()  # sin Input Driver: queda 1 grado de libertad
    assert under.asm.analyze(under.state0).dof == 1

    over = four_bar(D, R, L, RR)
    over.asm.add(Drive(over.crank))  # restricción duplicada pero consistente
    a = over.asm.analyze(over.state0)
    assert (a.dof, a.redundant) == (0, 1)


def test_inconsistent_constraints_fail_to_solve():
    fb = four_bar(D, R, L, RR)
    fb.asm.add(Drive(fb.crank, offset=1.0))  # contradice al Drive original
    assert not fb.asm.solve(fb.state0, 0.0).ok
