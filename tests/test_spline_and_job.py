import threading
import time

import numpy as np
import pytest

from mechchar.core.spline import closed_catmull_rom, pick_point
from mechchar.core.viewport import Viewport
from mechchar.optim.fourbar_fit import to_four_bar, trace_curve
from mechchar.optim.job import LatestJobRunner

from .test_fourbar_fit import P_TEST


def test_catmull_rom_interpolates_control_points_and_closes():
    pts = np.array([[0, 0], [2, 0], [3, 2], [1, 3], [-1, 1]], dtype=float)
    k = 12
    curve = closed_catmull_rom(pts, k)
    assert curve.shape == (5 * k, 2)
    assert curve[::k] == pytest.approx(pts)
    # cierra: el último punto muestreado está cerca del primer control, sin salto grande
    assert np.linalg.norm(curve[-1] - curve[0]) < 1.0


def test_catmull_rom_needs_three_points():
    with pytest.raises(ValueError):
        closed_catmull_rom(np.zeros((2, 2)))


def test_pick_point_uses_screen_distance_and_radius():
    vp = Viewport(scale=10.0, ox=100.0, oy=100.0)
    pts = [(0.0, 0.0), (5.0, 0.0), (5.2, 0.0)]
    assert pick_point(pts, vp, 101.0, 100.0, 8.0) == 0
    assert pick_point(pts, vp, 149.0, 100.0, 8.0) == 1
    assert pick_point(pts, vp, 300.0, 300.0, 8.0) is None
    assert pick_point(pts, vp, 152.0, 100.0, 8.0) == 2  # el más cercano gana


def test_to_four_bar_reproduces_the_fitted_curve():
    fb = to_four_bar(P_TEST, 1)
    assert fb is not None
    res = fb.asm.trace_cycle(fb.state0, n_steps=60, max_jump=1.0)
    assert res.ok
    ref = trace_curve(P_TEST, 1, 60)
    assert ref is not None
    pts = np.array([fb.tracer_world(s) for s in res.states])
    assert pts == pytest.approx(ref, abs=1e-9)
    assert set(fb.lengths) == {fb.crank, fb.coupler, fb.rocker}


def test_runner_skips_obsolete_requests_and_reports_errors():
    done: list[tuple[int, object, object]] = []
    gate = threading.Event()

    def work(item):
        gate.wait(2.0)
        if item == 99:
            raise ValueError("boom")
        return item * 2

    runner = LatestJobRunner(work, lambda item, res, err: done.append((item, res, err)))
    runner.submit(1)
    time.sleep(0.05)
    assert runner.busy
    runner.submit(2)
    runner.submit(3)  # reemplaza a 2 mientras 1 corre
    gate.set()
    assert runner.wait_idle(3.0)
    assert [d[0] for d in done] == [3]  # el 1 quedó obsoleto, el 2 fue reemplazado
    assert done[0][1] == 6 and done[0][2] is None

    runner.submit(99)
    assert runner.wait_idle(3.0)
    assert isinstance(done[-1][2], ValueError)
    assert not runner.busy
