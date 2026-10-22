import numpy as np
import pytest

from mechchar.core.mechanisms import four_bar
from mechchar.core.playback import MAX_DT, Playback


def make() -> tuple:
    fb = four_bar(4.0, 1.0, 3.5, 3.0)
    return fb, Playback(fb.asm, fb.state0, period=2.0)


def test_full_revolution_returns_to_start_without_branch_jump():
    fb, pb = make()
    start = fb.tracer_world(pb.state)
    for _ in range(200):  # 200 x 0.01 s = 2 s = 1 ciclo
        pb.advance(0.01)
        assert pb.ok
    assert pb.t == pytest.approx(1.0)
    assert fb.tracer_world(pb.state) == pytest.approx(start, abs=1e-6)


def test_paused_playback_does_not_move():
    _, pb = make()
    pb.playing = False
    s = pb.state.copy()
    assert pb.advance(0.05) is False
    assert np.array_equal(pb.state, s) and pb.t == 0.0


def test_large_dt_is_clamped():
    _, pb = make()
    pb.advance(10.0)
    assert pb.t == pytest.approx(MAX_DT / pb.period)


def test_seek_phase_matches_continuous_advance():
    fb, pb = make()
    for _ in range(50):  # t = 0.25
        pb.advance(0.01)
    via_advance = fb.tracer_world(pb.state)
    _, pb2 = make()
    pb2.seek_phase(0.25)
    assert fb.tracer_world(pb2.state) == pytest.approx(via_advance, abs=1e-6)
    assert pb2.phase == pytest.approx(0.25)


def test_phase_wraps_but_total_time_keeps_growing():
    _, pb = make()
    pb.seek_phase(0.9)
    pb.step_to(1.1)
    assert pb.phase == pytest.approx(0.1)
    assert pb.t == pytest.approx(1.1)


def test_invalid_period():
    fb = four_bar(4.0, 1.0, 3.5, 3.0)
    with pytest.raises(ValueError):
        Playback(fb.asm, fb.state0, period=0.0)
