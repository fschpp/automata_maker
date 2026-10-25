import math

import numpy as np
import pytest

from mechchar.core.assembly import world_point
from mechchar.core.character import LEG_ATTACH, LEG_MARKER, PointRef, biped_character, leg_character
from mechchar.core.kinematics import POSE_TOL, point_curve, solve_pose, track_marker


def _ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])


def _world(character, state, ref: PointRef) -> np.ndarray:
    return world_point(state, character.index(ref.part), character.local_point(ref.part, ref.point))


def test_solve_pose_at_rest_is_a_no_op() -> None:
    leg = leg_character()
    rest = leg.build()[1]
    state, err = solve_pose(leg, [(LEG_MARKER, _world(leg, rest, LEG_MARKER))])
    assert err < POSE_TOL
    assert state == pytest.approx(rest, abs=1e-6)


def test_solve_pose_reaches_a_reachable_point_and_keeps_the_constraints() -> None:
    leg = leg_character()
    goal = np.array([1.0, -2.0])  # dentro del espacio alcanzable del pie
    state, err = solve_pose(leg, [(LEG_MARKER, goal)])
    assert err < POSE_TOL
    assert _world(leg, state, LEG_MARKER) == pytest.approx(goal, abs=1e-6)
    assert np.abs(leg.build()[0].residual(state, 0.0)).max() < POSE_TOL


def test_solve_pose_from_rest_reaches_every_reachable_attach_point() -> None:
    """Los puntos de actuación que recorre la pierna al seguir elipses alcanzables, algunos a más
    de 1 de su posición de reposo, se resuelven directo desde el reposo (sin continuación)."""
    leg = leg_character()
    rest_attach = _world(leg, leg.build()[1], LEG_ATTACH)
    far = 0.0
    for center, radii in (((0.9, -2.1), (0.3, 0.12)), ((1.0, -2.0), (0.35, 0.15))):
        states, _ = track_marker(leg, LEG_MARKER, _ellipse(*center, *radii))
        for goal in point_curve(leg, LEG_ATTACH, states):
            state, err = solve_pose(leg, [(LEG_ATTACH, goal)])
            assert err < POSE_TOL
            far = max(far, float(np.linalg.norm(goal - rest_attach)))
    assert far > 1.0


def test_solve_pose_unreachable_point_reports_the_error_and_stays_valid() -> None:
    leg = leg_character()
    state, err = solve_pose(leg, [(LEG_MARKER, np.array([0.9, -5.0]))])  # muy por debajo
    assert err > 0.1
    assert np.abs(leg.build()[0].residual(state, 0.0)).max() < 1e-4  # la pose sigue siendo válida


def test_solve_pose_with_two_points_on_the_biped() -> None:
    biped = biped_character()
    rest = biped.build()[1]
    left = PointRef("l_" + LEG_MARKER.part, LEG_MARKER.point)
    right = PointRef("r_" + LEG_MARKER.part, LEG_MARKER.point)
    goals = [_world(biped, rest, left) + [0.05, 0.1], _world(biped, rest, right) + [-0.05, 0.1]]
    state, err = solve_pose(biped, [(left, goals[0]), (right, goals[1])])
    assert err < POSE_TOL
    assert _world(biped, state, left) == pytest.approx(goals[0], abs=1e-6)
    assert _world(biped, state, right) == pytest.approx(goals[1], abs=1e-6)


def test_track_marker_follows_a_reachable_curve() -> None:
    leg = leg_character()
    curve = _ellipse(0.9, -2.1, 0.3, 0.12)
    states, worst = track_marker(leg, LEG_MARKER, curve)
    assert states.shape == (len(curve), leg.build()[1].size)
    assert worst < 1e-5
    assert np.abs(point_curve(leg, LEG_MARKER, states) - curve).max() < 1e-5
    asm = leg.build()[0]
    assert max(np.abs(asm.residual(s, 0.0)).max() for s in states) < POSE_TOL


def test_track_marker_reports_an_unreachable_curve() -> None:
    """La elipse de la escena de la pierna cae casi toda fuera del alcance del pie."""
    leg = leg_character()
    states, worst = track_marker(leg, LEG_MARKER, _ellipse(0.9, -2.4, 0.55, 0.25))
    assert worst > 0.1
    asm = leg.build()[0]
    assert max(np.abs(asm.residual(s, 0.0)).max() for s in states) < 1e-4  # poses válidas igual


def test_the_attach_curve_is_not_the_marker_curve_translated() -> None:
    """Por eso trasladar la curva del marcador era un mal punto de partida: el punto de actuación
    recorre otra forma y otro tamaño."""
    leg = leg_character()
    curve = _ellipse(0.9, -2.1, 0.3, 0.12)
    states, _ = track_marker(leg, LEG_MARKER, curve)
    attach = point_curve(leg, LEG_ATTACH, states)
    assert np.ptp(attach[:, 0]) > 1.5 * np.ptp(curve[:, 0])  # 1,04 contra 0,60
