import math

import numpy as np
import pytest

from mechchar.db.metric import (
    N_NORM,
    descriptor,
    features,
    frechet_closed,
    normalize,
    self_intersections,
    shape_distance,
    shape_distance_many,
    tangents,
    to_complex,
    xcorr,
)


def curve(fx, fy, m: int = 300) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return np.column_stack([fx(t), fy(t)])


ELLIPSE = curve(lambda t: 3 * np.cos(t), lambda t: 1.5 * np.sin(t))
BEAN = curve(lambda t: 2 * np.cos(t), lambda t: 1.2 * np.sin(t) + 0.5 * np.cos(2 * t))
EIGHT = curve(lambda t: np.cos(t), lambda t: 0.5 * np.sin(2 * t))
CIRCLE = curve(np.cos, np.sin)


def similar(c: np.ndarray, angle: float, scale: float, shift: np.ndarray) -> np.ndarray:
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return scale * c @ r.T + shift


def test_normalize_sets_centroid_and_rms_radius():
    nz = normalize(similar(ELLIPSE, 0.7, 2.5, np.array([4.0, -3.0])))
    assert nz.pts.shape == (N_NORM, 2)
    assert nz.pts.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert np.sqrt((nz.pts**2).sum(1).mean()) == pytest.approx(1.0)
    assert nz.center == pytest.approx([4.0, -3.0], abs=0.05)
    assert nz.scale > 0


def test_xcorr_fft_matches_bruteforce():
    rng = np.random.default_rng(0)
    n = 16
    a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b = rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))
    got = xcorr(a, b)
    for m in range(3):
        for s in range(n):
            expected = sum(a[k] * np.conj(b[m, (k + s) % n]) for k in range(n))
            assert got[m, s] == pytest.approx(expected)


def test_shape_distance_is_invariant_to_pose_start_and_direction():
    a = normalize(BEAN).pts
    base = shape_distance(a, a)
    assert base == pytest.approx(0.0, abs=1e-6)
    moved = normalize(similar(BEAN, 1.9, 0.4, np.array([7.0, 2.0]))).pts
    assert shape_distance(a, moved) < 0.02
    assert shape_distance(a, np.roll(a, 17, axis=0)) == pytest.approx(0.0, abs=1e-6)
    assert shape_distance(a, a[::-1].copy()) == pytest.approx(0.0, abs=1e-6)


def test_shape_distance_separates_different_shapes_and_returns_the_pose():
    bean, ellipse, circle = (normalize(c).pts for c in (BEAN, ELLIPSE, CIRCLE))
    assert shape_distance(bean, ellipse) > 0.1
    assert shape_distance(ellipse, circle) > 0.1
    ang = 0.9
    rotated = normalize(similar(ELLIPSE, ang, 1.0, np.zeros(2))).pts
    res = shape_distance_many(ellipse, rotated[None])
    # a ≈ e^{iα} b: el ángulo recuperado es el opuesto de la rotación aplicada (mod π: la elipse
    # es simétrica)
    assert math.sin(res.rotation[0] + ang) == pytest.approx(0.0, abs=1e-2)


def test_tangents_are_unit_and_perpendicular_for_a_circle():
    z = to_complex(normalize(CIRCLE).pts)
    t = tangents(z)
    assert np.abs(t) == pytest.approx(1.0)
    assert np.abs((t * np.conj(z)).real).max() < 0.1  # tangente ⟂ radio


def test_features_values_and_invariance():
    f_circle = features(normalize(CIRCLE).pts)
    assert f_circle[0] == pytest.approx(1.0, abs=1e-3)  # perímetro / 2π
    assert f_circle[1] == pytest.approx(1.0, abs=1e-2)  # área / π
    assert f_circle[2] == pytest.approx(1.0, abs=1e-2)  # elipticidad
    f_e = features(normalize(ELLIPSE).pts)
    # con muestreo uniforme en longitud de arco la razón no es b/a = 0,5 exacta, pero sí < 1
    assert 0.5 < f_e[2] < 0.7
    thin = features(normalize(curve(lambda t: 3 * np.cos(t), lambda t: 0.5 * np.sin(t))).pts)
    assert thin[2] < f_e[2] < f_circle[2]
    f_e2 = features(normalize(similar(ELLIPSE, 2.1, 3.0, np.array([1.0, 1.0]))).pts)
    assert f_e2 == pytest.approx(f_e, abs=0.02)


def test_self_intersections_counts_the_figure_eight_only():
    assert self_intersections(normalize(EIGHT).pts) == 1
    assert self_intersections(normalize(ELLIPSE).pts) == 0
    assert self_intersections(normalize(BEAN).pts) == 0
    assert features(normalize(EIGHT).pts)[5] == 1.0
    assert features(normalize(EIGHT).pts, with_intersections=False)[5] == 0.0


def test_descriptor_is_invariant_and_reversal_swaps_harmonics():
    nz = normalize(BEAN).pts
    f = features(nz)
    d = descriptor(nz, f)
    moved = normalize(similar(BEAN, 2.4, 1.7, np.array([-5.0, 3.0]))).pts
    assert descriptor(moved, features(moved)) == pytest.approx(d, abs=0.02)
    assert descriptor(np.roll(nz, 9, axis=0), f) == pytest.approx(d, abs=1e-9)
    rev = nz[::-1].copy()
    expected = descriptor(nz, f, reverse=True)
    assert descriptor(rev, features(rev)) == pytest.approx(expected, abs=0.02)


def test_frechet_baseline_ranks_like_the_shapes():
    bean = normalize(BEAN).pts
    bean_moved = normalize(similar(BEAN, 1.1, 2.0, np.zeros(2))).pts
    ellipse = normalize(ELLIPSE).pts
    same, _ = frechet_closed(bean, bean_moved)
    different, _ = frechet_closed(bean, ellipse)
    assert same < 0.15
    assert different > same
