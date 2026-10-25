import math

import numpy as np
import pytest

from mechchar.db.metric import N_NORM, normalize, shape_distance
from mechchar.db.motion_db import (
    MotionDB,
    build_database,
    make_candidate_source,
    poisson_disk,
    poisson_disk_select,
    query,
    suggest,
    suggest_many,
)
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.families import FOUR_BAR, GEARED_FIVE_BAR
from mechchar.optim.fitcore import _builder, fit, trace_curve_generic

from .test_metric import BEAN, ELLIPSE, similar


@pytest.fixture(scope="module")
def db4() -> MotionDB:
    return build_database("four_bar", n_target=500, n_samples=16_000, workers=1)


def entry_curve(db: MotionDB, i: int) -> np.ndarray:
    return db.center[i] + db.scale[i] * db.pts[i]


# --- Poisson-disk -------------------------------------------------------------------------------


def test_poisson_disk_keeps_a_minimum_separation():
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(2000, 3))
    chosen = poisson_disk(pts, 0.15, rng.permutation(len(pts)))
    sel = pts[chosen]
    d = np.linalg.norm(sel[:, None] - sel[None], axis=-1) + np.eye(len(sel)) * 9
    assert d.min() >= 0.15
    assert len(chosen) > 10


def test_poisson_disk_select_hits_the_target_count_with_separation():
    rng = np.random.default_rng(1)
    pts = rng.uniform(size=(3000, 4))
    sel = poisson_disk_select(pts, 200, np.random.default_rng(2))
    assert len(sel) == 200
    assert len(poisson_disk_select(pts[:50], 200, np.random.default_rng(2))) == 50
    s = pts[sel]
    d = np.linalg.norm(s[:, None] - s[None], axis=-1) + np.eye(len(s)) * 9
    random_pick = pts[np.random.default_rng(3).choice(3000, 200, replace=False)]
    dr = np.linalg.norm(random_pick[:, None] - random_pick[None], axis=-1) + np.eye(200) * 9
    assert d.min() > dr.min()  # más separadas que una selección al azar


# --- base ---------------------------------------------------------------------------------------


def test_database_entries_are_feasible_mechanisms(db4):
    assert len(db4) == 500 and db4.pts.shape == (500, N_NORM, 2)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(db4), 12, replace=False):
        p0 = FOUR_BAR.assemble(db4.shape[i], db4.variant(i), 1.0, 0.0, np.zeros(2))
        res = _builder(FOUR_BAR, db4.variant(i))
        assert trace_curve_generic(res, p0, 60) is not None  # completa el ciclo en el simulador


def test_database_save_and_load_roundtrip(db4, tmp_path):
    db4.save(tmp_path / "db.npz")
    other = MotionDB.load(tmp_path / "db.npz")
    assert other.family == "four_bar" and len(other) == len(db4)
    assert np.array_equal(other.pts, db4.pts) and np.array_equal(other.branch, db4.branch)
    q = entry_curve(db4, 7)
    assert query(other, q, k=1)[0].index == query(db4, q, k=1)[0].index


# --- consulta -----------------------------------------------------------------------------------


def transformed_entry(db: MotionDB, rng: np.random.Generator, reverse: bool = False):
    i = int(rng.integers(len(db)))
    q = similar(
        entry_curve(db, i), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 4.0),
        rng.normal(size=2) * 5,
    )
    q = np.roll(q, int(rng.integers(len(q))), axis=0)
    return i, (q[::-1].copy() if reverse else q)


@pytest.mark.parametrize("seed", range(6))
def test_query_is_at_least_as_good_as_the_true_entry(db4, seed):
    """La entrada de origen está en la base, así que el mejor resultado no puede ser peor que ella
    (aunque no siempre es la misma: hay mecanismos de curva casi idéntica)."""
    rng = np.random.default_rng(seed)
    i, q = transformed_entry(db4, rng, reverse=bool(seed % 2))
    d_true = shape_distance(normalize(q).pts, db4.pts[i])
    top = query(db4, q, k=1, max_ratio=None)[0]
    assert top.shape_distance <= d_true + 1e-9


def test_transformed_entries_are_close_in_the_metric_with_a_resampling_floor(db4):
    """Tras remuestrear, la distancia a la propia entrada es baja (mediana); en curvas con cúspides
    el término de tangentes la eleva, por eso no se exige un máximo."""
    rng = np.random.default_rng(42)
    ds = []
    for _ in range(30):
        i, q = transformed_entry(db4, rng, reverse=bool(rng.integers(2)))
        ds.append(shape_distance(normalize(q).pts, db4.pts[i]))
    assert np.median(ds) < 0.02


def test_match_transform_reproduces_the_query_with_the_real_mechanism(db4):
    """Extremo a extremo: con (escala, rotación, origen) del match se arma el mecanismo en el
    simulador y su curva coincide con la del match."""
    rng = np.random.default_rng(5)
    for _ in range(3):
        i = int(rng.integers(len(db4)))
        q = similar(entry_curve(db4, i), rng.uniform(0, 6.28), 2.5, rng.normal(size=2) * 3)
        m = query(db4, q, k=1, max_ratio=None)[0]
        v = db4.variant(m.index)
        p0 = FOUR_BAR.assemble(db4.shape[m.index], v, m.scale, m.rotation, m.origin)
        traced = trace_curve_generic(_builder(FOUR_BAR, v), p0, 180)
        assert traced is not None
        # el mecanismo armado traza la curva del propio match, salvo el piso de discretizar la
        # curva guardada en 64 puntos (mediana 0,7 %, máximo 1,5 % del tamaño)...
        size = bbox_diag(q)
        assert chamfer_rms(traced, m.curve) < 0.02 * size
        # ...y el test es sensible: una rotación equivocada de solo 0,15 rad lo haría fallar
        p_bad = FOUR_BAR.assemble(db4.shape[m.index], v, m.scale, m.rotation + 0.15, m.origin)
        traced_bad = trace_curve_generic(_builder(FOUR_BAR, v), p_bad, 180)
        assert traced_bad is not None
        assert chamfer_rms(traced_bad, m.curve) > 0.02 * size
        # ...y esa curva se parece a la consultada (el mejor match puede ser otra entrada similar)
        assert chamfer_rms(m.curve, q) < 0.03 * bbox_diag(q)


def test_prescreen_agrees_with_exhaustive_search(db4):
    rng = np.random.default_rng(11)
    agree = 0
    trials = 12
    for _ in range(trials):
        i = int(rng.integers(len(db4)))
        q = entry_curve(db4, i) + rng.normal(scale=0.02, size=(N_NORM, 2)) * db4.scale[i]
        a = query(db4, q, k=1, max_ratio=None)[0]
        b = query(db4, q, k=1, max_ratio=None, exhaustive=True)[0]
        agree += a.index == b.index or abs(a.distance - b.distance) < 1e-3
    assert agree >= trials - 1


def test_query_respects_the_compactness_filter(db4):
    q = entry_curve(db4, 3)
    assert query(db4, q, k=3, max_ratio=1e-6) == []
    assert len(query(db4, q, k=3, max_ratio=None)) == 3


def test_query_is_fast(db4):
    import time

    q = entry_curve(db4, 0)
    query(db4, q)  # calentamiento
    t0 = time.perf_counter()
    for _ in range(5):
        query(db4, q)
    assert (time.perf_counter() - t0) / 5 < 0.1


# --- integración con el ajuste ------------------------------------------------------------------


@pytest.mark.parametrize(("name", "target"), [("elipse", ELLIPSE), ("frijol", BEAN)])
def test_fit_with_database_candidates_reaches_three_percent(db4, name, target):
    source = make_candidate_source([db4])
    result = fit(target, [FOUR_BAR], candidate_source=source)
    assert result is not None, name
    assert result.rms_rel < 0.03, f"{name}: {result.rms_rel:.4f}"


def test_candidate_source_ignores_families_without_a_database(db4):
    source = make_candidate_source([db4])
    assert source(GEARED_FIVE_BAR, ELLIPSE[:: 5], 2) == []
    assert len(source(FOUR_BAR, ELLIPSE[:: 5], 2)) > 0


def test_five_bar_database_builds():
    db = build_database("geared_five_bar", n_target=150, n_samples=16_000, workers=1)
    assert len(db) > 20
    assert set(np.unique(db.ratio)) <= {-2.0, -1.0, 1.0, 2.0}


def test_suggest_returns_ranked_candidates_that_build_real_mechanisms(db4):
    out = suggest([db4], BEAN, k=3)
    assert len(out) == 3
    assert [s.distance for s in out] == sorted(s.distance for s in out)
    for s in out:
        assert s.family == "four_bar" and s.rel_error < 0.1
        traced = trace_curve_generic(_builder(FOUR_BAR, s.variant), s.p0, 180)
        assert traced is not None
        assert chamfer_rms(traced, s.curve) < 0.02 * bbox_diag(BEAN)


def test_suggest_merges_several_databases():
    d4 = build_database("four_bar", n_target=200, n_samples=8_000, workers=1)
    d5 = build_database("geared_five_bar", n_target=200, n_samples=16_000, workers=1)
    out = suggest([d4, d5], ELLIPSE, k=3)
    assert 1 <= len(out) <= 3
    assert {s.family for s in out} <= {"four_bar", "geared_five_bar"}
    assert [s.distance for s in out] == sorted(s.distance for s in out)


def test_suggest_many_answers_each_curve_like_suggest_does(db4):
    curves = [entry_curve(db4, i) for i in (3, 40, 211)] + [ELLIPSE * 2.0]
    many = suggest_many([db4], curves, k=2)
    assert len(many) == len(curves)
    for curve, got in zip(curves, many, strict=True):
        want = suggest([db4], curve, k=2)
        assert [(g.family, g.distance) for g in got] == [(w.family, w.distance) for w in want]
    assert all(len(m) == 2 for m in many)


def test_suggest_many_with_no_curves_is_empty(db4):
    assert suggest_many([db4], []) == []
