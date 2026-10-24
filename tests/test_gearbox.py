import math

import numpy as np
import pytest

from mechchar.core.character import LEG_ATTACH, LEG_MARKER, leg_character
from mechchar.db.motion_db import MotionDB, build_database, make_candidate_source, suggest
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.curves import bbox_diag, resample_closed
from mechchar.optim.families import FOUR_BAR, fit_best
from mechchar.optim.fitcore import Variant
from mechchar.optim.gearbox import (
    BOX_RESIDUALS,
    BoxedFamily,
    GearBox,
    box_families,
    filter_candidates,
    fits,
    mechanism_bounds,
)

T = np.linspace(0, 2 * math.pi, 120, endpoint=False)
TARGET = np.column_stack([6 * np.cos(T), 3 * np.sin(T)])
SIZE = bbox_diag(TARGET)


def around(pad: float) -> GearBox:
    """Caja alrededor de la curva objetivo con un margen de ``pad`` tamaños en cada lado."""
    return GearBox(-6 - pad * SIZE, -3 - pad * SIZE, 6 + pad * SIZE, 3 + pad * SIZE)


@pytest.fixture(scope="module")
def pool():
    y = resample_closed(TARGET, 48)
    return FOUR_BAR.candidates(y, 48, 16000, 24, np.random.default_rng(0))


@pytest.fixture(scope="module")
def db4() -> MotionDB:
    return build_database("four_bar", n_target=500, n_samples=16_000, workers=1)


# --- la caja ---------------------------------------------------------------------------------


def test_box_from_corners_in_any_order_and_overshoot():
    box = GearBox.from_corners((4.0, 3.0), (-1.0, -2.0))
    assert (box.x0, box.y0, box.x1, box.y1) == (-1.0, -2.0, 4.0, 3.0)
    assert (box.width, box.height) == (5.0, 5.0)
    assert box.overshoot((0.0, 0.0, 1.0, 1.0)) == (0.0, 0.0, 0.0, 0.0)
    # se sale 0,5 por la izquierda, 2 por la derecha, 0 por abajo y 1 por arriba
    assert box.overshoot((-1.5, -2.0, 6.0, 4.0)) == (0.5, 2.0, 0.0, 1.0)
    assert box.worst_overshoot((-1.5, -2.0, 6.0, 4.0)) == 2.0
    assert box.contains((-1.0, -2.0, 4.0, 3.0)) and not box.contains((-1.1, 0.0, 0.0, 0.0))
    assert box.slack == pytest.approx(0.005)  # 0,1 % del lado mayor
    assert box.contains((-1.004, 0.0, 0.0, 0.0), box.slack)


# --- extensión del mecanismo -----------------------------------------------------------------


def test_bounds_match_a_dense_sweep_of_the_cycle(pool):
    c = pool[0]
    b = mechanism_bounds(FOUR_BAR, c.p0, c.variant)
    dense = mechanism_bounds(FOUR_BAR, c.p0, c.variant, n=240)
    assert b is not None and dense is not None
    assert np.allclose(b, dense, atol=0.01 * SIZE)  # 24 fases bastan para decidir si cabe
    assert dense[0] <= b[0] + 1e-9 and dense[2] >= b[2] - 1e-9  # más fases solo agrandan


def test_a_broken_mechanism_has_no_bounds():
    assert mechanism_bounds(FOUR_BAR, np.full(FOUR_BAR.n_params, -1.0), Variant(1)) is None


def test_filter_candidates_keeps_only_those_that_fit(pool):
    huge = GearBox(-1e3, -1e3, 1e3, 1e3)
    tiny = GearBox(-0.1, -0.1, 0.1, 0.1)
    assert filter_candidates(FOUR_BAR, pool, huge) == list(pool)
    assert filter_candidates(FOUR_BAR, pool, tiny) == []
    # una caja a medida del 3.º candidato deja pasar a los que caben y a ninguno que se salga
    b = mechanism_bounds(FOUR_BAR, pool[2].p0, pool[2].variant)
    box = GearBox(b[0] - 1e-6, b[1] - 1e-6, b[2] + 1e-6, b[3] + 1e-6)
    kept = filter_candidates(FOUR_BAR, pool, box)
    assert pool[2] in kept and all(fits(FOUR_BAR, c.p0, c.variant, box) for c in kept)
    assert len(kept) < len(pool)


# --- familia restringida a una caja ------------------------------------------------------------


def test_boxed_family_keeps_the_protocol_and_the_name(pool):
    box = around(1.2)
    bf = BoxedFamily(FOUR_BAR, box)
    assert bf.name == FOUR_BAR.name  # la base de movimientos sigue aplicándose
    assert bf.n_params == FOUR_BAR.n_params
    assert bf.n_compact == FOUR_BAR.n_compact + BOX_RESIDUALS
    c = pool[0]
    r, j = bf.compactness(c.p0, SIZE, c.variant)
    assert r.shape == (bf.n_compact,) and j.shape == (bf.n_compact, bf.n_params)
    assert box_families((FOUR_BAR,), None) == (FOUR_BAR,)
    assert isinstance(box_families((FOUR_BAR,), box)[0], BoxedFamily)


def test_box_residuals_are_zero_inside_and_positive_outside_with_a_correct_jacobian(pool):
    c = pool[0]
    b = mechanism_bounds(FOUR_BAR, c.p0, c.variant)
    inside = BoxedFamily(FOUR_BAR, GearBox(b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1))
    r_in, _ = inside.compactness(c.p0, SIZE, c.variant)
    assert np.all(r_in[-BOX_RESIDUALS:] == 0.0)
    # una caja que recorta 1 unidad por la derecha: ese residuo vale 2 (peso) y los demás 0
    cut = BoxedFamily(FOUR_BAR, GearBox(b[0] - 1, b[1] - 1, b[2] - 1.0, b[3] + 1))
    r, j = cut.compactness(c.p0, SIZE, c.variant)
    assert r[-BOX_RESIDUALS:] == pytest.approx([0.0, 2.0, 0.0, 0.0], abs=0.05)
    assert np.any(j[-BOX_RESIDUALS + 1] != 0.0)  # el residuo derecho depende de los parámetros
    # el Jacobiano coincide con diferencias finitas hechas con otro paso
    k = int(np.argmax(np.abs(j[-BOX_RESIDUALS + 1])))
    q = c.p0.copy()
    q[k] += 1e-4
    r2, _ = cut.compactness(q, SIZE, c.variant)
    assert (r2[-BOX_RESIDUALS + 1] - r[-BOX_RESIDUALS + 1]) / 1e-4 == pytest.approx(
        j[-BOX_RESIDUALS + 1, k], rel=0.05, abs=1e-3
    )


def test_boxed_candidates_all_fit(pool):
    box = around(1.2)
    bf = BoxedFamily(FOUR_BAR, box)
    cands = bf.candidates(resample_closed(TARGET, 48), 48, 16000, 2, np.random.default_rng(0))
    assert cands and all(fits(FOUR_BAR, c.p0, c.variant, box) for c in cands)


def test_fit_stays_inside_the_box_while_the_free_fit_does_not():
    box = around(1.2)
    free = fit_best(TARGET, (FOUR_BAR,))
    assert free is not None
    b_free = mechanism_bounds(FOUR_BAR, free.params, free.variant)
    assert box.worst_overshoot(b_free) > 0.05 * SIZE  # sin caja el mecanismo se sale
    boxed = fit_best(TARGET, box_families((FOUR_BAR,), box))
    assert boxed is not None
    b = mechanism_bounds(FOUR_BAR, boxed.params, boxed.variant)
    assert box.worst_overshoot(b) < 0.01 * SIZE
    assert boxed.rms_rel < 0.03  # y la curva sigue ajustando bien


def test_a_box_that_is_too_small_gives_no_mechanism():
    assert fit_best(TARGET, box_families((FOUR_BAR,), around(0.5))) is None


# --- base de movimientos ---------------------------------------------------------------------


def test_suggest_discards_candidates_outside_the_box_and_reports_bounds(db4):
    curve = resample_closed(TARGET, 64)
    free = suggest([db4], curve)
    assert len(free) == 3 and all(s.bounds is not None for s in free)
    b = free[0].bounds
    box = GearBox(b[0] - 1e-6, b[1] - 1e-6, b[2] + 1e-6, b[3] + 1e-6)
    boxed = suggest([db4], curve, box=box)
    assert boxed and all(box.contains(s.bounds) for s in boxed)
    assert boxed[0].distance == pytest.approx(free[0].distance)  # el mejor ya cabía
    assert suggest([db4], curve, box=GearBox(-0.1, -0.1, 0.1, 0.1)) == []


def test_candidate_source_discards_those_that_do_not_fit(db4):
    source = make_candidate_source([db4])
    y = resample_closed(TARGET, 48)
    assert source(FOUR_BAR, y, 2)  # sin caja, como siempre
    assert source(BoxedFamily(FOUR_BAR, GearBox(-0.1, -0.1, 0.1, 0.1)), y, 2) == []
    big = source(BoxedFamily(FOUR_BAR, GearBox(-1e3, -1e3, 1e3, 1e3)), y, 2)
    assert [c.rel_error for c in big] == [c.rel_error for c in source(FOUR_BAR, y, 2)]


# --- con personaje ---------------------------------------------------------------------------


def test_bounds_with_a_character_use_the_mechanism_simulated_with_the_leg():
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    target = np.column_stack([0.9 + 0.55 * np.cos(T), -2.4 + 0.25 * np.sin(T)])
    cand = fam.candidates(target, 40, 16000, 1, np.random.default_rng(0))[0]
    b = mechanism_bounds(fam, cand.p0, cand.variant)
    assert b is not None
    # el punto de actuación (donde el mecanismo se une a la pantorrilla) está dentro
    x, y = LEG_ATTACH.point
    assert b[0] - 1e-6 <= x <= b[2] + 1e-6 and b[1] - 1e-6 <= y <= b[3] + 1e-6
    assert fits(fam, cand.p0, cand.variant, GearBox(-50, -50, 50, 50))
    assert not fits(fam, cand.p0, cand.variant, GearBox(b[0], b[1], b[0] + 0.01, b[3]))
    boxed = BoxedFamily(fam, GearBox(-50, -50, 50, 50))
    assert boxed.character is fam.character  # la extensión se mide con el personaje
    assert [c.rel_error for c in boxed.candidates(target, 40, 16000, 1, np.random.default_rng(0))]
