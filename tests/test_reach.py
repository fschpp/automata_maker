"""Informe de alcance y curva por defecto de la pierna (bundle 4 de la mejora de la pierna)."""

import math

import numpy as np
import pytest

from mechchar.core.character import (
    LEG_ATTACH,
    LEG_DEFAULT_ELLIPSE,
    LEG_MARKER,
    leg_character,
)
from mechchar.core.kinematics import reach_report, track_marker
from mechchar.core.spline import closed_catmull_rom
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.curves import resample_closed
from mechchar.optim.families import FOUR_BAR
from mechchar.optim.fitcore import fit
from mechchar.ui.spline_editor import default_points


def ellipse(cx: float, cy: float, a: float, b: float, n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def _default_curve() -> np.ndarray:
    cx, cy, a, b = LEG_DEFAULT_ELLIPSE
    return closed_catmull_rom(np.array(default_points(8, a, b, (cx, cy))), 24)


def test_reach_report_of_a_reachable_curve_is_ok() -> None:
    rep = reach_report(leg_character(), LEG_MARKER, ellipse(1.0, -2.0, 0.35, 0.15))
    assert rep.ok and rep.n_outside == 0 and rep.n_points == 60 and rep.worst < 1e-5


def test_reach_report_counts_the_points_of_the_old_default_that_fall_outside() -> None:
    """La elipse por defecto anterior: 53 de 120 puntos fuera, al peor le faltan 0,22."""
    rep = reach_report(leg_character(), LEG_MARKER, ellipse(0.9, -2.4, 0.55, 0.25, 120))
    assert not rep.ok
    assert 0.3 < rep.n_outside / rep.n_points < 0.6
    assert rep.worst == pytest.approx(0.2235, abs=0.01)


def test_reach_report_agrees_with_track_marker() -> None:
    leg, y = leg_character(), ellipse(0.9, -2.4, 0.55, 0.25, 40)
    assert reach_report(leg, LEG_MARKER, y).worst == pytest.approx(
        track_marker(leg, LEG_MARKER, y)[1]
    )


def test_the_default_curve_of_the_leg_scene_is_reachable() -> None:
    """Lo que ve el usuario al abrir la escena: el spline por los 8 puntos, no la elipse ideal."""
    rep = reach_report(leg_character(), LEG_MARKER, resample_closed(_default_curve(), 32))
    assert rep.ok, rep


def test_the_default_curve_can_be_fitted_below_the_acceptance_threshold() -> None:
    """Antes la escena por defecto quedaba en ~8 %; con la curva alcanzable baja de 3 %."""
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    res = fit(_default_curve(), [fam], n_refine=2)
    assert res is not None and res.rms_rel < 0.03
