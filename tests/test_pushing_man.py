"""El personaje Pushing Man y las especificaciones de personaje (CharacterSpec)."""

import math

import numpy as np
import pytest

from mechchar.core.actuation import (
    _dof_basis,
    _mobility_matrix,
    advise,
    pick_part,
    point_mobility,
)
from mechchar.core.character import (
    LEG_SPEC,
    PUSHING_MAN_ATTACH,
    PUSHING_MAN_DEFAULT_ELLIPSE,
    PUSHING_MAN_MARKER,
    PUSHING_MAN_SPEC,
    PartOverride,
    pushing_man_character,
)
from mechchar.core.kinematics import POSE_TOL, solve_pose, track_marker
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.families import ALL_FAMILIES, fit_best

SPECS = (LEG_SPEC, PUSHING_MAN_SPEC)
T = np.linspace(0, 2 * math.pi, 120, endpoint=False)


def _ellipse(cx: float, cy: float, a: float, b: float) -> np.ndarray:
    return np.column_stack([cx + a * np.cos(T), cy + b * np.sin(T)])


# --- el personaje ------------------------------------------------------------------------------


def test_the_character_has_two_degrees_of_freedom_and_no_redundancy() -> None:
    ch = pushing_man_character()
    analysis = ch.analyze()
    assert (analysis.dof, analysis.redundant) == (2, 0)
    assert [p.name for p in ch.parts if p.fixed] == ["base"]


def test_the_rest_pose_satisfies_every_constraint() -> None:
    asm, s0 = pushing_man_character().build()
    assert np.abs(asm.residual(s0, 0.0)).max() < 1e-12


def test_default_attach_and_marker_are_usable_and_on_their_parts() -> None:
    ch = pushing_man_character()
    advice = advise(ch, PUSHING_MAN_ATTACH, PUSHING_MAN_MARKER)
    assert advice.ok and (advice.attach_rank, advice.marker_rank) == (2, 2)
    assert pick_part(ch, PUSHING_MAN_ATTACH.point, 1e-9) == "forearm"
    assert pick_part(ch, PUSHING_MAN_MARKER.point, 1e-9) == "forearm"


def test_a_point_on_the_torso_is_not_a_valid_actuation_point() -> None:
    """El torso solo gira alrededor de la cadera fija: un pasador lo sobrerrestringiría."""
    ch = pushing_man_character()
    mid_torso = ch.parts[ch.index("torso")]
    point = (mid_torso.x + 0.15, mid_torso.y + 0.75)
    assert point_mobility(ch, type(PUSHING_MAN_ATTACH)("torso", point)) == 1


def test_overrides_change_the_character_through_its_factory() -> None:
    assert pushing_man_character([PartOverride("base", fixed=False)]).analyze().dof == 5
    assert pushing_man_character([PartOverride("forearm", fixed=True)]).analyze().dof == 0
    z = pushing_man_character([PartOverride("arm", z=2.0)])
    assert z.parts[z.index("arm")].z == 2.0


def test_the_rest_hand_is_not_stuck_in_a_corner_of_its_reachable_region() -> None:
    """Con el antebrazo casi horizontal la mano en reposo no podía moverse hacia adelante ni hacia
    arriba (segundo valor singular 0,07). Inclinado, tiene margen en todas las direcciones."""
    ch = pushing_man_character()
    rest = ch.build()[1]
    hand = np.array(PUSHING_MAN_MARKER.point)
    for direction, margin in (((1, 0), 0.15), ((-1, 0), 0.3), ((0, 1), 0.3), ((0, -1), 0.3)):
        _, err = solve_pose(ch, [(PUSHING_MAN_MARKER, hand + margin * np.array(direction))], rest)
        assert err < POSE_TOL, f"la mano no llega a {margin} en {direction}"
    basis = _dof_basis(ch)
    sv = np.linalg.svd(_mobility_matrix(ch, PUSHING_MAN_MARKER, basis), compute_uv=False)
    assert sv[-1] > 0.2


def test_the_hand_cannot_go_arbitrarily_far() -> None:
    """Es un personaje con un espacio alcanzable acotado (y por eso hay aviso de alcance)."""
    ch = pushing_man_character()
    far = np.array(PUSHING_MAN_MARKER.point) + np.array([2.0, 0.0])
    _, err = solve_pose(ch, [(PUSHING_MAN_MARKER, far)], ch.build()[1])
    assert err > 0.5


# --- especificaciones --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_each_spec_is_consistent(spec) -> None:
    ch = spec.factory(())
    assert ch.analyze().dof == 2
    assert pick_part(ch, spec.attach.point, 1e-9) == spec.attach.part
    assert pick_part(ch, spec.marker.point, 1e-9) == spec.marker.part
    assert advise(ch, spec.attach, spec.marker).ok
    x0, y0, x1, y1 = spec.view
    assert x0 < x1 and y0 < y1
    cx, cy, a, b = spec.ellipse
    assert x0 < cx - a and cx + a < x1 and y0 < cy - b and cy + b < y1  # la curva se ve entera
    assert spec.who and spec.reach_zone and spec.intro


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_the_default_curve_of_each_spec_is_reachable(spec) -> None:
    ch = spec.factory(())
    _, worst = track_marker(ch, spec.marker, _ellipse(*spec.ellipse))
    assert worst < 1e-5


def test_the_specs_have_distinct_keys_and_characters() -> None:
    assert len({s.key for s in SPECS}) == len(SPECS)
    assert LEG_SPEC.factory(()).name != PUSHING_MAN_SPEC.factory(()).name


def test_the_default_ellipse_constant_matches_the_spec() -> None:
    assert PUSHING_MAN_SPEC.ellipse == PUSHING_MAN_DEFAULT_ELLIPSE


# --- ajuste con el pipeline real ---------------------------------------------------------------


def test_the_default_curve_fits_below_three_percent() -> None:
    ch = pushing_man_character()
    fams = [CharacterFamily(b, ch, PUSHING_MAN_ATTACH, PUSHING_MAN_MARKER) for b in ALL_FAMILIES]
    result = fit_best(_ellipse(*PUSHING_MAN_DEFAULT_ELLIPSE), tuple(fams))  # type: ignore[arg-type]
    assert result is not None
    assert result.rms_rel < 0.03, f"{result.rms_rel:.4f}"
