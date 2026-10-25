import json
import math

import pytest

from mechchar.core.actuation import (
    advise,
    pick_part,
    pick_point,
    point_mobility,
    reach,
    suggest_attach_points,
)
from mechchar.core.character import (
    LEG_ATTACH,
    LEG_MARKER,
    Character,
    Part,
    PointRef,
    biped_character,
    leg_character,
    leg_refs,
)
from mechchar.core.project import PointPick, ProjectState, load_project, save_project

THIGH_MID = PointRef("thigh", (0.45, -0.6))  # el muslo gira alrededor de la cadera fija
PELVIS = PointRef("pelvis", (0.0, 0.0))


# --- movilidad -------------------------------------------------------------------------------


def test_mobility_of_the_test_leg_points() -> None:
    leg = leg_character()
    assert point_mobility(leg, LEG_ATTACH) == 2  # medio de la pantorrilla: se mueve en el plano
    assert point_mobility(leg, LEG_MARKER) == 2  # el pie
    assert point_mobility(leg, THIGH_MID) == 1  # solo recorre un arco
    assert point_mobility(leg, PELVIS) == 0  # pieza fija


def test_mobility_is_a_property_of_the_point_not_of_the_part_name() -> None:
    """Cada pierna del bípedo se comporta como la pierna sola (las dos son independientes)."""
    biped = biped_character()
    for prefix in ("l_", "r_"):
        attach, marker = leg_refs(prefix)
        assert point_mobility(biped, attach) == 2
        assert point_mobility(biped, marker) == 2
        assert point_mobility(biped, PointRef(prefix + "thigh", THIGH_MID.point)) == 1


def test_reach_is_between_zero_and_one_and_zero_when_undefined() -> None:
    leg = leg_character()
    r = reach(leg, LEG_ATTACH, LEG_MARKER)
    assert 0.0 < r <= 1.0
    assert reach(leg, THIGH_MID, LEG_MARKER) == 0.0  # el punto de actuación no se mueve en 2D
    assert reach(leg, LEG_MARKER, LEG_MARKER) == pytest.approx(1.0)  # mismo punto: identidad


# --- diagnóstico -----------------------------------------------------------------------------


def test_advice_accepts_the_default_leg_points() -> None:
    advice = advise(leg_character(), LEG_ATTACH, LEG_MARKER)
    assert advice.ok
    assert advice.dof == 2 and advice.attach_rank == 2 and advice.marker_rank == 2


def test_advice_rejects_a_point_that_a_pin_would_overconstrain() -> None:
    advice = advise(leg_character(), THIGH_MID, LEG_MARKER)
    assert not advice.ok
    assert advice.attach_rank == 1
    assert "sobrerrestringe" in advice.text


def test_advice_rejects_a_fixed_part() -> None:
    advice = advise(leg_character(), PELVIS, LEG_MARKER)
    assert not advice.ok
    assert "fija" in advice.text


def test_advice_rejects_a_marker_that_does_not_move() -> None:
    advice = advise(leg_character(), LEG_ATTACH, PELVIS)
    assert not advice.ok
    assert advice.marker_rank == 0
    assert "no se mueve" in advice.text


def test_advice_for_a_character_without_freedom() -> None:
    rigid = Character("rigido", [Part("base", 0, 0, 0, [(0, 0), (1, 0), (1, 1)], fixed=True)])
    advice = advise(rigid, PointRef("base", (0.5, 0.2)), PointRef("base", (1.0, 0.0)))
    assert not advice.ok and advice.dof == 0
    assert "grados de libertad" in advice.text


def test_advice_warns_but_allows_when_more_mechanisms_are_needed() -> None:
    biped = biped_character()  # 4 grados de libertad: un mecanismo solo no alcanza
    attach, marker = leg_refs("l_")
    advice = advise(biped, attach, marker)
    assert advice.ok and advice.dof == 4
    assert "otro mecanismo" in advice.text


# --- sugerencias -----------------------------------------------------------------------------


def test_suggestions_are_usable_sorted_and_one_per_part() -> None:
    leg = leg_character()
    found = suggest_attach_points(leg, LEG_MARKER, limit=10)
    assert found
    scores = [s for _, s in found]
    assert scores == sorted(scores, reverse=True)
    parts = [ref.part for ref, _ in found]
    assert len(parts) == len(set(parts)) and "pelvis" not in parts and "thigh" not in parts
    for ref, _ in found:
        assert point_mobility(leg, ref) == 2
        assert advise(leg, ref, LEG_MARKER).ok
        # el punto está realmente sobre su pieza
        assert pick_part(leg, ref.point, 1e-9) == ref.part


def test_suggestions_respect_the_limit_and_need_freedom() -> None:
    assert len(suggest_attach_points(leg_character(), LEG_MARKER, limit=2)) == 2
    rigid = Character("rigido", [Part("base", 0, 0, 0, [(0, 0), (1, 0), (1, 1)], fixed=True)])
    assert suggest_attach_points(rigid, PointRef("base", (0.5, 0.2))) == []


# --- elegir con el clic ----------------------------------------------------------------------


def test_pick_part_inside_outside_and_overlapping() -> None:
    leg = leg_character()
    assert pick_part(leg, LEG_ATTACH.point) == "shank"
    assert pick_part(leg, (5.0, 5.0), tol=0.1) is None
    assert pick_part(leg, (0.0, 0.0)) == "thigh"  # la cadera está en la pelvis y en el muslo


def test_pick_point_snaps_to_the_part_outline_within_tolerance() -> None:
    leg = leg_character()
    inside = pick_point(leg, LEG_ATTACH.point)
    assert inside is not None and inside.part == "shank"
    assert inside.point == pytest.approx(LEG_ATTACH.point)
    # un clic cerca pero fuera de la pantorrilla se pega a su contorno
    near = pick_point(leg, (0.45 + 0.4, -1.8), tol=0.5)
    assert near is not None
    assert pick_part(leg, near.point, 1e-9) == near.part
    assert pick_point(leg, (9.0, 9.0), tol=0.1) is None


# --- proyecto --------------------------------------------------------------------------------

SPLINE = tuple((math.cos(a), math.sin(a)) for a in (0.0, 2.0, 4.0))


def test_project_roundtrips_the_picked_points(tmp_path) -> None:
    state = ProjectState(
        1, SPLINE, attach=PointPick("shank", 0.45, -1.8), marker=PointPick("foot", 0.9, -2.5)
    )
    path = tmp_path / "p.json"
    save_project(path, state)
    assert load_project(path) == state


def test_old_projects_without_picked_points_still_open(tmp_path) -> None:
    path = tmp_path / "old.json"
    save_project(path, ProjectState(0, SPLINE))
    raw = json.loads(path.read_text())
    del raw["attach"], raw["marker"]
    path.write_text(json.dumps(raw))
    state = load_project(path)
    assert state.attach is None and state.marker is None


@pytest.mark.parametrize(
    "bad",
    [
        {"part": "foot", "point": [float("nan"), 0.0]},
        {"part": "foot", "point": [1.0]},
        {"part": "", "point": [0.0, 0.0]},
        {"point": [0.0, 0.0]},
    ],
)
def test_project_rejects_malformed_picked_points(tmp_path, bad) -> None:
    path = tmp_path / "bad.json"
    save_project(path, ProjectState(1, SPLINE))
    raw = json.loads(path.read_text())
    raw["marker"] = bad
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_project(path)
