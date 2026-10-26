"""Cambios del usuario sobre las piezas (``z`` y piezas fijas) del panel de propiedades."""

import dataclasses
import json
import math

import pytest

from mechchar.core.actuation import advise
from mechchar.core.character import (
    LEG_ATTACH,
    LEG_MARKER,
    PartOverride,
    apply_overrides,
    biped_character,
    leg_character,
    merge_override,
)
from mechchar.core.project import History, ProjectState, load_project, save_project


def _state(**kw) -> ProjectState:
    return ProjectState(1, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), **kw)


def _part(ch, name):
    return ch.parts[ch.index(name)]


# --- aplicar los cambios ------------------------------------------------------------------------


def test_apply_overrides_changes_z_and_fixed_and_leaves_the_original_alone():
    base = leg_character()
    out = apply_overrides(base, [PartOverride("shank", z=2.5), PartOverride("rod", fixed=True)])
    assert _part(out, "shank").z == 2.5 and not _part(out, "shank").fixed
    assert _part(out, "rod").fixed and _part(out, "rod").z == 0.0
    assert _part(base, "shank").z == 0.0 and not _part(base, "rod").fixed  # no se tocó
    assert _part(out, "thigh") == _part(base, "thigh")
    assert out.pivots == base.pivots and out.name == base.name


def test_none_keeps_what_the_character_brings():
    out = apply_overrides(leg_character(), [PartOverride("pelvis", z=3.0)])
    assert _part(out, "pelvis").fixed  # fixed=None: sigue siendo del chasis
    assert _part(out, "pelvis").z == 3.0


def test_unknown_part_and_duplicates_are_rejected():
    with pytest.raises(ValueError, match="no existe"):
        apply_overrides(leg_character(), [PartOverride("cola", z=1.0)])
    with pytest.raises(ValueError, match="dos cambios"):
        apply_overrides(leg_character(), [PartOverride("foot", z=1.0), PartOverride("foot", z=2.0)])


def test_override_validation():
    with pytest.raises(ValueError):
        PartOverride("", z=1.0)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finito"):
            PartOverride("foot", z=bad)
    assert PartOverride("foot").is_noop and not PartOverride("foot", fixed=False).is_noop


def test_merge_override_replaces_per_part_and_drops_noops():
    a = merge_override((), "foot", z=1.0)
    b = merge_override(a, "foot", fixed=True)
    assert b == (PartOverride("foot", 1.0, True),)
    c = merge_override(b, "shank", z=2.0)
    assert {ov.name for ov in c} == {"foot", "shank"}
    # volver un valor a «el del personaje» no es posible con None; sí quitar la pieza entera
    assert merge_override((PartOverride("foot"),), "foot") == ()


# --- efecto sobre la simulación ----------------------------------------------------------------


def test_fixing_a_part_removes_degrees_of_freedom():
    free = leg_character().analyze().dof
    fixed = leg_character([PartOverride("thigh", fixed=True)]).analyze().dof
    assert free == 2 and fixed < free


def test_unfixing_the_pelvis_frees_the_whole_character():
    assert leg_character([PartOverride("pelvis", fixed=False)]).analyze().dof > 2


def test_advice_rejects_an_actuation_point_on_a_part_made_fixed():
    ok = advise(leg_character(), LEG_ATTACH, LEG_MARKER)
    assert ok.ok
    fixed = leg_character([PartOverride(LEG_ATTACH.part, fixed=True)])
    assert not advise(fixed, LEG_ATTACH, LEG_MARKER).ok


def test_z_edit_does_not_change_the_2d_simulation():
    """La simulación es 2D por plano: ``z`` se guarda y se muestra, pero no mueve la pierna."""
    a, b = leg_character(), leg_character([PartOverride("shank", z=4.0)])
    assert a.analyze().dof == b.analyze().dof
    assert a.build()[1].tolist() == b.build()[1].tolist()


# --- bípedo -------------------------------------------------------------------------------------


def test_biped_applies_fixed_to_both_legs_and_shifts_z_per_leg():
    out = biped_character([PartOverride("foot", z=2.0, fixed=True)])
    assert _part(out, "l_foot").fixed and _part(out, "r_foot").fixed
    assert _part(out, "l_foot").z == 2.0 and _part(out, "r_foot").z == 3.0  # 0+2 y 1+2
    assert _part(out, "l_shank").z == 0.0 and _part(out, "r_shank").z == 1.0  # el resto igual


def test_biped_pelvis_override_is_applied_once():
    out = biped_character([PartOverride("pelvis", z=1.5)])
    assert _part(out, "pelvis").z == 1.5
    assert sum(p.name == "pelvis" for p in out.parts) == 1


def test_biped_without_overrides_is_unchanged():
    assert biped_character(()).parts == biped_character().parts


# --- proyecto -----------------------------------------------------------------------------------


def test_project_roundtrip_keeps_the_part_changes(tmp_path):
    parts = (PartOverride("shank", z=2.0), PartOverride("pelvis", fixed=False))
    st = _state(parts=parts)
    save_project(tmp_path / "p.json", st)
    assert load_project(tmp_path / "p.json") == st
    assert json.loads((tmp_path / "p.json").read_text())["parts"][0]["name"] == "shank"


def test_old_projects_without_parts_still_open():
    raw = _state().to_dict()
    del raw["parts"]
    assert ProjectState.from_dict(raw).parts == ()
    raw["parts"] = None
    assert ProjectState.from_dict(raw).parts == ()


@pytest.mark.parametrize(
    "bad",
    [
        [{"name": "foot", "z": float("nan")}],
        [{"z": 1.0}],
        ["foot"],
        [{"name": "foot", "z": 1.0}, {"name": "foot", "fixed": True}],
    ],
)
def test_bad_part_changes_are_value_errors(bad):
    raw = _state().to_dict()
    raw["parts"] = bad
    with pytest.raises(ValueError):
        ProjectState.from_dict(raw)


def test_part_changes_are_undoable_in_the_history():
    h: History[ProjectState] = History(_state())
    assert h.commit(_state(parts=(PartOverride("foot", z=1.0),)))
    assert not h.commit(_state(parts=(PartOverride("foot", z=1.0),)))  # sin cambio real
    assert h.undo() == _state()
    assert h.redo().parts == (PartOverride("foot", z=1.0),)


def test_override_is_immutable():
    ov = PartOverride("foot", z=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ov.z = 2.0  # type: ignore[misc]
