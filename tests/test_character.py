import math

import numpy as np
import pytest

from mechchar.core.assembly import Assembly, Drive, FixedState, Phase, Pin, PointOnLine, world_point
from mechchar.core.character import (
    LEG_ATTACH,
    LEG_MARKER,
    Character,
    Part,
    Pivot,
    RiggedMechanism,
    leg_character,
)
from mechchar.core.playback import Playback
from mechchar.optim.character_fit import CharacterFamily
from mechchar.optim.curves import bbox_diag
from mechchar.optim.families import FOUR_BAR, GEARED_FIVE_BAR
from mechchar.optim.fitcore import Variant, fit, trace_curve_generic

TARGET_T = np.linspace(0, 2 * math.pi, 120, endpoint=False)


def ellipse(cx: float, cy: float, a: float, b: float) -> np.ndarray:
    return np.column_stack([cx + a * np.cos(TARGET_T), cy + b * np.sin(TARGET_T)])


# --- ensamblajes: fusión y reindexado -------------------------------------------------------------


def test_constraints_shift_every_component_index():
    assert Pin(0, (1.0, 2.0), 3, (0.0, 0.0)).shifted(5) == Pin(5, (1.0, 2.0), 8, (0.0, 0.0))
    assert FixedState(2, 1.0, 2.0, 0.5).shifted(4) == FixedState(6, 1.0, 2.0, 0.5)
    assert Drive(1, 2.0, 0.3).shifted(3) == Drive(4, 2.0, 0.3)
    assert Phase(0, 2, -1.0, 0.1).shifted(1) == Phase(1, 3, -1.0, 0.1)
    assert PointOnLine(1, (0.0, 1.0), 0, (0.0, 0.0), (1.0, 0.0)).shifted(2) == PointOnLine(
        3, (0.0, 1.0), 2, (0.0, 0.0), (1.0, 0.0)
    )


def test_assembly_append_offsets_components_and_keeps_the_motion():
    from mechchar.core.mechanisms import four_bar

    fb = four_bar(4.0, 1.0, 3.5, 3.0)
    host = Assembly()
    host.add_component("extra", 10.0, 10.0)
    offset = host.append(fb.asm)
    assert offset == 1 and len(host.components) == 1 + len(fb.asm.components)
    assert host.n_eq == fb.asm.n_eq
    s0 = np.concatenate([[10.0, 10.0, 0.0], fb.state0])
    cyc = host.trace_cycle(s0, 60)
    # la pieza suelta no tiene conexiones: queda quieta y el resto se mueve igual que solo
    assert cyc.ok
    alone = fb.asm.trace_cycle(fb.state0, 60)
    assert np.allclose(cyc.states[:, 3:], alone.states, atol=1e-8)


# --- personaje -----------------------------------------------------------------------------------


def test_leg_has_two_degrees_of_freedom_and_no_redundancy():
    a = leg_character().analyze()
    assert (a.dof, a.redundant) == (2, 0)


def test_rest_pose_satisfies_all_constraints_and_local_points_are_consistent():
    ch = leg_character()
    asm, s0 = ch.build()
    assert np.max(np.abs(asm.residual(s0, 0.0))) < 1e-12
    p = ch.index(LEG_MARKER.part)
    assert world_point(s0, p, ch.local_point(LEG_MARKER.part, LEG_MARKER.point)) == pytest.approx(
        LEG_MARKER.point
    )
    # la pelvis es el chasis: es la única pieza fija
    assert [pt.name for pt in ch.parts if pt.fixed] == ["pelvis"]


def test_character_json_roundtrip(tmp_path):
    ch = leg_character()
    ch.parts[1].z = 1.5
    ch.save(tmp_path / "pierna.json")
    other = Character.load(tmp_path / "pierna.json")
    assert other.to_dict() == ch.to_dict()
    assert other.parts[1].z == 1.5
    assert other.analyze().dof == ch.analyze().dof == 2


def test_character_validation():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    with pytest.raises(ValueError, match="mismo nombre"):
        Character("x", [Part("a", 0, 0, 0, sq), Part("a", 1, 0, 0, sq)]).validate()
    with pytest.raises(ValueError, match="inexistente"):
        Character("x", [Part("a", 0, 0, 0, sq)], [Pivot("a", "zzz", (0.0, 0.0))]).validate()
    with pytest.raises(ValueError, match="consigo"):
        Character("x", [Part("a", 0, 0, 0, sq)], [Pivot("a", "a", (0.0, 0.0))]).validate()
    with pytest.raises(KeyError):
        leg_character().index("no existe")
    bad = leg_character().to_dict()
    bad["version"] = 99
    with pytest.raises(ValueError, match="versión"):
        Character.from_dict(bad)


# --- mecanismo + personaje -----------------------------------------------------------------------


def test_combining_a_mechanism_removes_exactly_the_free_motion_of_the_leg():
    ch = leg_character()
    fam = CharacterFamily(FOUR_BAR, ch, LEG_ATTACH, LEG_MARKER)
    cand = fam.candidates(ellipse(0.9, -2.4, 0.55, 0.25), 40, 16000, 1, np.random.default_rng(0))[0]
    built = fam.build(cand.p0, cand.variant)
    assert built is not None
    a = built.asm.analyze(built.state0)
    assert (a.dof, a.redundant) == (0, 0)  # 2 de la pierna menos 2 del pasador, mecanismo con motor
    # el pasador es factible en reposo: el marcador del mecanismo cae sobre el punto de actuación
    assert np.max(np.abs(built.asm.residual(built.state0, 0.0))) < 1e-9


def test_character_family_candidates_are_feasible_full_cycles():
    ch = leg_character()
    for base in (FOUR_BAR, GEARED_FIVE_BAR):
        fam = CharacterFamily(base, ch, LEG_ATTACH, LEG_MARKER)
        target = ellipse(0.9, -2.4, 0.55, 0.25)
        cands = fam.candidates(target, 40, 16000, 3, np.random.default_rng(0))
        assert cands, base.name
        assert [c.rel_error for c in cands] == sorted(c.rel_error for c in cands)
        for c in cands:
            curve = trace_curve_generic(fam._builder(c.variant), c.p0, 60)
            assert curve is not None  # el personaje sigue al mecanismo durante todo el ciclo


def test_character_family_naming_and_protocol_delegation():
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    assert fam.name == "four_bar+pierna" and fam.name != FOUR_BAR.name  # no usa la base de datos
    assert (fam.n_params, fam.n_compact) == (FOUR_BAR.n_params, FOUR_BAR.n_compact)
    assert fam.build(np.full(FOUR_BAR.n_params, -1.0), Variant(1)) is None  # parámetros imposibles


def test_rigged_mechanism_animates_and_exposes_character_geometry():
    ch = leg_character()
    fam = CharacterFamily(FOUR_BAR, ch, LEG_ATTACH, LEG_MARKER)
    cand = fam.candidates(ellipse(0.9, -2.4, 0.55, 0.25), 40, 16000, 1, np.random.default_rng(0))[0]
    rig = fam.to_mechanism(cand.p0, cand.variant)
    assert isinstance(rig, RiggedMechanism) and rig.n_char == len(ch.parts)
    assert len(rig.mech_state(rig.state0)) == 3 * len(rig.base.asm.components)
    assert len(rig.part_polygons(rig.state0)) == rig.n_char
    assert rig.outline_points(rig.state0).shape[1] == 2
    pb = Playback(rig.asm, rig.state0, period=2.0)
    for _ in range(100):
        pb.advance(0.02)
        assert pb.ok
    # con el personaje en el lazo, el marcador del pie traza el ciclo y vuelve al inicio
    assert rig.marker_world(pb.state) == pytest.approx(rig.marker_world(rig.state0), abs=1e-5)


# --- ajuste con el personaje en la simulación ----------------------------------------------------


def test_fit_recovers_a_foot_curve_that_the_leg_can_really_draw():
    """Objetivo alcanzable: la curva del pie con un mecanismo factible. El ajuste con el personaje
    dentro del lazo debe volver a encontrarla con precisión."""
    ch = leg_character()
    fam = CharacterFamily(FOUR_BAR, ch, LEG_ATTACH, LEG_MARKER)
    seed = fam.candidates(ellipse(0.9, -2.4, 0.4, 0.2), 48, 16000, 1, np.random.default_rng(0))[0]
    reachable = trace_curve_generic(fam._builder(seed.variant), seed.p0, 120)
    assert reachable is not None
    result = fit(reachable, [fam], seed=3)
    assert result is not None
    assert result.rms_rel < 0.03, f"{result.rms_rel:.4f}"


def test_fit_improves_a_hard_ellipse_but_the_leg_cannot_draw_it_exactly():
    """Con esta pierna y un pasador en la pantorrilla el pie no llega a dibujar bien elipses de
    este tipo (en las pruebas queda en 6-8 %, no en 3 %). Solo se exige que el ajuste mejore mucho
    el candidato inicial; el límite está documentado en PROGRESS.md."""
    fam = CharacterFamily(FOUR_BAR, leg_character(), LEG_ATTACH, LEG_MARKER)
    target = ellipse(0.9, -2.4, 0.55, 0.23)
    result = fit(target, [fam], seed=0)
    assert result is not None
    assert result.rms_rel < 0.5 * result.initial_rms_rel
    assert bbox_diag(result.curve) > 0.5 * bbox_diag(target)  # no colapsa a una curva diminuta
