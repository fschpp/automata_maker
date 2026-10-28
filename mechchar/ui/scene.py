"""Dibujo del cuadrilátero con Cairo. Solo presentación: toda la geometría viene de ``core``."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import numpy as np

from mechchar.core.assembly import world_point
from mechchar.core.character import Character, Part, PointRef, RiggedMechanism
from mechchar.core.gearnet import GearTrain
from mechchar.core.gearshape import gear_outline, gear_rotations
from mechchar.core.mechanisms import FourBar, GearedFiveBar, SliderCrank
from mechchar.core.rig import MultiRig
from mechchar.core.viewport import Viewport

LINK_COLOR = (0.15, 0.25, 0.45)
GROUND_COLOR = (0.35, 0.35, 0.35)
TRACER_COLOR = (0.85, 0.15, 0.15)
GEAR_FILL = (0.86, 0.68, 0.25, 0.35)  # dorado translúcido, como en el video
GEAR_EDGE = (0.55, 0.40, 0.05)
PROGRESS_COLOR = (0.95, 0.5, 0.05)
LAYER_FILLS = (  # un color por capa del tren, dorado primero como en el video
    (0.86, 0.68, 0.25, 0.45),
    (0.29, 0.47, 0.76, 0.45),
    (0.29, 0.63, 0.42, 0.45),
    (0.71, 0.32, 0.48, 0.45),
)
SELECTED_FILL = (0.05, 0.12, 0.45, 0.6)  # azul oscuro: engranajes de los ejes elegidos
INPUT_FILL = (0.95, 0.5, 0.05, 0.6)  # naranja: engranajes del eje de entrada


def _screen(vp: Viewport, p: np.ndarray) -> tuple[float, float]:
    return vp.world_to_screen(float(p[0]), float(p[1]))


def _line(cr: Any, vp: Viewport, a: np.ndarray, b: np.ndarray, width: float, color) -> None:
    cr.set_source_rgb(*color)
    cr.set_line_width(width)
    cr.set_line_cap(1)  # redondeado
    cr.move_to(*_screen(vp, a))
    cr.line_to(*_screen(vp, b))
    cr.stroke()


def _joint(cr: Any, vp: Viewport, p: np.ndarray, radius: float = 5.0) -> None:
    x, y = _screen(vp, p)
    cr.set_source_rgb(1, 1, 1)
    cr.arc(x, y, radius, 0, 6.283185307179586)
    cr.fill_preserve()
    cr.set_source_rgb(0.1, 0.1, 0.1)
    cr.set_line_width(1.5)
    cr.stroke()


def draw_path(cr: Any, vp: Viewport, path: list[np.ndarray]) -> None:
    """Curva cerrada trazada por el marcador a lo largo de un ciclo."""
    if len(path) < 2:
        return
    cr.set_source_rgba(*TRACER_COLOR, 0.45)
    cr.set_line_width(1.5)
    cr.move_to(*_screen(vp, path[0]))
    for p in path[1:]:
        cr.line_to(*_screen(vp, p))
    cr.close_path()
    cr.stroke()


def draw_four_bar(cr: Any, vp: Viewport, fb: FourBar, state: np.ndarray) -> None:
    w = fb.asm.world_point
    ground_a = w(state, fb.ground, (0.0, 0.0))
    ground_d = w(state, fb.rocker, (0.0, 0.0))  # pivote del balancín = extremo de la bancada
    crank_tip = w(state, fb.crank, (_len(fb, fb.crank), 0.0))
    coupler_end = w(state, fb.coupler, (_len(fb, fb.coupler), 0.0))
    tracer = fb.tracer_world(state)

    _line(cr, vp, ground_a, ground_d, 3.0, GROUND_COLOR)
    _line(cr, vp, ground_a, crank_tip, 5.0, LINK_COLOR)
    _line(cr, vp, ground_d, coupler_end, 5.0, LINK_COLOR)
    # acoplador: eslabón triangular hacia el marcador
    coupler_start = w(state, fb.coupler, (0.0, 0.0))
    _line(cr, vp, coupler_start, coupler_end, 5.0, LINK_COLOR)
    _line(cr, vp, coupler_start, tracer, 2.0, LINK_COLOR)
    _line(cr, vp, coupler_end, tracer, 2.0, LINK_COLOR)

    for p in (ground_a, ground_d, crank_tip, coupler_end):
        _joint(cr, vp, p)
    x, y = _screen(vp, tracer)
    cr.set_source_rgb(*TRACER_COLOR)
    cr.arc(x, y, 5.0, 0, 6.283185307179586)
    cr.fill()


def _len(fb: FourBar, comp: int) -> float:
    """Largo de un eslabón: distancia entre sus dos pines (guardada al construir el mecanismo)."""
    return fb.lengths[comp]


def _disc(cr: Any, vp: Viewport, center: np.ndarray, radius: float) -> None:
    """Engranaje dibujado como disco (los dientes aparecen recién en la etapa del tren, M5)."""
    x, y = _screen(vp, center)
    cr.arc(x, y, radius * vp.scale, 0, 6.283185307179586)
    cr.set_source_rgba(*GEAR_FILL)
    cr.fill_preserve()
    cr.set_source_rgb(*GEAR_EDGE)
    cr.set_line_width(1.5)
    cr.stroke()


def draw_five_bar(cr: Any, vp: Viewport, fb: GearedFiveBar, state: np.ndarray) -> None:
    a, d, b1, b2, j, tracer = fb.joints(state)
    for center, radius in fb.discs(state):
        _disc(cr, vp, center, radius)
    _line(cr, vp, a, b1, 4.0, LINK_COLOR)  # manivela 1
    _line(cr, vp, d, b2, 4.0, LINK_COLOR)  # manivela 2
    _line(cr, vp, b1, j, 5.0, LINK_COLOR)  # biela 1
    _line(cr, vp, b2, j, 5.0, LINK_COLOR)  # biela 2
    _line(cr, vp, j, tracer, 3.0, LINK_COLOR)  # prolongación de la biela 1 hasta el marcador
    for p in (a, d, b1, b2, j):
        _joint(cr, vp, p)
    x, y = _screen(vp, tracer)
    cr.set_source_rgb(*TRACER_COLOR)
    cr.arc(x, y, 5.0, 0, 6.283185307179586)
    cr.fill()


def draw_slider_crank(cr: Any, vp: Viewport, sc: SliderCrank, state: np.ndarray) -> None:
    a, b, slider, tracer = sc.joints(state)
    g0, g1 = sc.guide(state)
    _line(cr, vp, g0, g1, 3.0, GROUND_COLOR)  # guía de la corredera
    _line(cr, vp, a, b, 4.0, LINK_COLOR)  # manivela
    _line(cr, vp, b, slider, 5.0, LINK_COLOR)  # biela
    _line(cr, vp, b, tracer, 2.0, LINK_COLOR)
    _line(cr, vp, slider, tracer, 2.0, LINK_COLOR)
    for p in (a, b, slider):
        _joint(cr, vp, p)
    x, y = _screen(vp, tracer)
    cr.set_source_rgb(*TRACER_COLOR)
    cr.arc(x, y, 5.0, 0, 6.283185307179586)
    cr.fill()


PART_FILL = (0.86, 0.88, 0.92)
FIXED_FILL = (0.62, 0.64, 0.70)
PART_EDGE = (0.30, 0.33, 0.42)
ATTACH_COLOR = (0.95, 0.65, 0.10)


def _polygon(cr: Any, vp: Viewport, poly: np.ndarray, fill: tuple, edge: tuple) -> None:
    cr.move_to(*_screen(vp, poly[0]))
    for p in poly[1:]:
        cr.line_to(*_screen(vp, p))
    cr.close_path()
    cr.set_source_rgb(*fill)
    cr.fill_preserve()
    cr.set_source_rgb(*edge)
    cr.set_line_width(1.5)
    cr.stroke()


def draw_character(
    cr: Any,
    vp: Viewport,
    character: Character,
    state: np.ndarray,
    marks: list[PointRef] | None = None,
    select: Callable[[Part], bool] | None = None,
) -> None:
    """Piezas del personaje (las fijas, más oscuras) y, si se pide, sus puntos marcados:
    naranja el punto de actuación (primero) y rojo el marcador de la curva (segundo).
    ``select`` limita qué piezas se dibujan."""
    for i, part in enumerate(character.parts):
        if select is not None and not select(part):
            continue
        poly = character.part_polygon(state, i)
        _polygon(cr, vp, poly, FIXED_FILL if part.fixed else PART_FILL, PART_EDGE)
    for ref, color in zip(marks or [], (ATTACH_COLOR, TRACER_COLOR), strict=False):
        i = character.index(ref.part)
        x, y = _screen(vp, world_point(state, i, character.local_point(ref.part, ref.point)))
        cr.arc(x, y, 6.0, 0, 6.283185307179586)
        cr.set_source_rgb(*color)
        cr.fill()


def draw_rigged(cr: Any, vp: Viewport, rig: RiggedMechanism, state: np.ndarray) -> None:
    draw_character(cr, vp, rig.character, state)
    mech_state = rig.mech_state(state)
    draw_mechanism(cr, vp, rig.base, mech_state)
    # el marcador del mecanismo es el punto de actuación (naranja); el rojo es el del personaje
    x, y = _screen(vp, rig.base.tracer_world(mech_state))
    cr.arc(x, y, 6.0, 0, 6.283185307179586)
    cr.set_source_rgb(*ATTACH_COLOR)
    cr.fill()
    x, y = _screen(vp, rig.marker_world(state))
    cr.arc(x, y, 6.0, 0, 6.283185307179586)
    cr.set_source_rgb(*TRACER_COLOR)
    cr.fill()


def _in_plane(z: float) -> Callable[[Part], bool]:
    return lambda part: not part.fixed and part.z == z


Z_SHIFT_PX = 26.0  # desplazamiento en pantalla por plano de profundidad (solo al dibujar)
INACTIVE_ALPHA = 0.4


def draw_multirig(
    cr: Any, vp: Viewport, rig: MultiRig, state: np.ndarray, active: int | None = None
) -> None:
    """Personaje con todos sus mecanismos; el marcador de cada pierna en rojo y su punto de
    actuación en naranja.

    Las piezas móviles de cada mecanismo se dibujan en el plano ``z`` de su pieza de actuación,
    corridas ``Z_SHIFT_PX`` píxeles por plano para que dos piernas superpuestas en 2D se lean por
    separado (la simulación no cambia). Si ``active`` es un índice de mecanismo, los demás se
    atenúan: así se ve qué pierna controla el slider que se está moviendo."""
    ch = rig.character
    draw_character(cr, vp, ch, state, select=lambda part: part.fixed)
    zs = sorted({ch.parts[ch.index(sub.attach.part)].z for sub in rig.subs})
    for k, sub in enumerate(rig.subs):
        z = ch.parts[ch.index(sub.attach.part)].z
        cr.save()
        cr.translate(Z_SHIFT_PX * zs.index(z), 0.0)
        dim = active is not None and active != k
        if dim:
            cr.push_group()
        draw_character(cr, vp, ch, state, select=_in_plane(z))
        mech_state = rig.mech_state(state, k)
        draw_mechanism(cr, vp, sub.mechanism, mech_state)
        x, y = _screen(vp, sub.mechanism.tracer_world(mech_state))
        cr.arc(x, y, 6.0, 0, 6.283185307179586)
        cr.set_source_rgb(*ATTACH_COLOR)
        cr.fill()
        x, y = _screen(vp, rig.marker_world(state, k))
        cr.arc(x, y, 6.0, 0, 6.283185307179586)
        cr.set_source_rgb(*TRACER_COLOR)
        cr.fill()
        if dim:
            cr.pop_group_to_source()
            cr.paint_with_alpha(INACTIVE_ALPHA)
        cr.restore()


def draw_mechanism(
    cr: Any, vp: Viewport, mech: Any, state: np.ndarray, active: int | None = None
) -> None:
    if isinstance(mech, MultiRig):
        draw_multirig(cr, vp, mech, state, active)
    elif isinstance(mech, RiggedMechanism):
        draw_rigged(cr, vp, mech, state)
    elif isinstance(mech, GearedFiveBar):
        draw_five_bar(cr, vp, mech, state)
    elif isinstance(mech, SliderCrank):
        draw_slider_crank(cr, vp, mech, state)
    else:
        draw_four_bar(cr, vp, mech, state)


CANDIDATE_COLORS = ((0.84, 0.15, 0.16), (0.17, 0.63, 0.17), (0.12, 0.47, 0.71))  # rojo, verde, azul


def draw_candidates(cr: Any, vp: Viewport, curves: list[np.ndarray]) -> None:
    """Curvas de los candidatos de la base (0 rojo, 1 verde, 2 azul), con su número."""
    for idx, curve in enumerate(curves[: len(CANDIDATE_COLORS)]):
        color = CANDIDATE_COLORS[idx]
        cr.set_source_rgba(*color, 0.9)
        cr.set_line_width(2.0)
        cr.move_to(*_screen(vp, curve[0]))
        for p in curve[1:]:
            cr.line_to(*_screen(vp, p))
        cr.close_path()
        cr.stroke()
        top = curve[np.argmax(curve[:, 1])]  # etiqueta sobre el punto más alto de la curva
        x, y = _screen(vp, top)
        cr.set_font_size(15.0)
        cr.move_to(x + 6, y - 6)
        cr.show_text(str(idx))


def draw_progress_curve(cr: Any, vp: Viewport, curve: np.ndarray) -> None:
    """Curva que se está evaluando durante el ajuste (la curva blanca del video)."""
    if len(curve) < 2:
        return
    cr.set_source_rgba(*PROGRESS_COLOR, 0.9)
    cr.set_line_width(2.0)
    cr.move_to(*_screen(vp, curve[0]))
    for p in curve[1:]:
        cr.line_to(*_screen(vp, p))
    cr.close_path()
    cr.stroke()


def draw_gear_train(
    cr: Any,
    vp: Viewport,
    train: GearTrain,
    theta: float = 0.0,
    selected: Collection[int] = (),
) -> None:
    """Red de engranajes con dientes, girada ``theta`` (radianes del eje de entrada).

    Cada conexión tiene el color de su capa; los engranajes de los ejes en ``selected`` (índices
    en ``train.axes``) van en azul oscuro y los del eje de entrada en naranja. Se dibuja un punto
    en cada eje."""
    for c, rots in zip(train.connections, gear_rotations(train, theta), strict=True):
        base = LAYER_FILLS[c.layer % len(LAYER_FILLS)]
        for k, (g, rot) in enumerate(zip(c.gears, rots, strict=True)):
            axis = c.a if k == 0 else c.b if k == len(c.gears) - 1 else None
            fill = base
            if axis is not None and axis in selected:
                fill = SELECTED_FILL
            elif axis is not None and axis == train.root:
                fill = INPUT_FILL
            pts = gear_outline(g, rot)
            cr.move_to(*_screen(vp, pts[0]))
            for p in pts[1:]:
                cr.line_to(*_screen(vp, p))
            cr.close_path()
            cr.set_source_rgba(*fill)
            cr.fill_preserve()
            cr.set_source_rgb(*GEAR_EDGE)
            cr.set_line_width(1.2)
            cr.stroke()
    for a in train.axes:
        _joint(cr, vp, np.array([a.x, a.y]), 3.5)
