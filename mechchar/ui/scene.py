"""Dibujo del cuadrilátero con Cairo. Solo presentación: toda la geometría viene de ``core``."""

from __future__ import annotations

from typing import Any

import numpy as np

from mechchar.core.mechanisms import FourBar, GearedFiveBar, Mechanism
from mechchar.core.viewport import Viewport

LINK_COLOR = (0.15, 0.25, 0.45)
GROUND_COLOR = (0.35, 0.35, 0.35)
TRACER_COLOR = (0.85, 0.15, 0.15)
GEAR_FILL = (0.86, 0.68, 0.25, 0.35)  # dorado translúcido, como en el video
GEAR_EDGE = (0.55, 0.40, 0.05)
PROGRESS_COLOR = (0.95, 0.5, 0.05)


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


def draw_mechanism(cr: Any, vp: Viewport, mech: Mechanism, state: np.ndarray) -> None:
    if isinstance(mech, GearedFiveBar):
        draw_five_bar(cr, vp, mech, state)
    else:
        draw_four_bar(cr, vp, mech, state)


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
