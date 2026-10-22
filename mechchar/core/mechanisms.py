"""Mecanismos básicos listos para simular, con una configuración inicial factible (analítica)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Pin, PointOnLine, Vec


def circle_intersections(c0: Vec, r0: float, c1: Vec, r1: float) -> list[np.ndarray]:
    """Puntos de intersección de dos circunferencias; ValueError si no se cortan."""
    p0, p1 = np.asarray(c0, dtype=float), np.asarray(c1, dtype=float)
    d = float(np.linalg.norm(p1 - p0))
    if d == 0.0:
        raise ValueError("circunferencias concéntricas")
    a = (r0**2 - r1**2 + d**2) / (2.0 * d)
    h2 = r0**2 - a**2
    if h2 < 0.0:
        raise ValueError("las circunferencias no se cortan")
    h = math.sqrt(h2)
    mid = p0 + a * (p1 - p0) / d
    perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
    return [mid + h * perp, mid - h * perp]


@dataclass
class FourBar:
    asm: Assembly
    state0: np.ndarray
    ground: int
    crank: int
    coupler: int
    rocker: int
    tracer: Vec  # punto trazador, en coordenadas locales del acoplador

    def tracer_world(self, state: np.ndarray) -> np.ndarray:
        return self.asm.world_point(state, self.coupler, self.tracer)


def four_bar(
    d: float,
    r: float,
    l: float,  # noqa: E741
    rr: float,
    tracer: Vec | None = None,
    branch: int = 1,
) -> FourBar:
    """Cuadrilátero: bancada de largo ``d`` (de (0,0) a (d,0)), manivela ``r``, acoplador ``l`` y
    balancín ``rr``. ``branch`` = +1 / -1 elige el ensamblado con el acoplador por arriba / abajo
    en t = 0. La manivela la mueve un Input Driver (θ = 2π t)."""
    tracer = tracer if tracer is not None else (0.5 * l, 1.0)
    crank_tip = (r, 0.0)
    cands = circle_intersections(crank_tip, l, (d, 0.0), rr)
    p = max(cands, key=lambda q: branch * q[1])
    asm = Assembly()
    g = asm.add_component("ground")
    c = asm.add_component("crank")
    k = asm.add_component(
        "coupler", r, 0.0, math.atan2(p[1] - crank_tip[1], p[0] - crank_tip[0])
    )
    b = asm.add_component("rocker", d, 0.0, math.atan2(p[1], p[0] - d))
    asm.add(FixedState(g, 0.0, 0.0, 0.0))
    asm.add(Pin(g, (0.0, 0.0), c, (0.0, 0.0)))
    asm.add(Pin(c, (r, 0.0), k, (0.0, 0.0)))
    asm.add(Pin(k, (l, 0.0), b, (rr, 0.0)))
    asm.add(Pin(b, (0.0, 0.0), g, (d, 0.0)))
    asm.add(Drive(c))
    return FourBar(asm, asm.initial_state(), g, c, k, b, tracer)


@dataclass
class CrankSlider:
    asm: Assembly
    state0: np.ndarray
    ground: int
    crank: int
    rod: int
    r: float
    l: float  # noqa: E741

    def slider_x(self, state: np.ndarray) -> float:
        return float(self.asm.world_point(state, self.rod, (self.l, 0.0))[0])


def crank_slider(r: float, l: float) -> CrankSlider:  # noqa: E741
    """Manivela ``r`` y biela ``l`` con el extremo de la biela guiado sobre el eje x."""
    if l <= r:
        raise ValueError("la biela debe ser más larga que la manivela")
    asm = Assembly()
    g = asm.add_component("ground")
    c = asm.add_component("crank")
    k = asm.add_component("rod", r, 0.0, 0.0)
    asm.add(FixedState(g, 0.0, 0.0, 0.0))
    asm.add(Pin(g, (0.0, 0.0), c, (0.0, 0.0)))
    asm.add(Pin(c, (r, 0.0), k, (0.0, 0.0)))
    asm.add(PointOnLine(k, (l, 0.0), g, (0.0, 0.0), (1.0, 0.0)))
    asm.add(Drive(c))
    return CrankSlider(asm, asm.initial_state(), g, c, k, r, l)
