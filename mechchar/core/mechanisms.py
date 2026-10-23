"""Mecanismos básicos listos para simular, con una configuración inicial factible (analítica)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mechchar.core.assembly import (
    Assembly,
    Drive,
    FixedState,
    Phase,
    Pin,
    PointOnLine,
    Vec,
    rot,
)


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
    lengths: dict[int, float]  # largo de cada eslabón móvil (distancia entre sus pines)

    def tracer_world(self, state: np.ndarray) -> np.ndarray:
        return self.asm.world_point(state, self.coupler, self.tracer)

    def joints(self, state: np.ndarray) -> np.ndarray:
        """Articulaciones (bancada A y D, punta de manivela, extremo del acoplador) y marcador."""
        w = self.asm.world_point
        return np.array(
            [
                w(state, self.ground, (0.0, 0.0)),
                w(state, self.rocker, (0.0, 0.0)),
                w(state, self.crank, (self.lengths[self.crank], 0.0)),
                w(state, self.coupler, (self.lengths[self.coupler], 0.0)),
                self.tracer_world(state),
            ]
        )

    def outline_points(self, state: np.ndarray) -> np.ndarray:
        """Puntos que delimitan el dibujo del mecanismo (para encuadrar la vista)."""
        return self.joints(state)


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
    return FourBar(asm, asm.initial_state(), g, c, k, b, tracer, {c: r, k: l, b: rr})


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


@dataclass
class GearedFiveBar:
    """Mecanismo de dos engranajes (Fig. 4 del paper): cada engranaje lleva una manivela, y las
    dos bielas se unen en un pin; la primera se prolonga hasta el marcador.

    Los pivotes de los engranajes están en A = (0, 0) y D = (d, 0) del marco de la bancada. El
    engranaje 1 lo mueve el Input Driver y el 2 gira con θ₂ = ratio·θ₁ + δ. Para dibujar, los
    radios de los engranajes salen de la relación de velocidades: R₁/R₂ = |ratio|, R₁ + R₂ = d."""

    asm: Assembly
    state0: np.ndarray
    ground: int
    gear1: int
    gear2: int
    link1: int
    link2: int
    tracer: Vec  # marcador, en coordenadas locales de ``link1``
    ratio: float
    d: float
    r1: float
    r2: float
    l1: float
    l2: float

    def tracer_world(self, state: np.ndarray) -> np.ndarray:
        return self.asm.world_point(state, self.link1, self.tracer)

    def gear_radii(self) -> tuple[float, float]:
        k = abs(self.ratio)
        return self.d * k / (1.0 + k), self.d / (1.0 + k)

    def joints(self, state: np.ndarray) -> np.ndarray:
        """Pivotes A y D, puntas de las manivelas, pin entre bielas y marcador."""
        w = self.asm.world_point
        return np.array(
            [
                w(state, self.gear1, (0.0, 0.0)),
                w(state, self.gear2, (0.0, 0.0)),
                w(state, self.gear1, (self.r1, 0.0)),
                w(state, self.gear2, (self.r2, 0.0)),
                w(state, self.link1, (self.l1, 0.0)),
                self.tracer_world(state),
            ]
        )

    def discs(self, state: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Centro y radio de cada engranaje (solo para dibujar)."""
        j = self.joints(state)
        r1, r2 = self.gear_radii()
        return [(j[0], r1), (j[1], r2)]

    def outline_points(self, state: np.ndarray) -> np.ndarray:
        pts = [self.joints(state)]
        for c, r in self.discs(state):
            pts.append(c + r * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))
        return np.vstack(pts)


def geared_five_bar(
    *,
    d: float,
    r1: float,
    r2: float,
    l1: float,
    l2: float,
    ratio: float,
    tracer: Vec | None = None,
    branch: int = 1,
    delta: float = 0.0,
    psi: float = 0.0,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GearedFiveBar:
    """5-bar con dos engranajes.

    ``psi`` es el ángulo de la manivela 1 en t = 0 y ``delta`` el desfase entre manivelas
    (θ₂ = ratio·θ₁ + δ), ambos en el marco de la bancada. ``pose`` = (x, y, ángulo) de la
    bancada en el mundo. ``branch`` = ±1 elige el ensamblado del lazo. Lanza ValueError si el
    mecanismo no se puede armar en t = 0."""
    if min(d, r1, r2, l1, l2) <= 0.0:
        raise ValueError("los largos deben ser positivos")
    if ratio == 0.0:
        raise ValueError("la relación de engranajes no puede ser cero")
    ax, ay, phi = pose
    tracer = tracer if tracer is not None else (0.5 * l1, 1.0)
    th1, th2 = psi, ratio * psi + delta
    b1 = np.array([r1 * math.cos(th1), r1 * math.sin(th1)])
    b2 = np.array([d + r2 * math.cos(th2), r2 * math.sin(th2)])
    cands = circle_intersections((float(b1[0]), float(b1[1])), l1, (float(b2[0]), float(b2[1])), l2)
    j = cands[0] if branch > 0 else cands[1]

    origin, rot_g = np.array([ax, ay]), rot(phi)

    def to_world(q: np.ndarray) -> np.ndarray:
        return origin + rot_g @ q

    b1w, b2w, dw = to_world(b1), to_world(b2), to_world(np.array([d, 0.0]))
    asm = Assembly()
    g = asm.add_component("ground", ax, ay, phi)
    c1 = asm.add_component("gear1", ax, ay, phi + th1)
    c2 = asm.add_component("gear2", dw[0], dw[1], phi + th2)
    k1 = asm.add_component(
        "link1", b1w[0], b1w[1], phi + math.atan2(j[1] - b1[1], j[0] - b1[0])
    )
    k2 = asm.add_component(
        "link2", b2w[0], b2w[1], phi + math.atan2(j[1] - b2[1], j[0] - b2[0])
    )
    asm.add(FixedState(g, ax, ay, phi))
    asm.add(Pin(g, (0.0, 0.0), c1, (0.0, 0.0)))
    asm.add(Pin(g, (d, 0.0), c2, (0.0, 0.0)))
    asm.add(Drive(c1, offset=phi + psi))
    # θ₂(mundo) = ratio·θ₁(mundo) + φ(1 − ratio) + δ, porque los ángulos locales se miden desde φ
    asm.add(Phase(c1, c2, ratio, phi * (1.0 - ratio) + delta))
    asm.add(Pin(c1, (r1, 0.0), k1, (0.0, 0.0)))
    asm.add(Pin(c2, (r2, 0.0), k2, (0.0, 0.0)))
    asm.add(Pin(k1, (l1, 0.0), k2, (l2, 0.0)))
    return GearedFiveBar(
        asm, asm.initial_state(), g, c1, c2, k1, k2, tracer, ratio, d, r1, r2, l1, l2
    )


Mechanism = FourBar | GearedFiveBar  # mecanismos que el ajuste sabe producir y la UI sabe dibujar
