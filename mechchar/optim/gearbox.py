"""Caja de engranajes (M4, paper §5): el rectángulo, en la vista lateral, donde tiene que caber
el mecanismo durante todo su ciclo.

Se usa en dos lugares, como en el video:
- al buscar candidatos se descartan los que no caben (``filter_candidates``, ``BoxedFamily``);
- al refinar, ``BoxedFamily.compactness`` suma residuos de bisagra que empujan al mecanismo a
  quedarse dentro, así que el ajuste continuo no lo saca de la caja.

«Cabe» mira los puntos del dibujo del mecanismo (articulaciones y discos de engranaje), no el
personaje. Sin GTK.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mechchar.optim.fitcore import Built, Candidate, Family, LocalSample, Variant

Bounds = tuple[float, float, float, float]  # x0, y0, x1, y1
N_PHASES = 24  # fases del ciclo con las que se mide la extensión
BOX_RESIDUALS = 4  # izquierda, derecha, abajo, arriba
BOX_WEIGHT = 2.0  # peso de los residuos de la caja frente a los de distancia a la curva
POOL_FACTOR = 4  # cuántos candidatos pedir por cada uno que se quiere conservar (base de datos)
MIN_POOL = 120  # pool mínimo al buscar al azar: los de menor error suelen ser los más grandes
MIN_POOL_CHARACTER = 40  # con personaje cada candidato se simula completo, así que es más caro


@dataclass(frozen=True)
class GearBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_corners(cls, a: tuple[float, float], b: tuple[float, float]) -> GearBox:
        """Caja a partir de dos esquinas opuestas (en cualquier orden)."""
        return cls(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def overshoot(self, bounds: Bounds) -> tuple[float, float, float, float]:
        """Cuánto se sale ``bounds`` por la izquierda, derecha, abajo y arriba (≥ 0)."""
        bx0, by0, bx1, by1 = bounds
        return (
            max(0.0, self.x0 - bx0), max(0.0, bx1 - self.x1),
            max(0.0, self.y0 - by0), max(0.0, by1 - self.y1),
        )

    @property
    def slack(self) -> float:
        """Tolerancia para decir que algo «cabe»: el refinamiento deja al mecanismo apoyado en el
        borde y la extensión se mide en un número finito de fases (0,1 % de la caja)."""
        return 1e-3 * max(self.width, self.height)

    def worst_overshoot(self, bounds: Bounds) -> float:
        return max(self.overshoot(bounds))

    def contains(self, bounds: Bounds, tol: float = 1e-9) -> bool:
        return self.worst_overshoot(bounds) <= tol


def _points_over_cycle(family: Any, p: np.ndarray, variant: Variant, n: int) -> np.ndarray | None:
    """Puntos del dibujo del mecanismo (sin el personaje) en ``n`` fases del ciclo. Si la familia
    lleva un personaje, el mecanismo se simula con el personaje dentro del lazo."""
    if hasattr(family, "character"):  # CharacterFamily: el estado es personaje + mecanismo
        built = family.build(p, variant)
        mech = family.base.to_mechanism(p, variant)
        skip = 3 * len(family.character.parts)
    else:
        mech = family.to_mechanism(p, variant)
        built = family.build(p, variant)
        skip = 0
    if built is None or mech is None:
        return None
    # el salto de rama no se vigila (el umbral absoluto de ``trace_cycle`` no escala con el
    # tamaño del mecanismo); los mecanismos que llegan acá ya se validaron al armarlos
    cyc = built.asm.trace_cycle(built.state0, n_steps=n, max_jump=math.inf)
    if not cyc.ok:
        return None
    return np.vstack([np.asarray(mech.outline_points(s[skip:])) for s in cyc.states])


def mechanism_bounds(
    family: Any, p: np.ndarray, variant: Variant, n: int = N_PHASES
) -> Bounds | None:
    """Rectángulo mínimo que contiene al mecanismo durante todo el ciclo, o None si el mecanismo
    no completa el ciclo. Se mide en ``n`` fases, así que puede quedar apenas por debajo del
    extremo real (menos del 1 % del radio de una manivela con 24 fases)."""
    pts = _points_over_cycle(family, p, variant, n)
    if pts is None:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def fits(family: Any, p: np.ndarray, variant: Variant, box: GearBox, tol: float = 1e-9) -> bool:
    b = mechanism_bounds(family, p, variant)
    return b is not None and box.contains(b, tol)


def filter_candidates(
    family: Any, cands: Sequence[Candidate], box: GearBox, tol: float = 1e-9
) -> list[Candidate]:
    """Los candidatos que caben en la caja, en el mismo orden."""
    return [c for c in cands if fits(family, c.p0, c.variant, box, tol)]


class BoxedFamily:
    """Una familia restringida a una caja: cumple el mismo protocolo que la familia que envuelve
    (y conserva su ``name``, así que la base de movimientos sigue aplicándose)."""

    def __init__(self, base: Family, box: GearBox) -> None:
        self.base, self.box = base, box
        self.name = base.name
        self.n_params = base.n_params
        self.n_compact = base.n_compact + BOX_RESIDUALS
        if hasattr(base, "character"):  # para que la extensión se mida con el personaje
            self.character = base.character  # type: ignore[attr-defined]

    # --- protocolo de familia -------------------------------------------------------------
    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        return self.base.build(p, variant)

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> Any:
        return self.base.to_mechanism(p, variant)

    def sample_local(self, rng: np.random.Generator, k: int, n: int) -> LocalSample | None:
        return self.base.sample_local(rng, k, n)

    def assemble(
        self, shape: np.ndarray, variant: Variant, scale: float, phi: float, origin: np.ndarray
    ) -> np.ndarray:
        return self.base.assemble(shape, variant, scale, phi, origin)

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        floor = MIN_POOL_CHARACTER if hasattr(self.base, "character") else MIN_POOL
        pool = self.base.candidates(y, n, n_candidates, max(POOL_FACTOR * n_keep, floor), rng)
        return filter_candidates(self.base, pool, self.box)[:n_keep]

    # --- residuos de la caja --------------------------------------------------------------
    def _box_residuals(self, p: np.ndarray, variant: Variant) -> np.ndarray | None:
        b = mechanism_bounds(self.base, p, variant)
        if b is None:
            return None
        return BOX_WEIGHT * np.asarray(self.box.overshoot(b))

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]:
        r_base, j_base = self.base.compactness(p, size, variant)
        r_box = self._box_residuals(p, variant)
        if r_box is None:
            r_box = np.zeros(BOX_RESIDUALS)
        jac = np.zeros((BOX_RESIDUALS, len(p)))
        if np.any(r_box > 0.0):  # solo cuesta el Jacobiano cuando la caja está activa
            for j in range(len(p)):
                step = 1e-6 * (1.0 + abs(p[j]))
                q = p.copy()
                q[j] += step
                r_up = self._box_residuals(q, variant)
                if r_up is not None:
                    jac[:, j] = (r_up - r_box) / step
        return np.concatenate([r_base, r_box]), np.vstack([j_base, jac])


def box_families(families: Sequence[Any], box: GearBox | None) -> tuple[Any, ...]:
    """Las mismas familias, restringidas a ``box`` (o sin cambios si no hay caja)."""
    if box is None:
        return tuple(families)
    return tuple(BoxedFamily(f, box) for f in families)
