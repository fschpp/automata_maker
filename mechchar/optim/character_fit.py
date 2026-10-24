"""Ajuste de un mecanismo unido a un personaje (M4): la curva objetivo es la de un marcador del
personaje, no la del mecanismo, y ambos se simulan juntos.

``CharacterFamily`` envuelve a una familia de mecanismos (cuadrilátero, 5-bar...) y cumple su mismo
protocolo, así que ``fitcore.fit`` y el gradiente por función implícita funcionan sin cambios: el
estado incluye los grados de libertad del personaje y el marcador es el del personaje.

Los candidatos iniciales se generan para la curva objetivo trasladada por el desplazamiento entre
el punto de actuación y el marcador en reposo (el punto de actuación se mueve parecido al marcador)
y se colocan de modo que el mecanismo coincida con el punto de actuación en t = 0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mechchar.core.assembly import world_point
from mechchar.core.character import Character, PointRef, RiggedMechanism, combine
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.fitcore import (
    Built,
    Candidate,
    Family,
    LocalSample,
    Variant,
    trace_curve_generic,
)


class CharacterFamily:
    def __init__(
        self, base: Family, character: Character, attach: PointRef, marker: PointRef
    ) -> None:
        self.base, self.character, self.attach, self.marker = base, character, attach, marker
        # nombre distinto del de la base: así no se le aplica la base de movimientos de la base
        self.name = f"{base.name}+{character.name}"
        self.n_params = base.n_params
        self.n_compact = base.n_compact

    # --- protocolo de familia -------------------------------------------------------------
    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        built = self.base.build(p, variant)
        if built is None:
            return None
        return combine(self.character, built, self.attach, self.marker)

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base.compactness(p, size, variant)

    def sample_local(self, rng: np.random.Generator, k: int, n: int) -> LocalSample | None:
        return self.base.sample_local(rng, k, n)

    def assemble(
        self, shape: np.ndarray, variant: Variant, scale: float, phi: float, origin: np.ndarray
    ) -> np.ndarray:
        return self.base.assemble(shape, variant, scale, phi, origin)

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> RiggedMechanism | None:
        built = self.build(p, variant)
        base = self.base.to_mechanism(p, variant)
        if built is None or base is None:
            return None
        return RiggedMechanism(
            built.asm, built.state0, self.character, len(self.character.parts), base,
            built.coupler, built.tracer,
        )

    def _builder(self, variant: Variant) -> Callable[[np.ndarray], Built | None]:
        def build(q: np.ndarray) -> Built | None:
            return self.build(q, variant)

        return build

    def place_on_attach_point(self, p: np.ndarray, variant: Variant) -> np.ndarray | None:
        """Traslada el mecanismo para que su marcador caiga sobre el punto de actuación en t = 0
        (el pasador es entonces factible con el personaje en reposo)."""
        built = self.base.build(p, variant)
        if built is None:
            return None
        tracer0 = world_point(built.state0, built.coupler, built.tracer)
        moved = p.copy()
        moved[:2] += np.asarray(self.attach.point) - tracer0
        return moved

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        """Candidatos cuyo marcador del personaje recorre una curva parecida a ``y``."""
        size = bbox_diag(y)
        shift = np.asarray(self.attach.point) - np.asarray(self.marker.point)
        pool: list[Candidate] = []
        for c in self.base.candidates(y + shift, n, n_candidates, max(8 * n_keep, 24), rng):
            p0 = self.place_on_attach_point(c.p0, c.variant)
            if p0 is None:
                continue
            curve = trace_curve_generic(self._builder(c.variant), p0, n)
            if curve is None:  # el personaje no puede seguir al mecanismo todo el ciclo
                continue
            pool.append(Candidate(p0, c.variant, chamfer_rms(curve, y) / size, curve))
        pool.sort(key=lambda cand: cand.rel_error)
        return pool[:n_keep]
