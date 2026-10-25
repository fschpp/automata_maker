"""Ajuste de un mecanismo unido a un personaje (M4): la curva objetivo es la de un marcador del
personaje, no la del mecanismo, y ambos se simulan juntos.

``CharacterFamily`` envuelve a una familia de mecanismos (cuadrilátero, 5-bar...) y cumple su mismo
protocolo, así que ``fitcore.fit`` y el gradiente por función implícita funcionan sin cambios: el
estado incluye los grados de libertad del personaje y el marcador es el del personaje.

Los candidatos iniciales se buscan para la curva que recorre el punto de actuación cuando el
marcador sigue la curva objetivo (cinemática inversa, ``core/kinematics.py``): en un personaje con
grados de libertad no se mueve como el marcador. La pose inicial del personaje se resuelve también
por cinemática inversa, con el punto de actuación sobre el marcador del mecanismo en t = 0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mechchar.core.assembly import world_point
from mechchar.core.character import Character, PointRef, RiggedMechanism, combine
from mechchar.core.kinematics import (
    POSE_TOL,
    REACH_TOL_REL,
    point_curve,
    solve_pose,
    track_marker,
)
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.fitcore import (
    Built,
    Candidate,
    Family,
    LocalSample,
    Variant,
    trace_curve_generic,
)

HINT_RADIUS = 0.05  # distancia máxima para reusar la última pose como arranque


class CharacterFamily:
    def __init__(
        self, base: Family, character: Character, attach: PointRef, marker: PointRef
    ) -> None:
        self.base, self.character, self.attach, self.marker = base, character, attach, marker
        # nombre distinto del de la base: así no se le aplica la base de movimientos de la base
        self.name = f"{base.name}+{character.name}"
        self.n_params = base.n_params
        self.n_compact = base.n_compact
        # última pose por IK y su destino: arranque tibio (~1 ms en vez de ~40 ms desde el reposo)
        self._hint: tuple[np.ndarray, np.ndarray] | None = None
        self.reach_error = 0.0  # peor distancia del marcador a la última curva pedida (IK)
        self._bank: tuple[np.ndarray, np.ndarray] | None = (
            None  # (curva del punto de actuación, poses)
        )

    # --- protocolo de familia -------------------------------------------------------------
    def prepare(self, y: np.ndarray) -> float:
        """Resuelve las poses del personaje con su marcador sobre la curva ``y`` y las guarda como
        banco de arranque para ``initial_pose``. Devuelve el peor error de alcance.

        Hace falta porque la posición del punto de actuación no determina la pose: con 2 grados de
        libertad hay varias ramas, y la que sigue el marcador sobre ``y`` es la que importa."""
        states, self.reach_error = track_marker(self.character, self.marker, y)
        self._bank = (point_curve(self.character, self.attach, states), states)
        self._hint = None
        return self.reach_error

    def _try_pose(self, target: np.ndarray, start: np.ndarray | None) -> np.ndarray | None:
        state, err = solve_pose(self.character, [(self.attach, target)], start)
        if err > POSE_TOL:
            return None
        self._hint = (state, target)
        return state

    def initial_pose(self, target: np.ndarray) -> np.ndarray | None:
        """Pose del personaje con su punto de actuación sobre ``target``, o None si no alcanza.

        Arranca de la última pose si el destino casi no se movió (al refinar o perturbar un
        parámetro: ~1 ms contra ~40 ms), si no de la pose del banco más cercana, y si no hay banco
        llega por continuación desde el reposo (solo la rama del reposo)."""
        target = np.asarray(target, dtype=float)
        if self._hint is not None and np.linalg.norm(target - self._hint[1]) < HINT_RADIUS:
            state = self._try_pose(target, self._hint[0])
            if state is not None:
                return state
        if self._bank is not None:
            curve, states = self._bank
            k = int(np.argmin(np.linalg.norm(curve - target, axis=1)))
            state = self._try_pose(target, states[k])
            if state is not None:
                return state
        states_rest, err = track_marker(self.character, self.attach, target[None, :])
        if err > POSE_TOL:
            return None
        self._hint = (states_rest[0], target)
        return states_rest[0]

    def pose_for(
        self, p: np.ndarray, variant: Variant
    ) -> dict[str, tuple[float, float, float]] | None:
        """Pose inicial (t = 0) de las piezas móviles del personaje con este mecanismo, por nombre.

        Es la misma que usa ``build``; sirve para armar el personaje en otro conjunto (bípedo)
        en la misma rama que se midió al ajustar. None si el mecanismo no alcanza su punto."""
        built = self.base.build(p, variant)
        if built is None:
            return None
        pose = self.initial_pose(world_point(built.state0, built.coupler, built.tracer))
        if pose is None:
            return None
        return {
            part.name: (float(pose[3 * i]), float(pose[3 * i + 1]), float(pose[3 * i + 2]))
            for i, part in enumerate(self.character.parts)
            if not part.fixed
        }

    def build_unposed(self, p: np.ndarray, variant: Variant) -> Built | None:
        """Como ``build`` pero con el personaje en reposo, sin cinemática inversa: para armados
        de los que solo se usan las ecuaciones (el Jacobiano por perturbación)."""
        built = self.base.build(p, variant)
        if built is None:
            return None
        return combine(self.character, built, self.attach, self.marker)

    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        built = self.base.build(p, variant)
        if built is None:
            return None
        tracer0 = world_point(built.state0, built.coupler, built.tracer)
        # sin pose alcanzable queda el reposo y el solver decide en t = 0 si hay solución
        pose = self.initial_pose(tracer0)
        return combine(self.character, built, self.attach, self.marker, pose)

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
            built.asm,
            built.state0,
            self.character,
            len(self.character.parts),
            base,
            built.coupler,
            built.tracer,
        )

    def _builder(self, variant: Variant) -> Callable[[np.ndarray], Built | None]:
        def build(q: np.ndarray) -> Built | None:
            return self.build(q, variant)

        return build

    def place_on_attach_point(self, p: np.ndarray, variant: Variant) -> np.ndarray | None:
        """Traslada el mecanismo para que su marcador caiga sobre el punto de actuación en reposo
        (el pasador es entonces factible sin mover al personaje)."""
        built = self.base.build(p, variant)
        if built is None:
            return None
        tracer0 = world_point(built.state0, built.coupler, built.tracer)
        moved = p.copy()
        moved[:2] += np.asarray(self.attach.point) - tracer0
        return moved

    def _pool(
        self,
        y: np.ndarray,
        source: np.ndarray,
        size: float,
        n: int,
        n_candidates: int,
        n_pool: int,
        rng: np.random.Generator,
        *,
        place: bool,
    ) -> list[Candidate]:
        pool: list[Candidate] = []
        for c in self.base.candidates(source, n, n_candidates, n_pool, rng):
            p0 = self.place_on_attach_point(c.p0, c.variant) if place else c.p0
            if p0 is None:
                continue
            curve = trace_curve_generic(self._builder(c.variant), p0, n)
            if curve is None:  # el personaje no puede seguir al mecanismo todo el ciclo
                continue
            pool.append(Candidate(p0, c.variant, chamfer_rms(curve, y) / size, curve))
        return pool

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        """Candidatos cuyo marcador del personaje recorre una curva parecida a ``y``.

        El mecanismo se ajusta a la curva del punto de actuación (cinemática inversa del marcador
        sobre ``y``) y luego se mide el error con el personaje dentro. Si parte de ``y`` no es
        alcanzable (``reach_error`` guarda cuánto falta) esa curva queda al borde del espacio
        alcanzable y suele no servir, así que se suma el método anterior: ``y`` trasladada por el
        desplazamiento entre el punto de actuación y el marcador en reposo."""
        size = bbox_diag(y)
        n_pool = max(8 * n_keep, 24)
        self.prepare(y)
        assert self._bank is not None
        attach_curve = self._bank[0]
        pool = self._pool(y, attach_curve, size, n, n_candidates, n_pool, rng, place=False)
        if self.reach_error > REACH_TOL_REL * size or not pool:
            shift = np.asarray(self.attach.point) - np.asarray(self.marker.point)
            pool += self._pool(y, y + shift, size, n, n_candidates, n_pool, rng, place=True)
        pool.sort(key=lambda cand: cand.rel_error)
        return pool[:n_keep]
