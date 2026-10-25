"""Cinemática inversa del personaje: poses que llevan puntos a posiciones del mundo (M4). Sin GTK.

Un personaje articulado tiene grados de libertad y el mecanismo que lo acciona solo fija dónde
está su punto de actuación. Para saber en qué pose empieza el conjunto (y qué curva recorre en
realidad el punto de actuación cuando el marcador sigue una curva) hay que resolver la pose del
personaje dado un punto: ``solve_pose``. ``track_marker`` lo hace a lo largo de una curva, por
continuación desde la pose de reposo (así queda en la misma rama de ensamblado que el reposo).
Resolver desde el reposo alcanzó (error ~0) los 210 puntos de actuación alcanzables que se
probaron en la pierna de prueba, así que no hace falta homotopía para llegar. Ojo: llegar no
fija la pose. Un punto con 2 grados de libertad tiene varias ramas y ``solve_pose`` directo puede
devolver otra que la que sigue el marcador; para esa rama hay que partir de una pose cercana
(``track_marker`` la deja por continuación). Si un punto lejano no converge, el error lo dice.

Las restricciones del personaje se imponen con un peso alto frente a los puntos objetivo, de modo
que si un punto no es alcanzable la pose sigue siendo válida y el error queda en el punto.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from mechchar.core.assembly import Assembly, drot, world_point
from mechchar.core.character import Character, PointRef

CONSTRAINT_WEIGHT = 1e3
POSE_TOL = 1e-6  # error máximo (restricciones o puntos) para dar una pose por válida
APPROACH_STEPS = 10  # pasos para llevar el marcador de su reposo al inicio de la curva

Target = tuple[PointRef, np.ndarray]


class _Goal(NamedTuple):
    comp: int
    local: tuple[float, float]
    world: np.ndarray


def _resolve(character: Character, targets: Sequence[Target]) -> list[_Goal]:
    return [
        _Goal(
            character.index(ref.part),
            character.local_point(ref.part, ref.point),
            np.asarray(world, dtype=float),
        )
        for ref, world in targets
    ]


def _pose_error(asm: Assembly, state: np.ndarray, items: list[_Goal]) -> float:
    err = float(np.max(np.abs(asm.residual(state, 0.0)), initial=0.0))
    for g in items:
        err = max(err, float(np.linalg.norm(world_point(state, g.comp, g.local) - g.world)))
    return err


def _least_squares(
    asm: Assembly, state: np.ndarray, items: list[_Goal], goals: Sequence[np.ndarray]
) -> np.ndarray:
    n = state.size

    def residual(x: np.ndarray) -> np.ndarray:
        parts = [CONSTRAINT_WEIGHT * asm.residual(x, 0.0)]
        parts += [
            world_point(x, g.comp, g.local) - goal for g, goal in zip(items, goals, strict=True)
        ]
        return np.concatenate(parts)

    def jacobian(x: np.ndarray) -> np.ndarray:
        rows = [CONSTRAINT_WEIGHT * asm.jacobian(x, 0.0)]
        for g in items:
            block = np.zeros((2, n))
            block[0, 3 * g.comp] = block[1, 3 * g.comp + 1] = 1.0
            block[:, 3 * g.comp + 2] = drot(x[3 * g.comp + 2]) @ np.asarray(g.local)
            rows.append(block)
        return np.vstack(rows)

    return least_squares(
        residual, state, jac=jacobian, xtol=1e-12, ftol=1e-12, gtol=1e-12
    ).x


def solve_pose(
    character: Character, targets: Sequence[Target], s_init: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Pose del personaje que lleva cada punto de ``targets`` a su posición del mundo.

    Parte de ``s_init`` (por defecto el reposo) y devuelve (estado, error), con el error en
    unidades del mundo: el mayor entre el residuo de las restricciones y la distancia de cada
    punto a su objetivo. Si un objetivo no es alcanzable, la pose sigue cumpliendo las
    restricciones y el error es lo que falta; se da por válida si ``error <= POSE_TOL``."""
    asm, s_rest = character.build()
    s_init = s_rest if s_init is None else np.asarray(s_init, dtype=float)
    items = _resolve(character, targets)
    state = _least_squares(asm, s_init, items, [g.world for g in items])
    return state, _pose_error(asm, state, items)


def track_marker(
    character: Character, marker: PointRef, curve: np.ndarray
) -> tuple[np.ndarray, float]:
    """Poses del personaje con su ``marker`` sobre cada punto de ``curve`` (n, 2), por
    continuación desde el reposo: (estados (n, n_vars), peor distancia del marcador a la curva).

    Si parte de la curva no es alcanzable las poses siguen siendo válidas y se acercan lo más
    posible; el segundo valor mide cuánto falta en el peor punto."""
    s_rest = character.build()[1]
    comp = character.index(marker.part)
    local = character.local_point(marker.part, marker.point)
    state = s_rest
    start = world_point(s_rest, comp, local)
    for a in np.linspace(0.0, 1.0, APPROACH_STEPS + 1)[1:]:
        state, _ = solve_pose(character, [(marker, start * (1.0 - a) + curve[0] * a)], state)
    states = np.empty((len(curve), state.size))
    worst = 0.0
    for k, goal in enumerate(curve):
        state, _ = solve_pose(character, [(marker, goal)], state)
        states[k] = state
        worst = max(worst, float(np.linalg.norm(world_point(state, comp, local) - goal)))
    return states, worst


def point_curve(character: Character, ref: PointRef, states: np.ndarray) -> np.ndarray:
    """Posición del mundo de un punto del personaje en cada estado de ``states``."""
    comp = character.index(ref.part)
    local = character.local_point(ref.part, ref.point)
    return np.array([world_point(s, comp, local) for s in states])
