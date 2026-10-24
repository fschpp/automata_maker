"""Varios mecanismos sobre un mismo personaje, con un único motor y un desfase por mecanismo (M4).

Todos los mecanismos comparten la fase global ``t`` del Input Driver (el único motor). El control
de desfase del video («Phase Shift Control Box», un slider por sub-ensamblaje) se materializa
como un corrimiento del offset del ``Drive`` de cada mecanismo: con desfase δ (en fracciones de
ciclo) el mecanismo hace en ``t`` lo que habría hecho en ``t + δ``. Sin GTK.

Cambiar un desfase en vivo mueve la pose de ese mecanismo, así que ``MultiRig.set_phase`` lo hace
por continuación (pasos chicos desde la pose actual) para no saltar a otra rama de ensamblado.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mechchar.core.assembly import TWO_PI, Analysis, Assembly, Drive, Pin, world_point
from mechchar.core.character import Character, PointRef

REST_TOL = 1e-6  # un pasador mayor que esto en reposo es infactible
MAX_PHASE_STEP = 0.02  # fracción de ciclo por paso de continuación


@dataclass
class SubSpec:
    """Un mecanismo para unir al personaje: el armado (``Built``), el mecanismo para dibujarlo,
    el punto de actuación del personaje y el marcador cuya curva se controla."""

    name: str
    built: Any  # optim.fitcore.Built (asm, state0, coupler, tracer)
    mechanism: Any  # FourBar | GearedFiveBar
    attach: PointRef
    marker: PointRef


@dataclass
class SubAssembly:
    name: str
    mechanism: Any
    attach: PointRef
    marker: PointRef
    first: int  # índice de su primer componente en el ensamblaje conjunto
    n_comp: int
    drive: Drive  # el Input Driver de este mecanismo (copia dentro del ensamblaje conjunto)
    offset0: float  # offset original del Drive, sin desfase
    marker_part: int
    marker_local: tuple[float, float]
    phase: float = 0.0  # desfase actual, en fracciones de ciclo

    def apply_phase(self, phase: float) -> None:
        self.phase = phase
        self.drive.offset = self.offset0 + self.drive.ratio * TWO_PI * phase


class MultiRig:
    """Personaje + N mecanismos accionados por el mismo motor, cada uno con su desfase."""

    def __init__(self, character: Character, specs: Sequence[SubSpec]) -> None:
        if not specs:
            raise ValueError("hace falta al menos un mecanismo")
        names = [sp.name for sp in specs]
        if len(set(names)) != len(names):
            raise ValueError("hay mecanismos con el mismo nombre")
        self.character = character
        self.n_char = len(character.parts)
        asm, s_char = character.build()
        states = [s_char]
        self.subs: list[SubAssembly] = []
        for sp in specs:
            n_before = len(asm.constraints)
            first = asm.append(sp.built.asm)
            drives = [c for c in asm.constraints[n_before:] if isinstance(c, Drive)]
            if len(drives) != 1:
                raise ValueError(f"{sp.name}: el mecanismo debe tener exactamente un Input Driver")
            asm.add(
                Pin(
                    sp.built.coupler + first, sp.built.tracer,
                    character.index(sp.attach.part),
                    character.local_point(sp.attach.part, sp.attach.point),
                )
            )
            states.append(np.asarray(sp.built.state0, dtype=float))
            self.subs.append(
                SubAssembly(
                    sp.name, sp.mechanism, sp.attach, sp.marker, first,
                    len(sp.built.asm.components), drives[0], drives[0].offset,
                    character.index(sp.marker.part),
                    character.local_point(sp.marker.part, sp.marker.point),
                )
            )
        self.asm: Assembly = asm
        self.state0 = np.concatenate(states)
        worst = float(np.max(np.abs(asm.residual(self.state0, 0.0))))
        if worst > REST_TOL:
            raise ValueError(
                f"los pasadores no son factibles en reposo (residuo {worst:.2e}): el marcador "
                "de cada mecanismo debe caer sobre su punto de actuación en t = 0"
            )

    # --- consulta ---------------------------------------------------------------------------
    @property
    def phases(self) -> list[float]:
        return [sub.phase for sub in self.subs]

    def analyze(self, state: np.ndarray | None = None, t: float = 0.0) -> Analysis:
        """Grados de libertad del conjunto: 0 y sin redundancia si cada pierna tiene su motor."""
        return self.asm.analyze(self.state0 if state is None else state, t)

    def mech_state(self, state: np.ndarray, k: int) -> np.ndarray:
        sub = self.subs[k]
        lo = 3 * sub.first
        return state[lo : lo + 3 * sub.n_comp]

    def marker_world(self, state: np.ndarray, k: int) -> np.ndarray:
        sub = self.subs[k]
        return world_point(state, sub.marker_part, sub.marker_local)

    def part_polygons(self, state: np.ndarray) -> list[np.ndarray]:
        return [self.character.part_polygon(state, i) for i in range(self.n_char)]

    # --- desfase ----------------------------------------------------------------------------
    def set_phase(
        self, k: int, phase: float, state: np.ndarray, t: float = 0.0
    ) -> tuple[np.ndarray, bool]:
        """Cambia el desfase del mecanismo ``k`` y devuelve la nueva pose en la fase global ``t``.

        Avanza por continuación en pasos de a lo sumo ``MAX_PHASE_STEP`` ciclos. Si algún paso no
        converge, deja el desfase y la pose como estaban y devuelve ``(state, False)``."""
        sub = self.subs[k]
        start = sub.phase
        n = max(1, math.ceil(abs(phase - start) / MAX_PHASE_STEP))
        s = np.asarray(state, dtype=float)
        for i in range(1, n + 1):
            sub.apply_phase(start + (phase - start) * i / n)
            res = self.asm.solve(s, t)
            if not res.ok:
                sub.apply_phase(start)
                return np.asarray(state, dtype=float), False
            s = res.state
        return s, True
