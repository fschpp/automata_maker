"""Reproducción cíclica de un ensamblaje: avanza la fase y resuelve en vivo.

La fase total ``t`` crece sin envolverse (θ = 2π·t en el Input Driver), así que el estado es siempre
continuo; solo ``phase = t mod 1`` se muestra al usuario. Sin dependencias de GTK.
"""

from __future__ import annotations

import math

import numpy as np

from mechchar.core.assembly import Assembly

MAX_DT = 0.1  # s; evita saltos grandes si la ventana estuvo oculta o la UI se trabó


class Playback:
    def __init__(self, asm: Assembly, state0: np.ndarray, period: float = 4.0) -> None:
        if period <= 0:
            raise ValueError("period debe ser positivo")
        self.asm = asm
        self.state = np.asarray(state0, dtype=float).copy()
        self.period = period  # segundos por ciclo
        self.t = 0.0  # fase total (ciclos recorridos)
        self.playing = True
        self.ok = True  # False si el último paso no convergió

    @property
    def phase(self) -> float:
        return self.t % 1.0

    def step_to(self, t: float) -> bool:
        """Resuelve en la fase total ``t`` partiendo del estado actual."""
        res = self.asm.solve(self.state, t)
        self.ok = res.ok
        self.t = t
        if res.ok:
            self.state = res.state
        return res.ok

    def advance(self, dt: float) -> bool:
        """Avanza ``dt`` segundos si está reproduciendo; devuelve True si el estado cambió."""
        if not self.playing:
            return False
        dt = min(max(dt, 0.0), MAX_DT)
        before = self.state
        self.step_to(self.t + dt / self.period)
        return self.state is not before

    def seek_phase(self, phase: float) -> bool:
        """Salta a una fase ∈ [0, 1] dentro de la vuelta actual."""
        return self.step_to(math.floor(self.t) + min(max(phase, 0.0), 1.0))
