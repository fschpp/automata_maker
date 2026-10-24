"""Ensamblajes 2D: piezas rígidas (x, y, θ), conexiones y solver con continuación por fase.

El estado es un vector plano ``s`` con 3 grados de libertad por componente:
``s[3*i:3*i+3] = (x_i, y_i, θ_i)``. Cada conexión aporta ecuaciones C(s, t) = 0 y su Jacobiano
analítico; el ensamblaje resuelve min ½‖C‖² con ``least_squares`` partiendo de la solución anterior
(continuación por fase), lo que mantiene la misma rama de ensamblado.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares

Vec = tuple[float, float]
TWO_PI = 2.0 * math.pi


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def drot(theta: float) -> np.ndarray:
    """Derivada de ``rot(theta)`` respecto de theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[-s, -c], [c, -s]])


def world_point(state: np.ndarray, comp: int, local: Vec) -> np.ndarray:
    """Posición en el mundo de un punto dado en coordenadas locales de ``comp``."""
    i = 3 * comp
    c, s = math.cos(state[i + 2]), math.sin(state[i + 2])
    lx, ly = local
    return np.array([state[i] + c * lx - s * ly, state[i + 1] + s * lx + c * ly])


# --- conexiones --------------------------------------------------------------------------------


class Constraint(ABC):
    """Una conexión: ``n_eq`` ecuaciones sobre el estado y la fase t ∈ [0, 1]."""

    n_eq: ClassVar[int]

    @abstractmethod
    def residual(self, s: np.ndarray, t: float) -> np.ndarray: ...

    @abstractmethod
    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        """Acumula (+=) las derivadas en ``out`` (forma n_eq x n_vars, inicializada en 0)."""

    def shifted(self, k: int) -> Constraint:
        """Copia con todos los índices de componente desplazados ``k`` posiciones."""
        moved = {
            f.name: getattr(self, f.name) + k
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
            if f.name in ("a", "b", "comp")
        }
        return dataclasses.replace(self, **moved)  # type: ignore[type-var]


@dataclass
class FixedState(Constraint):
    """Fija una pieza a una pose (típicamente el chasis)."""

    comp: int
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    n_eq: ClassVar[int] = 3

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        i = 3 * self.comp
        return np.array([s[i] - self.x, s[i + 1] - self.y, s[i + 2] - self.theta])

    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        i = 3 * self.comp
        out[0, i] += 1.0
        out[1, i + 1] += 1.0
        out[2, i + 2] += 1.0


@dataclass
class Pin(Constraint):
    """Pasador: un punto local de ``a`` coincide con un punto local de ``b``."""

    a: int
    pa: Vec
    b: int
    pb: Vec
    n_eq: ClassVar[int] = 2

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        return world_point(s, self.a, self.pa) - world_point(s, self.b, self.pb)

    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        for comp, local, sign in ((self.a, self.pa, 1.0), (self.b, self.pb, -1.0)):
            i = 3 * comp
            out[0, i] += sign
            out[1, i + 1] += sign
            out[:, i + 2] += sign * (drot(s[i + 2]) @ np.asarray(local, dtype=float))


@dataclass
class PointOnLine(Constraint):
    """Un punto local de ``a`` queda sobre la recta de ``b`` que pasa por ``q`` con dirección ``u``
    (ambos en coordenadas locales de ``b``). Sirve para correderas."""

    a: int
    pa: Vec
    b: int
    q: Vec
    u: Vec
    n_eq: ClassVar[int] = 1

    def _normal(self) -> np.ndarray:
        ux, uy = self.u
        norm = math.hypot(ux, uy)
        return np.array([-uy / norm, ux / norm])

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        nw = rot(s[3 * self.b + 2]) @ self._normal()
        d = world_point(s, self.a, self.pa) - world_point(s, self.b, self.q)
        return np.array([nw @ d])

    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        ia, ib = 3 * self.a, 3 * self.b
        n_local = self._normal()
        nw = rot(s[ib + 2]) @ n_local
        d = world_point(s, self.a, self.pa) - world_point(s, self.b, self.q)
        out[0, ia : ia + 2] += nw
        out[0, ia + 2] += nw @ (drot(s[ia + 2]) @ np.asarray(self.pa, dtype=float))
        out[0, ib : ib + 2] -= nw
        db = drot(s[ib + 2])
        out[0, ib + 2] += (db @ n_local) @ d - nw @ (db @ np.asarray(self.q, dtype=float))


@dataclass
class Phase(Constraint):
    """Engranes: θ_b = ratio·θ_a + offset (ratio = -r_a/r_b para un par externo)."""

    a: int
    b: int
    ratio: float
    offset: float = 0.0
    n_eq: ClassVar[int] = 1

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        return np.array([s[3 * self.b + 2] - self.ratio * s[3 * self.a + 2] - self.offset])

    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        out[0, 3 * self.b + 2] += 1.0
        out[0, 3 * self.a + 2] -= self.ratio


@dataclass
class Drive(Constraint):
    """Input Driver: θ = ratio·2π·t + offset, con t ∈ [0, 1] la fase de un ciclo."""

    comp: int
    ratio: float = 1.0
    offset: float = 0.0
    n_eq: ClassVar[int] = 1

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        return np.array([s[3 * self.comp + 2] - (self.ratio * TWO_PI * t + self.offset)])

    def jacobian(self, s: np.ndarray, t: float, out: np.ndarray) -> None:
        out[0, 3 * self.comp + 2] += 1.0


# --- ensamblaje --------------------------------------------------------------------------------


@dataclass
class Component:
    name: str
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    z: float = 0.0  # plano de profundidad; la simulación es 2D por plano


@dataclass
class SolveResult:
    state: np.ndarray
    residual_norm: float
    ok: bool


@dataclass
class Analysis:
    """Resultado del chequeo de restricciones en una configuración."""

    rank: int
    dof: int  # n_vars - rank: > 0 subrestringido
    redundant: int  # n_eq - rank: > 0 hay restricciones redundantes (o inconsistentes)


@dataclass
class CycleResult:
    states: np.ndarray  # (n_steps, n_vars) para t = k / n_steps
    ok: bool
    reason: str = ""
    max_jump: float = 0.0


@dataclass
class Assembly:
    components: list[Component] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    def add_component(
        self, name: str, x: float = 0.0, y: float = 0.0, theta: float = 0.0, z: float = 0.0
    ) -> int:
        self.components.append(Component(name, x, y, theta, z))
        return len(self.components) - 1

    def append(self, other: Assembly) -> int:
        """Agrega los componentes y conexiones de ``other`` (reindexados) y devuelve el índice
        que tiene su primer componente en este ensamblaje."""
        offset = len(self.components)
        self.components.extend(dataclasses.replace(c) for c in other.components)
        self.constraints.extend(c.shifted(offset) for c in other.constraints)
        return offset

    def add(self, constraint: Constraint) -> Constraint:
        self.constraints.append(constraint)
        return constraint

    @property
    def n_vars(self) -> int:
        return 3 * len(self.components)

    @property
    def n_eq(self) -> int:
        return sum(c.n_eq for c in self.constraints)

    def initial_state(self) -> np.ndarray:
        return np.array([v for c in self.components for v in (c.x, c.y, c.theta)], dtype=float)

    def residual(self, s: np.ndarray, t: float) -> np.ndarray:
        if not self.constraints:
            return np.zeros(0)
        return np.concatenate([c.residual(s, t) for c in self.constraints])

    def jacobian(self, s: np.ndarray, t: float) -> np.ndarray:
        jac = np.zeros((self.n_eq, self.n_vars))
        row = 0
        for c in self.constraints:
            c.jacobian(s, t, jac[row : row + c.n_eq])
            row += c.n_eq
        return jac

    def _newton(
        self, s0: np.ndarray, t: float, tol: float, max_iter: int = 30
    ) -> SolveResult | None:
        """Newton amortiguado (paso de mínimos cuadrados + búsqueda en línea) con el Jacobiano
        analítico. Devuelve None si no converge; ``solve`` recurre entonces a ``least_squares``."""
        s = np.array(s0, dtype=float)
        r = self.residual(s, t)
        nr = float(np.linalg.norm(r))
        for _ in range(max_iter):
            if nr < tol:
                return SolveResult(s, nr, True)
            step = np.linalg.lstsq(self.jacobian(s, t), -r, rcond=None)[0]
            lam = 1.0
            while lam >= 1e-3:
                s_new = s + lam * step
                r_new = self.residual(s_new, t)
                nr_new = float(np.linalg.norm(r_new))
                if nr_new < (1.0 - 1e-4 * lam) * nr:
                    break
                lam *= 0.5
            else:
                return None
            s, r, nr = s_new, r_new, nr_new
        return SolveResult(s, nr, True) if nr < tol else None

    def solve(self, s0: np.ndarray, t: float, tol: float = 1e-9) -> SolveResult:
        """Minimiza ½‖C(s, t)‖² partiendo de ``s0``. ``ok`` si el residuo final es < tol.

        Primero Newton con el Jacobiano analítico (rápido cerca de la solución, como en la
        continuación por fase); si no converge, ``least_squares`` (trf)."""
        fast = self._newton(s0, t, tol)
        if fast is not None:
            return fast
        res = least_squares(
            lambda s: self.residual(s, t),
            s0,
            jac=lambda s: self.jacobian(s, t),
            method="trf",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        norm = float(np.linalg.norm(res.fun))
        return SolveResult(res.x, norm, norm < tol)

    def analyze(self, s: np.ndarray, t: float = 0.0, rank_tol: float = 1e-9) -> Analysis:
        """Rango del Jacobiano: grados de libertad restantes y restricciones redundantes."""
        sv = np.linalg.svd(self.jacobian(s, t), compute_uv=False)
        rank = int(np.sum(sv > rank_tol * sv[0])) if sv.size else 0
        return Analysis(rank=rank, dof=self.n_vars - rank, redundant=self.n_eq - rank)

    def world_point(self, s: np.ndarray, comp: int, local: Vec) -> np.ndarray:
        return world_point(s, comp, local)

    def _wrapped_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        idx = np.arange(2, self.n_vars, 3)
        d[idx] = (d[idx] + math.pi) % TWO_PI - math.pi
        return d

    def trace_cycle(
        self,
        s0: np.ndarray,
        n_steps: int = 360,
        *,
        max_jump: float = 0.5,
        tol: float = 1e-9,
        closure_tol: float = 1e-6,
    ) -> CycleResult:
        """Recorre un ciclo t = 0 … 1 con continuación por fase.

        Se rechaza el mecanismo si el solver no converge en algún paso, si el estado salta más de
        ``max_jump`` entre fases consecutivas (cambio de rama) o si al cerrar el ciclo no vuelve
        a la configuración inicial (salvo múltiplos de 2π en los ángulos).
        """
        states: list[np.ndarray] = []
        s = np.asarray(s0, dtype=float)
        worst = 0.0
        for k in range(n_steps):
            t = k / n_steps
            r = self.solve(s, t, tol)
            if not r.ok:
                return CycleResult(
                    np.array(states), False, f"sin convergencia en t={t:.4f}", worst
                )
            if k > 0:
                jump = float(np.max(np.abs(r.state - s)))
                worst = max(worst, jump)
                if jump > max_jump:
                    return CycleResult(
                        np.array(states), False, f"salto de rama en t={t:.4f}", worst
                    )
            states.append(r.state)
            s = r.state
        r = self.solve(s, 1.0, tol)
        if not r.ok:
            return CycleResult(np.array(states), False, "sin convergencia en t=1", worst)
        if np.max(np.abs(self._wrapped_diff(r.state, states[0]))) > closure_tol:
            return CycleResult(np.array(states), False, "el ciclo no cierra", worst)
        return CycleResult(np.array(states), True, "", worst)
