"""Núcleo común del ajuste de mecanismos a una curva cerrada objetivo (M2).

Cada familia de mecanismos (cuadrilátero, 5-bar con engranajes, …) implementa el protocolo
``Family``: sabe armar su ensamblaje a partir de un vector de parámetros ``p``, proponer candidatos
iniciales y penalizar mecanismos poco compactos. Todo lo demás es común:

* gradiente de la curva del marcador por el Teorema de la Función Implícita,
  ∂s/∂p = −(∂C/∂s)⁺ ∂C/∂p, con ∂C/∂s analítico y ∂C/∂p por diferencias finitas;
* objetivo sin correspondencia punto a punto (distancias punto-a-curva en los dos sentidos);
* refinamiento con ``least_squares`` y búsqueda entre familias.

El objetivo no usa correspondencia porque el marcador no recorre su curva a velocidad uniforme:
exigir x(p, k/n) = x̂_k deja un error que ningún optimizador puede bajar (el paper también descarta
el timing en la métrica). Se minimizan distancias punto-a-curva con residuo escalar a lo largo de
la normal del segmento más cercano, lo que deja libre el deslizamiento tangencial.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares

from mechchar.core.assembly import Assembly, drot, world_point
from mechchar.core.mechanisms import Mechanism
from mechchar.optim.curves import bbox_diag, chamfer_rms, resample_closed

# Ningún eslabón ni el marcador puede superar este múltiplo del tamaño de la curva (compacidad).
MAX_LINK_RATIO = 1.5
TWO_PI = 2.0 * math.pi

ProgressFn = Callable[[str, np.ndarray], None]


@dataclass
class Built:
    """Un mecanismo armado en t = 0, listo para simular."""

    asm: Assembly
    state0: np.ndarray
    coupler: int  # pieza que lleva el marcador
    tracer: tuple[float, float]  # marcador, en coordenadas locales de ``coupler``


@dataclass(frozen=True)
class Variant:
    """Elección discreta dentro de una familia (no se optimiza de forma continua)."""

    branch: int = 1  # ensamblado del lazo cerrado: +1 / -1
    ratio: float = 0.0  # relación de velocidades de los engranajes (0 si la familia no tiene)


@dataclass
class Candidate:
    p0: np.ndarray
    variant: Variant
    rel_error: float  # error chamfer relativo al tamaño de la curva, antes de refinar
    curve: np.ndarray  # curva del candidato ya alineada con el objetivo (n, 2)


class Family(Protocol):
    name: str
    n_params: int
    n_compact: int  # cantidad de residuos de compacidad

    def build(self, p: np.ndarray, variant: Variant) -> Built | None: ...

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]: ...

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> Mechanism | None: ...


# --- curva del marcador y su gradiente -----------------------------------------------------------


def trace_curve_generic(
    build_fn: Callable[[np.ndarray], Built | None], p: np.ndarray, n: int
) -> np.ndarray | None:
    """Curva del marcador en n fases t = k/n, o None si el mecanismo no completa el ciclo."""
    b = build_fn(p)
    if b is None:
        return None
    res = b.asm.trace_cycle(b.state0, n_steps=n, max_jump=1.0)
    if not res.ok:
        return None
    return np.array([world_point(s, b.coupler, b.tracer) for s in res.states])


def curve_and_jacobian_generic(
    build_fn: Callable[[np.ndarray], Built | None], p: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Curva x(p) (n, 2) y su Jacobiano dx/dp (n, 2, P), por función implícita."""
    base = build_fn(p)
    if base is None:
        return None
    res = base.asm.trace_cycle(base.state0, n_steps=n, max_jump=1.0)
    if not res.ok:
        return None

    n_params = len(p)
    steps = 1e-6 * (1.0 + np.abs(p))
    plus: list[Built] = []
    minus: list[Built] = []
    for j in range(n_params):
        pp, pm = p.copy(), p.copy()
        pp[j] += steps[j]
        pm[j] -= steps[j]
        bp, bm = build_fn(pp), build_fn(pm)
        if bp is None or bm is None:
            return None
        plus.append(bp)
        minus.append(bm)

    x = np.zeros((n, 2))
    dx = np.zeros((n, 2, n_params))
    i = 3 * base.coupler
    for k in range(n):
        t, s = k / n, res.states[k]
        jac_s = base.asm.jacobian(s, t)
        c_p = np.empty((jac_s.shape[0], n_params))
        e_p = np.empty((2, n_params))
        for j in range(n_params):
            c_p[:, j] = (plus[j].asm.residual(s, t) - minus[j].asm.residual(s, t)) / (2 * steps[j])
            e_p[:, j] = (
                world_point(s, base.coupler, plus[j].tracer)
                - world_point(s, base.coupler, minus[j].tracer)
            ) / (2 * steps[j])
        ds_dp = -np.linalg.lstsq(jac_s, c_p, rcond=None)[0]
        e_s = np.zeros((2, jac_s.shape[1]))
        e_s[0, i] = 1.0
        e_s[1, i + 1] = 1.0
        e_s[:, i + 2] = drot(s[i + 2]) @ np.asarray(base.tracer)
        x[k] = world_point(s, base.coupler, base.tracer)
        dx[k] = e_s @ ds_dp + e_p
    return x, dx


def curve_and_jacobian_finite_differences(
    build_fn: Callable[[np.ndarray], Built | None], p: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Referencia: el mismo Jacobiano por diferencias finitas completas (re-simula el ciclo 2P
    veces). Solo para comparar velocidad y exactitud con la función implícita."""
    x = trace_curve_generic(build_fn, p, n)
    if x is None:
        return None
    steps = 1e-6 * (1.0 + np.abs(p))
    dx = np.zeros((n, 2, len(p)))
    for j in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[j] += steps[j]
        pm[j] -= steps[j]
        xp, xm = trace_curve_generic(build_fn, pp, n), trace_curve_generic(build_fn, pm, n)
        if xp is None or xm is None:
            return None
        dx[:, :, j] = (xp - xm) / (2 * steps[j])
    return x, dx


# --- objetivo sin correspondencia ----------------------------------------------------------------


def closest_on_polyline(
    q: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmento más cercano de la poligonal cerrada ``v`` (M, 2) para cada punto de ``q`` (m, 2).

    Devuelve (índice del segmento, parámetro λ ∈ [0, 1] del pie, normal unitaria del segmento)."""
    a, ab = v, np.roll(v, -1, axis=0) - v
    ab2 = np.maximum((ab**2).sum(1), 1e-18)
    lam = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)
    foot = a[None] + lam[..., None] * ab[None]
    idx = ((q[:, None, :] - foot) ** 2).sum(-1).argmin(1)
    rows = np.arange(len(q))
    seg = ab[idx]
    norm = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-12)
    normal = np.column_stack([-seg[:, 1], seg[:, 0]]) / norm[:, None]
    return idx, lam[rows, idx], normal


def distance_residuals(
    x: np.ndarray, dx: np.ndarray, y: np.ndarray, y_dense: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Residuos y Jacobiano del objetivo sin correspondencia (n + len(y) residuos).

    x (n, 2) curva del mecanismo, dx (n, 2, P) su Jacobiano, y (m, 2) muestras del objetivo y
    y_dense (M, 2) su poligonal densa."""
    n, n_params = len(x), dx.shape[2]
    # mecanismo -> objetivo
    idx_f, lam_f, nrm_f = closest_on_polyline(x, y_dense)
    foot = y_dense[idx_f] + lam_f[:, None] * (np.roll(y_dense, -1, axis=0)[idx_f] - y_dense[idx_f])
    r_f = (nrm_f * (x - foot)).sum(1)
    j_f = np.einsum("ki,kip->kp", nrm_f, dx)
    # objetivo -> mecanismo
    idx_b, lam_b, nrm_b = closest_on_polyline(y, x)
    nxt = (idx_b + 1) % n
    foot_b = (1 - lam_b)[:, None] * x[idx_b] + lam_b[:, None] * x[nxt]
    r_b = (nrm_b * (y - foot_b)).sum(1)
    d_foot = (1 - lam_b)[:, None, None] * dx[idx_b] + lam_b[:, None, None] * dx[nxt]
    j_b = -np.einsum("mi,mip->mp", nrm_b, d_foot)
    return np.concatenate([r_f, r_b]), np.vstack([j_f, j_b]).reshape(-1, n_params)


# --- candidatos iniciales: alineación de curvas locales con el objetivo ---------------------------


def principal_angle(curves: np.ndarray) -> np.ndarray:
    """Ángulo del eje principal de cada curva (K, n, 2) -> (K,)."""
    d = curves - curves.mean(1, keepdims=True)
    cxx, cyy = (d[..., 0] ** 2).sum(1), (d[..., 1] ** 2).sum(1)
    cxy = (d[..., 0] * d[..., 1]).sum(1)
    return 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)


@dataclass
class Aligned:
    index: int  # posición dentro de las curvas locales recibidas
    scale: float  # factor de escala (unidades locales -> mundo); d_mundo = scale (d local = 1)
    phi: float  # ángulo de la bancada en el mundo
    origin: np.ndarray  # pivote de la bancada en el mundo
    cost: float  # error chamfer (en unidades del mundo)
    curve: np.ndarray  # curva alineada (n, 2)


def rank_aligned(
    local: np.ndarray, extent: np.ndarray, y: np.ndarray, n_keep: int
) -> list[Aligned]:
    """Alinea curvas locales (K, n, 2) con el objetivo ``y`` por centro, escala y ejes principales.

    ``extent`` (K,) es la mayor dimensión del mecanismo en unidades locales (para el filtro de
    compacidad). Se prueban los dos sentidos del eje principal y se devuelven hasta ``n_keep``
    mejores por sentido, sin ordenar entre sentidos."""
    center, size = y.mean(0), bbox_diag(y)
    ang_y = float(principal_angle(y[None])[0])
    c0 = local.mean(1, keepdims=True)
    diag = np.hypot(np.ptp(local[..., 0], axis=1), np.ptp(local[..., 1], axis=1))
    f = size / diag
    compact_idx = np.where(f * extent <= MAX_LINK_RATIO * size)[0]
    if compact_idx.size == 0:
        return []
    loc, c0, f = local[compact_idx], c0[compact_idx], f[compact_idx]
    ang_l = principal_angle(loc)
    out: list[Aligned] = []
    for flip in (0.0, math.pi):
        phi = ang_y - ang_l + flip
        cph, sph = np.cos(phi)[:, None], np.sin(phi)[:, None]
        dxy = (loc - c0) * f[:, None, None]
        pts = (
            np.stack(
                [cph * dxy[..., 0] - sph * dxy[..., 1], sph * dxy[..., 0] + cph * dxy[..., 1]],
                axis=-1,
            )
            + center
        )
        d2 = ((pts[:, :, None, :] - y[None, None, :, :]) ** 2).sum(-1)
        cost = np.sqrt(0.5 * (d2.min(2).mean(1) + d2.min(1).mean(1)))
        for idx in np.argsort(cost)[:n_keep]:
            fc = f[idx] * c0[idx, 0]  # centroide local escalado
            c_, s_ = cph[idx, 0], sph[idx, 0]
            origin = center - np.array([c_ * fc[0] - s_ * fc[1], s_ * fc[0] + c_ * fc[1]])
            out.append(
                Aligned(
                    int(compact_idx[idx]), float(f[idx]), float(phi[idx]), origin,
                    float(cost[idx]), pts[idx],
                )
            )
    return out


# --- refinamiento y búsqueda ---------------------------------------------------------------------


def refine(
    family: Family,
    p0: np.ndarray,
    variant: Variant,
    y: np.ndarray,
    y_dense: np.ndarray,
    n: int,
    max_nfev: int,
    on_progress: ProgressFn | None = None,
    label: str = "",
) -> np.ndarray:
    """Refina ``p0`` con mínimos cuadrados (trf) y el Jacobiano por función implícita."""
    size = bbox_diag(y)
    n_res = n + len(y) + family.n_compact
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def build_fn(q: np.ndarray) -> Built | None:
        return family.build(q, variant)

    def evaluate(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = p.tobytes()
        if key not in cache:
            cache.clear()
            out = curve_and_jacobian_generic(build_fn, p, n)
            if out is None:  # mecanismo no ensamblable: penalización para rechazar el paso
                cache[key] = (np.full(n_res, size), np.zeros((n_res, family.n_params)))
            else:
                if on_progress is not None:
                    on_progress(label, out[0])
                r_d, j_d = distance_residuals(out[0], out[1], y, y_dense)
                r_c, j_c = family.compactness(p, size, variant)
                cache[key] = (np.concatenate([r_d, r_c]), np.vstack([j_d, j_c]))
        return cache[key]

    res = least_squares(
        lambda p: evaluate(p)[0],
        p0,
        jac=lambda p: evaluate(p)[1],
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return np.asarray(res.x)


@dataclass
class FitResult:
    family: str
    params: np.ndarray
    variant: Variant
    rms: float  # distancia chamfer RMS entre la curva final y el objetivo
    rms_rel: float  # rms / diagonal de la caja del objetivo
    curve: np.ndarray  # curva final, densa (n_dense, 2)
    initial_rms_rel: float  # error del mejor candidato antes del refinamiento continuo

    @property
    def branch(self) -> int:
        return self.variant.branch


def _builder(family: Family, variant: Variant) -> Callable[[np.ndarray], Built | None]:
    """``build`` de una familia con la variante fija, como función de los parámetros."""

    def build_fn(p: np.ndarray) -> Built | None:
        return family.build(p, variant)

    return build_fn


class Throttle:
    """Limita la frecuencia de un callback de progreso (el cálculo corre en un hilo aparte)."""

    def __init__(self, fn: ProgressFn, min_interval: float = 0.15) -> None:
        self._fn = fn
        self._min = min_interval
        self._last = 0.0

    def __call__(self, label: str, curve: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last >= self._min:
            self._last = now
            self._fn(label, curve)


def fit(
    target: np.ndarray,
    families: Sequence[Family],
    *,
    n: int = 48,
    n_candidates: int = 16000,
    n_refine: int = 2,
    max_nfev: int = 25,
    n_dense: int = 180,
    seed: int = 0,
    on_progress: ProgressFn | None = None,
) -> FitResult | None:
    """Ajusta el mejor mecanismo de ``families`` a ``target`` (curva cerrada (m, 2)).

    Devuelve None si ninguna familia produce un mecanismo válido. ``on_progress(etiqueta, curva)``
    recibe la curva del candidato inicial y luego la de cada evaluación del refinamiento."""
    y = resample_closed(target, n)
    y_dense = resample_closed(target, n_dense)
    size = bbox_diag(y)
    best: FitResult | None = None
    for fi, fam in enumerate(families):
        rng = np.random.default_rng(seed + fi)
        starts = fam.candidates(y, n, n_candidates, n_refine, rng)
        if starts and on_progress is not None:
            on_progress(f"{fam.name}: candidato inicial", starts[0].curve)
        for c in starts:
            p = refine(fam, c.p0, c.variant, y, y_dense, n, max_nfev, on_progress, fam.name)
            curve = trace_curve_generic(_builder(fam, c.variant), p, n_dense)
            if curve is None:
                continue
            rms = chamfer_rms(curve, y_dense)
            if best is None or rms < best.rms:
                best = FitResult(fam.name, p, c.variant, rms, rms / size, curve, c.rel_error)
    return best


def refine_from(
    family: Family,
    p0: np.ndarray,
    variant: Variant,
    target: np.ndarray,
    *,
    n: int = 48,
    max_nfev: int = 60,
    n_dense: int = 180,
    on_progress: ProgressFn | None = None,
) -> FitResult | None:
    """Refina un mecanismo dado (p. ej. un inicial deliberadamente malo) sin búsqueda aleatoria."""
    y = resample_closed(target, n)
    y_dense = resample_closed(target, n_dense)
    size = bbox_diag(y)
    start = trace_curve_generic(_builder(family, variant), p0, n_dense)
    init_rel = chamfer_rms(start, y_dense) / size if start is not None else math.inf
    p = refine(family, p0, variant, y, y_dense, n, max_nfev, on_progress, family.name)
    curve = trace_curve_generic(_builder(family, variant), p, n_dense)
    if curve is None:
        return None
    rms = chamfer_rms(curve, y_dense)
    return FitResult(family.name, p, variant, rms, rms / size, curve, init_rel)
