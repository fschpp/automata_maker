"""Familia «cuadrilátero articulado» para el ajuste a una curva cerrada objetivo (M2).

Parámetros p = (ax, ay, phi, d, r, lc, rr, tx, ty, psi):
  (ax, ay, phi)  pose de la bancada en el mundo (pivote de la manivela y ángulo de la bancada)
  d, r, lc, rr   bancada, manivela, acoplador y balancín
  (tx, ty)       marcador, en coordenadas locales del acoplador
  psi            ángulo de la manivela (mundo) en t = 0, es decir el desfase del ciclo

El gradiente, el refinamiento y la búsqueda son comunes a todas las familias (``fitcore``). Los
candidatos se generan al azar y se ordenan con una evaluación analítica vectorizada, solo para
elegir puntos de partida.
"""

from __future__ import annotations

import math

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Pin
from mechchar.core.mechanisms import FourBar, circle_intersections
from mechchar.optim.fitcore import (
    MAX_LINK_RATIO,
    Built,
    Candidate,
    FitResult,
    ProgressFn,
    Variant,
    curve_and_jacobian_generic,
    distance_residuals,
    fit,
    principal_angle,
    rank_aligned,
    trace_curve_generic,
)

__all__ = [
    "MAX_LINK_RATIO", "N_PARAMS", "FOUR_BAR", "Built", "FitResult", "FourBarFamily",
    "build", "curve_and_jacobian", "distance_residuals", "fit_four_bar", "to_four_bar",
    "trace_curve",
]

N_PARAMS = 10
TWO_PI = 2.0 * math.pi


def build(p: np.ndarray, branch: int) -> Built | None:
    """Ensamblaje para los parámetros ``p``; None si no se puede armar en t = 0."""
    ax, ay, phi, d, r, lc, rr, tx, ty, psi = (float(v) for v in p)
    if min(d, r, lc, rr) <= 0.0:
        return None
    th_local = psi - phi
    b_loc = np.array([r * math.cos(th_local), r * math.sin(th_local)])
    d_loc = np.array([d, 0.0])
    try:
        cands = circle_intersections(tuple(b_loc), lc, (d, 0.0), rr)
    except ValueError:
        return None
    p_loc = cands[0] if branch > 0 else cands[1]
    c, s = math.cos(phi), math.sin(phi)
    rot_g = np.array([[c, -s], [s, c]])
    origin = np.array([ax, ay])
    b_w, d_w = origin + rot_g @ b_loc, origin + rot_g @ d_loc

    asm = Assembly()
    g = asm.add_component("ground", ax, ay, phi)
    cr = asm.add_component("crank", ax, ay, psi)
    k = asm.add_component(
        "coupler", b_w[0], b_w[1], phi + math.atan2(p_loc[1] - b_loc[1], p_loc[0] - b_loc[0])
    )
    ro = asm.add_component(
        "rocker", d_w[0], d_w[1], phi + math.atan2(p_loc[1] - d_loc[1], p_loc[0] - d_loc[0])
    )
    asm.add(FixedState(g, ax, ay, phi))
    asm.add(Pin(g, (0.0, 0.0), cr, (0.0, 0.0)))
    asm.add(Pin(cr, (r, 0.0), k, (0.0, 0.0)))
    asm.add(Pin(k, (lc, 0.0), ro, (rr, 0.0)))
    asm.add(Pin(ro, (0.0, 0.0), g, (d, 0.0)))
    asm.add(Drive(cr, offset=psi))
    return Built(asm, asm.initial_state(), k, (tx, ty))


def to_four_bar(p: np.ndarray, branch: int) -> FourBar | None:
    """El mecanismo de parámetros ``p`` como ``FourBar`` (para animarlo y dibujarlo)."""
    b = build(p, branch)
    if b is None:
        return None
    _, _, _, _, r, lc, rr, _, _, _ = (float(v) for v in p)
    return FourBar(b.asm, b.state0, 0, 1, b.coupler, 3, b.tracer, {1: r, b.coupler: lc, 3: rr})


def trace_curve(p: np.ndarray, branch: int, n: int) -> np.ndarray | None:
    """Curva del marcador en n fases t = k/n, o None si el mecanismo no completa el ciclo."""
    return trace_curve_generic(lambda q: build(q, branch), p, n)


def curve_and_jacobian(
    p: np.ndarray, branch: int, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Curva x(p) (n, 2) y su Jacobiano dx/dp (n, 2, N_PARAMS), por función implícita."""
    return curve_and_jacobian_generic(lambda q: build(q, branch), p, n)


# --- evaluación analítica vectorizada (solo para elegir candidatos) -----------------------------


def _fast_local_curves(
    r: np.ndarray, lc: np.ndarray, rr: np.ndarray, tx: np.ndarray, ty: np.ndarray,
    branch: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Curvas del marcador en el marco de la bancada (d = 1, manivela en ángulo 0 en t = 0).

    Devuelve (curvas (K, n, 2), válidas (K,)); válida = el ciclo completo es ensamblable."""
    th = TWO_PI * np.arange(n) / n
    bx, by = r[:, None] * np.cos(th)[None, :], r[:, None] * np.sin(th)[None, :]
    vx, vy = 1.0 - bx, -by
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.hypot(vx, vy)
        a = (lc[:, None] ** 2 - rr[:, None] ** 2 + dist**2) / (2.0 * dist)
        h2 = lc[:, None] ** 2 - a**2
        h = np.sqrt(np.maximum(h2, 0.0))
        ux, uy = vx / dist, vy / dist
    valid = np.isfinite(h2).all(1) & (h2 > (0.02 * lc[:, None]) ** 2).all(1)
    px = bx + a * ux - branch[:, None] * h * uy
    py = by + a * uy + branch[:, None] * h * ux
    ang = np.arctan2(py - by, px - bx)
    c, s = np.cos(ang), np.sin(ang)
    qx = bx + c * tx[:, None] - s * ty[:, None]
    qy = by + s * tx[:, None] + c * ty[:, None]
    return np.stack([qx, qy], axis=-1), valid


_principal_angle = principal_angle  # nombre histórico


def _compactness(p: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Residuos de bisagra que penalizan eslabones o marcador más grandes que el límite."""
    limit = MAX_LINK_RATIO * size
    res = np.zeros(5)
    jac = np.zeros((5, N_PARAMS))
    for row, idx in enumerate((3, 4, 5, 6)):
        if p[idx] > limit:
            res[row] = p[idx] - limit
            jac[row, idx] = 1.0
    h = math.hypot(p[7], p[8])
    if h > limit:
        res[4] = h - limit
        jac[4, 7], jac[4, 8] = p[7] / h, p[8] / h
    return res, jac


class FourBarFamily:
    name = "four_bar"
    n_params = N_PARAMS
    n_compact = 5

    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        return build(p, variant.branch)

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]:
        return _compactness(p, size)

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> FourBar | None:
        return to_four_bar(p, variant.branch)

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        """Mejores candidatos aleatorios, ya alineados con el objetivo ``y``."""
        size = float(np.hypot(*np.ptp(y, axis=0)))
        pool: list[tuple[float, np.ndarray, int, np.ndarray]] = []
        chunk = 4000
        for _ in range(max(1, n_candidates // chunk)):
            r = rng.uniform(0.1, 1.0, chunk)
            lc = rng.uniform(0.6, 3.0, chunk)
            rr = rng.uniform(0.6, 3.0, chunk)
            tx = lc * rng.uniform(-0.5, 1.5, chunk)
            ty = lc * rng.uniform(-1.0, 1.0, chunk)
            br = rng.choice([-1, 1], chunk)
            local, valid = _fast_local_curves(r, lc, rr, tx, ty, br.astype(float), n)
            keep = np.where(valid)[0]
            if keep.size == 0:
                continue
            local, r, lc, rr, tx, ty, br = (a[keep] for a in (local, r, lc, rr, tx, ty, br))
            extent = np.maximum.reduce([np.ones_like(r), r, lc, rr, np.hypot(tx, ty)])
            for al in rank_aligned(local, extent, y, n_keep):
                i, f = al.index, al.scale
                p0 = np.array(
                    [al.origin[0], al.origin[1], al.phi, f, f * r[i], f * lc[i], f * rr[i],
                     f * tx[i], f * ty[i], al.phi]
                )
                pool.append((al.cost, p0, int(br[i]), al.curve))
        pool.sort(key=lambda item: item[0])
        return [
            Candidate(p0, Variant(br), cost / size, curve) for cost, p0, br, curve in pool[:n_keep]
        ]


FOUR_BAR = FourBarFamily()


def fit_four_bar(
    target: np.ndarray,
    *,
    n: int = 48,
    n_candidates: int = 16000,
    n_refine: int = 2,
    max_nfev: int = 25,
    n_dense: int = 180,
    seed: int = 0,
    on_progress: ProgressFn | None = None,
) -> FitResult | None:
    """Ajusta un cuadrilátero a ``target`` (curva cerrada (m, 2)); None si no hay ninguno válido."""
    return fit(
        target, [FOUR_BAR], n=n, n_candidates=n_candidates, n_refine=n_refine,
        max_nfev=max_nfev, n_dense=n_dense, seed=seed, on_progress=on_progress,
    )
