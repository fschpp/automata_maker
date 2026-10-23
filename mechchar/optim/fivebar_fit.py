"""Familia «5-bar con dos engranajes» (Fig. 4 del paper) para el ajuste a una curva cerrada (M2).

Parámetros p = (ax, ay, phi, d, r1, r2, l1, l2, tx, ty, psi, delta):
  (ax, ay, phi)  pose de la bancada en el mundo (pivote A del engranaje 1 y ángulo de la bancada)
  d              distancia entre los pivotes de los engranajes (A en (0, 0), D en (d, 0))
  r1, r2         radios de las manivelas (distancia del pin al centro de su engranaje)
  l1, l2         largos de las dos bielas (la biela 1 se prolonga hasta el marcador)
  (tx, ty)       marcador, en coordenadas locales de la biela 1
  psi            ángulo de la manivela 1 en t = 0, en el marco de la bancada
  delta          desfase entre las manivelas: θ₂ = ratio·θ₁ + δ

La relación de engranajes ``ratio`` ∈ {−2, −1, 1, 2} es discreta (variante, no se optimiza): con
ella el período de la curva coincide con el del Input Driver (§7.1 del paper). Los radios de los
engranajes salen de la relación (R₁/R₂ = |ratio|, R₁ + R₂ = d), y las manivelas deben caber dentro
de su engranaje.
"""

from __future__ import annotations

import math

import numpy as np

from mechchar.core.mechanisms import GearedFiveBar, geared_five_bar
from mechchar.optim.fitcore import (
    MAX_LINK_RATIO,
    Built,
    Candidate,
    LocalSample,
    Variant,
    curve_and_jacobian_generic,
    rank_aligned,
    trace_curve_generic,
)

N_PARAMS = 12
RATIOS = (-2.0, -1.0, 1.0, 2.0)
CRANK_FIT = 0.9  # la manivela debe quedar a menos de esta fracción del radio de su engranaje
TWO_PI = 2.0 * math.pi


def _ratio_split(ratio: float) -> tuple[float, float]:
    """(R₁/d, R₂/d): fracciones de la distancia entre pivotes que ocupa cada engranaje."""
    k = abs(ratio)
    return k / (1.0 + k), 1.0 / (1.0 + k)


def build(p: np.ndarray, variant: Variant) -> Built | None:
    """Ensamblaje para los parámetros ``p``; None si no se puede armar en t = 0."""
    ax, ay, phi, d, r1, r2, l1, l2, tx, ty, psi, delta = (float(v) for v in p)
    if min(d, r1, r2, l1, l2) <= 0.0:
        return None
    try:
        m = geared_five_bar(
            d=d, r1=r1, r2=r2, l1=l1, l2=l2, ratio=variant.ratio, tracer=(tx, ty),
            branch=variant.branch, delta=delta, psi=psi, pose=(ax, ay, phi),
        )
    except ValueError:
        return None
    return Built(m.asm, m.state0, m.link1, m.tracer)


def to_five_bar(p: np.ndarray, variant: Variant) -> GearedFiveBar | None:
    """El mecanismo de parámetros ``p`` (para animarlo y dibujarlo)."""
    ax, ay, phi, d, r1, r2, l1, l2, tx, ty, psi, delta = (float(v) for v in p)
    try:
        return geared_five_bar(
            d=d, r1=r1, r2=r2, l1=l1, l2=l2, ratio=variant.ratio, tracer=(tx, ty),
            branch=variant.branch, delta=delta, psi=psi, pose=(ax, ay, phi),
        )
    except ValueError:
        return None


def trace_curve(p: np.ndarray, variant: Variant, n: int) -> np.ndarray | None:
    return trace_curve_generic(lambda q: build(q, variant), p, n)


def curve_and_jacobian(
    p: np.ndarray, variant: Variant, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    return curve_and_jacobian_generic(lambda q: build(q, variant), p, n)


# --- evaluación analítica vectorizada (solo para elegir candidatos) -----------------------------


def _fast_local_curves(
    r1: np.ndarray, r2: np.ndarray, l1: np.ndarray, l2: np.ndarray, tx: np.ndarray,
    ty: np.ndarray, ratio: np.ndarray, delta: np.ndarray, branch: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Curvas del marcador en el marco de la bancada (d = 1, ψ = 0).

    Devuelve (curvas (K, n, 2), válidas (K,)); válida = el ciclo completo es ensamblable."""
    th1 = TWO_PI * np.arange(n) / n
    th2 = ratio[:, None] * th1[None, :] + delta[:, None]
    b1x, b1y = r1[:, None] * np.cos(th1)[None, :], r1[:, None] * np.sin(th1)[None, :]
    b2x, b2y = 1.0 + r2[:, None] * np.cos(th2), r2[:, None] * np.sin(th2)
    vx, vy = b2x - b1x, b2y - b1y
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.hypot(vx, vy)
        a = (l1[:, None] ** 2 - l2[:, None] ** 2 + dist**2) / (2.0 * dist)
        h2 = l1[:, None] ** 2 - a**2
        h = np.sqrt(np.maximum(h2, 0.0))
        ux, uy = vx / dist, vy / dist
    margin = (0.02 * np.minimum(l1, l2)[:, None]) ** 2
    valid = np.isfinite(h2).all(1) & (h2 > margin).all(1)
    px = b1x + a * ux - branch[:, None] * h * uy
    py = b1y + a * uy + branch[:, None] * h * ux
    ang = np.arctan2(py - b1y, px - b1x)
    c, s = np.cos(ang), np.sin(ang)
    qx = b1x + c * tx[:, None] - s * ty[:, None]
    qy = b1y + s * tx[:, None] + c * ty[:, None]
    return np.stack([qx, qy], axis=-1), valid


def _compactness(
    p: np.ndarray, size: float, variant: Variant
) -> tuple[np.ndarray, np.ndarray]:
    """Residuos de bisagra: eslabones y marcador acotados por el tamaño de la curva, y cada
    manivela dentro de su engranaje."""
    limit = MAX_LINK_RATIO * size
    res = np.zeros(8)
    jac = np.zeros((8, N_PARAMS))
    for row, idx in enumerate((3, 4, 5, 6, 7)):  # d, r1, r2, l1, l2
        if p[idx] > limit:
            res[row] = p[idx] - limit
            jac[row, idx] = 1.0
    h = math.hypot(p[8], p[9])
    if h > limit:
        res[5] = h - limit
        jac[5, 8], jac[5, 9] = p[8] / h, p[9] / h
    frac1, frac2 = _ratio_split(variant.ratio)
    for row, (r_idx, frac) in zip((6, 7), ((4, frac1), (5, frac2)), strict=True):
        over = p[r_idx] - CRANK_FIT * frac * p[3]
        if over > 0.0:
            res[row] = over
            jac[row, r_idx] = 1.0
            jac[row, 3] = -CRANK_FIT * frac
    return res, jac


class GearedFiveBarFamily:
    name = "geared_five_bar"
    n_params = N_PARAMS
    n_compact = 8

    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        return build(p, variant)

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]:
        return _compactness(p, size, variant)

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> GearedFiveBar | None:
        return to_five_bar(p, variant)

    def sample_local(self, rng: np.random.Generator, k: int, n: int) -> LocalSample | None:
        ratio = rng.choice(RATIOS, k)
        frac1 = np.abs(ratio) / (1.0 + np.abs(ratio))
        frac2 = 1.0 / (1.0 + np.abs(ratio))
        r1 = rng.uniform(0.15, 1.0, k) * CRANK_FIT * frac1
        r2 = rng.uniform(0.15, 1.0, k) * CRANK_FIT * frac2
        l1 = rng.uniform(0.4, 3.0, k)
        l2 = rng.uniform(0.4, 3.0, k)
        tx = l1 * rng.uniform(-0.5, 1.5, k)
        ty = l1 * rng.uniform(-1.0, 1.0, k)
        delta = rng.uniform(0.0, TWO_PI, k)
        br = rng.choice([-1, 1], k)
        local, valid = _fast_local_curves(r1, r2, l1, l2, tx, ty, ratio, delta, br.astype(float), n)
        keep = np.where(valid)[0]
        if keep.size == 0:
            return None
        r1, r2, l1, l2, tx, ty, ratio, delta, br = (
            a[keep] for a in (r1, r2, l1, l2, tx, ty, ratio, delta, br)
        )
        extent = np.maximum.reduce([np.ones_like(r1), r1, r2, l1, l2, np.hypot(tx, ty)])
        shape = np.stack([r1, r2, l1, l2, tx, ty, delta], axis=1)
        variants = [Variant(int(b), float(q)) for b, q in zip(br, ratio, strict=True)]
        return LocalSample(local[keep], extent, shape, variants)

    def assemble(
        self, shape: np.ndarray, variant: Variant, scale: float, phi: float, origin: np.ndarray
    ) -> np.ndarray:
        r1, r2, l1, l2, tx, ty, delta = (float(v) for v in shape)
        f = scale
        return np.array(
            [origin[0], origin[1], phi, f, f * r1, f * r2, f * l1, f * l2, f * tx, f * ty,
             0.0, delta]
        )

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        """Mejores candidatos aleatorios, ya alineados con el objetivo ``y``."""
        size = float(np.hypot(*np.ptp(y, axis=0)))
        pool: list[tuple[float, np.ndarray, Variant, np.ndarray]] = []
        chunk = 4000
        for _ in range(max(1, n_candidates // chunk)):
            s = self.sample_local(rng, chunk, n)
            if s is None:
                continue
            for al in rank_aligned(s.curves, s.extent, y, n_keep):
                i = al.index
                p0 = self.assemble(s.shape[i], s.variants[i], al.scale, al.phi, al.origin)
                pool.append((al.cost, p0, s.variants[i], al.curve))
        pool.sort(key=lambda item: item[0])
        return [Candidate(p0, v, cost / size, curve) for cost, p0, v, curve in pool[:n_keep]]


GEARED_FIVE_BAR = GearedFiveBarFamily()
