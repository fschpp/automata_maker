"""Familia «manivela-corredera» para el ajuste a una curva cerrada objetivo (M2).

Parámetros p = (ax, ay, phi, r, l, h, tx, ty, psi):
  (ax, ay, phi)  pose de la bancada en el mundo (pivote de la manivela y ángulo de la bancada)
  r, l           manivela y biela
  h              descentrado de la guía: la corredera se mueve sobre la recta y = h del marco de
                 la bancada (h = 0 es la manivela-corredera centrada)
  (tx, ty)       marcador, en coordenadas locales de la biela
  psi            ángulo de la manivela (mundo) en t = 0, es decir el desfase del ciclo

Sirve para curvas con un tramo casi recto o muy asimétricas que el cuadrilátero hace mal. El
gradiente, el refinamiento y la búsqueda son comunes a todas las familias (``fitcore``).
"""

from __future__ import annotations

import math

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Pin, PointOnLine
from mechchar.core.mechanisms import SliderCrank
from mechchar.optim.fitcore import (
    MAX_LINK_RATIO,
    Built,
    Candidate,
    LocalSample,
    Variant,
    random_candidates,
    trace_curve_generic,
)

N_PARAMS = 9
N_COMPACT = 4
TWO_PI = 2.0 * math.pi
MIN_CLEARANCE = 0.02  # la biela no se puede acercar a la vertical de la guía más que esto (× l)


def build(p: np.ndarray, branch: int) -> Built | None:
    """Ensamblaje para los parámetros ``p``; None si no se puede armar en t = 0."""
    ax, ay, phi, r, l, h, tx, ty, psi = (float(v) for v in p)  # noqa: E741
    if min(r, l) <= 0.0:
        return None
    b_loc = np.array([r * math.cos(psi - phi), r * math.sin(psi - phi)])
    dy = h - b_loc[1]
    if abs(dy) >= l:
        return None
    dx = branch * math.sqrt(l * l - dy * dy)
    c, s = math.cos(phi), math.sin(phi)
    b_w = np.array([ax, ay]) + np.array([[c, -s], [s, c]]) @ b_loc

    asm = Assembly()
    g = asm.add_component("ground", ax, ay, phi)
    cr = asm.add_component("crank", ax, ay, psi)
    k = asm.add_component("rod", b_w[0], b_w[1], phi + math.atan2(dy, dx))
    asm.add(FixedState(g, ax, ay, phi))
    asm.add(Pin(g, (0.0, 0.0), cr, (0.0, 0.0)))
    asm.add(Pin(cr, (r, 0.0), k, (0.0, 0.0)))
    asm.add(PointOnLine(k, (l, 0.0), g, (0.0, h), (1.0, 0.0)))
    asm.add(Drive(cr, offset=psi))
    return Built(asm, asm.initial_state(), k, (tx, ty))


def to_slider_crank(p: np.ndarray, branch: int) -> SliderCrank | None:
    """El mecanismo de parámetros ``p`` como ``SliderCrank`` (para animarlo y dibujarlo)."""
    b = build(p, branch)
    if b is None:
        return None
    r, l, h = float(p[3]), float(p[4]), float(p[5])  # noqa: E741
    return SliderCrank(b.asm, b.state0, 0, 1, b.coupler, b.tracer, r, l, h)


# --- evaluación analítica vectorizada (solo para elegir candidatos) -----------------------------


def _fast_local_curves(
    r: np.ndarray, l: np.ndarray, h: np.ndarray, tx: np.ndarray, ty: np.ndarray,  # noqa: E741
    branch: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Curvas del marcador en el marco de la bancada (manivela en ángulo 0 en t = 0).

    Devuelve (curvas (K, n, 2), válidas (K,)); válida = el ciclo completo es ensamblable."""
    th = TWO_PI * np.arange(n) / n
    bx, by = r[:, None] * np.cos(th)[None, :], r[:, None] * np.sin(th)[None, :]
    dy = h[:, None] - by
    h2 = l[:, None] ** 2 - dy**2
    valid = (h2 > (MIN_CLEARANCE * l[:, None]) ** 2).all(1)
    dx = branch[:, None] * np.sqrt(np.maximum(h2, 0.0))
    ang = np.arctan2(dy, dx)
    c, s = np.cos(ang), np.sin(ang)
    qx = bx + c * tx[:, None] - s * ty[:, None]
    qy = by + s * tx[:, None] + c * ty[:, None]
    return np.stack([qx, qy], axis=-1), valid


def _compactness(p: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Residuos de bisagra que penalizan manivela, biela, descentrado o marcador mayores que el
    límite."""
    limit = MAX_LINK_RATIO * size
    res = np.zeros(N_COMPACT)
    jac = np.zeros((N_COMPACT, N_PARAMS))
    for row, idx in enumerate((3, 4)):
        if p[idx] > limit:
            res[row] = p[idx] - limit
            jac[row, idx] = 1.0
    if abs(p[5]) > limit:
        res[2] = abs(p[5]) - limit
        jac[2, 5] = math.copysign(1.0, p[5])
    hyp = math.hypot(p[6], p[7])
    if hyp > limit:
        res[3] = hyp - limit
        jac[3, 6], jac[3, 7] = p[6] / hyp, p[7] / hyp
    return res, jac


class SliderCrankFamily:
    name = "slider_crank"
    n_params = N_PARAMS
    n_compact = N_COMPACT

    def build(self, p: np.ndarray, variant: Variant) -> Built | None:
        return build(p, variant.branch)

    def compactness(
        self, p: np.ndarray, size: float, variant: Variant
    ) -> tuple[np.ndarray, np.ndarray]:
        return _compactness(p, size)

    def to_mechanism(self, p: np.ndarray, variant: Variant) -> SliderCrank | None:
        return to_slider_crank(p, variant.branch)

    def sample_local(self, rng: np.random.Generator, k: int, n: int) -> LocalSample | None:
        r = rng.uniform(0.1, 1.0, k)
        l = rng.uniform(0.6, 3.0, k)  # noqa: E741
        h = rng.uniform(-1.0, 1.0, k)
        tx = l * rng.uniform(-0.5, 1.5, k)
        ty = l * rng.uniform(-1.0, 1.0, k)
        br = rng.choice([-1, 1], k)
        local, valid = _fast_local_curves(r, l, h, tx, ty, br.astype(float), n)
        keep = np.where(valid)[0]
        if keep.size == 0:
            return None
        r, l, h, tx, ty, br = (a[keep] for a in (r, l, h, tx, ty, br))  # noqa: E741
        extent = np.maximum.reduce([np.ones_like(r), r, l, np.abs(h), np.hypot(tx, ty)])
        shape = np.stack([r, l, h, tx, ty], axis=1)
        return LocalSample(local[keep], extent, shape, [Variant(int(b)) for b in br])

    def assemble(
        self, shape: np.ndarray, variant: Variant, scale: float, phi: float, origin: np.ndarray
    ) -> np.ndarray:
        r, l, h, tx, ty = (float(v) for v in shape)  # noqa: E741
        f = scale
        return np.array([origin[0], origin[1], phi, f * r, f * l, f * h, f * tx, f * ty, phi])

    def candidates(
        self, y: np.ndarray, n: int, n_candidates: int, n_keep: int, rng: np.random.Generator
    ) -> list[Candidate]:
        return random_candidates(self, y, n, n_candidates, n_keep, rng)


SLIDER_CRANK = SliderCrankFamily()


def trace_curve(p: np.ndarray, branch: int, n: int) -> np.ndarray | None:
    """Curva del marcador en n fases t = k/n, o None si el mecanismo no completa el ciclo."""
    return trace_curve_generic(lambda q: build(q, branch), p, n)


__all__ = [
    "N_PARAMS", "SLIDER_CRANK", "SliderCrankFamily", "build", "to_slider_crank", "trace_curve",
]
