"""Métrica entre curvas cerradas (M3). Sin dependencias de GTK.

Las curvas se normalizan (centroide en el origen, radio RMS = 1) y se comparan con una distancia
que es invariante al punto de inicio, al sentido de recorrido y a la rotación:

    D²(s) = d_pos²(s) + λ·d_angle²(s) = 2(1 + λ) − 2·|C[s]| / N

donde C[s] = Σ_k a_k·conj(b_{k+s}) + λ·Σ_k t^a_k·conj(t^b_{k+s}) (posiciones y tangentes unitarias
como números complejos). Para cada desplazamiento s la rotación óptima es arg C[s] y el valor
óptimo es |C[s]|, y C se obtiene para todos los s con una correlación cruzada por FFT, O(N log N).

Diferencias con el plan: se normaliza por el radio RMS (no por l_max), porque deja Σ|a|² = N y
simplifica la fórmula; no se alinea por PCA (inestable en curvas casi circulares): la rotación se
optimiza dentro de la métrica. Los coeficientes son fijos (el entrenamiento queda para después).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechchar.optim.curves import resample_closed

N_NORM = 64  # puntos por curva normalizada
LAMBDA_TAN = 0.5  # peso de la dirección de la tangente (d_angle) frente a la posición (d_pos)
SELF_INTERSECTION_PENALTY = 0.25
N_HARMONICS = 6  # armónicos de Fourier en el descriptor de preselección
N_FEATURES = 6


@dataclass
class Normalized:
    pts: np.ndarray  # (n, 2), centroide 0, radio RMS 1
    center: np.ndarray  # (2,) centroide original
    scale: float  # radio RMS original: curva = center + scale * pts


def normalize(curve: np.ndarray, n: int = N_NORM) -> Normalized:
    """Remuestrea a n puntos equiespaciados y normaliza traslación y escala."""
    x = resample_closed(curve, n)
    center = x.mean(axis=0)
    x0 = x - center
    scale = float(np.sqrt((x0**2).sum(axis=1).mean()))
    if scale <= 0.0:
        raise ValueError("curva degenerada")
    return Normalized(x0 / scale, center, scale)


def to_complex(pts: np.ndarray) -> np.ndarray:
    return pts[..., 0] + 1j * pts[..., 1]


def tangents(z: np.ndarray) -> np.ndarray:
    """Tangentes unitarias (complejas) de poligonales cerradas, a lo largo del último eje."""
    d = np.roll(z, -1, axis=-1) - z
    mag = np.abs(d)
    return d / np.where(mag > 0.0, mag, 1.0)


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """g[..., s] = Σ_k a_k · conj(b_{(k+s) mod N}) para todos los s, con FFT.

    ``a``: (N,) y ``b``: (..., N), complejos."""
    n = a.shape[-1]
    return np.fft.ifft(n * np.fft.ifft(a) * np.fft.fft(np.conj(b), axis=-1), axis=-1)


@dataclass
class ShapeMatch:
    distance: np.ndarray  # (M,)
    shift: np.ndarray  # (M,) desplazamiento del índice de inicio
    rotation: np.ndarray  # (M,) ángulo α tal que a_k ≈ e^{iα} b_{k+shift}
    reverse: np.ndarray  # (M,) True si b se recorrió en sentido inverso


def shape_distance_many(a: np.ndarray, b_many: np.ndarray, lam: float = LAMBDA_TAN) -> ShapeMatch:
    """Distancia de la curva normalizada ``a`` (n, 2) a cada curva de ``b_many`` (M, n, 2),
    minimizando sobre desplazamiento, rotación y sentido."""
    n = a.shape[0]
    za, ta = to_complex(a), tangents(to_complex(a))
    m = b_many.shape[0]
    rows = np.arange(m)
    best_d = np.full(m, np.inf)
    best_s = np.zeros(m, dtype=int)
    best_alpha = np.zeros(m)
    best_rev = np.zeros(m, dtype=bool)
    for rev in (False, True):
        zb = to_complex(b_many)
        if rev:
            zb = zb[:, ::-1]
        corr = xcorr(za, zb) + lam * xcorr(ta, tangents(zb))
        mag = np.abs(corr)
        s = mag.argmax(axis=1)
        top = mag[rows, s]
        d = np.sqrt(np.maximum(2.0 * (1.0 + lam) - 2.0 * top / n, 0.0))
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, s, best_s)
        best_alpha = np.where(better, np.angle(corr[rows, s]), best_alpha)
        best_rev = np.where(better, rev, best_rev)
    return ShapeMatch(best_d, best_s, best_alpha, best_rev)


def shape_distance(a: np.ndarray, b: np.ndarray, lam: float = LAMBDA_TAN) -> float:
    return float(shape_distance_many(a, b[None], lam).distance[0])


# --- features y descriptor ----------------------------------------------------------------------


def self_intersections(pts: np.ndarray) -> int:
    """Número de cruces propios de la poligonal cerrada (segmentos no adyacentes)."""
    n = len(pts)
    p, q = pts, np.roll(pts, -1, axis=0)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
            c[..., 0] - a[..., 0]
        )

    pi, qi = p[:, None, :], q[:, None, :]
    pj, qj = p[None, :, :], q[None, :, :]
    cross = (orient(pi, qi, pj) * orient(pi, qi, qj) < 0) & (
        orient(pj, qj, pi) * orient(pj, qj, qi) < 0
    )
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    nonadjacent = (j > i) & (j != (i + 1) % n) & (i != (j + 1) % n)
    return int(np.count_nonzero(cross & nonadjacent))


def features(pts: np.ndarray, with_intersections: bool = True) -> np.ndarray:
    """f1..f6 de una curva normalizada: longitud, área, elipticidad, asimetría a lo largo de cada
    eje principal y autointersecciones. Todas invariantes a traslación, escala y rotación."""
    d = np.roll(pts, -1, axis=0) - pts
    perimeter = np.hypot(d[:, 0], d[:, 1]).sum() / (2.0 * np.pi)
    x, y = pts[:, 0], pts[:, 1]
    area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    evals, evecs = np.linalg.eigh(np.cov(pts.T, bias=True))  # ascendente
    ellipticity = float(np.sqrt(max(evals[0], 0.0) / max(evals[1], 1e-12)))

    def skew(u: np.ndarray) -> float:
        proj = pts @ u
        m2 = float((proj**2).mean())
        return abs(float((proj**3).mean()) / max(m2, 1e-12) ** 1.5)

    return np.array(
        [
            perimeter,
            area / np.pi,
            ellipticity,
            skew(evecs[:, 1]),
            skew(evecs[:, 0]),
            float(self_intersections(pts)) if with_intersections else 0.0,
        ]
    )


def descriptor(pts: np.ndarray, feats: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Vector de preselección: f1..f5 y magnitudes de los armónicos ±1..±J (ponderadas por el
    orden). Es invariante a rotación, traslación, escala y punto de inicio; invertir el sentido
    intercambia los armónicos positivos y negativos."""
    z_hat = np.fft.fft(to_complex(pts)) / len(pts)
    k = np.arange(1, N_HARMONICS + 1)
    pos, neg = k * np.abs(z_hat[k]), k * np.abs(z_hat[-k])
    if reverse:
        pos, neg = neg, pos
    return np.concatenate([feats[: N_FEATURES - 1], pos, neg])


# --- baseline: distancia de Fréchet discreta ----------------------------------------------------


def _pca_angle(pts: np.ndarray) -> float:
    c = np.cov(pts.T, bias=True)
    return 0.5 * float(np.arctan2(2.0 * c[0, 1], c[0, 0] - c[1, 1]))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def frechet_closed(a: np.ndarray, b: np.ndarray, n: int = 32) -> tuple[float, float]:
    """Fréchet discreto entre dos curvas normalizadas, alineadas por PCA y probando los 2 signos
    del eje, los 2 sentidos y los n puntos de inicio. Devuelve (distancia, rotación de b sobre a).

    Es la referencia de la Fig. 9 del paper; el DP está vectorizado sobre las alternativas."""
    step_a, step_b = len(a) // n, len(b) // n
    pa, pb = a[::step_a] @ _rot(-_pca_angle(a)).T, b[::step_b] @ _rot(-_pca_angle(b)).T
    variants, meta = [], []
    for flip in (0, 1):
        base = pb @ _rot(np.pi * flip).T
        for rev in (False, True):
            cur = base[::-1] if rev else base
            for s in range(n):
                variants.append(np.roll(cur, -s, axis=0))
                meta.append((flip, rev))
    bv = np.stack(variants)  # (V, n, 2)
    dist = np.linalg.norm(pa[:, None, None, :] - bv.transpose(1, 0, 2)[None], axis=-1)  # (n, n, V)
    ca = np.empty_like(dist)
    ca[0, 0] = dist[0, 0]
    for j in range(1, n):
        ca[0, j] = np.maximum(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        ca[i, 0] = np.maximum(ca[i - 1, 0], dist[i, 0])
        for j in range(1, n):
            ca[i, j] = np.maximum(
                np.minimum(np.minimum(ca[i - 1, j], ca[i - 1, j - 1]), ca[i, j - 1]), dist[i, j]
            )
    best = int(ca[n - 1, n - 1].argmin())
    flip, _ = meta[best]
    # b (en su marco original) ≈ a rotada: a ≈ R(α_a + π·flip − α_b) b
    rotation = _pca_angle(a) + np.pi * flip - _pca_angle(b)
    return float(ca[n - 1, n - 1, best]), rotation
