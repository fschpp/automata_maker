"""Utilidades para curvas cerradas muestreadas (arreglos (n, 2)). Sin dependencias de GTK."""

from __future__ import annotations

import numpy as np


def resample_closed(points: np.ndarray, m: int) -> np.ndarray:
    """Remuestrea una poligonal cerrada en ``m`` puntos equiespaciados por longitud de arco."""
    pts = np.asarray(points, dtype=float)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0.0:
        raise ValueError("la curva tiene longitud cero")
    u = np.linspace(0.0, total, m, endpoint=False)
    return np.column_stack([np.interp(u, arc, closed[:, 0]), np.interp(u, arc, closed[:, 1])])


def bbox_diag(points: np.ndarray) -> float:
    """Diagonal de la caja delimitadora: medida de tamaño de la curva."""
    ext = np.ptp(np.asarray(points, dtype=float), axis=0)
    return float(np.hypot(ext[0], ext[1]))


def chamfer_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia RMS simétrica entre dos conjuntos de puntos, sin necesitar correspondencia."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(0.5 * (d2.min(1).mean() + d2.min(0).mean())))


def best_correspondence(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """Mejor desplazamiento cíclico y sentido para emparejar ``x`` con ``y`` (mismo n).

    Devuelve ``(shift, direction, cost)``: con ``y_dir = y`` (direction=+1) o ``y[::-1]``
    (direction=-1), la muestra k de ``x`` se empareja con ``y_dir[(k + shift) % n]`` y ``cost`` es
    la suma de distancias al cuadrado.
    """
    n = len(x)
    k = np.arange(n)
    idx = (k[:, None] + k[None, :]) % n  # idx[k, s] = (k + s) % n
    best = (0, 1, np.inf)
    for direction, y_dir in ((1, y), (-1, y[::-1])):
        d2 = ((x[:, None, :] - y_dir[None, :, :]) ** 2).sum(-1)  # (n, n)
        cost = d2[k[:, None], idx].sum(0)  # cost[s]
        s = int(np.argmin(cost))
        if cost[s] < best[2]:
            best = (s, direction, float(cost[s]))
    return best
