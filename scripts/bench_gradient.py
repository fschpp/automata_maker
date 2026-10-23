"""Compara el gradiente por función implícita con diferencias finitas completas (M2).

Uso:  PYTHONPATH=. python3 scripts/bench_gradient.py

Para cada familia mide el tiempo de ``curve_and_jacobian`` (curva del marcador + Jacobiano
dx/dp en n fases) con ambos métodos y el error relativo entre ellos.
"""

import time

import numpy as np

from mechchar.optim import fivebar_fit, fourbar_fit
from mechchar.optim.fitcore import (
    Variant,
    curve_and_jacobian_finite_differences,
    curve_and_jacobian_generic,
)

N = 48
CASES = {
    "cuadrilátero (10 parámetros)": (
        lambda q: fourbar_fit.build(q, 1),
        np.array([0.4, -0.3, 0.35, 4.0, 1.0, 3.5, 3.0, 1.7, 0.9, 0.6]),
    ),
    "5-bar con engranajes, ratio -1 (12 parámetros)": (
        lambda q: fivebar_fit.build(q, Variant(1, -1.0)),
        np.array([0.2, -0.1, 0.3, 4.0, 0.9, 0.7, 3.4, 3.0, 1.7, 0.8, 0.0, 0.9]),
    ),
    "5-bar con engranajes, ratio 2 (12 parámetros)": (
        lambda q: fivebar_fit.build(q, Variant(1, 2.0)),
        np.array([0.2, -0.1, 0.3, 4.0, 0.9, 0.7, 3.4, 3.0, 1.7, 0.8, 0.0, 0.9]),
    ),
}


def timed(method, build_fn, p, repeats):
    best = np.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = method(build_fn, p, N)
        best = min(best, time.perf_counter() - t0)
    return out, best


for name, (build_fn, p) in CASES.items():
    fast, t_fast = timed(curve_and_jacobian_generic, build_fn, p, 3)
    slow, t_slow = timed(curve_and_jacobian_finite_differences, build_fn, p, 2)
    assert fast is not None and slow is not None
    err = np.max(np.abs(fast[1] - slow[1])) / max(1.0, np.max(np.abs(fast[1])))
    print(
        f"{name}: implícita {t_fast * 1e3:.0f} ms | diferencias finitas {t_slow * 1e3:.0f} ms | "
        f"ganancia {t_slow / t_fast:.1f}x | error relativo {err:.1e}"
    )
