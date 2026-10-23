"""Mide la base de movimientos (M3): construcción, tiempo de consulta y calidad frente a Fréchet.

Uso:
    PYTHONPATH=. python scripts/bench_motion_db.py [--entries 3000] [--curves 40] [--five-bar]

La calidad es el error chamfer (relativo al tamaño de la curva) del mecanismo top-1 *antes* de
refinar. Se compara la métrica propia con la distancia de Fréchet discreta (la referencia de la
Fig. 9 del paper) sobre las mismas candidatas preseleccionadas, y con el mejor posible dentro de
ellas. Las curvas "a mano" son formas conocidas con ondulaciones aleatorias y pose arbitraria.
"""

import argparse
import math
import time

import numpy as np

from mechchar.db.metric import frechet_closed, normalize
from mechchar.db.motion_db import MotionDB, build_database, make_candidate_source, query
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.families import FOUR_BAR, GEARED_FIVE_BAR
from mechchar.optim.fitcore import fit


def hand_drawn(rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    kind = rng.integers(5)
    if kind == 0:  # elipse
        a, b = 1.0, rng.uniform(0.35, 0.8)
        x, y = a * np.cos(t), b * np.sin(t)
    elif kind == 1:  # frijol
        x, y = np.cos(t), 0.6 * np.sin(t) + rng.uniform(0.15, 0.35) * np.cos(2 * t)
    elif kind == 2:  # triángulo redondeado
        r = 1 + rng.uniform(0.15, 0.3) * np.cos(3 * t)
        x, y = r * np.cos(t), r * np.sin(t)
    elif kind == 3:  # gota
        x, y = np.cos(t), 0.7 * np.sin(t) * (1 + rng.uniform(0.2, 0.5) * np.cos(t))
    else:  # círculo achatado
        x, y = np.cos(t), rng.uniform(0.7, 0.95) * np.sin(t)
    wobble = np.ones_like(t)
    for k in range(2, 6):  # ondulación de mano temblorosa
        wobble += rng.uniform(0.0, 0.04) * np.cos(k * t + rng.uniform(0, 2 * math.pi))
    x, y = x * wobble, y * wobble
    ang, scale = rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 5.0)
    c, s = math.cos(ang), math.sin(ang)
    pts = scale * np.column_stack([c * x - s * y, s * x + c * y])
    return pts + rng.normal(size=2) * 3


def match_error(db: MotionDB, curve: np.ndarray, i: int, rotation: float, q) -> float:
    """Error relativo del mecanismo ``i`` de la base llevado a ``curve`` con la rotación dada."""
    z = (db.pts[i, :, 0] + 1j * db.pts[i, :, 1]) * np.exp(1j * rotation) * q.scale
    z = z + complex(q.center[0], q.center[1])
    return chamfer_rms(np.column_stack([z.real, z.imag]), curve) / bbox_diag(curve)


def benchmark_queries(db: MotionDB, n_curves: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    curves = [hand_drawn(rng) for _ in range(n_curves)]
    query(db, curves[0])  # calentamiento
    times = []
    for c in curves:
        t0 = time.perf_counter()
        query(db, c, k=3)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  consulta top-3 con la base completa: media {np.mean(times):.1f} ms, "
          f"p95 {np.percentile(times, 95):.1f} ms, máx {np.max(times):.1f} ms (objetivo < 100 ms)")

    ours, frechet, oracle, t_fr = [], [], [], []
    for c in curves:
        q = normalize(c)
        top = query(db, c, k=1, max_ratio=None)[0]
        ours.append(match_error(db, c, top.index, top.rotation, q))
        # candidatas preseleccionadas (las mismas para las dos métricas)
        pool = query(db, c, k=60, max_ratio=None)
        oracle.append(min(match_error(db, c, m.index, m.rotation, q) for m in pool))
        t0 = time.perf_counter()
        scored = [(frechet_closed(q.pts, db.pts[m.index]), m.index) for m in pool]
        t_fr.append((time.perf_counter() - t0) * 1000)
        (fd, rot), idx = min(scored, key=lambda item: item[0][0])
        frechet.append(match_error(db, c, idx, rot, q))
    ours, frechet, oracle = map(np.array, (ours, frechet, oracle))
    print(f"  error del top-1 antes de refinar (% del tamaño), {n_curves} curvas a mano:")
    print(f"    métrica propia (FFT)   mediana {np.median(ours) * 100:5.2f}  media {ours.mean() * 100:5.2f}  "
          f"p90 {np.percentile(ours, 90) * 100:5.2f}")
    print(f"    Fréchet discreto       mediana {np.median(frechet) * 100:5.2f}  media {frechet.mean() * 100:5.2f}  "
          f"p90 {np.percentile(frechet, 90) * 100:5.2f}   ({np.mean(t_fr):.0f} ms por consulta sobre 60 candidatas)")
    print(f"    mejor posible (60 cand.) mediana {np.median(oracle) * 100:5.2f}  media {oracle.mean() * 100:5.2f}")
    print(f"    la métrica propia es mejor o igual que Fréchet en {np.mean(ours <= frechet + 1e-12) * 100:.0f} % "
          f"de las curvas (estrictamente mejor en {np.mean(ours < frechet - 1e-12) * 100:.0f} %)")


def benchmark_fit(dbs: list[MotionDB]) -> None:
    from tests.test_metric import BEAN, ELLIPSE  # formas de referencia

    fams = [f for f in (FOUR_BAR, GEARED_FIVE_BAR) if f.name in {d.family for d in dbs}]
    source = make_candidate_source(dbs)
    print("  ajuste completo (consulta + refinamiento) frente a la búsqueda aleatoria:")
    for name, target in (("elipse", ELLIPSE), ("frijol", BEAN)):
        row = []
        for label, src in (("aleatoria", None), ("con base", source)):
            t0 = time.perf_counter()
            res = fit(target, fams, candidate_source=src)
            dt = time.perf_counter() - t0
            row.append(f"{label}: {res.rms_rel * 100:.2f} % en {dt:.1f} s ({res.family})" if res else f"{label}: sin resultado")
        print(f"    {name:7s}  " + "   |   ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=int, default=3000)
    ap.add_argument("--curves", type=int, default=40)
    ap.add_argument("--five-bar", action="store_true", help="incluir la base del 5-bar con engranajes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dbs = []
    for family in ("four_bar", "geared_five_bar") if args.five_bar else ("four_bar",):
        t0 = time.perf_counter()
        db = build_database(family, n_target=args.entries, n_samples=args.entries * 40)
        print(f"[{family}] base de {len(db)} entradas construida en {time.perf_counter() - t0:.1f} s")
        benchmark_queries(db, args.curves, args.seed)
        dbs.append(db)
    benchmark_fit(dbs)


if __name__ == "__main__":
    main()
