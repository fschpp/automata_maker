"""Base de movimientos (M3, paper §4.1): muestreo de mecanismos y consulta curva → mecanismo.

Una base guarda, para una familia, curvas normalizadas de mecanismos factibles y sus parámetros de
forma (bancada de largo 1). Se construye muestreando al azar (descartando los que no completan el
ciclo, vía ``Family.sample_local``) y quedándose con un subconjunto tipo Poisson-disk: ningún par
de muestras elegidas está más cerca que un radio en el espacio del descriptor, lo que reparte la
base de forma pareja en el espacio de *formas* y no en el de parámetros.

Diferencia con el paper: el radio de Poisson-disk se mide en el espacio del descriptor (features y
armónicos de Fourier, barato, indexado con KD-tree) y no con la métrica completa, que sí se usa
para ordenar las candidatas preseleccionadas al consultar.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mechchar.db.metric import (
    N_FEATURES,
    SELF_INTERSECTION_PENALTY,
    descriptor,
    features,
    normalize,
    self_intersections,
    shape_distance_many,
)
from mechchar.optim.curves import bbox_diag, chamfer_rms
from mechchar.optim.families import family_by_name
from mechchar.optim.fitcore import MAX_LINK_RATIO, Candidate, CandidateSource, Family, Variant

N_DENSE = 256  # puntos con los que se evalúa cada curva antes de normalizarla
DB_VERSION = 1


@dataclass
class MotionDB:
    family: str
    pts: np.ndarray  # (K, N_NORM, 2) curvas normalizadas (centroide 0, radio RMS 1)
    center: np.ndarray  # (K, 2) centroide en el marco local (curva = center + scale * pts)
    scale: np.ndarray  # (K,)
    extent: np.ndarray  # (K,) mayor dimensión del mecanismo, en unidades locales
    shape: np.ndarray  # (K, m) parámetros de forma de la familia
    branch: np.ndarray  # (K,)
    ratio: np.ndarray  # (K,)
    feats: np.ndarray  # (K, N_FEATURES)
    desc: np.ndarray  # (K, D)
    tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.desc)

    def __len__(self) -> int:
        return len(self.pts)

    def variant(self, i: int) -> Variant:
        return Variant(int(self.branch[i]), float(self.ratio[i]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, version=DB_VERSION, family=self.family, pts=self.pts, center=self.center,
            scale=self.scale, extent=self.extent, shape=self.shape, branch=self.branch,
            ratio=self.ratio, feats=self.feats, desc=self.desc,
        )

    @classmethod
    def load(cls, path: str | Path) -> MotionDB:
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != DB_VERSION:
                raise ValueError("versión de base de movimientos incompatible")
            return cls(
                str(z["family"]), z["pts"], z["center"], z["scale"], z["extent"], z["shape"],
                z["branch"], z["ratio"], z["feats"], z["desc"],
            )


# --- construcción -------------------------------------------------------------------------------


def _sample_chunk(args: tuple[str, int, int]) -> dict[str, np.ndarray] | None:
    """Trabajo de un proceso: muestras válidas, ya normalizadas y con su descriptor."""
    name, seed, k = args
    sample = family_by_name(name).sample_local(np.random.default_rng(seed), k, N_DENSE)
    if sample is None:
        return None
    rows: list[int] = []
    out: dict[str, list] = {key: [] for key in ("pts", "center", "scale", "feats", "desc")}
    for i, curve in enumerate(sample.curves):
        try:
            nz = normalize(curve)
        except ValueError:  # curva degenerada
            continue
        f = features(nz.pts, with_intersections=False)
        rows.append(i)
        out["pts"].append(nz.pts)
        out["center"].append(nz.center)
        out["scale"].append(nz.scale)
        out["feats"].append(f)
        out["desc"].append(descriptor(nz.pts, f))
    if not rows:
        return None
    res = {key: np.array(val) for key, val in out.items()}
    idx = np.array(rows)
    res["extent"] = sample.extent[idx]
    res["shape"] = sample.shape[idx]
    res["branch"] = np.array([sample.variants[i].branch for i in rows])
    res["ratio"] = np.array([sample.variants[i].ratio for i in rows])
    return res


def poisson_disk(desc: np.ndarray, radius: float, order: np.ndarray) -> np.ndarray:
    """Selección voraz tipo Poisson-disk: recorre ``order`` y se queda con cada punto que no está
    a menos de ``radius`` de uno ya elegido."""
    tree = cKDTree(desc)
    alive = np.ones(len(desc), dtype=bool)
    chosen: list[int] = []
    for i in order:
        if alive[i]:
            chosen.append(int(i))
            alive[tree.query_ball_point(desc[i], radius)] = False
    return np.array(chosen, dtype=int)


def poisson_disk_select(
    desc: np.ndarray, target: int, rng: np.random.Generator, iters: int = 12
) -> np.ndarray:
    """Elige ``target`` puntos con la mayor separación mínima que alcanza (bisección del radio)."""
    if len(desc) <= target:
        return np.arange(len(desc))
    order = rng.permutation(len(desc))
    hi = float(np.linalg.norm(np.ptp(desc, axis=0)))
    lo = hi * 1e-5
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if len(poisson_disk(desc, mid, order)) >= target:
            lo = mid
        else:
            hi = mid
    # cualquier subconjunto de un conjunto Poisson-disk conserva la separación mínima
    return poisson_disk(desc, lo, order)[:target]


def build_database(
    family: str,
    n_target: int = 3000,
    n_samples: int = 100_000,
    seed: int = 0,
    workers: int | None = None,
    chunk: int = 4000,
) -> MotionDB:
    """Construye la base de una familia. ``workers`` = 1 corre en el proceso actual."""
    jobs = [(family, seed * 10_007 + i, chunk) for i in range(max(1, n_samples // chunk))]
    if workers == 1:
        parts = [_sample_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_sample_chunk, jobs))
    good = [p for p in parts if p is not None]
    if not good:
        raise RuntimeError(f"no se obtuvo ninguna muestra válida para {family!r}")
    cat = {key: np.concatenate([p[key] for p in good]) for key in good[0]}
    keep = poisson_disk_select(cat["desc"], n_target, np.random.default_rng(seed))
    sel = {key: val[keep] for key, val in cat.items()}
    sel["feats"][:, N_FEATURES - 1] = [self_intersections(p) for p in sel["pts"]]
    return MotionDB(
        family, sel["pts"], sel["center"], sel["scale"], sel["extent"], sel["shape"],
        sel["branch"], sel["ratio"], sel["feats"], sel["desc"],
    )


def default_db_path(family: str) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "mechchar" / f"{family}_db_v{DB_VERSION}.npz"


def load_or_build(family: str, path: Path | None = None, **kwargs: object) -> MotionDB:
    """Carga la base desde el caché; si no existe (o no es compatible) la construye y la guarda."""
    path = path or default_db_path(family)
    if path.exists():
        try:
            return MotionDB.load(path)
        except (ValueError, KeyError, OSError):
            pass
    db = build_database(family, **kwargs)  # type: ignore[arg-type]
    db.save(path)
    return db


# --- consulta -----------------------------------------------------------------------------------


@dataclass
class Match:
    index: int  # posición en la base
    distance: float  # forma + penalización por autointersecciones distintas
    shape_distance: float
    scale: float  # factor de escala (marco local -> mundo); d_mundo = scale
    rotation: float  # ángulo de la bancada en el mundo
    origin: np.ndarray  # pivote de la bancada en el mundo
    curve: np.ndarray  # curva del mecanismo ya llevada al mundo (N_NORM, 2)


def query(
    db: MotionDB,
    curve: np.ndarray,
    k: int = 3,
    prescreen: int = 150,
    max_ratio: float | None = MAX_LINK_RATIO,
    exhaustive: bool = False,
) -> list[Match]:
    """Los ``k`` mecanismos de la base que mejor se parecen a ``curve`` (cerrada, (m, 2)).

    Se preseleccionan ``prescreen`` candidatos por sentido con el KD-tree y se ordenan con la
    métrica completa. ``max_ratio`` descarta los que no caben (eslabón mayor que ese múltiplo del
    tamaño de la curva); ``exhaustive`` compara contra toda la base (referencia)."""
    q = normalize(curve)
    f_q = features(q.pts)
    size = bbox_diag(curve)
    if exhaustive:
        cand = np.arange(len(db))
    else:
        m = min(prescreen, len(db))
        i1 = np.atleast_1d(db.tree.query(descriptor(q.pts, f_q), k=m)[1])
        i2 = np.atleast_1d(db.tree.query(descriptor(q.pts, f_q, reverse=True), k=m)[1])
        cand = np.unique(np.concatenate([i1, i2]))
    sm = shape_distance_many(q.pts, db.pts[cand])
    total = sm.distance + SELF_INTERSECTION_PENALTY * np.abs(f_q[5] - db.feats[cand, 5])
    f_scale = q.scale / db.scale[cand]
    fits = np.ones(len(cand), dtype=bool)
    if max_ratio is not None:
        fits = f_scale * db.extent[cand] <= max_ratio * size
    qc = complex(q.center[0], q.center[1])
    out: list[Match] = []
    for j in np.argsort(total):
        if not fits[j]:
            continue
        i = int(cand[j])
        rot = np.exp(1j * sm.rotation[j])
        c_e = complex(db.center[i, 0], db.center[i, 1])
        origin = qc - f_scale[j] * rot * c_e
        z = (db.pts[i, :, 0] + 1j * db.pts[i, :, 1]) * rot * q.scale + qc
        out.append(
            Match(
                i, float(total[j]), float(sm.distance[j]), float(f_scale[j]),
                float(sm.rotation[j]), np.array([origin.real, origin.imag]),
                np.column_stack([z.real, z.imag]),
            )
        )
        if len(out) == k:
            break
    return out


def make_candidate_source(dbs: Sequence[MotionDB], prescreen: int = 150) -> CandidateSource:
    """Fuente de candidatos para ``fitcore.fit`` a partir de una base por familia."""
    by_family = {db.family: db for db in dbs}

    def source(family: Family, y: np.ndarray, n_keep: int) -> list[Candidate]:
        db = by_family.get(family.name)
        if db is None:
            return []
        size = bbox_diag(y)
        cands = []
        for m in query(db, y, k=n_keep, prescreen=prescreen):
            v = db.variant(m.index)
            p0 = family.assemble(db.shape[m.index], v, m.scale, m.rotation, m.origin)
            cands.append(Candidate(p0, v, chamfer_rms(m.curve, y) / size, m.curve))
        return cands

    return source


@dataclass
class Suggestion:
    """Un candidato listo para mostrar o refinar: familia, parámetros y curva en el mundo."""

    family: str
    variant: Variant
    p0: np.ndarray  # vector de parámetros completo del mecanismo, ya colocado en el mundo
    curve: np.ndarray  # curva que traza (N_NORM, 2)
    distance: float  # métrica total contra la curva consultada
    rel_error: float  # error chamfer relativo al tamaño de la curva


def suggest(
    dbs: Sequence[MotionDB], curve: np.ndarray, k: int = 3, prescreen: int = 150
) -> list[Suggestion]:
    """Los ``k`` mejores candidatos entre todas las bases, ordenados por la métrica."""
    y = np.asarray(curve, dtype=float)
    size = bbox_diag(y)
    found: list[tuple[float, MotionDB, Match]] = []
    for db in dbs:
        for m in query(db, y, k=k, prescreen=prescreen):
            found.append((m.distance, db, m))
    found.sort(key=lambda item: item[0])
    out: list[Suggestion] = []
    for _, db, m in found[:k]:
        v = db.variant(m.index)
        p0 = family_by_name(db.family).assemble(db.shape[m.index], v, m.scale, m.rotation, m.origin)
        rel = chamfer_rms(m.curve, y) / size
        out.append(Suggestion(db.family, v, p0, m.curve, m.distance, rel))
    return out
