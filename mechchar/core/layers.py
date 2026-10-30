"""Capas sin colisión (M6, paper §6.2). Sin GTK.

Todas las piezas se mueven en planos paralelos, así que dos piezas que se pisan en algún
instante del ciclo tienen que ir en capas distintas. Este módulo recibe, por cada pieza, su
contorno proyectado en cada paso del ciclo y devuelve la capa de cada una:

* ``collisions``: pares de piezas que se pisan en el mismo paso (shapely + STRtree sobre las cajas
  de lo que barre cada pieza, y recién después el chequeo exacto paso por paso). Tocarse por el
  borde no es pisarse; ``clearance`` agrega una holgura (por ejemplo la de impresión).
* ``Pin``: un pasador atraviesa todas las capas entre la más baja y la más alta de las piezas que
  une (``members``). Ninguna otra pieza puede estar entre ellas **ni en las capas de los
  extremos** si en algún paso pisa al pasador (``pin_conflicts``): es el «C no entre A y B» del
  paper. Los extremos se prohíben también porque en la capa de una pieza del pasador ya está su
  agujero; con un pasador dentro de sus piezas es lo mismo que prohibir solo «en medio» (esa
  pieza chocaría con la del extremo), y si el pasador no está dentro de ellas es la opción segura.
* ``meshes``: pares de piezas que engranan (dos engranajes vecinos de una cadena). Tienen que ir
  en la **misma** capa y sus dientes, que se tocan a propósito, no cuentan como choque entre sí.
* ``assign_layers``: CP-SAT. Prueba con 1, 2, … capas hasta que haya solución (el mínimo) y con
  ese número minimiza el largo total de los pasadores (suma de ``max − min`` de capa de sus
  piezas) con un límite de tiempo.
* ``layer_offsets``: el espesor de una capa es el de su pieza más gruesa; devuelve el espesor y
  el ``z`` (de la cara inferior) de cada capa.

Las capas que ya trae una pieza (``Body.layer``, por ejemplo la del tren de engranajes de M5) se
respetan.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model
from shapely import STRtree, prepare
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

OVERLAP_TOL = 1e-9  # área mínima de intersección para contar como pisarse (unidades del mundo²)


@dataclass(frozen=True)
class Body:
    """Una pieza: su contorno (N, 2) en cada paso del ciclo, el espesor y, opcional, la capa."""

    name: str
    shapes: tuple[np.ndarray, ...]
    thickness: float = 1.0
    layer: int | None = None

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError(f"{self.name}: hace falta el contorno de al menos un paso")
        if not (math.isfinite(self.thickness) and self.thickness > 0.0):
            raise ValueError(f"{self.name}: el espesor debe ser positivo")
        if self.layer is not None and self.layer < 0:
            raise ValueError(f"{self.name}: la capa no puede ser negativa")


@dataclass(frozen=True)
class Pin:
    """Pasador (o eje) que une las piezas ``members``; ``path`` es su posición (n_pasos, 2)."""

    members: tuple[str, ...]
    path: np.ndarray
    radius: float = 0.1

    def __post_init__(self) -> None:
        if len(set(self.members)) < 2:
            raise ValueError("un pasador une al menos dos piezas distintas")
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("el radio del pasador debe ser positivo")


@dataclass(frozen=True)
class LayerPlan:
    layers: dict[str, int]  # nombre de la pieza -> capa (0 = la de abajo)
    n_layers: int
    pin_length: int  # suma de capas que cruza cada pasador
    optimal: bool  # el largo de pasadores es el mínimo (no se cortó por tiempo)
    problems: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.problems


def _polys(body: Body) -> list[Polygon]:
    out = []
    for s in body.shapes:
        p = Polygon(np.asarray(s, dtype=float))
        out.append(p if p.is_valid else p.buffer(0))
    return out


def _check_steps(bodies: Sequence[Body]) -> int:
    names = [b.name for b in bodies]
    if len(set(names)) != len(names):
        raise ValueError("hay piezas con el mismo nombre")
    steps = {len(b.shapes) for b in bodies}
    if len(steps) > 1:
        raise ValueError(f"las piezas tienen distinta cantidad de pasos: {sorted(steps)}")
    return steps.pop() if steps else 0


def collisions(
    bodies: Sequence[Body],
    clearance: float = 0.0,
    tol: float = OVERLAP_TOL,
    ignore: Collection[frozenset[int]] = (),
) -> set[frozenset[int]]:
    """Pares (índices) de piezas que se pisan en algún paso del ciclo.

    Con ``clearance`` > 0 también cuentan las que quedan a menos de esa distancia. Los pares de
    ``ignore`` (por ejemplo los engranajes que engranan) no se consideran."""
    if clearance < 0.0:
        raise ValueError("la holgura no puede ser negativa")
    _check_steps(bodies)
    polys = [_polys(b) for b in bodies]
    swept = [unary_union(ps).buffer(clearance) for ps in polys]  # lo que barre cada pieza
    tree = STRtree(swept)
    out: set[frozenset[int]] = set()
    for i, sw in enumerate(swept):
        for j in tree.query(sw, predicate="intersects"):
            j = int(j)
            if j <= i or frozenset((i, j)) in out or frozenset((i, j)) in ignore:
                continue
            for p, q in zip(polys[i], polys[j], strict=True):
                grown = p.buffer(clearance) if clearance else p
                prepare(grown)
                if grown.intersection(q).area > tol or (clearance and grown.intersects(q)):
                    out.add(frozenset((i, j)))
                    break
    return out


def pin_conflicts(
    bodies: Sequence[Body], pins: Sequence[Pin], clearance: float = 0.0
) -> list[tuple[int, int]]:
    """Pares (pasador, pieza) en los que la pieza, que no es del pasador, pisa al pasador en
    algún paso. Esa pieza no puede quedar en ninguna capa de las que cruza el pasador."""
    n_steps = _check_steps(bodies)
    index = {b.name: i for i, b in enumerate(bodies)}
    polys = [_polys(b) for b in bodies]
    out: list[tuple[int, int]] = []
    for k, pin in enumerate(pins):
        path = np.asarray(pin.path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) != n_steps:
            raise ValueError(f"el recorrido del pasador {k} no tiene {n_steps} pasos de (x, y)")
        for name in pin.members:
            if name not in index:
                raise ValueError(f"el pasador {k} une una pieza que no existe: {name!r}")
        discs = [Point(*p).buffer(pin.radius + clearance) for p in path]
        for i in range(len(bodies)):
            if bodies[i].name in pin.members:
                continue
            if any(d.intersects(p) for d, p in zip(discs, polys[i], strict=True)):
                out.append((k, i))
    return out


def pin_length(layers: dict[str, int], pins: Sequence[Pin]) -> int:
    """Suma, sobre los pasadores, de las capas que cruza cada uno (``max − min``)."""
    total = 0
    for pin in pins:
        ls = [layers[n] for n in pin.members]
        total += max(ls) - min(ls)
    return total


def _solve(
    bodies: Sequence[Body],
    pairs: set[frozenset[int]],
    pins: Sequence[Pin],
    conflicts: Sequence[tuple[int, int]],
    m: int,
    time_limit: float,
    minimize_pins: bool,
    same_layer: Collection[frozenset[int]] = (),
) -> tuple[dict[str, int], int, bool] | None:
    index = {b.name: i for i, b in enumerate(bodies)}
    model = cp_model.CpModel()
    lay = [model.NewIntVar(0, m - 1, f"l{i}") for i in range(len(bodies))]
    for i, b in enumerate(bodies):
        if b.layer is not None:
            if b.layer >= m:
                return None
            model.Add(lay[i] == b.layer)
    for pair in pairs:
        i, j = sorted(pair)
        model.Add(lay[i] != lay[j])
    for pair in same_layer:
        i, j = sorted(pair)
        model.Add(lay[i] == lay[j])
    lows, highs = [], []
    for pin in pins:
        ids = [index[n] for n in pin.members]
        for a in range(len(ids)):
            for c in range(a + 1, len(ids)):
                model.Add(lay[ids[a]] != lay[ids[c]])  # dos piezas en un pasador no comparten capa
        lo = model.NewIntVar(0, m - 1, "lo")
        hi = model.NewIntVar(0, m - 1, "hi")
        model.AddMinEquality(lo, [lay[i] for i in ids])
        model.AddMaxEquality(hi, [lay[i] for i in ids])
        lows.append(lo)
        highs.append(hi)
    for k, i in conflicts:
        below = model.NewBoolVar(f"below{k}_{i}")
        above = model.NewBoolVar(f"above{k}_{i}")
        model.Add(lay[i] < lows[k]).OnlyEnforceIf(below)
        model.Add(lay[i] > highs[k]).OnlyEnforceIf(above)
        model.AddBoolOr([below, above])  # nunca en [lo, hi]
    total = sum(highs[k] - lows[k] for k in range(len(pins))) if pins else 0
    if minimize_pins and pins:
        model.Minimize(total)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    layers = {b.name: int(solver.Value(lay[i])) for i, b in enumerate(bodies)}
    length = sum(int(solver.Value(highs[k]) - solver.Value(lows[k])) for k in range(len(pins)))
    return layers, length, status == cp_model.OPTIMAL


def assign_layers(
    bodies: Sequence[Body],
    pins: Sequence[Pin] = (),
    *,
    max_layers: int = 12,
    clearance: float = 0.0,
    time_limit: float = 10.0,
    meshes: Sequence[tuple[str, str]] = (),
) -> LayerPlan:
    """Capa de cada pieza con el mínimo de capas y, con ese mínimo, pasadores lo más cortos.

    Si con ``max_layers`` no hay solución, el plan trae el motivo en ``problems`` y ninguna
    capa inventada."""
    if max_layers < 1:
        raise ValueError("max_layers debe ser al menos 1")
    _check_steps(bodies)
    if not bodies:
        return LayerPlan({}, 0, 0, True)
    index = {b.name: i for i, b in enumerate(bodies)}
    same: set[frozenset[int]] = set()
    for a, b in meshes:
        for n in (a, b):
            if n not in index:
                raise ValueError(f"un par de engranajes nombra una pieza que no existe: {n!r}")
        if a == b:
            raise ValueError(f"{a!r} no engrana consigo mismo")
        same.add(frozenset((index[a], index[b])))
    pairs = collisions(bodies, clearance, ignore=same)
    for pin in pins:
        for n in pin.members:
            if n not in index:
                raise ValueError(f"un pasador une una pieza que no existe: {n!r}")
    conflicts = pin_conflicts(bodies, pins, clearance)
    floor = max([b.layer + 1 for b in bodies if b.layer is not None], default=1)
    start = max(floor, 2 if pairs or pins else 1)  # (los engranajes engranados comparten capa)
    for m in range(start, max_layers + 1):
        found = _solve(bodies, pairs, pins, conflicts, m, time_limit, False, same)
        if found is None:
            continue
        best = _solve(bodies, pairs, pins, conflicts, m, time_limit, True, same)
        layers, _, optimal = best if best is not None else found
        used = sorted(set(layers.values()))
        if all(b.layer is None for b in bodies):
            # Capas vacías no se fabrican: se compactan sin cambiar el orden. Si alguna pieza
            # trae su capa se deja todo como está para no moverla.
            remap = {old: new for new, old in enumerate(used)}
            layers = {k: remap[v] for k, v in layers.items()}
            return LayerPlan(layers, len(used), pin_length(layers, pins), optimal)
        return LayerPlan(layers, max(layers.values()) + 1, pin_length(layers, pins), optimal)
    return LayerPlan(
        {},
        0,
        0,
        False,
        (f"no hay una asignación de capas con {max_layers} capas como máximo",),
    )


def layer_offsets(plan: LayerPlan, bodies: Sequence[Body]) -> tuple[list[float], list[float]]:
    """Espesor de cada capa (el de su pieza más gruesa) y ``z`` de su cara inferior."""
    if not plan.ok:
        raise ValueError("el plan de capas tiene problemas")
    thickness = [0.0] * plan.n_layers
    for b in bodies:
        k = plan.layers[b.name]
        thickness[k] = max(thickness[k], b.thickness)
    z = [0.0]
    for t in thickness[:-1]:
        z.append(z[-1] + t)
    return thickness, z
