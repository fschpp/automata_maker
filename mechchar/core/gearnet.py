"""Red de engranajes que une los ejes de entrada de todos los mecanismos a uno solo (M5). Sin GTK.

Cada mecanismo gira sobre un eje que fija su posición, así que **los ejes no se mueven**: mover
uno cambiaría el punto de actuación y el ajuste ya hecho. En cambio se elige el módulo según las
distancias y se absorbe la holgura con engranajes intermedios, que se colocan fuera de la recta
(ver ``geartrain.connect``).

``plan_train`` arma la red:

* **Qué ejes se unen**: un árbol de expansión mínimo (por distancia) que parte del eje de entrada.
  Cada rama es una conexión, que se resuelve con ``geartrain.connect``.
* **Sentido de giro**: cada eje pide girar igual (``same_direction``) o al revés que el de
  entrada. La relación entre los extremos de una conexión sale de ahí y fija la paridad de sus
  intermedios (impar = igual, par = al revés); se puede forzar la cantidad.
* **Velocidad**: todos los engranajes de los extremos tienen los mismos dientes, así que todos los
  ejes giran a la misma velocidad que el de entrada; los intermedios son libres.
* **Capas**: cada conexión lleva sus propios engranajes en una capa (``z``). Las conexiones que
  llegan al mismo eje quedan apiladas sobre él, como engranajes en un eje común, y si dos cadenas
  se pisan en una capa la segunda pasa a la siguiente. Las capas son solo las del tren: la
  asignación global con las piezas de los mecanismos es de M6.
* **Módulo**: uno solo para toda la red. Si no se da, sale de las distancias: los engranajes de
  los extremos ocupan el 80 % de la menor distancia entre ejes unidos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Phase, Pin
from mechchar.core.geartrain import (
    MIN_TEETH,
    Gear,
    chain_problems,
    connect,
    intermediates_for,
    meshes,
    overlaps,
)

END_FILL = 0.8  # fracción de la menor distancia entre ejes que ocupan los dos engranajes extremos
DEFAULT_END_TEETH = 16  # dientes de los extremos si no se da el módulo
MAX_INNER_FACTOR = 4  # tope de dientes de un intermedio, en veces los de los extremos
MAX_LAYERS = 8


@dataclass(frozen=True)
class Axis:
    """Eje de entrada de un mecanismo. ``same_direction``: gira igual que el eje de entrada."""

    name: str
    x: float
    y: float
    same_direction: bool = True


@dataclass(frozen=True)
class Connection:
    """Una rama de la red: los engranajes (extremos incluidos) que unen dos ejes."""

    a: int  # índice del eje (el más cercano a la entrada)
    b: int
    gears: tuple[Gear, ...]  # [engranaje en a, intermedios…, engranaje en b]
    layer: int
    same_direction: bool  # lo que se pidió; ``len(gears) - 2`` impar ⇔ True

    @property
    def intermediates(self) -> int:
        return len(self.gears) - 2


@dataclass(frozen=True)
class GearTrain:
    axes: tuple[Axis, ...]
    root: int
    module: float
    end_teeth: int
    connections: tuple[Connection, ...]
    problems: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.problems

    def gears(self) -> list[Gear]:
        return [g for c in self.connections for g in c.gears]


# --- planificación -----------------------------------------------------------------------------


def _distance(p: Axis, q: Axis) -> float:
    return math.hypot(p.x - q.x, p.y - q.y)


def spanning_tree(axes: Sequence[Axis], root: int = 0) -> list[tuple[int, int]]:
    """Ramas (a, b) del árbol de expansión mínimo desde ``root`` (algoritmo de Prim), en el orden
    en que se agregan: ``a`` ya está unido y ``b`` es el eje nuevo. Empata por índice."""
    n = len(axes)
    if not 0 <= root < n:
        raise ValueError("el eje de entrada no existe")
    in_tree = {root}
    edges: list[tuple[int, int]] = []
    while len(in_tree) < n:
        best = min(
            ((_distance(axes[a], axes[b]), a, b) for a in sorted(in_tree) for b in range(n)
             if b not in in_tree),
        )
        edges.append((best[1], best[2]))
        in_tree.add(best[2])
    return edges


def _check_axes(axes: Sequence[Axis]) -> None:
    for i, p in enumerate(axes):
        if not all(math.isfinite(v) for v in (p.x, p.y)):
            raise ValueError(f"el eje {p.name!r} tiene una posición no finita")
        for q in axes[i + 1 :]:
            if _distance(p, q) < 1e-9:
                raise ValueError(f"los ejes {p.name!r} y {q.name!r} coinciden")


def module_for(axes: Sequence[Axis], edges: Sequence[tuple[int, int]]) -> float:
    """Módulo por defecto: con ``DEFAULT_END_TEETH`` dientes en los extremos, estos ocupan
    ``END_FILL`` de la menor distancia entre ejes unidos."""
    d_min = min(_distance(axes[a], axes[b]) for a, b in edges)
    return END_FILL * d_min / DEFAULT_END_TEETH


def _inner_candidates(k: int, z_end: int):
    """Dientes de los ``k`` intermedios por probar, de los más chicos a los más grandes: primero
    todos iguales y después con una diferencia de hasta 2 dientes entre vecinos."""
    top = MAX_INNER_FACTOR * z_end
    if k == 1:
        for z in range(MIN_TEETH, top + 1):
            yield [z]
    elif k == 2:
        for z in range(MIN_TEETH, top + 1):
            yield [z, z]
        for z in range(MIN_TEETH, top + 1):
            for dz in (1, 2):
                yield [z, z + dz]
                yield [z + dz, z]
    else:
        for z in range(MIN_TEETH, top + 1):
            yield [z] * k


def _connection_chain(
    pa: Axis, pb: Axis, z_end: int, module: float, layer_z: float, k: int, others: list[Gear]
) -> list[Gear] | None:
    """La primera cadena válida de ``k`` intermedios entre los ejes ``pa`` y ``pb`` en la capa
    ``layer_z`` que no pise a ``others`` (los engranajes ya puestos en esa capa)."""
    a = Gear(pa.x, pa.y, z_end, module, layer_z)
    b = Gear(pb.x, pb.y, z_end, module, layer_z)
    if not _free_ends(a, b, others):
        return None  # los extremos ya chocan en esta capa: ninguna cadena sirve, ni se prueba
    if k == 0:
        return [a, b] if meshes(a, b) and not _clash([a, b], others, {frozenset((0, 1))}) else None
    for inner in _inner_candidates(k, z_end):
        for side in (1, -1):
            placed = connect(a, b, inner, side, avoid=others)
            if placed is not None:
                return [a, *placed, b]
    return None


def _free_ends(a: Gear, b: Gear, others: list[Gear]) -> bool:
    """Los engranajes extremos, que tienen posición y dientes fijos, no pisan a ``others``."""
    return not overlaps([a, b, *others], ignore={frozenset((0, 1))} | {
        frozenset((i, j)) for i in range(2, 2 + len(others)) for j in range(i + 1, 2 + len(others))
    })


def _clash(new: list[Gear], others: list[Gear], neighbours: set[frozenset[int]]) -> bool:
    """La cadena ``new`` se pisa a sí misma (salvo sus vecinos) o con ``others``."""
    pairs = overlaps([*new, *others], ignore=neighbours)
    return bool(pairs)


def plan_train(
    axes: Sequence[Axis],
    root: int = 0,
    *,
    module: float | None = None,
    intermediates: Mapping[tuple[int, int], int] | None = None,
    layer_pitch: float = 1.0,
) -> GearTrain:
    """Une todos los ejes al de entrada ``root`` con una red de engranajes (ver el módulo).

    ``module`` fija el módulo (si no, sale de las distancias). ``intermediates`` fuerza la
    cantidad de intermedios de una conexión, indexada por el par de ejes (en cualquier orden); tiene
    que tener la paridad que pide el sentido de giro, o ``ValueError``. Lo que no se logra
    (ejes demasiado juntos para el módulo, cadena que no cierra) queda en ``problems`` y la
    conexión no aparece en ``connections``."""
    axes = tuple(axes)
    if not axes:
        raise ValueError("hace falta al menos un eje")
    _check_axes(axes)
    if len(axes) == 1:
        return GearTrain(axes, root, module or 1.0, 0, ())
    edges = spanning_tree(axes, root)
    m = module if module is not None else module_for(axes, edges)
    if not (math.isfinite(m) and m > 0.0):
        raise ValueError("el módulo debe ser positivo")
    forced = {frozenset(k): v for k, v in (intermediates or {}).items()}
    d_min = min(_distance(axes[a], axes[b]) for a, b in edges)
    z_end = int(END_FILL * d_min / m)  # dientes de los extremos: ocupan el 80 % de la distancia
    problems: list[str] = []
    if z_end < MIN_TEETH:
        problems.append(
            f"el módulo {m:g} es demasiado grande: con la menor distancia entre ejes "
            f"({d_min:g}) los engranajes de los extremos quedarían con {z_end} dientes "
            f"(mínimo {MIN_TEETH})"
        )
        return GearTrain(axes, root, m, z_end, (), tuple(problems))

    placed: dict[int, list[Gear]] = {}  # capa -> engranajes ya puestos en ella
    connections: list[Connection] = []
    for a, b in edges:
        same = axes[a].same_direction == axes[b].same_direction
        k = forced.get(frozenset((a, b)))
        if k is not None:
            if k < 0 or (k % 2 == 1) != same:
                raise ValueError(
                    f"{k} intermedios entre {axes[a].name!r} y {axes[b].name!r} no dan el "
                    f"sentido pedido ({'el mismo' if same else 'el contrario'}): "
                    f"hacen falta {'impar' if same else 'par'}"
                )
            ks = [k]
        else:
            ks = [intermediates_for(same, 0), intermediates_for(same, 0) + 2]
        chain_found: tuple[int, list[Gear]] | None = None
        for k_try in ks:
            if _connection_chain(axes[a], axes[b], z_end, m, 0.0, k_try, []) is None:
                continue  # esa cantidad de intermedios no cierra ni con la capa vacía
            for layer in range(MAX_LAYERS):
                full = _connection_chain(
                    axes[a], axes[b], z_end, m, layer * layer_pitch, k_try, placed.get(layer, [])
                )
                if full is not None:
                    chain_found = (layer, full)
                    break
            if chain_found is not None:
                break
        if chain_found is None:
            problems.append(
                f"no se pudo unir {axes[a].name!r} con {axes[b].name!r} con módulo {m:g}"
                + (f" y {ks[0]} intermedios" if k is not None else "")
            )
            continue
        layer, full = chain_found
        placed.setdefault(layer, []).extend(full)
        connections.append(Connection(a, b, tuple(full), layer, same))
    return GearTrain(axes, root, m, z_end, tuple(connections), tuple(problems))


# --- verificación ------------------------------------------------------------------------------


def simulate_train(train: GearTrain, n_steps: int = 24) -> np.ndarray:
    """Ángulo de cada eje (n_steps, n_ejes) a lo largo de una vuelta del eje de entrada.

    La red entera es un ensamblaje: cada engranaje pinado a la bancada en su eje, los de un mismo
    eje rígidos entre sí (``Phase`` con relación +1), cada par engranado con ``Phase`` (−z_a/z_b)
    y el eje de entrada con un Input Driver. Se rechaza una red incompleta, porque habría ejes
    que no se mueven."""
    if not train.ok:
        raise ValueError("la red tiene problemas: " + "; ".join(train.problems))
    if not train.connections:
        raise ValueError("la red no tiene conexiones: no hay nada que simular")
    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    by_axis: dict[int, list[int]] = {i: [] for i in range(len(train.axes))}
    for c in train.connections:
        comps = []
        for g in c.gears:
            idx = asm.add_component(f"engranaje{len(asm.components)}", g.x, g.y, 0.0, g.z)
            asm.add(Pin(ground, (g.x, g.y), idx, (0.0, 0.0)))
            comps.append(idx)
        for k in range(len(c.gears) - 1):
            asm.add(Phase(comps[k], comps[k + 1], -c.gears[k].teeth / c.gears[k + 1].teeth))
        by_axis[c.a].append(comps[0])
        by_axis[c.b].append(comps[-1])
    for members in by_axis.values():  # los engranajes de un mismo eje giran juntos
        for other in members[1:]:
            asm.add(Phase(members[0], other, 1.0))
    root_members = by_axis[train.root]
    if root_members:
        asm.add(Drive(root_members[0]))
    result = asm.trace_cycle(
        asm.initial_state(), n_steps, max_jump=math.pi, closure_tol=float("inf")
    )
    if not result.ok:
        raise ValueError(f"la red no se pudo simular: {result.reason}")
    return np.array(
        [[s[3 * by_axis[i][0] + 2] if by_axis[i] else 0.0 for i in range(len(train.axes))]
         for s in result.states]
    )


def train_problems(train: GearTrain) -> list[str]:
    """Revisa la red terminada (de punta a punta, sin confiar en cómo se armó): cada cadena engrana
    y no se pisa, y no hay dos engranajes de la misma capa que se pisen entre conexiones."""
    out = list(train.problems)
    for c in train.connections:
        label = f"{train.axes[c.a].name}–{train.axes[c.b].name}"
        out += [f"{label}: {p}" for p in chain_problems(c.gears)]
    for i, c in enumerate(train.connections):
        for d in train.connections[i + 1 :]:
            for p in overlaps([*c.gears, *d.gears]):
                if (p[0] < len(c.gears)) != (p[1] < len(c.gears)):
                    out.append(
                        f"{train.axes[c.a].name}–{train.axes[c.b].name} y "
                        f"{train.axes[d.a].name}–{train.axes[d.b].name} se pisan"
                    )
    return out
