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
* **Ejes que coinciden**: dos mecanismos cuyos ejes de entrada están en el mismo punto (como las
  dos piernas del bípedo, que usan el mismo ajuste y se superponen en 2D) comparten un eje: no
  hacen falta engranajes entre ellos. Solo hay problema si piden girar en sentidos opuestos, que
  en un mismo eje es imposible.
* **Módulo**: uno solo para toda la red. Si no se da, sale de las distancias: los engranajes de
  los extremos ocupan el 80 % de la menor distancia entre ejes unidos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Phase, Pin, world_point
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
COINCIDENT_TOL = 1e-6  # dos ejes a menos de esto (unidades del mundo) son el mismo eje


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
    # grupos de ejes (índices) que coinciden y comparten un eje; las conexiones usan el primero
    shared: tuple[tuple[int, ...], ...] = field(default=())

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
    for p in axes:
        if not all(math.isfinite(v) for v in (p.x, p.y)):
            raise ValueError(f"el eje {p.name!r} tiene una posición no finita")


def representatives(axes: Sequence[Axis]) -> list[int]:
    """Para cada eje, el índice del primer eje que está en su misma posición (él mismo si es
    único): los que coinciden comparten un eje."""
    rep: list[int] = []
    for i, p in enumerate(axes):
        rep.append(next((j for j in range(i) if _distance(axes[j], p) < COINCIDENT_TOL), i))
    return rep


def input_axis(
    asm: Assembly, state: np.ndarray, drive: Drive | None = None
) -> tuple[float, float] | None:
    """Posición del eje de entrada de un mecanismo armado: el punto de la bancada donde está
    pinada la pieza que lleva el Input Driver (la manivela, o el engranaje de entrada del 5-bar),
    o ``None`` si el ensamblaje no tiene Input Driver o esa pieza no gira sobre un punto fijo.
    Con varios mecanismos en un mismo ensamblaje (``MultiRig``) ``drive`` dice cuál; por defecto
    se usa el primero."""
    if drive is None:
        drive = next((c for c in asm.constraints if isinstance(c, Drive)), None)
    if drive is None:
        return None
    fixed = {c.comp for c in asm.constraints if isinstance(c, FixedState)}
    for c in asm.constraints:
        if not isinstance(c, Pin):
            continue
        if c.a == drive.comp and c.b in fixed:
            p = world_point(state, c.a, c.pa)
            return float(p[0]), float(p[1])
        if c.b == drive.comp and c.a in fixed:
            p = world_point(state, c.b, c.pb)
            return float(p[0]), float(p[1])
    return None


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
    conexión no aparece en ``connections``. Los ejes que coinciden comparten un eje (``shared``)
    y las conexiones se indexan por el primero de cada grupo."""
    axes = tuple(axes)
    if not axes:
        raise ValueError("hace falta al menos un eje")
    if not 0 <= root < len(axes):
        raise ValueError("el eje de entrada no existe")
    _check_axes(axes)
    rep = representatives(axes)
    unique = sorted(set(rep))
    shared = tuple(
        tuple(i for i in range(len(axes)) if rep[i] == r)
        for r in unique
        if rep.count(r) > 1
    )
    problems: list[str] = []
    for group in shared:
        if len({axes[i].same_direction for i in group}) > 1:
            names = ", ".join(repr(axes[i].name) for i in group)
            problems.append(
                f"los ejes {names} coinciden pero piden girar en sentidos opuestos: en un mismo "
                "eje no se puede; hay que separarlos"
            )
    if len(unique) == 1:
        return GearTrain(axes, root, module or 1.0, 0, (), tuple(problems), shared)

    sub = [axes[r] for r in unique]
    index = {r: k for k, r in enumerate(unique)}
    forced = {
        frozenset(index[rep[i]] for i in k): v for k, v in (intermediates or {}).items()
    }
    result = _plan_unique(sub, index[rep[root]], module, forced, layer_pitch)
    connections = tuple(
        Connection(unique[c.a], unique[c.b], c.gears, c.layer, c.same_direction)
        for c in result.connections
    )
    return GearTrain(
        axes, root, result.module, result.end_teeth, connections,
        tuple(problems) + result.problems, shared,
    )


def _plan_unique(
    axes: Sequence[Axis],
    root: int,
    module: float | None,
    forced: Mapping[frozenset[int], int],
    layer_pitch: float,
) -> GearTrain:
    """``plan_train`` para ejes todos en posiciones distintas."""
    axes = tuple(axes)
    edges = spanning_tree(axes, root)
    m = module if module is not None else module_for(axes, edges)
    if not (math.isfinite(m) and m > 0.0):
        raise ValueError("el módulo debe ser positivo")
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
    rep_ = representatives(train.axes)
    if not train.connections:  # un solo eje (o todos coinciden): giran juntos con el de entrada
        turn = 2.0 * math.pi * np.arange(n_steps) / n_steps
        return np.tile(turn[:, None], (1, len(train.axes)))
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
    root_members = by_axis[rep_[train.root]]
    if root_members:
        asm.add(Drive(root_members[0]))
    result = asm.trace_cycle(
        asm.initial_state(), n_steps, max_jump=math.pi, closure_tol=float("inf")
    )
    if not result.ok:
        raise ValueError(f"la red no se pudo simular: {result.reason}")
    return np.array(
        [
            [s[3 * by_axis[rep_[i]][0] + 2] if by_axis[rep_[i]] else 0.0
             for i in range(len(train.axes))]
            for s in result.states
        ]
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


def describe_train(train: GearTrain) -> tuple[str, bool]:
    """Resumen de la red en una frase para el panel, y si está bien (``False`` si hay problemas).
    Con todos los ejes en el mismo punto no hay engranajes: se dice que comparten el eje."""
    if train.problems:
        return " ".join(train.problems), False
    n = len(train.axes)
    if not train.connections:
        if n == 1:
            return "Hay un solo mecanismo: no hace falta ningún engranaje.", True
        x, y = train.axes[train.root].x, train.axes[train.root].y
        return (
            f"Los {n} mecanismos comparten el eje de entrada en ({x:.2f}, {y:.2f}): un solo "
            "eje, sin engranajes.",
            True,
        )
    layers = len({c.layer for c in train.connections})
    text = (
        f"{len(train.gears())} engranajes en {layers} capa{'s' if layers != 1 else ''}, "
        f"módulo {train.module:.3g}, extremos de {train.end_teeth} dientes."
    )
    if train.shared:
        text += f" {len(train.shared)} eje{'s' if len(train.shared) != 1 else ''} compartido" + (
            "s." if len(train.shared) != 1 else "."
        )
    return text, True
