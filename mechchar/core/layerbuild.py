"""Del rig y la red de engranajes al problema de capas (M6, bundle 2). Sin GTK.

``build_layer_problem`` recorre un ciclo del ``MultiRig`` y arma lo que ``layers.assign_layers``
necesita: las piezas con su contorno en cada paso, los pasadores con su recorrido y los pares de
engranajes que engranan.

* **Piezas del personaje**: su contorno (``Character.part_polygon``), fijas o no.
* **Engranajes del mecanismo** (el 5-bar): los dos engranajes engranan entre sí, así que van como
  ``meshes`` (misma capa, sin chocar entre ellos).
* **Eslabones de los mecanismos**: los eslabones son segmentos, no tienen polígono; se les da
  ancho (``link_width``): el casco convexo de los puntos donde se clavan sus pasadores, con ese
  ancho. Un engranaje del mecanismo (el 5-bar con dos engranajes) suma su disco. Las bancadas de
  los mecanismos (componentes de estado fijo) no son piezas: son el soporte (M8).
* **Pasadores**: cada pasador entre dos piezas con cuerpo; los que comparten el mismo punto de
  una pieza (tres eslabones en un mismo pasador) son uno solo con varios miembros. Un pasador
  contra una pieza sin cuerpo (la bancada) es un eje que entra a la pared y no se modela acá.
* **Red de engranajes** (opcional): cada engranaje con sus dientes girando con el eje de entrada,
  los vecinos de cada cadena como ``meshes``, y un eje común por cada eje con engranajes que
  une los engranajes de ese eje y las manivelas de los mecanismos que lo comparten.

No se asigna ninguna capa acá (eso es ``layers.assign_layers``) ni se usa ``Part.z`` del
personaje como restricción.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

from mechchar.core.assembly import FixedState, PointOnLine, world_point
from mechchar.core.assembly import Pin as PinConstraint
from mechchar.core.gearlink import RigAxes
from mechchar.core.gearnet import GearTrain, representatives
from mechchar.core.gearshape import gear_outline, gear_rotations
from mechchar.core.layers import Body, Pin
from mechchar.core.rig import MultiRig

CYCLE_RESOLUTION = 360  # pasos con que se recorre el ciclo antes de tomar los pedidos
KEY_DECIMALS = 9  # dos puntos locales que coinciden hasta aquí son el mismo pasador


@dataclass(frozen=True)
class LayerProblem:
    bodies: tuple[Body, ...]
    pins: tuple[Pin, ...]
    meshes: tuple[tuple[str, str], ...]
    n_steps: int


def _ring(geom) -> np.ndarray:
    """Contorno (N, 2) de un polígono de shapely, sin repetir el primer punto."""
    return np.asarray(geom.exterior.coords)[:-1]


def _disc_radii(rig: MultiRig) -> dict[int, float]:
    """Radio del engranaje de cada componente que lo tiene (el 5-bar con dos engranajes)."""
    out: dict[int, float] = {}
    for sub in rig.subs:
        mech = sub.mechanism
        if hasattr(mech, "gear_radii") and hasattr(mech, "gear1") and hasattr(mech, "gear2"):
            r1, r2 = mech.gear_radii()
            out[sub.first + mech.gear1] = float(r1)
            out[sub.first + mech.gear2] = float(r2)
    return out


def _local_points(rig: MultiRig) -> dict[int, list[tuple[float, float]]]:
    """Puntos locales de cada componente donde se clava algo (sin repetir)."""
    pts: dict[int, list[tuple[float, float]]] = {}

    def add(comp: int, p) -> None:
        q = (round(float(p[0]), KEY_DECIMALS), round(float(p[1]), KEY_DECIMALS))
        if q not in pts.setdefault(comp, []):
            pts[comp].append(q)

    for c in rig.asm.constraints:
        if isinstance(c, PinConstraint):
            add(c.a, c.pa)
            add(c.b, c.pb)
        elif isinstance(c, PointOnLine):
            add(c.a, c.pa)  # el punto que desliza; la guía es de la otra pieza
    return pts


def build_layer_problem(
    rig: MultiRig,
    train: GearTrain | None = None,
    axes: RigAxes | None = None,
    *,
    n_steps: int = 24,
    link_width: float = 0.12,
    part_thickness: float = 1.0,
    link_thickness: float = 1.0,
    gear_thickness: float = 1.0,
    pin_radius: float = 0.08,
    axle_radius: float = 0.15,
) -> LayerProblem:
    """Piezas, pasadores y engranajes del rig a lo largo de un ciclo.

    ``train`` y ``axes`` (de ``gearlink.plan_rig_train``) agregan la red de engranajes. Lanza
    ``ValueError`` si el ciclo del rig no se puede recorrer."""
    for name, v in (
        ("link_width", link_width),
        ("part_thickness", part_thickness),
        ("link_thickness", link_thickness),
        ("gear_thickness", gear_thickness),
        ("pin_radius", pin_radius),
        ("axle_radius", axle_radius),
    ):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} debe ser positivo")
    if n_steps < 2:
        raise ValueError("hacen falta al menos 2 pasos del ciclo")
    if (train is None) != (axes is None):
        raise ValueError("la red de engranajes necesita el tren y sus ejes (plan_rig_train)")

    # Se recorre con la resolución con que se validan las ramas y de ahí se toman los pasos
    # pedidos: con pocos pasos el salto entre fases consecutivas parecería un cambio de rama.
    fine = max(CYCLE_RESOLUTION, n_steps)
    cyc = rig.asm.trace_cycle(rig.state0, fine)
    if not cyc.ok:
        raise ValueError(f"no se pudo recorrer el ciclo del rig: {cyc.reason}")
    states = cyc.states[[round(k * fine / n_steps) % fine for k in range(n_steps)]]
    fixed = {c.comp for c in rig.asm.constraints if isinstance(c, FixedState)}
    radii = _disc_radii(rig)
    local = _local_points(rig)
    comps = rig.asm.components

    bodies: list[Body] = []
    body_of: dict[int, str] = {}  # componente del rig -> nombre de su pieza

    for i, part in enumerate(rig.character.parts):  # piezas del personaje
        shapes = tuple(rig.character.part_polygon(s, i) for s in states)
        bodies.append(Body(part.name, shapes, part_thickness))
        body_of[i] = part.name

    for sub in rig.subs:  # eslabones de los mecanismos
        for comp in range(sub.first, sub.first + sub.n_comp):
            if comp in fixed or comp not in local:
                continue
            shapes = []
            for s in states:
                geom = MultiPoint([world_point(s, comp, p) for p in local[comp]]).convex_hull
                geom = geom.buffer(link_width / 2.0, quad_segs=4)
                if comp in radii:
                    geom = unary_union(
                        [geom, Point(*world_point(s, comp, (0.0, 0.0))).buffer(radii[comp])]
                    ).convex_hull
                shapes.append(_ring(geom))
            name = f"{sub.name}:{comps[comp].name}"
            bodies.append(Body(name, tuple(shapes), link_thickness))
            body_of[comp] = name

    # pasadores: se unen los que comparten el mismo punto de una misma pieza
    parent: dict[tuple[int, tuple[float, float]], tuple[int, tuple[float, float]]] = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    joints: list[PinConstraint] = [c for c in rig.asm.constraints if isinstance(c, PinConstraint)]

    def key(comp: int, p) -> tuple[int, tuple[float, float]]:
        return (comp, (round(float(p[0]), KEY_DECIMALS), round(float(p[1]), KEY_DECIMALS)))

    for c in joints:
        parent[find(key(c.a, c.pa))] = find(key(c.b, c.pb))
    groups: dict[tuple[int, tuple[float, float]], list[tuple[int, tuple[float, float]]]] = {}
    for c in joints:
        for k in (key(c.a, c.pa), key(c.b, c.pb)):
            members = groups.setdefault(find(k), [])
            if k not in members:
                members.append(k)

    pins: list[Pin] = []
    for members in groups.values():
        with_body = [k for k in members if k[0] in body_of]
        names = sorted({body_of[k[0]] for k in with_body})
        if len(names) < 2:
            continue  # un eje contra la bancada: entra a la pared (M8)
        if all(k[0] in fixed for k in with_body):
            continue  # piezas del chasis unidas entre sí: son una sola cosa rígida
        comp, p = with_body[0]
        path = np.array([world_point(s, comp, p) for s in states])
        pins.append(Pin(tuple(names), path, pin_radius))

    meshes: list[tuple[str, str]] = []
    for sub in rig.subs:  # los dos engranajes de un 5-bar engranan entre sí (R1 + R2 = d)
        mech = sub.mechanism
        if hasattr(mech, "gear_radii") and hasattr(mech, "gear1") and hasattr(mech, "gear2"):
            pair = tuple(body_of.get(sub.first + g) for g in (mech.gear1, mech.gear2))
            if pair[0] is not None and pair[1] is not None:
                meshes.append((pair[0], pair[1]))
    if train is not None and axes is not None:
        _add_gear_train(
            rig, train, axes, states, bodies, pins, meshes, body_of,
            gear_thickness, axle_radius,
        )  # fmt: skip
    return LayerProblem(tuple(bodies), tuple(pins), tuple(meshes), n_steps)


def _add_gear_train(
    rig: MultiRig,
    train: GearTrain,
    axes: RigAxes,
    states: np.ndarray,
    bodies: list[Body],
    pins: list[Pin],
    meshes: list[tuple[str, str]],
    body_of: dict[int, str],
    gear_thickness: float,
    axle_radius: float,
) -> None:
    n_steps = len(states)
    rotations = [gear_rotations(train, 2.0 * math.pi * k / n_steps) for k in range(n_steps)]
    names: list[list[str]] = []
    for ci, c in enumerate(train.connections):
        row = []
        for gi, g in enumerate(c.gears):
            name = f"engranaje {ci + 1}.{gi + 1}"
            shapes = tuple(gear_outline(g, rotations[k][ci][gi]) for k in range(n_steps))
            bodies.append(Body(name, shapes, gear_thickness))
            row.append(name)
        names.append(row)
        meshes.extend(zip(row, row[1:], strict=False))

    rep = representatives(train.axes)
    on_axis: dict[int, list[str]] = {}
    for ci, c in enumerate(train.connections):
        on_axis.setdefault(c.a, []).append(names[ci][0])
        on_axis.setdefault(c.b, []).append(names[ci][-1])
    for i, gears in on_axis.items():
        cranks = [
            body_of[rig.subs[k].drive.comp]
            for k in range(len(rig.subs))
            if rep[k] == i and rig.subs[k].drive.comp in body_of
        ]
        members = tuple(dict.fromkeys([*gears, *cranks]))
        if len(members) >= 2:
            ax = train.axes[i]
            pins.append(Pin(members, np.array([[ax.x, ax.y]] * n_steps), axle_radius))
