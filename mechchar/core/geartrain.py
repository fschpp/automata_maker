"""Tren de engranajes entre ejes de entrada (M5, núcleo). Sin GTK.

Cada mecanismo gira sobre un eje de entrada. Para accionarlos todos con un solo motor se unen los
ejes con engranajes: un engranaje en cada eje y, entre ellos, una cadena de engranajes
intermedios. Este módulo resuelve **una conexión** entre dos ejes:

* ``Gear``: un engranaje con **dientes enteros y módulo común**, r = m·z/2, así que dos engranajes
  engranan cuando la distancia entre centros es m·(z₁ + z₂)/2 (el paso extra respecto del paper,
  que optimiza radios continuos y no son fabricables).
* **Paridad**: cada engranaje invierte el sentido de giro, así que con ``k`` intermedios los dos
  extremos giran en el mismo sentido si ``k`` es impar y en sentido contrario si es par (el
  «Orientation (odd: same, even: reverse)» del video).
* **Velocidad**: los intermedios no cambian la relación entre los extremos, que es solo
  z_primero / z_último. Para que dos mecanismos giren a la misma velocidad los engranajes de los
  extremos tienen que tener los mismos dientes; los intermedios quedan libres.
* ``connect`` coloca los intermedios entre dos extremos que ya están en su lugar (los ejes no se
  mueven: los fija la posición de cada mecanismo).
* ``overlaps`` detecta engranajes de la misma capa que se pisan sin engranar, y
  ``simulate_angles`` verifica las velocidades simulando el tren con la restricción ``Phase``.

No se modela la alineación de los dientes (que no choquen punta con punta al montar): es un
detalle de fabricación (M7).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from mechchar.core.assembly import Assembly, Drive, FixedState, Phase, Pin
from mechchar.core.mechanisms import circle_intersections

MIN_TEETH = 10  # menos que esto un engranaje real se socava; valor inicial, a validar al imprimir
MESH_TOL = 1e-6  # tolerancia relativa para dar dos engranajes por engranados
OVERLAP_TOL = 1e-6  # cuánto pueden pisarse (en unidades del mundo) antes de contar como choque
LAYER_TOL = 1e-9  # dos engranajes están en la misma capa si sus ``z`` difieren menos que esto


@dataclass(frozen=True)
class Gear:
    """Engranaje de dientes enteros en el plano (x, y), en la capa ``z``."""

    x: float
    y: float
    teeth: int
    module: float
    z: float = 0.0

    def __post_init__(self) -> None:
        if self.teeth < MIN_TEETH:
            raise ValueError(f"un engranaje necesita al menos {MIN_TEETH} dientes")
        if not (math.isfinite(self.module) and self.module > 0.0):
            raise ValueError("el módulo debe ser positivo")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z)):
            raise ValueError("la posición del engranaje debe ser finita")

    @property
    def radius(self) -> float:
        return self.module * self.teeth / 2.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])


def distance(a: Gear, b: Gear) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def mesh_distance(a: Gear, b: Gear) -> float:
    """Distancia entre centros a la que engranan: m·(z₁ + z₂)/2 (el mismo módulo)."""
    _same_module(a, b)
    return a.radius + b.radius


def meshes(a: Gear, b: Gear) -> bool:
    """``a`` y ``b`` engranan: misma capa, mismo módulo y a la distancia justa."""
    return (
        abs(a.z - b.z) <= LAYER_TOL
        and a.module == b.module
        and abs(distance(a, b) - mesh_distance(a, b)) <= MESH_TOL * mesh_distance(a, b)
    )


def _same_module(*gears: Gear) -> None:
    if any(g.module != gears[0].module for g in gears):
        raise ValueError("todos los engranajes de un tren tienen que tener el mismo módulo")


# --- paridad y velocidad -----------------------------------------------------------------------


def direction(n_intermediate: int) -> int:
    """+1 si los extremos giran en el mismo sentido (``n_intermediate`` impar), −1 si no."""
    if n_intermediate < 0:
        raise ValueError("no puede haber una cantidad negativa de intermedios")
    return 1 if n_intermediate % 2 == 1 else -1


def intermediates_for(same_direction: bool, at_least: int = 0) -> int:
    """Menor cantidad de intermedios, no menor que ``at_least``, con la paridad que pide el
    sentido de giro (impar = mismo sentido, par = contrario)."""
    k = max(at_least, 0)
    while (k % 2 == 1) != same_direction:
        k += 1
    return k


def speed_ratio(chain: Sequence[Gear]) -> float:
    """ω_último / ω_primero de una cadena engranada: ±z_primero / z_último (los intermedios no
    cuentan, salvo para el signo)."""
    if len(chain) < 2:
        raise ValueError("una cadena tiene al menos dos engranajes")
    return direction(len(chain) - 2) * chain[0].teeth / chain[-1].teeth


def end_teeth_for_distance(dist: float, module: float, tol: float = 1e-6) -> int | None:
    """Dientes de dos engranajes iguales que engranan directamente a la distancia ``dist``
    (dist = m·z), o ``None`` si no es un múltiplo entero del módulo o da menos de ``MIN_TEETH``."""
    z = dist / module
    n = round(z)
    return n if n >= MIN_TEETH and abs(z - n) <= tol else None


# --- colocación de los intermedios -------------------------------------------------------------


def _place(
    a: Gear, b: Gear, inner: Sequence[Gear], angle: float, side: int
) -> list[list[Gear]]:
    """Cadenas posibles con los primeros ``k − 1`` intermedios en fila saliendo de ``a`` con
    ``angle`` respecto de la recta a → b, y el último cerrando con ``b`` (hasta dos puntos)."""
    m = a.module
    base = math.atan2(b.y - a.y, b.x - a.x) + angle
    unit = np.array([math.cos(base), math.sin(base)])
    placed: list[Gear] = []
    prev = a
    for g in inner[:-1]:
        c = prev.center + unit * (prev.radius + g.radius)
        placed.append(Gear(float(c[0]), float(c[1]), g.teeth, m, a.z))
        prev = placed[-1]
    last = inner[-1]
    try:
        points = circle_intersections(
            (prev.x, prev.y), prev.radius + last.radius, (b.x, b.y), last.radius + b.radius
        )  # el primero queda a la izquierda de prev → b
    except ValueError:
        return []
    if side < 0:
        points = points[::-1]
    return [
        [*placed, Gear(float(p[0]), float(p[1]), last.teeth, m, a.z)] for p in points
    ]


def _free(chain_: Sequence[Gear], avoid: Sequence[Gear]) -> bool:
    """La cadena no se pisa a sí misma (salvo los vecinos, que engranan) ni con ``avoid``. No
    revisa el engrane: ``_place`` lo deja exacto por construcción."""
    n = len(chain_)
    for i in range(n):
        g = chain_[i]
        for j in range(i + 2, n):
            h = chain_[j]
            if abs(g.z - h.z) <= LAYER_TOL and distance(g, h) < g.radius + h.radius - OVERLAP_TOL:
                return False
        for h in avoid:
            if abs(g.z - h.z) <= LAYER_TOL and distance(g, h) < g.radius + h.radius - OVERLAP_TOL:
                return False
    return True


def connect(
    a: Gear,
    b: Gear,
    inner_teeth: Sequence[int],
    side: int = 1,
    avoid: Sequence[Gear] = (),
) -> list[Gear] | None:
    """Intermedios que unen los extremos ``a`` y ``b`` (que ya están en su lugar), o ``None`` si no
    hay forma de engranarlos con esos dientes sin que se pisen.

    ``inner_teeth`` da los dientes de cada intermedio, de ``a`` a ``b``; vacío = engrane directo
    (solo posible si la distancia es justo la de engrane). Con uno solo, la posición queda
    determinada salvo por el lado. Con más, los primeros ``k − 1`` salen en fila desde ``a`` con un
    ángulo libre y el último cierra con ``b`` cortando dos circunferencias; se prueba cada grado
    desde el más recto (0°) hasta el más torcido y se devuelve la primera cadena válida, es decir
    la más recta que engrana sin pisarse. ``side`` = +1 / −1 hace que la cadena se doble hacia la
    izquierda / derecha de la recta de ``a`` a ``b``. ``avoid`` son otros engranajes (de la misma
    capa) que la cadena tampoco puede pisar: el ángulo se sigue buscando hasta dar con uno libre,
    en lugar de devolver una cadena válida por sí sola que después haya que descartar.

    Límite: la familia «fila recta + cierre» no encuentra cadenas en zigzag, así que con varios
    intermedios chicos entre ejes cercanos (p. ej. cuatro de 10 dientes a 20 de distancia, con
    extremos de 20) devuelve ``None`` aunque exista alguna cadena torcida."""
    _same_module(a, b)
    if abs(a.z - b.z) > LAYER_TOL:
        raise ValueError("los extremos tienen que estar en la misma capa")
    if distance(a, b) == 0.0:
        raise ValueError("los dos ejes coinciden")
    inner = [Gear(0.0, 0.0, t, a.module, a.z) for t in inner_teeth]  # valida los dientes
    if not inner:
        return [] if meshes(a, b) and _free([a, b], avoid) else None
    reach = a.radius + b.radius + 2.0 * sum(g.radius for g in inner)
    if distance(a, b) > reach + OVERLAP_TOL:
        return None  # ni en línea recta llegan: no hay que buscar ángulos
    sign = 1 if side >= 0 else -1
    angles = [0.0] if len(inner) == 1 else [
        sign * math.radians(d) for d in range(0, 181)
    ]
    for angle in angles:
        for placed in _place(a, b, inner, angle, sign):
            if _free([a, *placed, b], avoid):
                return placed
    return None


def chain(a: Gear, b: Gear, inner_teeth: Sequence[int], side: int = 1) -> list[Gear] | None:
    """La cadena completa [a, intermedios…, b], o ``None`` si ``connect`` no cierra."""
    inner = connect(a, b, inner_teeth, side)
    return None if inner is None else [a, *inner, b]


# --- validación --------------------------------------------------------------------------------


def chain_problems(gears: Sequence[Gear]) -> list[str]:
    """Qué está mal en una cadena: pares consecutivos que no engranan y engranajes que se pisan
    sin ser vecinos. Vacío si la cadena es válida."""
    problems = [
        f"los engranajes {i} y {i + 1} no engranan"
        for i, (g, h) in enumerate(zip(gears, gears[1:], strict=False))
        if not meshes(g, h)
    ]
    neighbours = {frozenset((i, i + 1)) for i in range(len(gears) - 1)}
    problems += [
        f"los engranajes {i} y {j} se pisan"
        for i, j in overlaps(gears, ignore=neighbours)
    ]
    return problems


def overlaps(
    gears: Sequence[Gear], ignore: Collection[frozenset[int]] = ()
) -> list[tuple[int, int]]:
    """Pares (i, j) de engranajes de la misma capa cuyos círculos primitivos se pisan más de
    ``OVERLAP_TOL``. Los pares de ``ignore`` (los que engranan a propósito) no cuentan."""
    out = []
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            g, h = gears[i], gears[j]
            if frozenset((i, j)) in ignore or abs(g.z - h.z) > LAYER_TOL:
                continue
            if distance(g, h) < g.radius + h.radius - OVERLAP_TOL:
                out.append((i, j))
    return out


# --- velocidades por simulación ----------------------------------------------------------------


def simulate_angles(gears: Sequence[Gear], n_steps: int = 24) -> np.ndarray:
    """Ángulo de cada engranaje (n_steps, n) a lo largo de una vuelta del primero, con el tren
    armado como un ensamblaje: cada engranaje fijo en su eje, el primero con un Input Driver y
    cada par engranado con ``Phase`` (θ_b = −z_a/z_b·θ_a). Lanza ``ValueError`` si algún par no
    engrana, porque entonces la simulación no representa un tren."""
    if len(gears) < 2:
        raise ValueError("un tren necesita al menos dos engranajes")
    for g, h in zip(gears, gears[1:], strict=False):
        if not meshes(g, h):
            raise ValueError("hay engranajes consecutivos que no engranan")
    asm = Assembly()
    ground = asm.add_component("bancada")
    asm.add(FixedState(ground, 0.0, 0.0, 0.0))
    comps = []
    for i, g in enumerate(gears):
        c = asm.add_component(f"engranaje{i}", g.x, g.y, 0.0, g.z)
        asm.add(Pin(ground, (g.x, g.y), c, (0.0, 0.0)))
        comps.append(c)
    asm.add(Drive(comps[0]))
    for k in range(len(gears) - 1):
        asm.add(Phase(comps[k], comps[k + 1], -gears[k].teeth / gears[k + 1].teeth))
    # Un tren no tiene ramas de ensamblado (el salto entre fases es solo el paso de giro) y no
    # tiene por qué cerrar tras una vuelta del primero: un intermedio de z dientes gira
    # z_primero / z vueltas, que no es entero. Por eso no se exige el cierre.
    result = asm.trace_cycle(
        asm.initial_state(), n_steps, max_jump=math.pi, closure_tol=float("inf")
    )
    if not result.ok:
        raise ValueError(f"el tren no se pudo simular: {result.reason}")
    return np.array([[s[3 * c + 2] for c in comps] for s in result.states])
