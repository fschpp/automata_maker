"""Elegir puntos de actuación y marcador sobre un personaje, y avisar si la elección sirve (M4).
Sin GTK.

Un mecanismo se une al personaje con un pasador (2 ecuaciones) en el *punto de actuación*, y la
curva que se dibuja es la del *marcador*. Para que eso funcione, el punto de actuación tiene que
poder moverse en dos direcciones independientes: si el personaje solo lo deja recorrer una curva
(p. ej. un punto del muslo, que gira alrededor de la cadera fija) el pasador sobrerrestringe el
sistema, y si está en una pieza fija no se mueve nunca.

La «movilidad» de un punto es el rango (0, 1 o 2) de cómo cambia su posición al moverse el
personaje por sus grados de libertad, medido en la pose de reposo: ``P · N`` con ``N`` una base
del núcleo del Jacobiano de las restricciones y ``P`` la derivada del punto respecto del estado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from mechchar.core.assembly import drot
from mechchar.core.character import Character, PointRef

RANK_TOL = 1e-8
NULL_RCOND = 1e-9
MIN_REACH = 0.05  # por debajo, el marcador casi no se mueve en una de las dos direcciones
SUGGEST_FRACTIONS = (0.25, 0.5, 0.75)  # posiciones sobre el eje de cada pieza


# --- elegir con el clic -------------------------------------------------------------------------


def pick_part(character: Character, world: tuple[float, float], tol: float = 0.0) -> str | None:
    """Nombre de la pieza bajo ``world`` (pose de reposo), o la más cercana a menos de ``tol``.
    Si varias contienen el punto gana la de menor área (una barra sobre la pelvis)."""
    state = character.build()[1]
    pt = Point(world)
    best: tuple[float, float, str] | None = None
    for i, part in enumerate(character.parts):
        poly = Polygon(character.part_polygon(state, i))
        d = poly.distance(pt)
        if d > tol:
            continue
        key = (round(d, 12), poly.area, part.name)
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


def pick_point(
    character: Character, world: tuple[float, float], tol: float = 0.0
) -> PointRef | None:
    """Punto elegido con un clic: el mismo punto si cae dentro de una pieza, o el punto más
    cercano de su contorno si cae fuera pero a menos de ``tol``. ``None`` si no hay pieza."""
    name = pick_part(character, world, tol)
    if name is None:
        return None
    state = character.build()[1]
    poly = Polygon(character.part_polygon(state, character.index(name)))
    pt = Point(world)
    if poly.distance(pt) > 0.0:
        near = nearest_points(poly, pt)[0]
        world = (float(near.x), float(near.y))
    return PointRef(name, (float(world[0]), float(world[1])))


# --- movilidad ----------------------------------------------------------------------------------


def _mobility_matrix(character: Character, ref: PointRef, basis: np.ndarray) -> np.ndarray:
    """Derivada (2 × d) de la posición del punto respecto de los ``d`` grados de libertad."""
    state = character.build()[1]
    i = character.index(ref.part)
    local = np.asarray(character.local_point(ref.part, ref.point), dtype=float)
    theta = float(state[3 * i + 2])
    point_jac = np.zeros((2, state.size))
    point_jac[0, 3 * i] = 1.0
    point_jac[1, 3 * i + 1] = 1.0
    point_jac[:, 3 * i + 2] = drot(theta) @ local
    return point_jac @ basis


def _dof_basis(character: Character) -> np.ndarray:
    """Base ortonormal (n × d) de los movimientos del personaje en reposo."""
    asm, s0 = character.build()
    jac = asm.jacobian(s0, 0.0)
    if jac.shape[0] == 0:
        return np.eye(s0.size)
    return null_space(jac, rcond=NULL_RCOND)


def point_mobility(character: Character, ref: PointRef) -> int:
    """Cuántas direcciones independientes puede recorrer el punto (0, 1 o 2)."""
    basis = _dof_basis(character)
    if basis.shape[1] == 0:
        return 0
    m = _mobility_matrix(character, ref, basis)
    return int(np.linalg.matrix_rank(m, tol=RANK_TOL))


def reach(character: Character, attach: PointRef, marker: PointRef) -> float:
    """Qué tan bien el movimiento del punto de actuación se traduce en el del marcador, de 0 a 1.

    Es la razón entre el menor y el mayor valor singular de la aplicación que lleva un
    desplazamiento del punto de actuación a uno del marcador: 1 si el marcador se mueve igual de
    fácil en cualquier dirección, cerca de 0 si prácticamente solo se mueve a lo largo de una
    línea (y entonces no podría seguir bien una elipse). 0 si no está definida."""
    basis = _dof_basis(character)
    if basis.shape[1] == 0:
        return 0.0
    a = _mobility_matrix(character, attach, basis)
    m = _mobility_matrix(character, marker, basis)
    if np.linalg.matrix_rank(a, tol=RANK_TOL) < 2:
        return 0.0
    sv = np.linalg.svd(m @ np.linalg.pinv(a), compute_uv=False)
    return float(sv[-1] / sv[0]) if sv[0] > RANK_TOL else 0.0


# --- diagnóstico --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ActuationAdvice:
    ok: bool  # se puede ajustar con esta elección
    dof: int  # grados de libertad del personaje solo
    attach_rank: int
    marker_rank: int
    reach: float
    messages: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.messages)


def advise(character: Character, attach: PointRef, marker: PointRef) -> ActuationAdvice:
    """Chequea una elección de punto de actuación y marcador y explica lo que no sirve."""
    dof = character.analyze().dof
    a_rank = point_mobility(character, attach)
    m_rank = point_mobility(character, marker)
    msgs: list[str] = []
    ok = True
    if dof == 0:
        ok = False
        msgs.append("El personaje no tiene grados de libertad: no hay nada que accionar.")
    if character.parts[character.index(attach.part)].fixed:
        ok = False
        msgs.append(f"«{attach.part}» es una pieza fija: el mecanismo no puede moverla.")
    elif dof and a_rank == 0:
        ok = False
        msgs.append(f"El punto de actuación en «{attach.part}» no se mueve: queda trabado.")
    elif dof and a_rank == 1:
        ok = False
        msgs.append(
            f"El punto de actuación en «{attach.part}» solo puede recorrer una curva (1 grado de "
            "libertad): un pasador lo sobrerrestringe. Elegí un punto que se mueva en el plano."
        )
    if dof and m_rank == 0:
        ok = False
        msgs.append(f"El marcador en «{marker.part}» no se mueve: su curva sería un punto.")
    r = reach(character, attach, marker) if ok else 0.0
    if ok and r < MIN_REACH:
        msgs.append(
            "Con este punto de actuación el marcador casi solo se mueve en una dirección (en "
            "reposo): puede costar que siga una curva ancha como una elipse."
        )
    if ok and dof > 2:
        msgs.append(
            f"Con un solo mecanismo quedan {dof - 2} grados de libertad sin controlar: "
            "hace falta otro mecanismo (o más puntos de actuación)."
        )
    if ok and not msgs:
        msgs.append("El punto de actuación y el marcador se pueden usar.")
    return ActuationAdvice(ok, dof, a_rank, m_rank, r, tuple(msgs))


# --- sugerencias --------------------------------------------------------------------------------


def suggest_attach_points(
    character: Character, marker: PointRef, limit: int = 3
) -> list[tuple[PointRef, float]]:
    """Puntos de actuación posibles, de mejor a peor: sobre el eje de cada pieza móvil (a 1/4,
    1/2 y 3/4 de su largo) con movilidad 2, ordenados por ``reach`` y como mucho uno por pieza.

    Es una heurística de primer orden en la pose de reposo: en la pierna de prueba ordena los
    errores de ajuste en el mismo sentido que el alcance pero con diferencias chicas (7,4–8,1 %
    para una elipse), así que sirve para descartar lo peor y no para predecir el error."""
    state = character.build()[1]
    basis = _dof_basis(character)
    if basis.shape[1] == 0:
        return []
    out: list[tuple[PointRef, float]] = []
    for i, part in enumerate(character.parts):
        if part.fixed or not part.outline:
            continue
        xs = [v[0] for v in part.outline]
        x0, x1 = min(xs), max(xs)
        best: tuple[PointRef, float] | None = None
        for f in SUGGEST_FRACTIONS:
            lx = x0 + f * (x1 - x0)
            c, s = np.cos(state[3 * i + 2]), np.sin(state[3 * i + 2])
            # punto sobre el eje local x de la pieza
            wx, wy = state[3 * i] + c * lx, state[3 * i + 1] + s * lx
            ref = PointRef(part.name, (float(wx), float(wy)))
            if np.linalg.matrix_rank(_mobility_matrix(character, ref, basis), tol=RANK_TOL) < 2:
                continue
            score = reach(character, ref, marker)
            if best is None or score > best[1]:
                best = (ref, score)
        if best is not None:
            out.append(best)
    out.sort(key=lambda item: -item[1])
    return out[:limit]
