"""Forma de los engranajes de la red y cómo giran (M5, bundle 4). Sin GTK.

* ``gear_outline``: contorno con dientes de flancos rectos (trapecios), pensado para dibujar y
  para chequear choques; **no es un perfil evolvente** (eso llega con la exportación, M7). La
  circunferencia primitiva es la del ``Gear`` (r = m·z/2), la cabeza sobresale ``ADDENDUM``·m y
  el pie se hunde ``DEDENDUM``·m; el diente mide ``TOOTH_FILL`` de medio paso sobre la primitiva
  (el resto es juego).
* ``mesh_phase``: ángulo con que hay que girar un engranaje para que sus dientes entren en los
  huecos de su vecino. Sin esto, dos engranajes que engranan en distancia se dibujarían con los
  dientes pisados.
* ``gear_rotations``: ángulo de cada engranaje de la red cuando el eje de entrada gira ``theta``:
  cada engranaje gira ``(-1)^k · z₀/z_k`` veces el del primero de su cadena, y los de un eje giran
  como ese eje (``Axis.same_direction``). La fase de montaje de cada cadena sale de ``mesh_phase``
  y el primer engranaje de cada conexión parte en 0 (los de un mismo eje son piezas distintas y
  su orientación relativa es libre).
"""

from __future__ import annotations

import math

import numpy as np

from mechchar.core.gearnet import Connection, GearTrain
from mechchar.core.geartrain import Gear

ADDENDUM = 1.0  # cabeza del diente, en módulos
DEDENDUM = 1.25  # pie del diente, en módulos
TOOTH_FILL = 0.9  # fracción de medio paso que ocupa el diente sobre la primitiva
TIP_TO_PITCH = 0.55  # ancho en la cabeza respecto del ancho sobre la primitiva
TWO_PI = 2.0 * math.pi


def root_radius(g: Gear) -> float:
    return g.radius - DEDENDUM * g.module


def tip_radius(g: Gear) -> float:
    return g.radius + ADDENDUM * g.module


def gear_outline(g: Gear, rotation: float = 0.0) -> np.ndarray:
    """Contorno del engranaje (N, 2), en sentido antihorario, con un diente centrado en
    ``rotation`` (y uno cada 2π/z). Cada diente son cuatro vértices: pie izquierdo, cabeza
    izquierda, cabeza derecha y pie derecho."""
    z = g.teeth
    w_pitch = TOOTH_FILL * math.pi / (2.0 * z)  # semiancho angular del diente sobre la primitiva
    w_tip = TIP_TO_PITCH * w_pitch
    r_root, r_tip = root_radius(g), tip_radius(g)
    # el semiancho angular varía linealmente con el radio y vale w_pitch en la primitiva
    slope = (w_tip - w_pitch) / (r_tip - g.radius)
    w_root = w_pitch + slope * (r_root - g.radius)
    if not (r_root > 0.0 and 0.0 < w_root < math.pi / z):
        raise ValueError(f"engranaje de {z} dientes sin forma válida")
    pts = []
    for i in range(z):
        c = rotation + i * TWO_PI / z
        pts += [
            (r_root, c - w_root),
            (r_tip, c - w_tip),
            (r_tip, c + w_tip),
            (r_root, c + w_root),
        ]
    a = np.array(pts)
    return np.column_stack([g.x + a[:, 0] * np.cos(a[:, 1]), g.y + a[:, 0] * np.sin(a[:, 1])])


def _wrap(angle: float, period: float) -> float:
    """``angle`` llevado a [-period/2, period/2)."""
    return (angle + period / 2.0) % period - period / 2.0


def mesh_phase(prev: Gear, prev_rotation: float, g: Gear) -> float:
    """Rotación de ``g`` (con un diente en ese ángulo) para engranar con ``prev``, que tiene un
    diente en ``prev_rotation``: donde ``prev`` mete un diente, ``g`` tiene un hueco."""
    psi = math.atan2(g.y - prev.y, g.x - prev.x)  # dirección de prev hacia g
    e = _wrap(psi - prev_rotation, TWO_PI / prev.teeth)  # cuánto se pasa el contacto del diente
    return psi + math.pi + e * prev.radius / g.radius + math.pi / g.teeth


def chain_phases(gears: tuple[Gear, ...]) -> list[float]:
    """Rotaciones de montaje de una cadena (θ = 0): la primera es 0 y cada una engrana con la
    anterior."""
    phases = [0.0]
    for prev, g in zip(gears, gears[1:], strict=False):
        phases.append(mesh_phase(prev, phases[-1], g))
    return phases


def connection_rotations(train: GearTrain, c: Connection, theta: float) -> list[float]:
    """Rotación de cada engranaje de la conexión ``c`` con el eje de entrada girado ``theta``."""
    s_a = 1.0 if train.axes[c.a].same_direction else -1.0
    z0 = c.gears[0].teeth
    out = []
    for k, (g, phase) in enumerate(zip(c.gears, chain_phases(c.gears), strict=True)):
        ratio = (-1.0) ** k * z0 / g.teeth
        out.append(phase + ratio * s_a * theta)
    return out


def gear_rotations(train: GearTrain, theta: float = 0.0) -> list[list[float]]:
    """Rotación de todos los engranajes de la red: una lista por conexión, en el orden de
    ``train.connections`` y de ``Connection.gears``."""
    return [connection_rotations(train, c, theta) for c in train.connections]
