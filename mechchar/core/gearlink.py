"""Puente entre los mecanismos ya unidos al personaje (``MultiRig``) y la red de engranajes (M5).

``gearnet.plan_train`` trabaja con ejes sueltos (posición y sentido de giro). Acá se sacan esos
ejes de un ``MultiRig`` y se arma la red; no se toca ni el ajuste ni la simulación del rig.

* **Eje de un mecanismo**: el punto de la bancada donde está pinada la pieza que lleva su Input
  Driver (``gearnet.input_axis``), leído en el estado dado (por omisión el de ``rig.state0``).
  Hay un eje por mecanismo, en el orden de ``rig.subs``; ``root`` es el índice del mecanismo cuyo
  eje es la entrada de la red.
* **Sentido de giro**: sale del signo de ``Drive.ratio`` respecto del eje de entrada.
* **Velocidad**: todos los ejes de la red giran a la velocidad del de entrada, así que el módulo
  de ``Drive.ratio`` tiene que ser el mismo en todos; si no, es un error (no se inventa una
  relación de dientes distinta por eje).
* **Ejes coaxiales**: dos mecanismos cuyas manivelas quedan en el mismo punto (por ejemplo las
  dos piernas del bípedo, en planos z distintos) comparten un solo eje y no llevan engranajes
  entre sí: lo resuelve ``plan_train`` (``GearTrain.shared``), también cuando piden sentidos
  opuestos, que queda en ``problems``. ``RigAxes.members`` agrupa los mecanismos por eje y las
  conexiones del tren (``Connection.a``/``b``) usan el primer mecanismo de cada grupo.
* **Desfase**: la red deja a cada eje a un ángulo fijo respecto de la entrada; el desfase de cada
  mecanismo se materializa al montar la manivela en su engranaje (M7), no acá.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mechchar.core.gearnet import Axis, GearTrain, input_axis, plan_train, representatives
from mechchar.core.rig import MultiRig


@dataclass(frozen=True)
class RigAxes:
    """Un eje por mecanismo de ``rig.subs`` (mismo orden) y cuál es el de entrada."""

    axes: tuple[Axis, ...]
    root: int = 0

    @property
    def members(self) -> tuple[tuple[int, ...], ...]:
        """Mecanismos agrupados por eje: los que coinciden en el mismo punto comparten uno. El
        grupo del eje de entrada va primero."""
        rep = representatives(self.axes)
        groups = [tuple(i for i in range(len(rep)) if rep[i] == r) for r in sorted(set(rep))]
        groups.sort(key=lambda g: self.root not in g)
        return tuple(groups)

    def axis_of(self, k: int) -> int:
        """Índice del eje que usan las conexiones del tren para el mecanismo ``k``: el primer
        mecanismo de su grupo."""
        if not 0 <= k < len(self.axes):
            raise IndexError(f"no hay un mecanismo {k}")
        return representatives(self.axes)[k]


def rig_axes(rig: MultiRig, state: np.ndarray | None = None, root: int = 0) -> RigAxes:
    """Ejes de entrada de los mecanismos de ``rig``.

    ``root`` es el mecanismo cuyo eje es la entrada de la red. Lanza ``ValueError`` si el
    mecanismo no existe, si el eje de su manivela no está fijo a la bancada o si los módulos de
    ``Drive.ratio`` no coinciden."""
    n = len(rig.subs)
    if not 0 <= root < n:
        raise ValueError(f"el eje de entrada {root} no existe (hay {n} mecanismos)")
    s = np.asarray(rig.state0 if state is None else state, dtype=float)
    speed = abs(rig.subs[root].drive.ratio)
    if speed == 0.0 or not math.isfinite(speed):
        raise ValueError(f"{rig.subs[root].name}: el Input Driver no gira (ratio 0)")
    turning_root = rig.subs[root].drive.ratio > 0

    axes = []
    for sub in rig.subs:
        d = sub.drive
        if not math.isclose(abs(d.ratio), speed, rel_tol=1e-9):
            raise ValueError(
                f"{sub.name}: gira a {abs(d.ratio):g} veces el motor y la entrada a {speed:g}; "
                "la red une ejes que giran a la misma velocidad"
            )
        xy = input_axis(rig.asm, s, d)
        if xy is None:
            raise ValueError(f"{sub.name}: el eje de su manivela no está fijo a la bancada")
        axes.append(Axis(sub.name, xy[0], xy[1], (d.ratio > 0) == turning_root))
    return RigAxes(tuple(axes), root)


def plan_rig_train(
    rig: MultiRig,
    root: int = 0,
    state: np.ndarray | None = None,
    **kwargs: object,
) -> tuple[GearTrain, RigAxes]:
    """Red de engranajes que une los ejes de ``rig`` al eje del mecanismo ``root``.

    ``kwargs`` pasa a ``gearnet.plan_train`` (``module``, ``intermediates``, ``layer_pitch``);
    ``intermediates`` se indexa por mecanismo (el primero de cada grupo de ejes coincidentes).
    Con un solo eje distinto no hay nada que conectar: el tren queda sin conexiones."""
    ra = rig_axes(rig, state, root)
    train = plan_train(list(ra.axes), root, **kwargs)  # type: ignore[arg-type]
    return train, ra


def axis_positions(ra: RigAxes) -> Sequence[tuple[float, float]]:
    """Posición (x, y) de cada eje, en el orden de ``ra.axes``."""
    return [(a.x, a.y) for a in ra.axes]
