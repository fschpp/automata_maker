"""Puente entre los mecanismos ya unidos al personaje (``MultiRig``) y la red de engranajes (M5).

``gearnet.plan_train`` trabaja con ejes sueltos (posición y sentido de giro). Acá se sacan esos
ejes de un ``MultiRig`` y se arma la red; no se toca ni el ajuste ni la simulación del rig.

* **Eje de un mecanismo**: el origen del componente que mueve su Input Driver (la manivela), que
  está clavado a una bancada fija. Se lee en el estado dado (por omisión el de ``rig.state0``),
  así que sigue la posición real que tiene el mecanismo en el mundo.
* **Sentido de giro**: sale del signo de ``Drive.ratio`` respecto del eje de entrada.
* **Velocidad**: todos los ejes de la red giran a la velocidad del de entrada, así que el módulo
  de ``Drive.ratio`` tiene que ser el mismo en todos; si no, es un error (no se inventa una
  relación de dientes distinta por eje).
* **Ejes coaxiales**: dos mecanismos cuyas manivelas quedan en el mismo punto (por ejemplo las
  dos piernas del bípedo, en planos z distintos) comparten un solo eje y no necesitan engranajes
  entre sí. Si piden sentidos opuestos no hay tren que lo resuelva sin engranajes: queda en
  ``problems``.
* **Desfase**: la red deja a cada eje a un ángulo fijo respecto de la entrada; el desfase de cada
  mecanismo se materializa al montar la manivela en su engranaje (M7), no acá.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from mechchar.core.assembly import FixedState, Pin
from mechchar.core.gearnet import Axis, GearTrain, plan_train
from mechchar.core.rig import MultiRig

SAME_POINT = 1e-6  # dos ejes a menos de esto (en unidades del mundo) son el mismo eje


@dataclass(frozen=True)
class RigAxes:
    """Los ejes distintos de un rig y qué mecanismos (índices en ``rig.subs``) lleva cada uno."""

    axes: tuple[Axis, ...]
    members: tuple[tuple[int, ...], ...]
    problems: tuple[str, ...] = ()

    def axis_of(self, k: int) -> int:
        """Índice del eje que lleva al mecanismo ``k``."""
        for i, group in enumerate(self.members):
            if k in group:
                return i
        raise IndexError(f"no hay un mecanismo {k}")


def _crank_is_grounded(rig: MultiRig, comp: int) -> bool:
    """¿El origen de la manivela está clavado a un componente de estado fijo?"""
    fixed = {c.comp for c in rig.asm.constraints if isinstance(c, FixedState)}
    for c in rig.asm.constraints:
        if not isinstance(c, Pin):
            continue
        if c.a == comp and tuple(c.pa) == (0.0, 0.0) and c.b in fixed:
            return True
        if c.b == comp and tuple(c.pb) == (0.0, 0.0) and c.a in fixed:
            return True
    return False


def rig_axes(
    rig: MultiRig, state: np.ndarray | None = None, root: int = 0, tol: float = SAME_POINT
) -> RigAxes:
    """Ejes de entrada de los mecanismos de ``rig``, con los coaxiales agrupados.

    ``root`` es el mecanismo cuyo eje es la entrada de la red. Lanza ``ValueError`` si el
    mecanismo no existe, si su manivela no está clavada a la bancada o si los módulos de
    ``Drive.ratio`` no coinciden."""
    n = len(rig.subs)
    if not 0 <= root < n:
        raise ValueError(f"el eje de entrada {root} no existe (hay {n} mecanismos)")
    s = np.asarray(rig.state0 if state is None else state, dtype=float)
    speed = abs(rig.subs[root].drive.ratio)
    if speed == 0.0 or not math.isfinite(speed):
        raise ValueError(f"{rig.subs[root].name}: el Input Driver no gira (ratio 0)")
    turning_root = rig.subs[root].drive.ratio > 0

    pos: list[tuple[float, float]] = []
    same: list[bool] = []
    for sub in rig.subs:
        d = sub.drive
        if not math.isclose(abs(d.ratio), speed, rel_tol=1e-9):
            raise ValueError(
                f"{sub.name}: gira a {abs(d.ratio):g} veces el motor y la entrada a {speed:g}; "
                "la red une ejes que giran a la misma velocidad"
            )
        if not _crank_is_grounded(rig, d.comp):
            raise ValueError(f"{sub.name}: el eje de su manivela no está fijo a la bancada")
        pos.append((float(s[3 * d.comp]), float(s[3 * d.comp + 1])))
        same.append((d.ratio > 0) == turning_root)

    # Agrupar por posición; el grupo de la entrada va primero para que sea el índice 0.
    order = [root] + [k for k in range(n) if k != root]
    groups: list[list[int]] = []
    for k in order:
        for g in groups:
            if math.dist(pos[g[0]], pos[k]) <= tol:
                g.append(k)
                break
        else:
            groups.append([k])

    problems: list[str] = []
    axes = []
    for g in groups:
        first = g[0]
        for k in g[1:]:
            if same[k] != same[first]:
                problems.append(
                    f"{rig.subs[first].name} y {rig.subs[k].name} comparten eje pero piden "
                    "sentidos de giro opuestos"
                )
        axes.append(
            Axis("+".join(rig.subs[k].name for k in g), pos[first][0], pos[first][1], same[first])
        )
    return RigAxes(tuple(axes), tuple(tuple(g) for g in groups), tuple(problems))


def plan_rig_train(
    rig: MultiRig,
    root: int = 0,
    state: np.ndarray | None = None,
    **kwargs: object,
) -> tuple[GearTrain, RigAxes]:
    """Red de engranajes que une los ejes de ``rig`` al eje del mecanismo ``root``.

    ``kwargs`` pasa a ``gearnet.plan_train`` (``module``, ``intermediates``, ``layer_pitch``).
    Los problemas de ``rig_axes`` (ejes coaxiales con sentidos opuestos) se suman a los del tren.
    Con un solo eje distinto no hay nada que conectar: el tren queda sin conexiones."""
    ra = rig_axes(rig, state, root)
    train = plan_train(list(ra.axes), 0, **kwargs)  # type: ignore[arg-type]
    if ra.problems:
        train = replace(train, problems=tuple(train.problems) + ra.problems)
    return train, ra


def axis_positions(ra: RigAxes) -> Sequence[tuple[float, float]]:
    """Posición (x, y) de cada eje, en el orden de ``ra.axes``."""
    return [(a.x, a.y) for a in ra.axes]
