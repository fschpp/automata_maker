"""Personaje articulado 2D (M4): piezas rígidas, articulaciones y piezas fijas. Sin GTK.

Todo se define en la *pose de reposo* y en coordenadas del mundo: cada pieza tiene su pose
(x, y, θ) y un contorno en coordenadas locales; cada articulación (pivote) es un punto del mundo
compartido por dos piezas. Los puntos locales de los pasadores se calculan desde ahí, así que el
reposo siempre cumple las restricciones. Las piezas fijas forman el chasis. ``z`` es el plano de
profundidad de la pieza (la simulación es 2D por plano; cada sub-ensamblaje vive en el plano de
la pieza que acciona).

Formato en disco: JSON (``Character.save`` / ``Character.load``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mechchar.core.assembly import Analysis, Assembly, FixedState, Pin, rot, world_point

Vec = tuple[float, float]
FORMAT_VERSION = 1


@dataclass
class Part:
    name: str
    x: float
    y: float
    theta: float
    outline: list[Vec]  # polígono en coordenadas locales (dibujo y exportación)
    fixed: bool = False  # parte del chasis
    z: float = 0.0  # plano de profundidad


@dataclass
class Pivot:
    a: str
    b: str
    point: Vec  # articulación, en el mundo y en la pose de reposo


@dataclass
class PointRef:
    """Un punto sobre una pieza (marcador o punto de actuación), en el mundo en reposo."""

    part: str
    point: Vec


@dataclass
class Character:
    name: str
    parts: list[Part] = field(default_factory=list)
    pivots: list[Pivot] = field(default_factory=list)

    # --- consulta ---------------------------------------------------------------------------
    def index(self, name: str) -> int:
        for i, p in enumerate(self.parts):
            if p.name == name:
                return i
        raise KeyError(f"pieza desconocida: {name!r}")

    def local_point(self, part: str, world: Vec) -> Vec:
        """Punto del mundo (en reposo) expresado en coordenadas locales de ``part``."""
        p = self.parts[self.index(part)]
        v = rot(-p.theta) @ (np.asarray(world, dtype=float) - np.array([p.x, p.y]))
        return float(v[0]), float(v[1])

    def validate(self) -> None:
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names):
            raise ValueError("hay piezas con el mismo nombre")
        for pv in self.pivots:
            if pv.a not in names or pv.b not in names:
                raise ValueError(f"el pivote {pv.a}-{pv.b} usa una pieza inexistente")
            if pv.a == pv.b:
                raise ValueError("un pivote no puede unir una pieza consigo misma")

    # --- simulación -------------------------------------------------------------------------
    def build(self) -> tuple[Assembly, np.ndarray]:
        """Ensamblaje del personaje y su estado de reposo (una pieza por componente, en orden)."""
        self.validate()
        asm = Assembly()
        for p in self.parts:
            asm.add_component(p.name, p.x, p.y, p.theta, p.z)
        for i, p in enumerate(self.parts):
            if p.fixed:
                asm.add(FixedState(i, p.x, p.y, p.theta))
        for pv in self.pivots:
            asm.add(
                Pin(
                    self.index(pv.a), self.local_point(pv.a, pv.point),
                    self.index(pv.b), self.local_point(pv.b, pv.point),
                )
            )
        return asm, asm.initial_state()

    def analyze(self) -> Analysis:
        """Grados de libertad que le quedan al personaje solo (con su chasis fijo)."""
        asm, s0 = self.build()
        return asm.analyze(s0)

    def part_polygon(self, state: np.ndarray, i: int) -> np.ndarray:
        """Contorno de la pieza ``i`` en el mundo para un estado dado."""
        return np.array([world_point(state, i, v) for v in self.parts[i].outline])

    # --- JSON -------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "name": self.name,
            "parts": [
                {
                    "name": p.name,
                    "pose": [p.x, p.y, p.theta],
                    "outline": [list(v) for v in p.outline],
                    "fixed": p.fixed,
                    "z": p.z,
                }
                for p in self.parts
            ],
            "pivots": [{"a": v.a, "b": v.b, "point": list(v.point)} for v in self.pivots],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Character:
        if int(d.get("version", 0)) != FORMAT_VERSION:
            raise ValueError("versión de personaje incompatible")
        parts = []
        for p in d["parts"]:
            x, y, theta = (float(v) for v in p["pose"])
            outline = [(float(a), float(b)) for a, b in p["outline"]]
            parts.append(
                Part(str(p["name"]), x, y, theta, outline, bool(p.get("fixed", False)),
                     float(p.get("z", 0.0)))
            )
        pivots = [
            Pivot(str(v["a"]), str(v["b"]), (float(v["point"][0]), float(v["point"][1])))
            for v in d["pivots"]
        ]
        ch = cls(str(d["name"]), parts, pivots)
        ch.validate()
        return ch

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Character:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- unión con un mecanismo ---------------------------------------------------------------------


@dataclass
class RiggedMechanism:
    """Un mecanismo unido a un personaje, simulados juntos. El estado es el del personaje
    (``n_char`` piezas) seguido del de los componentes del mecanismo."""

    asm: Assembly
    state0: np.ndarray
    character: Character
    n_char: int
    base: Any  # el mecanismo (FourBar / GearedFiveBar), indexado desde 0 sobre su propio tramo
    marker_part: int
    marker_local: Vec

    def mech_state(self, state: np.ndarray) -> np.ndarray:
        return state[3 * self.n_char :]

    def marker_world(self, state: np.ndarray) -> np.ndarray:
        return world_point(state, self.marker_part, self.marker_local)

    def part_polygons(self, state: np.ndarray) -> list[np.ndarray]:
        return [self.character.part_polygon(state, i) for i in range(self.n_char)]

    def outline_points(self, state: np.ndarray) -> np.ndarray:
        pts = [poly for poly in self.part_polygons(state)]
        pts.append(np.asarray(self.base.outline_points(self.mech_state(state))))
        return np.vstack(pts)


def combine(
    character: Character,
    built: Any,
    attach: PointRef,
    marker: PointRef,
    s_char: np.ndarray | None = None,
) -> Any:
    """Une un mecanismo armado (``Built``) al personaje con un pasador entre su marcador y el
    punto de actuación ``attach``. Devuelve un ``Built`` cuyo marcador es el de ``marker`` (el
    punto del personaje cuya curva se quiere controlar). ``s_char`` es la pose inicial del
    personaje (por defecto el reposo)."""
    from mechchar.optim.fitcore import Built  # import tardío: ``optim`` depende de ``core``

    asm, s_rest = character.build()
    s_char = s_rest if s_char is None else np.asarray(s_char, dtype=float)
    offset = asm.append(built.asm)
    asm.add(
        Pin(
            built.coupler + offset, built.tracer,
            character.index(attach.part), character.local_point(attach.part, attach.point),
        )
    )
    state0 = np.concatenate([s_char, built.state0])
    return Built(
        asm, state0, character.index(marker.part), character.local_point(marker.part, marker.point)
    )


# --- personajes de prueba -----------------------------------------------------------------------


def _bar(length: float, half_width: float = 0.12) -> list[Vec]:
    """Contorno de una barra que va de (0, 0) a (length, 0) en coordenadas locales."""
    w = half_width
    return [(-w, -w), (length + w, -w), (length + w, w), (-w, w)]


def _link(name: str, a: Vec, b: Vec, **kw: Any) -> Part:
    """Pieza que va del punto ``a`` al punto ``b`` (su eje local x apunta de a hacia b)."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return Part(name, a[0], a[1], math.atan2(b[1] - a[1], b[0] - a[0]), _bar(length), **kw)


_LEG_HIP, _LEG_KNEE = (0.0, 0.0), (0.9, -1.2)
_LEG_ANKLE, _LEG_TOE = (0.0, -2.4), (0.9, -2.5)


def _leg_chain(prefix: str = "", z: float = 0.0) -> tuple[list[Part], list[Pivot]]:
    """Piezas y pivotes de una pierna (sin pelvis): muslo, pantorrilla, pie y varilla pasiva.
    ``prefix`` se antepone a los nombres y ``z`` es el plano de profundidad de la pierna; el
    primer pivote une la pelvis (que pone quien llama) con el muslo."""
    hip, knee, ankle, toe = _LEG_HIP, _LEG_KNEE, _LEG_ANKLE, _LEG_TOE
    rod_top = (0.5 * knee[0], 0.5 * knee[1])  # sobre el muslo
    heel = (ankle[0] + 0.75 * (toe[0] - ankle[0]), ankle[1] + 0.75 * (toe[1] - ankle[1]))
    foot = _link(prefix + "foot", ankle, toe, z=z)
    foot_len = math.hypot(toe[0] - ankle[0], toe[1] - ankle[1])
    foot.outline = [(-0.12, -0.12), (foot_len, -0.12), (foot_len, 0.12), (-0.12, 0.12)]
    parts = [
        _link(prefix + "thigh", hip, knee, z=z),
        _link(prefix + "shank", knee, ankle, z=z),
        foot,
        _link(prefix + "rod", rod_top, heel, z=z),
    ]
    pivots = [
        Pivot("pelvis", prefix + "thigh", hip),
        Pivot(prefix + "thigh", prefix + "shank", knee),
        Pivot(prefix + "shank", prefix + "foot", ankle),
        Pivot(prefix + "thigh", prefix + "rod", rod_top),
        Pivot(prefix + "foot", prefix + "rod", heel),
    ]
    return parts, pivots


def _pelvis() -> Part:
    outline = [(-0.7, 0.25), (0.7, 0.25), (0.7, -0.25), (-0.7, -0.25)]
    return Part("pelvis", 0.0, 0.0, 0.0, outline, fixed=True)


def leg_character() -> Character:
    """Pierna de prueba (vista lateral): cadera fija, muslo, pantorrilla, pie y una varilla pasiva
    del muslo al pie. Sola tiene 2 grados de libertad (el ángulo del muslo y el del lazo
    muslo-pantorrilla-pie-varilla), de modo que un mecanismo unido por un pasador a un punto de
    la pantorrilla la deja con 0.

    La rodilla está flexionada en reposo y la varilla se engancha lejos del tobillo: con la pierna
    estirada, o con el enganche pegado al tobillo, el reposo queda en el borde de la zona
    alcanzable y el mecanismo no puede moverla en todas las direcciones."""
    parts, pivots = _leg_chain()
    return Character("pierna", [_pelvis(), *parts], pivots)


def biped_character() -> Character:
    """Bípedo de prueba: una pelvis fija y dos piernas idénticas (prefijos ``l_`` y ``r_``) en
    planos de profundidad distintos (z = 0 y 1), como las patas de cada lado del cuadrúpedo del
    video. En la simulación 2D se superponen; solo se distinguen por ``z``. Solo tiene 4 grados
    de libertad: 2 por pierna."""
    left, left_pivots = _leg_chain("l_", z=0.0)
    right, right_pivots = _leg_chain("r_", z=1.0)
    return Character("bipedo", [_pelvis(), *left, *right], [*left_pivots, *right_pivots])


def leg_refs(
    prefix: str = "", attach: PointRef | None = None, marker: PointRef | None = None
) -> tuple[PointRef, PointRef]:
    """Punto de actuación y marcador de una pierna del personaje con ese prefijo. Sin ``attach``
    ni ``marker`` son los de la pierna de prueba; si no, los elegidos sobre una pierna sola (sin
    prefijo), que se trasladan a la pierna con ese prefijo."""
    attach, marker = attach or LEG_ATTACH, marker or LEG_MARKER
    return (
        PointRef(prefix + attach.part, attach.point),
        PointRef(prefix + marker.part, marker.point),
    )


def prefix_pose(
    pose: dict[str, tuple[float, float, float]], prefix: str
) -> dict[str, tuple[float, float, float]]:
    """Pose de las piezas de una pierna sola, con los nombres de la pierna con ese prefijo."""
    return {prefix + name: xya for name, xya in pose.items()}


# Puntos de la pierna de prueba: dónde se une el mecanismo y qué punto debe seguir la curva.
LEG_ATTACH = PointRef("shank", (0.45, -1.8))  # medio de la pantorrilla
LEG_MARKER = PointRef("foot", (0.9, -2.5))  # punta del pie
