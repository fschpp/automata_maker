"""Proyecto guardable y historial de deshacer/rehacer (M4). Sin GTK.

El estado del proyecto es una foto inmutable (``ProjectState``) de lo que el usuario edita: la
escena, los puntos de la spline, la caja de engranajes, el mecanismo ajustado y los desfases. No
guarda ni la curva densa ni el mecanismo armado: se reconstruyen desde los parámetros, así que el
archivo es chico y no puede quedar inconsistente con ellos.

``History`` guarda fotos; cada acción que el usuario hace (arrastrar un punto, dibujar la caja,
ajustar, mover un slider) es una foto nueva, y deshacer/rehacer solo cambian cuál está vigente.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

FORMAT_VERSION = 1
MAX_PARAMS = 64  # cota holgada: las familias actuales usan menos de 16
MAX_POINTS = 256

T = TypeVar("T")


def _finite(values: Any, what: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: hay valores que no son números") from exc
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what}: hay valores no finitos")
    return out


@dataclass(frozen=True)
class FitRecord:
    """Un mecanismo ajustado: lo mínimo para volver a armarlo."""

    family: str
    params: tuple[float, ...]
    branch: int = 1
    ratio: float = 0.0
    rms_rel: float = 0.0  # error relativo al ajustar (solo informativo)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("el mecanismo guardado no tiene familia")
        if not 0 < len(self.params) <= MAX_PARAMS:
            raise ValueError("el mecanismo guardado tiene una cantidad de parámetros inválida")
        if self.branch not in (-1, 1):
            raise ValueError("la rama del mecanismo guardado debe ser +1 o −1")
        if not (math.isfinite(self.ratio) and math.isfinite(self.rms_rel) and self.rms_rel >= 0):
            raise ValueError("el mecanismo guardado tiene valores inválidos")
        _finite(self.params, "parámetros del mecanismo")


@dataclass(frozen=True)
class ProjectState:
    scene: int = 0
    spline: tuple[tuple[float, float], ...] = ()
    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    fit: FitRecord | None = None
    phases: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.scene < 0:
            raise ValueError("escena inválida")
        if not 3 <= len(self.spline) <= MAX_POINTS:
            raise ValueError("la spline necesita entre 3 y 256 puntos")
        for pt in self.spline:
            if len(pt) != 2:
                raise ValueError("cada punto de la spline tiene dos coordenadas")
            _finite(pt, "puntos de la spline")
        if self.box is not None:
            if len(self.box) != 4:
                raise ValueError("la caja de engranajes necesita 4 valores")
            x0, y0, x1, y1 = _finite(self.box, "caja de engranajes")
            if not (x0 < x1 and y0 < y1):
                raise ValueError("la caja de engranajes está mal formada")
        _finite(self.phases, "desfases")

    # --- JSON -------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "scene": self.scene,
            "spline": [list(p) for p in self.spline],
            "box": None if self.box is None else list(self.box),
            "fit": None
            if self.fit is None
            else {
                "family": self.fit.family,
                "params": list(self.fit.params),
                "branch": self.fit.branch,
                "ratio": self.fit.ratio,
                "rms_rel": self.fit.rms_rel,
            },
            "phases": list(self.phases),
        }

    @classmethod
    def from_dict(cls, d: Any) -> ProjectState:
        if not isinstance(d, dict):
            raise ValueError("el archivo no es un proyecto")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError("versión de proyecto incompatible")
        try:
            raw_fit = d.get("fit")
            fit = (
                None
                if raw_fit is None
                else FitRecord(
                    str(raw_fit["family"]),
                    _finite(raw_fit["params"], "parámetros del mecanismo"),
                    int(raw_fit.get("branch", 1)),
                    float(raw_fit.get("ratio", 0.0)),
                    float(raw_fit.get("rms_rel", 0.0)),
                )
            )
            raw_box = d.get("box")
            return cls(
                int(d["scene"]),
                tuple(_point(pt) for pt in d["spline"]),
                None if raw_box is None else _box_tuple(raw_box),
                fit,
                _finite(d.get("phases", ()), "desfases"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"proyecto incompleto o mal formado ({exc!r})") from exc


def _point(raw: Any) -> tuple[float, float]:
    vals = _finite(raw, "puntos de la spline")
    if len(vals) != 2:
        raise ValueError("cada punto de la spline tiene dos coordenadas")
    return vals[0], vals[1]


def _box_tuple(raw: Any) -> tuple[float, float, float, float]:
    vals = _finite(raw, "caja de engranajes")
    if len(vals) != 4:
        raise ValueError("la caja de engranajes necesita 4 valores")
    return vals[0], vals[1], vals[2], vals[3]


def save_project(path: str | Path, state: ProjectState) -> None:
    """Escribe el proyecto de forma atómica: un fallo a mitad no deja un archivo roto."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_project(path: str | Path) -> ProjectState:
    """Lee un proyecto; cualquier problema del archivo es un ``ValueError`` con un mensaje claro."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("el archivo no es un proyecto válido (no es JSON)") from exc
    return ProjectState.from_dict(raw)


# --- historial ---------------------------------------------------------------------------------


class History(Generic[T]):
    """Pila de fotos con deshacer/rehacer. ``commit`` ignora lo que no cambió y, con la misma
    ``key`` en un intervalo corto (arrastrar un slider), reemplaza la foto vigente en vez de
    apilar una por cada pixel."""

    def __init__(self, initial: T, limit: int = 100, coalesce_s: float = 1.0) -> None:
        if limit < 2:
            raise ValueError("el historial necesita lugar para al menos 2 fotos")
        self._states: list[T] = [initial]
        self._keys: list[str | None] = [None]
        self._times: list[float] = [-math.inf]
        self._i = 0
        self.limit, self.coalesce_s = limit, coalesce_s

    @property
    def present(self) -> T:
        return self._states[self._i]

    @property
    def can_undo(self) -> bool:
        return self._i > 0

    @property
    def can_redo(self) -> bool:
        return self._i < len(self._states) - 1

    def commit(self, state: T, key: str | None = None, now: float | None = None) -> bool:
        """Registra ``state`` como la nueva foto vigente; False si no cambió nada."""
        if state == self.present:
            return False
        now = time.monotonic() if now is None else now
        merge = (
            key is not None
            and self._keys[self._i] == key
            and self._i > 0
            and now - self._times[self._i] < self.coalesce_s
            and not self.can_redo
        )
        if merge:
            self._states[self._i], self._times[self._i] = state, now
            return True
        del self._states[self._i + 1 :], self._keys[self._i + 1 :], self._times[self._i + 1 :]
        self._states.append(state)
        self._keys.append(key)
        self._times.append(now)
        if len(self._states) > self.limit:
            for lst in (self._states, self._keys, self._times):
                del lst[0]
        self._i = len(self._states) - 1
        return True

    def undo(self) -> T | None:
        if not self.can_undo:
            return None
        self._i -= 1
        return self.present

    def redo(self) -> T | None:
        if not self.can_redo:
            return None
        self._i += 1
        return self.present

    def reset(self, state: T) -> None:
        """Empieza un historial nuevo (al abrir un proyecto)."""
        self._states, self._keys, self._times, self._i = [state], [None], [-math.inf], 0
