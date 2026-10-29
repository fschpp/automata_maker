"""Modelo de edición de la escena «Red de engranajes» (M5). Sin GTK.

Es lo que la ventana edita: los ejes (posición, sentido de giro, cuál es la entrada), los pares
que el usuario eligió conectar, la cantidad de intermedios que forzó en alguno y los ejes
seleccionados para conectar. ``plan`` arma la red con ``gearnet.plan_train`` y **nunca lanza**:
si lo que hay no se puede armar devuelve el motivo en ``PlanOutcome.error`` (la ventana lo
muestra) para que se pueda seguir editando.

* Seleccionar dos ejes y conectarlos agrega un par; un par que repite otro, une ejes que
  coinciden o cierra un lazo se rechaza con ``ValueError`` (la red no admite lazos).
* ``auto``: con los pares elegidos como base, se completan solos los más cortos que falten
  (``complete_links``). Sin ``auto`` los pares tienen que unir todos los ejes.
* El eje de entrada es la referencia de giro: ``Axis.same_direction`` de cada uno es respecto de
  él. Cambiar de entrada reexpresa esos sentidos para que no cambie cómo gira cada eje.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from mechchar.core.gearnet import (
    Axis,
    GearTrain,
    complete_links,
    describe_train,
    plan_train,
    representatives,
)

MAX_SELECTED = 2


@dataclass(frozen=True)
class PlanOutcome:
    """Resultado de armar la red con lo que hay: el tren (``None`` si no se pudo ni empezar),
    los pares que se usaron (elegidos primero) y, si falló, el motivo."""

    train: GearTrain | None
    links: tuple[tuple[int, int], ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.train is not None and self.train.ok


def _key(i: int, j: int) -> frozenset[int]:
    return frozenset((i, j))


@dataclass
class GearNetModel:
    axes: list[Axis] = field(default_factory=list)
    root: int = 0
    links: list[tuple[int, int]] = field(default_factory=list)
    auto: bool = True
    forced: dict[frozenset[int], int] = field(default_factory=dict)  # par -> intermedios
    selected: list[int] = field(default_factory=list)

    # --- ejes ----------------------------------------------------------------------------------
    def _check(self, i: int) -> None:
        if not 0 <= i < len(self.axes):
            raise IndexError(f"no hay un eje {i}")

    def _free_name(self) -> str:
        names = {a.name for a in self.axes}
        n = len(self.axes) + 1
        while f"eje {n}" in names:
            n += 1
        return f"eje {n}"

    def add_axis(
        self, x: float, y: float, name: str | None = None, same_direction: bool = True
    ) -> int:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("la posición del eje no es finita")
        self.axes.append(Axis(name or self._free_name(), float(x), float(y), same_direction))
        return len(self.axes) - 1

    def remove_axis(self, i: int) -> None:
        """Quita el eje y los pares que lo usaban; los índices siguientes bajan en uno."""
        self._check(i)

        def shift(k: int) -> int:
            return k - 1 if k > i else k

        del self.axes[i]
        self.links = [(shift(a), shift(b)) for a, b in self.links if i not in (a, b)]
        self.forced = {_key(*map(shift, k)): v for k, v in self.forced.items() if i not in k}
        self.selected = [shift(k) for k in self.selected if k != i]
        if self.root == i:
            if self.axes:
                self.set_root(0)  # el nuevo eje de entrada pasa a ser la referencia de giro
            self.root = 0
        elif self.root > i:
            self.root -= 1

    def move_axis(self, i: int, x: float, y: float) -> None:
        self._check(i)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("la posición del eje no es finita")
        self.axes[i] = replace(self.axes[i], x=float(x), y=float(y))

    def rename_axis(self, i: int, name: str) -> None:
        self._check(i)
        if not name.strip():
            raise ValueError("el nombre no puede estar vacío")
        self.axes[i] = replace(self.axes[i], name=name.strip())

    def set_same_direction(self, i: int, same: bool) -> None:
        """``same``: el eje gira igual que el de entrada. La entrada siempre gira igual que sí
        misma, así que a ella no se le puede cambiar."""
        self._check(i)
        if i == self.root and not same:
            raise ValueError("el eje de entrada es la referencia: no puede girar al revés de sí")
        self.axes[i] = replace(self.axes[i], same_direction=bool(same))

    def set_root(self, i: int) -> None:
        """Cambia el eje de entrada y reexpresa los sentidos de giro respecto del nuevo."""
        self._check(i)
        ref = self.axes[i].same_direction
        self.axes = [replace(a, same_direction=(a.same_direction == ref)) for a in self.axes]
        self.root = i

    # --- selección y pares ---------------------------------------------------------------------
    def toggle_select(self, i: int) -> None:
        """Marca o desmarca el eje; con más de ``MAX_SELECTED`` se suelta el más viejo."""
        self._check(i)
        if i in self.selected:
            self.selected.remove(i)
            return
        self.selected.append(i)
        del self.selected[:-MAX_SELECTED]

    def clear_selection(self) -> None:
        self.selected.clear()

    def can_connect_selected(self) -> bool:
        return len(self.selected) == MAX_SELECTED

    def connect(self, i: int, j: int) -> tuple[int, int]:
        """Agrega el par (i, j). ``ValueError`` si repite uno, une ejes que coinciden o cierra un
        lazo con los pares elegidos."""
        self._check(i)
        self._check(j)
        if i == j:
            raise ValueError("un eje no se conecta consigo mismo")
        if _key(i, j) in {_key(a, b) for a, b in self.links}:
            raise ValueError(f"{self.axes[i].name} y {self.axes[j].name} ya están conectados")
        complete_links(self.axes, [*self.links, (i, j)])  # valida coincidencia y lazos
        self.links.append((i, j))
        return (i, j)

    def connect_selected(self) -> tuple[int, int]:
        if not self.can_connect_selected():
            raise ValueError("hay que seleccionar dos ejes")
        i, j = self.selected
        pair = self.connect(i, j)
        self.selected.clear()
        return pair

    def unlink(self, i: int, j: int) -> None:
        before = len(self.links)
        self.links = [(a, b) for a, b in self.links if _key(a, b) != _key(i, j)]
        if len(self.links) == before:
            raise ValueError("ese par no estaba conectado")
        self.forced.pop(_key(i, j), None)

    def clear_links(self) -> None:
        self.links.clear()
        self.forced.clear()

    def set_intermediates(self, i: int, j: int, k: int | None) -> None:
        """Fuerza ``k`` engranajes intermedios en el par (``None`` = automático). La paridad
        (impar = mismo sentido, par = contrario) se revisa al armar la red."""
        if k is not None and k < 0:
            raise ValueError("la cantidad de intermedios no puede ser negativa")
        if _key(i, j) not in {_key(a, b) for a, b in self._effective_links()}:
            raise ValueError("ese par no está conectado")
        if k is None:
            self.forced.pop(_key(i, j), None)
        else:
            self.forced[_key(i, j)] = k

    # --- la red --------------------------------------------------------------------------------
    def _effective_links(self) -> list[tuple[int, int]]:
        if not self.axes:
            return []
        try:
            return complete_links(self.axes, self.links) if self.auto else list(self.links)
        except ValueError:
            return list(self.links)

    def plan(self, **kwargs: object) -> PlanOutcome:
        """Arma la red con lo que hay. ``kwargs`` pasa a ``plan_train`` (``module``,
        ``layer_pitch``). No lanza: el motivo queda en ``error``."""
        if not self.axes:
            return PlanOutcome(None, (), "Agregá ejes para armar la red.")
        links: list[tuple[int, int]] = []
        try:
            links = complete_links(self.axes, self.links) if self.auto else list(self.links)
            intermediates = {tuple(sorted(k)): v for k, v in self.forced.items()}
            if len(self.axes) == 1:
                train = plan_train(self.axes, self.root, **kwargs)  # type: ignore[arg-type]
            else:
                train = plan_train(
                    self.axes,
                    self.root,
                    links=links,
                    intermediates=intermediates,
                    **kwargs,  # type: ignore[arg-type]
                )
        except ValueError as exc:
            return PlanOutcome(None, tuple(links), str(exc))
        return PlanOutcome(train, tuple(links))

    def describe(self, outcome: PlanOutcome | None = None) -> tuple[str, bool]:
        """Frase para el estado de la ventana y si está bien."""
        out = outcome or self.plan()
        if out.error is not None:
            return out.error, False
        assert out.train is not None
        return describe_train(out.train)

    def auto_links(self) -> list[tuple[int, int]]:
        """Los pares que se agregan solos (los que no eligió el usuario)."""
        chosen = {_key(a, b) for a, b in self.links}
        return [(a, b) for a, b in self._effective_links() if _key(a, b) not in chosen]

    def representatives(self) -> list[int]:
        return representatives(self.axes)

    @classmethod
    def example(cls) -> GearNetModel:
        """Cuatro ejes repartidos, el segundo gira al revés: sirve para probar la ventana."""
        m = cls()
        for x, y, name, same in (
            (0.0, 0.0, "entrada", True),
            (9.0, 1.0, "pierna izq.", True),
            (4.0, 8.0, "pierna der.", False),
            (13.0, 9.0, "brazo", True),
        ):
            m.add_axis(x, y, name, same)
        return m
