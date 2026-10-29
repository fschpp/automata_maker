"""Dibujo de la escena «Red de engranajes»: el editor sobre el canvas y la vista lateral.

Solo recibe un contexto ``cr`` de Cairo (sin importar GTK), así que se prueba con un ``cr`` de
mentira.

* ``draw_editor``: los pares (los elegidos con línea continua, los que se completan solos
  punteados), la red con dientes (``scene.draw_gear_train``) y los ejes con su nombre; el de
  entrada lleva un aro naranja, los seleccionados uno azul oscuro y los que giran al revés de la
  entrada un ↺.
* ``draw_side_view``: la red vista de frente (x horizontal, z vertical): cada engranaje es un
  rectángulo de su diámetro y el espesor puesto en su capa, y cada eje cilíndrico una barra de su
  profundidad (``gearnet.axles``); el de entrada, en naranja, sobresale.
"""

from __future__ import annotations

import math
from typing import Any

from mechchar.core.gearedit import GearNetModel, PlanOutcome
from mechchar.core.gearnet import GearTrain, axles
from mechchar.core.viewport import Viewport
from mechchar.ui.scene import (
    GEAR_EDGE,
    INPUT_FILL,
    LAYER_FILLS,
    SELECTED_FILL,
    draw_gear_train,
)

AXIS_RADIUS_PX = 6.0
RING_PX = 11.0
CHOSEN_LINK = (0.10, 0.12, 0.45)
AUTO_LINK = (0.45, 0.50, 0.62)
AXLE_FILL = (0.22, 0.22, 0.25)
INPUT_AXLE_FILL = (0.95, 0.5, 0.05)
TEXT_COLOR = (0.15, 0.15, 0.2)
SIDE_MARGIN_PX = 28.0


def _dashed(cr: Any, on: float, off: float) -> None:
    cr.set_dash([on, off], 0.0)


def _solid(cr: Any) -> None:
    cr.set_dash([], 0.0)


def draw_editor(
    cr: Any,
    vp: Viewport,
    model: GearNetModel,
    outcome: PlanOutcome,
    theta: float = 0.0,
) -> None:
    """Dibuja el estado del editor sobre la vista ``vp``."""
    chosen = {frozenset(p) for p in model.links}
    for a, b in outcome.links:
        solid = frozenset((a, b)) in chosen
        cr.set_source_rgb(*(CHOSEN_LINK if solid else AUTO_LINK))
        cr.set_line_width(2.0 if solid else 1.5)
        if not solid:
            _dashed(cr, 6.0, 5.0)
        cr.move_to(*vp.world_to_screen(model.axes[a].x, model.axes[a].y))
        cr.line_to(*vp.world_to_screen(model.axes[b].x, model.axes[b].y))
        cr.stroke()
        _solid(cr)

    if outcome.train is not None:
        draw_gear_train(cr, vp, outcome.train, theta, set(model.selected))

    cr.set_font_size(13.0)
    for i, axis in enumerate(model.axes):
        sx, sy = vp.world_to_screen(axis.x, axis.y)
        if i == model.root or i in model.selected:
            cr.arc(sx, sy, RING_PX, 0.0, 2.0 * math.pi)
            cr.set_source_rgba(*(INPUT_FILL if i == model.root else SELECTED_FILL)[:3], 1.0)
            cr.set_line_width(3.0)
            cr.stroke()
        if i == model.root and i in model.selected:  # los dos: aro azul por fuera del naranja
            cr.arc(sx, sy, RING_PX + 4.0, 0.0, 2.0 * math.pi)
            cr.set_source_rgba(*SELECTED_FILL[:3], 1.0)
            cr.set_line_width(2.0)
            cr.stroke()
        cr.arc(sx, sy, AXIS_RADIUS_PX, 0.0, 2.0 * math.pi)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.fill_preserve()
        cr.set_source_rgb(*TEXT_COLOR)
        cr.set_line_width(1.5)
        cr.stroke()
        label = axis.name + ("" if axis.same_direction else " ↺")
        cr.set_source_rgb(*TEXT_COLOR)
        cr.move_to(sx + RING_PX + 4.0, sy - RING_PX)
        cr.show_text(label)


def side_view_bounds(
    train: GearTrain, thickness: float, protrusion: float
) -> tuple[float, float, float, float]:
    """(x0, z0, x1, z1) en mundo de lo que hay que mostrar en la vista lateral."""
    gears = train.gears()
    xs0 = [g.x - g.radius for g in gears]
    xs1 = [g.x + g.radius for g in gears]
    zs = [a.z1 for a in axles(train, thickness, protrusion)]
    return min(xs0), 0.0, max(xs1), max(zs + [thickness])


def draw_side_view(
    cr: Any,
    width: float,
    height: float,
    train: GearTrain | None,
    thickness: float = 1.0,
    protrusion: float = 2.0,
) -> None:
    """Vista de frente de la red: engranajes por capa y ejes cilíndricos con su profundidad."""
    cr.set_source_rgb(0.97, 0.97, 0.97)
    cr.paint()
    cr.set_font_size(12.0)
    if train is None or not train.connections:
        cr.set_source_rgb(*TEXT_COLOR)
        cr.move_to(SIDE_MARGIN_PX, height / 2.0)
        cr.show_text("Sin engranajes que mostrar en profundidad")
        return
    x0, z0, x1, z1 = side_view_bounds(train, thickness, protrusion)
    avail_w = max(width - 2 * SIDE_MARGIN_PX, 1.0)
    avail_h = max(height - 2 * SIDE_MARGIN_PX, 1.0)
    scale = min(avail_w / max(x1 - x0, 1e-9), avail_h / max(z1 - z0, 1e-9))
    left = (width - (x1 - x0) * scale) / 2.0
    base = height - SIDE_MARGIN_PX

    def sx(x: float) -> float:
        return left + (x - x0) * scale

    def sz(z: float) -> float:
        return base - (z - z0) * scale

    def rect(xa: float, za: float, xb: float, zb: float, fill: tuple, edge: tuple) -> None:
        cr.rectangle(sx(xa), sz(zb), (xb - xa) * scale, (zb - za) * scale)
        if len(fill) == 4:
            cr.set_source_rgba(*fill)
        else:
            cr.set_source_rgb(*fill)
        cr.fill_preserve()
        cr.set_source_rgb(*edge)
        cr.set_line_width(1.0)
        cr.stroke()

    for c in train.connections:
        fill = LAYER_FILLS[c.layer % len(LAYER_FILLS)]
        for g in c.gears:
            rect(g.x - g.radius, g.z, g.x + g.radius, g.z + thickness, fill, GEAR_EDGE)
    half = 0.5 * train.module
    for a in axles(train, thickness, protrusion):
        rect(
            a.x - half,
            a.z0,
            a.x + half,
            a.z1,
            INPUT_AXLE_FILL if a.input else AXLE_FILL,
            TEXT_COLOR,
        )
        if not a.name.startswith("intermedio"):
            cr.set_source_rgb(*TEXT_COLOR)
            cr.move_to(sx(a.x) - 3.0 * len(a.name), base + 16.0)
            cr.show_text(a.name)
    cr.set_source_rgb(*TEXT_COLOR)  # el piso (z = 0)
    cr.set_line_width(1.0)
    cr.move_to(SIDE_MARGIN_PX / 2.0, base)
    cr.line_to(width - SIDE_MARGIN_PX / 2.0, base)
    cr.stroke()
