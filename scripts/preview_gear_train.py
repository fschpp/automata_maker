"""Dibuja una red de engranajes con dientes en un PNG (matplotlib; no hace falta GTK).

    python scripts/preview_gear_train.py [salida.png] [theta_en_grados]

Sirve para revisar a ojo que los dientes engranan y que el eje de entrada y las capas se ven bien.
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mechchar.core.gearnet import Axis, plan_train  # noqa: E402
from mechchar.core.gearshape import gear_outline, gear_rotations  # noqa: E402

AXES = [Axis("entrada", 0, 0), Axis("b", 9, 1), Axis("c", 4, 8, same_direction=False)]
LAYER_COLORS = ["#d9a520", "#4a78c2", "#4aa06a", "#b5527a"]


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "gear_train.png"
    theta = math.radians(float(sys.argv[2])) if len(sys.argv) > 2 else 0.0
    train = plan_train(AXES, 0)
    print("ok" if train.ok else f"problemas: {train.problems}", f"módulo {train.module:.3f}")
    fig, ax = plt.subplots(figsize=(7, 7))
    for c, rots in zip(train.connections, gear_rotations(train, theta), strict=True):
        color = LAYER_COLORS[c.layer % len(LAYER_COLORS)]
        for g, r in zip(c.gears, rots, strict=True):
            poly = gear_outline(g, r)
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.35, ec="k", lw=0.8)
            ax.plot(
                [g.x, g.x + g.radius * math.cos(r)],
                [g.y, g.y + g.radius * math.sin(r)],
                "k",
                lw=0.6,
            )
    for a in train.axes:
        ax.plot(a.x, a.y, "ko", ms=4)
        ax.annotate(a.name, (a.x, a.y), textcoords="offset points", xytext=(4, 4))
    ax.set_aspect("equal")
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
