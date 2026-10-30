"""Dibuja las capas de un rig de ejemplo (dos cuadriláteros unidos por una red de engranajes) en
un PNG con matplotlib: un panel por capa, con las piezas en un instante del ciclo.

    python scripts/preview_layers.py [salida.png] [paso]

No necesita GTK. Sirve para revisar a ojo qué piezas quedaron en cada capa.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mechchar.core.assembly import Assembly, Drive, FixedState  # noqa: E402
from mechchar.core.character import Character  # noqa: E402
from mechchar.core.gearlink import plan_rig_train  # noqa: E402
from mechchar.core.layerbuild import build_layer_problem  # noqa: E402
from mechchar.core.layers import assign_layers, layer_offsets  # noqa: E402
from mechchar.core.mechanisms import four_bar  # noqa: E402
from mechchar.core.rig import SubAssembly  # noqa: E402

PALETTE = ["#d9a520", "#4a78c2", "#4aa06a", "#b5527a", "#7a6bc2", "#c2704a"]


def example_rig(placements=((0.0, 0.0), (16.0, 3.0))):
    asm, states, subs = Assembly(), [np.zeros(0)], []
    for k, (dx, dy) in enumerate(placements):
        fb = four_bar(d=6.0, r=1.2, l=5.0, rr=4.0)
        fb.asm.constraints[0] = FixedState(fb.ground, dx, dy, 0.0)
        s = fb.state0.copy()
        s[0::3] += dx
        s[1::3] += dy
        n_before = len(asm.constraints)
        first = asm.append(fb.asm)
        drive = next(c for c in asm.constraints[n_before:] if isinstance(c, Drive))
        subs.append(
            SubAssembly(f"m{k}", fb, None, None, first, len(fb.asm.components), drive,
                        drive.offset, 0, (0.0, 0.0))
        )  # fmt: skip
        states.append(s)
    return SimpleNamespace(
        character=Character("vacío"), n_char=0, asm=asm, state0=np.concatenate(states), subs=subs
    )


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "layers.png"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rig = example_rig()
    train, axes = plan_rig_train(rig)
    prob = build_layer_problem(rig, train, axes, n_steps=24)
    plan = assign_layers(prob.bodies, prob.pins, meshes=prob.meshes)
    if not plan.ok:
        print("sin solución:", plan.problems)
        return
    thick, z = layer_offsets(plan, prob.bodies)
    print(f"{plan.n_layers} capas, pasadores {plan.pin_length}, óptimo={plan.optimal}")
    for k in range(plan.n_layers):
        print(f"  capa {k} (z={z[k]:.1f}, espesor {thick[k]:.1f}):",
              ", ".join(b.name for b in prob.bodies if plan.layers[b.name] == k))  # fmt: skip
    fig, axs = plt.subplots(1, plan.n_layers, figsize=(4.2 * plan.n_layers, 4.2), squeeze=False)
    for k, ax in enumerate(axs[0]):
        for b in prob.bodies:
            if plan.layers[b.name] != k:
                continue
            poly = b.shapes[step]
            ax.fill(
                poly[:, 0], poly[:, 1], color=PALETTE[k % len(PALETTE)], alpha=0.45, ec="k", lw=0.7
            )
        for pin in prob.pins:
            ls = [plan.layers[n] for n in pin.members]
            if min(ls) <= k <= max(ls):
                ax.plot(*pin.path[step], "ko", ms=3)
        ax.set_aspect("equal")
        ax.set_title(f"capa {k}")
        ax.set_xlim(-6, 26)
        ax.set_ylim(-6, 12)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()
