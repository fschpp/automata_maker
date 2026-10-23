"""Precalcula las bases de movimientos con todos los núcleos (se guardan en el caché del usuario).

    python -m mechchar.db [--entries 3000] [--force]
"""

import argparse
import time

from mechchar.db.motion_db import build_database, default_db_path
from mechchar.optim.families import ALL_FAMILIES


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m mechchar.db")
    ap.add_argument("--entries", type=int, default=3000, help="muestras por familia")
    ap.add_argument("--force", action="store_true", help="reconstruir aunque ya exista")
    args = ap.parse_args()
    for fam in ALL_FAMILIES:
        path = default_db_path(fam.name)
        if path.exists() and not args.force:
            print(f"[{fam.name}] ya existe: {path}")
            continue
        t0 = time.perf_counter()
        db = build_database(fam.name, n_target=args.entries, n_samples=args.entries * 40)
        db.save(path)
        print(f"[{fam.name}] {len(db)} entradas en {time.perf_counter() - t0:.1f} s -> {path}")


if __name__ == "__main__":
    main()
