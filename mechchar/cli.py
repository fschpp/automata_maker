"""Punto de entrada. Sin argumentos abre la GUI; el pipeline sin GUI se agregará aquí."""

from __future__ import annotations

import argparse
import sys

from mechchar import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mechchar")
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--gears", action="store_true", help="abre solo el editor de la red de engranajes"
    )
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0

    try:
        from mechchar.ui.app import run
    except (ImportError, ValueError) as exc:
        print(
            f"No se pudo cargar GTK4/PyGObject: {exc!r}\n"
            f"Python en uso: {sys.executable} ({sys.version.split()[0]})\n"
            "En Debian/Ubuntu: sudo apt install python3-gi gir1.2-gtk-4.0 python3-gi-cairo\n"
            "PyGObject de apt solo lo ve el Python del sistema: crea el venv con "
            "/usr/bin/python3 -m venv --system-site-packages .venv",
            file=sys.stderr,
        )
        return 1
    return run(gears=args.gears)


if __name__ == "__main__":
    raise SystemExit(main())
