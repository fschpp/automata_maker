# mechchar

Diseñador de personajes mecánicos (Coros et al., SIGGRAPH 2013) con Python + GTK4.
Plan y estado: ver `PROGRESS.md`.

## Instalación
```bash
sudo apt install python3-gi gir1.2-gtk-4.0 python3-gi-cairo   # GTK4 del sistema
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m mechchar          # abre la ventana (rueda = zoom, arrastrar = pan)
pytest && ruff check . && mypy mechchar
```
