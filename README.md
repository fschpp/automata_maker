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

## Escenas de la ventana
- **Curva libre:** arrastrá la curva y ajustá un mecanismo suelto (candidatos instantáneos de la base).
- **Pierna de prueba:** el mecanismo se une a una pierna articulada (`mechchar/core/character.py`) y se
  ajusta para que la punta del pie dibuje la curva. Un personaje propio se define en JSON
  (`Character.save` / `Character.load`).
- **Caja de engranajes:** el botón «Caja» dibuja el rectángulo donde tiene que caber el mecanismo; los
  candidatos y el ajuste lo respetan (`mechchar/optim/gearbox.py`).
- **Bípedo:** el ajuste de una pierna se usa en las dos (`mechchar/core/rig.py`), con marcha alternada de
  partida y un slider de desfase por pierna; todas comparten el mismo motor.

## Scripts
```bash
# precalcula las bases de movimientos con todos los núcleos (si no, la app las construye la
# primera vez, más lento); quedan en ~/.cache/mechchar/
python -m mechchar.db
# prueba de humo de la ventana sin pantalla: candidatos de la base, arrastrar un punto, refinar
# uno y ajustar (capturas en /tmp)
GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_ui.py
# escena «Bípedo»: ajusta una pierna, mueve el slider de desfase y dibuja a PNG (/tmp/shot_biped_*)
GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_biped.py
# caja de engranajes: la dibuja arrastrando, filtra candidatos y ajusta dentro (/tmp/shot_box_*)
GSK_RENDERER=cairo GDK_BACKEND=x11 PYTHONPATH=. xvfb-run -a python3 scripts/smoke_box.py
# gradiente por función implícita vs diferencias finitas (tiempo y error)
PYTHONPATH=. python3 scripts/bench_gradient.py
# base de movimientos: tiempo de consulta y calidad frente a Fréchet
PYTHONPATH=. python3 scripts/bench_motion_db.py --entries 3000 --curves 40
```
