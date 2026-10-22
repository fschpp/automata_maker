# PROGRESS.md — Mechanical Characters Designer (Python + GTK4)

Basado en: Coros et al., *Computational Design of Mechanical Characters* (SIGGRAPH 2013, Disney Research).
Última actualización: 2026-10-22 · Rama: `main` · Plan: v3.0 · Estado: M0 casi cerrado (falta verificar la ventana GTK en tu máquina)

## Current Focus
Cerrar M0 (abrir la ventana en tu máquina con `python -m mechchar` y confirmar grilla/zoom/pan) y pasar a M1: `Component`, `Assembly`, conexiones y solver. Objetivo general: llegar lo antes posible al primer "corte vertical" funcional: un cuadrilátero articulado animado en GTK que ajusta una curva dibujada con el mouse (M1 + M2), antes de construir la base de movimientos.

## Principios del plan v2
1. **Cortes verticales, no capas:** cada hito termina en algo que se ve y se prueba en la app. El paper construye la base de movimientos primero; acá la dejamos para M3 como *optimización de velocidad y calidad* de algo que ya funciona.
2. **Núcleo sin GTK:** todo `core/`, `db/`, `optim/` y `export/` corre en consola y con pytest. GTK es solo una capa de presentación.
3. **Criterio de aceptación medible por hito** (ver tabla), para saber cuándo avanzar.
4. **Simplificar donde el paper no obliga:** 2D con 3 DOF por pieza (x, y, θ) en vez de 6 DOF; sin métrica entrenada, sin engranajes no circulares y sin 3D en el MVP.

## Stack
| Necesidad | Librería |
|---|---|
| UI | GTK4 + PyGObject (libadwaita opcional); `Gtk.DrawingArea.set_draw_func` + Cairo; `Gtk.GestureDrag` para dibujar curvas; `add_tick_callback` para animar; `Gtk.FileDialog` (GTK ≥ 4.10) |
| Numérico / optimización | numpy, scipy (`least_squares`, `minimize` con BFGS y SLSQP, `RBFInterpolator`) |
| Geometría 2D | shapely (STRtree para colisiones) |
| Layering (CSP) | OR-tools CP-SAT |
| Mallas / STL | trimesh (extrusión de polígonos 2D) |
| Persistencia | JSON versionado para proyectos; `.npz` para la base de movimientos |
| Calidad | pytest, ruff, mypy; pytest-benchmark para la consulta a la base |

## Arquitectura
```
mechchar/
  core/     # Component, Connection, Assembly, solver, Mechanism (clase base), Curve
  db/       # muestreo, índice (features -> KD-tree), consulta
  optim/    # ajuste continuo, gear train, layering, soportes
  export/   # engranajes evolventes, eslabones, ejes, STL
  project/  # esquema JSON, comandos (undo/redo), migraciones
  ui/       # ventana, canvas, paneles, timeline  (solo presentación)
  cli.py    # ejecuta el pipeline sin GUI (útil para tests y precómputo)
tests/  data/  docs/
```
Reglas: `ui` depende de `core`, nunca al revés. Las optimizaciones largas corren en un hilo/proceso aparte y devuelven resultados con `GLib.idle_add`; la UI nunca se bloquea. Cada acción del usuario es un *comando* (undo/redo desde el día 1).

## Entrada y salida
**Entrada:**
1. Un **personaje articulado**: piezas rígidas con pivotes, definido en un SVG con capas (`part_*`) y círculos (`pivot_*`) o en un JSON equivalente.
2. Las **curvas de movimiento**: el usuario elige los puntos de actuación y dibuja cada curva como una spline cerrada editable.

**Pipeline:**
1. Cargar el personaje y elegir los puntos de actuación.
2. Dibujar las curvas y obtener, por cada una, un mecanismo optimizado (M2/M3).
3. Tren de engranajes (M5) → capas sin colisión (M6) → soportes (M8).
4. **Salida:** un STL por pieza del personaje + un STL por cada pieza del mecanismo (engranajes, eslabones, ejes, pines, paredes de soporte) + placa(s) de impresión + guía de ensamblaje con el orden de las capas.

**Límites del paper:** solo movimientos cíclicos; el pipeline completo es para mecanismos planares (el movimiento 3D se logra combinando planos). El *timing* no se controla en el MVP: se hace con engranajes no circulares (post-MVP), definiendo el perfil de fase con muestras (α₁, α₂).

## Hitos y criterios de aceptación

| Hito | Entregable visible | Criterio de aceptación | Tamaño* |
|---|---|---|---|
| M0 | Repo + ventana GTK con canvas | `pytest` y `ruff` en verde; ventana abre y dibuja una grilla con zoom/pan | S |
| M1 | Cuadrilátero animado en GTK | Curva del acoplador igual a la solución analítica (error < 1e-6); chequeo de gradiente vs diferencias finitas (error relativo < 1e-5) | M |
| M2 | Dibujás una curva y un mecanismo la ajusta | Elipse y curva "frijol" ajustadas con RMS < 3 % del tamaño de la curva, con reinicios aleatorios + `least_squares` (sin base de datos); en la pierna de prueba, el pie sigue una elipse objetivo | M |
| M3 | Base de movimientos + consulta | Consulta < 100 ms; el top-1 supera a Fréchet en un set de curvas dibujadas a mano (replicar Fig. 9); convergencia de M2 más rápida usando el inicial de la base | L |
| M4 | Personaje con 1–4 mecanismos y un solo motor | El personaje de ejemplo se anima con un único Input Driver; con 4 sub-ensamblajes, mover un slider de desfase actualiza la pose en vivo y de forma fluida; el sistema avisa si está sobre/subrestringido | L |
| M5 | Tren de engranajes entre mecanismos | Engranajes engranan (distancia = suma de radios), sin intersecciones, velocidades correctas | M |
| M6 | Capas sin colisión | Animación completa sin colisiones entre piezas; nº de capas mínimo encontrado en < 30 s para ~50 piezas | M |
| M7 | Exportación STL + primera impresión | Un STL por pieza + placa; mecanismo simple impreso que gira sin trabarse | M |
| M8 | Soportes y pulido | Ejes sostenidos por paredes; guardar/cargar proyecto; empaquetado básico | M |

\*S ≈ 1 semana, M ≈ 2 semanas, L ≈ 3 semanas, a ~10 h/semana. **MVP completo ≈ 16–20 semanas.**

## Todo List

### [x] Done
- [x] Lectura del paper y del video complementario (DisneyResearchHub, 2013)
- [x] Plan v1: stack, estructura de módulos y fases (2026-10-22)
- [x] Plan v2: reordenado en cortes verticales con criterios de aceptación, riesgos y decisiones de diseño (2026-10-22)
- [x] Plan v2.1: entrada = archivo de animación 3D (glTF/GLB), salida = set de STL para imprimir y ensamblar; M4 reescrito (2026-10-22) — la importación de animación se descartó en v2.4
- [x] Revisión de las capturas del video (0:05–0:59): Pushing Man, exploración del espacio de parámetros, consultas a la base y optimización continua (2026-10-22)
- [x] Plan v2.2: UI de curvas como spline editable, consulta en vivo, visualización de la optimización y opción de varillas externas (2026-10-22)
- [x] Revisión de capturas de la escena de la pierna (~1:32): marcador en un punto arbitrario del personaje (2026-10-22)
- [x] Plan v2.3: escena de prueba de la pierna en M2 y chequeo de grados de libertad en M4 (2026-10-22)
- [x] Plan v2.4: se descarta importar archivos de animación; la entrada vuelve a ser un personaje articulado propio + curvas dibujadas como splines (2026-10-22)
- [x] Revisión de capturas de la sesión interactiva con el cuadrúpedo (1:50–1:52 y siguientes): piezas fijas, caja de engranajes, marcador, spline, candidatos y desfase (2026-10-22)
- [x] Plan v2.5: flujo de herramientas de la UI, selección entre candidatos y control de desfase (2026-10-22)
- [x] Revisión de capturas de la continuación de la sesión (1:57–2:04 y siguientes): mecanismo colocado en la caja, segunda pata, dos sliders y vista 3D (2026-10-22)
- [x] Plan v2.6: plano z por sub-ensamblaje, vista simplificada en edición y visor 3D de capas (2026-10-22)
- [x] Revisión de capturas de la sección "Phase Shift Manipulation" (2:15–2:24): cuatro sliders, uno por pata (2026-10-22)
- [x] Plan v2.7: panel de desfase con actualización en vivo y offset de fase en las conexiones (2026-10-22)
- [x] Revisión de capturas del inicio del tren de engranajes con el T-Rex: selección de pares, diálogo de engranajes intermedios y eje común (2026-10-22)
- [x] Plan v2.8: M5 ampliado con selección, paridad de engranajes intermedios y eje común (2026-10-22)
- [x] Revisión de más capturas del tren de engranajes del T-Rex: conexión repetida par por par y eje de entrada (2026-10-22)
- [x] Plan v2.9: conexión iterativa hasta un único eje de entrada exportable (2026-10-22)
- [x] M0 (parte automática): `pyproject.toml`, estructura `mechchar/`, CI (ruff + mypy + pytest), `core/viewport.py` (mundo↔pantalla, zoom anclado al cursor, pan, grilla 1-2-5) con 16 tests en verde; `ui/` con `Canvas` (Cairo + `GestureDrag` + `EventControllerScroll`), ventana y `cli.py` (2026-10-22)
- [x] Revisión de capturas del tramo final (~3:40–4:10 según la barra de progreso, estimación mía): cuadrúpedo "EMA" en WALK y GALOP (render y prototipos físicos) y T-Rex con engranajes cónicos y ejes entre paredes (2026-10-22)
- [x] Plan v3.0: paredes de soporte en L, engranajes en varias capas sobre ejes perpendiculares a la pared, caja como pila de placas, eje de entrada para taladro y nota sobre engranajes no circulares en el galope (2026-10-22)

### [/] In Progress
- [/] M0 — Falta: instalar GTK4 (`sudo apt install python3-gi gir1.2-gtk-4.0 python3-gi-cairo`), ejecutar `python -m mechchar` y confirmar que la ventana dibuja la grilla y responde a rueda/arrastre; luego `git init` y primer commit

### [ ] Todo

**Investigación**
- [ ] Revisar capturas de los tramos aún no vistos del video (~1:00–1:31, ~1:34–1:49, ~2:05–2:14, ~2:25–3:40 salvo el T-Rex ya visto, y ~4:10–5:09): capas sin colisión, generación de soportes, engranajes no circulares y los demás personajes finales. Pendiente de confirmar: las miniaturas de la barra de tiempo hacían pensar en un tigre dientes de sable (Cyber Tiger del paper); las capturas nuevas muestran el cuadrúpedo "EMA" y no el tigre, que sigue sin verse

**M1 — Simulador + visor (paper §3)**
- [ ] `Component` 2D (x, y, θ) y `Assembly`, con un campo `z` (plano de profundidad) desde el principio, aunque la simulación sea 2D
- [ ] Conexiones: Pin, Point-On-Line, Phase (engranajes, ratio −r, con un offset de fase φ opcional para el desfase entre mecanismos), Fixed State
- [ ] Solver de min ½CᵀC con `least_squares` y Jacobiano analítico; luego Newton propio si hace falta
- [ ] **Continuación por fase:** cada paso arranca desde la solución anterior para seguir la misma rama de ensamblado (evita saltos de configuración del cuadrilátero)
- [ ] Detección de ciclo completo: rechazar mecanismos que se traban o saltan de rama (condición de Grashof / salto máximo entre fases consecutivas)
- [ ] Detección sobre/subrestringido (mínimo ≠ 0 / Hessiano singular)
- [ ] Input Driver con fase t ∈ [0, 1] y animación cíclica en el canvas
- [ ] Tests: cuadrilátero y manivela-corredera contra soluciones analíticas; test de gradiente

**M2 — Ajuste de curva sin base de datos (paper §4.2)**
- [ ] Mecanismos parametrizados: cuadrilátero, 5-bar con 2 engranajes (ratios −2, −1, 1, 2), manivela-corredera
- [ ] Marcador (punto que traza la curva) y curva en un ciclo
- [ ] Objetivo F = ½∫|x(p,s_t) − x̂_t|² dt discretizado
- [ ] Gradiente por Teorema de la Función Implícita: ∂s/∂p = −(∂C/∂s)⁻¹ ∂C/∂p (∂C/∂p por diferencias finitas, ∂C/∂s analítico, pseudo-inversa si hay restricciones redundantes)
- [ ] BFGS con reinicios aleatorios; comparar contra diferencias finitas completas (el paper reporta 20–30× de ganancia)
- [ ] UI: curva cerrada editable como spline con puntos de control arrastrables (como en el video) → ver mecanismo ajustado y animación
- [ ] Mostrar en vivo la curva objetivo (rojo) y la curva actual (blanco) mientras converge la optimización
- [ ] Prueba de robustez: partir de un inicial deliberadamente malo y comprobar que converge igual (el video lo muestra)
- [ ] **Marcador sobre una pieza del personaje** (Fig. 5 derecha): escena de prueba de una pierna (muslo con pivote fijo, tibia/pie y una varilla pasiva) accionada por un mecanismo de dos engranajes; optimizar para que el pie siga una elipse objetivo partiendo de una curva en forma de ocho, como en el video. Aquí el estado s_t incluye los grados de libertad del personaje
- [ ] Alternativa a evaluar: autodiff con JAX para ahorrar derivadas a mano

**M3 — Métrica y base de movimientos (paper §4.1, §5)**
- [ ] Remuestreo uniforme, normalización (ancla al origen, PCA, escala 1/l_max)
- [ ] Features f1..f6 (longitud, área, elipticidad, posición relativa, orientación, autointersecciones)
- [ ] d_pos y d_angle con desplazamiento del índice de inicio, **calculados por correlación cruzada con FFT** (O(n log n) en vez de O(n²))
- [ ] Decisión de diseño: la curva dibujada puede ir en sentido horario o antihorario → consultar también la versión invertida
- [ ] Baseline: Fréchet discreto, para comparar como en la Fig. 9
- [ ] Muestreo recursivo estilo Poisson-disk en espacio métrico, **descartando configuraciones infactibles** (que no completan el ciclo, según M1)
- [ ] Base de ≈ 3000 muestras por tipo; precómputo con `multiprocessing`; índice KD-tree sobre features para preseleccionar y luego refinar con la métrica completa
- [ ] Consulta: curva → mejor mecanismo + transformación (escala, rotación, traslación); descartar los que no caben en la caja de engranajes
- [ ] Consulta en vivo mientras se arrastran los puntos de control de la spline, con varias curvas a la vez; superponer la curva del mecanismo encontrado sobre la objetivo
- [ ] Galería de miniaturas de la base para depurar el muestreo
- [ ] Mostrar los 3 mejores candidatos en colores (rojo, verde, azul) con su caja delimitadora y dejar elegir uno, como en el video; los que no caben en la caja de engranajes se descartan
- [ ] (Post-MVP) Entrenamiento de coeficientes con SLSQP + UI de pares similares/distintos

**M4 — Personaje y múltiples mecanismos**
- [ ] Formato de personaje MVP: SVG con capas `part_*` y círculos `pivot_*` para las articulaciones, o JSON equivalente
- [ ] Chequeo de grados de libertad: comparar el número de mecanismos con los DOF libres del personaje (una pierna puede necesitar un solo mecanismo si tiene varillas pasivas) y guiar al usuario en la elección de puntos de actuación
- [ ] Personajes de prueba: la pierna de M2 y un personaje simple tipo Pushing Man (siluetas planas articuladas)
- [ ] Alternativa de formato a evaluar: una malla 3D por pieza (STL/OBJ) + JSON con los pivotes, trabajando en vista lateral ortogonal, como en el video (el personaje es un modelo 3D de piezas rígidas visto de perfil)
- [ ] Herramientas de la UI siguiendo la sesión del video: marcar piezas fijas (chasis), dibujar la caja de engranajes (rectángulo en vista lateral; en el video es una caja 3D con profundidad), elegir el punto marcador en una pieza, editar la spline objetivo y elegir candidato
- [ ] Control de desfase (phase shift): panel con un slider por mecanismo (el video muestra 4 sliders para las 4 patas), con escala de un ciclo y marcas; la pose se actualiza en vivo al arrastrar y se resalta la pieza controlada; el panel puede flotar o acoplarse arriba. Implementación propuesta: offset φ en la restricción de fase de cada mecanismo respecto al Input Driver
- [ ] Representación simplificada en la edición: engranajes como discos y eslabones como líneas; los dientes aparecen al pasar a la etapa del tren de engranajes (M5), como en el video
- [ ] Cada sub-ensamblaje tiene su plano de profundidad (z) según la pieza que acciona (p. ej. patas izquierda y derecha en planos distintos); se procesa un sub-ensamblaje a la vez y los anteriores quedan fijos
- [ ] Seleccionar punto de actuación, dibujar curva, instanciar mecanismo y unirlo al personaje con pines
- [ ] Varios mecanismos accionados por un único Input Driver
- [ ] Panel de propiedades, undo/redo, guardar/cargar proyecto
- [ ] Perfil de velocidad de la curva (preparación para engranajes no circulares)

**M5 — Tren de engranajes (paper §6.1)**
- [ ] El usuario elige pares de engranajes a conectar (paralelo / secuencial / combinado)
- [ ] Selección en la vista: los engranajes elegidos se resaltan (azul oscuro) y, al conectarlos, los dos extremos de la conexión se distinguen (naranja y azul), como en el video
- [ ] Diálogo con el número de engranajes intermedios (en el video: "Orientation (odd: same, even: reverse)"; impar = mismo sentido de giro, par = sentido contrario). El paper dice que el número se determina automáticamente: ofrecer el cálculo automático con opción de sobrescribirlo
- [ ] Engranajes en un eje común (conexión en paralelo) sobre un eje cilíndrico, y vista 3D del tren con dientes; si los ejes no son paralelos, engranajes cónicos (bevel) como indica el paper
- [ ] Repetir la conexión par por par hasta que todos los mecanismos queden unidos a un único eje de entrada; ese eje sobresale del conjunto para acoplar una manivela o un motor
- [ ] Engranajes en varias capas por pared: en los renders del cuadrúpedo y del T-Rex los engranajes dorados con dientes quedan apilados en distintas profundidades sobre ejes cilíndricos negros perpendiculares a la pared (visto de frente en el T-Rex se ven como discos finos apilados); el modelo `z` de M1 ya lo cubre, y la UI debe mostrar el eje (largo y profundidad) de cada engranaje
- [ ] Engranajes cónicos (bevel) cuando dos ejes no son paralelos: en el T-Rex se ve un engranaje cónico pequeño entre ejes perpendiculares, lo que confirma el caso que menciona el paper; en el MVP solo si el layout lo pide
- [ ] Optimización de parámetros (x, y, z, r, ω) con SLSQP (Ec. 9): engrane, no-intersección, radio mínimo
- [ ] **Paso extra respecto al paper:** pasar de radios continuos a *dientes enteros con módulo común*: r = m·z/2, distancia entre centros = m(z₁+z₂)/2; redondear z y resolver de nuevo las distancias
- [ ] Fijar los engranajes ya optimizados como restricciones duras

**M6 — Capas sin colisión (paper §6.2)**
- [ ] Colisiones por pares proyectando polígonos al plano de movimiento y barriendo el ciclo (shapely + STRtree)
- [ ] Asignación de capas en CP-SAT: A≠B para pares en colisión; "C no entre A y B" con booleanos; empezar con m = 4 y subir hasta ser factible
- [ ] Objetivo secundario: minimizar la longitud de pines (varias soluciones y elegir la mejor)
- [ ] Espesor de capa = mayor espesor de sus piezas
- [ ] Vista 3D de solo lectura (con cámara orbital) para revisar planos y capas

**M7 — Exportación e impresión (paper §6.4)**
- [ ] Engranajes evolventes, eslabones, ejes y pines desde polígonos 2D extruidos
- [ ] Opción de eslabones como varillas externas (filamento, varilla metálica o brochetas de bambú) en lugar de piezas impresas: exportar solo largo y diámetro, y agujeros del diámetro correcto
- [ ] Personaje como siluetas planas con orificios para los pines, como en los prototipos del video
- [ ] El desfase de cada mecanismo se materializa en el ángulo de montaje de sus manivelas sobre los engranajes (cada una rotada según su relación de velocidades); es una propuesta a validar con una impresión
- [ ] Parámetros FDM configurables: holgura (≈ 0,3 mm por defecto), espesor mínimo de pared, altura de capa
- [ ] Un STL por pieza + placa de impresión ordenada; opción "una sola pieza"
- [ ] Exportar también el eje de entrada, con largo suficiente para acoplar la manivela o el motor
- [ ] Caja del mecanismo como pila de placas planas con agujeros (una placa por capa), como en los prototipos del cuadrúpedo, con los ejes atravesándolas; las patas inferiores son piezas de apoyo/soporte
- [ ] Acople del eje de entrada a un taladro o manivela: en el prototipo el motor es un taladro de mano (Black+Decker) acoplado al eje que sobresale; exportar el eje con largo y diámetro configurables (¿extremo hexagonal para el mandril? a decidir)
- [ ] Prueba real con cuadrilátero + manivela y ajuste de holguras

**M8 — Soportes y pulido (paper §6.3)**
- [ ] Voxelizar el volumen barrido (numpy) y marcar vóxeles libres
- [ ] El usuario elige planos de pared; conexión automática de ejes alcanzables; feedback de ejes sin soporte
- [ ] Paredes de soporte en L: en los renders del T-Rex y del cuadrúpedo los engranajes y ejes están entre dos paredes translúcidas que forman una esquina (dos planos perpendiculares); el modelo de paredes debe admitir varios planos, no solo uno
- [ ] Empaquetado (pip primero, Flatpak después)

**Post-MVP**
- [ ] Engranajes no circulares para timing (§4.3: RBF thin-plate, r₁ + r₂ = a). En los renders de GALOP algunos engranajes parecen ovalados o triangulares redondeados; sería coherente con usarlos para el timing del galope, pero no lo puedo confirmar por las capturas
- [ ] Mecanismos no planares y movimiento 3D
- [ ] Análisis estructural y optimización de torque

## Observaciones del video (capturas 0:05–0:59, ~1:32, 1:50–2:24 y tren de engranajes del T-Rex)
- **Pushing Man (render y prototipo):** marco de dos placas con las piezas en capas; engranajes en la parte baja; el personaje es una silueta plana articulada con pines; los eslabones parecen **varillas finas de otro material**, no piezas impresas; hay patas/clips negros impresos como soporte; el motor es un engranaje girado a mano. El material verdoso parece resina.
- **Mecanismo base:** dos engranajes con una varilla fijada a cada uno; las varillas se unen en un pin y una se prolonga hasta el punto marcador rojo (acoplador con extensión). En las capturas los tamaños de los engranajes parecen 1:1 y 2:1, coherente con las relaciones ±1, ±2 del paper (§7.1).
- **Exploración del espacio de parámetros:** se ve cómo se barren mecanismos y se acumulan miniaturas de las curvas ya muestreadas.
- **Consulta a la base:** el usuario edita una **curva cerrada con puntos de control arrastrables** (spline) y, mientras la mueve, aparece el mecanismo y la curva que traza (en rojo sobre la objetivo en negro). Se hacen varias consultas a la vez. Por eso la consulta tiene que ser rápida (M3).
- **Optimización continua:** muestra la curva objetivo (rojo) y la actual (blanco) convergiendo, y afirma que funciona incluso con inicializaciones relativamente malas.
- **Curvas en puntos arbitrarios del personaje (~1:32):** una pierna (muslo con pivote fijo, rodilla, tobillo y pie) con una varilla pasiva y un único mecanismo de dos engranajes unido a la zona de la rodilla. Hay dos marcadores rojos: uno en el punto de unión del mecanismo y otro en el pie, donde está la curva objetivo (elipse roja). El pie arranca trazando una curva en forma de ocho y, a medida que se optimiza, la curva blanca converge a la elipse. Los engranajes quedan superpuestos con las siluetas del personaje, lo que se resuelve después con la asignación de capas.
- **Sesión interactiva (1:50–1:52 y siguientes):** ventana de una aplicación de investigación ("CDMASketch") con grilla y un cuadrúpedo 3D articulado visto de perfil. El flujo que se alcanza a ver (los colores son interpretación mía): (1) las piezas del cuerpo (cabeza, torso, cadera) pasan de naranja, que parece ser el resaltado, a azul, que parece indicar piezas fijas; (2) se dibuja un rectángulo naranja debajo del personaje, la caja de engranajes; (3) se elige una pieza de la pata (naranja) y un punto marcador (rosa) en el pie; (4) se dibuja una elipse como spline con puntos de control; (5) aparecen varias curvas candidatas en colores y rectángulos de colores, que parecen ser las cajas delimitadoras de cada candidato, y un diálogo "Select Curve" pide elegir 0: rojo, 1: verde o 2: azul; (6) un diálogo "Phase Shift Control Box" con un slider ("The UI controls 1 sub-assembly") ajusta el desfase del sub-ensamblaje. Antes de crear el primero, el diálogo dice que no hay sub-ensamblajes que controlar.
- **Continuación de la sesión (1:57–2:04 y siguientes):** al elegir el candidato, el mecanismo aparece dentro de la caja de engranajes y se conecta por una varilla (línea amarilla) al marcador del pie. En esta vista los engranajes son discos azules, uno grande y uno pequeño, que parecen marcadores de posición: no se ve que su tamaño sea el de los engranajes finales y no tienen dientes. Después se repite el proceso con la otra pata (pieza naranja, marcador rosa): nueva spline, nuevos candidatos con rectángulos de colores y el diálogo "Select Curve"; el diálogo de desfase pasa a "The UI controls 2 sub-assemblies" con dos sliders. Una vista en perspectiva muestra que la caja de engranajes es una caja 3D y que los dos mecanismos parecen estar en planos paralelos distintos, uno por cada lado del cuadrúpedo.
- **Manipulación del desfase (2:15–2:24, sección "Phase Shift Manipulation"):** con las patas ya accionadas, el diálogo pasa a "The UI controls 4 sub-assemblies" y muestra cuatro sliders horizontales con marcas, todos al centro al principio. El diálogo se mueve y queda acoplado en la parte superior de la ventana. Al arrastrar cada slider cambia la fase de un mecanismo y la pose de la pata correspondiente se actualiza en vivo; la pieza controlada se resalta en naranja (interpretación mía). Así se configura la marcha. En la vista lateral solo se distinguen dos mecanismos; los otros dos parecen quedar superpuestos.
- **Tren de engranajes con el T-Rex (sin marca de tiempo visible, posterior a 2:24):** otro personaje 3D (un T-Rex) con varillas grises que bajan desde las patas hasta engranajes azules con dientes, en una vista en perspectiva con ejes de coordenadas. Se ven tres engranajes de mecanismo; el usuario selecciona dos (se resaltan en azul oscuro) y aparece un diálogo, con el título cortado (parece "Number of gears in between"), que dice "Orientation (odd: same, even: reverse)" y tiene OK/Cancel: el número de engranajes intermedios decide si el sentido de giro se mantiene (impar) o se invierte (par). Tras aceptar se generan engranajes intermedios (blancos, con dientes radiales que podrían ser cónicos; no puedo confirmarlo) y un eje cilíndrico; los dos engranajes extremos quedan resaltados en naranja y azul y, desde otro ángulo, parecen montados en un mismo eje horizontal con los intermedios entre ellos. Debajo del cuerpo hay una estructura de varillas y collares que podría ser el soporte, sin confirmar.
- **Más del tren de engranajes del T-Rex (posterior a 2:24):** el diálogo "Number of gears in between" vuelve a aparecer, así que el proceso se repite par por par. Los engranajes conectados quedan repartidos en varios planos; los intermedios (blancos) rodean un eje cilíndrico largo que sobresale del conjunto y que parece ser el eje de entrada. Hay engranajes extremos azules y uno naranja, que parece ser el seleccionado en cada paso. Las miniaturas oscuras de la esquina derecha no son parte de la aplicación: son la vista previa de la barra de tiempo de YouTube; por los números truncados ("3:4", "3:5", "3:3") corresponden a ~3:30–3:50, donde se ven un prototipo físico y lo que parece un tigre dientes de sable.
- **Cuadrúpedo "EMA" (WALK y GALOP), render y prototipos (~3:40–4:10, estimación mía por la barra de progreso):** los renders llevan el rótulo "EMA WALK" y "EMA GALOP" (no sé qué significa "EMA"). Es un cuadrúpedo gris translúcido con cabeza, torso y dos conjuntos de patas, con varillas negras que bajan desde las patas hasta engranajes dorados con dientes. En WALK los engranajes están sobre una sola placa gris, con varios tamaños (uno grande abajo a la izquierda, uno grande abajo a la derecha, otros medianos y pequeños arriba) y algunos más oscuros, parecen estar en una capa de atrás; las varillas llegan a pequeños pines sobre los engranajes. En GALOP los engranajes se reparten sobre dos paredes translúcidas en esquina, con muchos más engranajes en varios planos y ejes cilíndricos negros que atraviesan los engranajes.
- **Prototipos físicos del cuadrúpedo:** uno oscuro (WALK) con una caja de placas con agujeros y pernos, patas y cuerpo en piezas separadas con varillas largas, un engranaje que asoma por el lado derecho de la caja y apoyos en la base; y otro translúcido (GALOP), con la caja hecha de placas apiladas con agujeros y engranajes visibles por dentro. En el translúcido el motor es un taladro naranja (se lee "BLACK+DECKER") acoplado a un lado de la caja, y las patas se ven movidas con desenfoque de movimiento en una de las capturas. No puedo asegurar el material de cada prototipo.
- **T-Rex (render, rótulo "TREX"):** cuerpo metálico con cabeza, patas y cola, varillas negras que bajan hasta engranajes dorados repartidos en varias capas entre dos paredes en L. Hay un engranaje cónico pequeño, a la derecha, entre ejes perpendiculares. Una vista frontal muestra los ejes negros horizontales saliendo de la pared con varios engranajes apilados en cada uno (se ven de canto, como discos finos) y un engranaje cónico pequeño cerca del centro.
- **Lectura general:** las paredes translúcidas en esquina parecen ser la etapa de soportes (§6.3) ya resuelta: los ejes quedan sostenidos por planos de pared y no flotan. Es una interpretación mía.
- **Corrección a una nota anterior:** las miniaturas de la barra de tiempo que había atribuido a "un prototipo físico y lo que parece un tigre dientes de sable" correspondían, al menos en parte, a estos prototipos del cuadrúpedo; el tigre no aparece en estas capturas.

## Riesgos y mitigaciones
| Riesgo | Mitigación |
|---|---|
| Mínimos locales en la optimización | Inicial desde la base de movimientos (M3); reinicios aleatorios en M2 |
| Saltos de rama del cuadrilátero | Continuación por fase + rechazo de ciclos no continuos (M1) |
| Layout de engranajes combinatorio | Semiautomático como en el paper: el usuario elige los pares |
| Radios continuos no son fabricables | Redondeo a dientes enteros con módulo común (M5) |
| Holguras de impresión | Parámetros configurables + prueba temprana (M7) |
| Rendimiento de scipy en la UI | Hilos/procesos aparte, cancelación y barra de progreso |
| Personajes con partes en distintos planos de profundidad (patas izquierda y derecha) | Plano z por sub-ensamblaje; la simulación sigue siendo 2D por plano y la asignación de capas (M6) usa esos planos |
| Alcance | MVP definido abajo; todo lo demás es post-MVP |
| Licencia | El paper no publica código; si pensás distribuir o vender la app, conviene revisar licencias o patentes de la técnica |

## Definición de MVP
Personaje 2D → dibujar 1–3 curvas → mecanismo optimizado por curva → un único motor con tren de engranajes → capas sin colisión → STL por pieza.
Fuera del MVP: entrenamiento de la métrica, engranajes no circulares, movimiento 3D.

## Latest Blockers / Discoveries
- M0: el código GTK (`ui/`) no se pudo ejecutar en el entorno donde se escribió (GTK4 no disponible); solo está verificada la lógica sin GTK. Si algo falla al abrir la ventana, empezar por `ui/canvas.py`.
- El video de YouTube (2013) es el material complementario oficial del paper; solo se pudo leer su descripción, que coincide con el abstract. Los ejemplos hay que verlos en el video.
- El paper no publica código ni la biblioteca exacta de mecanismos: hay que diseñar los tipos propios a partir de la Fig. 4.
- Decisiones tomadas: 2D con 3 DOF por pieza; hitos como cortes verticales; núcleo sin GTK; dientes enteros en M5.
- Del video: la entrada de curvas es una spline cerrada editable (no trazo libre); los eslabones del prototipo parecen varillas externas, relevante para impresión FDM.
- Las capturas disponibles cubren el primer minuto, la escena de la pierna (~1:32) y la sesión interactiva con el cuadrúpedo (1:50–2:24) y la edición del tren de engranajes con el T-Rex; falta ver el final del tren, las capas, los soportes y los resultados finales.
- La curva de un punto como el pie depende de la cinemática del personaje (varillas pasivas incluidas); de ahí el chequeo de DOF en M4.
- Decisión tomada (v2.4): se descarta importar archivos de animación (idea de v2.1); la entrada es un personaje articulado propio más curvas dibujadas como splines.
- Decisión tomada (v2.6): el modelo de datos incluye un plano z por sub-ensamblaje desde el inicio, aunque la simulación sea 2D.
- Del tramo final: las paredes de soporte pueden ser varios planos (esquina en L), el eje de entrada se acopla a un taladro en el prototipo y algunos engranajes del galope podrían ser no circulares (sin confirmar). Falta ver cómo se genera cada paso intermedio (capas y soportes).
- Decisiones abiertas: formato de personaje (SVG con capas, JSON o mallas 3D por pieza); `least_squares` vs Newton propio; derivadas a mano vs JAX.
