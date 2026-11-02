# Experimentos del informe

Scripts que generan todas las figuras y tablas del informe. Usan los módulos
del código base (`param`, `solvers`, `distortion`, ...) y Matplotlib, además de
NumPy, SciPy y Polyscope.

Se ejecutan desde la carpeta `tarea3_param`:

```bash
pip install matplotlib                     # además de requirements.txt
python experimentos/p1_pesos.py            # Parte 1: pesos negativos, flips, reproducción lineal
python experimentos/p2_bordes.py           # Parte 2: tablas --batch, cuadrado, estrella, arc vs uniform
python experimentos/p3_lscm.py             # Parte 3: borde invertido, valores propios de Q, pins (~1 min)
python experimentos/p4_distorsion.py       # Parte 4: sistema local y experimentos controlados
python experimentos/p5_comparacion.py      # Parte 5: imágenes, orden de métodos, área vs distancia, tiempos
python experimentos/b1_cuadrado.py         # Bonus B1: cuadrado abombado (degen = 0)
python experimentos/b2_foto.py             # Bonus B2 y pregunta 5.3: fotografía como textura
python experimentos/tablas_latex.py        # tablas --batch en LaTeX para el anexo (después de p2)
```

Todo se guarda en `experimentos/resultados/` (PNG y CSV). Las imágenes 3D se
hacen con Polyscope (`ParamViewer` del código base), así que se abre una
ventana por un momento; sin pantalla, Polyscope intenta un backend sin ventana.

| archivo | contenido |
|---|---|
| `comun.py` | rutas, carga de mallas, alineamiento de Procrustes, capturas con Polyscope |
| `p1_pesos.py` ... `p5_comparacion.py` | un script por parte del enunciado |
| `b1_cuadrado.py`, `b2_foto.py` | bonus |
| `tablas_latex.py` | convierte `batch_<malla>.csv` en tablas LaTeX |

El borde del bonus B1 está en `boundary.py` (`bulged_square_points`) y se usa
como `parameterize(V, F, metodo, shape="bulged_square")`.
