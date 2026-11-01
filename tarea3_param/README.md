# CC5513 – Tarea 3: Parametrización de superficies

Código base para la tarea. Todo está escrito con NumPy/SciPy y se visualiza con
Polyscope; **no se necesita OpenMesh**.

## Instalación

```bash
conda activate procgeom          # o el entorno que uses
pip install -r requirements.txt  # numpy, scipy, polyscope
```

## Uso rápido

```bash
python run_tarea3.py --mesh data/nefertiti.off          # visor interactivo
python run_tarea3.py --mesh data/bunny.off --batch      # tabla de métricas
python tests.py                                         # pruebas automáticas
```

Con el código tal como se entrega funcionan el método `uniform` con borde
`circle` y todas las medidas de distorsión; el resto muestra `PENDIENTE` hasta
que implementes las Partes 1 y 2 (LSCM usa los pesos cotangentes de la Parte 1a). El panel "Tarea 3" del visor permite cambiar método, forma
del borde, espaciado y qué se colorea (tablero, distorsión conforme, de área o
triángulos invertidos). Con el método `lscm` aparecen controles para elegir los
*pins* y se marcan en azul.

Otras opciones de `run_tarea3.py` (ver `--help`):

| opción | descripción |
|---|---|
| `--method {uniform,harmonic,meanvalue,lscm}` | método de parametrización |
| `--boundary {circle,square,star}` | forma del borde (métodos de borde fijo) |
| `--spacing {arc,uniform}` | distribución de los vértices del borde |
| `--color {checker,conformal,area,flips}` | qué se dibuja sobre la malla |
| `--checker 0.05` | tamaño del tablero |
| `--pins A B` | LSCM: posiciones en el borde de los dos vértices fijos |
| `--batch --csv tabla.csv` | métricas de todas las combinaciones |
| `--shot fig` | guarda `fig_3d.png` y `fig_uv.png` sin abrir ventana |
| `--save-uv plano.off` | guarda la malla (u, v, 0) |

## Archivos

| archivo | contenido | ¿modificar? |
|---|---|---|
| `mesh_io.py` | lectura/escritura OFF | no |
| `mesh_utils.py` | borde, ángulos, áreas, coordenadas locales, malla plana de prueba | no |
| `laplacians.py` | pesos uniformes (dado), **cotangentes y mean value** | **Parte 1** |
| `boundary.py` | espaciado del borde y círculo (dados), **polígonos** | **Parte 2** |
| `solvers.py` | borde fijo, LSCM y energía conforme (dados) | no; se analiza en la **Parte 3** |
| `distortion.py` | Jacobianos y medidas de distorsión (dados) | no; se analiza en la **Parte 4** |
| `param.py` | une todo: `parameterize(V, F, method, shape, spacing)` | no |
| `viewer.py` | visor Polyscope | no |
| `run_tarea3.py` | programa principal | no |
| `tests.py` | pruebas automáticas | no (puedes agregar) |

Las funciones que debes completar (Partes 1 y 2) lanzan `NotImplementedError`. Lee el
docstring de cada una: explica la fórmula, el formato de entrada/salida y da
pistas de implementación.

## Notas para Windows

* Si `pip install polyscope` falla, prueba `conda install -c conda-forge polyscope`.
* Los textos del visor no usan tildes a propósito (versiones antiguas de
  Polyscope no dibujan caracteres fuera de ASCII).
* Si el visor no abre (sin OpenGL), puedes trabajar con `--batch` y `--shot`.
