# CC5513: Parametrización de superficies

Tarea del curso CC5513 (Procesamiento Geométrico y Análisis de Formas): parametrización
de mallas de triángulos con topología de disco.

Métodos: Tutte (`uniform`), `harmonic` y `meanvalue` con borde fijo (círculo, cuadrado o
estrella), y `lscm` con borde libre. Se implementaron los pesos cotangentes y mean value,
los bordes poligonales y los dos bonus.

## Contenido

| ruta | contenido |
|---|---|
| `Tarea2_SGP.pdf` | enunciado |
| `tarea3_param/` | código, pruebas y mallas ([README](tarea3_param/README.md)) |
| `tarea3_param/experimentos/` | scripts de figuras y tablas, y sus resultados ([README](tarea3_param/experimentos/README.md)) |
| `informe/` | informe (`informe.pdf`) y su fuente LaTeX |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tarea3_param/requirements.txt
```

## Uso

Todos los comandos se ejecutan desde `tarea3_param/`:

```bash
python tests.py                                       # pruebas automáticas
python run_tarea3.py --mesh data/nefertiti.off        # visor interactivo
python run_tarea3.py --mesh data/bunny.off --batch    # tabla de métricas
```

Sin OpenGL, usa `--batch` o `--shot`. Las demás opciones están en
[`tarea3_param/README.md`](tarea3_param/README.md).
