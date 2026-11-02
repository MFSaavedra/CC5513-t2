# -*- coding: utf-8 -*-
"""
Funciones compartidas por los scripts de experimentos.

Los scripts se ejecutan desde la carpeta tarea3_param, por ejemplo
    python experimentos/p1_pesos.py
y guardan figuras y tablas en experimentos/resultados/.

Al importar este módulo se agrega la carpeta del código base a sys.path,
así que los scripts pueden importar param, solvers, distortion, etc.
"""
import csv
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)                  # carpeta tarea3_param
sys.path.insert(0, BASE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")                         # solo archivos, sin ventanas
import matplotlib.pyplot as plt  # noqa: E402

from distortion import distortion_report  # noqa: E402
from mesh_io import read_off  # noqa: E402
from mesh_utils import boundary_loop  # noqa: E402

RESULTADOS = os.path.join(AQUI, "resultados")
os.makedirs(RESULTADOS, exist_ok=True)

MALLAS = ["nefertiti", "bunny", "lucy"]

# Dirección desde la que se mira cada malla en las figuras 3D
VISTAS = {"nefertiti": (0.25, 0.0, 1.0), "bunny": (-1.0, 0.15, 0.55),
          "lucy": (0.15, 0.1, 1.0)}

# Las figuras se generan del tamaño con que aparecen en el informe (letra de 7 pt)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


def ruta(nombre_archivo):
    """Ruta de un archivo dentro de experimentos/resultados/."""
    return os.path.join(RESULTADOS, nombre_archivo)


def cargar(nombre):
    """Lee data/<nombre>.off y devuelve (V, F, loop)."""
    V, F = read_off(os.path.join(BASE, "data", nombre + ".off"))
    return V, F, boundary_loop(F)


def guardar_tabla(nombre_archivo, encabezado, filas):
    """Guarda una tabla como CSV y la imprime."""
    with open(ruta(nombre_archivo), "w", newline="") as f:
        escritor = csv.writer(f)          # pone comillas si un campo tiene comas
        escritor.writerow(encabezado)
        escritor.writerows(filas)
    print(" | ".join(encabezado))
    for fila in filas:
        print(" | ".join(str(x) for x in fila))


# ---------------------------------------------------------------------------
# Alineamiento de Procrustes (pregunta 3.3; también se usa en el bonus B2)
# ---------------------------------------------------------------------------
def semejanza(X, Y):
    """Mejor semejanza que lleva los puntos Y (k, 2) a los puntos X (k, 2).

    Busca la escala s, la rotación R (sin reflexión) y la traslación c que
    minimizan ||s Y R + c - X|| (Procrustes ortogonal: SVD de Y0^T X0).
    """
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    U, S, Vt = np.linalg.svd(Y0.T @ X0)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])   # evita reflexiones
    R = U @ D @ Vt
    s = np.sum(S * np.diag(D)) / np.sum(Y0 ** 2)
    c = X.mean(axis=0) - s * Y.mean(axis=0) @ R
    return s, R, c


def procrustes(X, Y):
    """Alinea Y con X usando la mejor semejanza (traslación, rotación y escala).

    Devuelve (Y_alineado, error), donde
        error = ||Y_alineado - X|| / ||X - media(X)||
    vale 0 solo si Y es exactamente una semejanza de X.
    """
    s, R, c = semejanza(X, Y)
    Y_alineado = s * Y @ R + c
    error = np.linalg.norm(Y_alineado - X) / np.linalg.norm(X - X.mean(axis=0))
    return Y_alineado, error


# ---------------------------------------------------------------------------
# Imágenes con Polyscope (se reutiliza ParamViewer del código base)
# ---------------------------------------------------------------------------
_polyscope_listo = False


def polyscope():
    """Inicializa Polyscope una sola vez y lo devuelve."""
    global _polyscope_listo
    import polyscope as ps
    if not _polyscope_listo:
        try:
            ps.init()
        except Exception:
            ps.set_allow_headless_backends(True)
            ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_window_size(1000, 1000)
        ps.set_SSAA_factor(2)
        ps.set_up_dir("y_up")
        _polyscope_listo = True
    return ps


def visor(V, F, UV, method="harmonic", color="checker", checker=0.05, pins=None):
    """ParamViewer con una parametrización UV ya calculada (no la recalcula)."""
    from viewer import ParamViewer
    v = ParamViewer(V, F, method=method, color=color, checker=checker, pins=pins)
    v.UV = UV
    v.per_face, v.summary = distortion_report(V, F, UV)
    return v


def invertidas_y_degeneradas(V, F, UV):
    """Índices de las caras invertidas y de las degeneradas (distortion_report).

    distortion_report marca ambas en per_face["flipped"]; las degeneradas
    además tienen conformal = 1e12.
    """
    cara, _ = distortion_report(V, F, UV)
    degeneradas = cara["conformal"] >= 1e12
    invertidas = cara["flipped"] & ~degeneradas
    return np.where(invertidas)[0], np.where(degeneradas)[0]


def recortar(archivo, margen=6):
    """Quita el fondo blanco alrededor de una imagen PNG."""
    img = plt.imread(archivo)[..., :3]
    hay = (img < 0.97).any(axis=2)
    filas = np.where(hay.any(axis=1))[0]
    cols = np.where(hay.any(axis=0))[0]
    if len(filas) == 0:
        return
    r0, r1 = max(filas[0] - margen, 0), filas[-1] + margen + 1
    c0, c1 = max(cols[0] - margen, 0), cols[-1] + margen + 1
    plt.imsave(archivo, img[r0:r1, c0:c1])


def registrar(v, **opciones):
    """Registra en Polyscope las mallas de un ParamViewer (ver v.register).

    Se registra dos veces a propósito: Polyscope recuerda, por nombre, qué
    cantidad estaba activa en la imagen anterior y al crearla de nuevo puede
    desactivar la que se pide ahora (la imagen sale sin color). En el segundo
    registro ese estado ya está limpio.
    """
    ps = polyscope()
    for _ in range(2):
        ps.remove_all_structures()
        v.register(**opciones)
    return ps


def foto_3d(v, archivo, direccion, zoom=1.0, objetivo=None):
    """Guarda la superficie de un ParamViewer vista desde `direccion`.

    objetivo: punto al que mira la cámara (por defecto, el centro de la malla).
    zoom > 1 acerca la cámara.
    """
    ps = registrar(v, show_surface=True, show_flat=False)
    ps.set_view_projection_mode("perspective")
    if objetivo is None:
        objetivo = (v.V.max(axis=0) + v.V.min(axis=0)) / 2
    d = np.asarray(direccion, dtype=float)
    d /= np.linalg.norm(d)
    ps.look_at(objetivo + 2.0 * v.size / zoom * d, objetivo)
    ps.screenshot(archivo, transparent_bg=False)
    recortar(archivo)


def foto_uv(v, archivo):
    """Guarda la malla aplanada (u, v) de un ParamViewer, vista desde arriba."""
    ps = registrar(v, show_surface=False, show_flat=True)
    ps.set_view_projection_mode("orthographic")
    c = v.offset + np.array([v.size / 2, v.size / 2, 0.0])
    ps.look_at(c + np.array([0.0, 0.0, 2.2 * v.size]), c)
    ps.screenshot(archivo, transparent_bg=False)
    recortar(archivo)


# ---------------------------------------------------------------------------
# Dibujos 2D con Matplotlib
# ---------------------------------------------------------------------------
def dibujar_malla_2d(ax, P, F, rojas=(), lw=0.3):
    """Dibuja las aristas de una malla plana P (n, 2) y rellena en rojo las caras `rojas`."""
    ax.triplot(P[:, 0], P[:, 1], F, color="0.35", lw=lw)
    for t in rojas:
        ax.fill(P[F[t], 0], P[F[t], 1], color="tab:red", alpha=0.8, lw=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def encuadrar(ax, puntos, margen=1.2):
    """Ajusta los ejes a un cuadrado que contiene los puntos (k, 2)."""
    c = (puntos.max(axis=0) + puntos.min(axis=0)) / 2
    r = margen * np.abs(puntos - c).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)


def tablero(u, v, tam=0.1):
    """Textura de tablero: 0 o 1 según la casilla en que cae (u, v)."""
    return (np.floor(u / tam) + np.floor(v / tam)) % 2
