# -*- coding: utf-8 -*-
"""
Parte 2: bordes poligonales.

    python experimentos/p2_bordes.py

Calcula la tabla --batch de las tres mallas (resultados/batch_<malla>.csv,
también se usa en la Parte 5) y responde:
2.1  dónde están los triángulos degenerados del borde cuadrado en Bunny;
2.2  dónde se concentran los triángulos invertidos con la estrella;
2.3  diferencia entre los espaciados arc y uniform.
"""
import numpy as np

# comun va primero: agrega la carpeta del código base a sys.path
from comun import (MALLAS, VISTAS, cargar, dibujar_malla_2d, encuadrar, foto_3d,
                   guardar_tabla, invertidas_y_degeneradas, plt, ruta, visor)
from boundary import star_corners
from distortion import distortion_report
from mesh_utils import face_areas
from param import parameterize
from run_tarea3 import run_batch


def tablas_batch():
    for nombre in MALLAS:
        V, F, _ = cargar(nombre)
        run_batch(V, F, nombre, ruta("batch_%s.csv" % nombre))


def pregunta_2_1():
    print("\n== 2.1 Triangulos degenerados con el borde cuadrado (Bunny) ==")
    V, F, loop = cargar("bunny")
    posicion = {int(v): k for k, v in enumerate(loop)}       # vértice -> posición en el borde
    filas = []
    for metodo in ["uniform", "harmonic", "meanvalue"]:
        for espaciado in ["arc", "uniform"]:
            UV, _ = parameterize(V, F, metodo, "square", espaciado, loop=loop)
            _, degeneradas = invertidas_y_degeneradas(V, F, UV)
            pos = [sorted(posicion.get(int(v), -1) for v in F[t]) for t in degeneradas]
            filas.append([metodo, espaciado, len(degeneradas), " ".join(str(p) for p in pos)])
    guardar_tabla("p2_degenerados_bunny.csv",
                  ["metodo", "espaciado", "degen", "posiciones en el borde"], filas)
    # Las 4 caras tienen sus 3 vértices consecutivos en el borde ("orejas"):
    # la arista que une el primero con el tercero es interior.
    UV, _ = parameterize(V, F, "meanvalue", "square", "arc", loop=loop)
    UVb, _ = parameterize(V, F, "meanvalue", "bulged_square", "arc", loop=loop)
    _, degeneradas = invertidas_y_degeneradas(V, F, UV)

    # Figura: (a) dominio completo con las orejas marcadas; (b, c) zoom de dos
    # orejas vecinas con el cuadrado y con el cuadrado abombado del bonus B1
    fig, axs = plt.subplots(1, 3, figsize=(2.9, 1.3))
    dibujar_malla_2d(axs[0], UV, F, lw=0.05)
    for t in degeneradas:
        x, y = UV[F[t]].mean(axis=0)
        axs[0].plot(x, y, "o", mfc="none", mec="tab:red", ms=5, mew=1)
    axs[0].set_title("square: degen = %d" % len(degeneradas))
    # las dos orejas vecinas (posiciones 58-60 y 60-62) comparten el vértice 60 del borde
    orejas = [t for t in degeneradas if 60 in [posicion[int(v)] for v in F[t]]]
    for ax, P, titulo in ((axs[1], UV, "zoom"), (axs[2], UVb, "bulged (B1)")):
        dibujar_malla_2d(ax, P, F, lw=0.3)
        for t in orejas:
            tri = P[F[t][[0, 1, 2, 0]]]
            ax.plot(tri[:, 0], tri[:, 1], color="tab:red", lw=1)
            ax.plot(tri[:3, 0], tri[:3, 1], "o", color="tab:red", ms=2)
        encuadrar(ax, P[F[orejas]].reshape(-1, 2), margen=1.15)
        ax.set_title(titulo)
    fig.savefig(ruta("p2_bunny_orejas_uv.png"))
    plt.close(fig)


def pregunta_2_2():
    print("\n== 2.2 Triangulos invertidos con la estrella (espaciado arc) ==")
    estrella = star_corners()
    internas, externas = estrella[1::2], estrella[0::2]   # vértices cóncavos y convexos
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        for metodo in ["uniform", "harmonic", "meanvalue"]:
            UV, _ = parameterize(V, F, metodo, "star", "arc", loop=loop)
            invertidas, _ = invertidas_y_degeneradas(V, F, UV)
            centro = UV[F[invertidas]].mean(axis=1)
            d_int = np.linalg.norm(centro[:, None] - internas[None], axis=2).min(axis=1)
            d_ext = np.linalg.norm(centro[:, None] - externas[None], axis=2).min(axis=1)
            filas.append([nombre, metodo, len(invertidas),
                          "%.0f%%" % (100 * np.mean(d_int < 0.2)),
                          "%.0f%%" % (100 * np.mean(d_ext < 0.2)),
                          "%.3f" % np.median(d_int)])
    guardar_tabla("p2_estrella_flips.csv",
                  ["malla", "metodo", "flips", "% a <0.2 de vertice concavo",
                   "% a <0.2 de vertice convexo", "mediana dist. concavo"], filas)

    # Figura: dominio (u, v) de Bunny con las caras invertidas en rojo, zoom en
    # el vértice cóncavo con más caras invertidas y la superficie (modo flips)
    V, F, loop = cargar("bunny")
    UV, _ = parameterize(V, F, "meanvalue", "star", "arc", loop=loop)
    invertidas, _ = invertidas_y_degeneradas(V, F, UV)
    centro = UV[F[invertidas]].mean(axis=1)
    cercano = np.linalg.norm(centro[:, None] - internas[None], axis=2).argmin(axis=1)
    k = np.bincount(cercano).argmax()               # vértice cóncavo con más invertidas
    fig, axs = plt.subplots(1, 2, figsize=(2.3, 1.3))
    for ax in axs:
        dibujar_malla_2d(ax, UV, F, rojas=invertidas, lw=0.05 if ax is axs[0] else 0.3)
        ax.plot(*np.vstack([estrella, estrella[:1]]).T, color="tab:red", lw=0.4)
    axs[0].plot(*internas[k], "s", mfc="none", mec="k", ms=6, mew=0.8)
    encuadrar(axs[1], centro[cercano == k], margen=1.6)
    axs[0].set_title("star: %d flips" % len(invertidas))
    axs[1].set_title("zoom")
    fig.savefig(ruta("p2_bunny_estrella_uv.png"))
    plt.close(fig)
    v = visor(V, F, UV, method="meanvalue", color="flips")
    foto_3d(v, ruta("p2_bunny_estrella_3d.png"), (0.3, -1.0, 0.4))


def pregunta_2_3():
    print("\n== 2.3 Espaciado arc vs uniform (borde circular) ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        largo = np.linalg.norm(V[np.roll(loop, -1)] - V[loop], axis=1)  # aristas del borde 3D
        en_borde = np.isin(F, loop).any(axis=1)     # caras que tocan el borde
        A = face_areas(V, F)
        for metodo in ["harmonic", "meanvalue"]:
            fila = [nombre, len(loop), "%.2f" % (largo.std() / largo.mean()),
                    "%.1f" % (largo.max() / largo.min()), metodo]
            for espaciado in ["arc", "uniform"]:
                UV, _ = parameterize(V, F, metodo, "circle", espaciado, loop=loop)
                cara, resumen = distortion_report(V, F, UV)
                area_borde = np.sum(A[en_borde] * np.abs(cara["area"][en_borde])) / A[en_borde].sum()
                fila += ["%.3f" % resumen["conf_med"], "%.3f" % resumen["area_mean"],
                         "%.3f" % np.median(cara["conformal"][en_borde]), "%.3f" % area_borde]
            filas.append(fila)
    guardar_tabla("p2_arc_vs_uniform.csv",
                  ["malla", "nb", "cv largo borde", "max/min", "metodo",
                   "conf_med arc", "area_mean arc", "conf_borde arc", "area_borde arc",
                   "conf_med unif", "area_mean unif", "conf_borde unif", "area_borde unif"],
                  filas)

    # Figura: distorsión conforme de Nefertiti (harmonic) con los dos espaciados
    V, F, loop = cargar("nefertiti")
    for espaciado in ["arc", "uniform"]:
        UV, _ = parameterize(V, F, "harmonic", "circle", espaciado, loop=loop)
        v = visor(V, F, UV, method="harmonic", color="conformal")
        foto_3d(v, ruta("p2_nefertiti_%s_conf.png" % espaciado), VISTAS["nefertiti"])


if __name__ == "__main__":
    tablas_batch()
    pregunta_2_1()
    pregunta_2_2()
    pregunta_2_3()
