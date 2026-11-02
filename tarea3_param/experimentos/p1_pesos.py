# -*- coding: utf-8 -*-
"""
Parte 1: pesos armónicos (cotangentes) y de valor medio.

    python experimentos/p1_pesos.py

1.1  Cuenta los pesos cotangentes negativos de cada malla y verifica que
     coinciden con las aristas donde alpha + beta > pi. Busca triángulos
     invertidos con pesos armónicos y borde circular, y dibuja el de Lucy.
1.2  Verifica que los pesos de valor medio son positivos.
1.3  Error de reproducción lineal en la malla plana de prueba y distorsión
     conforme en las zonas casi planas de cada malla.
"""
import numpy as np

# comun va primero: agrega la carpeta del código base a sys.path
from comun import (MALLAS, cargar, dibujar_malla_2d, encuadrar, guardar_tabla,
                   invertidas_y_degeneradas, plt, ruta)
from distortion import distortion_report
from laplacians import (WEIGHTS, cotangent_weights, count_negative_weights,
                        laplacian_from_weights, mean_value_weights)
from mesh_utils import boundary_loop, corner_angles, make_plane_grid
from param import parameterize
from solvers import solve_fixed_boundary


def suma_angulos_opuestos(V, F):
    """Para cada arista (i, j), i < j, la suma de los ángulos opuestos.

    En una arista interior es alpha + beta; en una de borde, solo alpha.
    Devuelve un diccionario {(i, j): suma}.
    """
    ang = corner_angles(V, F)
    suma = {}
    for c in range(3):  # el ángulo de la esquina c es opuesto a la arista (c+1, c+2)
        a, b = F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        for i, j, x in zip(np.minimum(a, b), np.maximum(a, b), ang[:, c]):
            suma[(i, j)] = suma.get((i, j), 0.0) + x
    return suma


def pregunta_1_1_conteo():
    print("\n== 1.1 Pesos cotangentes negativos ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        W = cotangent_weights(V, F)
        neg = count_negative_weights(W)            # pares dirigidos (i, j) y (j, i)
        borde = set(zip(np.minimum(loop, np.roll(loop, -1)), np.maximum(loop, np.roll(loop, -1))))
        suma = suma_angulos_opuestos(V, F)
        interior_obtusas = sum(1 for e, x in suma.items() if e not in borde and x > np.pi)
        borde_obtusas = sum(1 for e, x in suma.items() if e in borde and x > np.pi / 2)
        obtusos = int((corner_angles(V, F).max(axis=1) > np.pi / 2).sum())
        Wm = mean_value_weights(V, F)
        filas.append([nombre, len(suma), neg, neg // 2, interior_obtusas, borde_obtusas,
                      obtusos, "%.0f%%" % (100 * obtusos / len(F)), "%.3g" % Wm.data.min()])
    guardar_tabla("p1_pesos_negativos.csv",
                  ["malla", "aristas", "neg_w (pares)", "aristas con D<0",
                   "interiores alfa+beta>pi", "borde alfa>pi/2", "triang. obtusos",
                   "% obtusos", "min peso valor medio"], filas)


def pregunta_1_1_flips():
    print("\n== 1.1 Triangulos invertidos con borde circular ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        for espaciado in ["arc", "uniform"]:
            fila = [nombre, espaciado]
            for metodo in ["harmonic", "meanvalue"]:
                UV, _ = parameterize(V, F, metodo, "circle", espaciado, loop=loop)
                _, resumen = distortion_report(V, F, UV)
                fila.append(resumen["flips"])
            filas.append(fila)
    guardar_tabla("p1_flips_circulo.csv", ["malla", "espaciado", "flips harmonic",
                                           "flips meanvalue"], filas)


def anillo(F, caras, k):
    """Caras a distancia <= k (en anillos de vecinos) de las caras dadas."""
    sel = np.zeros(len(F), dtype=bool)
    sel[caras] = True
    for _ in range(k):
        verts = np.unique(F[sel])
        sel = np.isin(F, verts).any(axis=1)
    return np.where(sel)[0]


def normal(V, f):
    """Normal unitaria del triángulo f = (a, b, c)."""
    n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
    return n / np.linalg.norm(n)


def figura_flip_lucy():
    """Figura del triángulo invertido de Lucy (pesos armónicos, círculo, arc).

    El triángulo queda casi colineal en (u, v) (más delgado que las líneas de
    las aristas), así que en Polyscope no se alcanza a ver. Se dibuja con
    Matplotlib usando el mismo arreglo per_face["flipped"] que el modo de
    color flips del visor: rojo si está invertido, verde si no.
    """
    V, F, loop = cargar("lucy")
    UV = {metodo: parameterize(V, F, metodo, "circle", "arc", loop=loop)[0]
          for metodo in ["harmonic", "meanvalue"]}
    invertidas, _ = invertidas_y_degeneradas(V, F, UV["harmonic"])
    t = invertidas[0]
    print("Lucy: cara invertida", t, "vertices", F[t],
          "angulos (grados)", np.degrees(corner_angles(V, F[t:t + 1])).round(1))
    # En 3D la cara está plegada sobre sus vecinas: sus normales casi se oponen
    for f in np.where((np.isin(F, F[t]).sum(axis=1) == 2))[0]:
        angulo = np.degrees(np.arccos(np.clip(normal(V, F[t]) @ normal(V, F[f]), -1, 1)))
        print("   angulo entre su normal y la de la cara vecina %d: %.0f grados" % (f, angulo))

    W = cotangent_weights(V, F).tocoo()
    negativas = [(i, j) for i, j, w in zip(W.row, W.col, W.data) if w < 0 and i < j]
    zona = anillo(F, [t], 2)
    vz = set(np.unique(F[zona]).tolist())
    fig, axs = plt.subplots(1, 2, figsize=(2.3, 1.3))
    for ax, metodo in zip(axs, UV):
        P = UV[metodo]
        dibujar_malla_2d(ax, P, F[zona], lw=0.3)
        if metodo == "harmonic":   # aristas con peso cotangente negativo
            for i, j in negativas:
                if i in vz and j in vz:
                    ax.plot(P[[i, j], 0], P[[i, j], 1], color="tab:blue", lw=0.6)
        invertidas, _ = invertidas_y_degeneradas(V, F, P)
        color = "tab:red" if t in invertidas else "tab:green"
        tri = P[F[t][[0, 1, 2, 0]]]
        ax.plot(tri[:, 0], tri[:, 1], color=color, lw=1.6)
        encuadrar(ax, tri, margen=1.6)
        ax.set_title(metodo)
    fig.savefig(ruta("p1_lucy_flip_uv.png"))
    plt.close(fig)


def pregunta_1_3():
    print("\n== 1.3 Reproduccion lineal (malla plana de tests.py) ==")
    Vg, Fg = make_plane_grid(n=14, jitter=0.35, seed=3)
    loop = boundary_loop(Fg)
    filas = []
    for metodo in ["uniform", "harmonic", "meanvalue"]:
        L = laplacian_from_weights(WEIGHTS[metodo](Vg, Fg))
        UV = solve_fixed_boundary(L, loop, Vg[loop, :2])
        _, resumen = distortion_report(Vg, Fg, UV)
        filas.append([metodo, "%.2e" % np.abs(UV - Vg[:, :2]).max(),
                      "%.3f" % resumen["conf_med"], "%.3f" % resumen["conf_p95"]])
    guardar_tabla("p1_reproduccion_lineal.csv",
                  ["metodo", "error max", "conf_med", "conf_p95"], filas)

    print("\n== 1.3 Distorsion conforme en zonas casi planas (borde circular, arc) ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        # Curvatura de Gauss discreta: defecto angular 2*pi - suma de ángulos
        suma = np.bincount(F.ravel(), corner_angles(V, F).ravel(), minlength=len(V))
        defecto = np.abs(2 * np.pi - suma)
        defecto[loop] = np.inf                      # los vértices de borde no cuentan
        peor = defecto[F].max(axis=1)               # la cara es tan curva como su peor vértice
        planas = peor <= np.percentile(peor[np.isfinite(peor)], 25)
        fila = [nombre, int(planas.sum())]
        for metodo in ["uniform", "harmonic", "meanvalue"]:
            UV, _ = parameterize(V, F, metodo, "circle", "arc", loop=loop)
            cara, _ = distortion_report(V, F, UV)
            fila.append("%.3f" % np.median(cara["conformal"][planas]))
        filas.append(fila)
    guardar_tabla("p1_zonas_planas.csv", ["malla", "caras casi planas (25%)",
                                          "conf_med uniform", "conf_med harmonic",
                                          "conf_med meanvalue"], filas)


if __name__ == "__main__":
    pregunta_1_1_conteo()
    pregunta_1_1_flips()
    figura_flip_lucy()
    pregunta_1_3()
