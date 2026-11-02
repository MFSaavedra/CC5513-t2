# -*- coding: utf-8 -*-
"""
Parte 5: análisis comparativo.

    python experimentos/p5_comparacion.py

- Imágenes de los cuatro métodos en las tres mallas (borde circular y
  espaciado arc para los de borde fijo): tablero sobre la superficie,
  dominio (u, v) y mapas de distorsión conforme y de área.
- 5.1  Orden de los métodos por distorsión conforme y de área.
- 5.2  Distorsión de área en función de la distancia geodésica al borde.
- 5.4  Tiempos de cada etapa en Lucy.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra

# comun va primero: agrega la carpeta del código base a sys.path
from comun import (MALLAS, VISTAS, cargar, foto_3d, foto_uv, guardar_tabla, plt,
                   ruta, visor)
from boundary import map_boundary
from distortion import distortion_report
from laplacians import WEIGHTS, laplacian_from_weights
from mesh_utils import unique_edges
from param import parameterize
from solvers import boundary_area_matrix, choose_pins, conformal_energy

METODOS = ["uniform", "harmonic", "meanvalue", "lscm"]
TABLERO = {"nefertiti": 0.04, "bunny": 0.02, "lucy": 0.015}   # tamaño de casilla (uv normalizado)
COLORES = {"uniform": "tab:gray", "harmonic": "tab:blue", "meanvalue": "tab:green",
           "lscm": "tab:red"}


def imagenes():
    print("\n== Imagenes de los cuatro metodos ==")
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        for metodo in METODOS:
            UV, _ = parameterize(V, F, metodo, "circle", "arc", loop=loop)
            base = "p5_%s_%s" % (nombre, metodo)
            for color in ["checker", "conformal", "area"]:
                v = visor(V, F, UV, method=metodo, color=color, checker=TABLERO[nombre])
                foto_3d(v, ruta("%s_%s.png" % (base, color)), VISTAS[nombre])
            foto_uv(visor(V, F, UV, method=metodo, color="checker"), ruta(base + "_uv.png"))
            print("  ", base)


def mosaico(nombre):
    """Figura de una malla: filas = tablero, (u, v), conforme, área; columnas = métodos."""
    filas = ["checker", "uv", "conformal", "area"]
    etiquetas = ["tablero", "dominio (u, v)", "conforme", "área"]
    fig, axs = plt.subplots(len(filas), len(METODOS), figsize=(7.2, 7.2))
    for j, metodo in enumerate(METODOS):
        for i, fila in enumerate(filas):
            ax = axs[i, j]
            ax.imshow(plt.imread(ruta("p5_%s_%s_%s.png" % (nombre, metodo, fila))))
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_visible(False)
            if i == 0:
                ax.set_title(metodo, fontsize=9)
            if j == 0:
                ax.set_ylabel(etiquetas[i], fontsize=9)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(ruta("p5_mosaico_%s.png" % nombre), dpi=200)  # composición de capturas
    plt.close(fig)


def mosaico_tableros():
    """Figura resumen: tablero sobre la superficie, filas = mallas, columnas = métodos."""
    fig, axs = plt.subplots(len(MALLAS), len(METODOS), figsize=(4.3, 2.75))
    for i, nombre in enumerate(MALLAS):
        for j, metodo in enumerate(METODOS):
            ax = axs[i, j]
            ax.imshow(plt.imread(ruta("p5_%s_%s_checker.png" % (nombre, metodo))))
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_visible(False)
            if i == 0:
                ax.set_title(metodo)
            if j == 0:
                ax.set_ylabel(nombre)
    fig.tight_layout(h_pad=0.05, w_pad=0.05)
    fig.savefig(ruta("p5_tableros.png"))
    plt.close(fig)


def pregunta_5_1():
    print("\n== 5.1 Orden de los metodos (borde circular, arc) ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        conf, area = {}, {}
        for metodo in METODOS:
            UV, _ = parameterize(V, F, metodo, "circle", "arc", loop=loop)
            _, s = distortion_report(V, F, UV)
            conf[metodo], area[metodo] = s["conf_med"], s["area_mean"]
            filas.append([nombre, metodo, "%.3f" % s["conf_med"], "%.3f" % s["conf_p95"],
                          "%.3f" % s["area_mean"], s["flips"], s["degen"],
                          "%.4f" % conformal_energy(V, F, UV, loop)["ratio"]])
        print("  %s conforme: %s" % (nombre, " < ".join(sorted(conf, key=conf.get))))
        print("  %s area:     %s" % (nombre, " < ".join(sorted(area, key=area.get))))
    guardar_tabla("p5_metodos.csv", ["malla", "metodo", "conf_med", "conf_p95", "area_mean",
                                     "flips", "degen", "ED/|A|"], filas)


def distancia_al_borde(V, F, loop):
    """Distancia geodésica (por las aristas) de cada vértice al borde."""
    E = unique_edges(F)
    largo = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = sp.coo_matrix((largo, (E[:, 0], E[:, 1])), shape=(len(V), len(V)))
    return dijkstra(G, directed=False, indices=loop).min(axis=0)


def pregunta_5_2():
    print("\n== 5.2 Distorsion de area vs distancia al borde ==")
    fig, axs = plt.subplots(3, 1, figsize=(3.0, 2.75), sharex=True)
    filas = []
    for ax, nombre in zip(axs, MALLAS):
        V, F, loop = cargar(nombre)
        d = distancia_al_borde(V, F, loop)[F].mean(axis=1)
        d = d / d.max()                                   # distancia relativa de cada cara
        lejos = d > 0.6
        for metodo in METODOS:
            UV, _ = parameterize(V, F, metodo, "circle", "arc", loop=loop)
            cara, _ = distortion_report(V, F, UV)
            area = np.where(cara["conformal"] < 1e12, cara["area"], np.nan)  # sin degeneradas
            bins = np.linspace(0, 1, 21)
            k = np.digitize(d, bins) - 1
            med = [np.nanmedian(area[k == b]) if np.any(k == b) else np.nan for b in range(20)]
            ax.plot((bins[:-1] + bins[1:]) / 2, med, color=COLORES[metodo], lw=0.9, label=metodo)
            filas.append([nombre, metodo, "%.1f" % np.nanmedian(area[~lejos]),
                          "%.1f" % np.nanmedian(area[lejos]), "%.1f" % np.nanmin(area),
                          "%.0f%%" % (100 * lejos.mean())])
        ax.axhline(0, color="k", lw=0.5)
        ax.text(0.98, 0.92, nombre, transform=ax.transAxes, ha="right", va="top")
    axs[2].set_xlabel("distancia geodésica al borde / máxima")
    axs[1].set_ylabel("mediana de $\\log_2(\\sigma_1\\sigma_2)$")
    axs[1].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(h_pad=0.2)
    fig.savefig(ruta("p5_area_vs_distancia.png"))
    plt.close(fig)
    guardar_tabla("p5_area_distancia.csv",
                  ["malla", "metodo", "mediana log2 area (d<=0.6)", "mediana log2 area (d>0.6)",
                   "minimo log2 area", "% caras con d>0.6"], filas)


def cronometro(f, repeticiones=5):
    """Ejecuta f() varias veces y devuelve (mediana del tiempo, último resultado)."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        r = f()
        tiempos.append(time.perf_counter() - t0)
    return np.median(tiempos), r


def pregunta_5_4():
    """Tiempo de cada etapa, repitiendo a mano los pasos de param.py y solvers.py."""
    print("\n== 5.4 Tiempos en Lucy (mediana de 5 repeticiones, segundos) ==")
    V, F, loop = cargar("lucy")
    n = len(V)
    filas = []
    for metodo in ["uniform", "harmonic", "meanvalue"]:
        t_w, W = cronometro(lambda: WEIGHTS[metodo](V, F))
        t_l, L = cronometro(lambda: laplacian_from_weights(W))
        t_b, uv_b = cronometro(lambda: map_boundary(V, loop, "circle", "arc"))
        interior = np.setdiff1d(np.arange(n), loop)
        t_s, (L_II, L_IB) = cronometro(lambda: (L[interior][:, interior].tocsc(),
                                                L[interior][:, loop]))
        t_f, lu = cronometro(lambda: spla.splu(L_II))
        rhs = -(L_IB @ uv_b)
        t_r, _ = cronometro(lambda: (lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])))
        t_tot, (UV, _) = cronometro(lambda: parameterize(V, F, metodo, "circle", "arc", loop=loop))
        t_d, _ = cronometro(lambda: distortion_report(V, F, UV))
        filas.append([metodo, "%.3f" % t_w, "%.3f" % t_l, "%.4f" % t_b, "%.3f" % t_s,
                      "%.3f" % t_f, "%.4f" % t_r, "%.3f" % t_tot, "%.3f" % t_d,
                      "%d" % L_II.shape[0]])
    # LSCM
    p0, p1, d = choose_pins(V, loop)
    t_w, W = cronometro(lambda: WEIGHTS["harmonic"](V, F))
    t_l, L = cronometro(lambda: laplacian_from_weights(W))

    def armar_Q():
        K = boundary_area_matrix(n, loop)
        return sp.bmat([[L, -K], [K, L]]).tocsr()

    t_b, Q = cronometro(armar_Q)
    fijas = np.array([p0, p1, p0 + n, p1 + n])
    libres = np.setdiff1d(np.arange(2 * n), fijas)
    t_s, (Q_ff, Q_fp) = cronometro(lambda: (Q[libres][:, libres].tocsc(), Q[libres][:, fijas]))
    rhs = -(Q_fp @ np.array([0.0, d, 0.0, 0.0]))
    t_f, lu = cronometro(lambda: spla.splu(Q_ff))
    t_r, _ = cronometro(lambda: lu.solve(rhs))
    t_sp, _ = cronometro(lambda: spla.spsolve(Q_ff, rhs))  # lo que usa solvers.lscm
    t_tot, (UV, _) = cronometro(lambda: parameterize(V, F, "lscm", loop=loop))
    t_d, _ = cronometro(lambda: distortion_report(V, F, UV))
    filas.append(["lscm", "%.3f" % t_w, "%.3f" % t_l, "%.4f" % t_b, "%.3f" % t_s,
                  "%.3f" % t_f, "%.4f" % t_r, "%.3f" % t_tot, "%.3f" % t_d,
                  "%d" % Q_ff.shape[0]])
    print("  lscm: spsolve (factoriza y resuelve) = %.3f s" % t_sp)
    guardar_tabla("p5_tiempos_lucy.csv",
                  ["metodo", "pesos", "laplaciano", "borde / Q", "extraer submatrices",
                   "factorizacion LU", "sustitucion", "total parameterize",
                   "distortion_report", "incognitas"], filas)


if __name__ == "__main__":
    imagenes()
    for nombre in MALLAS:
        mosaico(nombre)
    mosaico_tableros()
    pregunta_5_1()
    pregunta_5_2()
    pregunta_5_4()
