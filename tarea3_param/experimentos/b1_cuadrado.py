# -*- coding: utf-8 -*-
"""
Bonus B1: cuadrado sin triángulos degenerados.

    python experimentos/b1_cuadrado.py

Compara el borde "square" con "bulged_square" (boundary.bulged_square_points:
el mismo cuadrado con los lados levemente curvados hacia afuera) en las tres
mallas y con los tres tipos de pesos. Para cada caso cuenta los triángulos
degenerados, cuántos tocan el borde (entre ellos, las orejas de la pregunta
2.1) y a cuántos anillos del borde está el degenerado más cercano.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

# comun va primero: agrega la carpeta del código base a sys.path
from comun import (MALLAS, cargar, foto_uv, guardar_tabla, invertidas_y_degeneradas,
                   ruta, visor)
from distortion import distortion_report
from mesh_utils import unique_edges
from param import parameterize


def anillos_al_borde(V, F, loop):
    """Número de aristas (anillos) entre cada vértice y el borde."""
    E = unique_edges(F)
    G = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V)))
    return dijkstra(G, directed=False, indices=loop, unweighted=True).min(axis=0)


if __name__ == "__main__":
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        anillos = anillos_al_borde(V, F, loop)[F].min(axis=1)   # anillo de cada cara
        for metodo in ["uniform", "harmonic", "meanvalue"]:
            for borde in ["circle", "square", "bulged_square"]:
                UV, _ = parameterize(V, F, metodo, borde, "arc", loop=loop)
                _, r = distortion_report(V, F, UV)
                _, degeneradas = invertidas_y_degeneradas(V, F, UV)
                en_borde = int(np.sum(anillos[degeneradas] == 0))
                cercano = int(anillos[degeneradas].min()) if len(degeneradas) else "-"
                filas.append([nombre, metodo, borde, r["degen"], en_borde, cercano, r["flips"],
                              "%.3f" % r["conf_med"], "%.3f" % r["area_mean"]])
    guardar_tabla("b1_cuadrado.csv",
                  ["malla", "metodo", "borde", "degen", "degen en el borde", "anillo minimo",
                   "flips", "conf_med", "area_mean"], filas)

    # Dominio (u, v) de Bunny con el cuadrado abombado
    V, F, loop = cargar("bunny")
    UV, _ = parameterize(V, F, "meanvalue", "bulged_square", "arc", loop=loop)
    foto_uv(visor(V, F, UV, method="meanvalue"), ruta("b1_bunny_uv.png"))
