# -*- coding: utf-8 -*-
"""
Lectura y escritura de mallas en formato OFF usando solo NumPy.

(No se necesita OpenMesh: la versión anterior de esta tarea dependía de
openmesh, que suele dar problemas de instalación en Windows.)

Este archivo está COMPLETO. No necesitas modificarlo.
"""
import numpy as np


def read_off(path):
    """Lee un archivo .OFF y devuelve (V, F).

    V : (n, 3) float64  -> coordenadas de los vértices
    F : (m, 3) int64    -> índices de los triángulos (base 0)

    Polígonos de más de 3 lados se triangulan en abanico.
    """
    with open(path, "r") as f:
        lines = [l.split("#")[0] for l in f]
    toks = " ".join(lines).split()
    if not toks or not toks[0].upper().endswith("OFF"):
        raise ValueError(f"{path}: no es un archivo OFF")
    # Algunos archivos escriben 'OFF' y los conteos en la misma línea
    i = 1
    nv, nf = int(toks[i]), int(toks[i + 1])
    i += 3  # se ignora el número de aristas
    V = np.array(toks[i:i + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i += 3 * nv
    F = []
    for _ in range(nf):
        k = int(toks[i])
        idx = [int(x) for x in toks[i + 1:i + 1 + k]]
        i += 1 + k
        for j in range(1, k - 1):  # triangulación en abanico
            F.append([idx[0], idx[j], idx[j + 1]])
    return V, np.asarray(F, dtype=np.int64)


def write_off(path, V, F):
    """Escribe (V, F) en formato OFF. Si V tiene 2 columnas se agrega z = 0."""
    V = np.asarray(V, dtype=np.float64)
    if V.shape[1] == 2:
        V = np.column_stack([V, np.zeros(len(V))])
    with open(path, "w") as f:
        f.write("OFF\n%d %d 0\n" % (len(V), len(F)))
        for p in V:
            f.write("%.10g %.10g %.10g\n" % tuple(p))
        for t in F:
            f.write("3 %d %d %d\n" % tuple(t))
