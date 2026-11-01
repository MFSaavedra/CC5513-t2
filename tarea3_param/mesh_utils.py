# -*- coding: utf-8 -*-
"""
Utilidades geométricas y topológicas para mallas de triángulos.

Este archivo está COMPLETO. Puedes (y deberías) usar estas funciones en tu
implementación. Todas están vectorizadas con NumPy: evita los ciclos for
sobre vértices o caras cuando puedas usar operaciones sobre arreglos.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Topología
# ---------------------------------------------------------------------------
def halfedges(F):
    """Devuelve las semi-aristas (i -> j) de cada cara, en el orden de la cara.

    Para la cara (a, b, c) se generan a->b, b->c, c->a. Resultado: (3m, 2).
    """
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)


def unique_edges(F):
    """Aristas no dirigidas (i < j), sin repetir. Resultado: (E, 2)."""
    he = np.sort(halfedges(F), axis=1)
    return np.unique(he, axis=0)


def boundary_loop(F):
    """Calcula el borde de una malla con topología de disco.

    Devuelve una lista de índices de vértices ordenada siguiendo la
    orientación de las caras (sentido antihorario si las normales apuntan
    hacia el observador). Esto importa en LSCM: el área con signo de la
    parametrización se calcula recorriendo el borde en este sentido.

    Una semi-arista i->j es de borde si su gemela j->i no existe.
    (Reemplaza a compute_boundary de la versión anterior, que usaba una
    matriz densa de n x n.)
    """
    he = halfedges(F)
    key = set(map(tuple, he.tolist()))
    nxt = {}
    for i, j in he.tolist():
        if (j, i) not in key:
            if i in nxt:
                raise ValueError("El borde no es una curva simple (vértice no-manifold).")
            nxt[i] = j
    if not nxt:
        raise ValueError("La malla no tiene borde (¿es cerrada?).")
    start = min(nxt)
    loop = [start]
    v = nxt[start]
    while v != start:
        loop.append(v)
        v = nxt[v]
        if len(loop) > len(nxt):
            raise ValueError("Error recorriendo el borde.")
    if len(loop) != len(nxt):
        raise ValueError("La malla tiene más de una componente de borde "
                         "(no es un disco).")
    return np.array(loop, dtype=np.int64)


def euler_characteristic(V, F):
    return len(V) - len(unique_edges(F)) + len(F)


# ---------------------------------------------------------------------------
# Geometría por triángulo
# ---------------------------------------------------------------------------
def face_areas(V, F):
    """Área de cada triángulo. Funciona con V de 2 o 3 columnas."""
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    if V.shape[1] == 2:
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def signed_areas_2d(UV, F):
    """Área con signo de cada triángulo en el plano (positiva = antihorario)."""
    a = UV[F[:, 1]] - UV[F[:, 0]]
    b = UV[F[:, 2]] - UV[F[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def corner_angles(V, F):
    """Ángulo interior en cada esquina de cada triángulo.

    Resultado A de tamaño (m, 3): A[t, c] es el ángulo en el vértice F[t, c].
    """
    A = np.zeros(F.shape, dtype=np.float64)
    for c in range(3):
        p = V[F[:, c]]
        u = V[F[:, (c + 1) % 3]] - p
        w = V[F[:, (c + 2) % 3]] - p
        nu = np.maximum(np.linalg.norm(u, axis=1), 1e-12)
        nw = np.maximum(np.linalg.norm(w, axis=1), 1e-12)
        cosang = np.clip(np.sum(u * w, axis=1) / (nu * nw), -1.0, 1.0)
        A[:, c] = np.arccos(cosang)
    return A


def local_triangle_coords(V, F):
    """Coordenadas 2D de cada triángulo en un sistema local isométrico.

    Para el triángulo (p0, p1, p2) se construye una base ortonormal del plano
    del triángulo con el eje x en la dirección de p1 - p0. El resultado Q
    tiene tamaño (m, 3, 2) y cumple:
        Q[t, 0] = (0, 0)
        Q[t, 1] = (|p1 - p0|, 0)
        Q[t, 2] = (x2, y2) con y2 > 0
    Las distancias y ángulos del triángulo 3D se preservan exactamente.
    Si V tiene 2 columnas, se devuelven las coordenadas trasladadas (sin
    rotar), preservando la orientación del triángulo.
    """
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    m = len(F)
    Q = np.zeros((m, 3, 2))
    if V.shape[1] == 2:
        Q[:, 1] = p1 - p0
        Q[:, 2] = p2 - p0
        return Q
    e1 = p1 - p0
    e2 = p2 - p0
    l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-12)
    x = e1 / l1[:, None]
    n = np.cross(e1, e2)
    y = np.cross(n, x)
    y /= np.maximum(np.linalg.norm(y, axis=1), 1e-12)[:, None]
    Q[:, 1, 0] = l1
    Q[:, 2, 0] = np.sum(e2 * x, axis=1)
    Q[:, 2, 1] = np.sum(e2 * y, axis=1)
    return Q


# ---------------------------------------------------------------------------
# Mallas sintéticas (para pruebas)
# ---------------------------------------------------------------------------
def make_plane_grid(n=12, jitter=0.3, seed=0):
    """Malla plana (z = 0) de un cuadrado [0,1]^2 con n x n celdas.

    Los vértices interiores se desplazan al azar (jitter, en fracción del
    tamaño de celda) para que los triángulos sean irregulares; algunos
    quedan obtusos, así que los pesos cotangentes pueden ser negativos.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    h = 1.0 / n
    interior = (P[:, 0] > 0) & (P[:, 0] < 1) & (P[:, 1] > 0) & (P[:, 1] < 1)
    P[interior] += rng.uniform(-jitter, jitter, size=(interior.sum(), 2)) * h
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    F = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if (i + j) % 2 == 0:
                F += [[a, b, c], [a, c, d]]
            else:
                F += [[a, b, d], [b, c, d]]
    V = np.column_stack([P, np.zeros(len(P))])
    return V, np.array(F, dtype=np.int64)


def normalize_uv(uv):
    """Traslada y escala uv para que quepa en [0,1]^2 (conserva proporciones)."""
    uv = uv - uv.min(axis=0)
    s = uv.max()
    return uv / s if s > 0 else uv
