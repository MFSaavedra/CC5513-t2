# -*- coding: utf-8 -*-
"""
Matrices de pesos y Laplacianos para parametrización con el modelo de resortes.

Convención usada en toda la tarea
---------------------------------
W es una matriz dispersa n x n con W[i, j] = D_ij > 0 (en general) si i y j
son vecinos, y 0 en otro caso. La fila i contiene los pesos que el vértice i
asigna a sus vecinos; W NO tiene por qué ser simétrica (mean value no lo es).

El Laplaciano es  L = diag(sum_j D_ij) - W,  es decir
    L_ii = sum_j D_ij,     L_ij = -D_ij.
Cada fila de L suma 0. La parametrización con borde fijo resuelve L U = 0
para los vértices interiores (ver solvers.py).

QUÉ DEBES IMPLEMENTAR AQUÍ (Parte 1)
    - cotangent_weights   (pesos armónicos)
    - mean_value_weights  (pesos de valor medio de Floater)
"""
import numpy as np
import scipy.sparse as sp

from mesh_utils import corner_angles, unique_edges


def _sparse_from_triplets(rows, cols, vals, n):
    """Construye una matriz CSR n x n. Las entradas repetidas (i, j) se SUMAN.

    Esto es muy útil: basta con generar una contribución por triángulo y
    dejar que SciPy acumule las de los dos triángulos que comparten una arista.
    """
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ---------------------------------------------------------------------------
# Pesos uniformes (Tutte) -- DADO, úsalo como ejemplo
# ---------------------------------------------------------------------------
def uniform_weights(V, F):
    """D_ij = 1 para cada par de vecinos (i, j)."""
    n = len(V)
    E = unique_edges(F)
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    vals = np.ones(len(rows))
    return _sparse_from_triplets(rows, cols, vals, n)


# ---------------------------------------------------------------------------
# Pesos armónicos (cotangentes) -- PARTE 1a
# ---------------------------------------------------------------------------
def cotangent_weights(V, F):
    """D_ij = 1/2 (cot(alpha_ij) + cot(beta_ij)).

    alpha_ij y beta_ij son los ángulos OPUESTOS a la arista (i, j) en los dos
    triángulos que la comparten. Si la arista está en el borde hay un solo
    triángulo y por lo tanto un solo término.

    Pista: en el triángulo (a, b, c), el ángulo en la esquina c es opuesto a
    la arista (a, b). Usa corner_angles(V, F) y _sparse_from_triplets: genera,
    para cada triángulo y cada esquina, las contribuciones a W[a, b] y W[b, a].
    No uses ciclos sobre los vértices.

    Ojo: estos pesos pueden ser NEGATIVOS (triángulos obtusos). No los
    corrijas: en la pregunta 1.1 vas a contar cuántos hay.
    """
    n = len(V)
    angles = corner_angles(V, F)  # angles[t, c] = ángulo en el vértice F[t, c]
    rows, cols, vals = [], [], []
    for c in range(3):
        # El ángulo en la esquina c es opuesto a la arista (a, b).
        a = F[:, (c + 1) % 3]
        b = F[:, (c + 2) % 3]
        half_cot = 0.5 / np.tan(angles[:, c])
        # Cada triángulo aporta un término a W[a, b] y el mismo a W[b, a].
        # Las aristas interiores reciben dos términos (cot alpha y cot beta),
        # que _sparse_from_triplets suma; las de borde reciben uno solo.
        rows += [a, b]
        cols += [b, a]
        vals += [half_cot, half_cot]
    return _sparse_from_triplets(np.concatenate(rows), np.concatenate(cols),
                                 np.concatenate(vals), n)


# ---------------------------------------------------------------------------
# Pesos de valor medio (mean value, Floater 2003) -- PARTE 1b
# ---------------------------------------------------------------------------
def mean_value_weights(V, F):
    """D_ij = ( tan(gamma_ij / 2) + tan(delta_ij / 2) ) / ||v_i - v_j||.

    gamma_ij y delta_ij son los ángulos EN EL VÉRTICE v_i de los dos
    triángulos que comparten la arista (i, j) (ver figura del enunciado).
    Como los ángulos se miden en v_i, W[i, j] != W[j, i]: la matriz NO es
    simétrica.

    Pista: en el triángulo (a, b, c), el ángulo en la esquina a contribuye
    a W[a, b] (dividido por ||v_a - v_b||) y a W[a, c] (dividido por
    ||v_a - v_c||). Recorre las 3 esquinas.
    """
    n = len(V)
    angles = corner_angles(V, F)  # angles[t, c] = ángulo en el vértice F[t, c]
    rows, cols, vals = [], [], []
    for c in range(3):
        i = F[:, c]               # vértice donde se mide el ángulo
        half_tan = np.tan(angles[:, c] / 2)
        # El ángulo en v_i es uno de los dos ángulos (gamma o delta) de las
        # aristas (i, j) e (i, k), los dos lados del triángulo que salen de i.
        for j in (F[:, (c + 1) % 3], F[:, (c + 2) % 3]):
            length = np.linalg.norm(V[i] - V[j], axis=1)
            rows.append(i)
            cols.append(j)
            vals.append(half_tan / length)
    return _sparse_from_triplets(np.concatenate(rows), np.concatenate(cols),
                                 np.concatenate(vals), n)


# ---------------------------------------------------------------------------
# Laplaciano a partir de pesos -- DADO
# ---------------------------------------------------------------------------
def laplacian_from_weights(W):
    """L = diag(suma de filas de W) - W."""
    d = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(d) - W).tocsr()


def count_negative_weights(W):
    """Número de pares (i, j) con D_ij < 0 (cada par dirigido cuenta una vez)."""
    W = W.tocsr()
    return int(np.sum(W.data < -1e-12))


WEIGHTS = {
    "uniform": uniform_weights,
    "harmonic": cotangent_weights,
    "meanvalue": mean_value_weights,
}
