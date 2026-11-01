# -*- coding: utf-8 -*-
"""
Resolución de los sistemas lineales de la parametrización.

    solve_fixed_boundary : modelo de resortes con borde fijo (DADO)
    lscm                 : Least Squares Conformal Maps, borde libre (DADO)
    conformal_energy     : E_D, A y E_C de una parametrización (DADO)

Este archivo está COMPLETO. En la Parte 3 debes leerlo, relacionar cada
paso con la formulación del enunciado y analizar su comportamiento.
LSCM usa los pesos cotangentes, así que funciona una vez que completes la
Parte 1a.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from laplacians import cotangent_weights, laplacian_from_weights


# ---------------------------------------------------------------------------
# Borde fijo -- DADO
# ---------------------------------------------------------------------------
def solve_fixed_boundary(L, loop, uv_boundary):
    """Resuelve L U = 0 en los vértices interiores con U = uv_boundary en el borde.

    En vez de reemplazar filas de L (como en la versión anterior), se separan
    los vértices en interiores (I) y de borde (B) y se resuelve el sistema
    reducido
            L_II U_I = - L_IB U_B
    que tiene una incógnita por vértice interior. Se factoriza L_II una sola
    vez (LU dispersa) y se resuelve para las dos columnas (u y v).
    """
    n = L.shape[0]
    L = L.tocsr()
    is_b = np.zeros(n, dtype=bool)
    is_b[loop] = True
    I = np.where(~is_b)[0]
    U = np.zeros((n, 2))
    U[loop] = uv_boundary
    if len(I) == 0:
        return U
    L_II = L[I][:, I].tocsc()
    L_IB = L[I][:, loop]
    rhs = -(L_IB @ uv_boundary)
    lu = spla.splu(L_II)
    U[I] = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
    return U


# ---------------------------------------------------------------------------
# LSCM -- DADO (se analiza en la Parte 3)
# ---------------------------------------------------------------------------
def choose_pins(V, loop):
    """Elige dos vértices del borde aproximadamente lo más lejanos posible. DADO.

    Devuelve (p0, p1, d) con d = ||V[p0] - V[p1]||.
    """
    P = V[loop]
    a = loop[np.argmax(np.linalg.norm(P - P[0], axis=1))]
    b = loop[np.argmax(np.linalg.norm(P - V[a], axis=1))]
    return int(a), int(b), float(np.linalg.norm(V[a] - V[b]))


def boundary_area_matrix(n, loop):
    """Matriz K (n x n) tal que  u^T K v = área con signo de la parametrización.

    Para una malla plana con borde recorrido en sentido antihorario
        Área(u, v) = 1/2 * sum_{aristas i->j del borde} (u_i v_j - u_j v_i)
    Por lo tanto K[i, j] = +1/2 y K[j, i] = -1/2 para cada arista i->j del
    borde (en el orden de loop, cerrando con loop[-1] -> loop[0]).
    """
    i = np.asarray(loop)
    j = np.roll(i, -1)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([np.full(len(i), 0.5), np.full(len(i), -0.5)])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def lscm(V, F, loop, pins=None):
    """Parametrización conforme por mínimos cuadrados (borde libre).

    Energía conforme (Lévy et al. 2002; forma de Mullen et al. 2008):
        E_C(u, v) = E_D(u, v) - Área(u, v)
                  = 1/2 (u^T L u + v^T L v) - u^T K v
    donde L es el Laplaciano COTANGENTE (Parte 1a) y K la matriz de área
    (boundary_area_matrix). Con x = [u; v] (vector de 2n):
        E_C = 1/2 x^T Q x,     Q = [[ L, -K ],
                                    [ K,  L ]]
    E_C >= 0 y vale 0 solo si el mapa es conforme. Su mínimo no es único
    (cualquier semejanza del plano lo preserva), así que se fijan dos
    vértices ("pins"): p0 -> (0, 0) y p1 -> (d, 0). Con f = variables libres y
    p = fijas, el mínimo se obtiene de
        Q_ff x_f = - Q_fp x_p.

    pins: None (usa choose_pins) o un par (p0, p1) de índices de VÉRTICES.

    Devuelve UV de tamaño (n, 2).
    """
    n = len(V)
    if pins is None:
        p0, p1, d = choose_pins(V, loop)
    else:
        p0, p1 = pins
        d = float(np.linalg.norm(V[p0] - V[p1]))
    L = laplacian_from_weights(cotangent_weights(V, F))
    K = boundary_area_matrix(n, loop)
    Q = sp.bmat([[L, -K], [K, L]]).tocsr()
    fixed = np.array([p0, p1, p0 + n, p1 + n])
    xp = np.array([0.0, d, 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * n), fixed)
    Q_ff = Q[free][:, free].tocsc()
    Q_fp = Q[free][:, fixed]
    x = np.zeros(2 * n)
    x[fixed] = xp
    x[free] = spla.spsolve(Q_ff, -(Q_fp @ xp))
    return np.column_stack([x[:n], x[n:]])


# ---------------------------------------------------------------------------
# Energías -- DADO (útil para la Parte 3)
# ---------------------------------------------------------------------------
def conformal_energy(V, F, UV, loop=None):
    """Evalúa las energías de LSCM para una parametrización cualquiera.

    Devuelve un diccionario con
        E_D   energía de Dirichlet  1/2 (u^T L u + v^T L v)
        A     área con signo de la parametrización  u^T K v
        E_C   energía conforme  E_D - A  (>= 0)
        ratio E_D / |A|  (>= 1; vale 1 solo si el mapa es conforme). A
              diferencia de E_C, no depende de la escala de UV, así que
              sirve para comparar métodos distintos.
    """
    from mesh_utils import boundary_loop
    if loop is None:
        loop = boundary_loop(F)
    n = len(V)
    L = laplacian_from_weights(cotangent_weights(V, F))
    K = boundary_area_matrix(n, loop)
    u, v = UV[:, 0], UV[:, 1]
    E_D = 0.5 * (u @ (L @ u) + v @ (L @ v))
    A = u @ (K @ v)
    return {"E_D": float(E_D), "A": float(A), "E_C": float(E_D - A),
            "ratio": float(E_D / abs(A)) if A != 0 else float("inf")}
