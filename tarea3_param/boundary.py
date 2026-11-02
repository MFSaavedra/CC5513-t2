# -*- coding: utf-8 -*-
"""
Mapeo del borde de la malla a una curva del plano.

La parametrización con borde fijo necesita, para cada vértice del borde
(en el orden de boundary_loop), una posición b_i en el plano. Se hace en
dos pasos:
    1) asignar a cada vértice del borde un parámetro t_i en [0, 1)
       (boundary_parameter: espaciado uniforme o por longitud de arco);
    2) llevar cada t_i a un punto de la curva destino
       (circle_points o polygon_points).

QUÉ DEBES IMPLEMENTAR AQUÍ (Parte 2)
    - polygon_points

Bonus B1: bulged_square_points, un cuadrado con los lados levemente
curvados que evita los triángulos degenerados (forma "bulged_square").
"""
import numpy as np


def boundary_parameter(V, loop, spacing="arc"):
    """Parámetro t_i en [0, 1) para cada vértice del borde.

    spacing = "arc"     -> proporcional a la longitud de arco acumulada
                           del borde 3D (como en la versión anterior)
    spacing = "uniform" -> t_i = i / nb, ignora la geometría
    """
    nb = len(loop)
    if spacing == "uniform":
        return np.arange(nb) / nb
    if spacing != "arc":
        raise ValueError("spacing debe ser 'arc' o 'uniform'")
    P = V[loop]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)  # |p_{i+1} - p_i|
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return s / seg.sum()


def circle_points(t):
    """Circunferencia unitaria recorrida en sentido antihorario. DADO."""
    return np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])


# ---------------------------------------------------------------------------
# Polígonos destino
# ---------------------------------------------------------------------------
def square_corners():
    """Cuadrado [-1, 1]^2, vértices en sentido antihorario."""
    return np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])


def star_corners(k=5, r_out=1.0, r_in=0.45):
    """Estrella de k puntas (NO convexa), vértices en sentido antihorario."""
    ang = np.pi / 2 + np.arange(2 * k) * np.pi / k
    r = np.where(np.arange(2 * k) % 2 == 0, r_out, r_in)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def polygon_points(t, corners):
    """Lleva parámetros t en [0, 1) a puntos sobre el perímetro de un polígono.

    corners : (k, 2) vértices del polígono cerrado, en orden antihorario.
              El lado q va de corners[q] a corners[(q+1) % k].

    El perímetro se recorre a velocidad constante: t = 0 es corners[0] y
    t = s / P (P = perímetro) es el punto que está a distancia s de corners[0]
    medida a lo largo del perímetro. Es decir, t se reparte entre los lados
    en proporción a su longitud (NO 1/k por lado).

    Debe devolver un arreglo (len(t), 2).
    Pista: longitudes acumuladas de los lados + np.searchsorted.
    """
    t = np.asarray(t, dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)
    closed = np.vstack([corners, corners[:1]])                # repite corners[0] al final
    side = np.linalg.norm(np.diff(closed, axis=0), axis=1)    # largo de cada lado
    dist = np.concatenate([[0.0], np.cumsum(side)])           # distancia de corners[0] a cada esquina
    s = (t % 1.0) * dist[-1]                                  # distancia a recorrer, s = t * P
    # Entre dos esquinas consecutivas la posición es lineal en s.
    # np.interp busca el lado de cada s (como searchsorted) e interpola en él.
    x = np.interp(s, dist, closed[:, 0])
    y = np.interp(s, dist, closed[:, 1])
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Bonus B1: cuadrado sin triángulos degenerados
# ---------------------------------------------------------------------------
def bulged_square_points(t, bulge=0.05):
    """Cuadrado [-1, 1]^2 con los lados levemente curvados hacia afuera.

    En el cuadrado normal, si los tres vértices de un triángulo caen en el
    mismo lado quedan colineales y el triángulo es degenerado (pregunta 2.1).
    Aquí cada punto se desplaza hacia afuera, perpendicular a su lado, en
    4 * bulge * s * (1 - s), donde s en [0, 1) es su posición dentro del
    lado: 0 en las esquinas y `bulge` en el centro. Cada lado queda como un
    arco de parábola, así que la curva es estrictamente convexa y tres puntos
    del borde nunca son colineales.

    Se usa como forma "bulged_square":
        parameterize(V, F, "meanvalue", shape="bulged_square")
    """
    t = np.asarray(t, dtype=np.float64) % 1.0
    corners = square_corners()
    side = np.minimum((4 * t).astype(int), 3)    # lado de cada t (los 4 lados miden lo mismo)
    s = 4 * t - side                             # posición dentro del lado, en [0, 1)
    direction = np.roll(corners, -1, axis=0) - corners
    outward = np.column_stack([direction[:, 1], -direction[:, 0]])  # normal exterior (borde antihorario)
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    offset = 4 * bulge * s * (1 - s)
    return polygon_points(t, corners) + offset[:, None] * outward[side]


BOUNDARY_SHAPES = {
    "circle": lambda t: circle_points(t),
    "square": lambda t: polygon_points(t, square_corners()),
    "star": lambda t: polygon_points(t, star_corners()),
    "bulged_square": lambda t: bulged_square_points(t),   # bonus B1
}


def map_boundary(V, loop, shape="circle", spacing="arc"):
    """Posiciones (nb, 2) en el plano para los vértices del borde. DADO."""
    t = boundary_parameter(V, loop, spacing)
    return BOUNDARY_SHAPES[shape](t)
