# -*- coding: utf-8 -*-
"""
Punto de entrada único para calcular una parametrización. DADO.

    UV, info = parameterize(V, F, method="harmonic", shape="circle", spacing="arc")

method  : "uniform" | "harmonic" | "meanvalue" | "lscm"
shape   : "circle" | "square" | "star"      (ignorado por lscm)
spacing : "arc" | "uniform"                 (ignorado por lscm)
"""
import time

import numpy as np

from boundary import map_boundary
from laplacians import WEIGHTS, count_negative_weights, laplacian_from_weights
from mesh_utils import boundary_loop
from solvers import lscm, solve_fixed_boundary

METHODS = ["uniform", "harmonic", "meanvalue", "lscm"]
SHAPES = ["circle", "square", "star"]
SPACINGS = ["arc", "uniform"]


def parameterize(V, F, method="harmonic", shape="circle", spacing="arc", loop=None,
                 pins=None):
    """pins (solo lscm): None = automáticos, o un par (a, b) de POSICIONES en
    el borde (0 <= a, b < |borde|), no índices de vértice."""
    if loop is None:
        loop = boundary_loop(F)
    t0 = time.perf_counter()
    info = {"method": method, "shape": shape, "spacing": spacing,
            "n_boundary": len(loop), "neg_weights": 0}
    if method == "lscm":
        if pins is None:
            UV = lscm(V, F, loop)
            info["pins"] = "auto"
        else:
            a, b = (int(p) % len(loop) for p in pins)
            if a == b:
                raise ValueError("Los dos pins deben ser distintos")
            UV = lscm(V, F, loop, pins=(int(loop[a]), int(loop[b])))
            info["pins"] = "%d,%d" % (a, b)
        info["shape"] = "-"
        info["spacing"] = "-"
    elif method in WEIGHTS:
        W = WEIGHTS[method](V, F)
        info["neg_weights"] = count_negative_weights(W)
        L = laplacian_from_weights(W)
        uv_b = map_boundary(V, loop, shape, spacing)
        UV = solve_fixed_boundary(L, loop, uv_b)
    else:
        raise ValueError("Metodo desconocido: %s" % method)
    info["time"] = time.perf_counter() - t0
    if not np.all(np.isfinite(UV)):
        raise FloatingPointError("La parametrizacion tiene valores NaN/inf")
    return UV, info
