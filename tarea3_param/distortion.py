# -*- coding: utf-8 -*-
"""
Medidas de distorsión de una parametrización.

Para cada triángulo t, la parametrización es una función afín del triángulo
3D (en coordenadas locales 2D, ver local_triangle_coords) al triángulo uv.
Su parte lineal es una matriz J_t de 2 x 2 (el Jacobiano). Sus valores
singulares sigma1 >= sigma2 >= 0 indican cuánto se estira el triángulo en
la dirección de máximo y mínimo estiramiento:

    sigma1 / sigma2      distorsión conforme (ángulos). Vale 1 si el
                         triángulo solo se rota y escala.
    log2(sigma1*sigma2)  distorsión de área. Vale 0 si conserva el área,
                         +1 si la duplica, -1 si la reduce a la mitad.
    det(J) < 0           el triángulo quedó invertido ("flip").

Como la escala global de uv es arbitraria (círculo de radio 1, LSCM con
pins...), antes de medir el área se reescala uv de modo que un triángulo
"típico" conserve su área (ver distortion_report).

Este archivo está COMPLETO. En la Parte 4 debes leerlo, explicar cada
decisión (sistema local, normalización de escala, estadísticas robustas,
umbral de degenerados) y analizar cómo se comportan las medidas.
"""
import numpy as np

from mesh_utils import face_areas, local_triangle_coords


def jacobians(V, F, UV):
    """Jacobiano (m, 2, 2) del mapa afín triángulo 3D -> triángulo uv.

    Sea Q = local_triangle_coords(V, F) (q0 = (0,0), q1, q2 en 2D) y sean
    u0, u1, u2 las coordenadas uv de los vértices del triángulo. J cumple
        J @ [q1 - q0 | q2 - q0] = [u1 - u0 | u2 - u0]
    (las barras separan columnas de matrices 2 x 2), es decir
        J = [u1 - u0 | u2 - u0] @ inv([q1 - q0 | q2 - q0]).
    Ojo: J depende del sistema local elegido en cada triángulo (una
    rotación), pero sus valores singulares no.
    """
    Q = local_triangle_coords(V, F)
    Qm = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], axis=2)       # (m,2,2)
    Um = np.stack([UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]], axis=2)
    return Um @ np.linalg.inv(Qm)


def face_distortion(J):
    """A partir de los Jacobianos (m, 2, 2) devuelve tres arreglos de largo m:

        conformal : sigma1 / sigma2   (usar np.inf si sigma2 es ~0)
        area      : log2(sigma1 * sigma2)   (= log2 |det J|)
        det       : det(J) con signo
    """
    s = np.linalg.svd(J, compute_uv=False)                 # (m, 2), s1 >= s2
    det = np.linalg.det(J)
    with np.errstate(divide="ignore", invalid="ignore"):
        conformal = np.where(s[:, 1] > 1e-14, s[:, 0] / s[:, 1], np.inf)
        area = np.log2(np.maximum(s[:, 0] * s[:, 1], 1e-300))
    return conformal, area, det


def distortion_report(V, F, UV, eps=1e-12):
    """Calcula las medidas por cara y un resumen. DADO.

    Escala global: como uv tiene una escala arbitraria, se reescala para que
    la media (ponderada por área 3D) de log2(sigma1 sigma2) sea 0; es decir,
    un triángulo "típico" conserva su área.

    Devuelve (per_face, summary):
        per_face["conformal"]  sigma1/sigma2 (1e12 si el triángulo es degenerado)
        per_face["area"]       log2(sigma1 sigma2) tras normalizar la escala
        per_face["flipped"]    True si el triángulo quedó invertido o degenerado
        summary["conf_med"]    mediana de sigma1/sigma2
        summary["conf_p95"]    percentil 95 de sigma1/sigma2
        summary["area_mean"]   media de |log2(sigma1 sigma2)| ponderada por área
        summary["flips"]       triángulos invertidos (det con signo contrario)
        summary["degen"]       triángulos degenerados: área uv < eps * mediana
                               (colineales o colapsados numéricamente).
                               Las estadísticas conf_* y area_mean se
                               calculan sin ellos.
    """
    A = face_areas(V, F)
    J = jacobians(V, F, UV)
    det0 = np.abs(np.linalg.det(J))
    ok = det0 > eps * np.median(det0)
    s = 2.0 ** (-0.5 * np.sum(A[ok] * np.log2(det0[ok])) / A[ok].sum()) if ok.any() else 1.0
    conformal, area, det = face_distortion(J * s)
    degen = ~ok
    orient = np.sign(np.sum(det * A)) or 1.0
    flipped = (det * orient < 0) & ~degen
    conformal = np.where(np.isfinite(conformal) & ~degen, conformal, 1e12)
    good = ~degen if (~degen).any() else np.ones(len(F), bool)
    w = A[good] / A[good].sum()
    summary = {
        "conf_med": float(np.median(conformal[good])),
        "conf_p95": float(np.percentile(conformal[good], 95)),
        "area_mean": float(np.sum(w * np.abs(area[good]))),
        "flips": int(flipped.sum()),
        "degen": int(degen.sum()),
    }
    per_face = {"conformal": conformal, "area": np.where(degen, 0.0, area),
                "flipped": flipped | degen}
    return per_face, summary
