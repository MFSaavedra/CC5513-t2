# -*- coding: utf-8 -*-
"""
Pruebas automáticas para la Tarea 3.   Uso:  python tests.py

Cada prueba imprime OK, FALLA o PENDIENTE (si la función todavía lanza
NotImplementedError). Que todas pasen NO garantiza que tu implementación
sea correcta, pero que alguna falle sí indica un error.

La prueba principal es la de REPRODUCCIÓN LINEAL: si la malla ya es plana
y fijamos el borde en su posición original, un método con "precisión
lineal" debe devolver exactamente las coordenadas originales de los
vértices interiores. Los pesos cotangentes y de valor medio la tienen;
los uniformes no.

Las pruebas marcadas "(dado)" verifican código que ya se entrega completo
(LSCM y medidas de distorsión); las de LSCM usan tus pesos cotangentes, así
que pasan una vez completada la Parte 1a.
"""
import os
import sys

import numpy as np

from boundary import polygon_points, square_corners, star_corners
from distortion import distortion_report, jacobians
from laplacians import (WEIGHTS, count_negative_weights, laplacian_from_weights)
from mesh_io import read_off
from mesh_utils import boundary_loop, make_plane_grid
from solvers import lscm, solve_fixed_boundary

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, fn):
    try:
        ok, msg = fn()
        tag = "OK" if ok else "FALLA"
    except NotImplementedError as e:
        tag, msg = "PENDIENTE", str(e)
    except Exception as e:  # noqa
        tag, msg = "FALLA", "%s: %s" % (type(e).__name__, e)
    results.append(tag)
    print("[%-9s] %s  %s" % (tag, name, msg))


# ---------------------------------------------------------------------------
Vg, Fg = make_plane_grid(n=14, jitter=0.35, seed=3)
loop_g = boundary_loop(Fg)


def reproduction_error(method):
    W = WEIGHTS[method](Vg, Fg)
    L = laplacian_from_weights(W)
    UV = solve_fixed_boundary(L, loop_g, Vg[loop_g, :2])
    return np.abs(UV - Vg[:, :2]).max(), W


def t_uniform_not_linear():
    err, _ = reproduction_error("uniform")
    return err > 1e-4, "error uniforme = %.2e (debe ser > 0: Tutte no tiene precision lineal)" % err


def t_cot_symmetric():
    W = WEIGHTS["harmonic"](Vg, Fg)
    asym = abs(W - W.T).max()
    # Triángulo rectángulo isósceles: pesos conocidos
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2]])
    W1 = WEIGHTS["harmonic"](V, F).toarray()
    # ángulo en 2 es 45 grados -> D_01 = 1/2 cot(45) = 0.5 ; en 0 es 90 -> D_12 = 0
    ok = asym < 1e-12 and abs(W1[0, 1] - 0.5) < 1e-9 and abs(W1[1, 2]) < 1e-9
    return ok, "asimetria = %.1e, D01 = %.3f (esperado 0.5), D12 = %.3f (esperado 0)" % (
        asym, W1[0, 1], W1[1, 2])


def t_cot_linear():
    err, W = reproduction_error("harmonic")
    return err < 1e-9, "error = %.2e, pesos negativos en la grilla = %d" % (
        err, count_negative_weights(W))


def t_mv_positive():
    W = WEIGHTS["meanvalue"](Vg, Fg)
    asym = abs(W - W.T).max()
    return (W.data.min() > 0 and asym > 1e-6), \
        "min peso = %.3f (debe ser > 0), asimetria = %.2e (debe ser > 0)" % (W.data.min(), asym)


def t_mv_linear():
    err, _ = reproduction_error("meanvalue")
    return err < 1e-9, "error = %.2e" % err


def t_polygon():
    sq = square_corners()
    t = np.array([0.0, 0.125, 0.25, 0.5, 0.875])
    P = polygon_points(t, sq)
    exp = np.array([[1, -1], [1, 0], [1, 1], [-1, 1], [0, -1]], float)
    e1 = np.abs(P - exp).max()
    st = star_corners()
    k = len(st)
    lens = np.linalg.norm(np.roll(st, -1, 0) - st, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])[:-1] / lens.sum()
    e2 = np.abs(polygon_points(cum, st) - st).max()
    return e1 < 1e-12 and e2 < 1e-12, "error cuadrado = %.1e, error esquinas estrella (%d) = %.1e" % (e1, k, e2)


def t_jacobian():
    # Una rotación+escala conocida aplicada a la grilla plana
    th, s = 0.7, 2.5
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    J = jacobians(Vg, Fg, Vg[:, :2] @ R.T)
    # J depende del sistema local de cada triángulo (una rotación), pero sus
    # valores singulares no: ambos deben ser s, y det(J) = s^2 > 0.
    sv = np.linalg.svd(J, compute_uv=False)
    err = max(np.abs(sv - s).max(), np.abs(np.linalg.det(J) - s * s).max())
    # Un estiramiento anisótropo: sigma1/sigma2 = 3, sin flips
    S = np.diag([3.0, 1.0])
    _, summ = distortion_report(Vg, Fg, Vg[:, :2] @ S.T)
    # Un reflejo de un solo vértice interior que invierte triángulos
    uv = Vg[:, :2].copy()
    inner = [i for i in range(len(Vg)) if i not in set(loop_g)][40]
    uv[inner] += np.array([0.3, 0.3])
    _, summ2 = distortion_report(Vg, Fg, uv)
    ok = err < 1e-9 and abs(summ["conf_med"] - 3) < 1e-9 and summ["flips"] == 0 \
        and summ2["flips"] > 0
    return ok, "error J = %.1e, sigma1/sigma2 = %.3f (esperado 3), flips = %d / %d (esperado 0 / >0)" % (
        err, summ["conf_med"], summ["flips"], summ2["flips"])


def t_lscm_plane():
    UV = lscm(Vg, Fg, loop_g)
    # LSCM reproduce una malla plana salvo semejanza: alinear con Procrustes
    X = Vg[:, :2] - Vg[:, :2].mean(0)
    Y = UV - UV.mean(0)
    U, S, Vt = np.linalg.svd(Y.T @ X)
    R = U @ Vt
    sc = S.sum() / (Y ** 2).sum()
    err = np.abs(sc * Y @ R - X).max()
    _, summ = distortion_report(Vg, Fg, UV)
    return err < 1e-8 and summ["flips"] == 0, \
        "error (salvo semejanza) = %.2e, sigma1/sigma2 mediana = %.6f" % (err, summ["conf_med"])


def t_lscm_nefertiti():
    V, F = read_off(os.path.join(HERE, "data", "nefertiti.off"))
    loop = boundary_loop(F)
    UV = lscm(V, F, loop)
    _, summ = distortion_report(V, F, UV)
    _, summ_c = distortion_report(V, F, solve_fixed_boundary(
        laplacian_from_weights(WEIGHTS["uniform"](V, F)), loop,
        np.column_stack([np.cos(np.linspace(0, 2 * np.pi, len(loop), endpoint=False)),
                         np.sin(np.linspace(0, 2 * np.pi, len(loop), endpoint=False))])))
    ok = summ["flips"] == 0 and summ["conf_med"] < summ_c["conf_med"]
    return ok, "LSCM sigma1/sigma2 = %.3f vs Tutte+circulo = %.3f (LSCM debe ser menor), flips = %d" % (
        summ["conf_med"], summ_c["conf_med"], summ["flips"])


if __name__ == "__main__":
    print("Tarea 3 - pruebas\n")
    check("1a  cotangente: simetria y valores", t_cot_symmetric)
    check("1a  cotangente: reproduccion lineal", t_cot_linear)
    check("1b  mean value: pesos positivos, no simetricos", t_mv_positive)
    check("1b  mean value: reproduccion lineal", t_mv_linear)
    check("--  uniforme NO reproduce (control)", t_uniform_not_linear)
    check("2   polygon_points", t_polygon)
    check("--  (dado) jacobianos y metricas", t_jacobian)
    check("--  (dado) LSCM reproduce malla plana [usa 1a]", t_lscm_plane)
    check("--  (dado) LSCM en Nefertiti [usa 1a]", t_lscm_nefertiti)
    n_ok = results.count("OK")
    print("\n%d / %d pruebas OK" % (n_ok, len(results)))
    sys.exit(0 if n_ok == len(results) else 1)
