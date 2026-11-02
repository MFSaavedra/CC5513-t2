# -*- coding: utf-8 -*-
"""
Parte 4: análisis de las medidas de distorsión.

    python experimentos/p4_distorsion.py

4.1  J depende del sistema local de cada triángulo, pero sigma1, sigma2 y el
     signo de det J no. Se comprueba rotando (y reflejando) los sistemas
     locales al azar. También se verifica que det J = área uv con signo /
     área 3D.
4.2  Experimentos controlados sobre la malla plana de prueba: se predicen
     conf_med, area_mean y flips de cinco mapas conocidos y se comparan con
     distortion_report. Figura del tablero en cada caso.
"""
import matplotlib.tri as mtri
import numpy as np

# comun va primero: agrega la carpeta del código base a sys.path
from comun import cargar, dibujar_malla_2d, guardar_tabla, plt, ruta, tablero
from distortion import distortion_report, jacobians
from mesh_utils import face_areas, local_triangle_coords, make_plane_grid, signed_areas_2d
from param import parameterize


def jacobianos_con_sistema(Q, F, UV):
    """La misma fórmula de distortion.jacobians, pero con coordenadas locales Q dadas."""
    Qm = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], axis=2)
    Um = np.stack([UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]], axis=2)
    return Um @ np.linalg.inv(Qm)


def pregunta_4_1():
    print("\n== 4.1 Dependencia del sistema local ==")
    V, F, loop = cargar("bunny")
    UV, _ = parameterize(V, F, "lscm", loop=loop)
    Q = local_triangle_coords(V, F)
    J = jacobians(V, F, UV)
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, len(F))
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], 1),
                  np.stack([np.sin(th), np.cos(th)], 1)], 1)      # (m, 2, 2) rotaciones
    reflejo = np.diag([1.0, -1.0])
    filas = []
    for nombre, M in (("rotacion al azar", R), ("rotacion + reflexion", R @ reflejo)):
        Jr = jacobianos_con_sistema(Q @ np.transpose(M, (0, 2, 1)), F, UV)  # q' = M q
        sv = np.linalg.svd(J, compute_uv=False)
        svr = np.linalg.svd(Jr, compute_uv=False)
        mismo_signo = np.mean(np.sign(np.linalg.det(J)) == np.sign(np.linalg.det(Jr)))
        filas.append([nombre, "%.2f" % np.abs(J - Jr).max(), "%.1e" % np.abs(sv - svr).max(),
                      "%.0f%%" % (100 * mismo_signo)])
    guardar_tabla("p4_sistema_local.csv",
                  ["cambio de sistema", "max |J - J'|", "max |sigma - sigma'|",
                   "caras con igual signo de det"], filas)
    razon = signed_areas_2d(UV, F) / face_areas(V, F)
    print("  max |det J - area_uv/area_3D| relativo: %.1e" %
          (np.abs(np.linalg.det(J) - razon).max() / np.abs(razon).max()))


# Los cinco mapas de la pregunta 4.2 (x, y son las coordenadas de la malla plana)
def mapa_a(x, y):  # rotación de 30 grados con escala uniforme 2
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    return 2 * (c * x - s * y), 2 * (s * x + c * y)


def mapa_e(x, y):  # z -> z^2 con z = (x + 1) + i y
    z = (x + 1) + 1j * y
    return (z ** 2).real, (z ** 2).imag


MAPAS = {
    "(a) rotacion+escala": mapa_a,
    "(b) (3x, y)": lambda x, y: (3 * x, y),
    "(c) (x+0.5y, y)": lambda x, y: (x + 0.5 * y, y),
    "(d) (-x, y)": lambda x, y: (-x, y),
    "(e) z^2": mapa_e,
}


TITULOS = ["(a) $2R$", "(b) $(3x, y)$", "(c) cizalle", "(d) $(-x, y)$", "(e) $z^2$"]


def predicciones():
    """Valores esperados (conf_med, area_mean, flips) ANTES de calcular.

    (a) J = 2R: sigma1 = sigma2 = 2 -> conformal 1. El área se multiplica por 4
        en todas partes, pero distortion_report normaliza la escala global,
        así que area_mean = 0.
    (b) J = diag(3, 1): sigma1/sigma2 = 3; área x3 uniforme -> area_mean = 0.
    (c) J = [[1, 0.5], [0, 1]], det = 1: sigma1 sigma2 = 1 y
        sigma1^2 + sigma2^2 = 2.25 -> sigma1/sigma2 = ((0.5 + sqrt(4.25)) / 2)^2.
    (d) J = diag(-1, 1): isometría que invierte TODAS las caras: esperaríamos
        flips = número de caras (392)... salvo que se cuente respecto de la
        orientación dominante.
    (e) f(z) = z^2 es holomorfa y f' = 2z != 0: conforme (sigma1/sigma2 = 1),
        con área local |f'|^2 = 4|z|^2. Tras normalizar, area_mean es la
        desviación media de log2(4|z|^2) en el cuadrado; se integra la
        fórmula continua con una grilla fina (no se usa la malla).
    """
    x, y = np.meshgrid((np.arange(1000) + 0.5) / 1000, (np.arange(1000) + 0.5) / 1000)
    log_area = np.log2(4 * ((x + 1) ** 2 + y ** 2))
    area_e = np.mean(np.abs(log_area - log_area.mean()))
    conf_c = ((0.5 + np.sqrt(4.25)) / 2) ** 2
    return {"(a) rotacion+escala": (1.0, 0.0, 0), "(b) (3x, y)": (3.0, 0.0, 0),
            "(c) (x+0.5y, y)": (conf_c, 0.0, 0), "(d) (-x, y)": (1.0, 0.0, 392),
            "(e) z^2": (1.0, area_e, 0)}


def pregunta_4_2():
    print("\n== 4.2 Experimentos controlados (malla plana de tests.py) ==")
    V, F = make_plane_grid(n=14, jitter=0.35, seed=3)
    x, y = V[:, 0], V[:, 1]
    pred = predicciones()
    filas = []
    fig, axs = plt.subplots(2, 5, figsize=(4.5, 2.0))
    # puntos donde se evalúa la textura (dentro de la malla plana)
    px, py = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
    malla = mtri.Triangulation(x, y, F)
    for k, (nombre, mapa) in enumerate(MAPAS.items()):
        UV = np.column_stack(mapa(x, y))
        _, s = distortion_report(V, F, UV)
        c, a, f = pred[nombre]
        filas.append([nombre, "%.3f" % c, "%.3f" % s["conf_med"], "%.3f" % a,
                      "%.3f" % s["area_mean"], f, s["flips"], "%.3f" % s["conf_p95"]])
        # Textura de tablero: (u, v) de cada punto por interpolación lineal en
        # su triángulo, igual que la textura en el visor (uv normalizado)
        uvn = (UV - UV.min(axis=0)) / (UV - UV.min(axis=0)).max()
        u = mtri.LinearTriInterpolator(malla, uvn[:, 0])(px, py)
        v = mtri.LinearTriInterpolator(malla, uvn[:, 1])(px, py)
        axs[0, k].imshow(tablero(u, v, 0.1), origin="lower", extent=(0, 1, 0, 1),
                         cmap="RdPu", vmin=-0.6, vmax=1.6)
        axs[0, k].set_xticks([])
        axs[0, k].set_yticks([])
        axs[0, k].set_title(TITULOS[k])
        dibujar_malla_2d(axs[1, k], UV, F, lw=0.1)
    axs[0, 0].set_ylabel("tablero")
    axs[1, 0].set_ylabel("(u, v)")
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(ruta("p4_experimentos.png"))
    plt.close(fig)
    guardar_tabla("p4_experimentos.csv",
                  ["mapa", "conf_med pred", "conf_med medido", "area_mean pred",
                   "area_mean medido", "flips pred", "flips medido", "conf_p95 medido"], filas)


if __name__ == "__main__":
    pregunta_4_1()
    pregunta_4_2()
