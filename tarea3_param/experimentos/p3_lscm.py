# -*- coding: utf-8 -*-
"""
Parte 3: análisis de LSCM (borde libre).

    python experimentos/p3_lscm.py

3.1  Qué pasa si el borde se recorre al revés de las caras (K cambia de signo).
3.2  Valores propios más pequeños de Q (Nefertiti y la malla plana de prueba)
     y qué pasa si se fija un solo vértice.
3.3  Dependencia de los pins: primer pin en la posición 0 del borde y el
     segundo recorriendo todo el borde; comparación con los pins automáticos
     después de alinear con Procrustes.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# comun va primero: agrega la carpeta del código base a sys.path
from comun import MALLAS, cargar, guardar_tabla, plt, procrustes, ruta
from distortion import distortion_report
from laplacians import cotangent_weights, laplacian_from_weights
from mesh_utils import boundary_loop, make_plane_grid, signed_areas_2d
from param import parameterize
from solvers import boundary_area_matrix, choose_pins, conformal_energy, lscm


def matriz_Q(V, F, loop):
    """Q = [[L, -K], [K, L]], igual que en solvers.lscm."""
    L = laplacian_from_weights(cotangent_weights(V, F))
    K = boundary_area_matrix(len(V), loop)
    return sp.bmat([[L, -K], [K, L]]).tocsr()


def pregunta_3_1():
    print("\n== 3.1 Borde recorrido al reves de las caras ==")
    filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        p0, p1, _ = choose_pins(V, loop)
        UV = lscm(V, F, loop, pins=(p0, p1))
        UVr = lscm(V, F, loop[::-1], pins=(p0, p1))          # K cambia de signo
        espejo = np.abs(UVr - UV * [1, -1]).max() / np.abs(UV).max()   # ¿UVr = (u, -v)?
        neg = np.mean(signed_areas_2d(UVr, F) < 0)             # caras con orientación opuesta
        # Las energías se miden siempre con el borde en el sentido correcto
        e_normal = conformal_energy(V, F, UV, loop)
        e_invertido = conformal_energy(V, F, UVr, loop)
        _, resumen = distortion_report(V, F, UVr)
        filas.append([nombre, "%.1e" % espejo, "%.0f%%" % (100 * neg),
                      "%.4g" % e_normal["A"], "%.4g" % e_invertido["A"],
                      "%.4g" % e_normal["E_C"], "%.4g" % e_invertido["E_C"],
                      "%.4f" % e_invertido["ratio"], resumen["flips"]])
    guardar_tabla("p3_borde_invertido.csv",
                  ["malla", "|UVr - (u,-v)| rel", "caras con det<0", "A normal", "A invertido",
                   "E_C normal", "E_C invertido", "ED/|A| invertido", "flips (report)"], filas)


def pregunta_3_2():
    print("\n== 3.2 Valores propios de Q ==")
    filas = []
    Vg, Fg = make_plane_grid(n=14, jitter=0.35, seed=3)
    for nombre, (V, F, loop) in (("nefertiti", cargar("nefertiti")),
                                 ("malla plana", (Vg, Fg, boundary_loop(Fg)))):
        n = len(V)
        Q = matriz_Q(V, F, loop)
        lam, vec = np.linalg.eigh(Q.toarray())
        ceros = np.abs(lam) < 1e-10 * np.abs(lam).max()
        # Proyección de los vectores propios nulos sobre las traslaciones (1, 0) y (0, 1)
        T = np.zeros((2 * n, 2))
        T[:n, 0] = 1 / np.sqrt(n)
        T[n:, 1] = 1 / np.sqrt(n)
        nulos = vec[:, ceros]
        fuera = np.linalg.norm(nulos - T @ (T.T @ nulos), axis=0)
        filas.append([nombre, 2 * n, int(ceros.sum())] + ["%.2e" % x for x in lam[:8]] +
                     ["%.1e" % fuera.max()])
        print("  %s: parte de los vectores nulos fuera de las traslaciones: %s" % (nombre, fuera.round(3)))
    guardar_tabla("p3_valores_propios.csv",
                  ["malla", "tamano Q", "ceros"] + ["lambda_%d" % k for k in range(8)] +
                  ["vec nulo fuera de traslaciones"], filas)

    print("\n== 3.2 Un solo pin (Nefertiti) ==")
    V, F, loop = cargar("nefertiti")
    n = len(V)
    Q = matriz_Q(V, F, loop)
    p0 = int(loop[0])
    fijas = np.array([p0, p0 + n])
    xp = np.array([0.3, -0.7])                  # el pin va a (0.3, -0.7)
    libres = np.setdiff1d(np.arange(2 * n), fijas)
    x = np.zeros(2 * n)
    x[fijas] = xp
    x[libres] = spla.spsolve(Q[libres][:, libres].tocsc(), -(Q[libres][:, fijas] @ xp))
    UV = np.column_stack([x[:n], x[n:]])
    print("  todos los vertices en el pin: max |uv - (0.3, -0.7)| = %.1e" % np.abs(UV - xp).max())


def pregunta_3_3():
    print("\n== 3.3 Dependencia de los pins ==")
    resumen_filas = []
    for nombre in MALLAS:
        V, F, loop = cargar(nombre)
        nb = len(loop)
        UV_auto, _ = parameterize(V, F, "lscm", loop=loop)
        r_auto = conformal_energy(V, F, UV_auto, loop)["ratio"]
        p0, p1, _ = choose_pins(V, loop)
        pos = {int(v): k for k, v in enumerate(loop)}
        filas = []
        for k in range(1, nb):
            UV, _ = parameterize(V, F, "lscm", loop=loop, pins=(0, k))
            _, s = distortion_report(V, F, UV)
            ratio = conformal_energy(V, F, UV, loop)["ratio"]
            _, error = procrustes(UV_auto, UV)
            filas.append([k / nb, ratio, s["conf_med"], s["area_mean"], error, s["flips"]])
        filas = np.array(filas)
        np.savetxt(ruta("p3_pins_%s.csv" % nombre), filas, delimiter=",", fmt="%.6g",
                   header="posicion/nb,ED/|A|,conf_med,area_mean,error_procrustes,flips")
        medio = np.argmin(np.abs(filas[:, 0] - 0.5))
        resumen_filas.append([nombre, nb, "%d,%d" % (pos[p0], pos[p1]),
                              "%.4f" % r_auto, "%.4f-%.4f" % (filas[:, 1].min(), filas[:, 1].max()),
                              "%.3f-%.3f" % (filas[:, 2].min(), filas[:, 2].max()),
                              "%.2f-%.2f" % (filas[:, 3].min(), filas[:, 3].max()),
                              "%.1e" % filas[:, 4].min(), "%.2f" % filas[0, 4],
                              "%.3f" % filas[medio, 4], int(filas[:, 5].max())])
    guardar_tabla("p3_pins_resumen.csv",
                  ["malla", "nb", "pins auto (pos)", "ED/|A| auto", "ED/|A| rango", "conf_med rango",
                   "area_mean rango", "error min", "error k=1", "error k=nb/2", "max flips"],
                  resumen_filas)


def figura_pins():
    """Gráfico de la pregunta 3.3, a partir de los CSV que guarda pregunta_3_3.

    Líneas segmentadas: el valor con los pins automáticos.
    """
    fig, axs = plt.subplots(2, 2, figsize=(4.2, 2.7), sharex=True)
    axs = axs.ravel()
    for nombre, color in zip(MALLAS, ["tab:blue", "tab:orange", "tab:green"]):
        filas = np.loadtxt(ruta("p3_pins_%s.csv" % nombre), delimiter=",")
        V, F, loop = cargar(nombre)
        UV_auto, _ = parameterize(V, F, "lscm", loop=loop)
        _, s_auto = distortion_report(V, F, UV_auto)
        referencia = [conformal_energy(V, F, UV_auto, loop)["ratio"], s_auto["conf_med"],
                      s_auto["area_mean"], None]
        for ax, j, ref in zip(axs, range(1, 5), referencia):
            ax.plot(filas[:, 0], filas[:, j], color=color, lw=0.9, label=nombre)
            if ref is not None:
                ax.axhline(ref, color=color, ls="--", lw=0.6)
    for ax, titulo in zip(axs, ["$E_D/|A|$", "conf_med", "area_mean", "error de Procrustes"]):
        ax.set_title(titulo)
    for ax in axs[2:]:
        ax.set_xlabel("posición del 2º pin / $n_b$")
    axs[3].set_yscale("log")
    axs[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(ruta("p3_pins.png"))
    plt.close(fig)


def figura_pins_bunny():
    """Dominio (u, v) de Bunny con los pins automáticos y con dos pins vecinos.

    Los resultados se alinean con Procrustes al de los pins automáticos y se
    colorean por distorsión conforme (sigma1 / sigma2).
    """
    V, F, loop = cargar("bunny")
    UV_auto, _ = parameterize(V, F, "lscm", loop=loop)
    p0, p1, _ = choose_pins(V, loop)
    fig, axs = plt.subplots(1, 2, figsize=(2.8, 1.4))
    casos = [(UV_auto, [p0, p1], "pins automáticos"),
             (parameterize(V, F, "lscm", loop=loop, pins=(0, 1))[0], loop[[0, 1]],
              "pins vecinos (0 y 1)")]
    for ax, (UV, pins, titulo) in zip(axs, casos):
        UV, error = procrustes(UV_auto, UV)
        cara, _ = distortion_report(V, F, UV)
        tp = ax.tripcolor(UV[:, 0], UV[:, 1], F, facecolors=np.minimum(cara["conformal"], 2.0),
                          cmap="Reds", vmin=1, vmax=2, edgecolors="0.6", linewidth=0.05)
        ax.plot(*np.vstack([UV[loop], UV[loop[:1]]]).T, color="k", lw=0.3)
        ax.plot(UV[pins, 0], UV[pins, 1], "o", color="tab:blue", ms=2.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("%s\nerror = %.2f" % (titulo, error))
    fig.colorbar(tp, ax=axs, shrink=0.8, label="$\\sigma_1/\\sigma_2$")
    fig.savefig(ruta("p3_pins_bunny.png"))
    plt.close(fig)


if __name__ == "__main__":
    pregunta_3_1()
    pregunta_3_2()
    pregunta_3_3()
    figura_pins()
    figura_pins_bunny()
