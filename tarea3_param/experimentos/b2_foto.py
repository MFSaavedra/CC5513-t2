# -*- coding: utf-8 -*-
"""
Bonus B2 (y pregunta 5.3): una fotografía como textura sobre Nefertiti.

    python experimentos/b2_foto.py

Se usa la foto de Grace Hopper que viene con Matplotlib (sample data). Para
que la cara de la foto quede sobre la cara de la malla se marcan tres puntos
(ojos y boca) en la malla y en la foto, y se calcula la semejanza que lleva
los primeros a los segundos (la misma función de Procrustes de la Parte 3).
Esa semejanza se aplica a todo el (u, v): la foto no se deforma más de lo
que ya deforma la parametrización.
"""
import matplotlib.cbook as cbook
import numpy as np

# comun va primero: agrega la carpeta del código base a sys.path
from comun import VISTAS, cargar, guardar_tabla, plt, polyscope, recortar, ruta, semejanza
from distortion import distortion_report
from param import parameterize
from solvers import conformal_energy

# Ojo izquierdo, ojo derecho y boca: en la malla (coordenadas x, y de la vista
# frontal) y en la foto (píxel: columna, fila)
PUNTOS_MALLA = np.array([[-0.85, 0.65], [0.95, 0.65], [0.0, -1.35]])
PUNTOS_FOTO = np.array([[205.0, 207.0], [275.0, 207.0], [240.0, 278.0]])

# Métodos que se comparan (LSCM no usa la forma del borde: su borde es libre)
OPCIONES = [("lscm", "libre"), ("harmonic", "circle"), ("meanvalue", "square")]


def leer_foto():
    """Foto de Grace Hopper que viene con Matplotlib, como arreglo (alto, ancho, 3)."""
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        return plt.imread(f, format="jpg") / 255.0      # (alto, ancho, 3) en [0, 1]


def coordenadas_de_textura(V, UV, foto):
    """Lleva (u, v) a coordenadas de textura (s, t) en [0, 1] de la foto."""
    alto, ancho = foto.shape[:2]
    # vértice más cercano a cada punto marcado (la cara es casi un gráfico sobre x, y)
    cercanos = [np.argmin(np.linalg.norm(V[:, :2] - p, axis=1)) for p in PUNTOS_MALLA]
    # píxeles con el eje vertical hacia arriba, como v
    destino = np.column_stack([PUNTOS_FOTO[:, 0], alto - PUNTOS_FOTO[:, 1]])
    s, R, c = semejanza(destino, UV[cercanos])
    pixeles = s * UV @ R + c
    return pixeles / [ancho, alto], pixeles


def foto_con_textura(V, F, tex, foto, archivo, direccion):
    """Guarda una imagen de la malla con la foto como textura (coordenadas tex)."""
    ps = polyscope()
    for _ in range(2):      # ver comun.registrar
        ps.remove_all_structures()
        m = ps.register_surface_mesh("cara", V, F)
        m.add_parameterization_quantity("tex", tex, defined_on="vertices")
        m.add_color_quantity("foto", foto, defined_on="texture", param_name="tex",
                             image_origin="upper_left", enabled=True)
    ps.set_view_projection_mode("perspective")
    centro = (V.max(axis=0) + V.min(axis=0)) / 2
    d = np.asarray(direccion, dtype=float)
    d /= np.linalg.norm(d)
    ps.look_at(centro + 2.0 * np.ptp(V, axis=0).max() * d, centro)
    ps.screenshot(archivo, transparent_bg=False)
    recortar(archivo)


if __name__ == "__main__":
    V, F, loop = cargar("nefertiti")
    foto = leer_foto()
    filas = []
    for metodo, borde in OPCIONES:
        UV, _ = parameterize(V, F, metodo, borde, "arc", loop=loop)
        _, r = distortion_report(V, F, UV)
        filas.append([metodo, borde, "%.3f" % r["conf_med"],
                      "%.3f" % r["conf_p95"], "%.3f" % r["area_mean"], r["flips"],
                      "%.4f" % conformal_energy(V, F, UV, loop)["ratio"]])
        tex, pixeles = coordenadas_de_textura(V, UV, foto)
        nombre = "b2_%s_%s" % (metodo, borde)
        foto_con_textura(V, F, tex, foto, ruta(nombre + "_frente.png"), VISTAS["nefertiti"])
        foto_con_textura(V, F, tex, foto, ruta(nombre + "_lado.png"), (1.0, 0.1, 0.8))
        # La foto con la malla (u, v) encima: qué parte de la imagen se usa
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        ax.imshow(foto, extent=(0, foto.shape[1], 0, foto.shape[0]))
        ax.triplot(pixeles[:, 0], pixeles[:, 1], F, color="w", lw=0.3)
        ax.plot(*np.vstack([pixeles[loop], pixeles[loop[:1]]]).T, color="tab:red", lw=0.8)
        ax.set_axis_off()
        fig.savefig(ruta(nombre + "_uv.png"))
        plt.close(fig)
    guardar_tabla("b2_opciones_nefertiti.csv",
                  ["metodo", "borde", "conf_med", "conf_p95", "area_mean", "flips", "ED/|A|"],
                  filas)
