# -*- coding: utf-8 -*-
"""
Convierte tablas CSV de resultados/ en tablas LaTeX para el informe:
    batch_<malla>.csv (p2_bordes.py)       -> batch_<malla>.tex (anexo)
    p5_tiempos_lucy.csv (p5_comparacion.py) -> tiempos_lucy.tex  (pregunta 5.4)

    python experimentos/tablas_latex.py
"""
import csv

from comun import MALLAS, ruta

COLUMNAS = ["metodo", "borde", "espaciado", "neg_w", "conf_med", "conf_p95", "area_mean",
            "flips", "degen", "ED/|A|", "tiempo_s"]
TITULOS = ["método", "borde", "espaciado", "neg\\_w", "conf\\_med", "conf\\_p95",
           "area\\_mean", "flips", "degen", "$E_D/|A|$", "tiempo [s]"]
FORMATOS = ["%s", "%s", "%s", "%d", "%.3f", "%.3f", "%.3f", "%d", "%d", "%.4f", "%.3f"]


def celda(valor, formato):
    """Texto de una celda: valor (leído del CSV como texto) con su formato."""
    if formato == "%s":
        return valor
    if formato == "%d":
        return "%d" % int(float(valor))
    return formato % float(valor)


def tabla(nombre):
    with open(ruta("batch_%s.csv" % nombre)) as f:
        filas = list(csv.DictReader(f))
    lineas = ["\\begin{tabular}{lll" + "r" * 8 + "}", "\\toprule",
              " & ".join(TITULOS) + " \\\\", "\\midrule"]
    metodo_anterior = None
    for fila in filas:
        if metodo_anterior is not None and fila["metodo"] != metodo_anterior:
            lineas.append("\\midrule")          # línea entre métodos
        metodo_anterior = fila["metodo"]
        celdas = [celda(fila[col], fmt) for col, fmt in zip(COLUMNAS, FORMATOS)]
        lineas.append(" & ".join(celdas) + " \\\\")
    lineas += ["\\bottomrule", "\\end{tabular}"]
    with open(ruta("batch_%s.tex" % nombre), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print("escrito", ruta("batch_%s.tex" % nombre))


def tabla_tiempos():
    """Tiempos de cada etapa en Lucy; "otras etapas" = Laplaciano + borde/Q + submatrices."""
    with open(ruta("p5_tiempos_lucy.csv")) as f:
        filas = list(csv.DictReader(f))
    lineas = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
              "método & pesos & otras etapas & factorización LU & sustitución & total & incógnitas \\\\",
              "\\midrule"]
    for fila in filas:
        otras = (float(fila["laplaciano"]) + float(fila["borde / Q"])
                 + float(fila["extraer submatrices"]))
        lineas.append("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %s \\\\" % (
            fila["metodo"], float(fila["pesos"]), otras, float(fila["factorizacion LU"]),
            float(fila["sustitucion"]), float(fila["total parameterize"]),
            "{:,}".format(int(fila["incognitas"])).replace(",", "\\,")))
    lineas += ["\\bottomrule", "\\end{tabular}"]
    with open(ruta("tiempos_lucy.tex"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print("escrito", ruta("tiempos_lucy.tex"))


if __name__ == "__main__":
    for nombre in MALLAS:
        tabla(nombre)
    tabla_tiempos()
