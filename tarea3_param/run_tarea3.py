# -*- coding: utf-8 -*-
"""
CC5513 - Tarea 3: Parametrización de superficies.

Ejemplos de uso
---------------
Visor interactivo (cambia método/borde desde el panel):
    python run_tarea3.py --mesh data/nefertiti.off

Elegir opciones desde la línea de comandos:
    python run_tarea3.py --mesh data/bunny.off --method meanvalue --boundary square

Tabla de métricas para TODAS las combinaciones (para el informe):
    python run_tarea3.py --mesh data/lucy.off --batch --csv lucy.csv

LSCM fijando las posiciones 0 y 10 del borde:
    python run_tarea3.py --mesh data/bunny.off --method lscm --pins 0 10

Capturas sin abrir ventana (genera fig_3d.png y fig_uv.png):
    python run_tarea3.py --mesh data/bunny.off --method lscm --color conformal --shot fig

Guardar la malla plana (u, v, 0) en OFF:
    python run_tarea3.py --mesh data/bunny.off --save-uv bunny_uv.off --no-view

Pruebas automáticas de tu implementación:
    python tests.py
"""
import argparse
import csv
import os
import sys

from distortion import distortion_report
from solvers import conformal_energy
from mesh_io import read_off, write_off
from mesh_utils import boundary_loop, euler_characteristic
from param import METHODS, SHAPES, SPACINGS, parameterize
from viewer import COLOR_MODES, ParamViewer

HERE = os.path.dirname(os.path.abspath(__file__))


def run_batch(V, F, name, csv_path=None):
    loop = boundary_loop(F)
    rows = []
    combos = [(m, s, sp) for m in METHODS if m != "lscm"
              for s in SHAPES for sp in SPACINGS] + [("lscm", "-", "-")]
    header = ("metodo", "borde", "espaciado", "neg_w", "conf_med",
              "conf_p95", "area_mean", "flips", "degen", "ED/|A|", "tiempo_s")
    print("\nMalla %s: |V|=%d |F|=%d |borde|=%d" % (name, len(V), len(F), len(loop)))
    print("%-10s %-7s %-8s %6s %9s %9s %9s %6s %6s %8s %8s" % header)
    for m, s, sp in combos:
        try:
            UV, info = parameterize(V, F, m, s if s != "-" else "circle",
                                    sp if sp != "-" else "arc", loop=loop)
            _, summ = distortion_report(V, F, UV)
            try:
                ratio = conformal_energy(V, F, UV, loop)["ratio"]
            except NotImplementedError:
                ratio = float("nan")  # requiere la Parte 1a
        except NotImplementedError as e:
            print("%-10s %-7s %-8s  PENDIENTE: %s" % (m, s, sp, e))
            continue
        row = (m, s, sp, info["neg_weights"], summ["conf_med"], summ["conf_p95"],
               summ["area_mean"], summ["flips"], summ["degen"], ratio, info["time"])
        rows.append(row)
        print("%-10s %-7s %-8s %6d %9.3f %9.3f %9.3f %6d %6d %8.4f %8.3f" % row)
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("CSV guardado en", csv_path)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tarea 3 - Parametrizacion")
    ap.add_argument("--mesh", default=os.path.join(HERE, "data", "nefertiti.off"))
    ap.add_argument("--method", default="uniform", choices=METHODS)
    ap.add_argument("--boundary", default="circle", choices=SHAPES)
    ap.add_argument("--spacing", default="arc", choices=SPACINGS)
    ap.add_argument("--pins", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="lscm: posiciones en el borde de los dos vertices fijos")
    ap.add_argument("--color", default="checker", choices=COLOR_MODES)
    ap.add_argument("--checker", type=float, default=0.05,
                    help="tamano del tablero en coordenadas uv normalizadas")
    ap.add_argument("--batch", action="store_true",
                    help="imprime metricas de todas las combinaciones")
    ap.add_argument("--csv", default=None, help="con --batch: guarda la tabla en CSV")
    ap.add_argument("--shot", default=None, metavar="PREFIJO",
                    help="guarda PREFIJO_3d.png y PREFIJO_uv.png")
    ap.add_argument("--save-uv", default=None, help="guarda la malla (u,v,0) en OFF")
    ap.add_argument("--no-view", action="store_true", help="no abrir la ventana")
    args = ap.parse_args(argv)

    V, F = read_off(args.mesh)
    name = os.path.splitext(os.path.basename(args.mesh))[0]
    chi = euler_characteristic(V, F)
    if chi != 1:
        print("[aviso] caracteristica de Euler = %d (un disco tiene 1)" % chi)

    if args.batch:
        run_batch(V, F, name, args.csv)
        if args.shot is None and not args.save_uv:
            return 0

    viewer = ParamViewer(V, F, name, args.method, args.boundary, args.spacing,
                         args.color, args.checker, pins=args.pins)
    viewer.compute()
    print(viewer.metrics_text())

    if args.save_uv and viewer.UV is not None:
        write_off(args.save_uv, viewer.UV, F)
        print("Malla plana guardada en", args.save_uv)
    if args.shot:
        for fn in viewer.screenshot(args.shot):
            print("Captura:", fn)
    if not args.no_view and not args.shot:
        viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
