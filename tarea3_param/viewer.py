# -*- coding: utf-8 -*-
"""
Visualización interactiva con Polyscope. DADO (no necesitas modificarlo).

Se muestran dos objetos:
    "superficie" : la malla 3D con la parametrización aplicada como textura
    "plano"      : la malla dibujada en coordenadas (u, v, 0), al lado

Panel de control (ventana "Tarea 3"):
    - método, forma del borde y espaciado
    - qué se colorea: textura de tablero, distorsión conforme, de área, flips
    - tamaño del tablero
    - métricas de la parametrización actual

Nota: los textos del panel están sin tildes a propósito; versiones antiguas
de Polyscope no dibujan bien caracteres fuera de ASCII.
"""
import traceback

import numpy as np
import polyscope as ps

try:
    import polyscope.imgui as psim
except ImportError:  # versiones muy antiguas
    psim = None

from distortion import distortion_report
from solvers import choose_pins, conformal_energy
from mesh_utils import boundary_loop
from param import METHODS, SHAPES, SPACINGS, parameterize

COLOR_MODES = ["checker", "conformal", "area", "flips"]  # flips incluye degenerados


# ---------------------------------------------------------------------------
# Envoltorios tolerantes a la versión de Polyscope
# ---------------------------------------------------------------------------
def _add_param(mesh, name, uv, checker_size, enabled):
    try:
        mesh.add_parameterization_quantity(name, uv, defined_on="vertices",
                                           viz_style="checker",
                                           checker_size=checker_size,
                                           enabled=enabled)
    except TypeError:
        mesh.add_parameterization_quantity(name, uv, defined_on="vertices",
                                           enabled=enabled)


def _add_face_scalar(mesh, name, vals, cmap, vminmax, enabled):
    try:
        mesh.add_scalar_quantity(name, vals, defined_on="faces", cmap=cmap,
                                 vminmax=vminmax, enabled=enabled)
    except TypeError:
        mesh.add_scalar_quantity(name, vals, defined_on="faces", enabled=enabled)


def _try(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception:
        pass


# ---------------------------------------------------------------------------
class ParamViewer:
    def __init__(self, V, F, name="malla", method="harmonic", shape="circle",
                 spacing="arc", color="checker", checker=0.05, pins=None):
        self.V, self.F, self.name = V, F, name
        self.loop = boundary_loop(F)
        self.method, self.shape, self.spacing = method, shape, spacing
        self.color = color
        self.checker = checker
        nb = len(self.loop)
        # pins de LSCM como posiciones en el borde; None = automáticos
        self.pins_auto = pins is None
        if pins is None:
            p0, p1, _ = choose_pins(V, self.loop)
            pos = {int(v): k for k, v in enumerate(self.loop)}
            pins = (pos[p0], pos[p1])
        self.pins = [int(pins[0]) % nb, int(pins[1]) % nb]
        self.energy = None
        self.UV = None
        self.info = {}
        self.summary = None
        self.per_face = None
        self.status = ""
        ext = V.max(axis=0) - V.min(axis=0)
        self.size = float(ext.max())
        self.offset = np.array([V.max(axis=0)[0] + 0.25 * self.size,
                                V.min(axis=0)[1], V.mean(axis=0)[2]])

    # ------------------------------------------------------------------
    def compute(self):
        """Calcula la parametrización y las métricas con las opciones actuales."""
        try:
            UV, info = parameterize(self.V, self.F, self.method, self.shape,
                                    self.spacing, loop=self.loop,
                                    pins=None if self.pins_auto else self.pins)
        except NotImplementedError as e:
            self.status = "PENDIENTE: %s" % e
            print("[tarea3] " + self.status)
            return False
        except Exception as e:
            self.status = "ERROR: %s" % e
            traceback.print_exc()
            return False
        self.UV, self.info = UV, info
        self.status = "OK (%.2f s)" % info["time"]
        try:
            self.per_face, self.summary = distortion_report(self.V, self.F, UV)
        except NotImplementedError as e:
            self.per_face, self.summary = None, None
            self.status += " | metricas: PENDIENTE (%s)" % e
        try:
            self.energy = conformal_energy(self.V, self.F, UV, self.loop)
        except NotImplementedError:
            self.energy = None  # requiere los pesos cotangentes (Parte 1a)
        return True

    def flat_vertices(self):
        """Coordenadas 3D de la malla plana: uv escalado y desplazado."""
        uv = self.UV - self.UV.min(axis=0)
        s = self.size / max(uv.max(), 1e-12)
        P = np.zeros((len(uv), 3))
        P[:, 0] = uv[:, 0] * s
        P[:, 1] = uv[:, 1] * s
        return P + self.offset

    # ------------------------------------------------------------------
    def register(self, show_surface=True, show_flat=True):
        if self.UV is None:
            m = ps.register_surface_mesh("superficie", self.V, self.F)
            _try(m.set_edge_width, 0.0)
            return
        structs = []
        if show_surface:
            structs.append(ps.register_surface_mesh("superficie", self.V, self.F))
        if show_flat:
            flat = ps.register_surface_mesh("plano", self.flat_vertices(), self.F)
            _try(flat.set_edge_width, 0.6)
            _try(flat.set_color, (0.93, 0.93, 0.95))
            structs.append(flat)
            loop = np.append(self.loop, self.loop[0])
            P = self.flat_vertices()[loop]
            E = np.column_stack([np.arange(len(loop) - 1), np.arange(1, len(loop))])
            c = ps.register_curve_network("borde", P, E)
            _try(c.set_radius, 0.002, relative=True)
            _try(c.set_color, (0.85, 0.1, 0.1))
            if self.method == "lscm":
                pv = self.flat_vertices()[self.loop[self.pins]]
                pc = ps.register_point_cloud("pins", pv)
                _try(pc.set_radius, 0.012, relative=True)
                _try(pc.set_color, (0.1, 0.35, 0.9))
        if show_surface and self.method == "lscm":
            pc3 = ps.register_point_cloud("pins 3D", self.V[self.loop[self.pins]])
            _try(pc3.set_radius, 0.012, relative=True)
            _try(pc3.set_color, (0.1, 0.35, 0.9))
        uvn = self.UV - self.UV.min(axis=0)
        uvn = uvn / max(uvn.max(), 1e-12)
        for m in structs:
            # En el plano el tablero no aporta (siempre se ve regular): en modo
            # "checker" se muestra la malla con aristas para ver la densidad.
            is_flat = show_flat and m is structs[-1]
            _add_param(m, "uv", uvn, self.checker,
                       self.color == "checker" and not is_flat)
            if self.per_face is not None:
                conf = self.per_face["conformal"]
                hi = max(1.5, float(np.percentile(conf, 95)))
                _add_face_scalar(m, "conformal", conf, "reds", (1.0, hi),
                                 self.color == "conformal")
                ar = self.per_face["area"]
                lim = max(0.5, float(np.percentile(np.abs(ar), 95)))
                _add_face_scalar(m, "area", ar, "coolwarm", (-lim, lim),
                                 self.color == "area")
                _add_face_scalar(m, "flips", self.per_face["flipped"].astype(float),
                                 "reds", (0.0, 1.0), self.color == "flips")

    # ------------------------------------------------------------------
    def metrics_text(self):
        lines = ["malla: %s  |V|=%d  |F|=%d  |borde|=%d" %
                 (self.name, len(self.V), len(self.F), len(self.loop))]
        if self.info:
            lines.append("metodo: %s  borde: %s  espaciado: %s" %
                         (self.info["method"], self.info["shape"], self.info["spacing"]))
            lines.append("pesos negativos: %d" % self.info["neg_weights"])
            if self.info["method"] == "lscm":
                lines.append("pins (posiciones en el borde): %s" % self.info.get("pins", "auto"))
        if self.summary:
            s = self.summary
            lines.append("sigma1/sigma2  mediana: %.3f   p95: %.3f" % (s["conf_med"], s["conf_p95"]))
            lines.append("|log2 area|    media: %.3f" % s["area_mean"])
            lines.append("triangulos invertidos: %d   degenerados: %d" % (s["flips"], s["degen"]))
        if self.energy:
            lines.append("E_D/|A|: %.4f   (1 = conforme)" % self.energy["ratio"])
        lines.append(self.status)
        return "\n".join(lines)

    def callback(self):
        if psim is None:
            return
        changed = False
        psim.PushItemWidth(160)
        for label, attr, options in (("metodo", "method", METHODS),
                                     ("borde", "shape", SHAPES),
                                     ("espaciado", "spacing", SPACINGS)):
            idx = options.index(getattr(self, attr))
            ch, idx = psim.Combo(label, idx, options)
            if ch:
                setattr(self, attr, options[idx])
                changed = True
        if self.method == "lscm":
            ch, self.pins_auto = psim.Checkbox("pins automaticos", self.pins_auto)
            changed |= ch
            if not self.pins_auto:
                nb = len(self.loop)
                for k in range(2):
                    ch, val = psim.SliderInt("pin %d (borde)" % k, self.pins[k], 0, nb - 1)
                    if ch and val != self.pins[1 - k]:
                        self.pins[k] = val
                        changed = True
        psim.Separator()
        idx = COLOR_MODES.index(self.color)
        ch, idx = psim.Combo("colorear", idx, COLOR_MODES)
        redraw = ch
        if ch:
            self.color = COLOR_MODES[idx]
        ch, val = psim.SliderFloat("tablero", self.checker, 0.005, 0.2)
        if ch:
            self.checker = val
            redraw = True
        psim.PopItemWidth()
        if psim.Button("Recalcular"):
            changed = True
        psim.Separator()
        psim.TextUnformatted(self.metrics_text())
        if changed:
            self.compute()
            ps.remove_all_structures()
            self.register()
        elif redraw:
            self.register()

    # ------------------------------------------------------------------
    def show(self):
        ps.init()
        _try(ps.set_ground_plane_mode, "none")
        _try(ps.set_program_name, "CC5513 - Tarea 3: Parametrizacion")
        self.register()
        ps.set_user_callback(self.callback)
        ps.show()

    def screenshot(self, prefix):
        """Guarda <prefix>_3d.png y <prefix>_uv.png sin abrir la ventana."""
        try:
            ps.init()
        except Exception:
            ps.set_allow_headless_backends(True)
            ps.init()
        _try(ps.set_ground_plane_mode, "none")
        _try(ps.set_up_dir, "y_up")
        outs = []
        for tag, kw in (("3d", dict(show_surface=True, show_flat=False)),
                        ("uv", dict(show_surface=False, show_flat=True))):
            if self.UV is None and tag == "uv":
                continue
            ps.remove_all_structures()
            self.register(**kw)
            if tag == "uv":
                _try(ps.set_view_projection_mode, "orthographic")
                c = self.offset + np.array([self.size / 2, self.size / 2, 0])
                ps.look_at(c + np.array([0, 0, 2.2 * self.size]), c)
            else:
                _try(ps.set_view_projection_mode, "perspective")
                ps.reset_camera_to_home_view()
            fn = "%s_%s.png" % (prefix, tag)
            ps.screenshot(fn, transparent_bg=False)
            outs.append(fn)
        return outs
