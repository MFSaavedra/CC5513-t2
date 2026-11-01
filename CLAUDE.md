# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Course assignment for CC5513 (Procesamiento Geométrico y Análisis de Formas): surface parameterization of disk-topology triangle meshes. The full statement is `Tarea2_SGP.pdf`. Its title says "Tarea 2", but its page headers and all the code say "Tarea 3"; they are the same assignment. All code lives in `tarea3_param/`. Code, docstrings, CLI output and the expected report are in Spanish.

## Assignment constraints (from the PDF; grading depends on them)

- Only `laplacians.py` (`cotangent_weights`, `mean_value_weights`, Part 1) and `boundary.py` (`polygon_points`, Part 2) are meant to be modified. **Do not change function signatures**, because the grader runs extra hidden tests against them. `tests.py` may be extended.
- `python tests.py` and `python run_tarea3.py` must work without modifications.
- Allowed libraries: NumPy, SciPy, Polyscope and Matplotlib only. Never use libraries that compute parameterizations or Laplacians (libigl/igl, pygeodesic, etc.).
- Weight functions must be vectorized. A loop over the 3 triangle corners is allowed; loops over vertices or faces are not. On Lucy (28k faces) each weight matrix should build in under 1 s.
- Do not "fix" negative cotangent weights. Question 1.1 asks for them to be counted.
- Parts 3 to 5 are analysis of given code (LSCM, distortion metrics) for a PDF report of at most 5 pages. Custom experiment scripts are allowed and should be included in the submission. Bonus B1 is a modified square mapping that gives `degen = 0`; B2 applies a real photo as the texture.

## Commands

Run everything from `tarea3_param/`. Modules import each other by bare name (there is no package).

```bash
pip install -r requirements.txt                          # numpy, scipy, polyscope
python tests.py                                          # all checks; exit 0 only if all OK
python run_tarea3.py --mesh data/nefertiti.off           # interactive Polyscope viewer
python run_tarea3.py --mesh data/bunny.off --batch --csv bunny.csv   # metrics for every method/boundary/spacing combo
python run_tarea3.py --mesh data/bunny.off --method lscm --color conformal --shot fig  # fig_3d.png + fig_uv.png, no window
python run_tarea3.py --mesh data/bunny.off --method lscm --pins 0 10  # LSCM with pins at boundary positions 0 and 10
```

- `tests.py` is a custom harness (not pytest). Each check prints `OK`, `FALLA` or `PENDIENTE`. To run one check, use `python -c "import tests; tests.check('cot', tests.t_cot_linear)"`; test functions are named `t_*`.
- `tests.py` needs only numpy and scipy. `run_tarea3.py` imports `viewer` (and therefore `polyscope`) at module level, so even `--batch` needs polyscope installed.
- Without OpenGL, use `--batch` and `--shot` (the viewer tries a headless backend).

## Architecture

`param.parameterize(V, F, method, shape, spacing, loop=None, pins=None)` is the single entry point. The viewer, the batch table and the tests all go through it or through its pieces.

- **Fixed-boundary methods** (`uniform`, `harmonic`, `meanvalue`): `laplacians.WEIGHTS[method](V, F)` builds a sparse `W` with `W[i, j] = D_ij`. Row `i` holds the weights vertex `i` gives its neighbors, so `W` is **not** necessarily symmetric (mean-value weights are not). Next, `laplacian_from_weights` builds `L = diag(rowsum) - W`, then `boundary.map_boundary` assigns positions in two steps: `boundary_parameter` maps each boundary vertex to `t ∈ [0,1)` by `arc` length or `uniform` spacing, and a shape function maps `t` to the plane. Finally, `solvers.solve_fixed_boundary` solves the reduced system `L_II U_I = -L_IB U_B` with one sparse LU factorization.
- **LSCM** (free boundary, `solvers.lscm`): uses the Mullen et al. formulation `Q = [[L, -K], [K, L]]`. Here `L` is the **cotangent** Laplacian and `K` is `boundary_area_matrix` (signed area along the boundary loop). Two pins are fixed and the solver solves `Q_ff x_f = -Q_fp x_p`. Both LSCM and `conformal_energy` (the `ED/|A|` column) depend on `cotangent_weights`, so they show PENDIENTE or NaN until Part 1a is done.
- **Distortion** (`distortion.py`): per-face Jacobians come from the isometric local 2D frames in `mesh_utils.local_triangle_coords`. `distortion_report` first rescales UV so that the area-weighted mean of `log2|det J|` is 0. It marks faces with `|det J| < eps·median` as degenerate and excludes them from the `conf_*` and `area_mean` statistics. It counts flips relative to the dominant orientation.

### Conventions to respect when implementing

- **The PENDIENTE contract**: unimplemented functions raise `NotImplementedError`. `tests.py`, `run_batch` and the viewer catch exactly that exception and show `PENDIENTE` without crashing. Other exceptions are reported as failures.
- `laplacians._sparse_from_triplets` **sums** duplicate `(i, j)` entries. The intended pattern is to emit one contribution per triangle corner and let the two triangles of an interior edge accumulate.
- `mesh_utils.corner_angles(V, F)[t, c]` is the angle at vertex `F[t, c]`. That angle is opposite edge `(F[t, c+1], F[t, c+2])`, which is what the cotangent weights need. Mean-value weights use the angle *at* `v_i`, divided by `|v_i - v_j|`.
- `boundary_loop(F)` follows face orientation (CCW). LSCM's signed-area term depends on this.
- `polygon_points(t, corners)`: corners are CCW, `t = 0` is `corners[0]`, and the perimeter is traversed at constant speed, so `t` is shared among sides in proportion to their length (not `1/k` per side). The star boundary is intentionally non-convex.
- Viewer UI strings are ASCII-only (no accents) on purpose, because old Polyscope versions cannot render non-ASCII text.
- `data/*.off` (nefertiti 299 vertices, bunny 1573, lucy 14156) are read-only test meshes.
