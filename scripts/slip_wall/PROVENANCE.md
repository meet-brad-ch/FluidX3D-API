# Provenance of the offline terrain builder

`builder3d.py` and the copied functions of `terrain_container.py` come from the research tree of the qualified model
(branch `volumetric-wall-model`, folder `research/wall_model/`), which is not part of this product. The copies keep the
statement order and the operation order of the originals, so the result is the same bytes.

| Source file (research tree) | Commit | SHA-256 | What is copied |
|---|---|---|---|
| integration/task26/builder3d.py | c575f4c2 | d7cb46a24f8e3ce92fbfb89090e694060a4366f36bba3195b29f731da1f6ba54 | the whole file, byte-identical (`builder3d.py`) |
| integration/task26/oracle3d.py | f2ae5516 | 58634c4838f35a39fd12879389cfbc4012730ac7ae32eefcde815937f5c76729 | plane_of, connectivity_check, domain_checks (as methods of TerrainManifest) |
| integration/task26/manifest3d.py | f2ae5516 | 4e7edf0af009081f9f2d9acc654f67439877bd99344ce3d4f6ea1b69b543afe5 | lowest_node_diagonals, height_text, manifest_record |
| integration/task22/terrain_builder.py | 24241119 | 44ef5923be2e0ec9f8ae6cee8238179f77d5212649f4dcfd9c85af9301f09acd | stencil_weights (without the module's shapely import) |
| integration/export_native_input.py | 6bdc5f63 | db582eac6c4630f4b371955975d9b02a4c2a54918ca378f06a2311e10a2a203f | Writer, DTYPE_* |
| integration/task87/t87_geometry.py | 57856b76 | bf0e6f38b0f380b92f13c968e54688aa8fd8319f6d5dacd159d4f70d4cf3bd19 | VelocitySet, MapAssembler identities/permute_rows/base_rows, VelocityExport.arms |
| integration/task95/t95_geometry.py | e9fb5fac | 7eb3a88b50dff3c17ac50f72ad8e5c685037714f3911895d24bfc1cf84eb8f42 | the velocity-set builder, assembly, gradient map, wall list, sections |
| integration/task53/moment_export.py | 84215c88 | fe3a2a59b55b7b703732b3b336bbe8549dd0bbc487611ec4043d84096e05aa6e | the gradient constants QUERY, OTHERS, TIE_RELATIVE |

New code: the CLI (`ContainerBuild`, `main`) and the diagnostic context of a refused cell (`cell_context`).

## Controls (historical: the pre-task-103 builder, terrain_container.py c51baab5)

These byte controls are not claimed for the task-103 builder (see "Task 103 changes"). With the M3 terrain of the qualification (the 64 heights of the task-26 manifest M3), the package reproduces every
numeric section of the accepted containers byte for byte; only the provenance and the scalars are excluded:

| Control | Original container (SHA-256) | Result |
|---|---|---|
| `--velocities D3Q19` | M3_terrain3d_rest.vwnat (3ba819665725fdcd4c3c4848cfa5c19f91253bb552d87cc5d6fb1e85d8ee7633) | identical; the package adds the 7 affine and friction sections |
| `--velocities D3Q27` | M3_d3q27.vwnat (348f3ded246f11e03ebc5cab8b94b61a7db4378500bd3608b870e2e280a58bf0, also tests/data/slip_wall) | identical |

The controls were run with an interpreter without shapely and verified independently (research tree,
integration/task100/development and the task-101 records).

## Task 103 changes (terrain_container.py fd4fc459; accepted)

The builder before task 103 (terrain_container.py c51baab5) built the example input into a container that the native checker
refused: donor partition 1.0877e-2 (normalized per donor volume) in 6 states, 3 x/y mirror pairs of horizontal links in cut
cells at builder layer k = 3; the builder exited 0. That record is kept unchanged in the research tree
(collaboration/task101_artifacts/run_b2, container 571223fc).

Cause: a binary64 clip at a zero-measure contact. In the donor pass, the back prism of a receiving facet that meets the donor
cell only in a face (for example facet (8, 13, T0, 4) against cell (7, 12, 3) along (0, -1, 0): contact plane z = 4) was
still clipped against the cell's pieces. A piece vertex that earlier clips had moved 3.088e-12 from the plane (ON tolerance
EPS_ON * max(1, |d|) = 3.018e-12) was classified strictly inside; the split made a malformed piece and created 1.3745e-3 of
volume. The builder recorded it as max_split_defect but did not check it.

Changes:
- Zero-measure contacts: the donor pass and the producer pass skip a candidate whose prism and the region overlap in zero
  measure. The test is exact and comes before any clip: the facet's integer cell box (terrain facet: column and layer; lid:
  column, z in [Z_top - 1, Z_top]), swept by the lattice vector, against the donor cell box or the producer's swept box. The
  real prism lies inside these bounds. No tolerance, height or precision change; no renormalization; no fallback that
  depends on the outcome.
- NativeIdentityGate: the donor partition, receiver coverage, prism coverage and return delivery with the native
  definitions and tolerance (1e-10). A violated or non-finite identity writes `OUT.rejected.json` (the identities, the worst
  states or rows, the builder diagnostics) and no container; exit code 3. The output folder is created first.

Limit: the skip removes the demonstrated zero-measure contact class only. The gate and the native checker stay mandatory for
every container.

Verification (task 103, independent, research commit 9116af03, collaboration/task103_artifacts/run; accepted):

| Check | Result |
|---|---|
| Original defect, old builder, state (7, 12, 3) (0, -1, 0) | reproduced |
| The six states, this builder | residual <= 1.67e-16; direct volume equal to an independent free-path quadrature within the fixed comparison tolerance 1e-8 (an engineering tolerance, not a proven quadrature error bound) |
| Exact-overlap controls (face, edge and point contacts; positive overlap, also across periodic images) | pass |
| CLI refusal of the saved defective maps | exit 3, rejection report, no container |
| M3 D3Q19 and D3Q27 rebuilt | same sparsity in every section; largest float difference 2.49e-13 (bound 1e-12 max(1, abs old, abs new)); native identities below 5e-14 |
| Full L24 build (the example input, D3Q27) | exit 0, 307.7 s; container 9af23bba (20629360 bytes), 7536 cells, 1920 facets, 22446 rows; native identities at most 8.66e-12 |
| Native checker on that container | PASS, exit 0 |

Byte equivalence: the controls in "Controls" (historical) hold for the pre-task-103 builder only. This builder changes the
M3 tables by round-off (above). The engine's acceptance fixture tests/data/slip_wall/M3_d3q27.vwnat (348f3ded) is unchanged
and stays the engine input of the core acceptance.

## V2 change: exact candidate window of the sampler (2026-11-10)

builder3d.build_sampler took the candidates of every facet from all cells in 9 periodic images and sorted them (cost
proportional to facets x cells). SamplerCandidates now builds them from a window of half width m around floor(x_s) and
keeps the window only when its entry number NEIGHBOURS (12) has d2 < m^2. Otherwise m doubles. If a window shift leaves
{-L, 0, L}, if m passes the box, or if a cell centroid lies outside its unit cell, the original full list is used.
Proof (class docstring): every cell outside the window has d2 >= m^2 after the monotone binary64 rounding. So the first
12 sorted tuples and the d2 == 0 set are the same objects as with the full list. The stencil rule, its gates, the tie
order and every table are unchanged. terrain_container passes NEIGHBOURS and logs the counts of the window and of the
full-list fallbacks. The build record is unchanged.

Verification: the registered byte comparison of M3 and L24 against the current builder (E1). Development check
(research tree v2/hill_inventory/e2/development/window_sampler_check.py): the window and the full list per facet, and
build_sampler with and without the window, on M3 and on a 3 x 3 case whose window crosses both box edges, and the same
case with a forced full-list fallback.

## V2 change: the 12 nearest VISIBLE stencil candidates (2026-11-11, #2283)

A SAMPLING-SELECTION rule change, not a change of fluid physics. Before: the sampler took the 12 nearest centroids by
distance and then refused the facet (R3) if one of them was not visible from the sample. The steep qualification input S8
was refused at 28 facets with one hidden candidate each, although an admissible visible stencil existed at every one of
them (research tree v2/hill_inventory/e2/steep/V2_STEEP_R3_DIAGNOSIS_E2_2026-11-11.md). Now:
SamplerCandidates.visible_prefix walks the same sorted list and keeps the exact (d2 == 0) entries and the candidates that
pass the existing R3 segment test, until 12 are kept. The direct-centroid handling, the rank, condition and reproduction
gates, and the R3 check of the chosen cells are unchanged. With fewer than 12 visible candidates, the facet is refused
(R3). The window is accepted when its 12th VISIBLE entry has d2 < m^2, so the exactness of the window is kept. Where the
12 nearest candidates are all visible (every facet of M3, L24 and the channel passed R3 under the old rule), the selected
stencil is the same as before. The byte identity of those containers is to be shown by the registered comparison.
R11 stays the default. `--allow-steep` skips only R11 (R10 and R1 stay) and records it in the build report
(`allow_steep`, `domain_checks`). The steep geometry checks S8 and S8X qualify their own bounded inputs.

## V2 change: local frames, exact piece volumes, parallel passes (2026-11-11, tasks 106 and 107)

A GEOMETRY ARITHMETIC change; the clipping planes, keys, signs, ownership and tolerances are unchanged.
- Facet area: a fan of vertex differences (translated_fan_area) in place of the global-coordinate shoelace.
- Clipping in integer-local frames: the cell, the producer and the donor clip in the frame of their cell, with planes from
  the exact triangle shifted by the integer origin; keys and moments are global.
- Piece volumes and moments: every fan determinant is an exact integer (a common power-of-two scale of the binary64
  coordinates), and every volume and moment weight is one correctly rounded division.
- `--workers N`: the producer and donor passes on N processes; the container is the same, byte for byte, as with 1.
- `--initial-velocity ux,uy,uz`: the uniform start f0 = feq3(1, u) (the reference equilibrium); the default stays the rest start.
Effect on the maps: whole-cell transfers that binary64 gave as 1 +- a few ulps are now exactly 1.0, so rule #253 adds fewer
cells to the wall list and the gradient sections are smaller (L24: 6619 -> 1936 gradient cells); the D, G, R, W values change
at the rounding level (L24: at most 3.6e-12 against the shipped container). Qualification: the hill local acceptance
(V2-HB-02/05), the all-facet area audit, and the M3, L24, S8 and S8X regressions (research tree, task 106).

## Example container

`examples/volumetric_wall/L24_r12_d3q27.vwnat` (SHA-256
9af23bbace3ca097feb0b71645ce49221112ced6eab73417825fe09c2b535edd, 20629360 bytes) is the container of the full L24 build
above, shipped unchanged. Its build record is `examples/volumetric_wall/L24_r12_d3q27.vwnat.json`. It was built by the V1
builder: a rebuild with the current builder is not byte-identical (see the V2 change above: 1936 instead of 6619 gradient
cells; D weights within 3.6e-12). The earlier task-101 container (571223fc) is older than the task-103 contact fix and differs
from the shipped one by the L24 contact defect (one donor volume by 1.37e-3); it is not a reference for the current builder.

## Example input

`examples/volumetric_wall/L24_heights_r12.csv` (SHA-256
660b5646d5caf72a9c70830024f9edaeb2a2a1da4674ca011b72031ed11cf11c) is the declared input of the example. The rule: 24
lines (j = 0 ... 23) of 24 comma-separated values (i = 0 ... 23), each `repr(round(3.5 + cos(2 pi i/24) cos(2 pi j/24),
12))` with Python's `math.cos` in binary64, CRLF line ends (git keeps the bytes: `-text` in `.gitattributes`). The rule
reproduces the file byte for byte. The largest change against the unrounded values is 4.8e-13 lattice units. No node is
within 1e-12 of an integer except the exact integers. The unrounded heights give a fluid sliver that the builder refuses
(see the guide, research/slip_wall/README.md).
