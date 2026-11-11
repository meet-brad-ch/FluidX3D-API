# Volumetric slip wall on terrain (D3Q27)

FluidX3D-API adds an opt-in wall boundary for a static triangulated terrain: the wall cuts the lattice cells (cut-cell
volumes, not voxel steps) and applies a sampled linear slip law. This page tells you how to build it, how to use it,
what the engine supports, and what it does not support.

> **Status (2026-11-10):** experimental, version 1. The model is qualified against an FP64 reference on an inclined channel
> and on one terrain fixture. The engine (FP64 arithmetic, FP32 storage) passes its acceptance on both: the channel for
> 65536 steps and the terrain fixture M3 for 2000 steps (see Validation). This qualifies the tested channel and terrain
> contract, not general terrain accuracy. The engine of this product passed its core acceptance
> (V1_PRODUCT_CORE_ACCEPTANCE_2026-11-10.md), and the ready example passed its checks: equivalence with the acceptance
> driver, finite outputs, the visual pair and one cost measurement (see Validation and Example results).

## What the feature does

| Part | What it does |
|---|---|
| Geometry | A height-field terrain (triangles) in a box, periodic in x and y, with a flat slip lid at the top. Each fluid cell has its fluid volume; each wall facet has its own normal and area. |
| Collision | Recursive regularized (RR) D3Q27 collision with the full force term in **every** cell, in FP64 arithmetic. |
| Wall | The fluid that streams into a facet returns from it. The return has an even part at a wall-extrapolated state, a friction part, and one mass correction per facet that keeps the mass of the facet. |
| Slip law | A linear slip law at the sampled velocity: the wall traction ON THE FLUID is the vector `-rho * beta * u_t`, with `beta = K / 3` (lattice units); its magnitude is `rho * beta * |u_t|`. `u_t` is the tangential fluid velocity sampled half a cell from the facet. `K = 0` is a free-slip wall. |
| Transfer between facets | Fluid that leaves one facet and reaches another facet within one step is held as a pending amount and arrives one step later. The total of fluid plus pending mass is conserved. |
| Storage | Populations in FP32 (FluidX3D's shifted storage `g = f - w`); pending amounts and all wall arithmetic in FP64. |

The slip law is a prescribed linear coefficient. It is **not** a rough-wall or log-law closure. The feature makes no claim
for atmospheric boundary layers, turbulence models (SGS) or stratification.

## How to build

The feature is a compile-time extension. Without it, FluidX3D does not change.

| Macro | Meaning |
|---|---|
| `VOLUMETRIC_WALL` | The native wall. |
| `VOLUMETRIC_WALL_RR27` | The qualified D3Q27 revision described on this page. Requires `D3Q27` and `VOLUMETRIC_WALL_FP64_COLLISION`. |
| `VOLUMETRIC_WALL_FP64_COLLISION` | FP64 collision arithmetic in the ordinary cells. |
| `VOLUME_FORCE` | A uniform, constant body force, the only supported forcing type. Zero force and a fluid at rest are supported. |

CMake: `add_fluidx3d_example(NAME <name> EXTENSIONS D3Q27 VOLUMETRIC_WALL VOLUMETRIC_WALL_FP64_COLLISION VOLUMETRIC_WALL_RR27 VOLUME_FORCE)`.

The build refuses (compile error) `VOLUMETRIC_WALL_RR27` without `D3Q27` or without `VOLUMETRIC_WALL_FP64_COLLISION`, and
with `SUBGRID`, `FP16S`, `FP16C`, `FORCE_FIELD`, `LEVEL_FORCING`, `TEMPERATURE`, `MOISTURE`, `SCALAR_FV`, `SURFACE`,
`PARTICLES`, `MOVING_BOUNDARIES` or `EQUILIBRIUM_BOUNDARIES`.

## How to use it

1. Get a container (`.vwnat` file) of the geometry. The container holds the cells, the cut volumes, the facets, the
   transfer maps, the friction rows, the gradient map, the scalars (`nu`, `omega`, `K`) and the initial populations and
   pending amounts. Make it with the offline builder (Python 3.10 or newer, numpy, scipy):

       python scripts/slip_wall/terrain_container.py --heights HEIGHTS.csv --z-top 16 --nu 0.1 --friction 0.06 --out terrain.vwnat

   `HEIGHTS.csv` holds Ny lines of Nx node heights (lattice units), periodic in x and y; the lid is at `--z-top`. The builder
   refuses a terrain with a slope of 1 or more, a lid clearance below 2 cells or a column that is not connected. The
   container starts at rest.

   The builder uses the heights exactly as given; it never rounds or moves them. A node height a tiny distance from an
   integer (a cell face), but not equal to it, can make a fluid sliver thinner than the binary64 clipping keeps (its
   vertex-on-plane tolerance is about 1e-12). The builder then refuses the cell, and the message names the node heights and
   their distances to the faces. Not every near-integer height is refused; whether a sliver is lost depends on the cell. If
   it happens, round or offset the heights yourself and record that change. Analytic terrain is the usual case: for example
   3.5 + cos(2 pi i/24) cos(2 pi j/24) gives 2.9999999999999996 where the exact value is 3.

   The example input `examples/volumetric_wall/L24_heights_r12.csv` is that analytic terrain with the heights rounded to
   12 decimals before the build (largest change 4.8e-13 lattice units); the rounding is part of the example's declared
   input, not of the builder. The ready container of this input is shipped; see "Run the example" below.

   The builder checks the map identities of every container with the native definitions and tolerance before it writes
   the file. When one fails, it writes `OUT.rejected.json` (the worst states) and no container, and exits with code 3.
2. Make an `LBM` of the container's box size with the container's viscosity and the force:
   `LBM lbm(Nx, Ny, Nz, nu, fx, fy, fz);`
3. Mark every cell outside the container's fluid cells `TYPE_S`.
4. Attach the wall before the first `run()`: `lbm.set_volumetric_wall("<file>.vwnat");`
5. Run: `lbm.run(steps);`

`examples/volumetric_wall` shows these steps. Its inputs are environment variables: `VOLUMETRIC_WALL_CONTAINER`,
`VOLUMETRIC_WALL_STEPS`, `VOLUMETRIC_WALL_FORCE` (`fx,fy,fz`), `VOLUMETRIC_WALL_OUT` (a folder for the fields; the wall build
also writes the raw state) and, in the visual targets, `VOLUMETRIC_WALL_FRAMES` (ascending steps for a frame). Its targets:

| Target | What it is |
|---|---|
| `volumetric_wall_rr27` | The slip wall described on this page (D3Q27, RR27), without graphics. |
| `volumetric_wall_noslip` | The baseline: the same container cells as plain voxels with FluidX3D's bounce-back and its FP32 SRT collision. Both the geometry (voxel steps instead of the cut surface) and the collision differ from the wall build, so a comparison shows the feature against the established baseline, not slip alone. |
| `volumetric_wall_rr27_visual`, `volumetric_wall_noslip_visual` | The same two builds with graphics: frames at the given steps with one fixed view (velocity on the middle y slice seen face-on, fixed camera, fixed colour scale). |
| `volumetric_wall`, `volumetric_wall_update_fields` | The earlier D3Q19 wall (kept for its earlier results). |

### Run the example

The commands are for PowerShell on Windows, from the root of a checkout. The build uses the `tools` submodule
(cmake-build-scripts). Initialize it once with `git submodule update --init tools`.

1. Configure and build the wall, the baseline and the container checker:

       tools\configure.bat Release Ninja -- -DFLUIDX3D_EXPERIMENTAL_VOLUMETRIC_WALL=ON
       tools\build.bat volumetric_wall_rr27 Release
       tools\build.bat volumetric_wall_noslip Release
       tools\build.bat volumetric_wall_native_check Release

   The executables are in `bin`. For the frames, build `volumetric_wall_rr27_visual` and `volumetric_wall_noslip_visual`.
2. Use the ready container `examples/volumetric_wall/L24_r12_d3q27.vwnat` (SHA-256 9af23bba..., 20.6 MB; box 24 x 24 x 16,
   7536 fluid cells, 1920 facets; `nu = 0.1`, `K = 0.06`, at rest). Its build record is
   `examples/volumetric_wall/L24_r12_d3q27.vwnat.json`, its provenance `scripts/slip_wall/PROVENANCE.md`.

   Optional, to make it again from the example input (CPU only; the recorded build took 307 s). The recorded build used
   Python 3.10.11, numpy 2.2.6 and scipy 1.15.3; byte identity of a rebuild with the shipped file is not claimed, so check
   the result with step 3. The builder refuses a container whose map identities fail and then writes
   `L24.vwnat.rejected.json` instead:

       python scripts/slip_wall/terrain_container.py --heights examples/volumetric_wall/L24_heights_r12.csv --velocities D3Q27 --z-top 16 --nu 0.1 --friction 0.06 --out L24.vwnat

3. Check the container (no GPU). The checker writes its cell lists with the given prefix and prints `PASS`:

       bin\volumetric_wall_native_check.exe examples\volumetric_wall\L24_r12_d3q27.vwnat L24_lists

4. Run the slip wall, then the baseline, on the same container and force:

       $env:VOLUMETRIC_WALL_CONTAINER = "examples\volumetric_wall\L24_r12_d3q27.vwnat"
       $env:VOLUMETRIC_WALL_STEPS = "20000"
       $env:VOLUMETRIC_WALL_FORCE = "0x1.2c1eb6p-17,0x1.2c1eb6p-18,0"
       New-Item -ItemType Directory -Force out_rr27, out_noslip
       $env:VOLUMETRIC_WALL_OUT = "out_rr27";   bin\volumetric_wall_rr27.exe
       $env:VOLUMETRIC_WALL_OUT = "out_noslip"; bin\volumetric_wall_noslip.exe

   Each output folder gets `rho.f32`, `u.f32` (the fields) and `meta.txt`; only the wall build also writes the raw state
   (`fi.f32`, `pending.f64`). For the frames, also set
   `$env:VOLUMETRIC_WALL_FRAMES` (for example `"5000,10000,20000"`) and run the two `_visual` executables.

The two images do not show slip alone. The left build is the smooth cut-cell D3Q27 RR wall with the slip law; the right
build is FluidX3D's voxel bounce-back (no-slip) with the SRT collision. Label them so.

**Units.** All values are lattice units. `K` is a container scalar; the wall traction on the fluid is `-rho * (K / 3) * u_t`
(the force on the wall is the opposite vector). In SI units, `beta_SI = (K / 3) * dx / dt`. The force density is momentum per volume per step. The velocity the fields hold is
the physical velocity `(j + F/2) / rho`.

**Enable and disable.** The feature is on only in a build with the macros above and only after `set_volumetric_wall()`.
A `VOLUMETRIC_WALL_RR27` build refuses `run()` without an attached wall, because its collision differs from the
baseline. To compare with the baseline, build the same setup without the macros. Switching needs a restart; there is no
live switching.

**Restart.** The state is the raw populations (`volumetric_wall_read_fi`), the pending amounts
(`volumetric_wall_read_pending`) and the time step `t`. To restart, attach the same container, call `run(0)`, write the
populations and the pending amounts back (`volumetric_wall_write_fi`, `volumetric_wall_write_pending`) and set `t`
(`volumetric_wall_set_time_step`). Never reset the pending amounts to zero.

## What the attachment refuses

`set_volumetric_wall()` stops with a message when:

- the container's velocity set (D3Q19 or D3Q27) differs from the build's;
- a `VOLUMETRIC_WALL_RR27` build gets a container without the affine sections (arms, gradient map) or without friction rows;
- a table identity of the container fails (volume partitions, prism coverage, friction tensor, sampler reproduction);
- a facet row has no fluid part (a fully blocked prism);
- the container's viscosity differs from the `LBM` viscosity;
- a fluid cell of the lattice has no container state;
- a wall is already attached, or the `LBM` has more than one domain;
- the device has no FP64 support.

At run time no value is clamped or repaired. The wall counts every facet with an invalid wall state: a non-positive or
non-finite wall density, or a non-finite wall velocity. The collision counts every velocity at the clamp limit. At the END of each `run()` call the engine reads the two counters
(one small read) and, when either is nonzero, stops the program with an error that gives both counts and the time step. The
detection is at the end of `run()`, not at the step where the failure happened: a long `run(N)` reports it after N steps.
To find the step, run in shorter calls.

## Precision and memory

- Arithmetic: FP64 in the collision of every cell and in every wall kernel. The force is the float32 value widened to
  double.
- Storage: FP32 populations, stored as `fl32(f - w)` with the double weights 8/27, 2/27, 1/54, 1/216.
- Memory: D3Q27 needs 125 bytes per cell (FP32) plus the wall buffers (FP64 per wall cell, row and facet).
- FP16 storage is not supported.

## Limits

| Limit | What it means |
|---|---|
| One GPU, one domain. | Multi-GPU is refused. |
| Static, periodic geometry with a lid. | No inlet, outlet or moving wall. |
| Uniform constant body force only. | Other forcing types are refused. |
| Linear slip law with a constant `K`. | No rough-wall or log-law closure. |
| No SGS. | `SUBGRID` is refused. |
| Laminar, low-Mach flow. | The qualification covers laminar channel and terrain flows only. |
| The transfer between facets is delayed by one step. | It is not exact fractional-flight transport. The fluid-only mass changes by the pending exchange; fluid plus pending mass is conserved. |
| Rows that are fully blocked, and directions that graze a facet, are not covered by the qualification fixtures. | Fully blocked rows are refused. Grazing directions have no row by construction. |
| Containers come from an offline Python builder (`scripts/slip_wall`). | There is no in-engine geometry builder; a large terrain takes minutes to hours to build. |
| A node height a tiny distance from an integer can make a sliver that the builder's binary64 clipping loses; the builder then refuses the cell. | Round or offset the input explicitly; the builder never changes it. |

## Validation

| Test | Result |
|---|---|
| FP64 reference, inclined channel, full force, two starts | Density departure 4.44e-5 (limit 1e-4), profile error 0.16 % (limit 2 %), normal velocity 1.8e-4 of the peak (limit 1 %) |
| FP64 reference, 65536-step channel screen on the GPU | Disturbance decays (L2 ratio 0.0036), final gates pass |
| FP64 reference, terrain M3 with transfer between facets, 2000 steps | Fluid plus pending mass and momentum within 1e-5, traction identity, positivity, active non-equilibrium transfer on all 152 links |
| Engine: build, default-off identity, refusals, one-step agreement with the reference, restart | Pass |
| Engine, inclined channel, 65536 steps (FP32 storage) | Mass drift 1.1e-9, momentum 2.8e-7 (limits 1e-5); profile error 0.16 %, density departure 4.4e-5, normal velocity 1.8e-4 of the peak at every checkpoint; one-step map agreement at most 8.9e-16 (see the note below) |
| Engine, terrain M3, 2000 steps (FP32 storage) | Mass drift 5.7e-11, momentum 7.7e-10 (limits 1e-5); one-step map agreement at most 8.9e-16 (see the note below); transfer active on all 152 links |

The one-step map agreement (at most 8.9e-16) is not a trajectory difference. At each checkpoint, the run captured the
input of one step and the output and intermediate arrays of the same step. The FP64 reference applies that step to the
captured input, and the result is compared with the recorded arrays. The 8.9e-16 is the comparison of the FP64
arithmetic only. The FP32 stored populations have a separate rounding (ULP) gate, and they do not agree to 8.9e-16.
No independent FP64 trajectory of 65536 (or 2000) steps was compared with the engine. The physical, profile and
conservation results in the same rows are separate gates.

The records are kept in the research tree of the project (branch `volumetric-wall-model`, folder `research/wall_model/`),
which is not part of this product. Each is named with its commit and SHA-256, so it can be checked unchanged:

| Record | Commit | SHA-256 |
|---|---|---|
| V1_PRODUCT_CORE_ACCEPTANCE_2026-11-10.md (this product, 8362bd32) | 90516a56 | 2e1ffff969709aa4a193e644120a88fd721d410e26abfc56ee423155a7566af7 |
| WALL_REFERENCE_CHANNEL_RESULT_2026-11-10.md | ed53f1e2 | 29da49aba3fd586409b1a628a983886cf2590d18be8e35aecd182b60364b3967 |
| LONG_STABILITY_RESULT_2026-11-10.md | b260a2a9 | 370746ee642541418c37361bde8a41ae0eeb0136125ab7fc894700e81ec23e65 |
| TERRAIN_AUGMENTED_RESULT_2026-11-10.md | 2d3cfa02 | fcfbbf8ed3ea52ddb26ecc0e18ba6773982380d53ea103ad513724af92fbb4fe |
| NATIVE_M1_INITIAL_RESULT_2026-11-10.md | dc92f8ed | 60a0a3a94536c171ba6f020ae03e9665385f64d5f85c292cbd4060611ad14597 |
| NATIVE_M1_PHYSICAL_ACCEPTANCE_2026-11-10.md | ed87f11e | 7a0725e139ac1bd7d9dcbae9c926dd99dd8d2524635d17d9e6ae0e26093acd5a |
| MATHEMATICAL_MODEL.md (the model, section 12) | ed87f11e | 7bd0181b1ef695bec8c039f04825fb44d422f334c9e15599391fbebc0ef08268 |
| collaboration/task103_artifacts/run/t103_summary.json (builder fix, L24 container) | 9116af03 | ec7a30034563852cf197bff6bf5be0f074cc356672643f4e2fcd1efcb7cf4a72 |
| collaboration/task104_artifacts/run/t104_summary.json (example, cost) | 728f8ce7 | 27cfc8b694e89141d27217b7165745141682588cd60ac08b5eed051028054ff1 |
| collaboration/task105_artifacts/run/t105_summary.json (camera fix, headless requalification, visual pair) | 5c505001 | 7effeae72b1c8b281ebb20342452f8a442619d886db2f2908b3262514c6e0ccc |
| collaboration/task105_artifacts/SATURATION_CORRECTION_E1.md (the colour limit 0.57735 x GRAPHICS_U_MAX) | 1323791c | 0eb0fd929755a237d37529e3b229e8a5f660afb5ebb0816aba27e53be31fc308 |

## Example results

The example on the ready container `examples/volumetric_wall/L24_r12_d3q27.vwnat` (24 x 24 x 16 = 9216 lattice cells, 7536
fluid cells), 20000 steps, force `0x1.2c1eb6p-17,0x1.2c1eb6p-18,0`, from rest. Record: task 104 (table above).

| Check | Result |
|---|---|
| Example against the acceptance driver on the same container, force and steps | Raw populations and pending amounts byte-identical |
| Graphics build against the headless build | Raw state byte-identical |
| Outputs | All finite, density positive, no invalid wall state reported |
| Slip wall (`volumetric_wall_rr27`) | Volume-weighted mean density 1.0000000014; mass-weighted mean velocity (3.5309e-3, 1.7656e-3, 1.0e-9); largest speed 4.715e-3 |
| Baseline (`volumetric_wall_noslip`) | Volume-weighted mean density 0.99999947; mass-weighted mean velocity (1.2616e-3, 6.308e-4, -6.3e-9); largest speed 2.118e-3 |

The baseline differs from the wall build in the geometry (voxel steps, bounce-back, no slip), the collision (FP32 SRT) and
the start (FluidX3D's own equilibrium initialization), so the difference of the two rows is not the effect of slip alone.

![Slip wall and baseline at step 20000](images/l24_pair_step20000.png)

The native frames of the two graphics builds at step 20000, side by side and unchanged (the text is outside the frames).
The view is the velocity magnitude on the slice y = 12, seen face-on, with x to the right and z up. The colour scale is
the same in both (`GRAPHICS_U_MAX` 0.008): full colour at |u| = 0.57735 x 0.008 = 0.004619 lattice units, and faster cells
show the same colour. At this step 6.05 % of the slip wall's fluid cells on the slice are at or above that limit (slice
maximum 0.004700); the baseline has none (slice maximum 0.002072). The thin lines are outlines of the cell flags, not
flow: grey for the solid cells, orange for the wall region of the slip wall. Do not read acceleration or a steady state
from the colours.

### Cost of this example

One run of each headless build, 20000 steps, on an NVIDIA GeForce RTX 4090 (driver 617.42), Intel Core i7-13700KF,
Windows 11. Record: k12_cost_table.json (task 104, 674f8b95).

| Build | End-to-end process wall time | Sampled peak of device VRAM in use (before the run) |
|---|---|---|
| `volumetric_wall_rr27` | 10.26 s | 2307 MiB (1882 MiB) |
| `volumetric_wall_noslip` | 0.94 s | 2214 MiB (1880 MiB) |
| Ratio | 10.9 | |

- The wall time is the whole process: start, the OpenCL compile, the setup, the run and the final export. How the time
  divides between these parts was not measured. The time and the ratio apply to this small case only and cannot be
  extrapolated to larger lattices.
- The VRAM column samples the memory in use on the whole device every 0.5 s (other processes included); it is not the
  allocation of the process. The engine's own "Memory Usage" line does not include the wall buffers.
- No MLUPs value is given. The task-104 record contains an MLUPs estimate that the measurement script derived from the
  arrival times of buffered console lines; that timing is not reliable, and the estimate is not a measured performance.
  It is a different number from the engine's own smoothed progress line.
- These numbers belong to the executables of that run (`volumetric_wall_rr27` 9f798291, `volumetric_wall_noslip`
  33e1afa8). A later rebuild (a change of comments and of the graphics camera only) overwrote those files by mistake;
  the MSVC link output is not bit-reproducible, so the rebuilt files have other hashes (c86e7bff, 0c59ca47). The rebuilt
  headless executables were requalified: their 20000-step raw state and fields are byte-identical to those of the cost
  run (task 105). The cost was not measured again.

## How to run the tests

    tools/configure.bat Release Ninja -- -DFLUIDX3D_BUILD_TESTS=ON -DFLUIDX3D_BUILD_VOLUMETRIC_WALL_NATIVE=ON -DFLUIDX3D_EXPERIMENTAL_VOLUMETRIC_WALL=ON
    tools/build.bat fluidx3d_volumetric_wall_native_tests Release
    tools/build.bat physics_volumetric_wall_native_rr27 Release

The unit tests (`fluidx3d_volumetric_wall_native_tests`) check the container reader, the direction tables, the address
algebra and the ownership of the cells, on the containers in `tests/data/slip_wall` (a missing file fails the tests).
`physics_volumetric_wall_native_rr27` is the instrument driver of the acceptance runs (`VW_MODE=attach`, `step`,
`restart`, `resume`, `protocol`, and `poison`: a synthetic invalid wall state that `run()` must report).

## References

- Chen, H., Teixeira, C., Molvig, K. (1998). Realization of fluid boundary conditions via discrete Boltzmann dynamics.
  International Journal of Modern Physics C 9, 1281-1292. doi:10.1142/S0129183198001151
- Li et al. (2026). Recursive regularized lattice Boltzmann method: forcing implementation, boundary treatment and
  application in large eddy simulation. Computers & Fluids 314, 107076. doi:10.1016/j.compfluid.2026.107076
- Lehmann, M. (2022). Esoteric Pull and Esoteric Push: two simple in-place streaming schemes for the lattice Boltzmann
  method on GPUs. Computation 10, 92. doi:10.3390/computation10060092
