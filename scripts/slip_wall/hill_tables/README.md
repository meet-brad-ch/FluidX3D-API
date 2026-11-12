# Hill table tools (T119 and SRD sections)

`prepare_hill_container.py` adds the sections `t119_gdelta`, `t119_wmoment` and `t121_*` to a geometry container of the V2 hill. The targets `volumetric_wall_rr27_sgs_rough_open_visual` and `_live`
are built with `VOLUMETRIC_WALL_T119` and `VOLUMETRIC_WALL_SRD` and refuse a container without these sections.

- `tree/` holds 22 files of the research tree **verbatim** (same relative paths, so that the tools find each other). `MANIFEST.sha256` lists the sha256 of each file, its source path and the
  commit of the research tree that the copies were taken from. The files keep their own headers and provenance notes; the licence of this repository applies.
- The script copies `tree/` into the work folder, writes the few task copies (the pinned container and input constants, the fixture rows, and the rest-start rule of `make_t119_container.py`) and runs
  the four stages. It stops at the first failing stage and writes `preparation_record.json` with the hashes.
- Inputs: the heights CSV and the geometry container made from it by `scripts/slip_wall/terrain_container.py`. For the registered extended hill: `examples/volumetric_wall/hill_64_apron8_ext80_heights.csv`
  (sha256 8fb6d1fd...) and a geometry container with sha256 a0a1d18c1745c5c4d5e7d498a9817b2494ba4f62320020d94b136d8ecb2f3792 (the build command of `research/slip_wall/README.md` with the roughness written as a quoted string and `--name hill`; the metadata and provenance of the container differ from the registered one, the physical arrays are equal).
- Not shipped: the independent verifier of the SRD tables (it needs saved flow states of the research tree). Run it in the research tree if you need it.
- Status: experimental. The container made with these tools was used in the accepted hill run (see `research/slip_wall/README.md` for the scope).
