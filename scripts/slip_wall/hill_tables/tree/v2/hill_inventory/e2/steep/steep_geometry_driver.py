"""V2 steep geometry (E2, #2258): the research-only D3Q27 adapter and driver. Nothing in task 26 or the product is edited.

Stages (each writes into a NEW --out folder; E1 registers and runs them):
  control  the task-26 exact oracle with the 26 moving D3Q27 directions on the task-26 manifest M3 (manifest3d.CASES["M3"]).
           The 18-direction subset of every table must be EXACTLY equal (same Fraction text) to the retained
           tables_M3.json; cells and facets equal in full; the oracle's exact identities hold for all 26 directions.
  s8       the frozen steep input S8 (S8_heights.csv): the PRODUCT builder (scripts/slip_wall, the accepted window sampler)
           in D3Q27 with R11 skipped only here (SteepManifest: R10 and R1 kept), then its NativeIdentityGate; the builder runs
           FIRST and a refusal stops the stage. Then the exact oracle (26 directions, check_slope False; R10 and R1 kept) and
           the existing comparator compare3d.oracle_vs_builder on the D3Q27 list (TOL 1e-12, key sets first). On a pass the
           container is written (the existing Writer), for the pinned native checker (E1).

Direction lists (DirectionScope, try/finally): oracle3d.LEX_C = the 26 moving D3Q27 directions; the comparator's
compare3d.B.LEX19 = the 27 D3Q27 directions in the product builder's VelocitySet order (lexicographic, rest at 13). Both are
restored after the stage. Assessment: V2_STEEP_D3Q27_ORACLE_ASSESSMENT_E2_2026-11-11.md (631461ef).

Exit codes and status (#2261): 0 PASS / EQUAL; 1 NOT PASS / NOT EQUAL; 2 INVALID (GeometryInvalid of the oracle or the
builder = a violated exact identity, a violated native identity gate, any other exception, an existing --out); 3 REFUSED
(GeometryRejected = an explicit admissibility rejection). E1's wrapper maps the status to the batch verdict.

Module-name note: task 26 and the product both have a module "builder3d". The driver imports the product builder first;
compare3d then binds that module as B and reads only B.LEX19 (patched here) and type names. The report records the file.

  envs/solver_poc/Scripts/python.exe -B v2/hill_inventory/e2/steep/steep_geometry_driver.py --stage control|s8 --out DIR
      [--builder-dir ports/20261110_v2_builder/source/scripts/slip_wall]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[3]
TASK26 = WALL / "integration" / "task26"
RETAINED_M3 = WALL / "collaboration" / "task26_artifacts" / "oracle" / "run1" / "tables_M3.json"
S8_CSV = HERE / "S8_heights.csv"
S8_Z_TOP = 8
NU, FRICTION = 0.1, 0.0          # container scalars only; the geometry tables do not depend on them
DEFAULT_BUILDER = WALL / "ports" / "20261110_v2_builder" / "source" / "scripts" / "slip_wall"
LEX19_MOVING = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if 0 < x * x + y * y + z * z <= 2]
LEX27_MOVING = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if 0 < x * x + y * y + z * z]


class Modules:
    """Imports in a fixed order: the product builder first, then task 26 (oracle, manifests, comparator, table writer)."""

    def __init__(self, builder_dir: Path) -> None:
        sys.path.insert(0, str(builder_dir))
        import terrain_container as T  # noqa: PLC0415  (product; binds the product builder3d)
        sys.path.insert(1, str(TASK26))
        import oracle3d as O  # noqa: PLC0415
        import manifest3d as M  # noqa: PLC0415
        import compare3d as C  # noqa: PLC0415
        import oracle_stage2 as S2  # noqa: PLC0415  (table_record only; nothing runs on import)
        self.T, self.O, self.M, self.C, self.S2 = T, O, M, C, S2
        self.files = {name: str(Path(mod.__file__).resolve()) for name, mod in
                      (("terrain_container", T), ("builder3d_bound_by_compare3d", C.B), ("oracle3d", O), ("compare3d", C))}
        self.sha256 = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in self.files.items()}


class DirectionScope:
    """Sets the oracle's and the comparator's direction lists for one stage and restores them in finally."""

    def __init__(self, mods: Modules, comparator_list: list | None) -> None:
        self.mods, self.comparator_list = mods, comparator_list

    def __enter__(self) -> "DirectionScope":
        self.saved_oracle = self.mods.O.LEX_C
        self.saved_comparator = self.mods.C.B.LEX19
        self.mods.O.LEX_C = list(LEX27_MOVING)
        if self.comparator_list is not None:
            self.mods.C.B.LEX19 = list(self.comparator_list)
        return self

    def __exit__(self, *exc) -> None:
        self.mods.O.LEX_C = self.saved_oracle
        self.mods.C.B.LEX19 = self.saved_comparator


def steep_manifest_class(T):
    class SteepManifest(T.TerrainManifest):
        """TerrainManifest whose domain_checks keeps R10 and R1 (the product lines 182-184 and 191-197, unchanged) and skips
        ONLY the R11 slope loop (#2253). The product class is not edited."""

        def domain_checks(self) -> None:
            hmax = max(self.heights.values())
            if self.Z_top - hmax < 2:
                raise T.B.GeometryRejected(f"R10: clearance {self.Z_top - hmax} < 2")
            for i in range(self.Lx):
                for j in range(self.Ly):
                    diag, off = self.diagonal_nodes(i, j)
                    lo, hi = max(off), min(diag)
                    K = math.floor(lo) + 1
                    if K <= hi:
                        raise T.B.GeometryRejected(f"R1: column ({i}, {j}) disconnected at level {K} (max off {lo}, min diag {hi})")
    return SteepManifest


def steep_manifest(T):
    """S8 as a SteepManifest. TerrainManifest.from_csv is a static method that returns the base class, so the product reader
    builds the base instance and its fields are copied into the subclass (the same heights and diagonals)."""
    base = T.TerrainManifest.from_csv(S8_CSV, S8_Z_TOP, "S8")
    return steep_manifest_class(T)(base.name, base.Lx, base.Ly, base.Z_top, base.heights, base.diagonal)


class Stages:
    def __init__(self, mods: Modules, out: Path) -> None:
        self.mods, self.out = mods, out
        self.report = {"status": "RUNNING", "modules": mods.files, "module_sha256": mods.sha256}

    def write_report(self) -> None:
        (self.out / "report.json").write_text(json.dumps(self.report, indent=1, default=str), encoding="utf-8")

    def tables(self, orc, name: str) -> Path:
        path = self.out / f"tables_{name}.json"
        path.write_text(json.dumps(self.mods.S2.table_record(orc)), encoding="ascii")
        return path

    def control(self) -> int:
        M, O = self.mods.M, self.mods.O
        with DirectionScope(self.mods, None):
            begin = time.perf_counter()
            orc = O.Oracle(M.CASES["M3"]())
            identities = orc.run()
            self.report["oracle_seconds"] = time.perf_counter() - begin
            self.report["identities_26"] = identities
            new = json.loads(self.tables(orc, "M3_d3q27").read_text(encoding="ascii"))
        old = json.loads(RETAINED_M3.read_text(encoding="ascii"))
        moving19 = set(LEX19_MOVING)
        result = {}
        for name in ("cells", "facets"):
            result[name] = new[name] == old[name]
        for name in ("direct", "gather", "scatter", "W", "immediate"):
            subset = {k: v for k, v in new[name].items() if tuple(ast.literal_eval(k)[1]) in moving19}
            extra_dirs = {tuple(ast.literal_eval(k)[1]) for k in new[name]} - moving19
            result[name] = {"subset_equal": subset == old[name], "subset_entries": len(subset), "retained_entries": len(old[name]),
                            "entries_of_the_8_corner_directions": len(new[name]) - len(subset),
                            "other_directions_are_corners_only": all(sum(c * c for c in d) == 3 for d in extra_dirs)}
        self.report["retained_M3_sha256"] = hashlib.sha256(RETAINED_M3.read_bytes()).hexdigest()
        self.report["comparison"] = result
        ok = all(v if isinstance(v, bool) else (v["subset_equal"] and v["other_directions_are_corners_only"]) for v in result.values())
        self.report["status"] = "PASS" if ok else "NOT PASS"
        return 0 if ok else 1

    def s8(self) -> int:
        T, O, C = self.mods.T, self.mods.O, self.mods.C
        manifest = steep_manifest(T)
        self.report["S8_csv_sha256"] = hashlib.sha256(S8_CSV.read_bytes()).hexdigest()
        run = T.ContainerBuild(manifest, "D3Q27", NU, FRICTION)
        logs = []
        b, sam, m, p, identities, arm, grad, added, record = run.build(logs.append)   # a refusal raises and stops the stage
        gate = T.NativeIdentityGate(run.vs).check(m, b)
        self.report.update(builder_seconds=run.seconds, builder_log=logs, cells=len(m["keys"]), facets=len(m["facet_keys"]),
                           rows=len(m["rows"]), native_identities=gate["identities"], native_gate_passed=gate["passed"],
                           sampler_candidates=sam.get("candidate_source"))
        if not gate["passed"]:
            # #2261: a violated map identity is an invalid builder computation (the product CLI calls it a refusal, exit 3;
            # here it is classed with GeometryInvalid)
            self.report["status"] = "INVALID (native identity gate violated)"
            self.report["native_gate_failing"] = {k: v[:5] for k, v in gate.get("failing", {}).items()}
            return 2
        self.write_report()
        directions27 = list(run.vs.directions)
        case = O.Case("S8", manifest.Lx, manifest.Ly, manifest.Z_top, dict(manifest.heights), dict(manifest.diagonal),
                      check_slope=False, check_clearance=True)
        with DirectionScope(self.mods, directions27):
            begin = time.perf_counter()
            orc = O.Oracle(case)
            self.report["oracle_identities_26"] = orc.run()
            self.report["oracle_seconds"] = time.perf_counter() - begin
            table_path = self.tables(orc, "S8_d3q27")
            comparison = C.oracle_vs_builder(C.load_tables(table_path), b)
        self.report["comparison"] = comparison
        if not comparison["pass"]:
            self.report["status"] = "NOT EQUAL"
            return 1
        writer = run.sections(b, sam, m, p, arm, grad, added, record, identities)
        container = self.out / "S8_d3q27.vwnat"
        writer.write(container)
        self.report["container_sha256"] = hashlib.sha256(container.read_bytes()).hexdigest()
        self.report["status"] = "EQUAL (native checker pending)"
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("control", "s8"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--builder-dir", type=Path, default=DEFAULT_BUILDER)
    args = ap.parse_args()
    if args.out.exists():
        print(f"REFUSED: {args.out} exists")
        return 2
    args.out.mkdir(parents=True)
    mods = Modules(args.builder_dir)
    stages = Stages(mods, args.out)
    code = 2
    try:
        code = stages.control() if args.stage == "control" else stages.s8()
    except (mods.O.GeometryInvalid, mods.T.B.GeometryInvalid) as error:
        # #2261: a violated exact identity is an INVALID oracle or builder computation, not an out-of-domain geometry
        stages.report.update(status="INVALID", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        code = 2
    except (mods.O.GeometryRejected, mods.T.B.GeometryRejected) as error:
        # an explicit geometry admissibility rejection (R1, R2/R3, R4, R5, R10, ...): REFUSED
        stages.report.update(status="REFUSED", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        code = 3
    except Exception as error:  # noqa: BLE001  every crash is recorded as INVALID with its traceback
        stages.report.update(status="INVALID", reason=repr(error), traceback=traceback.format_exc())
        code = 2
    stages.write_report()
    print(stages.report["status"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
