"""Prepares the V2 hill container for the wall targets that carry VOLUMETRIC_WALL_T119 and VOLUMETRIC_WALL_SRD.

The container builder (scripts/slip_wall/terrain_container.py) writes the geometry container. The T119 and SRD targets also need the sections `t119_gdelta`, `t119_wmoment` and the 18 sections `t121_*`.
This script adds them with the qualified tools of the research tree, which are shipped verbatim in the folder `tree/` (MANIFEST.sha256 lists every file with its sha256 and its source path).
No algorithm is changed. Four stages, each with an explicit input and output; the first failing stage stops the run:

  1. companion   tree/v2/hill_demo/e2/task119/w_moment_extractor.py  : the owned first moments of the stored W entries (record verdict PASS required)
  2. t119        tree/v2/hill_demo/e2/task119/make_t119_container.py : the geometry container plus t119_gdelta and t119_wmoment, force 0 (the rest start)
  3. srd         tree/v2/hill_demo/e2/task120/srd_construct.py       : the neighbourhoods and weights of the state redistribution
  4. final       tree/v2/hill_demo/e1/make_t121_container.py         : the 18 sections t121_* added to the T119 container

Three of these tools pin the container and the input file of the registered 80-column hill in module constants. This script works on a COPY of `tree/` in the work folder and replaces only those constants (and, in
make_t119_container.py, the readback rule that demands a changed `f0`, which the rest start does not have because its force-compatible f0 equals the stored one). Each replacement must match exactly once.
The original files are not edited.

Usage:
  python prepare_hill_container.py --heights HEIGHTS.csv --geometry hill_ext_d3q27.vwnat --workdir WORK [--workers 20] [--fixture-rows A,B] [--companion-npz FILE --companion-json FILE]
The fixture rows are two row indices of the geometry container that the companion tool recomputes serially as a scheduling check; both must have a W entry. For the registered extended hill
(geometry container sha256 a0a1d18c1745c5c4d5e7d498a9817b2494ba4f62320020d94b136d8ecb2f3792) the rows 241654 and 241678 are the default.
With --companion-npz and --companion-json an existing companion table is used after its record (verdict PASS, container sha256) and its own sha256 are checked; stage 1 is then skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTERED_GEOMETRY_SHA = "a0a1d18c1745c5c4d5e7d498a9817b2494ba4f62320020d94b136d8ecb2f3792"
REGISTERED_FINAL_SHA = "2acd5121ddd696101aa78c4e65d628c35661d9adf539877e016521d01520835a"
REGISTERED_FIXTURE_ROWS = (241654, 241678)
OLD_CONTAINER_LINE = 'WALL / "v2" / "hill_demo" / "e1" / "hf_inputs" / "hill_uniform.vwnat"'
OLD_PREFIX_LINE = 'CONTAINER_SHA_PREFIX = "a404dc7463c9bc1a"'
OLD_CSV_LINE = 'CSV = WALL / "v2" / "hill_demo" / "e2" / "input" / "hill_64_apron8_heights.csv"'
OLD_CSV_SHA_LINE = 'CSV_SHA = "d60b66771befcef8776a22bb80416d86c6baec56cf7520e2b1cd3a22a919f0d6"'
OLD_FIXTURE_LINE = "FIXTURE_ROWS = (126454, 126478)"


def sha256_file(path: Path) -> str:
    """@brief sha256 of a file. @param path the file @return the hexadecimal digest"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(source: Path, target: Path, pairs: list) -> None:
    """@brief Writes a copy of `source` with exact replacements.
    @param source the verbatim file
    @param target the copy to write
    @param pairs (old, new) texts; each `old` must occur exactly once
    @throws SystemExit when a replacement does not match exactly once"""
    text = source.read_text(encoding="utf-8")
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{source.name}: the text to replace matched {text.count(old)} times (need 1): {old[:80]}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8", newline="\n")


class Preparation:
    """@brief One preparation run in a work folder."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.heights = args.heights.resolve()
        self.geometry = args.geometry.resolve()
        self.work = args.workdir.resolve()
        self.workers = args.workers
        self.label = args.release_label
        self.companion_npz = args.companion_npz
        self.companion_json = args.companion_json
        self.record = {"stages": {}, "registered_geometry_sha256": REGISTERED_GEOMETRY_SHA, "registered_final_sha256": REGISTERED_FINAL_SHA}
        self.geometry_sha = sha256_file(self.geometry)
        self.fixture_rows = self.choose_fixture_rows(args.fixture_rows)

    def choose_fixture_rows(self, text: str) -> tuple:
        """@brief The fixture rows of the companion tool. @param text "A,B" or "" @return two row indices
        @throws SystemExit when none are given and the geometry is not the registered one"""
        if text:
            rows = tuple(int(item) for item in text.split(","))
            if len(rows) != 2:
                raise SystemExit("--fixture-rows needs exactly two row indices")
            return rows
        if self.geometry_sha != REGISTERED_GEOMETRY_SHA:
            raise SystemExit("this geometry container is not the registered one: give --fixture-rows A,B (two rows with a W entry)")
        return REGISTERED_FIXTURE_ROWS

    def verify_tree_hashes(self) -> None:
        """@brief Checks every file of `tree/` against MANIFEST.sha256 (the pins of the verbatim tools).
        @throws SystemExit when a file is missing or its sha256 differs; a changed end-of-line style is the usual cause (check .gitattributes: `tree/** -text`, and core.autocrlf)"""
        manifest = HERE / "MANIFEST.sha256"
        checked = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, relative = line.split()[0], line.split()[1]
            path = HERE / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise SystemExit(f"{relative}: sha256 differs from MANIFEST.sha256 (or the file is missing). The packaged tools must be byte-identical to the pins; a changed line ending is the usual cause.")
            checked += 1
        print(f"tree/: {checked} files match MANIFEST.sha256", flush=True)

    def prepare_tools(self) -> Path:
        """@brief Copies `tree/` into the work folder and writes the task copies. @return the folder of the generated copies"""
        tree = self.work / "tree"
        if tree.exists():
            raise SystemExit(f"{tree} exists: use a new work folder")
        self.verify_tree_hashes()
        shutil.copytree(HERE / "tree", tree)
        generated = tree / "v2/hill_demo/e1/_generated"
        generated.mkdir()
        container = f'Path(r"{self.geometry}")'
        prefix = f'CONTAINER_SHA_PREFIX = "{self.geometry_sha[:16]}"'
        e2_119, e2_120 = tree / "v2/hill_demo/e2/task119", tree / "v2/hill_demo/e2/task120"
        derive(e2_119 / "w_moment_extractor.py", generated / "w_moment_extractor.py", [
            (f"CONTAINER = {OLD_CONTAINER_LINE}", f"CONTAINER = {container}"), (OLD_PREFIX_LINE, prefix),
            (OLD_CSV_LINE, f'CSV = Path(r"{self.heights}")'), (OLD_CSV_SHA_LINE, f'CSV_SHA = "{sha256_file(self.heights)}"'),
            (OLD_FIXTURE_LINE, f"FIXTURE_ROWS = {self.fixture_rows!r}")])
        derive(e2_120 / "srd_build.py", generated / "srd_build.py", [(f"CONTAINER = {OLD_CONTAINER_LINE}", f"CONTAINER = {container}"), (OLD_PREFIX_LINE, prefix)])
        shutil.copyfile(e2_120 / "srd_construct.py", generated / "srd_construct.py")
        nl, quote = chr(10), chr(34)
        derive(e2_119 / "make_t119_container.py", generated / "make_t119_container.py", [
            ("sys.path.insert(0, str(HERE))" + nl + "import r1_correction as R1", "sys.path.insert(0, str(HERE.parents[1] / " + quote + "e2" + quote + " / " + quote + "task119" + quote + "))" + nl + "import r1_correction as R1"),
            ("    if changed != [" + quote + "f0" + quote + "] or sorted(added)", "    if changed not in ([" + quote + "f0" + quote + "], []) or sorted(added)")])
        return generated

    def run_stage(self, label: str, argv: list) -> None:
        """@brief Runs one stage and stops the whole run when it fails. @param label the stage name @param argv the command
        @throws SystemExit when the exit code is not 0"""
        start = time.perf_counter()
        with (self.work / f"{label}_stdout.txt").open("w", encoding="utf-8") as out, (self.work / f"{label}_stderr.txt").open("w", encoding="utf-8") as err:
            code = subprocess.call([str(item) for item in argv], stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        seconds = round(time.perf_counter() - start, 1)
        self.record["stages"][label] = {"exit": code, "seconds": seconds}
        print(f"stage {label}: exit {code} ({seconds} s)", flush=True)
        if code != 0:
            self.write_record("STOPPED at " + label)
            raise SystemExit(f"stage {label} failed with exit {code}; see {self.work / (label + '_stderr.txt')}")

    def write_record(self, verdict: str) -> None:
        """@brief Writes preparation_record.json. @param verdict the result text"""
        self.record["verdict"] = verdict
        (self.work / "preparation_record.json").write_text(json.dumps(self.record, indent=1), encoding="utf-8")

    def validate_companion(self, record_path: Path, table_path: Path) -> None:
        """@brief Reads a companion record and checks it against the geometry container and the table. Used for a reused table AND for a freshly written one.
        @param record_path the JSON record of the companion tool
        @param table_path the npz table that belongs to it
        @throws SystemExit when a file is missing, the verdict is not PASS, the container sha256 differs from this geometry container, or the table hash differs"""
        if not record_path.is_file() or not table_path.is_file():
            raise SystemExit(f"companion record or table is missing: {record_path}, {table_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("verdict") != "PASS" or record.get("container_sha256") != self.geometry_sha or record.get("npz_sha256") != sha256_file(table_path):
            self.write_record("STOPPED: companion record is not a PASS for this geometry container, or its table hash differs")
            raise SystemExit("the companion record is not a PASS for this geometry container, or its table hash differs")
        self.record["companion_record"] = {"verdict": record["verdict"], "container_sha256": record["container_sha256"], "npz_sha256": record["npz_sha256"], "producer_rows_with_W": record.get("producer_rows_with_W"),
                                           "W_entries": record.get("W_entries"), "second_image_rows": record.get("second_image_rows")}

    def use_existing_companion(self) -> Path:
        """@brief Checks an existing companion table and returns its path. @return the npz path
        @throws SystemExit when the record is not a PASS for this geometry or the hashes differ"""
        self.validate_companion(self.companion_json, self.companion_npz)
        self.record["stages"]["companion"] = {"exit": 0, "seconds": 0.0, "reused": str(self.companion_npz)}
        return self.companion_npz

    def run(self) -> int:
        """@brief Runs the four stages. @return 0 when the final container is written"""
        self.work.mkdir(parents=True, exist_ok=True)
        generated = self.prepare_tools()
        py = sys.executable
        if self.companion_npz is not None:
            companion = self.use_existing_companion()
        else:
            (self.work / "companion").mkdir()
            stem = self.work / "companion/companion"
            self.run_stage("companion", [py, "-B", generated / "w_moment_extractor.py", "--release", self.label, "--out", stem, "--workers", str(self.workers)])
            companion = stem.with_suffix(".npz")
            self.validate_companion(stem.with_suffix(".json"), companion)      # the exit code alone is not trusted: the record must be a PASS for this container
        companion_sha = sha256_file(companion)
        t119 = self.work / "hill_t119.vwnat"
        self.run_stage("t119", [py, "-B", generated / "make_t119_container.py", "--release", self.label, "--base", self.geometry, "--base-sha", self.geometry_sha, "--companion", companion,
                                "--companion-sha", companion_sha, "--force", "0,0,0", "--out", t119])
        (self.work / "srd").mkdir()
        tables = self.work / "srd/srd"
        self.run_stage("srd", [py, "-B", generated / "srd_construct.py", "--release", self.label, "--out", tables])
        final = self.work / "hill_final.vwnat"
        self.run_stage("final", [py, "-B", self.work / "tree/v2/hill_demo/e1/make_t121_container.py", "--base", t119, "--base-sha", sha256_file(t119), "--tables", tables.with_suffix(".npz"),
                                 "--tables-sha", sha256_file(tables.with_suffix(".npz")), "--out", final])
        self.record.update(geometry_sha256=self.geometry_sha, companion_sha256=companion_sha, t119_container_sha256=sha256_file(t119), srd_tables_sha256=sha256_file(tables.with_suffix(".npz")),
                           final_container_sha256=sha256_file(final), fixture_rows=list(self.fixture_rows), final_equals_registered=sha256_file(final) == REGISTERED_FINAL_SHA)
        self.write_record("PREPARED")
        print("PREPARED: final container sha256", self.record["final_container_sha256"], "(equals the registered one: " + str(self.record["final_equals_registered"]) + ")", flush=True)
        return 0


def main() -> int:
    """@brief Command line entry. @return the exit code"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--heights", required=True, type=Path, help="the heights CSV the geometry container was built from")
    parser.add_argument("--geometry", required=True, type=Path, help="the geometry container (hill_d3q27.vwnat) from scripts/slip_wall/terrain_container.py")
    parser.add_argument("--workdir", required=True, type=Path, help="a new folder for all outputs")
    parser.add_argument("--workers", type=int, default=20, help="processes of the companion extraction")
    parser.add_argument("--fixture-rows", default="", help="two row indices A,B with a W entry (default: the registered rows for the registered geometry)")
    parser.add_argument("--release-label", default="PACKAGED", help="the label the research tools demand with --release")
    parser.add_argument("--companion-npz", type=Path, default=None, help="an existing companion table (stage 1 is skipped)")
    parser.add_argument("--companion-json", type=Path, default=None, help="the record of that companion table")
    args = parser.parse_args()
    if (args.companion_npz is None) != (args.companion_json is None):
        parser.error("--companion-npz and --companion-json go together")
    return Preparation(args).run()


if __name__ == "__main__":
    sys.exit(main())
