"""Task 119: reference extraction of the owned-piece first moments of the stored W entries (companion table).

STATUS: written, NOT executed. Refuses to run without --release (the scientist's release message id). Reference-only preprocessing:
no container, geometry or builder file is written or changed. The output is a hashed companion table aligned entry by entry with
the W sparse table of the pinned hill container.

Source (scientist #2842, #2846): the pinned geom_par builder (builder3d.py, terrain_container.py) computes every owned piece in
VelocityMomentTerrainBuilder.producer_pass -> producer_partition -> add_w(fkey, d, g, img, p), with the piece p in the frame of the
integer origin = f.cell (terrain_container.py geom_par 429-448). This tool subclasses that builder, makes add_h a no-op, and lets
add_w also keep the volume and the first moment of p. producer_partition is unchanged. It runs producer_pass only for the
(facet, direction) rows that have a W entry in the pinned container.

Definitions (all in the producer frame, no large global moments subtracted):
  C_full_local = centroid of the framed facet + c/2          (exact prism centroid, E2 #2794)
  m_piece      = moment_local - volume * C_full_local         (first moment of the piece about C_full)
  m_b          = sum of m_piece over the pieces and periodic images of one container W entry (recv_row, prod_row)
  delta_j      = (cell centroid + prism_arm) - C_full for each G entry of a row with W, taken at the periodic image nearest to
                 C_full. Reach (scientist #2850): a facet lies in one unit cell, any facet coordinate differs from its area
                 centroid by <= 1, and the extrusion c (t - 1/2) adds <= 1/2, so every prism point has |x - C_full| <= 1.5 per axis
                 in the producer frame. Stored offsets are checked against 1.5 + 1e-9, and |m_b| <= w_b (1.5 + 1e-9). With
                 L = 80, 64 > 3 the image inside the bound is unique; the tool checks the chosen image and that no other image
                 lies inside the bound.
Checks (recorded; nothing is corrected or allocated to satisfy them):
  1. the key set of the recomputed entries equals the key set of the container W exactly (receiver row, producer row);
  2. recomputed volume (math.fsum over pieces and images) equals the stored W weight: max |diff| / v <= 1e-13 (E1 registered); the count
     of bit-equal entries is reported;
  3. every moment is finite; every volume is finite and positive;
  4. partition: residual_r = M_r + sum_b m_b with M_r = sum_j h_j delta_j (exact zero for an exact partition); |residual| / v is
     reported per row and gated at PARTITION_REL_TOL (the engineering tolerance of task 118, not a proof); coverage
     |H + W - v| / v is gated at the same tolerance; selected G weights must be finite and positive;
     every gate violation is COUNTED and the residuals are written even when a gate fails (no early abort on a gate);
  5. the direction table, the row list and prism_volume of the container agree with the builder (vi = c.a bit-equal);
  6. H = 0 on a row with W is a MODEL-DOMAIN BLOCKER (exit 3, reported before any extraction), not a fallback;
  7. scheduling equivalence: a fixed deterministic serial subset (the two fixture rows, the row with the most destinations,
     the first 3 rows with W whose C_full is within 1.5 of a periodic edge, the first 2 rows with W) is recomputed serially in
     the parent and must equal the pool result exactly (keys and floats).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[3]
sys.path.insert(0, str(WALL / "integration" / "task39"))
import t39_lib as T39  # noqa: E402  (container section reader)

BUILDER_DIR = WALL / "ports" / "20261112_v2_geom_par" / "source" / "scripts" / "slip_wall"
BUILDER_SHA = {"builder3d.py": "58cff4cda16e78d054793259b29110bd5e6c8e630ba5d8e6ac1f73b598148bf5",
               "terrain_container.py": "4f6d9ad9932e74d104cab03eaced6fde5e7913395c3fd8646aef1d8752b47c04"}
STEEP_DIR = WALL / "v2" / "hill_inventory" / "e2" / "steep"
CONTAINER = WALL / "v2" / "hill_demo" / "e1" / "hf_inputs" / "hill_uniform.vwnat"
CONTAINER_SHA_PREFIX = "a404dc7463c9bc1a"
CSV = WALL / "v2" / "hill_demo" / "e2" / "input" / "hill_64_apron8_heights.csv"
CSV_SHA = "d60b66771befcef8776a22bb80416d86c6baec56cf7520e2b1cd3a22a919f0d6"
Z_TOP = 24
VOLUME_REL_TOL = 1e-13
PARTITION_REL_TOL = 1e-9
REACH = 1.5
REACH_TOL = 1e-9
FIXTURE_ROWS = (126454, 126478)


class DomainBlocker(RuntimeError):
    """A model-domain blocker (H = 0 on a row with W): reported before any evaluation, never a silent fallback."""


class Invalid(RuntimeError):
    """An input, a convention or a check is outside the contract: stop, change nothing."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_pinned():
    """The pinned builder modules, imported by path after the hash check; the steep manifest driver is hashed and recorded."""
    for name, sha in BUILDER_SHA.items():
        got = sha256_file(BUILDER_DIR / name)
        if got != sha:
            raise Invalid(f"{name} sha256 {got[:16]} is not the pinned {sha[:16]}")
    sys.path.insert(0, str(BUILDER_DIR))
    import terrain_container as T  # noqa: PLC0415
    sys.path.insert(0, str(STEEP_DIR))
    import steep_geometry_driver as SD  # noqa: PLC0415
    return T, SD


def hill_manifest(T, SD):
    got = sha256_file(CSV)
    if got != CSV_SHA:
        raise Invalid(f"hill heights CSV sha256 {got[:16]} is not the pinned {CSV_SHA[:16]}")
    base = T.TerrainManifest.from_csv(CSV, Z_TOP, "hill")
    return SD.steep_manifest_class(T)(base.name, base.Lx, base.Ly, base.Z_top, base.heights, base.diagonal)


# ---------------------------------------------------------------- worker side (module level: picklable under spawn)
_WORKER: dict = {}


def make_extractor_class(T):
    B = sys.modules["builder3d"]

    class OwnedPieceBuilder(T.VelocityMomentTerrainBuilder):
        """The pinned builder with add_h as a no-op and add_w also keeping volume and centred first moment of each owned piece."""

        def __init__(self, record, velocity_set):
            self.pieces: dict = {}
            self._frame_cache: dict = {}
            super().__init__(record, velocity_set)

        def add_h(self, fkey, d, p, origin=B.ORIGIN0):
            return

        def frame_centroid(self, fkey, origin):
            if fkey not in self._frame_cache:
                self._frame_cache[fkey] = self.facet_in_frame(self.facets[fkey], origin).centroid()
            return self._frame_cache[fkey]

        def add_w(self, fkey, d, g, img, p):
            super().add_w(fkey, d, g, img, p)
            volume, moment = p.moment()
            origin = tuple(self.facets[fkey].cell)
            cen = self.frame_centroid(fkey, origin)
            c = self.velocity_set.directions[d]
            cfull = tuple(cen[k] + 0.5 * c[k] for k in range(3))
            centred = tuple(moment[k] - volume * cfull[k] for k in range(3))
            self.pieces.setdefault((fkey, d, g.key, img), []).append((float(volume), centred))

    return OwnedPieceBuilder


def worker_init(record: dict, velocities: str, wanted: dict) -> None:
    try:
        T, _ = import_pinned()
        cls = make_extractor_class(T)
        _WORKER.update(b=cls(record, T.VelocitySet.named(velocities)), wanted=wanted)
    except Exception as error:  # noqa: BLE001  (kept and raised by every task: fail closed)
        _WORKER.update(error=f"{type(error).__name__}: {error}")


def worker_facet(fkey) -> list:
    if "error" in _WORKER:
        raise RuntimeError(f"INVALID: worker initialization failed: {_WORKER['error']}")
    b = _WORKER["b"]
    out = []
    for d in _WORKER["wanted"][fkey]:
        b.pieces = {}
        b.producer_pass(fkey, d)
        out.append((fkey, d, b.vi[(fkey, d)], dict(b.pieces)))
    return out


# ---------------------------------------------------------------- parent side
class ContainerTables:
    """The container arrays used for the row map, the W keys and the delta_j of the rows with W."""

    def __init__(self) -> None:
        raw = CONTAINER.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        if not self.sha.startswith(CONTAINER_SHA_PREFIX):
            raise Invalid(f"container hash {self.sha[:16]} is not the pinned {CONTAINER_SHA_PREFIX}")
        a = T39.arrays(T39.read_sections(raw))
        self.box = np.asarray(a["box"], dtype=np.int64)
        counts = np.asarray(a["counts"], dtype=np.int64)
        self.cells, self.facets, self.rows = (int(x) for x in counts)
        self.q = 27
        self.centroid = np.asarray(a["centroid"], dtype=np.float64).reshape(-1, 3)
        self.row_direction = np.asarray(a["row_direction"], dtype=np.int64)
        self.prism_volume = np.asarray(a["prism_volume"], dtype=np.float64)
        offsets = np.asarray(a["facet_offsets"], dtype=np.int64)
        self.facet_of_row = np.repeat(np.arange(self.facets), np.diff(offsets))
        self.prism_arm = np.asarray(a["prism_arm"], dtype=np.float64).reshape(-1, 3)
        self.G = self._csr(a, "G")
        self.W = self._csr(a, "W")
        if self.W.shape != (self.rows, self.rows) or self.G.shape != (self.rows, self.cells * self.q):
            raise Invalid(f"W {self.W.shape} / G {self.G.shape} do not match rows {self.rows}, cells {self.cells}")
        if len(self.prism_arm) != self.G.nnz:
            raise Invalid("prism_arm length differs from G nnz")
        if self.box[0] <= 3 or self.box[1] <= 3:
            raise Invalid(f"box {self.box.tolist()}: x and y must exceed 3 cells for a unique image inside the reach")
        for name in ("centroid", "prism_volume", "prism_arm"):
            if not np.isfinite(getattr(self, name)).all():
                raise Invalid(f"{name} has a non-finite value")
        if not np.isfinite(self.W.data).all() or not (self.W.data > 0).all():
            raise Invalid("a stored W weight is not finite and positive")
        if not np.isfinite(self.G.data).all():
            raise Invalid("a stored G weight is not finite")

    @staticmethod
    def _csr(a: dict, name: str) -> csr_matrix:
        shape = tuple(int(x) for x in a[f"{name}_shape"])
        return csr_matrix((np.asarray(a[f"{name}_weights"], dtype=np.float64), np.asarray(a[f"{name}_indices"], dtype=np.int64),
                           np.asarray(a[f"{name}_offsets"], dtype=np.int64)), shape=shape)


class MomentExtractor:
    """Runs the owned-piece extraction on the producer rows with a W entry and validates the result against the container."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.T, self.SD = import_pinned()
        self.t = ContainerTables()
        self.manifest = hill_manifest(self.T, self.SD)
        self.manifest.domain_checks()
        self.record = self.manifest.record()
        self.record_sha = hashlib.sha256(json.dumps(self.record, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        self.vs = self.T.VelocitySet.named("D3Q27")
        self.builder = make_extractor_class(self.T)(self.record, self.vs)

    def has_fluid_gather(self, row: int) -> bool:
        lo, hi = int(self.t.G.indptr[row]), int(self.t.G.indptr[row + 1])
        return hi > lo and math.fsum(self.t.G.data[lo:hi].tolist()) > 0.0

    def row_map(self) -> tuple:
        """container row -> (facet key, direction); the inverse; checks vi = c.a against prism_volume (bit-equal)."""
        t, b = self.t, self.builder
        facet_keys = sorted(b.facets, key=str)
        if len(facet_keys) != t.facets:
            raise Invalid(f"builder facets {len(facet_keys)} != container facets {t.facets}")
        B = sys.modules["builder3d"]
        key_of_row, row_of_key = [], {}
        for r in range(t.rows):
            fk, d = facet_keys[int(t.facet_of_row[r])], int(t.row_direction[r])
            f, c = b.facets[fk], self.vs.directions[d]
            if f.sign(c) <= 0:
                raise Invalid(f"container row {r} (facet {fk}, direction {d}) is not an exact-positive producer row in the builder")
            if B.dotf(f.a, c) != float(t.prism_volume[r]):
                raise Invalid(f"container row {r}: builder vi {B.dotf(f.a, c)!r} != prism_volume {float(t.prism_volume[r])!r}")
            key_of_row.append((fk, d))
            row_of_key[(fk, d)] = r
        return key_of_row, row_of_key

    def run(self) -> dict:
        t = self.t
        key_of_row, row_of_key = self.row_map()
        Wc = csc_matrix(t.W)
        prod_rows = [r for r in range(t.rows) if Wc.indptr[r + 1] > Wc.indptr[r]]
        zero_h = [r for r in prod_rows if not self.has_fluid_gather(r)]
        if zero_h:
            raise DomainBlocker(f"{len(zero_h)} rows with W have H = 0 (first {zero_h[:10]}); the all-modified-row candidate is undefined there")
        wanted: dict = {}
        for r in prod_rows:
            fk, d = key_of_row[r]
            wanted.setdefault(fk, []).append(d)
        context = multiprocessing.get_context("spawn")
        start = time.time()
        gathered: dict = {}
        with context.Pool(self.workers, initializer=worker_init, initargs=(self.record, "D3Q27", wanted)) as pool:
            for n, result in enumerate(pool.imap_unordered(worker_facet, sorted(wanted, key=str), chunksize=4)):
                for fk, d, vi, pieces in result:
                    gathered[(fk, d)] = (vi, pieces)
                if (n + 1) % max(1, len(wanted) // 20) == 0:
                    print(f"  facet {n + 1}/{len(wanted)} {time.time() - start:.0f} s", flush=True)
        seconds = time.time() - start
        return self.validate(key_of_row, row_of_key, prod_rows, gathered, seconds)

    def validate(self, key_of_row: list, row_of_key: dict, prod_rows: list, gathered: dict, seconds: float) -> dict:
        t = self.t
        opp = self.vs.opposites
        volume_of: dict = {}
        moment_of: dict = {}
        for prod in prod_rows:
            fk, d = key_of_row[prod]
            if (fk, d) not in gathered:
                raise Invalid(f"no extraction result for producer row {prod}")
            for (pfk, pd, gkey, img), pieces in gathered[(fk, d)][1].items():
                recv_key = (gkey, opp[pd])
                if recv_key not in row_of_key:
                    raise Invalid(f"receiver row {recv_key} of producer row {prod} does not exist in the container")
                key = (row_of_key[recv_key], prod)
                volume_of.setdefault(key, []).extend(v for v, _ in pieces)
                moment_of.setdefault(key, []).extend(m for _, m in pieces)
        stored = {}
        W = t.W.tocoo()
        for r, c, w in zip(W.row.tolist(), W.col.tolist(), W.data.tolist()):
            stored[(r, c)] = stored.get((r, c), 0.0) + w
        if set(stored) != set(volume_of):
            raise Invalid(f"key sets differ: {len(set(stored) - set(volume_of))} stored-only, {len(set(volume_of) - set(stored))} recomputed-only")
        csr = t.W.tocsr()
        nnz = csr.nnz
        entry_recv = np.repeat(np.arange(t.rows), np.diff(csr.indptr))
        entry_prod = csr.indices.astype(np.int64)
        volume = np.empty(nnz)
        m_b = np.empty((nnz, 3))
        for n in range(nnz):
            key = (int(entry_recv[n]), int(entry_prod[n]))
            volume[n] = math.fsum(volume_of[key])
            m_b[n] = [math.fsum(m[k] for m in moment_of[key]) for k in range(3)]
        if not (np.isfinite(volume).all() and np.isfinite(m_b).all() and (volume > 0).all()):
            raise Invalid("a recomputed volume or moment is not finite or a volume is not positive")
        v_prod = t.prism_volume[entry_prod]
        diff = np.abs(volume - csr.data) / v_prod
        bit_equal = int(np.count_nonzero(volume == csr.data))
        duplicates = len(csr.data) - len(stored)
        return self.partition(prod_rows, csr, entry_prod, volume, m_b, diff, bit_equal, duplicates, seconds, key_of_row, gathered)

    def serial_subset(self, prod_rows: list, entry_prod: np.ndarray, c_full: np.ndarray) -> list:
        """The fixed deterministic subset for the scheduling-equivalence check (rule: module docstring, item 7)."""
        t = self.t
        chosen = []
        for r in FIXTURE_ROWS:
            if r not in prod_rows:
                raise Invalid(f"fixture row {r} has no W entry")
            chosen.append(r)
        counts = np.bincount(entry_prod, minlength=t.rows)
        chosen.append(int(np.argmax(counts)))
        edge = [r for n, r in enumerate(prod_rows)
                if any(c_full[n, a] < REACH or c_full[n, a] > t.box[a] - REACH for a in (0, 1))]
        chosen.extend(edge[:3])
        chosen.extend(prod_rows[:2])
        return sorted(set(chosen))

    def serial_equivalence(self, subset: list, gathered: dict, key_of_row: list) -> dict:
        b = self.builder
        bad = []
        for r in subset:
            fk, d = key_of_row[r]
            b.pieces = {}
            b.producer_pass(fk, d)
            if b.pieces != gathered[(fk, d)][1]:
                bad.append(r)
        return {"serial_subset": subset, "serial_equal": not bad, "serial_mismatch_rows": bad}

    def partition(self, prod_rows, csr, entry_prod, volume, m_b, diff, bit_equal, duplicates, seconds, key_of_row, gathered) -> dict:
        """C_full and delta_j for the rows with W; the partition residual M + sum m_b per row."""
        t, b = self.t, self.builder
        L = t.box[:2].astype(np.float64)
        z_lo = min(k[2] for k in b.cells) - 1          # terrain_container.sections: container z = builder z - z_lo
        if int(t.box[2]) != self.manifest.Z_top - z_lo + 1:
            raise Invalid(f"box z {int(t.box[2])} != Z_top - z_lo + 1 = {self.manifest.Z_top - z_lo + 1}")
        c_full = np.zeros((len(prod_rows), 3))
        g_index, delta_all, residual = [], [], np.zeros((len(prod_rows), 3))
        reach_max = 0.0
        second_image_rows = 0
        coverage = np.zeros(len(prod_rows))
        nonpositive_h_rows = 0
        for n, r in enumerate(prod_rows):
            fk, d = key_of_row[r]
            f = b.facets[fk]
            cen = f.centroid()
            c = self.vs.directions[d]
            c_full[n] = [cen[0] + 0.5 * c[0], cen[1] + 0.5 * c[1], cen[2] + 0.5 * c[2] - z_lo]
            lo, hi = int(t.G.indptr[r]), int(t.G.indptr[r + 1])
            h = t.G.data[lo:hi]
            nonpositive_h_rows += int(not (h > 0).all())
            cells = t.G.indices[lo:hi] // t.q
            delta = t.centroid[cells] + t.prism_arm[lo:hi] - c_full[n]
            for axis in (0, 1):
                delta[:, axis] -= L[axis] * np.round(delta[:, axis] / L[axis])
                other = np.minimum(np.abs(delta[:, axis] + L[axis]), np.abs(delta[:, axis] - L[axis]))
                second_image_rows += int((other <= REACH + REACH_TOL).any())
            reach_max = max(reach_max, float(np.abs(delta).max()))
            g_index.append(np.arange(lo, hi))
            delta_all.append(delta)
            M = np.array([math.fsum((h * delta[:, k]).tolist()) for k in range(3)])
            mask = entry_prod == r
            residual[n] = M + m_b[mask].sum(axis=0)
            coverage[n] = abs(math.fsum(h.tolist()) + math.fsum(volume[mask].tolist()) - t.prism_volume[r]) / t.prism_volume[r]
        moment_over = int((np.abs(m_b).max(axis=1) > volume * (REACH + REACH_TOL)).sum())
        v_rows = t.prism_volume[np.array(prod_rows)]
        part_rel = np.abs(residual).max(axis=1) / v_rows
        verdict = "PASS"
        reasons = []
        if diff.max() > VOLUME_REL_TOL:
            verdict = "FAILED"
            reasons.append(f"max recomputed-vs-stored W volume difference {diff.max()!r} v exceeds {VOLUME_REL_TOL}")
        if part_rel.max() > PARTITION_REL_TOL:
            verdict = "FAILED"
            reasons.append(f"max partition residual {part_rel.max()!r} v exceeds {PARTITION_REL_TOL}")
        if coverage.max() > PARTITION_REL_TOL:
            verdict = "FAILED"
            reasons.append(f"max |H + W - v| / v = {coverage.max()!r} exceeds {PARTITION_REL_TOL}")
        if reach_max > REACH + REACH_TOL:
            verdict = "FAILED"
            reasons.append(f"max |delta_j| = {reach_max!r} exceeds the reach {REACH} + {REACH_TOL}")
        if second_image_rows:
            verdict = "FAILED"
            reasons.append(f"{second_image_rows} rows have a second periodic image inside the reach")
        if moment_over:
            verdict = "FAILED"
            reasons.append(f"{moment_over} W entries have |m_b| above w_b (1.5 + 1e-9)")
        if nonpositive_h_rows:
            verdict = "FAILED"
            reasons.append(f"{nonpositive_h_rows} rows have a non-positive selected G weight")
        arrays = {"prod_rows": np.array(prod_rows, dtype=np.int64), "c_full": c_full, "w_recv_row": np.repeat(np.arange(t.rows), np.diff(csr.indptr)),
                  "w_prod_row": entry_prod, "w_volume_recomputed": volume, "w_volume_stored": csr.data, "w_moment": m_b,
                  "g_entry_index": np.concatenate(g_index), "g_delta": np.concatenate(delta_all), "partition_residual": residual,
                  "coverage_rel": coverage}
        subset = self.serial_subset(prod_rows, entry_prod, c_full)
        equivalence = self.serial_equivalence(subset, gathered, key_of_row)
        if not equivalence["serial_equal"]:
            verdict = "FAILED"
            reasons.append(f"serial subset differs from the pool result on rows {equivalence['serial_mismatch_rows']}")
        record = {**equivalence, "manifest_record_sha256": self.record_sha, "verdict": verdict, "reasons": reasons, "producer_rows_with_W": len(prod_rows), "W_entries": int(csr.nnz),
                  "stored_duplicate_entries": int(duplicates), "volume_bit_equal_entries": bit_equal,
                  "volume_max_rel_diff": float(diff.max()), "partition_max_rel_residual": float(part_rel.max()),
                  "delta_reach_max": reach_max, "coverage_max_rel": float(coverage.max()), "second_image_rows": second_image_rows,
                  "moment_over_bound_entries": moment_over, "nonpositive_h_rows": nonpositive_h_rows, "producer_keys_expected": len(csr.data), "z_lo": int(z_lo), "extraction_seconds": seconds}
        return {"record": record, "arrays": arrays}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", required=True, help="id of the scientist release message; the tool refuses without it")
    parser.add_argument("--out", required=True, type=Path, help="output stem; writes STEM.json and STEM.npz")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1:
        raise Invalid("workers must be >= 1")
    try:
        result = MomentExtractor(args.workers).run()
    except DomainBlocker as blocker:
        blocked = {"release": args.release, "verdict": "MODEL_DOMAIN_BLOCKER", "reason": str(blocker), "tool_sha256": sha256_file(Path(__file__))}
        args.out.with_suffix(".json").write_text(json.dumps(blocked, indent=1))
        print(f"MODEL_DOMAIN_BLOCKER: {blocker}")
        return 3
    npz = args.out.with_suffix(".npz")
    np.savez(npz, **result["arrays"])
    meta = {"release": args.release, "container_sha256": sha256_file(CONTAINER), "csv_sha256": sha256_file(CSV),
            "builder_sha256": {n: sha256_file(BUILDER_DIR / n) for n in BUILDER_SHA},
            "steep_driver_sha256": sha256_file(STEEP_DIR / "steep_geometry_driver.py"), "tool_sha256": sha256_file(Path(__file__)),
            "npz_sha256": sha256_file(npz), "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "workers": args.workers,
            "loaded_local_modules": {str(Path(m.__file__)): sha256_file(Path(m.__file__)) for m in list(sys.modules.values())
                                     if getattr(m, "__file__", None) and str(m.__file__).startswith(str(WALL))}}
    args.out.with_suffix(".json").write_text(json.dumps({**meta, **result["record"]}, indent=1))
    print(json.dumps(result["record"]))
    return 1 if result["record"]["verdict"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
