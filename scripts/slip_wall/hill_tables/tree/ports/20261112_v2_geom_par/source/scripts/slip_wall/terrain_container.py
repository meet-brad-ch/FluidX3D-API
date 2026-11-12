"""Offline builder of a VWNAT01 container for the volumetric slip wall on terrain (VOLUMETRIC_WALL_RR27).

    python terrain_container.py --heights HEIGHTS.csv --z-top 16 --nu 0.1 --friction 0.06 --out terrain.vwnat

HEIGHTS.csv: Ny lines of Nx comma-separated node heights (lattice units, binary64), periodic in x and y. The diagonals of the
columns follow the lowest-node rule. The box is periodic in x and y with a flat slip lid at z = Z_top. The container gets
D3Q27 tables, the scalars (nu, omega = 1/(3 nu + 1/2), K), the rest initial state f0 = w and s0 = W w_rd, the friction rows,
the arms and the gradient map. The force is a run-time input of the simulation, not part of the container.

Dependencies: Python 3.10 or newer, numpy, scipy. No other module.

SOURCE. Every function below except the CLI and the task-103 changes (at the end of this text) is copied from the research
tree of the qualified model (branch volumetric-wall-model; files and SHA-256 in PROVENANCE.md), with the statement order and
the operation order of the original. HISTORICAL: before task 103 the result was the same bytes as the originals (the controls
below); the task-103 builder changes the maps by round-off and claims no byte equivalence:
  builder3d.py (copied whole)          research/wall_model/integration/task26/builder3d.py
  domain checks R1, R10, R11           research/wall_model/integration/task26/oracle3d.py (plane_of, connectivity_check,
                                       domain_checks; adapted from methods of oracle3d.Case to functions of a manifest)
  lowest_node_diagonals, height_text,  research/wall_model/integration/task26/manifest3d.py
  the manifest record
  stencil_weights                      research/wall_model/integration/task22/terrain_builder.py
  Writer, DTYPE_*                      research/wall_model/integration/export_native_input.py
  VelocitySet, identities, permute_    research/wall_model/integration/task87/t87_geometry.py (VelocitySet, MapAssembler
  rows, base_rows, arms                identities/permute_rows/base_rows, VelocityExport.arms)
  the velocity-set builder, assembly,  research/wall_model/integration/task95/t95_geometry.py (TerrainBuilderFactory,
  gradient map, wall list, sections    TerrainAssembler, TerrainGradient, TerrainChain.wall_list, TerrainChain.sections)
Controls of the pre-task-103 builder (historical, PROVENANCE.md): with --velocities D3Q19 the numeric sections equal the accepted M3 container (3ba81966); with D3Q27
they equal the task-95 M3 container (348f3ded), except the provenance and the scalars.

Task 103 changes (not in the originals; PROVENANCE.md):
  - the donor and producer passes skip a prism that meets the region only in a face, an edge or a point (an exact integer
    test before any clip); a binary64 clip against such a plane gave a wrong split volume in the L24 example;
  - NativeIdentityGate: the four map identities with the native definitions and tolerance; a violation writes
    OUT.rejected.json and no container (exit 3).
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import json
import math
import platform
import struct
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import builder3d as B  # noqa: E402  (copied whole from task26)

# ---------------------------------------------------------------- export_native_input.py: the container writer
DTYPE_U32, DTYPE_I32, DTYPE_F64, DTYPE_TEXT = 1, 2, 3, 4


class Writer:
    """Sectioned VWNAT01 writer: magic, count, then (name[32], dtype u32, count u64, bytes)."""

    def __init__(self) -> None:
        self.sections: list[tuple[str, int, bytes]] = []

    def array(self, name: str, dtype: int, values: np.ndarray) -> None:
        kind = {DTYPE_U32: "<u4", DTYPE_I32: "<i4", DTYPE_F64: "<f8"}[dtype]
        self.sections.append((name, dtype, np.ascontiguousarray(values, dtype=kind).tobytes()))

    def text(self, name: str, value: str) -> None:
        self.sections.append((name, DTYPE_TEXT, value.encode("ascii")))

    def csr(self, name: str, matrix) -> None:
        m = matrix.tocsr()
        m.sort_indices()
        self.array(f"{name}_shape", DTYPE_U32, np.array(m.shape))
        self.array(f"{name}_offsets", DTYPE_U32, m.indptr)
        self.array(f"{name}_indices", DTYPE_U32, m.indices)
        self.array(f"{name}_weights", DTYPE_F64, m.data)

    def write(self, path: Path) -> None:
        with open(path, "wb") as out:
            out.write(b"VWNAT01\n")
            out.write(struct.pack("<I", len(self.sections)))
            for name, dtype, payload in self.sections:
                element = {DTYPE_U32: 4, DTYPE_I32: 4, DTYPE_F64: 8, DTYPE_TEXT: 1}[dtype]
                out.write(name.encode("ascii").ljust(32, b"\0"))
                out.write(struct.pack("<IQ", dtype, len(payload) // element))
                out.write(payload)


# ---------------------------------------------------------------- task22/terrain_builder.py: the sampler stencil
NEIGHBOURS = 12


def stencil_weights(cand: list, Ny: int) -> tuple:
    """Sampler convention (scientist #398). `cand`: sorted (d2, cell index, layer j, x shift, y shift, delta) candidates.
    DIRECT_CENTROID branch: the target coincides EXACTLY (d2 == 0) with ONE physical cell centroid: weight 1, no solve.
    Two DISTINCT physical cells at d2 == 0 are rejected. Otherwise LEAST_SQUARES (12 nearest, weights 1/d2, affine)."""
    exact = {(z[1], z[2] % Ny): z for z in cand if z[0] == 0.0}
    if len(exact) > 1:
        raise B.GeometryRejected(f"R3: sample coincides with {len(exact)} distinct cell centroids (ambiguous)")
    if len(exact) == 1:
        z = next(iter(exact.values()))
        return "direct_centroid", [z], np.array([1.0]), None, None
    chosen = cand[:NEIGHBOURS]
    delta = np.array([z[5] for z in chosen])
    Bm = np.c_[np.ones(len(chosen)), delta]
    w = 1 / np.array([z[0] for z in chosen])
    Mn = Bm.T @ (w[:, None] * Bm)
    if np.linalg.matrix_rank(np.sqrt(w)[:, None] * Bm) < 4:
        raise B.GeometryRejected("R3: stencil rank < 4")
    coeff = (w[:, None] * Bm) @ np.linalg.solve(Mn, np.array([1.0, 0, 0, 0]))
    return "least_squares", chosen, coeff, float(np.linalg.cond(Mn)), float(np.max(np.abs(coeff @ Bm - np.array([1.0, 0, 0, 0]))))


# ---------------------------------------------------------------- task26/manifest3d.py and oracle3d.py: the manifest and its rules
def lowest_node_diagonals(heights, Lx, Ly):
    """The diagonal through the column's lowest node; ties by the order (0,0), (1,0), (0,1), (1,1)."""
    out = {}
    for i in range(Lx):
        for j in range(Ly):
            nodes = [((0, 0), heights[(i, j)]), ((1, 0), heights[((i + 1) % Lx, j)]), ((0, 1), heights[(i, (j + 1) % Ly)]),
                     ((1, 1), heights[((i + 1) % Lx, (j + 1) % Ly)])]
            low = min(nodes, key=lambda e: e[1])[0]
            out[(i, j)] = "main" if low in ((0, 0), (1, 1)) else "anti"
    return out


def height_text(h: Fraction) -> str:
    try:
        f = float(h)
        if Fraction(f) == h:
            return f.hex()
    except OverflowError:
        pass
    return f"{h.numerator}/{h.denominator}"


def plane_of(tri):
    """(alpha, beta, gamma) with z = alpha x + beta y + gamma through the three vertices (exact)."""
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    alpha = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / det
    beta = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det
    return alpha, beta, z0 - alpha * x0 - beta * y0


@dataclass
class TerrainManifest:
    """A height field (oracle3d.Case data): exact node heights, the column diagonals, the lid."""
    name: str
    Lx: int
    Ly: int
    Z_top: int
    heights: dict      # (i, j) -> Fraction
    diagonal: dict     # (i, j) -> "main" or "anti"

    def h(self, i, j):
        return self.heights[(i % self.Lx, j % self.Ly)]

    def column_nodes(self, i, j):
        return {(0, 0): self.h(i, j), (1, 0): self.h(i + 1, j), (0, 1): self.h(i, j + 1), (1, 1): self.h(i + 1, j + 1)}

    def triangles(self, i, j):
        nd = self.column_nodes(i, j)
        P = lambda a, b: (Fraction(i + a), Fraction(j + b), nd[(a, b)])  # noqa: E731
        if self.diagonal[(i % self.Lx, j % self.Ly)] == "main":
            return [("T0", (P(0, 0), P(1, 0), P(1, 1))), ("T1", (P(0, 0), P(1, 1), P(0, 1)))]
        return [("T0", (P(0, 0), P(1, 0), P(0, 1))), ("T1", (P(1, 0), P(1, 1), P(0, 1)))]

    def diagonal_nodes(self, i, j):
        nd = self.column_nodes(i, j)
        if self.diagonal[(i % self.Lx, j % self.Ly)] == "main":
            return (nd[(0, 0)], nd[(1, 1)]), (nd[(1, 0)], nd[(0, 1)])
        return (nd[(1, 0)], nd[(0, 1)]), (nd[(0, 0)], nd[(1, 1)])

    def domain_checks(self) -> None:
        """oracle3d.domain_checks with both flags on: R10 lid clearance >= 2, R11 slopes |a|, |b| < 1, R1 connectivity."""
        hmax = max(self.heights.values())
        if self.Z_top - hmax < 2:
            raise B.GeometryRejected(f"R10: clearance {self.Z_top - hmax} < 2")
        for i in range(self.Lx):
            for j in range(self.Ly):
                for tag, tri in self.triangles(i, j):
                    a, b, _ = plane_of(tri)
                    if abs(a) >= 1 or abs(b) >= 1:
                        raise B.GeometryRejected(f"R11: slope ({a}, {b}) at column ({i}, {j}) {tag}")
        for i in range(self.Lx):
            for j in range(self.Ly):
                diag, off = self.diagonal_nodes(i, j)
                lo, hi = max(off), min(diag)
                K = math.floor(lo) + 1
                if K <= hi:
                    raise B.GeometryRejected(f"R1: column ({i}, {j}) disconnected at level {K} (max off {lo}, min diag {hi})")

    def record(self) -> dict:
        """manifest3d.manifest_record (the builder3d.load_manifest format)."""
        return {"name": self.name, "Lx": self.Lx, "Ly": self.Ly, "Z_top": self.Z_top, "check_slope": True, "check_clearance": True,
                "heights": {f"{i},{j}": height_text(h) for (i, j), h in sorted(self.heights.items())},
                "diagonal": {f"{i},{j}": d for (i, j), d in sorted(self.diagonal.items())}}

    @staticmethod
    def from_csv(path: Path, z_top: int, name: str) -> "TerrainManifest":
        rows = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        grid = [[float(v) for v in row.split(",")] for row in rows]
        Ly, Lx = len(grid), len(grid[0])
        if any(len(row) != Lx for row in grid) or Lx < 2 or Ly < 2:
            raise ValueError("HEIGHTS.csv: every line needs the same number (>= 2) of values, and >= 2 lines")
        if not all(math.isfinite(v) for row in grid for v in row):
            raise ValueError("HEIGHTS.csv: a height is not finite")
        heights = {(i, j): Fraction(grid[j][i]) for i in range(Lx) for j in range(Ly)}
        return TerrainManifest(name, Lx, Ly, int(z_top), heights, lowest_node_diagonals(heights, Lx, Ly))


# ---------------------------------------------------------------- task87/t87_geometry.py: velocity set and map identities
@dataclass(frozen=True)
class VelocitySet:
    """One lattice velocity set in LEXICOGRAPHIC order (x slowest, z fastest)."""

    name: str
    directions: tuple
    weights: tuple
    opposites: tuple
    rest: int

    @property
    def q(self) -> int:
        return len(self.directions)

    @property
    def c_array(self) -> np.ndarray:
        return np.array(self.directions, float)

    @property
    def w_array(self) -> np.ndarray:
        return np.array(self.weights, float)

    @staticmethod
    def named(name: str) -> "VelocitySet":
        """D3Q19: |c|^2 <= 2, weights 1/3, 1/18, 1/36. D3Q27: all 27, weights 8/27, 2/27, 1/54, 1/216."""
        tables = {"D3Q19": (2, {0: Fraction(1, 3), 1: Fraction(1, 18), 2: Fraction(1, 36)}),
                  "D3Q27": (3, {0: Fraction(8, 27), 1: Fraction(2, 27), 2: Fraction(1, 54), 3: Fraction(1, 216)})}
        if name not in tables:
            raise ValueError(f"unknown velocity set {name}; expected one of {sorted(tables)}")
        limit, weight_of_shell = tables[name]
        directions = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x * x + y * y + z * z <= limit)
        weights = tuple(float(weight_of_shell[c[0] * c[0] + c[1] * c[1] + c[2] * c[2]]) for c in directions)
        opposites = tuple(directions.index((-c[0], -c[1], -c[2])) for c in directions)
        rest = directions.index((0, 0, 0))
        if sum(Fraction(weight_of_shell[c[0] * c[0] + c[1] * c[1] + c[2] * c[2]]) for c in directions) != 1:
            raise ValueError(f"{name}: weights do not sum to 1")
        return VelocitySet(name, directions, weights, opposites, rest)


class MapIdentities:
    """t87_geometry.MapAssembler: identities, permute_rows, base_rows (velocity set as a parameter)."""

    def __init__(self, vs: VelocitySet) -> None:
        self.vs = vs

    def identities(self, m: dict) -> dict:
        V, D, G, R, W, vi = m["V"], m["D"], m["G"], m["R"], m["W"], m["vi"]
        Vs = np.repeat(V, self.vs.q)
        donor = Vs @ D + np.ones(G.shape[0]) @ G - Vs
        recv = D @ np.ones(D.shape[1]) + R @ vi - np.ones(len(Vs))
        prism = G @ np.ones(G.shape[1]) + W @ np.ones(W.shape[1]) - vi
        ret = Vs @ R + (np.ones(W.shape[0]) @ W) / vi - np.ones(len(vi))
        return {"donor_partition": float(np.max(np.abs(donor))), "receiver_coverage": float(np.max(np.abs(recv))),
                "prism_coverage": float(np.max(np.abs(prism))), "return_delivery": float(np.max(np.abs(ret)))}

    def permute_rows(self, m: dict, rf: np.ndarray, rd: np.ndarray) -> dict:
        rows = len(rf)
        order = np.argsort(rf, kind="stable")
        P = csr_matrix((np.ones(rows), (np.arange(rows), order)), shape=(rows, rows))
        s0 = m["W"] @ self.vs.w_array[rd]
        return {"D": m["D"].tocsr(), "S": m["S"].tocsr(), "G": (P @ m["G"]).tocsr(), "R": (m["R"] @ P.T).tocsr(),
                "W": (P @ m["W"] @ P.T).tocsr(), "rf": rf[order], "rd": rd[order], "vi": m["vi"][order], "s0": s0[order],
                "order": order}

    def base_rows(self, arrays: dict) -> np.ndarray:
        """phi0 = 6 w v^2 / A (c - (c.n) n) for every row (t55_pack.base_rows)."""
        direction = arrays["row_direction"].astype(np.int64)
        offsets = arrays["facet_offsets"].astype(np.int64)
        facets = len(arrays["area"])
        facet_of_row = np.repeat(np.arange(facets), np.diff(offsets))
        c = self.vs.c_array[direction].astype(float)
        w = self.vs.w_array[direction]
        prism = np.asarray(arrays["prism_volume"], float)
        area = np.asarray(arrays["area"], float)[facet_of_row]
        normal = np.asarray(arrays["normal"], float).reshape(facets, 3)[facet_of_row]
        tangent = c - (c * normal).sum(axis=1)[:, None] * normal
        return (6.0 * w * prism ** 2 / area)[:, None] * tangent


def arms(vs: VelocitySet, builder, maps: dict, permuted: dict) -> dict:
    """t87_geometry.VelocityExport.arms: part centroid - fluid centroid of the cell, per D and G entry."""
    q = vs.q
    keys, rows = maps["keys"], maps["rows"]
    centroid_local = np.array([builder.cells[k]["centroid_local"] for k in keys])   # task 106: about the cell corner, from the local frame
    direct = permuted["D"].tocoo()
    direct_arm = np.zeros((direct.nnz, 3))
    for n, column in enumerate(direct.col.tolist()):
        cell, d = column // q, column % q
        part = builder.direct_moment[(keys[cell], d)]
        direct_arm[n] = np.array(part[1:]) / part[0] - centroid_local[cell]
    gather = permuted["G"].tocoo()
    order = np.asarray(permuted["order"])
    prism_arm = np.zeros((gather.nnz, 3))
    for n, (row, column) in enumerate(zip(gather.row.tolist(), gather.col.tolist())):
        fkey, d = rows[int(order[row])]
        cell = column // q
        if column % q != vs.opposites[d]:
            raise RuntimeError("INVALID: a gather entry is not the opposite direction of its row")
        part = builder.h_moment[(fkey, d, keys[cell])]
        prism_arm[n] = np.array(part[1:]) / part[0] - centroid_local[cell]
    return {"direct_arm": direct_arm, "prism_arm": prism_arm, "wall_cells": np.unique(gather.col // q)}


# ---------------------------------------------------------------- parallel pass workers (module level: picklable under spawn)
_PASS_WORKER: dict = {}


def _pass_worker_init(record: dict, velocities: str) -> None:
    """Builds the same builder as the parent (cells and facets from the record) and records the Diag events of its passes:
    split() runs on a fresh probe Diag (0.0 + defect = defect exactly) and note_small appends the small volumes.
    An exception here is kept and raised by the first task: multiprocessing.Pool restarts a worker whose initializer fails,
    so a failing initializer would never end the build."""
    try:
        _pass_worker_setup(record, velocities)
    except Exception as error:  # noqa: BLE001  (kept and raised by every task: fail closed)
        _PASS_WORKER.update(error=f"{type(error).__name__}: {error}")


def _pass_worker_builder():
    """The worker's builder, or the error of its initializer as a RuntimeError."""
    if "error" in _PASS_WORKER:
        raise RuntimeError(f"INVALID: parallel worker initialization failed: {_PASS_WORKER['error']}")
    return _PASS_WORKER["b"]


def _pass_worker_setup(record: dict, velocities: str) -> None:
    b = VelocityMomentTerrainBuilder(record, VelocitySet.named(velocities))
    events: list = []
    original_split = B.split

    def split_recorded(P, n, d, diag):
        probe = B.Diag()
        lo, hi = original_split(P, n, d, probe)
        events.append(("d", probe.sum_defect))
        return lo, hi

    def note_small_recorded(v):
        if v <= B.SMALL:
            events.append(("s", v))

    B.split = split_recorded
    b.note_small = note_small_recorded
    _PASS_WORKER.update(b=b, events=events)


def _pass_worker_take() -> tuple:
    """The events and the pass state written since the last take, then empty containers for the next row or state."""
    b, events = _PASS_WORKER["b"], _PASS_WORKER["events"]
    state = []
    for name in VelocityMomentTerrainBuilder.PASS_STATE:
        items = list(getattr(b, name).items())
        if items:
            state.append((name, items))
        setattr(b, name, {})
    taken = list(events)
    events.clear()
    return taken, state


def _pass_worker_producer(fkey) -> list:
    b = _pass_worker_builder()
    _pass_worker_take()
    out = []
    for d in range(b.velocity_set.q):
        if b.velocity_set.directions[d] != (0, 0, 0):
            b.producer_pass(fkey, d)
            out.append(_pass_worker_take())
    return out


def _pass_worker_donor(cell) -> list:
    b = _pass_worker_builder()
    _pass_worker_take()
    out = []
    for d in range(b.velocity_set.q):
        b.donor_pass(cell, d)
        out.append(_pass_worker_take())
    return out


# ---------------------------------------------------------------- task95/t95_geometry.py: velocity-set builder, assembly, gradient, wall list
class VelocityMomentTerrainBuilder(B.Builder):
    """builder3d.Builder with the direction table of a VelocitySet and the task-53 moment lines."""

    def __init__(self, record: dict, velocity_set: VelocitySet) -> None:
        self.velocity_set = velocity_set
        self.h_moment, self.row_moment, self.direct_moment = {}, {}, {}
        super().__init__(record)

    # Task 103: a prism that meets a region only in a face, an edge or a point (zero measure) takes no volume from it. The
    # passes below skip such a candidate by an exact integer test, before any clipping. The bounds are the exact cell box
    # of the facet (terrain facet: its column and layer; lid: its column, z in [Z_top - 1, Z_top]) swept by the prism
    # vector; the real prism lies inside these bounds. Without the skip, the binary64 clip of a piece against a plane
    # that only touches it can classify a drifted vertex just beyond EPS_ON as strictly inside and give a split with a
    # wrong volume (task 103: +1.37e-3 in the L24 donor state (7, 12, 3) along (0, -1, 0)).
    @staticmethod
    def facet_box(g, img) -> tuple:
        lo = (g.cell[0] + img[0], g.cell[1] + img[1], g.cell[2])
        return lo, (lo[0] + 1, lo[1] + 1, lo[2] + 1)

    @staticmethod
    def swept(box: tuple, v: tuple) -> tuple:
        lo, hi = box
        return (tuple(lo[a] + min(0, v[a]) for a in range(3)), tuple(hi[a] + max(0, v[a]) for a in range(3)))

    @staticmethod
    def positive_overlap(a: tuple, b: tuple) -> bool:
        return all(a[0][k] < b[1][k] and b[0][k] < a[1][k] for k in range(3))

    def producer_pass(self, fkey, d):
        f = self.facets[fkey]
        c = self.velocity_set.directions[d]
        if f.sign(c) <= 0:
            return
        vi = B.dotf(f.a, c)
        if not (math.isfinite(vi) and vi > 0.0):
            raise B.GeometryInvalid(f"exact-positive row {fkey} {c} has FP64 vi {vi}")
        self.vi[(fkey, d)] = vi
        i0 = math.floor(min(p[0] for p in f.poly))
        j0 = math.floor(min(p[1] for p in f.poly))
        origin = tuple(f.cell)                       # task 106: the clipping runs in the frame of the producer's cell
        outgoing = self.swept(self.facet_box(f, (0, 0)), c)
        recv = [(g, img) for g, img in self.candidates(i0, j0, f.cell[2], exclude=fkey, origin=origin)
                if g.sign(c) < 0 and self.positive_overlap(outgoing, self.swept(self.facet_box(g, img), c))]
        free, owned = self.producer_partition(self.facet_in_frame(f, origin), c, recv)
        for p in free:
            self.add_h(fkey, d, p, origin)
        for p, (g, img) in owned:
            self.add_w(fkey, d, g, img, p)

    def add_h(self, fkey, d, p, origin=B.ORIGIN0):
        """The free piece p (in the frame of origin) split by the cells; keys and moments are global (task 106)."""
        xs = [q[0] for q in p.v]
        ys = [q[1] for q in p.v]
        zs = [q[2] for q in p.v]
        c = self.case
        ox, oy, oz = origin
        for li in range(math.floor(min(xs)), math.ceil(max(xs))):
            for lj in range(math.floor(min(ys)), math.ceil(max(ys))):
                for lk in range(math.floor(min(zs)), math.ceil(max(zs))):
                    q = p
                    for n, dd in B.Poly.box((float(li), float(lj), float(lk)), (float(li + 1), float(lj + 1), float(lk + 1))).halfspaces():
                        q = B.clip(q, n, dd) if q is not None else None
                    if q is None:
                        continue
                    ci, cj, ck = li + ox, lj + oy, lk + oz
                    v = B.checked_volume(q, ("h", fkey, d, ci, cj, ck))
                    self.note_small(v)
                    cell = (ci % c.Lx, cj % c.Ly, ck)
                    img = (ci - ci % c.Lx, cj - cj % c.Ly)
                    key = (fkey, d, cell, img)
                    self.h[key] = self.h.get(key, 0.0) + v
                    volume, moment = q.moment()
                    corner = (li, lj, lk)            # the cell corner in the frame of origin: the part moment about it is frame-free
                    part = self.h_moment.setdefault((fkey, d, cell), [0.0, 0.0, 0.0, 0.0])
                    row = self.row_moment.setdefault((fkey, d), [0.0, 0.0, 0.0, 0.0])
                    part[0] += volume
                    row[0] += volume
                    for k in range(3):
                        part[1 + k] += moment[k] - volume * corner[k]
                        row[1 + k] += moment[k] + volume * origin[k]   # the row moment stays global

    def donor_pass(self, cell, d):
        c = self.velocity_set.directions[d]
        i, j, k = cell
        pieces = list(self.cells[cell]["pieces"])            # task 106: in the cell's own frame
        cell_box = ((i, j, k), (i + 1, j + 1, k + 1))
        back = (-c[0], -c[1], -c[2])
        for g, img in self.candidates(i, j, k, origin=(i, j, k)):
            if g.sign(c) >= 0:
                continue
            if not self.positive_overlap(cell_box, self.swept(self.facet_box(g, img), back)):
                continue                             # task 103: zero-measure contact, no volume
            S = B.Poly.prism(g.poly, (-c[0], -c[1], -c[2]))
            _, pieces = B.subtract_convex(pieces, S.halfspaces(), self.diag)
            if not pieces:
                break
        v = 0.0
        part = [0.0, 0.0, 0.0, 0.0]
        for p in pieces:
            pv = B.checked_volume(p, ("direct", cell, d))
            self.note_small(pv)
            v += pv
            volume, moment = p.moment()
            part[0] += volume
            for q in range(3):
                part[1 + q] += moment[q]           # task 106: the pieces are in the cell's own frame, so this is the moment about the cell corner
        self.direct[(cell, d)] = v
        self.direct_moment[(cell, d)] = part

    # ---- parallel passes (bitwise the serial result) -----------------------------------------------------------------------
    # Every key the passes write belongs to ONE producer row (fkey, d) or ONE donor state (cell, d): h, W, vi, h_moment and
    # row_moment to its row, direct and direct_moment to its state. So a worker computes the final value of every key of its
    # rows. The parent inserts the rows in the serial order of run() (so every dict has the serial insertion order) and replays
    # each row's Diag events (the split defects and the small pieces, in the order they happened) on its own Diag with the
    # serial statements. The container is therefore the same, byte for byte, as with run().
    PASS_STATE = ("h", "W", "vi", "h_moment", "row_moment", "direct", "direct_moment")

    def merge(self, result) -> None:
        """One row or donor state from a worker: its Diag events replayed in order, then its keys in insertion order."""
        events, state = result
        before = self.diag.sum_defect
        for kind, value in events:
            if kind == "d":
                self.diag.max_defect = max(self.diag.max_defect, value)
                self.diag.sum_defect += value
            else:
                self.diag.small_count += 1
                self.diag.small_total += value
        self.diag.max_unit_sum = max(self.diag.max_unit_sum, self.diag.sum_defect - before)
        for name, items in state:
            target = getattr(self, name)
            for key, value in items:
                if key in target:
                    raise RuntimeError(f"INVALID: parallel merge: key {key!r} of {name} written by two rows")
                target[key] = value

    def run_parallel(self, record: dict, workers: int, progress=None) -> None:
        """run() on `workers` processes; the result equals run() bitwise (see PASS_STATE)."""
        q = self.velocity_set.q
        facets = sorted(self.facets, key=str)
        cells = sorted(self.cells)
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_pass_worker_init, initargs=(record, self.velocity_set.name)) as pool:
            moving = sum(1 for c in self.velocity_set.directions if c != (0, 0, 0))
            for n, rows in enumerate(pool.imap(_pass_worker_producer, facets, chunksize=4)):
                if len(rows) != moving:
                    raise RuntimeError("INVALID: parallel producer task returned a wrong number of rows")
                for result in rows:
                    self.merge(result)
                if progress:
                    progress("facet", n)
            for n, states in enumerate(pool.imap(_pass_worker_donor, cells, chunksize=16)):
                if len(states) != q:
                    raise RuntimeError("INVALID: parallel donor task returned a wrong number of states")
                for result in states:
                    self.merge(result)
                if progress:
                    progress("cell", n)

    def run(self, progress=None):
        q = self.velocity_set.q
        for n, fkey in enumerate(sorted(self.facets, key=str)):
            for d in range(q):
                if self.velocity_set.directions[d] != (0, 0, 0):
                    before = self.diag.sum_defect
                    self.producer_pass(fkey, d)
                    self.diag.max_unit_sum = max(self.diag.max_unit_sum, self.diag.sum_defect - before)
            if progress:
                progress("facet", n)
        for n, cell in enumerate(sorted(self.cells)):
            for d in range(q):
                before = self.diag.sum_defect
                self.donor_pass(cell, d)
                self.diag.max_unit_sum = max(self.diag.max_unit_sum, self.diag.sum_defect - before)
            if progress:
                progress("cell", n)


def assemble(vs: VelocitySet, b, sam: dict) -> dict:
    """export3d.assemble with the velocity set as a parameter (t95_geometry.TerrainAssembler)."""
    q, lex, opp = vs.q, vs.directions, vs.opposites
    case = b.case
    keys = sam["keys"]
    cidx = {k: n for n, k in enumerate(keys)}
    facet_keys = sam["facet_keys"]
    rows = [(fk, d) for fk in facet_keys for d in range(q) if (fk, d) in b.vi]
    ridx = {r: n for n, r in enumerate(rows)}
    V = np.array([b.cells[k]["volume"] for k in keys])
    states = len(keys) * q
    D, G, R, Wt = ([], [], []), ([], [], []), ([], [], []), ([], [], [])
    for (cell, d), vol in b.direct.items():
        if vol == 0.0:
            continue
        c = lex[d]
        recv = ((cell[0] + c[0]) % case.Lx, (cell[1] + c[1]) % case.Ly, cell[2] + c[2])
        if recv not in cidx:
            raise B.GeometryRejected(f"direct {vol} from {cell} along {c} into a non-fluid cell {recv}")
        D[0].append(cidx[recv] * q + d); D[1].append(cidx[cell] * q + d); D[2].append(vol / V[cidx[recv]])  # noqa: E702
    hsum = {}
    for (fk, d, cell, _img), v in b.h.items():
        hsum[(fk, d, cell)] = hsum.get((fk, d, cell), 0.0) + v
    for (fk, d, cell), v in hsum.items():
        r = ridx[(fk, d)]
        qv = b.vi[(fk, d)]
        G[0].append(r); G[1].append(cidx[cell] * q + opp[d]); G[2].append(v)  # noqa: E702
        R[0].append(cidx[cell] * q + d); R[1].append(r); R[2].append(v / (qv * V[cidx[cell]]))  # noqa: E702
    wsum = {}
    for (fk, d, ak, _img), v in b.W.items():
        key = ((ak, opp[d]), (fk, d))
        wsum[key] = wsum.get(key, 0.0) + v
    for (recv_row, prod_row), v in wsum.items():
        if recv_row not in ridx:
            raise B.GeometryRejected(f"W receiver row {recv_row} does not exist")
        Wt[0].append(ridx[recv_row]); Wt[1].append(ridx[prod_row]); Wt[2].append(v)  # noqa: E702
    S = (sam["rows"], sam["cols"], sam["vals"])
    nr, nf = len(rows), len(facet_keys)

    def mk(t, shape):
        return coo_matrix((t[2], (t[0], t[1])), shape=shape).tocsr()
    vi = np.array([b.vi[r] for r in rows])
    return {"V": V, "D": mk(D, (states, states)), "G": mk(G, (nr, states)), "R": mk(R, (states, nr)), "W": mk(Wt, (nr, nr)),
            "S": mk(S, (nf, len(keys))), "vi": vi, "rows": rows, "keys": keys, "facet_keys": facet_keys}


QUERY, OTHERS, TIE_RELATIVE = 40, 11, 1e-9   # task53/moment_export.py constants of the gradient stencil


def gradient_map(case, builder, keys: list, wall_cells: np.ndarray) -> dict:
    """t95_geometry.TerrainGradient.gradient_map: own centroid + the 11 nearest distinct fluid centroids (the tied shell),
    unweighted LSQ slope, x/y periodic images only."""
    centroid = np.array([builder.cells[k]["centroid"] for k in keys])
    shifts = np.array([(sa * case.Lx, sb * case.Ly, 0) for sa in (-1, 0, 1) for sb in (-1, 0, 1)], float)
    own_shift = 4
    points = (centroid[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tree = cKDTree(points)
    cells = len(keys)
    query = min(QUERY, len(points))
    indptr, indices, weights, conditions, reproductions, ties, sizes, gaps = [0], [], [], [], [], 0, [], []
    for a in wall_cells.tolist():
        _, found = tree.query(centroid[a], k=query)
        nearest, second = {}, {}
        for flat in found.tolist():
            shift, cell = divmod(flat, cells)
            delta = centroid[cell] + shifts[shift] - centroid[a]
            d2 = float(delta @ delta)
            if cell == a:
                if shift != own_shift:
                    second[cell] = min(second.get(cell, math.inf), d2)
                continue
            if cell not in nearest or (d2, shift) < (nearest[cell][0], nearest[cell][2]):
                if cell in nearest:
                    second[cell] = min(second.get(cell, math.inf), nearest[cell][0])
                nearest[cell] = (d2, cell, shift, delta)
            else:
                second[cell] = min(second.get(cell, math.inf), d2)
        candidates = sorted(nearest.values(), key=lambda z: (z[0], z[1], z[2]))
        if len(candidates) <= OTHERS:
            raise B.GeometryRejected("INVALID: the neighbour query is too short")
        limit = candidates[OTHERS - 1][0] * (1.0 + TIE_RELATIVE)
        others = [z for z in candidates if z[0] <= limit]
        if len(others) == len(candidates):
            raise B.GeometryRejected("INVALID: the neighbour query is too short for the tied shell")
        if any(second.get(z[1], math.inf) <= limit for z in others) or second.get(a, math.inf) <= limit:
            raise B.GeometryRejected("INVALID: a second periodic image of a stencil cell lies inside the stencil limit")
        ties += int(len(others) > OTHERS)
        gaps.append((candidates[len(others)][0] - others[-1][0]) / others[-1][0])
        chosen = [(0.0, a, own_shift, np.zeros(3))] + others
        sizes.append(len(chosen))
        delta = np.array([z[3] for z in chosen])
        basis = np.c_[np.ones(len(chosen)), delta]
        normal = basis.T @ basis
        if np.linalg.matrix_rank(basis) < 4:
            raise B.GeometryRejected(f"gradient stencil of cell {keys[a]}: rank < 4")
        coefficient = np.linalg.solve(normal, basis.T)[1:4]
        expected = np.c_[np.zeros(3), np.eye(3)]
        reproductions.append(float(np.abs(coefficient @ basis - expected).max()))
        conditions.append(float(np.linalg.cond(normal)))
        for axis in range(3):
            indices.extend(z[1] for z in chosen)
            weights.extend(coefficient[axis].tolist())
            indptr.append(len(indices))
    return {"gradient_cells": wall_cells.astype(np.int64), "gradient_offsets": np.array(indptr, np.int64),
            "gradient_indices": np.array(indices, np.int64), "gradient_weights": np.array(weights),
            "reproduction_max": max(reproductions), "condition_max": max(conditions), "stencil_size_min": min(sizes),
            "stencil_size_max": max(sizes), "stencils_with_a_distance_tie_at_the_cut": ties,
            "smallest_relative_gap_of_squared_distance_after_the_stencil": min(gaps)}


def wall_list(vs: VelocitySet, builder, maps: dict, permuted: dict, gather_donors: np.ndarray) -> tuple:
    """t95_geometry.TerrainChain.wall_list: gather donors + the cells of every non-pure state (rule #253, exact unit weight)."""
    q = vs.q
    D, R = permuted["D"].tocsr(), permuted["R"].tocsr()
    keys = maps["keys"]
    case = builder.case
    cidx = {k: n for n, k in enumerate(keys)}
    wall = set(gather_donors.tolist())
    added = set()
    for state in range(D.shape[0]):
        cell, d = divmod(state, q)
        a, b = D.indptr[state], D.indptr[state + 1]
        c = vs.directions[d]
        k = keys[cell]
        donor_key = ((k[0] - c[0]) % case.Lx, (k[1] - c[1]) % case.Ly, k[2] - c[2])
        expected = cidx.get(donor_key)
        pure = (b - a == 1 and D.data[a] == 1.0 and expected is not None and int(D.indices[a]) == expected * q + d
                and R.indptr[state + 1] == R.indptr[state])
        if not pure:
            for z in [cell] + [int(s) // q for s in D.indices[a:b]]:
                if z not in wall:
                    added.add(z)
    return np.array(sorted(wall | added), np.int64), sorted(int(z) for z in added)


def sample_support_closure(vs: VelocitySet, wall: np.ndarray, permuted: dict, direct_arm: np.ndarray) -> tuple:
    """V2 (#2357): wall list := rule #253 UNION the sampler support (every cell of an S stencil), before the gradient map.

    The wall-extrapolated reference reads S[L rho] and S[L u], so every sample cell needs a gradient; the native reader
    (lbm.cpp 'a sample cell has no gradient') and the CPU reference (t87_model._rows 'a sample cell is not a wall-list cell')
    refuse a container without it. A cell added here is outside rule #253, so it is pure: no gather entry, no return entry,
    and every one of its states is a single unit transfer from the expected neighbour. As a donor it moves its whole cell,
    so its direct arms are zero in exact geometry; in binary64 the polyhedron moments leave a residue (measured 8.9e-16 on
    the four-layer flat channel). The residue is NOT snapped (direct_arm is computed before the wall list and keeps its
    bytes); it is bounded here by ARM_ROUNDING (fail closed) and returned for the record. The gradient stencils do not depend
    on the wall list (gradient_map: nearest fluid centroids of all cells), so the stencils of the other cells are unchanged.
    Containers whose sample cells are all in rule #253 get no added cell and the same bytes as before."""
    q = vs.q
    support = np.unique(permuted["S"].tocsr().indices.astype(np.int64))
    added = np.setdiff1d(support, wall)
    arm_max = 0.0
    if added.size:
        donor_cell = permuted["D"].tocoo().col // q          # the entry order of arms() (the same permuted, sorted CSR)
        if len(donor_cell) != len(direct_arm):
            raise RuntimeError("INVALID: direct_arm and D have different entry counts")
        of_added = np.isin(donor_cell, added)
        arm_max = float(np.abs(direct_arm[of_added]).max())
        if not arm_max <= ARM_ROUNDING:
            raise B.GeometryRejected(f"INVALID: a sample-support cell outside rule #253 has a direct arm {arm_max!r} > "
                                     f"{ARM_ROUNDING} (it is not a whole-cell donor)")
    return np.union1d(wall, added).astype(np.int64), [int(z) for z in added], arm_max


ARM_ROUNDING = 1e-12   # binary64 residue bound of a whole-cell direct arm (native ignores arms < 1e-9 only WITHOUT a gradient)


# ---------------------------------------------------------------- the CLI (the only new code)
class RoughTerrain:
    """V2 (VOLUMETRIC_WALL_ROUGH): the facetwise rough law of the terrain facets, from the roughness length z0.
    C = (kappa / ln(d / z0))^2 with kappa = 0.4 and d = 0.5 (the sample distance), the law of
    research/wall_model/v2/rough_wall/e2/rough_model.py. Terrain facets ("T", ...) get (K0, C); lid facets ("L", ...) get
    (0, 0): the lid stays a free-slip wall. Refused: z0 not finite or <= 0, d <= z0, a computed C that is not finite and
    strictly positive, K0 not finite or < 0. d > z0 is a NECESSARY algebraic domain condition of the law (#2323); it is not
    evidence that a physical sample lies in a log layer."""

    KAPPA, SAMPLE_DISTANCE = 0.4, 0.5

    def __init__(self, z0: float, K0: float) -> None:
        if not (math.isfinite(z0) and z0 > 0.0):
            raise ValueError(f"--rough-z0 must be finite and > 0 (it is {z0!r})")
        if not self.SAMPLE_DISTANCE > z0:
            raise ValueError(f"--rough-z0 {z0!r} >= the sample distance {self.SAMPLE_DISTANCE}: the log law needs the sample above the roughness")
        if not (math.isfinite(K0) and K0 >= 0.0):
            raise ValueError(f"--rough-K0 must be finite and >= 0 (it is {K0!r})")
        self.z0, self.K0 = z0, K0
        self.C = (self.KAPPA / math.log(self.SAMPLE_DISTANCE / z0)) ** 2
        if not (math.isfinite(self.C) and self.C > 0.0):   # e.g. a subnormal z0: d/z0 overflows and C would silently be 0
            raise ValueError(f"--rough-z0 {z0!r} gives C = {self.C!r}, not a finite positive coefficient")

    def sections(self, facet_keys: list) -> tuple:
        """(facet_friction facets x 2, facet_kind facets) in the container facet order."""
        kinds = []
        for key in facet_keys:
            if not (isinstance(key, tuple) and len(key) > 0 and key[0] in ("T", "L")):
                raise ValueError(f"facet key {key!r} is neither terrain ('T') nor lid ('L')")
            kinds.append(0 if key[0] == "T" else 1)
        kind = np.array(kinds, dtype=np.uint32)
        law = np.zeros((len(facet_keys), 2))
        law[kind == 0, 0], law[kind == 0, 1] = self.K0, self.C
        return law, kind


class ContainerBuild:
    def __init__(self, manifest: TerrainManifest, velocities: str, nu: float, friction: float, rough: RoughTerrain | None = None,
                 workers: int = 1) -> None:
        if not (math.isfinite(nu) and nu > 0.0 and math.isfinite(friction) and friction >= 0.0):
            raise ValueError("nu must be > 0 and K >= 0")
        self.manifest, self.vs, self.nu, self.friction = manifest, VelocitySet.named(velocities), nu, friction
        self.rough = rough
        if not (isinstance(workers, int) and workers >= 1):
            raise ValueError("workers must be an integer >= 1")
        self.workers = workers
        self.seconds = {}
        self.support_added = []      # V2 (#2357): cells added to the wall list by sample_support_closure (set by build)
        self.support_arm_max = 0.0

    def build(self, log) -> tuple:
        t0 = time.time()
        self.manifest.domain_checks()
        record = self.manifest.record()
        try:
            b = VelocityMomentTerrainBuilder(record, self.vs)
        except B.GeometryRejected as error:
            raise B.GeometryRejected(self.cell_context(str(error))) from error
        log(f"cells {len(b.cells)}, facets {len(b.facets)}; producer and donor passes ...")
        start = time.time()
        facets, cells = len(b.facets), len(b.cells)

        def progress(kind, n):
            total = facets if kind == "facet" else cells
            if (n + 1) % max(1, total // 10) == 0 or n + 1 == total:
                log(f"  {kind} pass {n + 1}/{total} {time.time() - start:.0f} s")
        if self.workers > 1:
            b.run_parallel(record, self.workers, progress=progress)   # bitwise the serial result
        else:
            b.run(progress=progress)
        self.seconds["passes"] = time.time() - start
        t = time.time()
        sam = B.build_sampler(b, stencil_weights, NEIGHBOURS, visible_only=True)   # V2: exact window (#2213), 12 nearest VISIBLE (#2283)
        self.seconds["sampler"] = time.time() - t
        log(f"sampler candidates: {sam['candidate_source']}")
        ident_maker = MapIdentities(self.vs)
        m = assemble(self.vs, b, sam)
        identities = ident_maker.identities(m)
        fpos = {k: n for n, k in enumerate(m["facet_keys"])}
        rf, rd = np.array([fpos[r[0]] for r in m["rows"]]), np.array([r[1] for r in m["rows"]])
        p = ident_maker.permute_rows(m, rf, rd)
        for name in "DGRSW":
            p[name] = p[name].tocsr().copy()
            p[name].sort_indices()
        t = time.time()
        arm = arms(self.vs, b, m, p)
        wall, added = wall_list(self.vs, b, m, p, arm["wall_cells"])
        wall, self.support_added, self.support_arm_max = sample_support_closure(self.vs, wall, p, arm["direct_arm"])   # V2 (#2357)
        grad = gradient_map(b.case, b, m["keys"], wall)
        self.seconds["arms_and_gradient"] = time.time() - t
        self.seconds["total"] = time.time() - t0
        return b, sam, m, p, identities, arm, grad, added, record

    def cell_context(self, message: str) -> str:
        """The builder's refusal plus the context of its cell (diagnostic only; nothing is changed or snapped): the four node
        heights of the column (decimal and exact hex) and their signed distances to the cell's lower and upper faces.
        The builder classifies a cell with the exact heights but cuts its pieces in binary64 with a vertex-on-plane tolerance
        of about 1e-12 (builder3d EPS_ON); a node a few ulp from a cell face can therefore give a fluid cell without a piece."""
        import re  # noqa: PLC0415
        found = re.search(r"cell \((-?\d+), (-?\d+), (-?\d+)\)", message)
        if not found:
            return message
        i, j, k = (int(v) for v in found.groups())
        lines = [message, f"context of cell ({i}, {j}, {k}): faces z = {k} and z = {k + 1}; the column's node heights:"]
        for (a, c), h in sorted(self.manifest.column_nodes(i, j).items()):
            value = float(h)
            lines.append(f"  node ({(i + a) % self.manifest.Lx}, {(j + c) % self.manifest.Ly}): {value!r} ({value.hex()}), "
                         f"to z = {k}: {float(h - k):+.3e}, to z = {k + 1}: {float(h - (k + 1)):+.3e}")
        lines.append("A node within about 1e-12 of a cell face makes a sliver that the binary64 clipping removes. The builder does not "
                     "change the input: round or offset the heights explicitly (and record that change) and build again.")
        return "\n".join(lines)

    def sections(self, b, sam, m, p, arm, grad, added, record, identities) -> list:
        """t95_geometry.TerrainChain.sections: the section order of the task-95 container."""
        case = b.case
        keys, facet_keys, rows = m["keys"], m["facet_keys"], m["rows"]
        z_lo = min(k[2] for k in keys) - 1
        box = np.array([case.Lx, case.Ly, case.Z_top - z_lo + 1])
        cell_xyz = np.array([[k[0], k[1], k[2] - z_lo] for k in keys])
        cent = np.array([[b.cells[k]["centroid"][0], b.cells[k]["centroid"][1], b.cells[k]["centroid"][2] - z_lo] for k in keys])
        fct = [b.facets[k] for k in facet_keys]
        normals = np.array([f.unit for f in fct])
        areas = np.array([float(np.sqrt(np.dot(f.a, f.a))) for f in fct])
        samples = np.array([[q[0], q[1], q[2] - z_lo] for q in sam["points"]])
        w = Writer()
        w.array("box", DTYPE_I32, box)
        w.array("counts", DTYPE_U32, np.array([len(keys), len(facet_keys), len(rows)]))
        w.array("cell_xyz", DTYPE_I32, cell_xyz.reshape(-1))
        w.array("volume", DTYPE_F64, m["V"])
        w.array("centroid", DTYPE_F64, cent.reshape(-1))
        w.array("sample_point", DTYPE_F64, samples.reshape(-1))
        w.array("mode_phi", DTYPE_F64, np.ones(len(keys)))
        w.array("normal", DTYPE_F64, normals.reshape(-1))
        w.array("area", DTYPE_F64, areas)
        w.array("facet_offsets", DTYPE_U32, np.searchsorted(p["rf"], np.arange(len(facet_keys) + 1)))
        w.array("row_direction", DTYPE_U32, p["rd"])
        w.array("prism_volume", DTYPE_F64, p["vi"])
        w.array("row_origin", DTYPE_U32, p["order"])
        for name in ("D", "G", "R", "S", "W"):
            w.csr(name, p[name])
        w.array("scalars", DTYPE_F64, np.array([self.nu, 1.0 / (3.0 * self.nu + 0.5), self.friction]))
        w.array("f0", DTYPE_F64, np.tile(self.vs.w_array, len(keys)))
        w.array("s0", DTYPE_F64, p["s0"])
        arrays = {n: np.frombuffer(pl, {DTYPE_U32: "<u4", DTYPE_I32: "<i4", DTYPE_F64: "<f8"}[d]) for n, d, pl in w.sections if d != DTYPE_TEXT}
        phi0 = MapIdentities(self.vs).base_rows(arrays)
        if phi0.shape != (len(rows), 3) or not np.isfinite(phi0).all():
            raise RuntimeError("INVALID: bad base rows")
        w.array("friction_rows", DTYPE_F64, phi0)
        w.sections.append(("direct_arm", DTYPE_F64, np.ascontiguousarray(arm["direct_arm"], "<f8").tobytes()))
        w.sections.append(("prism_arm", DTYPE_F64, np.ascontiguousarray(arm["prism_arm"], "<f8").tobytes()))
        for name in ("gradient_cells", "gradient_offsets", "gradient_indices"):
            if grad[name].min() < 0 or grad[name].max() >= 2 ** 32:
                raise RuntimeError(f"INVALID: {name} does not fit uint32")
            w.sections.append((name, DTYPE_U32, grad[name].astype("<u4").tobytes()))
        w.sections.append(("gradient_weights", DTYPE_F64, np.ascontiguousarray(grad["gradient_weights"], "<f8").tobytes()))
        if self.rough is not None:   # V2: the optional facetwise rough law; written only with --rough-z0 (legacy bytes otherwise)
            law, kind = self.rough.sections(facet_keys)
            w.array("facet_friction", DTYPE_F64, law.reshape(-1))
            w.array("facet_kind", DTYPE_U32, kind)
        g = {k: v for k, v in grad.items() if not isinstance(v, np.ndarray)}
        prov = {"format": "VWNAT01", "case": record["name"], "velocity_set": self.vs.name, "Lx": case.Lx, "Ly": case.Ly,
                "Z_top": case.Z_top, "z_lo": int(z_lo), "box": box.tolist(), "nu": self.nu, "friction_K": self.friction,
                "f0": "rest w_i (rho0 = 1)", "s0": "W w_rd (rest prehistory)", "identities": identities, "gradient_report": g,
                "wall_list_cells_added_by_rule_253": len(added), "builder_diagnostics": b.diag.as_dict(),
                "sampler_branches": sam["branches"], "exporter": "scripts/slip_wall/terrain_container.py",
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "manifest_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode("ascii")).hexdigest()}
        if self.support_added:       # V2 (#2357): only when the closure adds a cell, so the legacy provenance text is unchanged
            prov["wall_list_cells_added_by_sample_support"] = len(self.support_added)
            prov["sample_support_direct_arm_max_abs"] = self.support_arm_max
        if self.rough is not None:   # only with --rough-z0, so the legacy provenance text is unchanged
            prov["rough"] = {"z0": self.rough.z0, "z0_hex": float(self.rough.z0).hex(), "K0_terrain": self.rough.K0,
                             "C_terrain": self.rough.C, "C_terrain_hex": float(self.rough.C).hex(), "kappa": RoughTerrain.KAPPA,
                             "sample_distance": RoughTerrain.SAMPLE_DISTANCE, "lid": "K0 = C = 0"}
        w.text("provenance", json.dumps(prov, indent=1))
        return w


class NativeIdentityGate:
    """Task 103: the map identities with the definitions and the tolerance of the native checker
    (VolumetricWallNativeInput::validate, identity_tolerance 1e-10), on the assembled maps before any file is written:
      donor partition   max |v^T D + 1^T G - v^T| / v   per state (relative to the donor cell volume)
      receiver coverage max |D 1 + R vi - 1|            per state
      prism coverage    max |G 1 + W 1 - vi| / vi       per row
      return delivery   max |v^T R + 1^T W diag(1/vi) - 1| per row
    A non-finite value or a residual above the tolerance refuses the container. The maps are never changed."""

    TOLERANCE = 1e-10   # identity_tolerance of src/experimental/volumetric_wall_native.cpp

    def __init__(self, vs: VelocitySet) -> None:
        self.vs = vs

    def check(self, m: dict, b) -> dict:
        q = self.vs.q
        V, D, G, R, W, vi = m["V"], m["D"], m["G"], m["R"], m["W"], m["vi"]
        Vs = np.repeat(V, q)
        donor = np.abs(Vs @ D + np.ones(G.shape[0]) @ G - Vs) / Vs
        recv = np.abs(D @ np.ones(D.shape[1]) + R @ vi - np.ones(len(Vs)))
        prism = np.abs(G @ np.ones(G.shape[1]) + W @ np.ones(W.shape[1]) - vi) / vi
        ret = np.abs(Vs @ R + (np.ones(W.shape[0]) @ W) / vi - np.ones(len(vi)))
        values = {"donor_partition": donor, "receiver_coverage": recv, "prism_coverage": prism, "return_delivery": ret}
        report = {"tolerance": self.TOLERANCE, "definitions": "native (src/experimental/volumetric_wall_native.cpp)",
                  "builder_diagnostics": b.diag.as_dict(), "identities": {}, "failing": {}}
        finite = all(np.isfinite(a).all() for a in (V, vi, D.data, G.data, R.data, W.data))
        report["finite"] = bool(finite)
        for name, value in values.items():
            worst = float(np.max(value)) if value.size and np.isfinite(value).all() else float("nan")
            report["identities"][name] = worst
            bad = np.flatnonzero(~np.isfinite(value) | (value > self.TOLERANCE))
            if bad.size:
                report["failing"][name] = self.listing(name, bad[np.argsort(-np.nan_to_num(value[bad], nan=np.inf))], value, m)
        report["passed"] = bool(finite and not report["failing"])
        return report

    def listing(self, name: str, order: np.ndarray, value: np.ndarray, m: dict) -> list:
        q, out = self.vs.q, []
        for n in order[:50].tolist():
            if name in ("donor_partition", "receiver_coverage"):
                cell, d = divmod(n, q)
                out.append({"state": n, "cell": list(m["keys"][cell]), "c": list(self.vs.directions[d]),
                            "volume": float(m["V"][cell]), "residual": float(value[n])})
            else:
                fkey, d = m["rows"][n]
                out.append({"row": n, "facet": str(fkey), "c": list(self.vs.directions[d]), "vi": float(m["vi"][n]),
                            "residual": float(value[n])})
        return out


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heights", required=True, type=Path, help="CSV: Ny lines of Nx node heights")
    parser.add_argument("--z-top", required=True, type=int, help="the lid height (integer, lattice units)")
    parser.add_argument("--nu", required=True, type=float, help="kinematic viscosity, lattice units")
    parser.add_argument("--friction", required=True, type=float, help="the slip coefficient K >= 0 (traction -rho (K/3) u_t)")
    parser.add_argument("--velocities", default="D3Q27", choices=("D3Q27", "D3Q19"), help="D3Q19 only for the recovery control")
    parser.add_argument("--name", default="terrain")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rough-z0", type=float, default=None, help="V2: the terrain roughness length z0 (lattice units); writes the "
                        "facet_friction and facet_kind sections (terrain (K0, C), lid (0, 0)); needs VOLUMETRIC_WALL_ROUGH to run")
    parser.add_argument("--rough-K0", type=float, default=0.0, help="V2: the terrain K0 of the rough law (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="processes for the producer and donor passes (the container is "
                        "the same, byte for byte, as with 1)")
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"REFUSED: {args.out} exists")
        return 2
    rough = None
    if args.rough_z0 is not None:
        try:
            rough = RoughTerrain(args.rough_z0, args.rough_K0)
        except ValueError as error:
            print(f"REFUSED: {error}")
            return 2
    elif args.rough_K0 != 0.0:
        print("REFUSED: --rough-K0 needs --rough-z0")
        return 2
    manifest = TerrainManifest.from_csv(args.heights, args.z_top, args.name)
    run = ContainerBuild(manifest, args.velocities, args.nu, args.friction, rough, workers=args.workers)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)
    b, sam, m, p, identities, arm, grad, added, record = run.build(log)
    gate = NativeIdentityGate(run.vs).check(m, b)
    args.out.parent.mkdir(parents=True, exist_ok=True)   # the folder of the container or of the rejection report
    if not gate["passed"]:
        rejected = Path(str(args.out) + ".rejected.json")
        gate.update({"container": None, "requested_output": str(args.out), "cells": len(m["keys"]), "rows": len(m["rows"]),
                     "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "builder3d_sha256": hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()})
        rejected.write_text(json.dumps(gate, indent=1, default=str) + "\n", encoding="utf-8")
        log(f"REFUSED: map identities violated (native definitions, tolerance {NativeIdentityGate.TOLERANCE:g}); no container written")
        log(json.dumps({"identities": gate["identities"], "finite": gate["finite"],
                        "worst": {k: v[0] for k, v in gate["failing"].items()}}, indent=1, default=str))
        log(f"diagnostic report: {rejected}")
        return 3
    writer = run.sections(b, sam, m, p, arm, grad, added, record, identities)
    writer.write(args.out)
    data = args.out.read_bytes()
    report = {"container": str(args.out), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "velocity_set": run.vs.name,
              "cells": len(m["keys"]), "facets": len(m["facet_keys"]), "rows": len(m["rows"]), "W_nnz": int(p["W"].nnz),
              "identities": identities, "native_identities": gate["identities"], "seconds": run.seconds, "pass_workers": run.workers, "wall_list_cells_added_by_rule_253": added,
              "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "builder3d_sha256": hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()}
    Path(str(args.out) + ".build.json").write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + "\n", encoding="utf-8")
    log(json.dumps({k: report[k] for k in ("sha256", "bytes", "cells", "facets", "rows", "W_nnz", "identities", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
