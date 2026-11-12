"""Task 120 (E2): static neighborhoods and weights of the conservative affine state redistribution (reference candidate). READ-ONLY on the container.

STATUS: written; a DEVELOPMENT build may be run to see the counts; the registered build is E1's. Needs --release (a label).
SPECIFICATION (frozen: scientist #2951, #2952, #2953; E1 protocol draft d0dd05f2; canonical MATHEMATICAL_MODEL.md section 15.12). Source weights: Berger
and Giuliani 2023, equation 4.2. The weighted centered fit on the SAME averaging neighborhood is OUR variant, not the source's neighbor-Q slope scheme.

  Cells. alpha_i = V_i. SEED: V_i < ALPHA_TARGET = 0.5. A seed j has the merging neighborhood M_j; every other cell is its own singleton (no slope) and
  may be a member of seeds' neighborhoods.
  Graph. Undirected FLUID FACE graph: i and k are linked when the direct table D has an entry in a shell-1 direction between them with the same direction
  index (a positive direct prism inside the fluid). Nothing is added across a solid.
  Periodic images. For a seed j and a cell k: s = (xyz_j + wrap(xyz_k - xyz_j)) - xyz_k, wrap = minimum image in x and y (box > 3 cells), none in z; the centroid of k
  in the frame of j is C_k + s; all offsets are in the frame of the seed.
  Expansion, deterministic. From {j}, add ONE cell at a time in the order of the key (graph distance, centroid distance to the seed, cell index), among cells
  at most MAX_GRAPH = 2 graph edges from j AND at most MAX_CHEB = 2 lattice cells from j (Chebyshev norm of the wrapped lattice offset), until
    (a) sum of V over M_j >= ALPHA_TARGET, and
    (b) FULL RANK of the dimensionless centered weighted covariance A_j = (1/Vhat_j) sum_i w_ij V_i d_i d_i^T, d_i = C_i - xhat_j: s_min/s_max >= RANK_RATIO_MIN = 1e-10
        (np.linalg.svd, FP64); a declared numerical-conditioning requirement, not a theorem.
  Weights depend on the other neighborhoods, so (b) is tested with the FINAL weights: the build iterates (add one candidate to every failing neighborhood,
  recompute all weights and tests) until none fails. A neighborhood that fails with no candidate left within the limits makes the setup REJECTED (exit 3): the
  exact failing neighborhoods are reported (seed, members, volume, rank ratio); no silent enlargement, no pseudoinverse, no zero-slope downgrade.
  Weights (source 4.2): W_i = {i} plus the seeds j with i in M_j (for a seed i, i is in M_i); N_i = |W_i|; for j in W_i, j != i:
  w_ij = (1 - alpha_j/ALPHA_TARGET)/(N_i - 1); w_ii = 1 - sum of the others (accumulated in ascending seed index). w >= 0 by construction.
  Moments. Vhat_j = sum w_ij V_i; xhat_j = sum w_ij V_i (C_i + s)/Vhat_j then ONE refinement xhat_j += sum w_ij V_i (C_i + s - xhat_j)/Vhat_j, so the weighted first
  moment vanishes to round-off (reported).
Outputs (NPZ, hashed): seeds, entries sorted by seed (seed, member, w, delta), per-seed Vhat, xhat, A, the rank ratio, the singleton coefficient c_i of the
non-seed members, and a JSON report (counts, size histogram, reach reached, rank ratios (min and the five worst), min Vhat, min and max weight, max N_i,
connectedness, identity residuals, hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[3]
sys.path.insert(0, str(WALL / "integration" / "task39"))
import t39_lib as T39  # noqa: E402

CONTAINER = WALL / "v2" / "hill_demo" / "e1" / "hf_inputs" / "hill_uniform.vwnat"
CONTAINER_SHA_PREFIX = "a404dc7463c9bc1a"
ALPHA_TARGET = 0.5
RANK_RATIO_MIN = 1e-10
MAX_GRAPH = 2
MAX_CHEB = 2
Q = 27


class Rejected(RuntimeError):
    """The setup cannot meet the frozen construction within the frozen limits."""

    def __init__(self, message: str, failing: list) -> None:
        super().__init__(message)
        self.failing = failing


class Container:
    """The arrays used (read-only)."""

    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        if not self.sha.startswith(CONTAINER_SHA_PREFIX):
            raise ValueError(f"container sha256 {self.sha[:16]} is not the pinned {CONTAINER_SHA_PREFIX}")
        a = T39.arrays(T39.read_sections(raw))
        self.box = np.asarray(a["box"], dtype=np.int64)
        self.V = np.asarray(a["volume"], dtype=np.float64)
        self.n = len(self.V)
        self.C = np.asarray(a["centroid"], dtype=np.float64).reshape(-1, 3)
        self.xyz = np.asarray(a["cell_xyz"], dtype=np.int64).reshape(-1, 3)
        self.D = csr_matrix((np.asarray(a["D_weights"], dtype=np.float64), np.asarray(a["D_indices"], dtype=np.int64),
                             np.asarray(a["D_offsets"], dtype=np.int64)), shape=tuple(int(x) for x in a["D_shape"]))
        if self.box[0] <= 3 or self.box[1] <= 3:
            raise ValueError("box x and y must exceed 3 cells (unique periodic image)")


class FluidFaceGraph:
    """Undirected face-adjacency of the cells through the direct transport table (shell-1 directions)."""

    def __init__(self, c: Container) -> None:
        coo = c.D.tocoo()
        lex = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
        rc, rd, dc, dd = coo.row // Q, coo.row % Q, coo.col // Q, coo.col % Q
        face = (rd == dd) & (np.abs(lex[rd]).sum(axis=1) == 1) & (rc != dc) & (coo.data > 0)
        e = csr_matrix((np.ones(int(face.sum())), (rc[face], dc[face])), shape=(c.n, c.n))
        adj = ((e + e.T) > 0).astype(np.int8).tocsr()
        adj.sort_indices()
        self.adj = adj

    def neighbours(self, i: int) -> np.ndarray:
        return self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i + 1]]


class Neighborhoods:
    """The deterministic construction of the frozen specification (see the module text)."""

    def __init__(self, c: Container) -> None:
        self.c, self.g = c, FluidFaceGraph(c)
        self.seeds = np.flatnonzero(c.V < ALPHA_TARGET)
        self.is_seed = np.zeros(c.n, dtype=bool)
        self.is_seed[self.seeds] = True
        self.reach: dict = {}
        self.candidates = {int(j): self._candidates(int(j)) for j in self.seeds}
        self.members = {int(j): [int(j)] for j in self.seeds}

    def shift(self, j: int, k: int) -> np.ndarray:
        c = self.c
        d = c.xyz[k] - c.xyz[j]
        wrapped = d.copy()
        for a in (0, 1):
            wrapped[a] -= c.box[a] * int(np.round(d[a] / c.box[a]))
        return (c.xyz[j] + wrapped) - c.xyz[k]

    def cheb(self, j: int, k: int) -> int:
        c = self.c
        d = c.xyz[k] - c.xyz[j]
        for a in (0, 1):
            d[a] -= c.box[a] * int(np.round(d[a] / c.box[a]))
        return int(np.abs(d).max())

    def _candidates(self, j: int) -> list:
        """(graph distance, centroid distance, index) order within the limits; the reach of each candidate is kept."""
        c = self.c
        seen = {j}
        out = []
        frontier = [j]
        for ring in range(1, MAX_GRAPH + 1):
            nxt = sorted({int(k) for i in frontier for k in self.g.neighbours(i)} - seen)
            seen.update(nxt)
            keyed = []
            for k in nxt:
                cheb = self.cheb(j, k)
                if cheb > MAX_CHEB:
                    continue
                dist = float(np.linalg.norm(c.C[k] + self.shift(j, k) - c.C[j]))
                keyed.append((dist, k, ring, cheb))
            out.extend(sorted(keyed))
            frontier = nxt            # the graph walk continues through all cells of the ring, also those beyond the Chebyshev limit
        return [(k, ring, cheb) for _, k, ring, cheb in out]

    def entries(self) -> tuple:
        s = [j for j in sorted(self.members) for _ in self.members[j]]
        m = [i for j in sorted(self.members) for i in self.members[j]]
        return np.array(s, dtype=np.int64), np.array(m, dtype=np.int64)

    def weights(self, es: np.ndarray, em: np.ndarray) -> tuple:
        c = self.c
        n_count = np.bincount(em, minlength=c.n) + (~self.is_seed).astype(np.int64)
        other = es != em
        w = np.zeros(len(es))
        denom = np.maximum(n_count[em] - 1, 1)
        w[other] = (1.0 - c.V[es[other]] / ALPHA_TARGET) / denom[other]
        sums = np.zeros(c.n)
        for pos in np.lexsort((es, em)):
            if other[pos]:
                sums[em[pos]] += w[pos]            # ascending seed index per member
        w[~other] = 1.0 - sums[em[~other]]
        coef = np.where(self.is_seed, 0.0, 1.0 - sums)
        return w, coef, n_count

    def geometry(self, es: np.ndarray, em: np.ndarray, w: np.ndarray) -> dict:
        c = self.c
        pos = np.stack([c.C[em[t]] + self.shift(int(es[t]), int(em[t])) for t in range(len(es))])
        wv = w * c.V[em]
        starts = np.flatnonzero(np.r_[True, es[1:] != es[:-1]])
        seed_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(es)]))
        vhat = np.add.reduceat(wv, starts)
        xhat = np.add.reduceat(wv[:, None] * pos, starts, axis=0) / vhat[:, None]
        xhat = xhat + np.add.reduceat(wv[:, None] * (pos - xhat[seed_of]), starts, axis=0) / vhat[:, None]
        delta = pos - xhat[seed_of]
        A = np.add.reduceat(wv[:, None, None] * delta[:, :, None] * delta[:, None, :], starts, axis=0) / vhat[:, None, None]
        sv = np.linalg.svd(A, compute_uv=False)
        ratio = sv[:, -1] / np.maximum(sv[:, 0], 1e-300)
        return {"pos": pos, "delta": delta, "wv": wv, "starts": starts, "seeds": es[starts], "seed_of": seed_of, "vhat": vhat, "xhat": xhat, "A": A,
                "ratio": ratio, "volume": np.add.reduceat(c.V[em], starts)}

    def build(self) -> dict:
        c = self.c
        limit = max((len(v) for v in self.candidates.values()), default=0) + 1
        for p in range(limit + 1):
            es, em = self.entries()
            w, coef, n_count = self.weights(es, em)
            geo = self.geometry(es, em, w)
            failing = [k for k in range(len(geo["seeds"])) if not (geo["volume"][k] >= ALPHA_TARGET and geo["ratio"][k] >= RANK_RATIO_MIN)]
            if not failing:
                out = {"es": es, "em": em, "w": w, "coef": coef, "n_count": n_count, "passes": np.array(p + 1), **geo}
                return out
            stuck = []
            for k in failing:
                j = int(geo["seeds"][k])
                taken = set(self.members[j])
                nxt = next((cand for cand in self.candidates[j] if cand[0] not in taken), None)
                if nxt is None:
                    stuck.append({"seed": j, "members": list(self.members[j]), "volume": float(geo["volume"][k]), "rank_ratio": float(geo["ratio"][k]),
                                  "candidates_within_limits": len(self.candidates[j])})
                else:
                    self.members[j].append(int(nxt[0]))
                    self.reach[(j, int(nxt[0]))] = (nxt[1], nxt[2])
            if stuck:
                raise Rejected(f"{len(stuck)} neighborhoods cannot reach volume >= {ALPHA_TARGET} and rank ratio >= {RANK_RATIO_MIN:g} within {MAX_GRAPH} graph edges "
                               f"and {MAX_CHEB} lattice cells", stuck)
        raise Rejected("the iteration did not stop", [])


def report(nb: Neighborhoods, out: dict) -> dict:
    c = nb.c
    sizes = np.diff(np.r_[out["starts"], len(out["es"])])
    wsum = np.zeros(c.n)
    np.add.at(wsum, out["em"], out["w"])
    wsum = wsum + np.where(nb.is_seed, 0.0, out["coef"])
    active = np.unique(out["em"])
    moment = np.add.reduceat(out["wv"][:, None] * out["delta"], out["starts"], axis=0)
    disconnected = []
    for j in out["seeds"].tolist():
        mem = set(nb.members[j])
        seen, stack = {j}, [j]
        while stack:
            u = stack.pop()
            for v in nb.g.neighbours(u):
                if int(v) in mem and int(v) not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        if seen != mem:
            disconnected.append(j)
    order = np.argsort(out["ratio"])[:5]
    reach_graph = max([r[0] for r in nb.reach.values()], default=0)
    reach_cheb = max([r[1] for r in nb.reach.values()], default=0)
    return {"cells": int(c.n), "seeds": int(len(out["seeds"])), "active_cells": int(len(active)),
            "size_histogram": {int(s): int((sizes == s).sum()) for s in sorted(set(sizes.tolist()))}, "passes": int(out["passes"]),
            "reach_graph_used": reach_graph, "reach_chebyshev_used": reach_cheb, "rank_ratio_min": float(out["ratio"].min()),
            "rank_ratio_five_worst": [{"seed": int(out["seeds"][k]), "ratio": float(out["ratio"][k]), "members": nb.members[int(out["seeds"][k])]} for k in order],
            "vhat_min": float(out["vhat"].min()), "volume_min": float(out["volume"].min()), "weight_min": float(out["w"].min()), "weight_max": float(out["w"].max()),
            "max_N_i": int(out["n_count"].max()), "weight_sum_max_dev": float(np.abs(wsum[active] - 1.0).max()),
            "first_moment_abs_max": float(np.abs(moment).max()), "first_moment_rel_vhat_max": float((np.abs(moment).max(axis=1) / out["vhat"]).max()),
            "disconnected_neighborhoods": disconnected, "max_abs_delta": float(np.abs(out["delta"]).max()),
            "constants": {"ALPHA_TARGET": ALPHA_TARGET, "RANK_RATIO_MIN": RANK_RATIO_MIN, "MAX_GRAPH": MAX_GRAPH, "MAX_CHEB": MAX_CHEB}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True, type=Path, help="output stem: STEM.npz and STEM.json")
    a = ap.parse_args()
    c = Container(CONTAINER)
    nb = Neighborhoods(c)
    try:
        out = nb.build()
    except Rejected as error:
        a.out.with_suffix(".json").write_text(json.dumps({"release": a.release, "verdict": "REJECTED", "reason": str(error), "failing": error.failing,
                                                          "seeds": int(len(nb.seeds))}, indent=1))
        print(f"REJECTED: {error}; failing neighborhoods listed in {a.out.with_suffix('.json').name}")
        return 3
    rep = report(nb, out)
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez(a.out.with_suffix(".npz"), **arrays)
    rep.update(release=a.release, verdict="BUILT", container_sha256=c.sha, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               npz_sha256=hashlib.sha256(a.out.with_suffix(".npz").read_bytes()).hexdigest())
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
