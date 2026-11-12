"""Task 120 revision 2 (E2): static tables of the conservative affine state redistribution, SOURCE-method reconstruction. READ-ONLY on the container.

STATUS: final candidate tool. A DEVELOPMENT build was run by E2 (dev2_srd.*, CPU, read-only) and its sets agree exactly with E1's independent builder (E1 #2968); the REGISTERED build is E1's. Needs --release (a label). This replaces the withdrawn same-neighborhood fit
(srd_build.py, preserved). FINAL choices: scientist #2962 on E1 revision 2 (f9b654e6); canonical MATHEMATICAL_MODEL.md section 15.12 (corrected, a4e1632f).
Source: Berger and Giuliani 2023, weights (4.2), reconstruction from the Qhat of neighboring neighborhoods with the distance rule (4.9, 4.10 with Q_-2) and the
stencil span rule (5.1); the 3D transcription is OURS.

AVERAGING neighborhoods M_j (no rank condition). Seeds: V_j < 0.5. Face graph of the fluid from the direct table D (see srd_build.FluidFaceGraph). A seed expands
one cell at a time in the order (graph distance, centroid distance to the seed, cell index) among cells within 2 graph edges and 2 lattice cells (Chebyshev,
wrapped offset) until the RAW volume sum of V_i over M_j is >= 0.5. Non-seeds are singletons. Weights (4.2) exactly as in srd_build. Vhat_j = sum w_ij V_i
must be positive (reported). A seed that cannot reach 0.5 within the limits makes the setup REJECTED (the exact list is reported, nothing is enlarged).
Every cell k labels a neighborhood: a seed's M_k, or the singleton {k} with Vhat_k = w_kk V_k and xhat_k = C_k.

RECONSTRUCTION stencil T_j (per seed j): all labels k != j whose cell is within graph distance 3 of j, with |xhat_k + s - xhat_j| >= 0.5 (s = the integer
periodic shift of cell k into the chart of cell j; the chart of a label is its own seed's chart). The stencil, with xhat_j, must have span >= 1 in every
coordinate and the Vhat-weighted second-moment matrix B_j = (1/sum Vhat_k) sum_k Vhat_k d_k d_k^T (d_k = xhat_k + s - xhat_j) must satisfy
s_min/s_max >= 1e-10 (SVD). Otherwise REJECTED with the exact list. No tuning, no nearest subset, no zero-slope substitution.
The slope at run time: sigma_j = B_j^{-1} (1/sum Vhat_k) sum_k Vhat_k (Q_k - Q_j) d_k (a weighted LSQ gradient through (xhat_j, Q_j); no member value of M_j is used
directly; the Q_k of singletons are cell values and of seeds neighborhood means). Affine data on the stencil gives sigma = b exactly.

Outputs: NPZ (members, weights, offsets, per-seed Vhat and xhat, label Vhat and xhat, the stencil entries and B_j) and a JSON report with the canonical member
and stencil sets (sha256 of their canonical text) so that two builders can be compared exactly (the earlier builder discrepancy must not recur unnoticed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import srd_build as SB  # noqa: E402

MAX_GRAPH_STENCIL = 3
H_HALF = 0.5
SPAN_MIN = 1.0
RANK_RATIO_MIN = 1e-10


class SrdConstruction:
    def __init__(self, c: SB.Container) -> None:
        self.nb = SB.Neighborhoods(c)
        self.c, self.g = c, self.nb.g
        self.rejected_average: list = []
        self.rejected_stencil: list = []

    def averaging(self) -> None:
        nb, c = self.nb, self.c
        for j in sorted(nb.members):
            vol = float(c.V[j])
            for cand in nb.candidates[j]:
                if vol >= SB.ALPHA_TARGET:
                    break
                nb.members[j].append(int(cand[0]))
                nb.reach[(j, int(cand[0]))] = (cand[1], cand[2])        # (graph distance, Chebyshev) of each added member
                vol += float(c.V[cand[0]])
            if vol < SB.ALPHA_TARGET:
                self.rejected_average.append({"seed": int(j), "members": list(nb.members[j]), "raw_volume": vol, "candidates": len(nb.candidates[j])})

    def labels(self, out: dict) -> tuple:
        """xhat (chart of the label's own seed) and Vhat of the neighborhood centred on every cell."""
        c, nb = self.c, self.nb
        xhat = c.C.copy()
        vhat = c.V * np.where(nb.is_seed, 1.0, out["coef_full"])      # a singleton: Vhat_k = w_kk V_k (the seeds are overwritten below)
        for k, j in enumerate(out["seeds"].tolist()):
            xhat[j] = out["xhat"][k]
            vhat[j] = out["vhat"][k]
        return xhat, vhat

    def stencil_cells(self, j: int) -> list:
        seen = {j}
        frontier = [j]
        order = []
        for _ in range(MAX_GRAPH_STENCIL):
            nxt = sorted({int(q) for i in frontier for q in self.g.neighbours(i)} - seen)
            seen.update(nxt)
            order.extend(nxt)
            frontier = nxt
        return order

    def build(self) -> dict:
        nb, c = self.nb, self.c
        self.averaging()
        if self.rejected_average:
            raise SB.Rejected(f"{len(self.rejected_average)} seeds cannot reach raw volume {SB.ALPHA_TARGET} within the limits", self.rejected_average)
        es, em = nb.entries()
        w, coef, n_count = nb.weights(es, em)
        geo = nb.geometry(es, em, w)
        coef_full = coef.copy()
        out = {"es": es, "em": em, "w": w, "coef": coef, "coef_full": coef_full, "n_count": n_count, **geo}
        xhat_l, vhat_l = self.labels(out)
        if not (vhat_l > 0).all():
            raise SB.Rejected("a label has Vhat <= 0", [int(i) for i in np.flatnonzero(~(vhat_l > 0))[:20]])
        seeds = out["seeds"]
        st_seed, st_cell, st_w, st_d, st_shift = [], [], [], [], []
        Bmat = np.zeros((len(seeds), 3, 3))
        ratio = np.zeros(len(seeds))
        for k, j in enumerate(seeds.tolist()):
            xj = out["xhat"][k]
            chosen = []
            for q in self.stencil_cells(j):
                shift = nb.shift(j, q)
                d = xhat_l[q] + shift - xj
                if np.linalg.norm(d) >= H_HALF:
                    chosen.append((q, shift, d))
            if len(chosen) < 3:
                self.rejected_stencil.append({"seed": j, "reason": "fewer than 3 stencil labels", "count": len(chosen)})
                continue
            D = np.array([d for _, _, d in chosen])
            Wk = np.array([vhat_l[q] for q, _, _ in chosen])
            B = (Wk[:, None, None] * D[:, :, None] * D[:, None, :]).sum(axis=0) / Wk.sum()
            sv = np.linalg.svd(B, compute_uv=False)
            rr = float(sv[-1] / max(sv[0], 1e-300))
            pts = np.vstack([D, np.zeros((1, 3))])                  # the centre j is part of the span check
            span = pts.max(axis=0) - pts.min(axis=0)
            if rr < RANK_RATIO_MIN or not (span >= SPAN_MIN).all():
                self.rejected_stencil.append({"seed": j, "rank_ratio": rr, "span": span.tolist(), "count": len(chosen),
                                              "stencil": [int(q) for q, _, _ in chosen], "members": list(nb.members[j])})
                continue
            Bmat[k], ratio[k] = B, rr
            for (q, shift, d), wk in zip(chosen, Wk):
                st_seed.append(k)
                st_cell.append(q)
                st_w.append(wk)
                st_d.append(d)
                st_shift.append(shift)
        if self.rejected_stencil:
            raise SB.Rejected(f"{len(self.rejected_stencil)} seeds have no admissible reconstruction stencil within graph distance {MAX_GRAPH_STENCIL}",
                              self.rejected_stencil)
        out.update(xhat_label=xhat_l, vhat_label=vhat_l, st_seed=np.array(st_seed, dtype=np.int64), st_cell=np.array(st_cell, dtype=np.int64),
                   st_w=np.array(st_w), st_d=np.array(st_d), st_shift=np.array(st_shift, dtype=np.int64), B=Bmat, b_ratio=ratio)
        return out


def combined_reach(sc: SrdConstruction, out: dict) -> dict:
    """The effective support of seed j: j, its stencil cells, and the members of the neighborhoods of those cells."""
    c, nb = sc.c, sc.nb
    graph, cheb = 0, 0
    starts = np.r_[np.flatnonzero(np.r_[True, out["st_seed"][1:] != out["st_seed"][:-1]]), len(out["st_seed"])]
    for k, j in enumerate(out["seeds"].tolist()):
        lo, hi = starts[k], starts[k + 1]
        cells = {int(q) for q in out["st_cell"][lo:hi]}
        support = set(cells) | {j}
        for q in cells | {j}:
            if q in nb.members:
                support.update(nb.members[q])
        cheb = max(cheb, max(nb.cheb(j, q) for q in support))
    return {"chebyshev_cells_from_seed": cheb, "stencil_graph_distance_cap": MAX_GRAPH_STENCIL, "members_graph_cap": SB.MAX_GRAPH}


def canonical_sets(sc: SrdConstruction, out: dict) -> dict:
    members = {str(j): sorted(sc.nb.members[j]) for j in out["seeds"].tolist()}
    starts = np.flatnonzero(np.r_[True, out["st_seed"][1:] != out["st_seed"][:-1]])
    ends = np.r_[starts[1:], len(out["st_seed"])]
    stencil = {str(int(out["seeds"][k])): sorted(int(q) for q in out["st_cell"][lo:hi]) for k, (lo, hi) in enumerate(zip(starts, ends))}
    text = json.dumps({"members": members, "stencil": stencil}, sort_keys=True, separators=(",", ":"))
    return {"members_sha256": hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "stencil_sha256": hashlib.sha256(json.dumps(stencil, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sets_sha256": hashlib.sha256(text.encode()).hexdigest(), "members": members, "stencil": stencil}


def report(sc: SrdConstruction, out: dict) -> dict:
    nb = sc.nb
    base = SB.report(nb, {**out, "ratio": out["b_ratio"], "passes": np.array(1)})
    base.pop("rank_ratio_min", None)
    base.pop("rank_ratio_five_worst", None)
    sizes = np.diff(np.r_[np.flatnonzero(np.r_[True, out["st_seed"][1:] != out["st_seed"][:-1]]), len(out["st_seed"])])
    order = np.argsort(out["b_ratio"])[:5]
    base.update(stencil_size_min_median_max=[int(sizes.min()), int(np.median(sizes)), int(sizes.max())], b_ratio_min=float(out["b_ratio"].min()),
                b_ratio_five_worst=[{"seed": int(out["seeds"][k]), "ratio": float(out["b_ratio"][k])} for k in order],
                label_vhat_min=float(out["vhat_label"].min()), label_vhat_max=float(out["vhat_label"].max()), combined_reach=combined_reach(sc, out),
                stencil_weight_min=float(out["st_w"].min()), averaging_rule="raw volume sum >= 0.5, no rank condition",
                constants={"ALPHA_TARGET": SB.ALPHA_TARGET, "MAX_GRAPH": SB.MAX_GRAPH, "MAX_CHEB": SB.MAX_CHEB, "MAX_GRAPH_STENCIL": MAX_GRAPH_STENCIL,
                           "H_HALF": H_HALF, "SPAN_MIN": SPAN_MIN, "RANK_RATIO_MIN": RANK_RATIO_MIN})
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True, type=Path, help="output stem: STEM.npz, STEM.json, STEM_sets.json")
    a = ap.parse_args()
    c = SB.Container(SB.CONTAINER)
    sc = SrdConstruction(c)
    try:
        out = sc.build()
    except SB.Rejected as error:
        a.out.with_suffix(".json").write_text(json.dumps({"release": a.release, "verdict": "REJECTED", "reason": str(error), "failing": error.failing}, indent=1))
        print(f"REJECTED: {error}; the exact failing neighborhoods are in {a.out.with_suffix('.json').name}")
        return 3
    rep = report(sc, out)
    sets = canonical_sets(sc, out)
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez(a.out.with_suffix(".npz"), **arrays)
    sets_path = a.out.parent / (a.out.name + "_sets.json")
    sets_path.write_text(json.dumps(sets, sort_keys=True, separators=(",", ":")))
    rep.update(release=a.release, verdict="BUILT", container_sha256=c.sha, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               build_tool_sha256=hashlib.sha256((HERE / "srd_build.py").read_bytes()).hexdigest(),
               npz_sha256=hashlib.sha256(a.out.with_suffix(".npz").read_bytes()).hexdigest(), sets_sha256=sets["sets_sha256"],
               members_sha256=sets["members_sha256"], stencil_sha256=sets["stencil_sha256"], sets_file_sha256=hashlib.sha256(sets_path.read_bytes()).hexdigest())
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
