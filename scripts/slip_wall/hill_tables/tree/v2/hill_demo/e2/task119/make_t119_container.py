"""Task 119 (E2): the container of the experimental engine port: the registered hill container with (a) the force-compatible start and
(b) the two companion sections of the whole-prism correction. Reference plumbing; writes one file, changes no input.

STATUS: written, NOT executed. Needs --release (a label of the message that releases the container build).

  1. f0 := feq(1, 0) - Phi(0, F)/2 for the force F, computed by the PINNED reference equilibrium (the task-115 start, MATHEMATICAL_MODEL
     15.9), in place of the stored f0. s0 is not changed; it must equal the T115 start s0 = W w_row (checked, reported).
  2. t119_gdelta: 3 doubles per G entry (stored CSR order), the offset of the prism part from the prism centroid, from the verified companion
     (zeros for the rows without W, which the kernel never reads). t119_wmoment: 3 doubles per W entry (stored CSR order), the owned first moment.
  Every other section is byte-identical (readback checked). The provenance section is not edited.
  Checks against the container before anything is written: companion sha256; the W structure (receiver row, producer row) and the stored W
  weights equal the companion's; the companion G entry list is exactly all G entries of all rows with W; all values finite.

  envs/solver_poc/Scripts/python.exe -B v2/hill_demo/e2/task119/make_t119_container.py --release ID --base FILE --base-sha PREFIX
      --companion NPZ --companion-sha SHA --force FX,FY,FZ --out FILE.vwnat
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
import r1_correction as R1  # noqa: E402

DH = R1.DH
T39 = DH.PR2.OM.T39
Invalid = R1.Invalid


def csr(a: dict, name: str):
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    shape = tuple(int(x) for x in a[f"{name}_shape"])
    return csr_matrix((np.asarray(a[f"{name}_weights"], dtype=np.float64), np.asarray(a[f"{name}_indices"], dtype=np.int64),
                       np.asarray(a[f"{name}_offsets"], dtype=np.int64)), shape=shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--companion", required=True, type=Path)
    ap.add_argument("--companion-sha", required=True)
    ap.add_argument("--force", required=True, help="FX,FY,FZ lattice units")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    raw = a.base.read_bytes()
    base_sha = hashlib.sha256(raw).hexdigest()
    if not base_sha.startswith(a.base_sha):
        raise Invalid(f"base container sha256 {base_sha[:16]} is not {a.base_sha}")
    comp_raw = a.companion.read_bytes()
    if hashlib.sha256(comp_raw).hexdigest() != a.companion_sha:
        raise Invalid("companion sha256 differs from the registered one")
    sections = T39.read_sections(raw)
    arrs = T39.arrays(sections)
    z = np.load(a.companion)
    W, G = csr(arrs, "W"), csr(arrs, "G")
    recv = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    if not (np.array_equal(recv, z["w_recv_row"]) and np.array_equal(W.indices.astype(np.int64), z["w_prod_row"])):
        raise Invalid("companion W structure differs from the container")
    if not np.array_equal(W.data, z["w_volume_stored"]):
        raise Invalid("companion stored W weights differ from the container")
    g_rows = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
    expected = np.flatnonzero(np.isin(g_rows, np.unique(z["w_prod_row"])))
    if not np.array_equal(z["g_entry_index"], expected):
        raise Invalid("companion G entry list is not exactly all G entries of all rows with W")
    delta = np.zeros((G.nnz, 3))
    delta[z["g_entry_index"]] = z["g_delta"]
    moment = np.asarray(z["w_moment"], dtype=np.float64)
    if not (np.isfinite(delta).all() and np.isfinite(moment).all() and moment.shape == (W.nnz, 3)):
        raise Invalid("companion values are not finite or have the wrong shape")
    # the force-compatible start from the pinned reference equilibrium (the task-115 start)
    force = np.array([float(v) for v in a.force.split(",")])
    m = DH.ForcedClosedModel(a.base, force)
    rho0 = 1.0
    feq, phi = m.eq.parts(np.full(m.cells, rho0), np.zeros((m.cells, 3)), force)
    f0 = feq - 0.5 * phi
    s0_ref = rho0 * (m.W @ m.w_row)
    s0_container = np.asarray(arrs["s0"], dtype=np.float64)
    s0_diff = float(np.abs(s0_ref - s0_container).max())
    if not (np.isfinite(f0).all() and f0.min() > 0.0):
        raise Invalid("the force-compatible f0 is not finite and positive")
    replace = {"f0": np.ascontiguousarray(f0, "<f8").tobytes()}
    out_sections = [(n, d, replace.get(n, p)) for n, d, p in sections]
    insert_at = next(i for i, (n, _, _) in enumerate(out_sections) if n == "provenance")
    out_sections.insert(insert_at, ("t119_wmoment", 3, np.ascontiguousarray(moment, "<f8").tobytes()))
    out_sections.insert(insert_at, ("t119_gdelta", 3, np.ascontiguousarray(delta, "<f8").tobytes()))
    data = T39.write_sections(out_sections)
    back = T39.read_sections(data)
    old = {n: p for n, _, p in sections}
    changed = [n for n, _, p in back if n in old and old[n] != p]
    added = [n for n, _, _ in back if n not in old]
    if changed != ["f0"] or sorted(added) != ["t119_gdelta", "t119_wmoment"]:
        raise Invalid(f"readback: changed {changed}, added {added}")
    back_arr = T39.arrays(back)
    if not (np.array_equal(back_arr["t119_gdelta"].reshape(-1, 3), delta) and np.array_equal(back_arr["t119_wmoment"].reshape(-1, 3), moment)
            and np.array_equal(back_arr["f0"].reshape(-1, 27), f0)):
        raise Invalid("readback of the new sections differs")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    record = {"release": a.release, "base": str(a.base), "base_sha256": base_sha, "companion_sha256": a.companion_sha,
              "out": str(a.out), "out_sha256": hashlib.sha256(data).hexdigest(), "force": force.tolist(), "changed_sections": changed,
              "added_sections": added, "s0_vs_reference_max_abs": s0_diff, "f0_min": float(f0.min()),
              "g_entries": int(G.nnz), "w_entries": int(W.nnz), "rows_with_w": int(len(np.unique(z["w_prod_row"]))), "tool_sha256":
              hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
