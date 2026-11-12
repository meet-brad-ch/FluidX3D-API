"""Task 121 (E1): add the SRD sections (schema T121_SRD_SECTION_SCHEMA_E1.md) to the T119 hill container.

  python -B make_t121_container.py --base hill_t119.vwnat --base-sha 4d931c28... --tables srd.npz --tables-sha 3edc0009... --out hill_t121.vwnat
The values are copied from the npz without change (FP64 bitwise); derived fields use integer or exact-sum operations only. A readback
asserts bitwise equality and that every pre-existing section is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "integration/task39"))
import t39_lib as T39  # noqa: E402

U32, F64 = 1, 3


def u32(a) -> bytes:
    arr = np.asarray(a)
    if arr.size and (arr.min() < 0 or arr.max() > 0xFFFFFFFF):
        raise ValueError("value outside u32")
    return np.ascontiguousarray(arr, "<u4").tobytes()


def f64(a) -> bytes:
    arr = np.asarray(a, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite value")
    return np.ascontiguousarray(arr, "<f8").tobytes()


def i32(a) -> bytes:
    return np.ascontiguousarray(np.asarray(a), "<i4").tobytes()


def build(z, cells: int) -> list:
    """The 18 t121 sections as (name, dtype code, payload)."""
    seeds, em, starts = z["seeds"].astype(np.int64), z["em"].astype(np.int64), z["starts"].astype(np.int64)
    S, E = len(seeds), len(em)
    st_seed, st_cell = z["st_seed"].astype(np.int64), z["st_cell"].astype(np.int64)
    if len(z["coef_full"]) != cells:
        raise ValueError("the tables do not belong to this container (cell count)")
    if not np.all(np.diff(z["seed_of"]) >= 0) or not np.array_equal(np.flatnonzero(np.r_[True, z["seed_of"][1:] != z["seed_of"][:-1]]), starts):
        raise ValueError("entries are not grouped by seed in order")
    entry_starts = np.r_[starts, E]
    st_starts = np.r_[np.flatnonzero(np.r_[True, st_seed[1:] != st_seed[:-1]]), len(st_seed)]
    if len(st_starts) != S + 1 or not np.array_equal(st_seed[st_starts[:-1]], np.arange(S)):
        raise ValueError("stencil entries are not grouped by seed in order")
    position = np.minimum(np.searchsorted(seeds, st_cell), S - 1)
    st_seed_index = np.where(seeds[position] == st_cell, position, -1)
    st_wsum = np.add.reduceat(z["st_w"], st_starts[:-1])
    perm = np.argsort(em, kind="stable")
    cells_a, first = np.unique(em[perm], return_index=True)
    active_starts = np.r_[first, E]
    coef_a = z["coef_full"][cells_a]
    return [("t121_seeds", U32, u32(seeds)), ("t121_entry_starts", U32, u32(entry_starts)), ("t121_entry_cell", U32, u32(em)),
            ("t121_entry_w", F64, f64(z["w"])), ("t121_entry_wv", F64, f64(z["wv"])), ("t121_entry_delta", F64, f64(z["delta"])),
            ("t121_vhat", F64, f64(z["vhat"])), ("t121_st_starts", U32, u32(st_starts)), ("t121_st_cell", U32, u32(st_cell)),
            ("t121_st_w", F64, f64(z["st_w"])), ("t121_st_d", F64, f64(z["st_d"])), ("t121_st_seed_index", 2, i32(st_seed_index)),
            ("t121_B", F64, f64(z["B"])), ("t121_st_wsum", F64, f64(st_wsum)), ("t121_active_cells", U32, u32(cells_a)),
            ("t121_active_coef", F64, f64(coef_a)), ("t121_active_starts", U32, u32(active_starts)), ("t121_active_entry", U32, u32(perm))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--tables-sha", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    raw = a.base.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.base_sha:
        raise SystemExit("base container sha256 differs from the registered one")
    if hashlib.sha256(a.tables.read_bytes()).hexdigest() != a.tables_sha:
        raise SystemExit("tables sha256 differs from the registered one")
    sections = T39.read_sections(raw)
    cells = int(T39.arrays(sections)["counts"][0])
    z = np.load(a.tables)
    new = build(z, cells)
    insert_at = next(i for i, (n, _, _) in enumerate(sections) if n == "provenance")
    out_sections = sections[:insert_at] + new + sections[insert_at:]
    data = T39.write_sections(out_sections)
    back = T39.read_sections(data)
    old = {n: p for n, _, p in sections}
    if [n for n, _, p in back if n in old and old[n] != p] or sorted(n for n, _, _ in back if n not in old) != sorted(n for n, _, _ in new):
        raise SystemExit("readback: a base section changed or the added set differs")
    arr = T39.arrays(back)
    checks = {"t121_entry_w": z["w"], "t121_entry_wv": z["wv"], "t121_entry_delta": z["delta"], "t121_vhat": z["vhat"], "t121_st_w": z["st_w"],
              "t121_st_d": z["st_d"], "t121_B": z["B"]}
    for name, ref in checks.items():
        if not np.array_equal(arr[name].reshape(ref.shape), ref):
            raise SystemExit(f"readback differs bitwise: {name}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    record = {"base_sha256": a.base_sha, "tables_sha256": a.tables_sha, "out": str(a.out), "out_sha256": hashlib.sha256(data).hexdigest(),
              "added_sections": [n for n, _, _ in new], "seeds": int(len(z["seeds"])), "entries": int(len(z["em"])), "stencil_entries": int(len(z["st_cell"])),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
