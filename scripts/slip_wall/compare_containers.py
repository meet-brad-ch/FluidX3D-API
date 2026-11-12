"""Compares two VWNAT01 containers section by section: the gate of the compiled builder (volumetric_wall_builder).

  python compare_containers.py REFERENCE.vwnat CANDIDATE.vwnat

Every section except "provenance" must be byte-identical. The provenance texts are compared as JSON: they may differ only in
the fields that name the exporter (the Python builder writes "exporter", "python", "numpy", "scipy"; the compiled one
"exporter", "openblas"). For a differing numeric section the first differing element and the count of differing elements
are printed. Exit code 0 when the containers pass, 1 otherwise.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

KINDS = {1: "<u4", 2: "<i4", 3: "<f8", 4: "S1"}
SIZES = {1: 4, 2: 4, 3: 8, 4: 1}
EXPORTER_FIELDS = {"exporter", "python", "numpy", "scipy", "openblas"}


class Container:
    """@brief The sections of one VWNAT01 file, in file order."""

    def __init__(self, path: Path) -> None:
        data = path.read_bytes()
        if data[:8] != b"VWNAT01\n":
            raise SystemExit(f"{path}: not a VWNAT01 file")
        (count,) = struct.unpack_from("<I", data, 8)
        at = 12
        self.order, self.sections = [], {}
        for _ in range(count):
            name = data[at:at + 32].rstrip(b"\0").decode("ascii")
            dtype, elements = struct.unpack_from("<IQ", data, at + 32)
            at += 44
            size = elements * SIZES[dtype]
            self.order.append(name)
            self.sections[name] = (dtype, data[at:at + size])
            at += size
        if at != len(data):
            raise SystemExit(f"{path}: {len(data) - at} bytes after the last section")


def compare_numeric(name: str, ref: tuple, cand: tuple) -> bool:
    """@brief Prints the difference of one numeric section. @return True when byte-identical"""
    if ref == cand:
        return True
    if ref[0] != cand[0]:
        print(f"{name}: dtype {ref[0]} vs {cand[0]}")
        return False
    a = np.frombuffer(ref[1], KINDS[ref[0]])
    b = np.frombuffer(cand[1], KINDS[cand[0]])
    if a.size != b.size:
        print(f"{name}: {a.size} vs {b.size} elements")
        return False
    raw = {1: "<u1", 4: "<u4", 8: "<u8"}[SIZES[ref[0]]]   # bit patterns: -0.0 and NaN payloads count as differences
    differ = np.flatnonzero(np.frombuffer(ref[1], raw) != np.frombuffer(cand[1], raw))
    first = int(differ[0])
    print(f"{name}: {differ.size} of {a.size} elements differ; first at {first}: {a[first]!r} vs {b[first]!r}")
    return False


def compare_provenance(ref: bytes, cand: bytes) -> bool:
    """@brief Compares the provenance JSON, ignoring the exporter fields. @return True when equal otherwise"""
    a, b = json.loads(ref.decode("ascii")), json.loads(cand.decode("ascii"))
    keys = [k for k in dict.fromkeys(list(a) + list(b)) if k not in EXPORTER_FIELDS]
    same = True
    for key in keys:
        if a.get(key) != b.get(key):
            print(f"provenance[{key!r}]: {a.get(key)!r} vs {b.get(key)!r}")
            same = False
    return same


def main() -> int:
    """@brief Command line entry. @return the exit code"""
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_containers.py REFERENCE.vwnat CANDIDATE.vwnat")
    ref, cand = Container(Path(sys.argv[1])), Container(Path(sys.argv[2]))
    passed = True
    if ref.order != cand.order:
        print(f"section order differs:\n  {ref.order}\n  {cand.order}")
        passed = False
    for name in ref.order:
        if name not in cand.sections:
            print(f"{name}: missing")
            passed = False
        elif name == "provenance":
            passed = compare_provenance(ref.sections[name][1], cand.sections[name][1]) and passed
        else:
            passed = compare_numeric(name, ref.sections[name], cand.sections[name]) and passed
    print("PASS: every numeric section byte-identical, provenance equal except the exporter fields" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
