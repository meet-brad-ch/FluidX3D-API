"""Writes the reference vectors of the C++ wall builder's exact arithmetic (tests/unit/wall_builder_test.cpp).

Python's own integers and true division are the reference: builder3d.py takes every volume as int / int (round to
nearest, ties to even) and every exact plane value as float(Fraction). One line per case:

  S a b a+b            sum
  M a b a*b            product
  R n d e hex          float(n / d * 2^e) as float.hex()
  D x y floor sum product   x, y as float.hex(); floor(x + y) exact; float(x + y) and float(x * y) as float.hex()
  A y x hex            math.atan2(y, x)
  H x y hex            math.hypot(x, y)

The file is deterministic (fixed seed). Usage: python make_wall_builder_vectors.py OUT.txt
"""
from __future__ import annotations

import math
import random
import sys
from fractions import Fraction
from pathlib import Path

SEED = 20261112
MAX_BITS = 1000          # the C++ integer has 1024 bits
SMALLEST_NORMAL = 2.2250738585072014e-308


class VectorWriter:
    """@brief Collects the reference lines."""

    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.lines: list = []

    def signed(self, bits: int) -> int:
        """@brief A random integer of exactly `bits` bits (0 for bits 0) with a random sign."""
        if bits == 0:
            return 0
        value = self.rng.getrandbits(bits) | (1 << (bits - 1))
        return -value if self.rng.random() < 0.5 else value

    def rounding(self, n: int, d: int, e: int) -> None:
        """@brief Adds an R line when the result is a normal binary64."""
        exact = Fraction(n * 2 ** max(e, 0), d * 2 ** max(-e, 0))
        try:
            value = float(exact)
        except OverflowError:
            return
        if value != 0.0 and abs(value) < SMALLEST_NORMAL:
            return
        if value in (float("inf"), float("-inf")):
            return
        self.lines.append(f"R {n} {d} {e} {value.hex()}")

    def arithmetic(self, count: int) -> None:
        """@brief Sums and products whose results stay inside 1024 bits."""
        for _ in range(count):
            a_bits = self.rng.randint(0, 500)
            b_bits = self.rng.randint(0, 500)
            a, b = self.signed(a_bits), self.signed(b_bits)
            self.lines.append(f"S {a} {b} {a + b}")
            self.lines.append(f"M {a} {b} {a * b}")

    def random_quotients(self, count: int) -> None:
        """@brief Random numerators, small divisors, exponents that keep the result normal."""
        for _ in range(count):
            n = self.signed(self.rng.randint(1, MAX_BITS - 70))
            d = self.rng.choice((1, 3, 5, 6, 7, 3 * 2 ** 10))
            target = self.rng.randint(-1000, 1000)
            e = target - (abs(n).bit_length() - d.bit_length())
            self.rounding(n, d, e)

    def ties(self, count: int) -> None:
        """@brief Exact halfway cases and their neighbours, for divisors 1 and 3."""
        for _ in range(count):
            m = self.rng.getrandbits(52) | (1 << 52)          # 53 significant bits
            k = self.rng.randint(0, 300)
            tie = (2 * m + 1) << k                              # exactly halfway between two binary64 values
            for n in (tie, tie - 1, tie + 1, -tie):
                self.rounding(n, 1, -k - 60)
                self.rounding(3 * n, 3, -k - 60)

    def volumes(self, count: int) -> None:
        """@brief The shape of a builder volume: a determinant sum over 6 D^3, D = 2^p."""
        for _ in range(count):
            p = self.rng.randint(0, 300)       # L24 measured D up to 2^164 (2026-11-12)
            total = abs(self.signed(self.rng.randint(1, min(MAX_BITS - 70, 3 * p + 8))))
            self.rounding(total, 3, -(1 + 3 * p))

    def height(self) -> float:
        """@brief A random binary64 in the range of terrain values, sometimes tiny or an integer."""
        kind = self.rng.random()
        if kind < 0.1:
            return float(self.rng.randint(-50, 50))
        if kind < 0.2:
            return self.rng.uniform(-1.0, 1.0) * 2.0 ** self.rng.randint(-200, -20)
        return self.rng.uniform(-60.0, 60.0)

    def dyadics(self, count: int) -> None:
        """@brief Exact sums and products of binary64 values (builder3d.py takes them as Fractions)."""
        for _ in range(count):
            x, y = self.height(), self.height()
            total = Fraction(x) + Fraction(y)
            product = Fraction(x) * Fraction(y)
            self.lines.append(f"D {x.hex()} {y.hex()} {math.floor(total)} {float(total).hex()} {float(product).hex()}")

    def angles(self, count: int) -> None:
        """@brief math.atan2 on the values order_cycle gives it: dot products of small vectors, signed zeros, axes."""
        specials = (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-300, -1e-300)
        for y in specials:
            for x in specials:
                self.lines.append(f"A {y.hex()} {x.hex()} {math.atan2(y, x).hex()}")
        for _ in range(count):
            y = self.rng.uniform(-2.0, 2.0) * 2.0 ** self.rng.randint(-40, 0)
            x = self.rng.uniform(-2.0, 2.0) * 2.0 ** self.rng.randint(-40, 0)
            self.lines.append(f"A {y.hex()} {x.hex()} {math.atan2(y, x).hex()}")

    def hypots(self) -> None:
        """@brief math.hypot on the only inputs builder3d.cell_pieces gives it: edge vectors with components -1, 0, 1.
        Python's math.hypot is its own algorithm, not the C library's: on random inputs 26 of 200 differ from std::hypot
        by one ulp (measured 2026-11-12), so the C++ builder accepts only these inputs and stops on any other."""
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                self.lines.append(f"H {x.hex()} {y.hex()} {math.hypot(x, y).hex()}")

    def small(self) -> None:
        """@brief Small numerators, where the C++ code must shift before it divides."""
        for n in range(1, 200):
            for d in (1, 3, 6, 7):
                self.rounding(n, d, 0)
                self.rounding(-n, d, -5)


def main() -> int:
    """@brief Command line entry. @return the exit code"""
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_wall_builder_vectors.py OUT.txt")
    writer = VectorWriter()
    writer.arithmetic(400)
    writer.random_quotients(1500)
    writer.ties(300)
    writer.volumes(1000)
    writer.small()
    writer.dyadics(2000)
    writer.angles(20000)
    writer.hypots()
    Path(sys.argv[1]).write_text("\n".join(writer.lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(writer.lines)} vectors written to {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
