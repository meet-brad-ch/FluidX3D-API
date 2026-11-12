"""Task 39 library: container I/O, the analytic slab target, exact cell averages, friction rows (present and P1), fi decoding.
No engine call and no run logic here. Geometry: walls q = 3 x + z = c1, c2; n = (3, 0, 1)/sqrt(10); eta = (q - c1)/sqrt(10).
"""
from __future__ import annotations

import itertools
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

LEX = np.array([c for c in itertools.product((-1, 0, 1), repeat=3) if sum(x * x for x in c) <= 2], float)     # 19, rest = index 9
LEX_W = np.array([1.0 / 3.0 if s == 0 else (1.0 / 18.0 if s == 1 else 1.0 / 36.0) for s in (LEX ** 2).sum(axis=1)])
FX = np.array([[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0],
               [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1],
               [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1]]).T              # engine direction table
FX_W = np.where((FX ** 2).sum(axis=1) == 0, 1 / 3, np.where((FX ** 2).sum(axis=1) == 1, 1 / 18, 1 / 36))
SQ10 = math.sqrt(10.0)
N_HAT = np.array([3.0, 0.0, 1.0]) / SQ10          # wall normal of wall B, into the fluid
T1 = np.array([1.0, 0.0, -3.0]) / SQ10            # in-plane tangent with a nonzero shell source
E_Y = np.array([0.0, 1.0, 0.0])                   # in-plane tangent with zero shell source (control)
ELEMENT = {1: ("<u4", 4), 2: ("<i4", 4), 3: ("<f8", 8), 4: (None, 1)}


# ---- container ---------------------------------------------------------------------------------------------------------------
def read_sections(raw: bytes) -> list:
    """Ordered (name, dtype code, payload bytes) of a VWNAT01 container."""
    if raw[:8] != b"VWNAT01\n":
        raise ValueError("not a VWNAT01 container")
    out, pos = [], 12
    for _ in range(struct.unpack_from("<I", raw, 8)[0]):
        name = raw[pos:pos + 32].rstrip(b"\0").decode("ascii")
        dtype, n = struct.unpack_from("<IQ", raw, pos + 32)
        size = ELEMENT[dtype][1]
        out.append((name, dtype, raw[pos + 44:pos + 44 + n * size]))
        pos += 44 + n * size
    if pos != len(raw):
        raise ValueError("trailing bytes")
    return out


def write_sections(sections: list) -> bytes:
    parts = [b"VWNAT01\n", struct.pack("<I", len(sections))]
    for name, dtype, payload in sections:
        parts += [name.encode("ascii").ljust(32, b"\0"), struct.pack("<IQ", dtype, len(payload) // ELEMENT[dtype][1]), payload]
    return b"".join(parts)


def arrays(sections: list) -> dict:
    return {name: (payload.decode("ascii") if ELEMENT[dtype][0] is None else np.frombuffer(payload, dtype=ELEMENT[dtype][0]))
            for name, dtype, payload in sections}


# ---- analytic target -----------------------------------------------------------------------------------------------------------
@dataclass
class Slab:
    """One case geometry and physics in lattice units."""
    c1: float
    c2: float
    Lz: int
    nu: float
    K: float

    @property
    def H(self) -> float:
        return (self.c2 - self.c1) / SQ10

    @property
    def beta(self) -> float:
        return self.K / 3.0                        # friction traction beta rho u_t with beta = K c_s^2

    @property
    def biot_half(self) -> float:
        return self.beta * self.H / (2.0 * self.nu)

    def roots(self, count: int = 40) -> np.ndarray:
        """x_n > 0 with x tan x = beta H / (2 nu): symmetric modes cos(2 x_n (eta - H/2) / H) of the WALL-VALUE Robin law."""
        B = self.biot_half
        return np.array([brentq(lambda x: x * math.tan(x) - B, n * math.pi + 1e-14, n * math.pi + math.pi / 2 - 1e-12) for n in range(count)])

    def rate(self, x: float) -> float:
        return self.nu * (2.0 * x / self.H) ** 2

    def sampled_root(self, d: float = 0.5) -> float:
        """Lowest symmetric root of the SAMPLED law nu u'(0) = beta u(d): nu q sin(x) = beta cos(x - q d), x = q H / 2."""
        H, nu, beta = self.H, self.nu, self.beta
        x0 = float(self.roots(1)[0])
        return brentq(lambda x: nu * (2 * x / H) * math.sin(x) - beta * math.cos(x - 2 * x * d / H), 0.5 * x0, min(1.5 * x0, math.pi / 2 - 1e-9))

    def series(self, eta: np.ndarray, t: float, count: int = 40) -> np.ndarray:
        """u(eta, t) / U for the uniform initial condition (point values)."""
        x = self.roots(count)
        A = 4.0 * np.sin(x) / (2.0 * x + np.sin(2.0 * x))
        return sum(A[n] * np.cos(2.0 * x[n] * (eta - self.H / 2.0) / self.H) * math.exp(-self.rate(x[n]) * t) for n in range(count))


GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def cell_average(slab: Slab, qc: np.ndarray, g) -> tuple:
    """Exact fluid volume and fluid average of g(eta) for unit cells whose centres have q = 3 x + z = qc (#895).
    The density of q - qc over a unit cube is the trapezoid f(r) = (r + 2)/3 on [-2, -1], 1/3 on [-1, 1], (2 - r)/3 on [1, 2].
    Each linear piece is clipped to the fluid interval of the slab image nearest to the cell and integrated with a 24-point
    Gauss rule (exact to rounding for the smooth integrands used here)."""
    qc = np.asarray(qc, float)
    m = np.round((qc - 0.5 * (slab.c1 + slab.c2)) / slab.Lz)                 # nearest slab image
    lo, hi = slab.c1 + m * slab.Lz, slab.c2 + m * slab.Lz
    vol, acc = np.zeros_like(qc), np.zeros_like(qc)
    for a, b, f in ((-2.0, -1.0, lambda r: (r + 2.0) / 3.0), (-1.0, 1.0, lambda r: np.full_like(r, 1.0 / 3.0)), (1.0, 2.0, lambda r: (2.0 - r) / 3.0)):
        A, Bq = np.maximum(qc + a, lo), np.minimum(qc + b, hi)
        w = np.maximum(Bq - A, 0.0)
        q = A[:, None] + 0.5 * w[:, None] * (GL_X[None, :] + 1.0)
        dens = f(q - qc[:, None]) * (0.5 * w[:, None] * GL_W[None, :])
        vol += dens.sum(axis=1)
        acc += (dens * g((q - lo[:, None]) / SQ10)).sum(axis=1)
    return vol, acc / vol


# ---- friction rows ---------------------------------------------------------------------------------------------------------------
def present_rows(g: dict) -> np.ndarray:
    """phi_k = 6 w_k vi_k^2 / A * P_t c_k: the kernel's own friction expression, per exported row (rows x 3)."""
    nrm, area = g["normal"].reshape(-1, 3), g["area"]
    fo, rd, vi = g["facet_offsets"], g["row_direction"].astype(int), g["prism_volume"]
    phi = np.zeros((len(rd), 3))
    for a in range(len(area)):
        k = np.arange(fo[a], fo[a + 1])
        Pt = np.eye(3) - np.outer(nrm[a], nrm[a])
        phi[k] = (6.0 * LEX_W[rd[k]] * vi[k] ** 2 / area[a])[:, None] * (LEX[rd[k]] @ Pt)
    return phi


def p1_rows(g: dict) -> tuple:
    """Minimum-norm correction per facet: zero sum of the rows in each speed shell, unchanged impulse tensor.
    Returns (rows x 3, report)."""
    nrm, area = g["normal"].reshape(-1, 3), g["area"]
    fo, rd, vi = g["facet_offsets"], g["row_direction"].astype(int), g["prism_volume"]
    phi = present_rows(g)
    out = phi.copy()
    rep = {"solve_residual": 0.0, "correction_rel_max": 0.0, "row_change_over_vi_max": 0.0, "shell_sum_max": 0.0, "tensor_change_max": 0.0}
    for a in range(len(area)):
        k = np.arange(fo[a], fo[a + 1])
        c = LEX[rd[k]]
        shell = ((c ** 2).sum(axis=1) > 1.5).astype(int)
        Pt = np.eye(3) - np.outer(nrm[a], nrm[a])
        tb = np.linalg.svd(Pt)[0][:, :2]
        m = len(k)
        A, b = [], []
        for s in (0, 1):
            for q in range(2):
                r = np.zeros((m, 2))
                r[shell == s, q] = 1.0
                A.append(r.ravel())
                b.append(-(phi[k][shell == s].sum(axis=0) @ tb[:, q]))
        for i in range(3):
            for q in range(2):
                r = np.zeros((m, 2))
                r[:, q] = c[:, i]
                A.append(r.ravel())
                b.append(0.0)
        A, b = np.array(A), np.array(b)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        out[k] = phi[k] + x.reshape(m, 2) @ tb.T
        rep["solve_residual"] = max(rep["solve_residual"], float(np.abs(A @ x - b).max()) / area[a])
        rep["correction_rel_max"] = max(rep["correction_rel_max"], float(np.linalg.norm(out[k] - phi[k]) / np.linalg.norm(phi[k])))
        rep["row_change_over_vi_max"] = max(rep["row_change_over_vi_max"], float((np.linalg.norm(out[k] - phi[k], axis=1) / vi[k]).max()))
        rep["shell_sum_max"] = max(rep["shell_sum_max"], max(float(np.abs(out[k][shell == s].sum(axis=0)).max()) for s in (0, 1)) / area[a])
        rep["tensor_change_max"] = max(rep["tensor_change_max"], float(np.abs(c.T @ (out[k] - phi[k])).max()) / area[a])
    return out, rep


def friction_impulse(g: dict, phi: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Friction-only impulse per facet after the shell balance for unit K rho and the uniform state u (facets x 3)."""
    nrm = g["normal"].reshape(-1, 3)
    fo, rd, vi = g["facet_offsets"], g["row_direction"].astype(int), g["prism_volume"]
    J = np.zeros((len(g["area"]), 3))
    for a in range(len(J)):
        k = np.arange(fo[a], fo[a + 1])
        c = LEX[rd[k]]
        shell = ((c ** 2).sum(axis=1) > 1.5).astype(int)
        ut = u - nrm[a] * (nrm[a] @ u)
        q = -(phi[k] @ ut)
        for s in (0, 1):
            msk = shell == s
            q[msk] += vi[k][msk] * (phi[k][msk] @ ut).sum() / vi[k][msk].sum()
        J[a] = c.T @ q
    return J


# ---- lattice decoding ----------------------------------------------------------------------------------------------------------
class Lattice:
    """Raw fi dump (slot * N + cell) -> level-t pre-collision densities of the fluid cells (engine direction order)."""

    def __init__(self, g: dict) -> None:
        self.box = [int(v) for v in g["box"]]
        self.N = self.box[0] * self.box[1] * self.box[2]
        self.xyz = g["cell_xyz"].reshape(-1, 3).astype(np.int64)
        self.n = self.index(self.xyz)
        self.vol = np.asarray(g["volume"], float)
        self.nb = [self.index(self.xyz + FX[i].astype(np.int64)) for i in range(19)]

    def index(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = (np.mod(xyz[:, a], self.box[a]) for a in range(3))
        return x + (y + z * self.box[1]) * self.box[0]

    def decode(self, fi: np.ndarray, t: int) -> np.ndarray:
        f = np.empty((len(self.n), 19))
        f[:, 0] = fi[self.n]
        for i in range(1, 19, 2):
            f[:, i] = fi[(i if t % 2 else i + 1) * self.N + self.n]
            f[:, i + 1] = fi[(i + 1 if t % 2 else i) * self.N + self.nb[i]]
        return f.astype(np.float64) + FX_W

    def moments(self, fi: np.ndarray, t: int) -> tuple:
        f = self.decode(fi, t)
        rho = f.sum(axis=1)
        return rho, (f @ FX) / rho[:, None], f


def equilibrium(u: np.ndarray) -> np.ndarray:
    """f^eq_d(1, u), lexicographic order (the bounded operator's second-order equilibrium)."""
    cu = LEX @ u
    return LEX_W * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * float(u @ u))


def sha(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False), encoding="utf-8")
