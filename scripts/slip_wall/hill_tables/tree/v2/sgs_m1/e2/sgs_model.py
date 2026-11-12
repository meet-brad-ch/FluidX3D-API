"""V2-SGS-01 (E2): the static Smagorinsky candidate in the M1 FP64 reference (CPU) and its parallel Jacobian route.

Released #2132 (reference only; no engine patch). Derivation: V2_SGS_01_M1_DERIVATION_E2_2026-11-10.md (586be055 + fdf5c8cf).
ONE change to the frozen model: the collision rate becomes a per-cell value omega_l,

    p = feq + (1 - omega_l) fneq_RR(A, u) + Phi/2                  (Phi unchanged)
    A = sum c c (f + Phi/2) - rho/3 I - rho u u                    (the frozen M1 moment, FULL A)
    tau_l = tau0/2 + sqrt(tau0^2 + k sqrt(A:A)/rho)/2,  omega_l = 1/tau_l,  tau0 = 1/omega0
    k = 18 sqrt(2) ell2,  ell2 = (C Delta)^2,  Delta = 1 in every cell.

The chosen constant (#2132): k = float32(0.76421222) widened exactly to binary64, i.e. ell2 = k / (18 sqrt(2)).
k == 0 calls the frozen collide unchanged (the zero-SGS path is the original code, bitwise).

Classes:
  SgsClosure                 the constant and the local rate (numpy or cupy arrays: the same expression on both)
  SgsWallReferenceModel27    t89_model.WallReferenceModel27 with the SGS collide (transport, wall pass, ledgers inherited)
The parallel Jacobian route is in sgs_parallel.py (as t89_parallel is apart from t89_model).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parents[2] / "integration"
for folder in ("task87", "task89"):
    sys.path.insert(0, str(INTEGRATION / folder))
import t89_model as M89  # noqa: E402  (frozen)

Invalid = M89.Invalid


@dataclass(frozen=True)
class SgsClosure:
    """The static Smagorinsky constant k = 18 sqrt(2) (C Delta)^2 and the local rate of (C2)."""

    k: float

    CHOSEN = float(np.float32(0.76421222))      # #2132: float32(0.76421222) widened exactly to binary64

    @staticmethod
    def chosen() -> "SgsClosure":
        return SgsClosure(SgsClosure.CHOSEN)

    @staticmethod
    def off() -> "SgsClosure":
        return SgsClosure(0.0)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k >= 0.0):
            raise Invalid("the SGS constant k must be finite and >= 0")

    @property
    def ell2(self) -> float:
        """(C Delta)^2 with Delta = 1; for the record only (the update uses k)."""
        return self.k / (18.0 * math.sqrt(2.0))

    def rate(self, A, rho, omega0: float, xp=np):
        """omega_l per cell. A: cells x 3 x 3 (the full M1 moment), rho: cells. xp: numpy or cupy (the same expression)."""
        Q = (A * A).sum(axis=(1, 2))
        tau0 = 1.0 / omega0
        tau = 0.5 * tau0 + 0.5 * xp.sqrt(tau0 * tau0 + self.k * xp.sqrt(Q) / rho)
        return 1.0 / tau


class SgsWallReferenceModel27(M89.WallReferenceModel27):
    """WallReferenceModel27 whose collision uses the per-cell rate of SgsClosure."""

    def __init__(self, container: Path, force: np.ndarray, closure: SgsClosure) -> None:
        super().__init__(container, force)
        if not isinstance(closure, SgsClosure):
            raise Invalid("an SgsClosure is required")
        self.closure = closure

    def collide(self, f: np.ndarray) -> np.ndarray:
        if self.closure.k == 0.0:
            return super().collide(f)                       # the frozen code path, bitwise
        rho = f.sum(axis=1)
        u = (f @ self.c + 0.5 * self.force[None, :]) / rho[:, None]
        feq, phi = self.eq.parts(rho, u, self.force)
        A = self.eq.stress(f, phi, rho, u)
        omega_l = self.closure.rate(A, rho, self.omega)
        return feq + (1.0 - omega_l)[:, None] * self.eq.non_equilibrium(A, u) + 0.5 * phi

    def local_rate(self, f: np.ndarray) -> np.ndarray:
        """omega_l of a (pre-collision) state; for diagnostics: nu_t = (1/omega_l - 1/omega0)/3."""
        rho = f.sum(axis=1)
        u = (f @ self.c + 0.5 * self.force[None, :]) / rho[:, None]
        _, phi = self.eq.parts(rho, u, self.force)
        A = self.eq.stress(f, phi, rho, u)
        if self.closure.k == 0.0:
            return np.full(len(rho), self.omega)
        return self.closure.rate(A, rho, self.omega)
