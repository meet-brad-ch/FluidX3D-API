"""V2 rough law (E2): the facetwise nonlinear friction K_a = K0 + 3 C |u_s,a| in the M1 reference, CPU and GPU. REFERENCE ONLY.

Released #2227 / #2245 (control proposal V2_ROUGH_LAW_CHANNEL_CONTROL_E2_2026-11-10.md, 5871ecc9; law RW
V2_ROUGH_WALL_M1_RECONCILIATION_E2_2026-11-10.md). No engine patch.

ONE change to the accepted SGS candidate (sgs_model.SgsWallReferenceModel27 on the CPU, sgs_gpu_map.SgsGpuMap27 on the GPU):
the scalar friction K of the return term (t89_model.WallReferenceModel27.transport line 108 and t93_gpu_map.GpuMap27.transport
line 154: "- self.friction * rho_r * (self.phi * u_r).sum(axis=1)") becomes the per-row value K_a of the row's facet,

    K_a = K0 + 3 C |u_s,a|,   u_s,a = P_a S u_p   (the projected sample velocity of the same transport call)

The transport code is not copied. transport(p) computes u_s with the frozen expressions (fields, S v, projection), sets
self.friction to the per-row array K_a[facet_of_row] for that one call (the return term broadcasts it), and restores the
scalar afterwards. The law is a run parameter (no container change). |u_s| = 0 gives K0; no division.
The temporary per-row friction is safe only because each model instance belongs to one process (the parallel route builds
one model per spawn worker): do not share one instance among concurrent threads (#2250).
Off path: C = 0 and K0 = the container K give the same values as the accepted map (the per-row array holds K in every row).
C from z0: C = (kappa / ln(d / z0))^2 with kappa = 0.4 and d = 0.5 (the sample distance); d <= z0 is refused (fail closed).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SGS_E2 = HERE.parents[1] / "sgs_m1" / "e2"
sys.path.insert(0, str(SGS_E2))
import sgs_model as SGS  # noqa: E402  (pinned SGS candidate; puts task87/task89 on the path)

Invalid = SGS.Invalid


@dataclass(frozen=True)
class RoughLaw:
    """K_a = K0 + 3 C |u_s,a| with one C and one K0 for all facets."""

    C: float
    K0: float = 0.0
    KAPPA = 0.4
    SAMPLE_DISTANCE = 0.5

    def __post_init__(self) -> None:
        if not (math.isfinite(self.C) and self.C >= 0.0 and math.isfinite(self.K0) and self.K0 >= 0.0):
            raise Invalid("the rough law needs finite C >= 0 and K0 >= 0")

    @staticmethod
    def from_z0(z0: float, K0: float = 0.0) -> "RoughLaw":
        """C = (kappa / ln(d / z0))^2. Refuses d <= z0 (the unphysical branch) and a z0 that is not finite and positive."""
        d = RoughLaw.SAMPLE_DISTANCE
        if not (math.isfinite(z0) and z0 > 0.0):
            raise Invalid("z0 must be finite and > 0")
        if not d > z0:
            raise Invalid(f"refused: sample distance d = {d} <= z0 = {z0} (the log law needs the sample above the roughness)")
        return RoughLaw((RoughLaw.KAPPA / math.log(d / z0)) ** 2, K0)

    @staticmethod
    def channel_control(beta: float, a: float, H: float) -> "RoughLaw":
        """The registered control's C = 2 beta^2 / (a H) (control proposal Section 2), K0 = 0."""
        return RoughLaw(2.0 * beta * beta / (a * H), 0.0)

    @property
    def z0(self) -> float:
        """The roughness length implied by C: z0 = d exp(-kappa / sqrt(C)). C = 0 (the off law) gives 0.0, the limit of z0 as
        C -> 0+ (#2250). from_z0 still requires z0 > 0; the off law is given by C = 0 explicitly."""
        return self.SAMPLE_DISTANCE * math.exp(-self.KAPPA / math.sqrt(self.C)) if self.C > 0.0 else 0.0

    def facet_K(self, speed, xp=np):
        """K_a per facet from |u_s,a| (numpy or cupy array)."""
        return self.K0 + 3.0 * self.C * speed


class RoughSgsModel27(SGS.SgsWallReferenceModel27):
    """SgsWallReferenceModel27 with the facetwise rough friction (CPU)."""

    def __init__(self, container: Path, force: np.ndarray, closure: SGS.SgsClosure, law: RoughLaw) -> None:
        super().__init__(container, force, closure)
        if not isinstance(law, RoughLaw):
            raise Invalid("a RoughLaw is required")
        self.law = law
        self.last_facet_K = None

    def sample_speed(self, p: np.ndarray) -> np.ndarray:
        """|u_s,a| per facet with the frozen transport expressions (t89 lines 92, 101-102)."""
        _, v = self.fields(p)
        u_s_full = self.S @ v
        u_s = u_s_full - self.normal * (u_s_full * self.normal).sum(axis=1)[:, None]
        return np.sqrt((u_s * u_s).sum(axis=1))

    def transport(self, p: np.ndarray) -> tuple:
        facet_K = self.law.facet_K(self.sample_speed(p))
        scalar = self.friction
        self.friction = facet_K[self.facet_of_row]
        try:
            result = super().transport(p)
        finally:
            self.friction = scalar
        self.last_facet_K = facet_K
        return result


CUDA12_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"


def pin_cuda12() -> None:
    """Set CUDA_PATH of THIS process to the CUDA 12 toolkit before cupy is imported (#2248). The global setting points to
    v13.3, whose headers the cupy-cuda12 NVRTC cannot parse (JIT of a CUB reduction failed, 2026-11-11). Fails closed if
    the toolkit is missing or cupy is already imported with another path."""
    if not Path(CUDA12_PATH, "include", "cuda.h").is_file():
        raise Invalid(f"the CUDA 12 toolkit is missing: {CUDA12_PATH}")
    if "cupy" in sys.modules:
        if os.environ.get("CUDA_PATH") != CUDA12_PATH:
            raise Invalid("cupy is already imported with CUDA_PATH != the CUDA 12 toolkit; pin it before the first import")
        return
    os.environ["CUDA_PATH"] = CUDA12_PATH


def gpu_map_class():
    """RoughSgsGpuMap27 (imported on demand: the GPU map needs cupy, the envs/gpu_map interpreter). Pins CUDA_PATH first."""
    pin_cuda12()
    import sgs_gpu_map as GPU  # noqa: PLC0415

    class RoughSgsGpuMap27(GPU.SgsGpuMap27):
        """SgsGpuMap27 with the facetwise rough friction (GPU)."""

        def __init__(self, model: RoughSgsModel27) -> None:
            if not isinstance(model, RoughSgsModel27):
                raise Invalid("the rough GPU map mirrors RoughSgsModel27 only")
            super().__init__(model)
            self.law = model.law
            self.last_facet_K = None

        def sample_speed(self, p):
            cp = GPU.cp
            _, v = self.fields(p)
            u_s_full = self.S @ v
            u_s = u_s_full - self.normal * (u_s_full * self.normal).sum(axis=1)[:, None]
            return cp.sqrt((u_s * u_s).sum(axis=1))

        def transport(self, p):
            facet_K = self.law.facet_K(self.sample_speed(p))
            scalar = self.friction
            self.friction = facet_K[self.facet_of_row]
            try:
                result = super().transport(p)
            finally:
                self.friction = scalar
            self.last_facet_K = facet_K
            return result

    return RoughSgsGpuMap27
