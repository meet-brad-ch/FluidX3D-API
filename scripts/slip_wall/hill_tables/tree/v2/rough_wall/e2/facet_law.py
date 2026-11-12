"""V2 rough law (E2, #2305): the FACETWISE law data adapter for the reference. Support for the accepted port plan
(V2_ROUGH_ENGINE_PORT_PLAN_E2_2026-11-11.md, f736bdec + 28764899). NOT a new law: K_a = K0_a + 3 C_a |u_s,a| as in
rough_model.py (unchanged, pinned by V2-RW-01). No engine patch.

FacetLaw holds one (K0_a, C_a) pair per facet, in the container facet order:
  - validated: lengths equal to the facet count, every value finite and >= 0 (fail closed); READ-ONLY after validation;
  - C_a == 0: K_a = K0_a EXACTLY (a masked copy; no arithmetic on those facets), the legacy constant-K branch (#2273 item 1);
  - explicit device conversion: on a cupy speed array, the per-facet arrays are converted once with cupy.asarray and
    cached (#2273 item 2); the CPU path uses the numpy arrays;
  - ownership from the geometry: from_facet_keys gives terrain facets ("T", ...) the terrain pair and lid facets ("L", ...)
    the pair (0, 0) (#2271); no normal-direction heuristic;
  - section form: as_section() is the facets x 2 float64 array of the planned container section facet_friction; from_section
    reads it back (the exact binary64 values, e.g. the registered C, no z0 round trip; #2273 item 3).
FacetRoughSgsModel27 is RoughSgsModel27 with a FacetLaw in place of the scalar RoughLaw. The transport is the inherited
one: it calls law.facet_K(speed) per facet and broadcasts K_a[facet_of_row]. The GPU map (rough_model.gpu_map_class())
accepts it unchanged, because it is a RoughSgsModel27.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rough_model as RM  # noqa: E402  (pinned; imported, not modified)

Invalid = RM.Invalid


class FacetLaw:
    """Per-facet (K0_a, C_a) with the exact C_a == 0 branch."""

    def __init__(self, K0: np.ndarray, C: np.ndarray, facets: int) -> None:
        K0 = np.asarray(K0, dtype=np.float64).copy()
        C = np.asarray(C, dtype=np.float64).copy()
        if K0.shape != (facets,) or C.shape != (facets,):
            raise Invalid(f"facet law arrays must have length {facets} (K0 {K0.shape}, C {C.shape})")
        if not (np.isfinite(K0).all() and np.isfinite(C).all()):
            raise Invalid("facet law: a K0 or C value is not finite")
        if (K0 < 0.0).any() or (C < 0.0).any():
            raise Invalid("facet law: a K0 or C value is negative")
        rough = C != 0.0
        for array in (K0, C, rough):
            array.setflags(write=False)        # #2307: immutable after validation (the GPU copies cannot fall out of sync)
        self.K0, self.C, self.rough, self.facets = K0, C, rough, facets
        self._device = None

    @staticmethod
    def uniform(facets: int, C: float, K0: float = 0.0) -> "FacetLaw":
        """Every facet (K0, C): the uniform channel control maps onto this (RoughLaw.channel_control's C, K0 = 0)."""
        return FacetLaw(np.full(facets, float(K0)), np.full(facets, float(C)), facets)

    @staticmethod
    def kinds_of(facet_keys: list) -> list:
        """The kind of every key ('T' terrain, 'L' lid). Refuses an empty key list, an empty or non-tuple key, and any other
        kind (#2307: the same validation for the law and for facet_kind)."""
        if len(facet_keys) == 0:
            raise Invalid("facet law: no facet keys")
        kinds = []
        for key in facet_keys:
            if not isinstance(key, tuple) or len(key) == 0 or key[0] not in ("T", "L"):
                raise Invalid(f"facet law: the facet key {key!r} is neither terrain ('T', ...) nor lid ('L', ...)")
            kinds.append(key[0])
        return kinds

    @staticmethod
    def from_facet_keys(facet_keys: list, terrain_K0: float, terrain_C: float) -> "FacetLaw":
        """Terrain facets ("T", ...) get (terrain_K0, terrain_C); lid facets ("L", ...) get (0, 0). Other keys are refused."""
        kinds = FacetLaw.kinds_of(facet_keys)
        if not (math.isfinite(terrain_K0) and math.isfinite(terrain_C)):
            raise Invalid("facet law: the terrain K0 or C is not finite")
        terrain = np.array([kind == "T" for kind in kinds])
        return FacetLaw(np.where(terrain, terrain_K0, 0.0), np.where(terrain, terrain_C, 0.0), len(facet_keys))

    @staticmethod
    def from_section(values: np.ndarray, facets: int) -> "FacetLaw":
        """The planned container section facet_friction: facets x 2 = (K0_a, C_a), exact binary64."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (facets, 2):
            raise Invalid(f"facet_friction must be {facets} x 2, it is {values.shape}")
        return FacetLaw(values[:, 0], values[:, 1], facets)

    def as_section(self) -> np.ndarray:
        return np.stack([self.K0, self.C], axis=1)

    def facet_kind(self, facet_keys: list) -> np.ndarray:
        """The planned section facet_kind (u32): 0 terrain, 1 lid (metadata; the law does not read it). The keys are
        validated as in from_facet_keys, and their count must equal this law's facet count (#2307)."""
        kinds = FacetLaw.kinds_of(facet_keys)
        if len(kinds) != self.facets:
            raise Invalid(f"facet_kind: {len(kinds)} keys for a law of {self.facets} facets")
        return np.array([0 if kind == "T" else 1 for kind in kinds], dtype=np.uint32)

    def _arrays_for(self, speed):
        cupy = sys.modules.get("cupy")
        if cupy is not None and isinstance(speed, cupy.ndarray):
            if self._device is None:
                self._device = (cupy.asarray(self.K0), cupy.asarray(self.C), cupy.asarray(self.rough))
            return cupy, self._device
        return np, (self.K0, self.C, self.rough)

    def facet_K(self, speed, xp=np):
        """K_a = K0_a + 3 C_a |u_s,a| on the rough facets; K_a = K0_a (a copy) on the C_a == 0 facets.
        Masked assignment (#2307): the C_a == 0 facets are copied, not computed, so no arithmetic touches them. A non-finite
        speed on a rough facet gives a non-finite K_a there, so the transport's fail-closed state checks still see it."""
        lib, (K0, C, rough) = self._arrays_for(speed)
        if speed.shape != K0.shape:
            raise Invalid(f"facet law: speed has shape {speed.shape}, expected {K0.shape}")
        K = K0.copy()
        K[rough] = K0[rough] + 3.0 * C[rough] * speed[rough]
        return K


class FacetRoughSgsModel27(RM.RoughSgsModel27):
    """RoughSgsModel27 with a FacetLaw. The base constructor requires a scalar RoughLaw, so this constructor builds the SGS
    model and sets the law itself; everything else (sample_speed, transport, last_facet_K) is inherited unchanged."""

    def __init__(self, container: Path, force: np.ndarray, closure, law: FacetLaw) -> None:
        RM.SGS.SgsWallReferenceModel27.__init__(self, container, force, closure)
        if not isinstance(law, FacetLaw) or law.facets != self.facets:
            raise Invalid("a FacetLaw with one entry per facet of the container is required")
        self.law = law
        self.last_facet_K = None
