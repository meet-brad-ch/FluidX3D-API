"""V2 rough law (E2, #2326): the PENDING-terrain reference with the facetwise rough friction. Oracle plumbing for the native
qualification (task-99 style R3 model) and the later S8 flow check. NOT a new model; a new file (no pinned file changed).

RoughSgsTerrainModel27 = sgs_terrain_model.SgsTerrainModel27 (the accepted SGS collision + the task-95 augmented pending
transport, both unchanged) with ONE change, the same as rough_model.RoughSgsModel27 for the channel: the scalar friction of
the return term (t95_model.TerrainModel27.transport_augmented line 115, "- self.friction * rho_r * (self.phi * u_r).sum(axis=1)")
is replaced, for that one call, by the per-row K_a[facet_of_row], K_a = FacetLaw.facet_K(|u_s,a|), where u_s,a is the projected
sample of the SAME post-collision state that the transport uses (the frozen expressions fields, S v, projection; the sample is
computed by rough_model.RoughSgsModel27.sample_speed, called on this model). The scalar is restored after the call.
Method resolution order:
  RoughSgsTerrainModel27 -> SgsTerrainModel27 -> SgsWallReferenceModel27 -> TerrainModel27 -> WallReferenceModel27 -> Model27
The law: an explicit FacetLaw, or the container section facet_friction (K0_a, C_a), or, for a container without it, the legacy
(K, 0) on every facet (K_a = K exactly: the accepted SGS terrain model). The per-row array is safe only because each instance
belongs to one process (#2250).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "sgs_m1" / "e2"))
sys.path.insert(0, str(HERE.parents[2] / "integration" / "task39"))
import facet_law as FL  # noqa: E402
import rough_model as RM  # noqa: E402  (pinned; only sample_speed is called)
import sgs_terrain_model as STM  # noqa: E402  (accepted SGS pending-terrain oracle)
import t39_lib as T39  # noqa: E402  (container section reader)

Invalid = FL.Invalid


def container_law(container: Path, facets: int, K: float) -> FL.FacetLaw:
    """The facet_friction section of the container, or the legacy (K, 0) on every facet when it is absent."""
    sections = {name: (dtype, payload) for name, dtype, payload in T39.read_sections(Path(container).read_bytes())}
    if "facet_friction" not in sections:
        return FL.FacetLaw.uniform(facets, 0.0, K)
    dtype, payload = sections["facet_friction"]
    if dtype != 3:
        raise Invalid("facet_friction is not float64")
    return FL.FacetLaw.from_section(np.frombuffer(payload, dtype="<f8").reshape(-1, 2), facets)


class RoughSgsTerrainModel27(STM.SgsTerrainModel27):
    """SgsTerrainModel27 with the facetwise rough friction in transport_augmented."""

    def __init__(self, container: Path, force: np.ndarray, closure, law: FL.FacetLaw | None = None) -> None:
        super().__init__(container, force, closure)
        if law is None:
            law = container_law(container, self.facets, float(self.friction))
        if not isinstance(law, FL.FacetLaw) or law.facets != self.facets:
            raise Invalid("a FacetLaw with one entry per facet of the container is required")
        self.law = law
        self.last_facet_K = None

    def sample_speed(self, p: np.ndarray) -> np.ndarray:
        return RM.RoughSgsModel27.sample_speed(self, p)

    def transport_augmented(self, p: np.ndarray, s: np.ndarray) -> tuple:
        facet_K = self.law.facet_K(self.sample_speed(p))
        scalar = self.friction
        self.friction = facet_K[self.facet_of_row]
        try:
            result = super().transport_augmented(p, s)
        finally:
            self.friction = scalar
        self.last_facet_K = facet_K
        return result

    def traction_defect(self, wall) -> np.ndarray:
        """t95 traction_defect with K_a RECOMPUTED from this wall pass (#2327): K_a = law.facet_K(|wall.u_s|), wall.u_s being
        the projected sample of that pass (the same expression as sample_speed). The base uses the scalar K, which is the
        restored legacy value after the call, so it would report the old law."""
        K_a = self.law.facet_K(np.sqrt((wall.u_s * wall.u_s).sum(axis=1)))
        n = self.normal
        J = wall.impulse
        tangential = J - n * (J * n).sum(axis=1)[:, None]
        expected = -K_a[:, None] * wall.rho_s[:, None] * (self.area / 3.0)[:, None] * wall.u_s
        return np.abs(tangential - expected).max(axis=1) / self.area


EXPECTED_ORDER = ("RoughSgsTerrainModel27", "SgsTerrainModel27", "SgsWallReferenceModel27", "TerrainModel27",
                  "WallReferenceModel27", "Model27", "object")
