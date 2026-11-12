"""V2-SGS-01 (E2): the pending-terrain oracle with the accepted SGS collision, for the native parity (#2229).

SgsTerrainModel27 = integration/task95/t95_model.TerrainModel27 (frozen: augmented pending transport, step, ledgers) with the
SAME collision override as sgs_model.SgsWallReferenceModel27 (pinned, unchanged). Nothing is copied: the class takes both
bases, and the method resolution order is

    SgsTerrainModel27 -> SgsWallReferenceModel27 -> TerrainModel27 -> WallReferenceModel27 -> Model27 -> object,

so collide and local_rate come from SgsWallReferenceModel27, everything else from TerrainModel27. With k == 0,
SgsWallReferenceModel27.collide calls super().collide, which resolves past TerrainModel27 (no collide of its own) to the
frozen Model27.collide: the k = 0 path is the unchanged terrain oracle. SgsWallReferenceModel27.__init__(container, force,
closure) calls super().__init__(container, force), which is TerrainModel27.__init__ with its default friction (the
container's K). No transport, wall pass or ledger is changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "integration" / "task95"))
import sgs_model as SGS  # noqa: E402  (pinned candidate)
import t95_model as T95  # noqa: E402  (frozen terrain oracle)


class SgsTerrainModel27(SGS.SgsWallReferenceModel27, T95.TerrainModel27):
    """TerrainModel27 with the SGS collision of SgsWallReferenceModel27 (see the module text for the resolution order)."""

    def __init__(self, container: Path, force: np.ndarray, closure: SGS.SgsClosure) -> None:
        super().__init__(container, force, closure)


EXPECTED_ORDER = ("SgsTerrainModel27", "SgsWallReferenceModel27", "TerrainModel27", "WallReferenceModel27", "Model27", "object")
