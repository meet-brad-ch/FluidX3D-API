"""Task 95 (E2): the frozen task-90 model (t89_model.WallReferenceModel27) with the pending facet-to-facet transport W.
Derivation: integration/task95/TASK95_AUGMENTED_TRANSPORT_E2_2026-11-10.md (Sections 2-6).

The state is (f, s): f pre-collision population densities (cells x 27), s pending AMOUNTS per row (the amount that moves with
-c_r towards facet a of row r, produced in the previous update). One update:
    p = collide(f)                                    (unchanged, Model27.collide)
    I = G p + G_arm L p + s                           (NEW: + s, the pending inventory is consumed once)
    Q = the t89 return with the one-per-facet delta over I (unchanged statements)
    f' = D p + D_arm L p + R Q + centred scatter      (unchanged)
    s' = W (Q / v)                                    (NEW)
With W = 0 and s = 0 the statements are those of WallReferenceModel27.transport; f' is bit-identical (check C7).
Frozen files are not changed. Model27.__init__ refuses W != 0; TerrainModel27.__init__ repeats its body without that refusal
and without the channel normal groups (Model27._groups is the channel's q_c = 3x + z rule; no reduced solve on terrain).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task87"))
sys.path.insert(0, str(HERE.parent / "task89"))
import t89_model as M89  # noqa: E402  (frozen)

M27 = M89.M27
Invalid = M27.Invalid


@dataclass
class WallPass95:
    gathered: np.ndarray          # rows: G p + G_arm L p (fluid part of the incoming amount)
    pending_in: np.ndarray        # rows: s (consumed)
    incoming: np.ndarray          # rows: I = gathered + s
    outgoing: np.ndarray          # rows: Q
    delta: np.ndarray             # facets
    impulse: np.ndarray           # facets x 3: J = sum_r c_r (Q_r + I_r) (on the fluid)
    rho_s: np.ndarray             # facets: sampled density (friction)
    u_s: np.ndarray               # facets x 3: sampled tangential velocity (friction)
    rho_w: np.ndarray             # facets: wall-extrapolated density (even reference)
    u_w: np.ndarray               # facets x 3: wall-extrapolated tangential velocity (even reference)


class TerrainModel27(M89.WallReferenceModel27):
    """WallReferenceModel27 with the augmented state (f, s)."""

    def __init__(self, container: Path, force: np.ndarray, friction: float | None = None) -> None:  # noqa: PLR0915
        # ---- body of Model27.__init__ without the W refusal and without _groups ------------------------------------------------
        t39 = M27.Readers.t39()
        a = t39.arrays(t39.read_sections(Path(container).read_bytes()))
        self.lattice = M27.Lattice27.make()
        self.c, self.w, self.q = self.lattice.c, self.lattice.w, 27
        self.eq = M27.Equilibrium(self.lattice)
        self.force = np.asarray(force, dtype=np.float64)
        self.nu, self.omega, container_friction = (float(a["scalars"][k]) for k in range(3))
        self.friction = container_friction if friction is None else float(friction)
        self.cells = len(a["volume"])
        self.states = self.q * self.cells
        self.volume = np.asarray(a["volume"], dtype=np.float64)
        if len(a["f0"]) != self.states:
            raise Invalid(f"f0 has {len(a['f0'])} values, not cells x 27: the container is not a D3Q27 container")
        self.initial = np.asarray(a["f0"], dtype=np.float64).reshape(self.cells, self.q)
        self.wall_cells = a["gradient_cells"].astype(np.int64)
        self.is_wall = np.zeros(self.cells, dtype=bool)
        self.is_wall[self.wall_cells] = True
        self._direct(a)
        self._gradient(a)
        self._rows(a)
        self._scatter(a)
        # ---- NEW: the pending map and the initial pending amounts ----------------------------------------------------------------
        shape = tuple(int(v) for v in a["W_shape"])
        if shape != (self.rows, self.rows):
            raise Invalid(f"W has shape {shape}, not rows x rows = {(self.rows, self.rows)}")
        self.W = csr_matrix((np.asarray(a["W_weights"], dtype=np.float64), a["W_indices"].astype(np.int64), a["W_offsets"].astype(np.int64)),
                            shape=shape)
        Wc = self.W.tocoo()
        if Wc.nnz and not np.array_equal(self.row_direction[Wc.row], self.lattice.opposite[self.row_direction[Wc.col]]):
            raise Invalid("a W entry does not join a receiver row with the opposite direction of its producer row")
        if Wc.nnz and (Wc.data <= 0.0).any():
            raise Invalid("a W volume is not positive")
        self.initial_pending = np.asarray(a["s0"], dtype=np.float64)
        if self.initial_pending.shape != (self.rows,) or not np.isfinite(self.initial_pending).all():
            raise Invalid("s0 is not one finite amount per row")
        self.area = np.asarray(a["area"], dtype=np.float64)

    # ---- the update ---------------------------------------------------------------------------------------------------------------
    def transport(self, p: np.ndarray) -> tuple:
        raise Invalid("TerrainModel27 has the augmented state: use transport_augmented(p, s) or step(f, s)")

    def transport_augmented(self, p: np.ndarray, s: np.ndarray) -> tuple:
        """WallReferenceModel27.transport with I = gathered + s and s' = W (Q / v). Every other statement is the t89 one."""
        q = self.q
        if s.shape != (self.rows,) or not np.isfinite(s).all():
            raise Invalid("the pending state is not one finite amount per row")
        rho, v = self.fields(p)
        grad_p = [g @ p for g in self.gradient]
        grad_rho = np.stack([g @ rho for g in self.gradient], axis=1)
        grad_v = np.stack([g @ v for g in self.gradient], axis=2)
        flat = p.reshape(-1)
        direct = self.D @ flat
        direct_increment = sum(self.D_arm[axis] @ grad_p[axis].reshape(-1) for axis in range(3))
        gathered = self.G @ flat + sum(self.G_arm[axis] @ grad_p[axis].reshape(-1) for axis in range(3))
        incoming = gathered + s                                                           # NEW: the pending arrivals
        rho_s = self.S @ rho
        u_s_full = self.S @ v
        u_s = u_s_full - self.normal * (u_s_full * self.normal).sum(axis=1)[:, None]
        rho_w, u_w = self.wall_state(rho_s, u_s_full, grad_rho, grad_v)
        rho_wr, u_wr = rho_w[self.facet_of_row], u_w[self.facet_of_row]
        rho_r, u_r = rho_s[self.facet_of_row], u_s[self.facet_of_row]
        cu = (self.c_row * u_wr).sum(axis=1)
        even = 2.0 * self.w_row * rho_wr * (1.0 + 4.5 * cu * cu - 1.5 * (u_wr * u_wr).sum(axis=1))
        ret = self.prism * even - incoming - self.friction * rho_r * (self.phi * u_r).sum(axis=1)
        delta = np.bincount(self.facet_of_row, weights=incoming - ret, minlength=self.facets) / self.facet_volume
        outgoing = ret + self.prism * delta[self.facet_of_row]
        part = self._outgoing_shape(self.c_row[self.g_row], self.w_row[self.g_row], rho[self.g_cell], v[self.g_cell], self.g_arm,
                                    grad_rho[self.g_cell], grad_v[self.g_cell])
        row_mean = np.bincount(self.g_row, weights=self.g_weight * part, minlength=self.rows) / self.g_total
        entry = self._outgoing_shape(self.c[self.r_direction], self.w[self.r_direction], rho[self.r_cell], v[self.r_cell], self.r_arm,
                                     grad_rho[self.r_cell], grad_v[self.r_cell])
        scatter = np.bincount(self.r_state, weights=self.r_part * (entry - row_mean[self.r_row]), minlength=self.states)
        returned = self.R @ outgoing + scatter
        new = (direct + direct_increment + returned).reshape(self.cells, q)
        pending_next = self.W @ (outgoing / self.prism)                                    # NEW: s' = W (Q / v)
        if not (np.isfinite(new).all() and np.isfinite(pending_next).all()):
            raise Invalid("a new population or pending amount is not finite")
        impulse_rows = self.c_row * (outgoing + incoming)[:, None]
        impulse = np.stack([np.bincount(self.facet_of_row, weights=impulse_rows[:, k], minlength=self.facets) for k in range(3)], axis=1)
        return new, pending_next, WallPass95(gathered, s, incoming, outgoing, delta, impulse, rho_s, u_s, rho_w, u_w)

    def step(self, f: np.ndarray, s: np.ndarray) -> tuple:
        new, pending_next, _ = self.transport_augmented(self.collide(f), s)
        return new, pending_next

    # ---- inventories and the ledgers (derivation Sections 4-5) --------------------------------------------------------------------
    def augmented_mass(self, f: np.ndarray, s: np.ndarray) -> float:
        return float(self.volume @ f.sum(axis=1)) + float(s.sum())

    def augmented_momentum(self, f: np.ndarray, s: np.ndarray) -> np.ndarray:
        """sum V sum c f + sum (-c_r) s_r: a pending amount of row r moves with -c_r."""
        return self.volume @ (f @ self.c) - s @ self.c_row

    def traction_defect(self, wall: WallPass95) -> np.ndarray:
        """Per facet: P J + K rho_s (A/3) u_s (zero for a complete planar row set, Section 5), divided by A."""
        n = self.normal
        J = wall.impulse
        tangential = J - n * (J * n).sum(axis=1)[:, None]
        expected = -self.friction * wall.rho_s[:, None] * (self.area / 3.0)[:, None] * wall.u_s
        return np.abs(tangential - expected).max(axis=1) / self.area
