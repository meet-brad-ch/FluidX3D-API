"""Task 119: reference wrapper of the whole-prism redistribution (MATHEMATICAL_MODEL.md section 15.11). Reference only; no engine.

STATUS: written, NOT executed. Nothing here runs a trajectory. The baseline transport is the pinned task-115 ForcedClosedModel
(driven_hill.py, unchanged): this file SUBCLASSES it, calls its transport_augmented unchanged, and adds the section-15.11
increments to the returned populations and pending amounts. Q, the incoming ledger, the friction law, the projector, the
collision and the force convention are therefore untouched.

Section 15.11 contract, per producer row r with a stored W entry (all other rows are not touched):
  fluid parts j (G entries): volume h_j, offset delta_j = C_j - C_full, first-order Taylor slope b_j of the existing shape E_j;
  owned parts b (stored W entries): volume w_b (the stored W weight), first moment m_b (companion table, extractor output);
  H = sum h_j,  W = sum w_b,  S = H + W,  a = sum h_j b_j / H,  B0 = -sum h_j (b_j . delta_j) / H,  L = W B0 + a . sum_b m_b
  fluid amount added to part j     : -h_j L / S           (divided by V_cell when added to the population, direction c_r)
  pending amount added to dest. b  : w_b B0 + a . m_b - w_b L / S   (added to the actual destination row)
H > 0 is required (a row with W and H = 0 is a model-domain blocker and raises DomainBlocker at construction).
The increments of one row sum to zero for any slopes and any stored moments. Shapes use the post-collision fields of the same
update; the new pending is used on the next update as before.

Classes: CompanionTables (load + verify the extractor output), R1Increments (the vectorized increments), R1CorrectedModel
(pinned model + increments), R1Components (component checks that need no trajectory).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[3]
sys.path.insert(0, str(WALL / "v2" / "hill_demo" / "e2" / "task115"))
import driven_hill as DH  # noqa: E402  (pinned baseline: ForcedClosedModel, Run)

Invalid = DH.Invalid


class DomainBlocker(Invalid):
    """A row with W has H = 0: the section 15.11 reconstruction is undefined there. Reported, never a silent fallback."""


class CompanionTables:
    """The extractor output (NPZ), verified against the model's own container tables before any use."""

    def __init__(self, npz: Path, npz_sha256: str, model) -> None:
        raw = Path(npz).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != npz_sha256:
            raise Invalid(f"companion NPZ sha256 {got[:16]} is not the registered {npz_sha256[:16]}")
        z = np.load(npz)
        self.prod_rows = np.asarray(z["prod_rows"], dtype=np.int64)
        self.w_recv = np.asarray(z["w_recv_row"], dtype=np.int64)
        self.w_prod = np.asarray(z["w_prod_row"], dtype=np.int64)
        self.w_stored = np.asarray(z["w_volume_stored"], dtype=np.float64)
        self.moment = np.asarray(z["w_moment"], dtype=np.float64)
        self.g_index = np.asarray(z["g_entry_index"], dtype=np.int64)
        self.g_delta = np.asarray(z["g_delta"], dtype=np.float64)
        self.verify(model)

    def verify(self, m) -> None:
        W = csr_matrix(m.W)
        recv = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        if not (np.array_equal(recv, self.w_recv) and np.array_equal(W.indices.astype(np.int64), self.w_prod)):
            raise Invalid("companion W structure (receiver, producer) differs from the model W table")
        if not np.array_equal(W.data, self.w_stored):
            raise Invalid("companion stored W weights differ from the model W table")
        if not (np.isfinite(self.moment).all() and np.isfinite(self.g_delta).all()):
            raise Invalid("companion moments or offsets are not finite")
        nnz = W.nnz
        if self.moment.shape != (nnz, 3) or self.w_stored.shape != (nnz,) or self.w_recv.shape != (nnz,) or self.w_prod.shape != (nnz,):
            raise Invalid("a companion W-entry array has the wrong shape")
        expected = np.flatnonzero(np.isin(m.g_row, self.prod_rows))
        if len(np.unique(self.g_index)) != len(self.g_index):
            raise Invalid("the companion G entry list has duplicates")
        if not np.array_equal(self.g_index, expected):
            raise Invalid(f"the companion G entry list ({len(self.g_index)}) is not exactly all G entries of all W-present rows ({len(expected)})")
        if self.g_delta.shape != (len(expected), 3):
            raise Invalid("the companion offsets have the wrong shape")
        if not np.array_equal(np.unique(self.w_prod), self.prod_rows):
            raise Invalid("producer rows with W differ between the W table and the companion")


class R1Increments:
    """The vectorized section 15.11 increments for a given post-collision state p (state arrays are not modified)."""

    def __init__(self, m, companion: CompanionTables) -> None:
        self.m, self.cp = m, companion
        self.nrw = len(companion.prod_rows)
        compact = np.full(m.rows, -1, dtype=np.int64)
        compact[companion.prod_rows] = np.arange(self.nrw)
        self.g_idx = companion.g_index
        self.g_c = compact[m.g_row[self.g_idx]]
        self.g_cell = m.g_cell[self.g_idx]
        self.g_dir = m.row_direction[m.g_row[self.g_idx]]
        self.h = m.g_weight[self.g_idx]
        self.last: dict = {}
        self.last_arrays: dict = {}
        self.delta = companion.g_delta
        self.w_c = compact[companion.w_prod]
        self.w = companion.w_stored
        self.wm = companion.moment
        H = np.bincount(self.g_c, weights=self.h, minlength=self.nrw)
        if not (H > 0.0).all():
            raise DomainBlocker(f"{int((H <= 0.0).sum())} rows with W have H <= 0")
        if not (np.isfinite(self.h).all() and (self.h > 0.0).all() and (self.w > 0.0).all()):
            raise Invalid("a selected fluid or owned volume is not finite and positive")

    def fixture_mask(self) -> np.ndarray:
        """Compact-row mask of the task-118 fixture: producer rows with an R entry into X in their own direction, that carry W."""
        m = self.m
        block = m.R[DH.X_CELL * m.q:(DH.X_CELL + 1) * m.q].tocoo()
        rows = {int(r) for st, r in zip(block.row.tolist(), block.col.tolist()) if int(st) == int(m.row_direction[r])}
        mask = np.zeros(self.nrw, dtype=bool)
        for n, r in enumerate(self.cp.prod_rows.tolist()):
            mask[n] = r in rows
        return mask

    def stored_part(self) -> np.ndarray:
        """h from the STORED R table per G entry: R[(cell, d), row] * v_row * V_cell, matched entry by entry (fail closed)."""
        m = self.m
        key_r = m.r_row * m.states + m.r_state
        order = np.argsort(key_r)
        wanted = m.g_row[self.g_idx] * m.states + self.g_cell * m.q + self.g_dir
        pos = np.minimum(np.searchsorted(key_r[order], wanted), len(order) - 1)
        match = order[pos]
        if not np.array_equal(key_r[match], wanted):
            raise Invalid("a modified G entry has no stored R entry")
        return m.R.data[match] * m.prism[m.g_row[self.g_idx]] * m.volume[self.g_cell]

    def slopes(self, rho, v, grad_rho, grad_v) -> np.ndarray:
        """b_j: the spatial gradient of the existing first-order Taylor shape E_j (t87_model.py 277-283) at the G entries."""
        m = self.m
        c = m.c[self.g_dir]
        w = m.w[self.g_dir]
        u = v[self.g_cell]
        cu = (c * u).sum(axis=1)
        uu = (u * u).sum(axis=1)
        B = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu + 4.5 * (cu ** 3 - cu * uu)
        slope = m.eq.slope(c, u)
        return w[:, None] * (B[:, None] * grad_rho[self.g_cell] + rho[self.g_cell][:, None] * np.einsum("nc,nca->na", slope, grad_v[self.g_cell]))

    def row_terms(self, b: np.ndarray) -> dict:
        """H, W, S, a, B0, L per compact row for given slopes b (n_g, 3)."""
        n = self.nrw
        H = np.bincount(self.g_c, weights=self.h, minlength=n)
        Wr = np.bincount(self.w_c, weights=self.w, minlength=n)
        a = np.stack([np.bincount(self.g_c, weights=self.h * b[:, k], minlength=n) for k in range(3)], axis=1) / H[:, None]
        B0 = -np.bincount(self.g_c, weights=self.h * (b * self.delta).sum(axis=1), minlength=n) / H
        sum_m = np.stack([np.bincount(self.w_c, weights=self.wm[:, k], minlength=n) for k in range(3)], axis=1)
        S = H + Wr
        L = Wr * B0 + (a * sum_m).sum(axis=1)
        return {"H": H, "W": Wr, "S": S, "a": a, "B0": B0, "L": L, "sum_m": sum_m}

    def amounts(self, b: np.ndarray) -> tuple:
        """(fluid amount per G entry, pending amount per W entry) for the slopes b."""
        t = self.row_terms(b)
        fluid = -self.h * t["L"][self.g_c] / t["S"][self.g_c]
        pending = self.w * t["B0"][self.w_c] + (t["a"][self.w_c] * self.wm).sum(axis=1) - self.w * t["L"][self.w_c] / t["S"][self.w_c]
        return fluid, pending, t

    def from_state(self, p: np.ndarray) -> np.ndarray:
        """Slopes from the post-collision populations p of one update (same field formulas as the pinned transport)."""
        m = self.m
        rho, v = m.fields(p)
        grad_rho = np.stack([g @ rho for g in m.gradient], axis=1)
        grad_v = np.stack([g @ v for g in m.gradient], axis=2)
        return self.slopes(rho, v, grad_rho, grad_v)

    def apply(self, new: np.ndarray, s_next: np.ndarray, p: np.ndarray) -> tuple:
        """The corrected (new, s_next): the increments added to copies. Fluid amounts are divided by V_cell."""
        m = self.m
        fluid, pending, terms = self.amounts(self.from_state(p))
        states = self.g_cell * m.q + self.g_dir
        increment = np.bincount(states, weights=fluid / m.volume[self.g_cell], minlength=new.size).reshape(new.shape)
        flat = new + increment
        s_new = s_next + np.bincount(self.cp.w_recv, weights=pending, minlength=m.rows)
        x = DH.X_CELL
        self.last_arrays = {"fluid_amount_per_g_entry": fluid.copy(), "pending_amount_per_w_entry": pending.copy(),
                            "L_per_row": terms["L"], "g_entry_index": self.g_idx, "w_recv_row": self.cp.w_recv,
                            "w_prod_row": self.cp.w_prod}
        self.last = {"rows_modified": self.nrw, "sum_abs_fluid_amount": float(np.abs(fluid).sum()),
                     "max_abs_fluid_amount": float(np.abs(fluid).max()), "sum_abs_pending_amount": float(np.abs(pending).sum()),
                     "max_abs_pending_amount": float(np.abs(pending).max()),
                     "max_abs_population_increment": float(np.abs(increment).max()),
                     "cells_with_increment": int((np.abs(increment).sum(axis=1) > 0).sum()),
                     "X_population_increment_sum": float(increment[x].sum()),
                     "mass_increment": float(m.volume @ increment.sum(axis=1) + (s_new - s_next).sum())}
        return flat, s_new


class R1CorrectedModel(DH.ForcedClosedModel):
    """The pinned closed hill map with the section 15.11 increments added after the unchanged baseline transport."""

    def __init__(self, container: Path, force: np.ndarray) -> None:
        super().__init__(container, force)
        self.container_path = Path(container)

    def attach(self, companion: CompanionTables) -> None:
        self.r1 = R1Increments(self, companion)
        self.r1_active = True

    def transport_augmented(self, p: np.ndarray, s: np.ndarray) -> tuple:
        new, s_next, wall = super().transport_augmented(p, s)
        if not getattr(self, "r1_active", False):
            raise Invalid("the corrected model has no companion attached")
        new, s_next = self.r1.apply(new, s_next, p)
        if not (np.isfinite(new).all() and np.isfinite(s_next).all()):
            raise Invalid("a corrected population or pending amount is not finite")
        return new, s_next, wall


class R1Components:
    """Component checks of the candidate that need no trajectory. Each returns numbers; the protocol sets the gates."""

    def __init__(self, model: R1CorrectedModel) -> None:
        self.m, self.r1 = model, model.r1

    def conservation(self, b: np.ndarray) -> dict:
        """Per-row sum of fluid and pending amounts, and the augmented mass and momentum increments, for slopes b.

        SCALES. The row-sum ratio, the mass gross and the momentum gross use the TERM gross: the sum of the magnitudes of every summed
        term (dot-product components separately, and the two terms of L): fluid parts |h_j L/S|; pending parts |w_b B0| + sum_k |a_k m_bk|
        + |w_b L/S|; and the two terms of L, |W B0| + sum_k |a_k sum_m_k|. The momentum gross weights each term by |c_k| of the row
        direction. The amount gross (sum of |fluid| + |pending|) is kept as a DIAGNOSTIC (*_amount_gross). Provenance: the scale was
        revised BEFORE registration, after an exploratory plumb by E1 (#2891) gave 1.26e-13 under the amount gross for rows with H much
        smaller than W; both ratios are recorded. 1e-13 is an engineering rounding budget, not a rigorous bound."""
        r1, m = self.r1, self.m
        fluid, pending, t = r1.amounts(b)
        n = r1.nrw
        per_row = np.bincount(r1.g_c, weights=fluid, minlength=n) + np.bincount(r1.w_c, weights=pending, minlength=n)
        amount_scale = np.bincount(r1.g_c, weights=np.abs(fluid), minlength=n) + np.bincount(r1.w_c, weights=np.abs(pending), minlength=n)
        a_w, B0_w, L_w, S_w = t["a"][r1.w_c], t["B0"][r1.w_c], t["L"][r1.w_c], t["S"][r1.w_c]
        fluid_terms = np.abs(r1.h * t["L"][r1.g_c] / t["S"][r1.g_c])
        pending_terms = np.abs(r1.w * B0_w) + np.abs(a_w * r1.wm).sum(axis=1) + np.abs(r1.w * L_w / S_w)
        l_terms = np.abs(t["W"] * t["B0"]) + np.abs(t["a"] * t["sum_m"]).sum(axis=1)
        term_scale = (np.bincount(r1.g_c, weights=fluid_terms, minlength=n) + np.bincount(r1.w_c, weights=pending_terms, minlength=n) + l_terms)
        c_prod = m.c_row[r1.cp.prod_rows]
        momentum = per_row[:, None] * c_prod
        mass_global = float(fluid.sum() + pending.sum())
        mom_global = (fluid[:, None] * m.c[r1.g_dir]).sum(axis=0) + (pending[:, None] * c_prod[r1.w_c]).sum(axis=0)
        mom_gross_amount = (np.abs(fluid[:, None] * m.c[r1.g_dir]).sum(axis=0) + np.abs(pending[:, None] * c_prod[r1.w_c]).sum(axis=0))
        mom_gross_term = (term_scale[:, None] * np.abs(c_prod)).sum(axis=0)
        safe_term = np.where(term_scale > 0, term_scale, 1.0)
        safe_amount = np.where(amount_scale > 0, amount_scale, 1.0)
        return {"row_sum_max_abs": float(np.abs(per_row).max()), "row_scale_max": float(term_scale.max()),
                "row_sum_max_rel": float((np.abs(per_row) / safe_term).max()),
                "row_sum_max_rel_amount_gross": float((np.abs(per_row) / safe_amount).max()),
                "row_scale_max_amount_gross": float(amount_scale.max()),
                "mass_global": mass_global, "mass_gross": float(term_scale.sum()), "mass_gross_amount_gross": float(amount_scale.sum()),
                "momentum_global": mom_global.tolist(), "momentum_gross_componentwise": mom_gross_term.tolist(),
                "momentum_gross_componentwise_amount_gross": mom_gross_amount.tolist(), "momentum_rowsum_max": float(np.abs(momentum).max())}

    def affine(self, g: np.ndarray) -> dict:
        """Common affine E with slope g (the same b_j = g for every part): the formula against its closed form and the exact targets."""
        r1 = self.r1
        b = np.tile(np.asarray(g, dtype=np.float64), (len(r1.h), 1))
        fluid, pending, t = r1.amounts(b)
        M = np.stack([np.bincount(r1.g_c, weights=r1.h * r1.delta[:, k], minlength=r1.nrw) for k in range(3)], axis=1)
        gM = M @ np.asarray(g, dtype=np.float64)
        closed_fluid = r1.h * (-(t["L"] / t["S"]))[r1.g_c]
        exact_fluid = r1.h * (gM / t["H"])[r1.g_c]
        exact_pending = r1.wm @ np.asarray(g, dtype=np.float64)
        return {"formula_vs_closed_form_fluid_max_abs": float(np.abs(fluid - closed_fluid).max()),
                "fluid_vs_exact_target_max_abs": float(np.abs(fluid - exact_fluid).max()),
                "pending_vs_exact_target_max_abs": float(np.abs(pending - exact_pending).max()),
                "fluid_scale": float(np.abs(exact_fluid).max()), "pending_scale": float(np.abs(exact_pending).max())}

    def affine_target(self, slope: float = 0.1) -> tuple:
        """The task-118 target on the CORRECTED map for ALL modified rows, per axis, with the PER-ROW scale (contract: errors divided by
        the prism volume v of their own row, not a global volume). Returns (metrics, arrays for the independent audit).
        E(x) = 1 + slope (x_k - C_full_k), Q = v. Exact: fluid h (1 + slope delta_k), pending w_b + slope m_bk.
        Baseline: stored-R part + h slope (delta_k - M_k/H) and w_b. Corrected = baseline + the section 15.11 increment."""
        r1, m = self.r1, self.m
        stored = r1.stored_part()
        v_part = m.prism[m.g_row[r1.g_idx]]
        v_dest = m.prism[r1.cp.w_prod]
        v_row = m.prism[r1.cp.prod_rows]
        fix = r1.fixture_mask()
        fix_part, fix_dest = fix[r1.g_c], fix[r1.w_c]
        metrics = {"modified_rows": int(r1.nrw), "fixture_rows_with_W": int(fix.sum()), "fixture_row_ids": r1.cp.prod_rows[fix].tolist()}
        arrays = {"prod_rows": r1.cp.prod_rows, "v_row": v_row, "g_entry_index": r1.g_idx, "v_part": v_part, "w_prod_row": r1.cp.w_prod,
                  "w_recv_row": r1.cp.w_recv, "v_dest": v_dest, "fixture_mask": fix}
        for k in range(3):
            gk = np.zeros(3)
            gk[k] = slope
            b = np.tile(gk, (len(r1.h), 1))
            fluid, pending, t = r1.amounts(b)
            Mk = np.bincount(r1.g_c, weights=r1.h * r1.delta[:, k], minlength=r1.nrw)
            base_f = stored + r1.h * slope * (r1.delta[:, k] - (Mk / t["H"])[r1.g_c])
            exact_f = r1.h * (1.0 + slope * r1.delta[:, k])
            base_p = r1.w.copy()
            exact_p = r1.w + slope * r1.wm[:, k]
            err_f0, err_f1 = base_f - exact_f, base_f + fluid - exact_f
            err_p0, err_p1 = base_p - exact_p, base_p + pending - exact_p
            row_f1 = np.bincount(r1.g_c, weights=err_f1, minlength=r1.nrw)
            row_p1 = np.bincount(r1.w_c, weights=err_p1, minlength=r1.nrw)
            row_f0 = np.bincount(r1.g_c, weights=err_f0, minlength=r1.nrw)
            row_p0 = np.bincount(r1.w_c, weights=err_p0, minlength=r1.nrw)
            entry = {"fluid_part_over_v_baseline_max": float((np.abs(err_f0) / v_part).max()),
                     "fluid_part_over_v_corrected_max": float((np.abs(err_f1) / v_part).max()),
                     "pending_part_over_v_baseline_max": float((np.abs(err_p0) / v_dest).max()),
                     "pending_part_over_v_corrected_max": float((np.abs(err_p1) / v_dest).max()),
                     "fluid_row_sum_over_v_baseline_max": float((np.abs(row_f0) / v_row).max()),
                     "fluid_row_sum_over_v_corrected_max": float((np.abs(row_f1) / v_row).max()),
                     "pending_row_sum_over_v_baseline_max": float((np.abs(row_p0) / v_row).max()),
                     "pending_row_sum_over_v_corrected_max": float((np.abs(row_p1) / v_row).max()),
                     "row_total_conserved_over_v_max": float((np.abs(row_f1 + row_p1 - (row_f0 + row_p0)) / v_row).max())}
            if fix.any():
                entry["fixture_fluid_part_over_v_baseline_max"] = float((np.abs(err_f0)[fix_part] / v_part[fix_part]).max())
                entry["fixture_fluid_part_over_v_corrected_max"] = float((np.abs(err_f1)[fix_part] / v_part[fix_part]).max())
                entry["fixture_pending_part_over_v_baseline_max"] = float((np.abs(err_p0)[fix_dest] / v_dest[fix_dest]).max())
                entry["fixture_pending_part_over_v_corrected_max"] = float((np.abs(err_p1)[fix_dest] / v_dest[fix_dest]).max())
            metrics[f"axis{k}"] = entry
            for name, value in (("err_f_baseline", err_f0), ("err_f_corrected", err_f1), ("err_p_baseline", err_p0), ("err_p_corrected", err_p1),
                                ("exact_f", exact_f), ("exact_p", exact_p), ("base_f", base_f), ("base_p", base_p), ("inc_f", fluid), ("inc_p", pending)):
                arrays[f"axis{k}_{name}"] = value
        return metrics, arrays

    def linearity(self, seed: int = 12345, count: int = 4096) -> dict:
        """The closed-form slopes against the pinned _outgoing_shape: E(arm) - E(0) = b . arm (it is linear in the arm)."""
        m, r1 = self.m, self.r1
        rng = np.random.default_rng(seed)
        n = min(count, len(r1.g_cell))
        pick = np.sort(rng.choice(len(r1.g_cell), size=n, replace=False))
        rho = rng.uniform(0.95, 1.05, m.cells)
        v = rng.uniform(-0.05, 0.05, (m.cells, 3))
        grad_rho = rng.uniform(-1e-2, 1e-2, (m.cells, 3))
        grad_v = rng.uniform(-1e-2, 1e-2, (m.cells, 3, 3))
        b = r1.slopes(rho, v, grad_rho, grad_v)[pick]
        cell, d = r1.g_cell[pick], r1.g_dir[pick]
        arm = rng.uniform(-1.5, 1.5, (n, 3))
        c, w = m.c[d], m.w[d]
        e1 = m._outgoing_shape(c, w, rho[cell], v[cell], arm, grad_rho[cell], grad_v[cell])
        e0 = m._outgoing_shape(c, w, rho[cell], v[cell], np.zeros((n, 3)), grad_rho[cell], grad_v[cell])
        diff = np.abs((e1 - e0) - (b * arm).sum(axis=1))
        return {"entries": int(n), "max_abs_difference": float(diff.max()), "scale": float(np.abs(e1 - e0).max())}

    def state_report(self, p: np.ndarray, s: np.ndarray) -> dict:
        """Baseline and corrected next populations and pending for a post-collision state p (no trajectory). Everything the physical
        gates need: finite and minimum of the corrected next f, next pending, next collision p, and outgoing Q (Q is unchanged by the
        correction), the augmented mass and momentum increments, and the increment sizes. Signed increments are reported only."""
        m, r1 = self.m, self.r1
        new, s_next, wall = DH.ForcedClosedModel.transport_augmented(m, p, s)
        new_c, s_c = r1.apply(new, s_next, p)
        p_next = m.collide(new_c)
        fluid, pending, t = r1.amounts(r1.from_state(p))
        dP = m.augmented_momentum(new_c, s_c) - m.augmented_momentum(new, s_next)
        # C6: the transport budgets against the INPUT (p, s) and the returned wall impulse J (collision is unchanged)
        V = m.volume
        mass_in = float(V @ p.sum(axis=1) + s.sum())
        mass_b = float(V @ new.sum(axis=1) + s_next.sum()) - mass_in
        mass_c = float(V @ new_c.sum(axis=1) + s_c.sum()) - mass_in
        J = wall.impulse.sum(axis=0)
        P_in = m.augmented_momentum(p, s)
        budget_b = m.augmented_momentum(new, s_next) - P_in - J
        budget_c = m.augmented_momentum(new_c, s_c) - P_in - J
        cnorm = np.linalg.norm(m.c, axis=1)
        crow = np.linalg.norm(m.c_row, axis=1)
        gross = (float(V @ (np.abs(new_c) @ cnorm)) + float(V @ (np.abs(p) @ cnorm)) + float(np.abs(s_c) @ crow) + float(np.abs(s) @ crow)
                 + float(np.abs(wall.impulse).sum()))
        return {"min_new_baseline": float(new.min()), "min_new_corrected": float(new_c.min()),
                "min_pending_baseline": float(s_next.min()), "min_pending_corrected": float(s_c.min()),
                "min_next_p_corrected": float(p_next.min()), "min_outgoing_q": float(wall.outgoing.min()),
                "all_finite": bool(np.isfinite(new_c).all() and np.isfinite(s_c).all() and np.isfinite(p_next).all() and np.isfinite(wall.outgoing).all()),
                "max_abs_population_increment": float(np.abs(new_c - new).max()), "max_abs_pending_increment": float(np.abs(s_c - s_next).max()),
                "rows_with_nonzero_increment": int((np.abs(t["L"]) > 0).sum()), "rows_modified": r1.nrw,
                "diag_mass_increment": float(m.volume @ (new_c - new).sum(axis=1) + (s_c - s_next).sum()),
                "diag_augmented_momentum_increment": dP.tolist(), "diag_augmented_momentum_increment_max_abs": float(np.abs(dP).max()),
                "traction_defect_max": float(m.traction_defect(wall).max()),
                "mass_budget_baseline": mass_b, "mass_budget_corrected": mass_c, "mass_input": mass_in,
                "momentum_budget_baseline_max_abs": float(np.abs(budget_b).max()), "momentum_budget_corrected_max_abs": float(np.abs(budget_c).max()),
                "momentum_gross_scale": gross, "wall_impulse_sum": J.tolist()}

    def rest_case(self) -> dict:
        """The UNFORCED constant/rest state f = w in every cell with the container pending s0: the baseline rest fixed point and the
        corrected map on it. It uses a SEPARATE zero-force model on the same container and companion (the driven model has F != 0 and
        fields(p) subtracts F/2, so p = w would be a moving state there). The driven model is not touched. Distinct from the forced
        first update. Slopes are computed from the state, not set to zero."""
        zero = R1CorrectedModel(self.m.container_path, np.zeros(3))
        zero.attach(self.r1.cp)
        if float(np.abs(zero.force).max()) != 0.0:
            raise Invalid("the rest model has a nonzero force")
        m, r1 = zero, zero.r1
        p = np.tile(m.w, (m.cells, 1))
        s0 = np.asarray(m.initial_pending, dtype=np.float64)
        new, s_next, _ = DH.ForcedClosedModel.transport_augmented(m, p, s0)
        new_c, s_c = r1.apply(new, s_next, p)
        b = r1.from_state(p)
        fluid, pending, _ = r1.amounts(b)
        return {"max_abs_slope": float(np.abs(b).max()), "max_abs_fluid_amount": float(np.abs(fluid).max()),
                "max_abs_pending_amount": float(np.abs(pending).max()),
                "baseline_max_abs_new_minus_rest": float(np.abs(new - p).max()), "baseline_max_abs_pending_minus_s0": float(np.abs(s_next - s0).max()),
                "corrected_max_abs_new_minus_rest": float(np.abs(new_c - p).max()), "corrected_max_abs_pending_minus_s0": float(np.abs(s_c - s0).max()),
                "corrected_minus_baseline_new_max_abs": float(np.abs(new_c - new).max()),
                "corrected_minus_baseline_pending_max_abs": float(np.abs(s_c - s_next).max()),
                "population_scale": float(np.abs(new).max())}

    def first_update_equality(self, f0: np.ndarray, s0: np.ndarray) -> dict:
        """The registered start: corrected update 1 against the baseline update 1 (absolute, population scale)."""
        m = self.m
        p = m.collide(f0)
        new, s_next, _ = DH.ForcedClosedModel.transport_augmented(m, p, s0)
        new_c, s_c, _ = m.transport_augmented(p, s0)
        return {"max_abs_new_difference": float(np.abs(new_c - new).max()), "max_abs_pending_difference": float(np.abs(s_c - s_next).max()),
                "population_scale": float(np.abs(new).max())}
