"""Task 89 (E2): the wall-extrapolated even reference of the D3Q27 RR channel (scientist selection #1850; derivation
TASK89_WALL_REFERENCE_E2 Sections 1-3 and TASK89_AFFINE_SHEAR_IDENTITY_E2). A derived extension, NOT a printed Chen rule,
not qualified.

The ONLY change against integration/task87/t87_model.Model27 (frozen, unchanged): the even part of the return of facet alpha
uses the wall-value state

    rho_w = rho_s - h_s (d_n rho)_s,     u_w = P [ u_s - h_s (d_n u)_s ],     h_s = 1/2 (sample at facet centroid + n/2),
    B_r = 2 v_r w_r rho_w [1 + (9/2)(c_r.u_w)^2 - (3/2)|u_w|^2],

with (d_n rho)_s = n_alpha . S[L rho_p] and (d_n u)_s = S[L u_p] n_alpha: the EXISTING gradient reconstruction L of the post-collision
fields (rho_p, u_p = (sum c p - F/2)/rho_p), sampled by the EXISTING sampler S (the time level of M0 Section 3).
Unchanged: the friction at the sample (-K rho_s phi_r . u_s), the facet-wide mass correction (#1760), the feq3 Taylor
redistribution, the RR collision with full Phi and the original force. rho_w and u_w are recorded per facet; a rho_w that is not
finite and positive stops the step (no clamp).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task87"))
import t87_model as M27  # noqa: E402  (frozen)

Invalid = M27.Invalid


@dataclass
class WallPass89(M27.WallPass):
    rho_w: np.ndarray = None          # facets
    u_w: np.ndarray = None            # facets x 3 (tangential)


class Dependency89:
    """Structural dependency of WallReferenceModel27 (#1858). Same as t87_solve.Dependency27 except the sample path: the even
    reference reads S[L rho_p], S[L u_p], so Q_r depends on the gradient stencils (reach) of the sample cells, not only on the
    sample cells: sample = facet_rows @ ones(S) @ reach. A superset of the Dependency27 pattern (reach contains the identity)."""

    @staticmethod
    def make(model, reduction):
        sys.path.insert(0, str(HERE.parent / "task87"))
        import t87_solve as S27  # noqa: PLC0415

        class _Dependency(S27.Dependency27):
            def __init__(self, m, r) -> None:
                from scipy.sparse import csr_matrix, diags, identity  # noqa: PLC0415
                q = m.q
                states = np.arange(m.states)
                collapse = csr_matrix((np.ones(m.states), (states // q, states)), shape=(m.cells, m.states))
                reach = self._ones(sum(abs(g) for g in m.gradient)) + identity(m.cells, format="csr")
                direct = collapse @ self._ones(m.D) @ collapse.T
                direct_arm = collapse @ self._ones(sum(abs(a) for a in m.D_arm)) @ collapse.T @ reach
                rows = np.arange(m.rows)
                gather = self._ones(m.G) @ collapse.T @ reach
                facet_rows = csr_matrix((np.ones(m.rows), (rows, m.facet_of_row)), shape=(m.rows, m.facets))
                sample = facet_rows @ self._ones(m.S) @ reach                         # NEW: gradient stencils of the sample cells
                outgoing = facet_rows @ facet_rows.T @ (gather + sample)
                scatter = collapse @ self._ones(m.R)
                receivers = (np.diff(scatter.tocsr().indptr) > 0).astype(np.float64)
                cell = direct + direct_arm + scatter @ (outgoing + gather) + diags(receivers) @ reach
                indicator = csr_matrix((np.ones(m.cells), (r.group, np.arange(m.cells))), shape=(r.groups, m.cells))
                self.pattern = np.asarray((indicator @ cell @ indicator.T).todense()) > 0.0

        return _Dependency(model, reduction)


class WallReferenceModel27(M27.Model27):
    """Model27 with the wall-extrapolated even reference. Everything else is inherited unchanged."""

    SAMPLE_HEIGHT = 0.5

    def wall_state(self, rho_s: np.ndarray, u_s_full: np.ndarray, grad_rho: np.ndarray, grad_v: np.ndarray) -> tuple:
        """(rho_w, u_w) per facet. grad_rho: cells x 3 (axis); grad_v: cells x 3 (component) x 3 (axis); u_s_full unprojected."""
        n = self.normal
        d_rho = ((self.S @ grad_rho) * n).sum(axis=1)                                   # n . S[L rho]
        sampled_grad_v = np.stack([self.S @ grad_v[:, k, :] for k in range(3)], axis=1)   # facets x component x axis
        d_u = (sampled_grad_v * n[:, None, :]).sum(axis=2)                                # S[L u] n
        rho_w = rho_s - self.SAMPLE_HEIGHT * d_rho
        u_w = u_s_full - self.SAMPLE_HEIGHT * d_u
        u_w = u_w - n * (u_w * n).sum(axis=1)[:, None]
        if not (np.isfinite(rho_w).all() and rho_w.min() > 0.0 and np.isfinite(u_w).all()):
            raise Invalid("the wall-extrapolated density is not finite and positive (no clamp)")
        return rho_w, u_w

    def transport(self, p: np.ndarray) -> tuple:
        """t87_model.Model27.transport with the even term at (rho_w, u_w); every other statement is the inherited one."""
        q = self.q
        rho, v = self.fields(p)
        grad_p = [g @ p for g in self.gradient]
        grad_rho = np.stack([g @ rho for g in self.gradient], axis=1)
        grad_v = np.stack([g @ v for g in self.gradient], axis=2)
        flat = p.reshape(-1)
        direct = self.D @ flat
        direct_increment = sum(self.D_arm[axis] @ grad_p[axis].reshape(-1) for axis in range(3))
        gathered = self.G @ flat + sum(self.G_arm[axis] @ grad_p[axis].reshape(-1) for axis in range(3))
        rho_s = self.S @ rho
        u_s_full = self.S @ v
        u_s = u_s_full - self.normal * (u_s_full * self.normal).sum(axis=1)[:, None]
        rho_w, u_w = self.wall_state(rho_s, u_s_full, grad_rho, grad_v)
        rho_wr, u_wr = rho_w[self.facet_of_row], u_w[self.facet_of_row]                  # even reference (NEW)
        rho_r, u_r = rho_s[self.facet_of_row], u_s[self.facet_of_row]                    # friction at the sample (unchanged)
        cu = (self.c_row * u_wr).sum(axis=1)
        even = 2.0 * self.w_row * rho_wr * (1.0 + 4.5 * cu * cu - 1.5 * (u_wr * u_wr).sum(axis=1))
        ret = self.prism * even - gathered - self.friction * rho_r * (self.phi * u_r).sum(axis=1)
        delta = np.bincount(self.facet_of_row, weights=gathered - ret, minlength=self.facets) / self.facet_volume
        outgoing = ret + self.prism * delta[self.facet_of_row]
        part = self._outgoing_shape(self.c_row[self.g_row], self.w_row[self.g_row], rho[self.g_cell], v[self.g_cell], self.g_arm,
                                    grad_rho[self.g_cell], grad_v[self.g_cell])
        row_mean = np.bincount(self.g_row, weights=self.g_weight * part, minlength=self.rows) / self.g_total
        entry = self._outgoing_shape(self.c[self.r_direction], self.w[self.r_direction], rho[self.r_cell], v[self.r_cell], self.r_arm,
                                     grad_rho[self.r_cell], grad_v[self.r_cell])
        scatter = np.bincount(self.r_state, weights=self.r_part * (entry - row_mean[self.r_row]), minlength=self.states)
        returned = self.R @ outgoing + scatter
        new = (direct + direct_increment + returned).reshape(self.cells, q)
        if not np.isfinite(new).all():
            raise Invalid("a new population is not finite")
        energy = np.bincount(self.facet_of_row, weights=self.energy_row * (outgoing - gathered), minlength=self.facets)
        impulse_rows = self.c_row * (outgoing + gathered)[:, None]
        impulse = np.stack([np.bincount(self.facet_of_row, weights=impulse_rows[:, k], minlength=self.facets) for k in range(3)], axis=1)
        return new, WallPass89(gathered, outgoing, delta, energy, (impulse * self.normal).sum(axis=1), rho_w, u_w)
