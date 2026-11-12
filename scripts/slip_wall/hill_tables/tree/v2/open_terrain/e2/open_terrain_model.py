"""V2 open terrain (E2, #2338): the REFERENCE-only open operator. No engine patch, no builder change, no frozen file changed.

Contract: v2/open_terrain/e2/V2_OPEN_DOMAIN_LEDGER_CONTRACT_E2_2026-11-10.md, Sections 3-4 (the current ownership and
ledger), 6c (decision #2338) and 7 (the isothermal characteristic E state).

The operator is the existing transport of the rough SGS pending-terrain reference (rough_terrain_model.RoughSgsTerrainModel27,
which ends in t95_model.TerrainModel27.transport_augmented) on a PERIODIC builder container whose columns x = 0 and x = Nx - 1
have a flat apron (contract A2). The open operator does not need new tables: it classifies the existing ones and forms only
the entries that the contract allows.
- E cells: the cells of the columns x = 0 (inlet) and x = Nx - 1 (outlet). Their post-collision state is feq3(rho_b, u_b)
  (A4 or Section 7) and their gradient rows are zero (O3).
- Exterior facets: the facets of the two E columns (O2). Their rows return Q*_r = v_r feq3_(c_r)(rho_b, u_b) of the E cell next
  to the facet (O3a), without scatter and without friction; their gathers from Omega donors are absorbed.
- Every table entry that the operator forms is checked against the x wrap (fail closed): no D, G, R, S, L or W entry that
  joins two places more than Nx/2 apart in x may touch Omega or an interior facet. Entries that join only E cells or
  exterior rows to E cells are not formed (their values land in E cells, which the boundary state overwrites).
- The ledger (Section 4) is computed in every update from the same quantities.

Method resolution order (OpenTransport95 sits below the rough and SGS overrides, so the rough per-row friction swap and the
SGS collision act unchanged; OpenTransport95 replaces only the body of TerrainModel27.transport_augmented):
  OpenRoughSgsTerrainModel27 -> RoughSgsTerrainModel27 -> SgsTerrainModel27 -> SgsWallReferenceModel27 -> OpenTransport95
  -> TerrainModel27 -> WallReferenceModel27 -> Model27 -> object
Without an open declaration (open_x = False) the body is the frozen statement list; check C0 verifies the equality.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import diags

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[2]
sys.path.insert(0, str(WALL / "v2" / "rough_wall" / "e2"))
sys.path.insert(0, str(WALL / "integration" / "task95"))
sys.path.insert(0, str(WALL / "integration" / "task39"))
import rough_terrain_model as RTM  # noqa: E402  (pending R3 adapter, unchanged)
import t39_lib as T39  # noqa: E402  (container section reader)
import t95_model as T95  # noqa: E402  (frozen)

Invalid = T95.Invalid
CS = 1.0 / np.sqrt(3.0)          # lattice speed of sound c


class Refused(Invalid):
    """A container or a boundary state that the open operator does not support (fail closed)."""


# ---- the classification of the periodic container ------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenLayout:
    """The open-x classification of one container (contract O1-O6). Index arrays only; no table is copied."""
    nx: int
    cell_x: np.ndarray            # cells: engine x index
    is_e: np.ndarray              # cells: True for E cells (x = 0 or x = Nx - 1)
    e_cells: np.ndarray           # E cell indices
    side_of_e: np.ndarray         # per E cell: -1 inlet (x = 0), +1 outlet (x = Nx - 1)  (outward normal sign s)
    inward: np.ndarray            # per E cell: (b-1, b-2) cell indices along the inward normal (Section 7.3)
    exterior_facet: np.ndarray    # facets: True for E-column facets
    e_cell_of_facet: np.ndarray   # facets: the E cell next to an exterior facet, -1 otherwise
    exterior_row: np.ndarray      # rows
    e_cell_of_row: np.ndarray     # rows: e_cell_of_facet[facet_of_row]

    @staticmethod
    def classify(container: Path, model) -> "OpenLayout":
        a = T39.arrays(T39.read_sections(Path(container).read_bytes()))
        box = np.asarray(a["box"], dtype=np.int64)
        xyz = np.asarray(a["cell_xyz"], dtype=np.int64).reshape(-1, 3)
        centroid = np.asarray(a["centroid"], dtype=np.float64).reshape(-1, 3)
        sample_point = np.asarray(a["sample_point"], dtype=np.float64).reshape(-1, 3)
        nx = int(box[0])
        if len(xyz) != model.cells or xyz[:, 0].min() != 0 or xyz[:, 0].max() != nx - 1:
            raise Refused("cell_xyz does not cover x = 0 .. Nx - 1 for every model cell")
        if not np.array_equal(np.floor(centroid[:, :2]).astype(np.int64), xyz[:, :2]):
            raise Refused("a cell centroid is not inside its x-y column (engine x, y = builder i, j is assumed)")
        cell_x = xyz[:, 0]
        is_e = (cell_x == 0) | (cell_x == nx - 1)
        e_cells = np.flatnonzero(is_e)
        if not np.all(model.volume[e_cells] == 1.0):
            raise Refused("an E cell is not fully fluid (V != 1): the apron is not flat at an integer face (A2)")
        side_of_e = np.where(cell_x[e_cells] == 0, -1, 1).astype(np.int64)
        # inward neighbours along x at the same (y, z): exact lookup by engine coordinates
        key = {tuple(int(v) for v in xyz[c]): c for c in range(model.cells)}
        inward = np.empty((len(e_cells), 2), dtype=np.int64)
        for k, c in enumerate(e_cells):
            step = -side_of_e[k]
            for d in (1, 2):
                n = key.get((int(cell_x[c] + step * d), int(xyz[c, 1]), int(xyz[c, 2])))
                if n is None or model.volume[n] != 1.0 or is_e[n]:
                    raise Refused(f"E cell {c}: no full interior cell {d} step(s) inward at the same y, z (apron too narrow)")
                inward[k, d - 1] = n
        # exterior facets: the facet point is the sample point minus n/2 (builder: sample at the facet point + n/2)
        normal = model.normal
        facet_point = sample_point - 0.5 * normal
        facet_column = np.floor(facet_point[:, 0]).astype(np.int64)
        exterior_facet = (facet_column == 0) | (facet_column == nx - 1)
        flat = (normal[:, 0] == 0.0) & (normal[:, 1] == 0.0) & (np.abs(normal[:, 2]) == 1.0)
        if not flat[exterior_facet].all():
            raise Refused("an E-column facet is not a flat face (normal != +-z): the apron is not flat (A2)")
        e_cell_of_facet = np.full(model.facets, -1, dtype=np.int64)
        e_by_point = {}
        for c in e_cells:
            e_by_point.setdefault((int(xyz[c, 0]), int(xyz[c, 1])), []).append(c)
        for f in np.flatnonzero(exterior_facet):
            column = (int(facet_column[f]), int(np.floor(facet_point[f, 1])))
            match = [c for c in e_by_point.get(column, []) if centroid[c, 2] == sample_point[f, 2]]
            if len(match) != 1:
                raise Refused(f"exterior facet {f}: not exactly one E cell next to it (found {len(match)})")
            e_cell_of_facet[f] = match[0]
        exterior_row = exterior_facet[model.facet_of_row]
        return OpenLayout(nx, cell_x, is_e, e_cells, side_of_e, inward, exterior_facet, e_cell_of_facet, exterior_row,
                          e_cell_of_facet[model.facet_of_row])


@dataclass(frozen=True)
class WrapReport:
    """Counts of the table entries across the x wrap, by what they touch (O8; the refusal is in WrapCheck)."""
    D_e_only: int
    G_exterior_e: int
    R_exterior_e: int
    S_exterior: int
    L_e: int


class WrapCheck:
    """Fail closed on any formed table entry across the x wrap (contract O8). An entry is across the wrap when the two x
    positions it joins are more than Nx/2 apart (every true reach is <= 3 cells)."""

    def __init__(self, model, layout: OpenLayout) -> None:
        self.m, self.L = model, layout

    def _wrapped(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return np.abs(x0 - x1) > 0.5 * self.L.nx

    def run(self) -> WrapReport:
        m, L, q = self.m, self.L, self.m.q
        facet_x = np.floor(m.layout_facet_point_x).astype(np.int64)
        # D: receiver state rows, donor state columns
        Dc = m.D.tocoo()
        rx, dx = L.cell_x[Dc.row // q], L.cell_x[Dc.col // q]
        w = self._wrapped(rx, dx)
        if (w & ~(L.is_e[Dc.row // q] & L.is_e[Dc.col // q])).any():
            raise Refused("a D entry across the x wrap touches an interior cell")
        # G: rows x donor states
        g_wrapped = self._wrapped(facet_x[m.facet_of_row[m.g_row]], L.cell_x[m.g_cell])
        if (g_wrapped & ~L.exterior_row[m.g_row]).any():
            raise Refused("a gather of an interior facet row is across the x wrap")
        if (g_wrapped & ~L.is_e[m.g_cell]).any():
            raise Refused("a gather across the x wrap has an interior donor")
        # R: receiver states x rows
        r_wrapped = self._wrapped(facet_x[m.facet_of_row[m.r_row]], L.cell_x[m.r_cell])
        if (r_wrapped & ~L.exterior_row[m.r_row]).any():
            raise Refused("a return of an interior facet row is across the x wrap")
        if (r_wrapped & ~L.is_e[m.r_cell]).any():
            raise Refused("a return across the x wrap lands in an interior cell")
        # S: facets x cells
        Sc = m.S.tocoo()
        s_wrapped = self._wrapped(facet_x[Sc.row], L.cell_x[Sc.col])
        if (s_wrapped & ~L.exterior_facet[Sc.row]).any():
            raise Refused("a sampler stencil of an interior facet is across the x wrap")
        # L: the gradient stencils of interior cells (E rows are zeroed by the operator)
        l_count = 0
        for g in m.frozen_gradient:
            Gc = g.tocoo()
            l_wrapped = self._wrapped(L.cell_x[Gc.row], L.cell_x[Gc.col])
            if (l_wrapped & ~L.is_e[Gc.row]).any():
                raise Refused("a gradient stencil of an interior cell is across the x wrap")
            l_count += int(l_wrapped.sum())
        # W: receiver rows x producer rows; none may touch an exterior facet (O4)
        Wc = m.W.tocoo()
        if (L.exterior_row[Wc.row] | L.exterior_row[Wc.col]).any():
            raise Refused("a pending (W) entry touches an exterior facet (O4)")
        if self._wrapped(facet_x[m.facet_of_row[Wc.row]], facet_x[m.facet_of_row[Wc.col]]).any():
            raise Refused("a pending (W) entry is across the x wrap")
        return WrapReport(int(w.sum()), int(g_wrapped.sum()), int(r_wrapped.sum()), int(s_wrapped.sum()), l_count)


# ---- the E states -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryState:
    """(rho_b, u_b) per E cell (layout.e_cells order)."""
    rho: np.ndarray
    u: np.ndarray

    def checked(self) -> "BoundaryState":
        if not (np.isfinite(self.rho).all() and np.isfinite(self.u).all() and self.rho.min() > 0.0):
            raise Refused("an E state is not finite with rho_b > 0 (7.6)")
        return self


class ReservoirBoundary:
    """Contract A4: the E state is fixed. A reservoir condition, not an outflow condition (A7)."""

    def admissible(self, state: BoundaryState, level: str) -> BoundaryState:
        return state.checked()

    def update(self, state: BoundaryState, rho_f: np.ndarray, u_f: np.ndarray) -> BoundaryState:
        return state


@dataclass(frozen=True)
class OutletRelaxation:
    """7.4 outlet: L_minus = K_p c^2 (rho_b - rho_inf), K_p = sigma (1 - Ma^2) c / L_x; all three declared, sigma > 0."""
    sigma: float
    mach: float
    length_x: float
    rho_inf: float

    @property
    def K_p(self) -> float:
        scalars = np.array([self.sigma, self.mach, self.length_x, self.rho_inf], dtype=np.float64)
        if not np.isfinite(scalars).all():
            raise Refused("outlet relaxation: sigma, Ma, L_x and rho_inf must be finite (#2353)")
        if not (self.sigma > 0.0 and 0.0 <= self.mach < 1.0 and self.length_x > 0.0 and self.rho_inf > 0.0):
            raise Refused("outlet relaxation: sigma > 0, 0 <= Ma < 1, L_x > 0 and rho_inf > 0 are required (K_p = 0 refused)")
        value = self.sigma * (1.0 - self.mach * self.mach) * CS / self.length_x
        if not (np.isfinite(value) and value > 0.0):
            raise Refused(f"outlet relaxation: the computed K_p = {value!r} is not finite and strictly positive (#2353)")
        return value


@dataclass(frozen=True)
class InletTarget:
    """7.4 inlet: steady target velocity U_b (lattice axes) relaxed with K_u >= 0, K_u < 1; or the hard algebraic limit."""
    velocity: np.ndarray          # (3,) lattice axes, the same for every inlet E cell
    K_u: float
    hard: bool

    def checked(self) -> "InletTarget":
        if not (np.isfinite(self.velocity).all() and self.velocity.shape == (3,)):
            raise Refused("inlet target velocity: three finite components")
        if not self.velocity[0] > 0.0:
            raise Refused("inlet target velocity: u_x > 0 (inflow through x = 0) is required")
        if not self.hard and not (np.isfinite(self.K_u) and 0.0 <= self.K_u < 1.0):
            raise Refused("inlet K_u: 0 <= K_u < 1 (explicit update); the hard limit is a separate mode")
        return self


class CharacteristicBoundary:
    """Contract Section 7: the isothermal LODI E state, explicit, Delta t = 1, level-n fields (7.5), refusals 7.6.
    Outward normal along x with sign s; normal velocity un = s u_x; transverse v = u_y, w = u_z."""

    def __init__(self, layout: OpenLayout, outlet: OutletRelaxation, inlet: InletTarget) -> None:
        self.L = layout
        self.K_p = outlet.K_p
        self.rho_inf = outlet.rho_inf
        self.inlet = inlet.checked()

    def admissible(self, state: BoundaryState, level: str) -> BoundaryState:
        """7.6 refusals of one E state, at the level where it is formed (#2353: also the NEW state, not on the next call)."""
        state.checked()
        un = self.L.side_of_e * state.u[:, 0]
        outlet = self.L.side_of_e > 0
        # task 108: closed at zero speed. At u_n = 0 the two transverse velocity waves (no entropy wave: isothermal) have zero speed (outlet L_v = L_w = 0, the
        # relaxed inlet prescribes them), so no transverse inflow data are missing; wrong-sign flow stays refused.
        if (un[outlet] < 0.0).any():
            raise Refused(f"outlet backflow: u_n < 0 at an outlet E node ({level} state, 7.6)")
        if (un[~outlet] > 0.0).any():
            raise Refused(f"inlet outflow: u_n > 0 at an inlet E node ({level} state, 7.6)")
        if (np.abs(un) >= CS).any():
            raise Refused(f"|u_n| >= c at an E node ({level} state, 7.6)")
        return state

    def update(self, state: BoundaryState, rho_f: np.ndarray, u_f: np.ndarray) -> BoundaryState:
        L, c, c2 = self.L, CS, CS * CS
        s = L.side_of_e.astype(np.float64)
        self.admissible(state, "current")
        rho_b, u_b = state.rho, state.u
        un_b = s * u_b[:, 0]
        b1, b2 = L.inward[:, 0], L.inward[:, 1]
        outlet = L.side_of_e > 0

        def d_n(phi_b, phi_1, phi_2):
            return 0.5 * (3.0 * phi_b - 4.0 * phi_1 + phi_2)

        d_rho = d_n(rho_b, rho_f[b1], rho_f[b2])
        d_un = d_n(un_b, s * u_f[b1, 0], s * u_f[b2, 0])
        d_v = d_n(u_b[:, 1], u_f[b1, 1], u_f[b2, 1])
        d_w = d_n(u_b[:, 2], u_f[b1, 2], u_f[b2, 2])
        L_plus = (un_b + c) * (c2 * d_rho + rho_b * c * d_un)
        L_minus = np.empty_like(L_plus)
        L_v = np.empty_like(L_plus)
        L_w = np.empty_like(L_plus)
        # outlet: computed outgoing L_v, L_w; incoming L_minus by the pressure relaxation
        L_minus[outlet] = self.K_p * c2 * (rho_b[outlet] - self.rho_inf)
        L_v[outlet] = un_b[outlet] * d_v[outlet]
        L_w[outlet] = un_b[outlet] * d_w[outlet]
        # inlet: steady target (d_t U_b = 0)
        inlet = ~outlet
        target_un = s[inlet] * self.inlet.velocity[0]
        new_rho = np.empty_like(rho_b)
        new_u = np.empty_like(u_b)
        if self.inlet.hard:
            new_rho[inlet] = rho_b[inlet] - L_plus[inlet] / c2              # L_minus = L_plus: d_t rho = -L_plus / c^2
            new_u[inlet] = self.inlet.velocity[None, :]
        else:
            K = self.inlet.K_u
            L_minus[inlet] = L_plus[inlet] - 2.0 * rho_b[inlet] * c * K * (un_b[inlet] - target_un)
            L_v[inlet] = K * (u_b[inlet, 1] - self.inlet.velocity[1])
            L_w[inlet] = K * (u_b[inlet, 2] - self.inlet.velocity[2])
        relaxed = outlet if self.inlet.hard else np.ones_like(outlet)
        new_rho[relaxed] = rho_b[relaxed] - (L_plus[relaxed] + L_minus[relaxed]) / (2.0 * c2)
        new_un = un_b[relaxed] - (L_plus[relaxed] - L_minus[relaxed]) / (2.0 * rho_b[relaxed] * c)
        new_u[relaxed, 0] = s[relaxed] * new_un
        new_u[relaxed, 1] = u_b[relaxed, 1] - L_v[relaxed]
        new_u[relaxed, 2] = u_b[relaxed, 2] - L_w[relaxed]
        return self.admissible(BoundaryState(new_rho, new_u), "new")


# ---- the open transport -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Transfer:
    """One boundary transfer of contract Section 4: its mass and its momentum (3)."""
    mass: float
    momentum: np.ndarray


@dataclass
class OpenPass:
    """The four boundary transfers of one update (contract Section 4, Omega masks), their sums, and the wall pass."""
    wall: T95.WallPass95
    e_to_omega: Transfer             # (1) IN: direct and interior gathers from E donors
    exterior_return: Transfer        # (2) IN: Q* of exterior rows into Omega receivers
    omega_to_e: Transfer             # (3) OUT: direct (with arms) and interior returns (with scatter) into E receivers
    exterior_gather: Transfer        # (4) OUT: gathers (with arms) of exterior rows from Omega donors
    interior_impulse: np.ndarray     # sum over interior facets of J_a

    @property
    def phi_in(self) -> float:
        return self.e_to_omega.mass + self.exterior_return.mass

    @property
    def phi_out(self) -> float:
        return self.omega_to_e.mass + self.exterior_gather.mass

    @property
    def pi_in(self) -> np.ndarray:
        return self.e_to_omega.momentum + self.exterior_return.momentum

    @property
    def pi_out(self) -> np.ndarray:
        return self.omega_to_e.momentum + self.exterior_gather.momentum


class OpenTransport95(T95.TerrainModel27):
    """TerrainModel27.transport_augmented with the open rules when self.layout is set (contract Sections 3-4, 6c)."""

    layout: OpenLayout | None = None
    last_open: OpenPass | None = None

    def transport_augmented(self, p: np.ndarray, s: np.ndarray) -> tuple:
        # ---- the frozen statement list of TerrainModel27.transport_augmented (t95_model.py 93-131) ------------------------------
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
        incoming = gathered + s
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
        # ---- OPEN (O3a): exterior rows return Q*_r = v_r feq3_(c_r)(rho_b, u_b); no balance on exterior facets ----------------
        L = self.layout
        if L is not None:
            ext = L.exterior_row
            e_of_row = L.e_cell_of_row[ext]
            rho_b, u_b = rho[e_of_row], v[e_of_row]
            c_ext = self.c_row[ext]
            cu_b = (c_ext * u_b).sum(axis=1)
            uu_b = (u_b * u_b).sum(axis=1)
            bracket = 1.0 + 3.0 * cu_b + 4.5 * cu_b * cu_b - 1.5 * uu_b + 4.5 * (cu_b * cu_b * cu_b - cu_b * uu_b)
            q_star = self.prism[ext] * self.w_row[ext] * rho_b * bracket
            outgoing = outgoing.copy()
            outgoing[ext] = q_star
            delta = np.where(L.exterior_facet, 0.0, delta)
        # ---- the frozen statements continue (scatter; r_part of exterior rows is zero, O3a) ------------------------------------
        part = self._outgoing_shape(self.c_row[self.g_row], self.w_row[self.g_row], rho[self.g_cell], v[self.g_cell], self.g_arm,
                                    grad_rho[self.g_cell], grad_v[self.g_cell])
        row_mean = np.bincount(self.g_row, weights=self.g_weight * part, minlength=self.rows) / self.g_total
        entry = self._outgoing_shape(self.c[self.r_direction], self.w[self.r_direction], rho[self.r_cell], v[self.r_cell], self.r_arm,
                                     grad_rho[self.r_cell], grad_v[self.r_cell])
        scatter = np.bincount(self.r_state, weights=self.r_part * (entry - row_mean[self.r_row]), minlength=self.states)
        returned = self.R @ outgoing + scatter
        new = (direct + direct_increment + returned).reshape(self.cells, q)
        pending_next = self.W @ (outgoing / self.prism)
        if not (np.isfinite(new).all() and np.isfinite(pending_next).all()):
            raise Invalid("a new population or pending amount is not finite")
        impulse_rows = self.c_row * (outgoing + incoming)[:, None]
        impulse = np.stack([np.bincount(self.facet_of_row, weights=impulse_rows[:, k], minlength=self.facets) for k in range(3)], axis=1)
        wall = T95.WallPass95(gathered, s, incoming, outgoing, delta, impulse, rho_s, u_s, rho_w, u_w)
        if L is not None:
            self.last_open = self._ledger(flat, grad_p, outgoing, scatter, impulse, wall)
        return new, pending_next, wall

    def _ledger(self, flat, grad_p, outgoing, scatter, impulse, wall) -> OpenPass:
        """Contract Section 4: the four boundary transfers (1)-(4) and their momenta, from the quantities of this update."""
        L, q = self.layout, self.q
        state_e = np.repeat(L.is_e, q)
        state_o = ~state_e
        ext = L.exterior_row
        vol = np.repeat(self.volume, q)
        c_state = np.tile(self.c, (self.cells, 1))
        flat_e = np.where(state_e, flat, 0.0)
        flat_o = np.where(state_o, flat, 0.0)
        gp = [gq.reshape(-1) for gq in grad_p]                       # E rows are zero (O3)

        def moments(values: np.ndarray, weights: np.ndarray) -> tuple:
            return float(weights @ values), (weights * values) @ c_state

        # (1) E -> Omega: direct into Omega receivers and gathers of interior rows from E donors
        d_in = self.D @ flat_e
        m1, p1 = moments(d_in, np.where(state_o, vol, 0.0))
        g_in_mass = float((self.G @ flat_e)[~ext].sum())
        g_in_mom = np.array([float((self.G @ (flat_e * c_state[:, k]))[~ext].sum()) for k in range(3)])
        # (2) exterior returns into Omega receivers
        r_ext = self.R @ np.where(ext, outgoing, 0.0)
        m2, p2 = moments(r_ext, np.where(state_o, vol, 0.0))
        # (3) Omega -> E: direct (with arms) and interior returns (with scatter) into E receivers
        d_out = self.D @ flat_o + sum(self.D_arm[axis] @ gp[axis] for axis in range(3))
        r_int = self.R @ np.where(ext, 0.0, outgoing) + scatter
        m3, p3 = moments(d_out + r_int, np.where(state_e, vol, 0.0))
        # (4) exterior gathers from Omega donors (with arms)
        g_out = [self.G @ (flat_o * c_state[:, k]) + sum(self.G_arm[axis] @ (gp[axis] * c_state[:, k]) for axis in range(3))
                 for k in range(3)]
        g_out_mass = float((self.G @ flat_o + sum(self.G_arm[axis] @ gp[axis] for axis in range(3)))[ext].sum())
        g_out_mom = np.array([float(g_out[k][ext].sum()) for k in range(3)])
        interior_impulse = impulse[~L.exterior_facet].sum(axis=0)
        return OpenPass(wall, Transfer(m1 + g_in_mass, p1 + g_in_mom), Transfer(m2, p2), Transfer(m3, p3),
                        Transfer(g_out_mass, g_out_mom), interior_impulse)


class OpenRoughSgsTerrainModel27(RTM.RoughSgsTerrainModel27, OpenTransport95):
    """The rough SGS pending-terrain reference with the open-x operator (open_x = True) or the closed one (open_x = False)."""

    def __init__(self, container: Path, force: np.ndarray, closure, law=None, open_x: bool = True) -> None:
        super().__init__(container, force, closure, law)
        self.frozen_gradient = list(self.gradient)
        self.layout = None
        self.wrap = None
        if not open_x:
            return
        if np.any(self.force != 0.0):
            raise Refused("body force with open boundaries (contract A5, Section 5)")
        a = T39.arrays(T39.read_sections(Path(container).read_bytes()))
        sample_point = np.asarray(a["sample_point"], dtype=np.float64).reshape(-1, 3)
        layout = OpenLayout.classify(container, self)
        self.layout_facet_point_x = (sample_point - 0.5 * self.normal)[:, 0]
        self.wrap = WrapCheck(self, layout).run()
        keep = diags((~layout.is_e).astype(np.float64))
        self.gradient = [(keep @ g).tocsr() for g in self.frozen_gradient]           # O3: no gradient at E cells
        self.r_part = np.where(layout.exterior_row[self.r_row], 0.0, self.r_part)    # O3a: no scatter on exterior rows
        if np.any(self.initial_pending[layout.exterior_row] != 0.0):
            raise Refused("an exterior row has an initial pending amount (O4)")
        self.layout = layout

    # ---- the open update ------------------------------------------------------------------------------------------------------
    def e_populations(self, state: BoundaryState) -> np.ndarray:
        return self.eq.parts(state.rho, state.u, np.zeros(3))[0]

    def open_step(self, f: np.ndarray, s: np.ndarray, state: BoundaryState, boundary) -> tuple:
        """One update (7.5): p = collide(f); p_E = feq3(state^n); transport; state^(n+1) from the level-n fields of f;
        f'_E = feq3(state^(n+1)). Returns f', s', state^(n+1) and the OpenPass of this update."""
        L = self.layout
        if L is None:
            raise Invalid("open_step needs the open layout (open_x = True)")
        state = boundary.admissible(state, "current")               # before the transport uses it (#2353)
        p = self.collide(f)
        p[L.e_cells] = self.e_populations(state)
        new, pending_next, _ = self.transport_augmented(p, s)
        rho_f = f.sum(axis=1)
        u_f = (f @ self.c) / rho_f[:, None]
        state_next = boundary.update(state, rho_f, u_f)
        new[L.e_cells] = self.e_populations(state_next)
        return new, pending_next, state_next, self.last_open

    def open_mass(self, f: np.ndarray, s: np.ndarray) -> float:
        omega = ~self.layout.is_e
        return float(self.volume[omega] @ f[omega].sum(axis=1)) + float(s[~self.layout.exterior_row].sum())

    def open_momentum(self, f: np.ndarray, s: np.ndarray) -> np.ndarray:
        omega = ~self.layout.is_e
        interior = ~self.layout.exterior_row
        return self.volume[omega] @ (f[omega] @ self.c) - s[interior] @ self.c_row[interior]

    def interior_traction_defect(self, wall) -> np.ndarray:
        """#2377 OC4 gate: the delivered wall traction identity on the INTERIOR facets only (the rough_terrain_model
        traction_defect, K_a recomputed from this pass's u_s; per area). Exterior facets have no balance and no impulse
        (contract O3a), so they are not part of the identity."""
        return self.traction_defect(wall)[~self.layout.exterior_facet]

    def initial_e_state(self, rho: float, u: np.ndarray) -> BoundaryState:
        n = len(self.layout.e_cells)
        return BoundaryState(np.full(n, float(rho)), np.tile(np.asarray(u, dtype=np.float64), (n, 1))).checked()


EXPECTED_ORDER = ("OpenRoughSgsTerrainModel27", "RoughSgsTerrainModel27", "SgsTerrainModel27", "SgsWallReferenceModel27",
                  "OpenTransport95", "TerrainModel27", "WallReferenceModel27", "Model27", "object")
