"""Task 87 (E2): reference update of the D3Q27 RR channel with the extended volumetric wall (FP64, no storage layer).

Specification: TASK87_WALL_EXTENSION_E2_2026-11-10.md (amended: facet-wide mass balance #1760, feq3 Taylor redistribution
#1754), task 81 (RR collision, full Phi). Structure and table semantics follow integration/task70/t70_model.py (the D3Q19
reference) line by line, with these changes only:
  - 27 directions, lexicographic order (rest index 13, opposite 26 - i), D3Q27 weights;
  - collision: the task-81 RR operator with full Phi at every cell (wall-list cells included);
  - wall return: the even equilibrium (unchanged form, D3Q27 weight), ONE mass adjustment per facet (all rows);
  - redistribution: the Taylor expression of the feq3 bracket;
  - recorded, not conserved: the facet lattice kinetic-energy exchange sum (|c|^2/2)(Q - I) and the normal impulse.
State: populations f (cells x 27), stored pre-collision: sum f = rho, sum c f = rho u - F/2.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

INTEGRATION = Path(__file__).resolve().parents[1]


class Invalid(Exception):
    """The container or a state does not have the structure that the specification needs."""


class Readers:
    @staticmethod
    def t39():
        name = "t39_lib"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, INTEGRATION / "task39" / "t39_lib.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


@dataclass(frozen=True)
class Lattice27:
    c: np.ndarray          # 27 x 3, lexicographic
    w: np.ndarray          # 27
    opposite: np.ndarray   # 27
    shell: np.ndarray      # |c|^2

    @staticmethod
    def make() -> "Lattice27":
        c = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64)
        shell = np.rint((c * c).sum(axis=1)).astype(np.int64)
        w = np.array([8.0 / 27.0, 2.0 / 27.0, 1.0 / 54.0, 1.0 / 216.0])[shell]
        opposite = np.array([int(np.flatnonzero((c == -c[i]).all(axis=1))[0]) for i in range(27)], dtype=np.int64)
        if not (np.array_equal(opposite, 26 - np.arange(27)) and shell[13] == 0):
            raise Invalid("the lexicographic D3Q27 order is not the expected one")
        return Lattice27(c, w, opposite, shell)


class Equilibrium:
    """The feq3 bracket B_i(u), its velocity derivative, the full force term and the regularized non-equilibrium (task 81)."""

    def __init__(self, lattice: Lattice27) -> None:
        self.c, self.w = lattice.c, lattice.w

    def bracket(self, cu: np.ndarray, uu: np.ndarray) -> np.ndarray:
        """B = 1 + 3 cu + 9/2 cu^2 - 3/2 uu + 9/2 (cu^3 - cu uu). cu: n x m, uu: n (broadcast)."""
        uu = uu[:, None]
        return 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu + 4.5 * (cu * cu * cu - cu * uu)

    def slope(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        """dB/du for row-wise pairs (c, u): n x 3."""
        cu = (c * u).sum(axis=1)[:, None]
        uu = (u * u).sum(axis=1)[:, None]
        return 3.0 * c + 9.0 * cu * c - 3.0 * u + 4.5 * (3.0 * cu * cu * c - uu * c - 2.0 * cu * u)

    def parts(self, rho: np.ndarray, u: np.ndarray, force: np.ndarray) -> tuple:
        """(feq, Phi) for cells: feq = w rho B, Phi = 3 (c.F - u.F) w B (no density: F is the force density)."""
        cu = u @ self.c.T
        B = self.bracket(cu, (u * u).sum(axis=1))
        feq = self.w[None, :] * rho[:, None] * B
        phi = 3.0 * ((self.c @ force)[None, :] - (u @ force)[:, None]) * self.w[None, :] * B
        return feq, phi

    def non_equilibrium(self, A: np.ndarray, u: np.ndarray) -> np.ndarray:
        """w [9/2 H2:A + 27/2 ((cu)(cAc) - (cu tr A + 2 u.A.c)/3)], A: n x 3 x 3."""
        c = self.c
        cAc = np.einsum("ia,nab,ib->ni", c, A, c)
        trace = np.trace(A, axis1=1, axis2=2)[:, None]
        cu = u @ c.T
        uAc = np.einsum("na,nab,ib->ni", u, A, c)
        second = cAc - trace / 3.0
        third = cu * cAc - (cu * trace + 2.0 * uAc) / 3.0
        return self.w[None, :] * (4.5 * second + 13.5 * third)

    def stress(self, f: np.ndarray, phi: np.ndarray, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
        """A = sum H2 (f + Phi/2) - rho u u (sum Phi = 0, so the trace term is rho/3 I)."""
        g = f + 0.5 * phi
        second = np.einsum("ni,ia,ib->nab", g, self.c, self.c)
        return second - (rho / 3.0)[:, None, None] * np.eye(3)[None, :, :] - rho[:, None, None] * np.einsum("na,nb->nab", u, u)


class Lift:
    """Populations from fields: f = feq3(rho, u) + fneq_RR(A, u) - Phi/2, so sum f = rho, sum c f = rho u - F/2 (#1754)."""

    def __init__(self, lattice: Lattice27, force: np.ndarray) -> None:
        self.eq, self.force = Equilibrium(lattice), np.asarray(force, dtype=np.float64)

    def populations(self, rho: np.ndarray, u: np.ndarray, A: np.ndarray) -> np.ndarray:
        feq, phi = self.eq.parts(rho, u, self.force)
        return feq + self.eq.non_equilibrium(A, u) - 0.5 * phi


@dataclass
class WallPass:
    gathered: np.ndarray           # rows: I_r
    outgoing: np.ndarray           # rows: Q_r
    delta: np.ndarray              # facets: the facet mass adjustment
    energy_exchange: np.ndarray    # facets: sum_r (|c_r|^2/2)(Q_r - I_r) (recorded, not conserved)
    normal_impulse: np.ndarray     # facets: n . sum_r c_r (Q_r + I_r)


class Model27:
    """The complete update of one level: RR collision of every cell, ordinary streaming, the extended wall pass."""

    WALL_B, WALL_U, PERIOD = 0.5, 96.5, 144.0          # the registered channel (task 59 Geometry rule for the normal groups)

    def __init__(self, container: Path, force: np.ndarray, friction: float | None = None) -> None:
        t39 = Readers.t39()
        a = t39.arrays(t39.read_sections(container.read_bytes()))
        self.lattice = Lattice27.make()
        self.c, self.w, self.q = self.lattice.c, self.lattice.w, 27
        self.eq = Equilibrium(self.lattice)
        self.force = np.asarray(force, dtype=np.float64)
        self.nu, self.omega, container_friction = (float(a["scalars"][k]) for k in range(3))
        self.friction = container_friction if friction is None else float(friction)
        self.cells = len(a["volume"])
        self.states = self.q * self.cells
        self.volume = np.asarray(a["volume"], dtype=np.float64)
        if len(a["f0"]) != self.states:
            raise Invalid(f"f0 has {len(a['f0'])} values, not cells x 27: the container is not a D3Q27 container")
        if len(a["W_weights"]) != 0 or np.any(np.asarray(a["s0"]) != 0.0):
            raise Invalid("the container has pending amounts or a pending map")
        self.initial = np.asarray(a["f0"], dtype=np.float64).reshape(self.cells, self.q)
        self.wall_cells = a["gradient_cells"].astype(np.int64)
        self.is_wall = np.zeros(self.cells, dtype=bool)
        self.is_wall[self.wall_cells] = True
        self._groups(t39, a)
        self._direct(a)
        self._gradient(a)
        self._rows(a)
        self._scatter(a)

    # ---- tables (semantics of t70_model, q = 27) ------------------------------------------------------------------------------
    def _groups(self, t39, a: dict) -> None:
        """Normal groups: cells with the same q_c = 3x + z (the task-59 rule, centred and periodic)."""
        xyz = t39.Lattice(a).xyz
        qc = 3.0 * (xyz[:, 0] + 0.5) + xyz[:, 2] + 0.5
        centre = 0.5 * (self.WALL_B + self.WALL_U)
        keys = np.round(np.mod(qc - centre + 0.5 * self.PERIOD, self.PERIOD) - 0.5 * self.PERIOD, 6)
        levels = np.unique(keys)
        if not np.array_equal(np.diff(levels), np.ones(len(levels) - 1)):
            raise Invalid("the normal groups are not consecutive in q")
        self.group = np.searchsorted(levels, keys)
        self.groups = len(levels)
        self.group_q = levels + centre

    def _csr(self, a: dict, name: str, shape: tuple, scale: np.ndarray | None = None) -> csr_matrix:
        weights = np.asarray(a[f"{name}_weights"], dtype=np.float64)
        if scale is not None:
            weights = weights * scale
        return csr_matrix((weights, a[f"{name}_indices"].astype(np.int64), a[f"{name}_offsets"].astype(np.int64)), shape=shape)

    def _direct(self, a: dict) -> None:
        q = self.q
        self.D = self._csr(a, "D", (self.states, self.states))
        arm = np.asarray(a["direct_arm"], dtype=np.float64).reshape(-1, 3)
        donor_cell = a["D_indices"].astype(np.int64) // q
        with_gradient = self.is_wall[donor_cell].astype(np.float64)
        self.D_arm = [self._csr(a, "D", (self.states, self.states), arm[:, axis] * with_gradient) for axis in range(3)]
        offsets = a["D_offsets"].astype(np.int64)
        count = np.diff(offsets)
        first = np.minimum(offsets[:-1], len(donor_cell) - 1)
        state_cell = np.arange(self.states) // q
        single = (count == 1) & (np.asarray(a["D_weights"])[first] == 1.0)
        donor_is_wall = np.zeros(self.states, dtype=bool)
        np.logical_or.at(donor_is_wall, np.repeat(np.arange(self.states), count), self.is_wall[donor_cell])
        native = single & ~donor_is_wall
        owned = (self.is_wall[state_cell] & ~native) | (~self.is_wall[state_cell] & donor_is_wall)
        if (~self.is_wall[state_cell] & ~native & ~owned).any():
            raise Invalid("a state of an ordinary cell is neither native nor owned")

    def _gradient(self, a: dict) -> None:
        cells = a["gradient_cells"].astype(np.int64)
        offsets = a["gradient_offsets"].astype(np.int64)
        index = a["gradient_indices"].astype(np.int64)
        weight = np.asarray(a["gradient_weights"], dtype=np.float64)
        if len(offsets) != 3 * len(cells) + 1 or not np.array_equal(index[offsets[0:-1:3]], cells):
            raise Invalid("the first stencil entry of a gradient cell is not the cell itself")
        segment = np.repeat(np.arange(3 * len(cells)), np.diff(offsets))
        own = cells[segment // 3]
        self.gradient = []
        for axis in range(3):
            use = segment % 3 == axis
            matrix = coo_matrix((weight[use], (own[use], index[use])), shape=(self.cells, self.cells)).tocsr()
            row_sum = np.asarray(matrix.sum(axis=1)).ravel()
            self.gradient.append((matrix - coo_matrix((row_sum, (np.arange(self.cells), np.arange(self.cells))), shape=matrix.shape)).tocsr())

    def _rows(self, a: dict) -> None:
        self.rows = len(a["row_direction"])
        self.facets = len(a["area"])
        facet_offsets = a["facet_offsets"].astype(np.int64)
        self.facet_of_row = np.repeat(np.arange(self.facets), np.diff(facet_offsets))
        self.row_direction = a["row_direction"].astype(np.int64)
        self.c_row = self.c[self.row_direction]
        self.w_row = self.w[self.row_direction]
        self.energy_row = 0.5 * (self.c_row * self.c_row).sum(axis=1)
        self.prism = np.asarray(a["prism_volume"], dtype=np.float64)
        self.phi = np.asarray(a["friction_rows"], dtype=np.float64).reshape(self.rows, 3)
        self.normal = np.asarray(a["normal"], dtype=np.float64).reshape(self.facets, 3)
        if np.any(self.lattice.shell[self.row_direction] == 0):
            raise Invalid("a wall row has the rest direction")
        self.facet_volume = np.bincount(self.facet_of_row, weights=self.prism, minlength=self.facets)
        if np.any(self.facet_volume <= 0.0):
            raise Invalid("a facet has no row volume")
        self.G = self._csr(a, "G", (self.rows, self.states))
        arm = np.asarray(a["prism_arm"], dtype=np.float64).reshape(-1, 3)
        self.G_arm = [self._csr(a, "G", (self.rows, self.states), arm[:, axis]) for axis in range(3)]
        self.g_row = np.repeat(np.arange(self.rows), np.diff(a["G_offsets"].astype(np.int64)))
        self.g_state = a["G_indices"].astype(np.int64)
        self.g_cell = self.g_state // self.q
        self.g_weight = np.asarray(a["G_weights"], dtype=np.float64)
        self.g_arm = arm
        self.g_total = np.bincount(self.g_row, weights=self.g_weight, minlength=self.rows)
        if not self.is_wall[self.g_cell].all():
            raise Invalid("a gather donor is not a wall-list cell")
        self.S = self._csr(a, "S", (self.facets, self.cells))
        if not self.is_wall[a["S_indices"].astype(np.int64)].all():
            raise Invalid("a sample cell is not a wall-list cell")

    def _scatter(self, a: dict) -> None:
        q = self.q
        self.R = self._csr(a, "R", (self.states, self.rows))
        self.r_state = np.repeat(np.arange(self.states), np.diff(a["R_offsets"].astype(np.int64)))
        self.r_row = a["R_indices"].astype(np.int64)
        cell, direction = self.r_state // q, self.r_state % q
        if not np.array_equal(self.row_direction[self.r_row], direction):
            raise Invalid("a scatter entry has another direction than its row")
        if not self.is_wall[cell].all():
            raise Invalid("a scatter receiver is not a wall-list cell")
        key_g = self.g_row * self.states + self.g_state
        order = np.argsort(key_g)
        wanted = self.r_row * self.states + cell * q + self.lattice.opposite[direction]
        position = np.minimum(np.searchsorted(key_g[order], wanted), len(order) - 1)
        match = order[position]
        if not np.array_equal(key_g[match], wanted):
            raise Invalid("a scatter entry has no gather entry of the same prism part")
        self.r_cell, self.r_direction = cell, direction
        self.r_part = self.g_weight[match] / self.volume[cell]
        self.r_arm = self.g_arm[match]

    # ---- the update ---------------------------------------------------------------------------------------------------------
    def collide(self, f: np.ndarray) -> np.ndarray:
        """Task-81 RR: p = feq3 + (1 - omega) fneq_RR(A) + Phi/2, u = (sum c f + F/2)/rho, A = sum H2 (f + Phi/2) - rho u u."""
        rho = f.sum(axis=1)
        u = (f @ self.c + 0.5 * self.force[None, :]) / rho[:, None]
        feq, phi = self.eq.parts(rho, u, self.force)
        A = self.eq.stress(f, phi, rho, u)
        return feq + (1.0 - self.omega) * self.eq.non_equilibrium(A, u) + 0.5 * phi

    def fields(self, p: np.ndarray) -> tuple:
        rho = p.sum(axis=1)
        return rho, (p @ self.c - 0.5 * self.force[None, :]) / rho[:, None]

    def _outgoing_shape(self, c, w, rho, u, arm, grad_rho, grad_u) -> np.ndarray:
        """E = w {[rho + d.L rho] B(u) + rho (dB/du) . [(L u) d]} (feq3 Taylor, #1754)."""
        cu = (c * u).sum(axis=1)
        B = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (u * u).sum(axis=1) + 4.5 * (cu ** 3 - cu * (u * u).sum(axis=1))
        d_rho = (arm * grad_rho).sum(axis=1)
        d_u = (grad_u * arm[:, None, :]).sum(axis=2)
        return w * (rho * B + d_rho * B + rho * (self.eq.slope(c, u) * d_u).sum(axis=1))

    def transport(self, p: np.ndarray) -> tuple:
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
        u_s = self.S @ v
        u_s = u_s - self.normal * (u_s * self.normal).sum(axis=1)[:, None]
        rho_r, u_r = rho_s[self.facet_of_row], u_s[self.facet_of_row]
        cu = (self.c_row * u_r).sum(axis=1)
        even = 2.0 * self.w_row * rho_r * (1.0 + 4.5 * cu * cu - 1.5 * (u_r * u_r).sum(axis=1))
        ret = self.prism * even - gathered - self.friction * rho_r * (self.phi * u_r).sum(axis=1)
        delta = np.bincount(self.facet_of_row, weights=gathered - ret, minlength=self.facets) / self.facet_volume   # ONE per facet
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
        return new, WallPass(gathered, outgoing, delta, energy, (impulse * self.normal).sum(axis=1))

    def step(self, f: np.ndarray) -> np.ndarray:
        return self.transport(self.collide(f))[0]

    def mass(self, f: np.ndarray) -> float:
        return float(self.volume @ f.sum(axis=1))

    def rest(self, mass: float) -> np.ndarray:
        """The kinetic rest state rho_bar w_i with the given volume-weighted mass."""
        return np.tile((mass / float(self.volume.sum())) * self.w, (self.cells, 1))
