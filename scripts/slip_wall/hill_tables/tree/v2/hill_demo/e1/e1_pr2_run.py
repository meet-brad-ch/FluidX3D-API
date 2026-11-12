"""V2-PR-02 (E1, task 111): 32-update reference feedback check of the positive wall-return projection. Protocol
V2-PR-02_PROTOCOL_E1_2026-11-11.md. Reference only (FP64 R3 statements), no engine.

The projected transport (E2 #2601, scientist #2603): ProjectedTransport.transport_augmented is the frozen statement list of
open_terrain_model.OpenTransport95.transport_augmented with ONE marked insertion, after the exterior rows are set and before
the scatter, R and W: outgoing = projection(q0). So one corrected q feeds R q, W (q / v), the impulse, the WallPass and the open
ledger. delta stays the original additive-correction diagnostic. Exterior rows are not projected. check_copy() compares the
two sources line by line (insertion removed) at every start: any other difference is INVALID.

Screening (#2599): per-facet min q0 (minimum.reduceat) and the P1 mass/tangent equality of q0 for ALL interior facets, by
bincount. Bypass (q = q0 bits) only when min q0 >= 0 AND P1 passes (#2581). Every other interior facet goes to E1's PR-01
Projection (e1_pr_check.Projection: semismooth Newton, KKT certificate, analytic infeasibility only, #2588).

Per update n (state f^(n-1), s^(n-1), E^(n-1)): E admissible; p = collide(f), p_E = feq3(E); transport; E^n from the level
n-1 fields; f^n_E = feq3(E^n). The collision of update n+1 is the next-collision check of update n; after update 32 one extra
collision covers update 32. A parent-model transport on the same (p, s) gives the instrument checks: q0 bitwise equal, and
f^n against new_R3 + R (q - q0) and s^n against s_R3 + W ((q - q0) / v) within 1e-13.

  envs/solver_poc/Scripts/python.exe -B v2/hill_demo/e1/e1_pr2_run.py --case H1|F1 --record FILE
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[2]
for folder in ("v2/open_terrain/e2", "integration/task39", "integration/task95", "integration/task99", "v2/sgs_m1/e2", "v2/sgs_m1",
               "v2/rough_wall/e2", "v2/open_terrain/e1/native", "v2/hill_demo/e1"):
    sys.path.insert(0, str(WALL / folder))
import e1_pr_check as PC  # noqa: E402  (V2-PR-01: Projection, the H1 field pins)
import open_terrain_model as OM  # noqa: E402

T95 = OM.T95            # names used by the copied statement list
Invalid = OM.Invalid
GATE = PC.GATE
UPDATES = 32
DENSITY_SCREEN = 0.05
SPEED_SCREEN = 0.3
# exceptions that are a physical inadmissible state (NOT PASSED, #2608); every other exception is INVALID
PHYSICAL_INVALID = ("a new population or pending amount is not finite", "the pending state is not one finite amount per row")
PHYSICAL_REFUSED_STAGES = ("collision", "boundary update")      # OM.Refused there = the open-boundary 7.6 refusal of the state
BEGIN = "# ---- E1 PROJECTION BEGIN"
END = "# ---- E1 PROJECTION END"
CASES = {
    "H1": {"container": "v2/hill_demo/e1/hf_inputs/hill_uniform.vwnat", "sha": "a404dc7463c9bc1a", "length": 78.0,
           "from_run": "v2/hill_demo/e1/hf_run/hf_run", "expect_inactive": False},
    "F1": {"container": "v2/open_terrain/e1/native/inputs/oc01_uniform.vwnat", "sha": "341b8c3dc6176fa6", "length": 158.0,
           "expect_inactive": True},
}


class FacetProjector:
    """Vectorized screen of every interior facet; E1's PR-01 Projection on the facets that need it. Results in self.last."""

    def __init__(self, m) -> None:
        fr = m.facet_of_row
        if not np.all(np.diff(fr) >= 0):
            raise Invalid("rows are not sorted by facet")
        counts = np.bincount(fr, minlength=m.facets)
        if np.any(counts == 0):
            raise Invalid("a facet without rows")
        self.m = m
        self.start = np.searchsorted(fr, np.arange(m.facets))
        self.bounds = np.append(self.start, m.rows)
        self.interior = ~m.layout.exterior_facet
        n = m.normal
        a = np.where((np.abs(n[:, 0]) < 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        t1 = np.cross(n, a)
        t1 = t1 / np.linalg.norm(t1, axis=1)[:, None]
        self.t1 = t1
        self.t2 = np.cross(n, t1)
        self.cnorm = np.linalg.norm(m.c_row, axis=1)
        self.last = None

    def screen(self, q0, incoming, T) -> dict:
        m = self.m
        fr = m.facet_of_row
        M = np.bincount(fr, weights=incoming, minlength=m.facets)
        sum_q0 = np.bincount(fr, weights=q0, minlength=m.facets)
        V = np.stack([np.bincount(fr, weights=m.c_row[:, k] * (q0 + incoming), minlength=m.facets) for k in range(3)], axis=1) - T
        S_a = np.bincount(fr, weights=self.cnorm * (np.abs(q0) + np.abs(incoming)), minlength=m.facets)
        mass_rel = np.abs(sum_q0 - M) / np.maximum(M, 1e-300)
        tan_rel = np.maximum(np.abs((V * self.t1).sum(axis=1)), np.abs((V * self.t2).sum(axis=1))) / np.maximum(S_a, 1e-300)
        min_q0 = np.minimum.reduceat(q0, self.start)
        rows_finite = np.logical_and.reduceat(np.isfinite(q0) & np.isfinite(incoming), self.start)
        finite = (rows_finite & np.isfinite(T).all(axis=1) & np.isfinite(M) & np.isfinite(S_a) & np.isfinite(mass_rel)
                  & np.isfinite(tan_rel) & np.isfinite(min_q0))
        nonfinite = self.interior & ~finite                    # #2608: never compared, never bypassed: a physical non-finite state
        ok = self.interior & finite
        p1_fail = ok & ((mass_rel > GATE) | (tan_rel > GATE))
        need = ok & ((min_q0 < 0.0) | p1_fail)
        return {"mass_rel": mass_rel, "tan_rel": tan_rel, "min_q0": min_q0, "p1_fail": p1_fail, "need": need, "nonfinite": nonfinite,
                "ok": ok}

    def project(self, q0, incoming, rho_s, u_s, K_row) -> np.ndarray:
        m = self.m
        T = -(K_row[self.start] * rho_s * m.area / 3.0)[:, None] * u_s
        sc = self.screen(q0, incoming, T)
        q = q0.copy()
        ok = sc["ok"]
        out = {"q0": q0, "active": [], "p1_failures": [], "infeasible": [], "unresolved": [], "audit": [],
               "nonfinite_facets": [int(a) for a in np.flatnonzero(sc["nonfinite"])[:100]],
               "nonfinite_count": int(sc["nonfinite"].sum()),
               "screen_mass_rel_max": float(sc["mass_rel"][ok].max()) if ok.any() else float("nan"),
               "screen_tan_rel_max": float(sc["tan_rel"][ok].max()) if ok.any() else float("nan"),
               "min_q0": float(sc["min_q0"][ok].min()) if ok.any() else float("nan")}
        for a in np.flatnonzero(sc["need"]):
            rows = np.arange(self.bounds[a], self.bounds[a + 1])
            if sc["p1_fail"][a]:
                out["p1_failures"].append({"facet": int(a), "mass_rel": float(sc["mass_rel"][a]), "tangent_rel": float(sc["tan_rel"][a])})
            pr = PC.Projection(m.prism[rows], m.c_row[rows], q0[rows], incoming[rows], m.normal[a], T[a])
            sol = pr.solve()
            cert = pr.certificate(sol["q"], sol["lam"])
            if cert["problems"]:
                inf = pr.infeasibility()
                target = out["infeasible"] if inf["certified"] else out["unresolved"]
                target.append({"facet": int(a), "problems": cert["problems"], "reason": inf["reason"]})
                continue
            q[rows] = sol["q"]
            out["active"].append(self.facet_record(a, rows, pr, sol, cert, incoming))
        self.last = out
        return q

    def facet_record(self, a, rows, pr, sol, cert, incoming) -> dict:
        m = self.m
        n = m.normal[a]
        mu = (sol["q"] - pr.q0) / pr.v + sol["lam"] @ pr.A
        active_rows = sol["q"] == 0.0
        Jn0 = float(n @ (m.c_row[rows].T @ (pr.q0 + incoming[rows])))
        dJn = float(n @ (m.c_row[rows].T @ (sol["q"] - pr.q0)))
        return {"facet": int(a), "rows": [int(rows[0]), int(rows[-1])], "iters": sol["iters"], "active_rows": cert["active_rows"],
                "mass_rel": cert["mass_rel"], "tangent_rel": cert["tangent_rel"], "stationarity_free": cert["stationarity_free"],
                "complementarity": cert["complementarity"], "dual_scale": cert["dual_scale"],
                "mu_min_active": float(mu[active_rows].min()) if active_rows.any() else None,
                "Jn0": Jn0, "dJn": dJn, "q0_min": float(pr.q0.min())}


class ProjectedTransport(OM.OpenTransport95):
    projector: FacetProjector | None = None

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
        # ---- E1 PROJECTION BEGIN (task 111): the one corrected q feeds R, W, the impulse, the WallPass and the open ledger ----
        K_row = np.broadcast_to(np.asarray(self.friction, dtype=np.float64), (self.rows,))
        outgoing = self.projector.project(outgoing, incoming, rho_s, u_s, K_row)
        # ---- E1 PROJECTION END ----------------------------------------------------------------------------------------------
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


class ProjectedOpenModel(OM.OpenRoughSgsTerrainModel27, ProjectedTransport):
    """The open rough SGS reference with the projected transport (open_x = True only)."""

    def __init__(self, container: Path) -> None:
        super().__init__(container, np.zeros(3), PC.SGS.SgsClosure(PC.K_SGS), None, True)
        self.projector = FacetProjector(self)


EXPECTED_ORDER = ("ProjectedOpenModel", "OpenRoughSgsTerrainModel27", "RoughSgsTerrainModel27", "SgsTerrainModel27",
                  "SgsWallReferenceModel27", "ProjectedTransport", "OpenTransport95", "TerrainModel27", "WallReferenceModel27",
                  "Model27", "object")


def check_copy() -> None:
    """The copied statement list equals the parent's, line by line, once the marked insertion is removed."""
    parent = inspect.getsource(OM.OpenTransport95.transport_augmented).splitlines()
    mine = inspect.getsource(ProjectedTransport.transport_augmented).splitlines()
    kept = []
    skip = False
    for line in mine:
        if BEGIN in line:
            skip = True
            continue
        if END in line:
            skip = False
            continue
        if not skip:
            kept.append(line)
    if kept != parent:
        first = next((i for i, (x, y) in enumerate(zip(kept, parent)) if x != y), min(len(kept), len(parent)))
        raise Invalid(f"the copied transport differs from OpenTransport95 at line {first}")
    order = tuple(k.__name__ for k in ProjectedOpenModel.__mro__)
    if order != EXPECTED_ORDER:
        raise Invalid(f"method resolution order {order}")


class Trajectory:
    """One case: the projected model and its state, the parent model for the instrument checks, and F1's plain R3 trajectory."""

    def __init__(self, cid: str) -> None:
        cfg = CASES[cid]
        self.cid = cid
        self.case = WALL / cfg["container"]
        self.sha = hashlib.sha256(self.case.read_bytes()).hexdigest()
        if not self.sha.startswith(cfg["sha"]):
            raise Invalid(f"{cid}: container hash {self.sha[:16]} is not the pinned {cfg['sha']}")
        self.expect_inactive = cfg["expect_inactive"]
        self.m = ProjectedOpenModel(self.case)
        self.ref = OM.OpenRoughSgsTerrainModel27(self.case, np.zeros(3), PC.SGS.SgsClosure(PC.K_SGS), None, True)
        self.boundary = self.make_boundary(cfg["length"])
        if cfg.get("from_run"):
            self.f, self.s, self.E = self.load_run(cfg["from_run"])
        else:
            self.f = np.asarray(self.m.initial, dtype=np.float64).copy()
            self.s = np.asarray(self.m.initial_pending, dtype=np.float64).copy()
            self.E = self.m.initial_e_state(1.0, np.array([PC.U, 0.0, 0.0]))
        self.plain = None
        if self.expect_inactive:
            self.plain = {"f": self.f.copy(), "s": self.s.copy(), "E": self.E, "boundary": self.make_boundary(cfg["length"])}

    def make_boundary(self, length: float):
        L = self.m.layout
        return OM.CharacteristicBoundary(L, OM.OutletRelaxation(0.58, PC.U * math.sqrt(3.0), length, 1.0),
                                         OM.InletTarget(np.array([PC.U, 0.0, 0.0]), 0.0, True))

    def load_run(self, run: str) -> tuple:
        for name, pin in PC.H1_FIELDS.items():
            got = hashlib.sha256((WALL / run / name).read_bytes()).hexdigest()
            if not got.startswith(pin):
                raise Invalid(f"H1 field {name} hash {got[:16]} is not the pinned {pin}")
        got = hashlib.sha256((WALL / PC.H1_GEOMETRY[0]).read_bytes()).hexdigest()
        if not got.startswith(PC.H1_GEOMETRY[1]):
            raise Invalid("H1 geometry container hash mismatch")
        import e1_ob_oracle as OB  # noqa: PLC0415
        x = OB.CheckpointRun(self.case, WALL / run, self.ref)
        f, _ = x.level(0)
        rho, u = x.open_state(0, self.m.layout.e_cells)
        return np.asarray(f, dtype=np.float64).copy(), np.asarray(x.pending_level(0), dtype=np.float64).copy(), OM.BoundaryState(rho, u)


def collide_open(m, f, E, boundary) -> tuple:
    E = boundary.admissible(E, "current")
    p = m.collide(f)
    p[m.layout.e_cells] = m.e_populations(E)
    return p, E


def location(m, values: np.ndarray) -> dict:
    k = int(np.argmin(values))
    cell = k // values.shape[1] if values.ndim == 2 else k
    return {"value": float(values.min()), "cell": int(cell), "volume": float(m.volume[cell]), "is_E": bool(m.layout.is_e[cell])}


class Runner:
    """The 32 updates of one case, the per-update record and the two verdicts."""

    def __init__(self, tr: Trajectory) -> None:
        self.tr = tr
        m = tr.m
        self.M0 = m.open_mass(tr.f, tr.s)
        self.P0 = float(np.linalg.norm(m.open_momentum(tr.f, tr.s)))
        self.rows = []
        self.audit = []
        self.stop = None
        self.completed = 0
        self.stage = "start"
        self.persist = {}
        self.persist_max = 0
        self.ever_active = set()

    def update(self, n: int) -> dict:
        tr, m = self.tr, self.tr.m
        L = m.layout
        t0 = time.perf_counter()
        self.stage = "collision"
        p, E = collide_open(m, tr.f, tr.E, tr.boundary)
        row = {"update": n, "collision_of_level": n - 1, "post_collision_min": location(m, p), "post_collision_finite": bool(np.isfinite(p).all())}
        self.stage = "projected transport"
        new, s_next, wall = m.transport_augmented(p, tr.s)
        led = m.last_open
        pj = m.projector.last
        rho_f = tr.f.sum(axis=1)
        u_f = (tr.f @ m.c) / rho_f[:, None]
        self.stage = "boundary update"
        E_next = tr.boundary.update(E, rho_f, u_f)
        new[L.e_cells] = m.e_populations(E_next)
        t_step = time.perf_counter() - t0
        self.stage = "records and instrument checks"
        row.update(self.projection_row(pj, wall, n))
        row.update(self.instrument_row(p, tr.s, new, s_next, pj, wall, E_next))
        row.update(self.ledger_row(p, tr.f, tr.s, new, s_next, led))
        row.update(self.screen_row(new))
        row["min_new"] = location(m, new)
        row["new_finite"] = bool(np.isfinite(new).all())
        row["pending_finite"] = bool(np.isfinite(s_next).all())
        row["min_pending"] = float(s_next.min())
        row["min_pending_row"] = int(np.argmin(s_next))
        if self.tr.plain is not None:
            row.update(self.plain_row(new, s_next))
        row["seconds_update"] = t_step
        row["seconds_with_checks"] = time.perf_counter() - t0
        self.pending_state = (p, new, s_next, E_next)
        return row

    def projection_row(self, pj, wall, n: int) -> dict:
        m = self.tr.m
        act = pj["active"]
        ids = [a["facet"] for a in act]
        for facet in ids:
            self.persist[facet] = self.persist.get(facet, 0) + 1
            self.persist_max = max(self.persist_max, self.persist[facet])
        for facet in [f for f in self.persist if f not in ids]:
            self.persist[facet] = 0
        self.ever_active.update(ids)
        for a in act:
            self.audit.append({"update": n, **a})
        Jn_all = (wall.impulse * m.normal).sum(axis=1)
        interior = m.projector.interior
        sum_abs_dJn = float(sum(abs(a["dJn"]) for a in act))
        sum_abs_Jn = float(np.abs(Jn_all[interior]).sum())
        mu_active = [a["mu_min_active"] for a in act if a["mu_min_active"] is not None]
        wall_rho = np.concatenate([np.asarray(wall.rho_s)[interior], np.asarray(wall.rho_w)[interior]])
        return {"wall_density_min": float(wall_rho.min()), "wall_density_finite": bool(np.isfinite(wall_rho).all()),
                "nonfinite_facet_count": pj["nonfinite_count"], "nonfinite_facets": pj["nonfinite_facets"],
                "active_facets": len(act), "active_rows": int(sum(a["active_rows"] for a in act)), "min_q0": pj["min_q0"],
                "min_q": float(wall.outgoing[np.isin(m.facet_of_row, np.flatnonzero(interior))].min()),
                "screen_mass_rel_max": pj["screen_mass_rel_max"], "screen_tan_rel_max": pj["screen_tan_rel_max"],
                "p1_failures": pj["p1_failures"], "infeasible_certified": pj["infeasible"], "unresolved": pj["unresolved"],
                "kkt_mass_rel_max": max((a["mass_rel"] for a in act), default=0.0),
                "kkt_tangent_rel_max": max((a["tangent_rel"] for a in act), default=0.0),
                "kkt_stationarity_max_over_scale": max((a["stationarity_free"] / a["dual_scale"] for a in act), default=0.0),
                "mu_min_active": min(mu_active) if mu_active else None, "newton_iterations_max": max((a["iters"] for a in act), default=0),
                "sum_abs_dJn": sum_abs_dJn, "sum_abs_Jn_interior": sum_abs_Jn,
                "dJn_rel_max": max((abs(a["dJn"]) / abs(a["Jn0"]) if a["Jn0"] != 0.0 else float("inf") for a in act), default=0.0),
                "persistence_max": self.persist_max, "facets_ever_active": len(self.ever_active)}

    def instrument_row(self, p, s, new, s_next, pj, wall, E_next) -> dict:
        m, ref = self.tr.m, self.tr.ref
        L = m.layout
        new_r, s_r, wall_r = ref.transport_augmented(p, s)
        q0, q = pj["q0"], wall.outgoing
        cross = new_r + (ref.R @ (q - q0)).reshape(m.cells, m.q)
        cross[L.e_cells] = m.e_populations(E_next)
        pend_cross = s_r + ref.W @ ((q - q0) / ref.prism)
        inactive_rows = ~np.isin(m.facet_of_row, [a["facet"] for a in pj["active"]])
        return {"q0_bitwise_parent": bool(np.array_equal(q0, wall_r.outgoing)),
                "q_bits_inactive_rows": bool(np.array_equal(q[inactive_rows], q0[inactive_rows])),
                "assembly_vs_cross": float(np.abs(new - cross).max()), "pending_vs_cross": float(np.abs(s_next - pend_cross).max())}

    def ledger_row(self, p, f, s, new, s_next, led) -> dict:
        m = self.tr.m
        B = m.open_mass(new, s_next) - m.open_mass(p, s) - (led.phi_in - led.phi_out)
        BP = m.open_momentum(new, s_next) - m.open_momentum(p, s) - (led.interior_impulse + led.pi_in - led.pi_out)
        C = m.open_mass(p, s) - m.open_mass(f, s)
        CP = m.open_momentum(p, s) - m.open_momentum(f, s)
        return {"ledger_B_mass_rel": abs(B) / self.M0, "ledger_B_momentum_rel": float(np.abs(BP).max()) / self.P0,
                "collision_C_mass_rel": abs(C) / self.M0, "collision_C_momentum_rel": float(np.abs(CP).max()) / self.P0,
                "open_mass": m.open_mass(new, s_next), "open_momentum": m.open_momentum(new, s_next).tolist(),
                "phi_in": led.phi_in, "phi_out": led.phi_out, "interior_impulse": led.interior_impulse.tolist()}

    def screen_row(self, new) -> dict:
        m = self.tr.m
        rho = new.sum(axis=1)
        dep = np.abs(rho - 1.0)
        speed = np.linalg.norm((new @ m.c) / rho[:, None], axis=1)
        kd = int(np.argmax(dep))
        ks = int(np.argmax(speed))
        return {"density_departure_max": float(dep[kd]), "density_cell": kd, "density_cell_volume": float(m.volume[kd]),
                "density_rho": float(rho[kd]), "speed_max": float(speed[ks]), "speed_cell": ks}

    def plain_row(self, new, s_next) -> dict:
        tr = self.tr
        pl = tr.plain
        f_r, s_r, E_r, _ = tr.ref.open_step(pl["f"], pl["s"], pl["E"], pl["boundary"])
        pl["f"], pl["s"], pl["E"] = f_r, s_r, E_r
        return {"plain_R3_bitwise_f": bool(np.array_equal(new, f_r)), "plain_R3_bitwise_s": bool(np.array_equal(s_next, s_r)),
                "plain_R3_max_abs_f": float(np.abs(new - f_r).max()), "plain_R3_max_abs_s": float(np.abs(s_next - s_r).max())}

    def failures(self, row: dict) -> tuple:
        """(stop reasons that make the feedback verdict NOT PASSED, instrument reasons that make it INVALID)."""
        bad = []
        if not row["post_collision_finite"] or row["post_collision_min"]["value"] < 0.0:
            bad.append(f"post-collision of level {row['collision_of_level']}")
        if not row["new_finite"] or row["min_new"]["value"] < 0.0:
            bad.append("negative or non-finite population")
        if not row["pending_finite"] or row["min_pending"] < 0.0:
            bad.append("negative or non-finite pending")
        scalars = [row["ledger_B_mass_rel"], row["ledger_B_momentum_rel"], row["collision_C_mass_rel"], row["collision_C_momentum_rel"],
                   row["density_departure_max"], row["speed_max"], row["assembly_vs_cross"], row["pending_vs_cross"], row["min_q"],
                   row["min_q0"], row["screen_mass_rel_max"], row["screen_tan_rel_max"], row["kkt_mass_rel_max"],
                   row["kkt_tangent_rel_max"], row["kkt_stationarity_max_over_scale"], row["sum_abs_dJn"], row["sum_abs_Jn_interior"]]
        if row["mu_min_active"] is not None:
            scalars.append(row["mu_min_active"])
        if not all(math.isfinite(x) for x in scalars):
            bad.append("non-finite ledger, screen or cross-check value")
        if row["nonfinite_facet_count"]:
            bad.append("non-finite q0, incoming or target on an interior facet")
        if not row["wall_density_finite"] or not row["wall_density_min"] > 0.0:
            bad.append("non-finite or nonpositive wall density")
        if row["infeasible_certified"]:
            bad.append("certified infeasible facet")
        if row["p1_failures"]:
            bad.append("P1 failure of q0")
        if row["min_q"] < 0.0:
            bad.append("negative interior return")
        if max(row["ledger_B_mass_rel"], row["ledger_B_momentum_rel"], row["collision_C_mass_rel"], row["collision_C_momentum_rel"]) > GATE:
            bad.append("ledger")
        if self.tr.expect_inactive and row["active_facets"] != 0:
            bad.append("F1 activation")
        if "plain_R3_max_abs_f" in row and max(row["plain_R3_max_abs_f"], row["plain_R3_max_abs_s"]) > GATE:
            bad.append("F1 differs from plain R3 beyond 1e-13")
        inst = []
        if row["unresolved"]:
            inst.append("facet without a KKT or infeasibility certificate")
        if not (row["q0_bitwise_parent"] and row["q_bits_inactive_rows"]):
            inst.append("q0 or inactive q bits differ from the parent transport")
        if max(row["assembly_vs_cross"], row["pending_vs_cross"]) > GATE:
            inst.append("assembly cross-check beyond 1e-13")
        return bad, inst

    def run(self, record: Path) -> dict:
        tr = self.tr
        t_all = time.perf_counter()
        for n in range(1, UPDATES + 1):
            try:
                row = self.update(n)
            except Exception as error:  # noqa: BLE001  (#2606, #2608: keep the state, stage and error for diagnosis)
                text = f"exception in {self.stage}: {type(error).__name__}: {error}"
                physical = ((isinstance(error, OM.Refused) and self.stage in PHYSICAL_REFUSED_STAGES)
                            or (type(error) is Invalid and str(error) in PHYSICAL_INVALID))
                self.stop = {"update": n, "not_passed": [text] if physical else [], "invalid": [] if physical else [text]}
                self.save_state_in(record)
                break
            self.rows.append(row)
            bad, inst = self.failures(row)
            p, new, s_next, E_next = self.pending_state
            if bad or inst:
                self.stop = {"update": n, "not_passed": bad, "invalid": inst}
                self.save_stop(record, p, new, s_next, E_next)
                break
            tr.f, tr.s, tr.E = new, s_next, E_next
            self.completed = n
        final = None
        if self.stop is None:
            self.stage = "collision"
            try:
                p, _ = collide_open(tr.m, tr.f, tr.E, tr.boundary)
            except OM.Refused as error:
                self.stop = {"update": UPDATES, "not_passed": [f"exception in final collision: Refused: {error}"], "invalid": []}
                self.save_state_in(record)
                return self.record(None, time.perf_counter() - t_all)
            final = {"collision_of_level": UPDATES, "min": location(tr.m, p), "finite": bool(np.isfinite(p).all())}
            if not final["finite"] or final["min"]["value"] < 0.0:
                self.stop = {"update": UPDATES, "not_passed": [f"post-collision of level {UPDATES}"], "invalid": []}
                self.save_state_in(record, p)
        return self.record(final, time.perf_counter() - t_all)

    def save_stop(self, record: Path, p, new, s_next, E_next) -> None:
        tr = self.tr
        path = record.with_name(record.stem + "_stop.npz")
        np.savez_compressed(path, f_in=tr.f, s_in=tr.s, E_in_rho=tr.E.rho, E_in_u=tr.E.u, p=p, new=new, s_next=s_next,
                            E_next_rho=E_next.rho, E_next_u=E_next.u)
        if self.rows and self.rows[-1]["unresolved"]:
            self.stop["label"] = ("new and s_next keep the ORIGINAL q0 rows on the unresolved facets: failure evidence only, "
                                  "not a valid projected step (#2608)")
        self.stop["file"] = path.name
        self.stop["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save_state_in(self, record: Path, p=None) -> None:
        """The state the failing step read (and its collision when there is one)."""
        tr = self.tr
        path = record.with_name(record.stem + "_stop.npz")
        arrays = {"f_in": tr.f, "s_in": tr.s, "E_in_rho": tr.E.rho, "E_in_u": tr.E.u}
        if p is not None:
            arrays["p"] = p
        np.savez_compressed(path, **arrays)
        self.stop["file"] = path.name
        self.stop["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def record(self, final, seconds: float) -> dict:
        tr = self.tr
        rows = self.rows
        inv = [x for x in (self.stop or {}).get("invalid", [])]
        bad = [x for x in (self.stop or {}).get("not_passed", [])]
        feedback = "INVALID" if inv else ("NOT PASSED" if bad else "PASS")
        exceed = [r["update"] for r in rows if r["density_departure_max"] > DENSITY_SCREEN or r["speed_max"] > SPEED_SCREEN]
        screens = "NOT PASSED" if exceed else "PASS"
        return {"case": tr.cid, "container_sha256": tr.sha, "updates_completed": self.completed,
                "M0": self.M0, "P0_norm": self.P0,
                "verdict": feedback, "detail": f"feedback {feedback}; HF-01 screens {screens} (reported separately, #2595)",
                "verdict_feedback": feedback, "verdict_hf01_screens": screens, "screen_first_exceedance_update": exceed[0] if exceed else None,
                "stop": self.stop, "final_extra_collision": final, "seconds_total": seconds,
                "seconds_update_mean": float(np.mean([r["seconds_update"] for r in rows])) if rows else None,
                "facets_ever_active": len(self.ever_active), "persistence_max": self.persist_max, "rows": rows,
                "note": "verdict_feedback: positivity/constraints/ledger of the corrected operator over the prefix only; "
                        "verdict_hf01_screens: the original HF-01 density <= 0.05 and speed <= 0.3 screens. A feedback PASS is not "
                        "hill acceptance. Later updates are states of the corrected operator, not the original trajectory."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--record", required=True, type=Path)
    args = ap.parse_args()
    args.record.parent.mkdir(parents=True, exist_ok=True)
    check_copy()
    tr = Trajectory(args.case)
    s0 = tr.s
    runner = Runner(tr)
    rec = runner.run(args.record)
    rec["initial_pending_min"] = float(s0.min())
    rec["initial_pending_nonzero"] = int(np.count_nonzero(s0))
    if runner.audit:
        npz = args.record.with_suffix(".npz")
        keys = [k for k in runner.audit[0] if k != "rows"]
        np.savez_compressed(npz, **{k: np.asarray([a[k] if a[k] is not None else np.nan for a in runner.audit]) for k in keys},
                            rows=np.asarray([a["rows"] for a in runner.audit]))
        rec["audit_npz_file"] = npz.name
        rec["audit_npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    args.record.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    last = rec["rows"][-1] if rec["rows"] else {}
    print(f"{args.case}: feedback {rec['verdict_feedback']}, screens {rec['verdict_hf01_screens']}, updates {rec['updates_completed']}, "
          f"stop {rec['stop']}, last min_new {last.get('min_new')}, seconds {rec['seconds_total']:.1f}")
    return 0 if rec["verdict_feedback"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Invalid as error:
        print(f"INVALID: {error}")
        sys.exit(2)
