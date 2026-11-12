"""V2-PR-01 (E1, task 110): reference-only check of the positive wall-return projection (protocol
V2-PR-01_PROTOCOL_E1_2026-11-11.md). No engine. One R3 update from a frozen state, then E1's own projection per interior
facet and E1's reconstruction of the full update from its parts.

Derivation: v2/hill_demo/e2/TASK110_POSITIVE_RETURN_DERIVATION_E2_2026-11-11.md as corrected in 4f1cac5f (#2581).
Outcomes (#2581): a certified infeasibility on a registered state is NOT PASSED (domain failure); a facet without a KKT
certificate and without an infeasibility certificate is INVALID (unresolved), never proof of infeasibility. The inactive
bypass applies only when q0 >= 0 AND its physical equality residual passes P1; b is never altered. The independent assembly
is compared with R3 within the registered envelope 1e-13 (not bitwise); bitwise equality is required only within the same
assembly (q == q0 and new + R (q - q0) == new on inactive states).
Projection (E1, independent of E2's development code, #2579): q = max(0, q0 - v (lambda_0 + lambda_t . P c)) with lambda from
a semismooth Newton method on the 3 dual equations A q(lambda) = b. Then an explicit KKT certificate that REJECTS (INVALID)
on: non-finite values; primal |A q - b| above the gate; a row with q > 0 and a stationarity residual; a row with q = 0 and a
negative dual mu; no convergence. b is the PHYSICAL data: b_0 = M = sum I, b_t = T - P sum c I with
T = -K_a rho_s (A_a / 3) u_s (K_a: the facet friction R3 used on this update). q0's defect |A q0 - b| is reported.
Full update (#2579): new_q = direct + direct_increment + R q + scatter, every term rebuilt by E1 from the post-collision state
with the reference statements. Cross-check: new_q against new_R3 + R (q - q0). pending_q = W (q / v).

  envs/solver_poc/Scripts/python.exe -B v2/hill_demo/e1/e1_pr_check.py --state ID --record FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[2]
for folder in ("v2/open_terrain/e2", "integration/task39", "integration/task95", "integration/task99", "v2/sgs_m1/e2", "v2/sgs_m1",
               "v2/rough_wall/e2", "v2/open_terrain/e1/native"):
    sys.path.insert(0, str(WALL / folder))
import open_terrain_model as OM  # noqa: E402
import sgs_model as SGS  # noqa: E402

Invalid = OM.Invalid
K_SGS = float.fromhex("0x1.8746d2p-1")
U = float.fromhex("0x1.999999999999ap-5")
GATE = 1e-13
PRESERVE = 1e-12


class Projection:
    """E1's projection of one facet's return. Rows: v (prism), c (row directions), q0, I. Facet: n, T."""

    MAX_ITER = 100

    def __init__(self, v, c, q0, inc, n, T) -> None:
        self.v, self.c, self.q0, self.inc, self.n = v, c, q0, inc, n
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        self.t = np.stack([t1, t2])
        self.A = np.vstack([np.ones(len(v)), c @ t1, c @ t2])                     # 3 x rows
        P = np.eye(3) - np.outer(n, n)
        bt = T - P @ (c.T @ inc)
        self.M = float(math.fsum(inc.tolist()))
        self.b = np.array([self.M, bt @ t1, bt @ t2])
        self.scale_t = float(np.sum(np.linalg.norm(c, axis=1) * (np.abs(q0) + np.abs(inc))))

    def q_of(self, lam):
        return np.maximum(0.0, self.q0 - self.v * (lam @ self.A))

    def solve(self) -> dict:
        lam = np.zeros(3)
        iters = 0
        for iters in range(1, self.MAX_ITER + 1):
            q = self.q_of(lam)
            phi = self.A @ q - self.b
            if np.all(np.abs(phi) <= np.array([GATE * max(self.M, 1e-300), GATE * self.scale_t, GATE * self.scale_t]) * 0.01):
                break
            free = q > 0.0
            J = -(self.A[:, free] * self.v[free]) @ self.A[:, free].T
            step = np.linalg.lstsq(J, -phi, rcond=None)[0]
            # damped step: halve until the residual norm decreases (semismooth Newton with backtracking)
            base = np.linalg.norm(phi)
            alpha = 1.0
            for _ in range(60):
                trial = lam + alpha * step
                if np.linalg.norm(self.A @ self.q_of(trial) - self.b) < base:
                    break
                alpha *= 0.5
            lam = lam + alpha * step
        q = self.q_of(lam)
        return {"q": q, "lam": lam, "iters": iters}

    def certificate(self, q, lam) -> dict:
        """KKT certificate (#2579, #2581). Scales: q_scale = max(max|q|, M); dual_scale = max(1, max|q0/v|, max|lambda|).
        Problems: non-finite; mass > 1e-13 M; tangent > 1e-13 S_a; stationarity on free rows > 1e-13 dual_scale; dual mu on
        active rows < -1e-13 dual_scale; complementarity |mu q| > 1e-13 q_scale dual_scale; q < 0."""
        mu = (q - self.q0) / self.v + lam @ self.A
        arrays_finite = bool(np.isfinite(q).all() and np.isfinite(lam).all() and np.isfinite(mu).all())
        q_scale = max(float(np.abs(q).max()) if arrays_finite else 0.0, self.M)
        dual_scale = max(1.0, float(np.abs(self.q0 / self.v).max()), float(np.abs(lam).max()) if arrays_finite else 0.0)
        mass = abs(float(math.fsum(q.tolist())) - self.M) if arrays_finite else float("nan")
        tang = float(np.abs((self.A @ q - self.b)[1:]).max()) if arrays_finite else float("nan")
        free = q > 0.0
        stat = float(np.abs(mu[free]).max()) if (arrays_finite and free.any()) else 0.0
        dual_min = float(mu[~free].min()) if (arrays_finite and (~free).any()) else 0.0
        comp = float(np.abs(mu * q).max()) if arrays_finite else float("nan")
        cert = {"finite": arrays_finite, "q_scale": q_scale, "dual_scale": dual_scale, "mass_abs": mass,
                "mass_rel": mass / max(self.M, 1e-300), "tangent_abs": tang, "tangent_rel": tang / max(self.scale_t, 1e-300),
                "stationarity_free": stat, "dual_min_active": dual_min, "complementarity": comp,
                "active_rows": int((~free).sum()), "min_q": float(q.min()) if arrays_finite else float("nan")}
        problems = []
        if not arrays_finite:
            problems.append("non-finite")
        else:
            if cert["mass_rel"] > GATE:
                problems.append("mass")
            if cert["tangent_rel"] > GATE:
                problems.append("tangent")
            if stat > GATE * dual_scale:
                problems.append("stationarity")
            if dual_min < -GATE * dual_scale:
                problems.append("dual sign")
            if comp > GATE * q_scale * dual_scale:
                problems.append("complementarity")
            if q.min() < 0.0:
                problems.append("negative q")
        cert["problems"] = problems
        return cert

    def infeasibility(self) -> dict:
        """A CERTIFIED infeasibility only (#2581, #2588): M < 0, or M = 0 with a non-zero tangent target (analytic). An LP
        phase-1 status is recorded but never counts as a certificate. Anything else is unresolved."""
        if self.M < 0.0:
            return {"certified": True, "reason": "M < 0"}
        if self.M == 0.0:
            nonzero = bool(np.any(self.b[1:] != 0.0))
            return {"certified": nonzero, "reason": "M = 0 with a non-zero tangent target" if nonzero else "M = 0, q = 0 feasible"}
        from scipy.optimize import linprog  # noqa: PLC0415
        res = linprog(np.zeros(len(self.v)), A_eq=self.A, b_eq=self.b, bounds=[(0, None)] * len(self.v), method="highs")
        # #2588: an LP status 2 is a numerical infeasibility REPORT, not an independently checked Farkas certificate:
        # recorded as unresolved (INVALID), never as a certified infeasibility
        return {"certified": False, "reason": f"linprog status {res.status} (numerical report, not certified): {res.message}"}


class State:
    """A frozen state: the model, the input populations, pending, and (open) the E state and boundary."""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        cfg = STATES[sid]
        self.case = WALL / cfg["container"]
        self.sha = hashlib.sha256(self.case.read_bytes()).hexdigest()
        if not self.sha.startswith(cfg["sha"]):
            raise Invalid(f"{sid}: container hash {self.sha[:16]} is not the pinned {cfg['sha']}")
        self.open = cfg["open"]
        self.m = OM.OpenRoughSgsTerrainModel27(self.case, np.zeros(3), SGS.SgsClosure(K_SGS), None, self.open)
        if cfg.get("from_run"):
            for name, pin in H1_FIELDS.items():
                got = hashlib.sha256((WALL / cfg["from_run"] / name).read_bytes()).hexdigest()
                if not got.startswith(pin):
                    raise Invalid(f"H1 field {name} hash {got[:16]} is not the pinned {pin}")
            got = hashlib.sha256((WALL / H1_GEOMETRY[0]).read_bytes()).hexdigest()
            if not got.startswith(H1_GEOMETRY[1]):
                raise Invalid("H1 geometry container hash mismatch")
            import e1_ob_oracle as OB  # noqa: PLC0415
            x = OB.CheckpointRun(self.case, WALL / cfg["from_run"], self.m)
            self.f, _ = x.level(0)
            self.s = x.pending_level(0)
            rho, u = x.open_state(0, self.m.layout.e_cells)
            self.E = OM.BoundaryState(rho, u)
        else:
            self.f = np.asarray(self.m.initial, dtype=np.float64)
            self.s = np.asarray(self.m.initial_pending, dtype=np.float64)
            if self.open:
                self.E = self.m.initial_e_state(1.0, np.array([U, 0.0, 0.0]))
        if self.open:
            L = self.m.layout
            self.boundary = OM.CharacteristicBoundary(L, OM.OutletRelaxation(0.58, U * math.sqrt(3.0), float(cfg["length"]), 1.0),
                                                      OM.InletTarget(np.array([U, 0.0, 0.0]), 0.0, True))
        self.target = cfg.get("target")


def parts(m, p):
    """E1's reconstruction of direct, direct_increment and scatter from the post-collision state (reference statements)."""
    rho, v = m.fields(p)
    grad_p = [g @ p for g in m.gradient]
    grad_rho = np.stack([g @ rho for g in m.gradient], axis=1)
    grad_v = np.stack([g @ v for g in m.gradient], axis=2)
    flat = p.reshape(-1)
    direct = m.D @ flat
    inc = sum(m.D_arm[axis] @ grad_p[axis].reshape(-1) for axis in range(3))
    part = m._outgoing_shape(m.c_row[m.g_row], m.w_row[m.g_row], rho[m.g_cell], v[m.g_cell], m.g_arm, grad_rho[m.g_cell], grad_v[m.g_cell])
    row_mean = np.bincount(m.g_row, weights=m.g_weight * part, minlength=m.rows) / m.g_total
    entry = m._outgoing_shape(m.c[m.r_direction], m.w[m.r_direction], rho[m.r_cell], v[m.r_cell], m.r_arm, grad_rho[m.r_cell], grad_v[m.r_cell])
    scatter = np.bincount(m.r_state, weights=m.r_part * (entry - row_mean[m.r_row]), minlength=m.states)
    return direct, inc, scatter


def feq3_row(m, rows, rho, u):
    c = m.c_row[rows]
    cu = c @ u
    uu = float(u @ u)
    return m.w_row[rows] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu + 4.5 * (cu ** 3 - cu * uu))


def table_diagnostics(m, interior) -> dict:
    """#2586: the hypotheses H-a..H-d and T0 of E2's derivation (2a212e6b, Sections 5.1-5.2), measured on the tables of this
    state. Reported, not gated: a P5 failure is attributed to a table/assembly defect or to the formula with these values."""
    q = m.q
    ones_cells = np.ones(m.cells)
    grad_const = max(float(np.abs(g @ ones_cells).max()) for g in m.gradient)
    S_rows = np.asarray(m.S @ ones_cells)
    sampler = float(np.abs(S_rows[interior] - 1.0).max()) if interior.any() else 0.0
    Gc = m.G.tocoo()
    g_sum = np.bincount(Gc.row, weights=Gc.data, minlength=m.rows)
    rows_int = np.flatnonzero(interior[m.facet_of_row])
    gather_vs_prism = float(np.max(np.abs(g_sum[rows_int] - m.prism[rows_int]) / m.prism[rows_int])) if rows_int.size else 0.0
    opp = {tuple(int(v) for v in c): d for d, c in enumerate(m.c)}
    want = np.array([opp[tuple(int(-v) for v in c)] for c in m.c_row])
    wrong_dir = int(np.count_nonzero(Gc.col % q != want[Gc.row]))
    cn = (m.c_row * m.normal[m.facet_of_row]).sum(axis=1)
    t0 = float(np.max(np.abs(m.prism[rows_int] - m.area[m.facet_of_row[rows_int]] * cn[rows_int]) / m.area[m.facet_of_row[rows_int]])) if rows_int.size else 0.0
    return {"W_entries": int(m.W.nnz), "s0_all_zero": bool(np.all(np.asarray(m.initial_pending) == 0.0)),
            "gradient_on_constant_max_abs": grad_const, "sampler_row_sum_defect_max": sampler,
            "gather_row_sum_vs_prism_rel_max": gather_vs_prism, "gather_entries_not_opposite_direction": wrong_dir,
            "T0_prism_vs_area_cn_per_area_max": t0}


def check(sid: str) -> dict:
    st = State(sid)
    m = st.m
    q = m.q
    if st.open:
        st.E = st.boundary.admissible(st.E, "current")
        p = m.collide(st.f)
        p[m.layout.e_cells] = m.e_populations(st.E)
    else:
        p = m.collide(st.f)
    new, s_next, wall = m.transport_augmented(p, st.s)
    K_a = np.asarray(m.last_facet_K, dtype=np.float64)
    if st.open:
        rho_f = st.f.sum(axis=1)
        u_f = (st.f @ m.c) / rho_f[:, None]
        E_next = st.boundary.update(st.E, rho_f, u_f)
    interior = ~m.layout.exterior_facet if st.open else np.ones(m.facets, dtype=bool)
    q0_all = wall.outgoing.copy()
    q_all = q0_all.copy()
    facets = []
    defect = {"mass_rel": 0.0, "tangent_rel": 0.0}
    order = np.argsort(m.facet_of_row, kind="stable")
    bounds = np.searchsorted(m.facet_of_row[order], np.arange(m.facets + 1))
    worst = {"mass_rel": 0.0, "tangent_rel": 0.0, "stationarity_free": 0.0, "dual_min_active": 0.0}
    active, iters_max, dJn = [], 0, []
    p1_failures, infeasible, unresolved = [], [], []
    audit = {"facet": [], "row": [], "prism": [], "c_row": [], "incoming": [], "q0": [], "q": [], "mu": [], "normal": [],
             "T": [], "b": [], "lam": [], "row_facet_index": []}
    for a in np.flatnonzero(interior):
        rows = order[bounds[a]:bounds[a + 1]]
        if len(rows) == 0:
            continue
        n = m.normal[a]
        T = -K_a[a] * wall.rho_s[a] * (m.area[a] / 3.0) * wall.u_s[a]
        pr = Projection(m.prism[rows], m.c_row[rows], q0_all[rows], wall.incoming[rows], n, T)
        d_mass = abs(float(math.fsum(q0_all[rows].tolist())) - pr.M) / max(pr.M, 1e-300)
        d_tan = float(np.abs((pr.A @ q0_all[rows] - pr.b)[1:]).max()) / max(pr.scale_t, 1e-300)
        defect["mass_rel"] = max(defect["mass_rel"], d_mass)
        defect["tangent_rel"] = max(defect["tangent_rel"], d_tan)
        p1_ok = d_mass <= GATE and d_tan <= GATE
        if not p1_ok:
            p1_failures.append({"facet": int(a), "mass_rel": d_mass, "tangent_rel": d_tan})
        if q0_all[rows].min() >= 0.0 and p1_ok:
            continue                                           # identity: q0 is admissible AND satisfies the physical data
        sol = pr.solve()
        cert = pr.certificate(sol["q"], sol["lam"])
        if cert["problems"]:
            inf = pr.infeasibility()
            if inf["certified"]:
                infeasible.append({"facet": int(a), **inf})
            else:
                unresolved.append({"facet": int(a), "problems": cert["problems"], "lp": inf["reason"]})
            continue
        for k in worst:
            worst[k] = min(worst[k], cert[k]) if k == "dual_min_active" else max(worst[k], cert[k])
        worst["complementarity"] = max(worst.get("complementarity", 0.0), cert["complementarity"])
        iters_max = max(iters_max, sol["iters"])
        q_all[rows] = sol["q"]
        mu = (sol["q"] - pr.q0) / pr.v + sol["lam"] @ pr.A
        k = len(audit["facet"])
        audit["facet"].append(int(a))
        audit["normal"].append(n)
        audit["T"].append(T)
        audit["b"].append(pr.b)
        audit["lam"].append(sol["lam"])
        for name, value in (("row", rows), ("prism", pr.v), ("c_row", pr.c), ("incoming", pr.inc), ("q0", pr.q0), ("q", sol["q"]), ("mu", mu)):
            audit[name].extend(list(value))
        audit["row_facet_index"].extend([k] * len(rows))
        Jn0 = float(n @ (m.c_row[rows].T @ (q0_all[rows] + wall.incoming[rows])))
        dJn.append(float(n @ (m.c_row[rows].T @ (sol["q"] - q0_all[rows]))) / abs(Jn0) if Jn0 != 0.0 else float("inf"))
        active.append({"facet": int(a), "active_rows": cert["active_rows"], "min_q": cert["min_q"], "iters": sol["iters"]})
    # full update rebuilt from its parts, and the cross-check
    direct, inc, scatter = parts(m, p)
    new_q = (direct + inc + m.R @ q_all + scatter).reshape(m.cells, q)
    cross = new + (m.R @ (q_all - q0_all)).reshape(m.cells, q)
    pend_q = m.W @ (q_all / m.prism)
    if st.open:
        new_q[m.layout.e_cells] = m.e_populations(E_next)
        cross[m.layout.e_cells] = m.e_populations(E_next)
    rebuild_old = (direct + inc + m.R @ q0_all + scatter).reshape(m.cells, q)
    if st.open:
        rebuild_old[m.layout.e_cells] = m.e_populations(E_next)
        new_ref = new.copy()
        new_ref[m.layout.e_cells] = m.e_populations(E_next)
    else:
        new_ref = new
    post = m.collide(new_q)
    if st.open:
        post[m.layout.e_cells] = m.e_populations(E_next)
    rec = {"state": sid, "container_sha256": st.sha, "facets_checked": int(interior.sum()), "facets_active": len(active),
           "q0_defect_rel": defect, "kkt_worst": worst, "newton_iterations_max": iters_max, "active": active[:200],
           "min_q0": float(q0_all[np.isin(m.facet_of_row, np.flatnonzero(interior))].min()),
           "min_q": float(q_all[np.isin(m.facet_of_row, np.flatnonzero(interior))].min()),
           "rebuild_old_vs_R3": float(np.abs(rebuild_old - new_ref).max()), "rebuild_vs_cross": float(np.abs(new_q - cross).max()),
           "P1_failures": p1_failures[:100], "P1_failure_count": len(p1_failures), "infeasible_certified": infeasible[:100],
           "unresolved": unresolved[:100],
           "min_new_q": float(new_q.min()), "min_new_q_cell": int(np.unravel_index(np.argmin(new_q), new_q.shape)[0]),
           "min_pending_q": float(pend_q.min()) if pend_q.size else 0.0, "min_post_next_collision": float(post.min()),
           "post_finite": bool(np.isfinite(post).all()), "dJn_rel_max_abs": float(max((abs(x) for x in dJn), default=0.0))}
    rec["min_new_q_cell_volume"] = float(m.volume[rec["min_new_q_cell"]])
    rec["table_diagnostics"] = table_diagnostics(m, interior)
    post_cell = int(np.unravel_index(np.argmin(post), post.shape)[0])
    rec["min_post_next_collision_cell"] = post_cell
    rec["min_post_next_collision_cell_volume"] = float(m.volume[post_cell])
    pend_row = int(np.argmin(pend_q)) if pend_q.size else -1
    rec["min_pending_q_row"] = pend_row
    if audit["facet"]:
        rec["audit_npz"] = {name: np.asarray(value) for name, value in audit.items()}
    inactive_state = len(active) == 0
    # bitwise only within the same assembly (#2581): q == q0, and the cross-check new + R (q - q0) == new
    rec["P3_bitwise_inactive"] = bool(inactive_state and np.array_equal(q_all, q0_all) and np.array_equal(cross, new_ref))
    if st.target is not None:
        rho_t, u_t = st.target
        rows = np.flatnonzero(np.isin(m.facet_of_row, np.flatnonzero(interior)))
        expect = m.prism[rows] * feq3_row(m, rows, rho_t, np.asarray(u_t))
        rec["P5_q0_vs_equilibrium_rel"] = float(np.abs(q0_all[rows] - expect).max() / np.abs(expect).max())
        rec["P5_preserve_max_abs"] = float(np.abs(new_ref - st.f).max())
    if sid == "S3":
        rec["P9_traction_defect"] = float(m.traction_defect(wall)[interior].max())
    if unresolved:
        raise Invalid(f"{sid}: {len(unresolved)} facet(s) without a KKT certificate or an infeasibility certificate: {unresolved[:3]}")
    gates = [not infeasible, not p1_failures, rec["min_q"] >= 0.0, rec["min_new_q"] >= 0.0, rec["min_pending_q"] >= 0.0, rec["post_finite"],
             rec["min_post_next_collision"] >= 0.0, rec["rebuild_old_vs_R3"] <= GATE, rec["rebuild_vs_cross"] <= GATE]
    if STATES[sid].get("expect_inactive"):
        gates.append(rec["P3_bitwise_inactive"])
    if "P5_q0_vs_equilibrium_rel" in rec:
        gates += [rec["P5_q0_vs_equilibrium_rel"] <= GATE, rec["P5_preserve_max_abs"] <= PRESERVE]
    if "P9_traction_defect" in rec:
        gates.append(rec["P9_traction_defect"] <= GATE)
    rec["pass"] = bool(all(gates))
    rec["verdict"] = "PASS" if rec["pass"] else "NOT PASSED"
    return rec


STATES = {
    "H1": {"container": "v2/hill_demo/e1/hf_inputs/hill_uniform.vwnat", "sha": "a404dc7463c9bc1a", "open": True, "length": 78.0,
           "from_run": "v2/hill_demo/e1/hf_run/hf_run"},
    "F1": {"container": "v2/open_terrain/e1/native/inputs/oc01_uniform.vwnat", "sha": "341b8c3dc6176fa6", "open": True, "length": 158.0,
           "target": (1.0, (U, 0.0, 0.0)), "expect_inactive": True},
    "F2": {"container": "v2/sample_support/e1/inputs/ss4_uniform.vwnat", "sha": "2a988ed7", "open": False,
           "target": (1.0, (U, 0.0, 0.0)), "expect_inactive": True},
    "S0": {"container": "v2/hill_demo/e1/pr_inputs/slab_S0.vwnat", "sha": "dd2a2bdb68546b41", "open": False,
           "target": (1.0, (0.0, 0.0, 0.0)), "expect_inactive": True},
    "S1": {"container": "v2/hill_demo/e1/pr_inputs/slab_S1.vwnat", "sha": "726e66503cd2101d", "open": False,
           "target": (1.0, tuple((U * np.array([1.0, 0.0, -3.0]) / math.sqrt(10.0)).tolist())), "expect_inactive": True},
    "S2": {"container": "v2/hill_demo/e1/pr_inputs/slab_S2.vwnat", "sha": "50a3ac0fc2ac62da", "open": False,
           "target": (1.0, (0.0, U, 0.0)), "expect_inactive": True},
    "S3": {"container": "v2/hill_demo/e1/pr_inputs/slab_S3.vwnat", "sha": "6c4e37d4e5356336", "open": False, "expect_inactive": True},
}
# H1: the exact failed-run state (#2581): geometry ecf08b8f, moving-state container a404dc74 (the run's input), and the saved
# level-0 fields of the HF-01 protocol folder, pinned by hash
H1_FIELDS = {"checkpoint_000000_fi.f32": "3c1c23da52d14de9", "checkpoint_000000_pending.f64": "769dda72d11df029",
             "checkpoint_000000_open_state.f64": "79c5318fb61100cf"}
H1_GEOMETRY = ("v2/hill_demo/e1/hb5b_run/hb_build/hill_d3q27.vwnat", "ecf08b8f7254c10d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--record", required=True, type=Path)
    args = ap.parse_args()
    if args.state not in STATES:
        raise Invalid(f"unknown state {args.state}")
    rec = check(args.state)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    arrays = rec.pop("audit_npz", None)
    if arrays is not None:
        npz = args.record.with_suffix(".npz")
        np.savez_compressed(npz, **arrays)
        rec["audit_npz_file"] = npz.name
        rec["audit_npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    args.record.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(rec["verdict"], flush=True)
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Invalid, ValueError) as error:
        print(f"INVALID: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
