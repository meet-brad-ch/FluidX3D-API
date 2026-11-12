"""Task 115 (E2): compatible-start constant-force run of the CLOSED periodic hill map. Reference only; no engine.

Map: OpenRoughSgsTerrainModel27(open_x = False, force = F) with the PR-02 projected transport (e1_pr2_run.ProjectedTransport,
unchanged, check_copy() run). That is the actual hill geometry, viscosity, rough law, SGS and positive projection, with
periodic x and y, the terrain and the frictionless lid. No boundary law, weight or geometry change. No ramp: F is constant
from the first collision on.

Start (MATHEMATICAL_MODEL 15.9; TASK115_SOURCE_ASSESSMENT Section 3): f0 = feq(rho0, 0) - Phi(0, F)/2, so the physical
u0 = 0 and A0 = 0. s0 = rho0 W w_row (the unforced prehistory, = the container s0). Recorded start checks: sum f0 - rho0,
sum c f0 + F/2, the physical u0, the RR stress A0, min f0 and min p0, and s0 against the container s0.

Per update n (vectorized, no full export): min f, min s, min p (the next collision); finite; max |rho - 1| with its cell
and volume; max physical speed; the volume-mean physical velocity (the achieved response); the augmented mass
sum V f + sum s (relative change); the augmented momentum budget P(n+1) - P(n) - (sum V F + sum J) (task-95 Section 5);
the max interior traction defect per area (the rough law); the active-projection count; at X = 67345 rho_X and, for its 4
returning facets, rho_s, rho_w, delta and sum q / sum v. Full f and s are saved at --checkpoints. The run STOPS at the
first registered failure and saves the input and output f/s (#2699, #2704): a FacetProjector P1 failure, a certified
infeasible or non-finite facet (REJECTED); an unresolved facet (INVALID); a non-finite or negative f, s, p or outgoing q
(REJECTED); a non-finite derived metric with finite fields (INVALID); the density screen |rho - 1| > 0.05 or the speed
screen |u| > 0.3 (REJECTED); or a model exception (REJECTED for a physical refusal, INVALID otherwise). A completed run
with a final volume-mean u_x < 0.03 is INCONCLUSIVE (#2695). The record holds class_suggested; the verdict is E1's.

  envs/solver_poc/Scripts/python.exe -B v2/hill_demo/e2/task115/driven_hill.py --force FX,FY,FZ --updates N
      --checkpoints 0,100,... --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
WALL = HERE.parents[3]
sys.path.insert(0, str(WALL / "v2" / "hill_demo" / "e1"))
import e1_pr2_run as PR2  # noqa: E402

OM, PC = PR2.OM, PR2.PC
Invalid = OM.Invalid
CASE = PR2.CASES["H1"]
X_CELL = 67345
X_FACETS = (12390, 12392, 12394, 12622)
DENSITY_SCREEN = 0.05
SPEED_SCREEN = 0.3
RESPONSE_MIN_MEAN_UX = 0.03            # #2695: below this the target regime is unqualified (INCONCLUSIVE), not a failure
PHYSICAL_EXCEPTIONS = PR2.PHYSICAL_INVALID + ("wall-extrapolated density",)   # model refusals of a physical state


class ForcedClosedModel(OM.OpenRoughSgsTerrainModel27, PR2.ProjectedTransport):
    def __init__(self, container: Path, force: np.ndarray) -> None:
        super().__init__(container, force, PC.SGS.SgsClosure(PC.K_SGS), None, False)
        if self.layout is not None:
            raise Invalid("the closed map must have no open layout")
        self.layout = SimpleNamespace(exterior_facet=np.zeros(self.facets, dtype=bool))    # all facets interior
        self.projector = PR2.FacetProjector(self)
        self.layout = None


class Run:
    def __init__(self, force: np.ndarray, out: Path) -> None:
        self.case = WALL / CASE["container"]
        self.sha = hashlib.sha256(self.case.read_bytes()).hexdigest()
        if not self.sha.startswith(CASE["sha"]):
            raise Invalid(f"container hash {self.sha[:16]} is not the pinned {CASE['sha']}")
        PR2.check_copy()
        self.F = np.asarray(force, dtype=np.float64)
        self.m = ForcedClosedModel(self.case, self.F)
        self.out = out
        self.t0 = time.perf_counter()
        self.last_print = self.t0

    def start(self, rho0: float = 1.0) -> tuple:
        m = self.m
        rho = np.full(m.cells, rho0)
        u = np.zeros((m.cells, 3))
        feq, phi = m.eq.parts(rho, u, self.F)
        f0 = feq - 0.5 * phi
        s0 = rho0 * (m.W @ m.w_row)
        A0 = m.eq.stress(f0, phi, rho, u)
        p0 = m.collide(f0)
        container_s0 = np.asarray(m.initial_pending, dtype=np.float64)
        checks = {"sum_f0_minus_rho0_max_abs": float(np.abs(f0.sum(axis=1) - rho0).max()),
                  "sum_c_f0_plus_F_half_max_abs": float(np.abs(f0 @ m.c + 0.5 * self.F[None, :]).max()),
                  "physical_u0_max_abs": float(np.abs((f0 @ m.c + 0.5 * self.F[None, :]) / rho0).max()),
                  "A0_max_abs": float(np.abs(A0).max()), "min_f0": float(f0.min()), "min_p0": float(p0.min()),
                  "s0_vs_container_s0_max_abs": float(np.abs(s0 - container_s0).max()),
                  "positivity_bound_|F|<2rho0/(3sqrt3)": bool(np.linalg.norm(self.F) < 2.0 * rho0 / (3.0 * np.sqrt(3.0)))}
        return f0, s0, checks

    def physical_u(self, f: np.ndarray) -> tuple:
        rho = f.sum(axis=1)
        return rho, (f @ self.m.c + 0.5 * self.F[None, :]) / rho[:, None]

    def x_local(self, f, wall) -> dict:
        m = self.m
        out = {"rho_X": float(f[X_CELL].sum())}
        for a in X_FACETS:
            rows = m.facet_of_row == a
            out[str(a)] = {"rho_s": float(wall.rho_s[a]), "rho_w": float(wall.rho_w[a]), "delta": float(wall.delta[a]),
                           "q_over_v_mean": float(wall.outgoing[rows].sum() / m.prism[rows].sum())}
        return out

    def projector_lists(self) -> dict:
        """#2704: the FacetProjector failure lists (it keeps q0 on those facets and reports without raising)."""
        pj = self.m.projector.last
        return {"projector_p1_failures": len(pj["p1_failures"]), "projector_infeasible": len(pj["infeasible"]),
                "projector_unresolved": len(pj["unresolved"]), "projector_nonfinite": int(pj["nonfinite_count"]),
                "projector_ids": {"p1": [x["facet"] for x in pj["p1_failures"][:10]],
                                  "infeasible": [x["facet"] for x in pj["infeasible"][:10]],
                                  "unresolved": [x["facet"] for x in pj["unresolved"][:10]],
                                  "nonfinite": pj["nonfinite_facets"][:10]}}

    @staticmethod
    def stop_reason(row: dict, finite: bool, min_p: float):
        """(class, kind) of the first registered failure (#2699, #2704), or None. REJECTED = a physical inadmissible state,
        a failed q0 equality, a certified infeasible or non-finite facet, or a screen crossing; INVALID = an unresolved
        projection (no certificate) or a non-finite derived metric with finite fields."""
        if row["projector_p1_failures"] or row["projector_infeasible"] or row["projector_nonfinite"]:
            return "REJECTED", "projector: P1 equality failure, certified infeasible or non-finite facet"
        if row["projector_unresolved"]:
            return "INVALID", "projector: facet without a KKT or infeasibility certificate"
        if not finite or row["min_f"] < 0.0 or row["min_s"] < 0.0 or min_p < 0.0 or row["min_outgoing_q"] < 0.0:
            return "REJECTED", "inadmissible (non-finite or negative f, s, p or outgoing q)"
        numbers = [v for k, v in row.items() if isinstance(v, float)] + row["mean_u"]
        if not all(np.isfinite(numbers)):
            return "INVALID", "non-finite derived metric with finite fields"
        if row["density_departure_max"] > DENSITY_SCREEN:
            return "REJECTED", "density screen |rho - 1| > 0.05"
        if row["speed_max"] > SPEED_SCREEN:
            return "REJECTED", "speed screen |u| > 0.3"
        return None

    def save(self, name: str, f, s) -> str:
        path = self.out / f"{name}.npz"
        np.savez_compressed(path, f=f, s=s)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run(self, updates: int, checkpoints: set) -> dict:
        m = self.m
        f, s, start = self.start()
        record = {"force": self.F.tolist(), "container_sha256": self.sha, "start_checks": start, "rows": [], "checkpoints": {},
                  "stop": None}
        if 0 in checkpoints:
            record["checkpoints"]["0"] = self.save("checkpoint_0", f, s)
        V = m.volume
        mass0 = float(V @ f.sum(axis=1) + s.sum())
        force_impulse = V.sum() * self.F
        P = m.augmented_momentum(f, s)
        for n in range(1, updates + 1):
            try:
                p = m.collide(f)
                min_p = float(p.min())
                new, s_next, wall = m.transport_augmented(p, s)
            except Exception as err:  # noqa: BLE001  (a model exception is a stop; keep the state)
                physical = any(key in str(err) for key in PHYSICAL_EXCEPTIONS)
                record["stop"] = {"update": n, "class": "REJECTED" if physical else "INVALID", "kind": "exception",
                                  "type": type(err).__name__, "error": str(err), "sha256_input": self.save(f"stop_{n}_input", f, s)}
                break
            finite = bool(np.isfinite(new).all() and np.isfinite(s_next).all() and np.isfinite(p).all())
            rho, u = self.physical_u(new)
            speed = np.sqrt((u * u).sum(axis=1))
            dep = np.abs(rho - 1.0)
            kd = int(np.argmax(dep))
            P_next = m.augmented_momentum(new, s_next)
            J = wall.impulse.sum(axis=0)
            budget = P_next - P - (force_impulse + J)
            # rounding scale fixed before the run (#2695): the gross magnitude of every term summed in the budget
            cnorm = np.linalg.norm(m.c, axis=1)
            gross = (float(V @ (np.abs(new) @ cnorm)) + float(V @ (np.abs(f) @ cnorm))
                     + float(np.abs(s_next) @ np.linalg.norm(m.c_row, axis=1)) + float(np.abs(s) @ np.linalg.norm(m.c_row, axis=1))
                     + float(np.abs(wall.impulse).sum()) + float(np.abs(force_impulse).sum()))
            traction = m.traction_defect(wall)
            row = {"n": n, "finite": finite, "min_f": float(new.min()), "min_s": float(s_next.min()), "min_p": min_p,
                   "density_departure_max": float(dep[kd]), "density_cell": kd, "density_cell_volume": float(V[kd]),
                   "speed_max": float(speed.max()), "mean_u": ((V @ u) / V.sum()).tolist(),
                   "mass_rel_change": float((V @ new.sum(axis=1) + s_next.sum()) / mass0 - 1.0),
                   "momentum_budget_max_abs": float(np.abs(budget).max()),
                   "momentum_budget_rel_force_impulse": float(np.abs(budget).max() / max(np.abs(force_impulse).max(), 1e-300)),
                   "momentum_gross_scale": gross, "momentum_budget_rel_gross": float(np.abs(budget).max() / gross),
                   "traction_defect_max": float(traction.max()),
                   "active_facets": len(m.projector.last["active"]), **self.projector_lists(),
                   "min_outgoing_q": float(wall.outgoing.min()),
                   "min_outgoing_q_over_v": float((wall.outgoing / m.prism).min()),
                   "X": self.x_local(new, wall)}
            record["rows"].append(row)
            reason = self.stop_reason(row, finite, min_p)
            if reason is not None:
                record["stop"] = {"update": n, "class": reason[0], "kind": reason[1],
                                  "sha256_input": self.save(f"stop_{n}_input", f, s),
                                  "sha256_output": self.save(f"stop_{n}_output", new, s_next)}
                break
            f, s, P = new, s_next, P_next
            if n in checkpoints:
                record["checkpoints"][str(n)] = self.save(f"checkpoint_{n}", f, s)
            now = time.perf_counter()
            if now - self.last_print >= 60.0:
                print(f"[{now - self.t0:7.0f} s] update {n}/{updates} dep {row['density_departure_max']:.4f} "
                      f"umax {row['speed_max']:.4f} mean_ux {row['mean_u'][0]:.5f}", flush=True)
                self.last_print = now
        rows = record["rows"]
        record["screens"] = {"density_first_exceedance": next((r["n"] for r in rows if r["density_departure_max"] > DENSITY_SCREEN), None),
                             "speed_first_exceedance": next((r["n"] for r in rows if r["speed_max"] > SPEED_SCREEN), None)}
        record["updates_completed"] = len(rows) if record["stop"] is None else record["stop"]["update"] - 1
        if record["stop"] is not None:
            record["class_suggested"] = record["stop"]["class"]
        elif rows and rows[-1]["mean_u"][0] < RESPONSE_MIN_MEAN_UX:
            record["class_suggested"] = "INCONCLUSIVE (insufficient response: mean u_x below 0.03)"
        else:
            record["class_suggested"] = "COMPLETED (the analyzer applies the registered budget and response gates)"
        return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", required=True, help="FX,FY,FZ lattice units")
    ap.add_argument("--updates", required=True, type=int)
    ap.add_argument("--checkpoints", default="0")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    force = np.array([float(v) for v in a.force.split(",")])
    ck = {int(v) for v in a.checkpoints.split(",") if v != ""}
    t = time.perf_counter()
    try:
        run = Run(force, a.out)
        rec = run.run(a.updates, ck)
    except Exception as err:  # noqa: BLE001  (setup or unexpected failure: keep a record and exit non-zero)
        partial = {"class_suggested": "INVALID", "error_type": type(err).__name__, "error": str(err),
                   "seconds": time.perf_counter() - t}
        (a.out / "driven_hill_FAILED.json").write_text(json.dumps(partial, indent=1), encoding="utf-8")
        print(f"FAILED: {type(err).__name__}: {err}", flush=True)
        return 2
    rec["seconds"] = time.perf_counter() - t
    import psutil  # noqa: PLC0415
    mem = psutil.Process().memory_info()
    rec["peak_working_set_GB"] = getattr(mem, "peak_wset", mem.rss) / 2**30
    (a.out / "driven_hill.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    summary = {k: v for k, v in rec.items() if k != "rows"}
    summary["last_row"] = rec["rows"][-1] if rec["rows"] else None
    print(json.dumps(summary, indent=1)[:4000])
    return 0 if rec["stop"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
