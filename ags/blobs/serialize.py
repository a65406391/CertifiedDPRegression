"""Total serialization for certified runs. Nothing computed is discarded.

Layout, one directory per run, under ags/blobs/runs/certified/<run_id>/:

    meta.json                  tier 3 -- config + provenance, written at start
    models/k{k}_w{w:05d}.lp.gz tier 0 -- the Gurobi formulation for that step
    solves.jsonl               tier 1 -- ONE record per Gurobi optimize() call
    steps/k{k}_w{w:05d}.npz    tier 2 -- (l, n, u) + derived widths per step
    progress.jsonl             heartbeat every 30 s: position, solver status, ETA
    MANIFEST.json              written at close -- counts, checksums, verdict

Capture works by wrapping two functions:

  gurobipy.Model.optimize          -> tier 1. Every solve, no exceptions. The
                                     direction comes from ModelSense, so min/max
                                     are distinguished without guessing.
  certified_training.bound_parameters
                                   -> tiers 0 and 2, and the step boundary. The
                                     library calls it exactly once per
                                     concretization, which is what defines a step.

Both are restored by uninstall(). All appends are flushed and fsynced, so a
kill -9 loses at most the record being written.
"""

from __future__ import annotations
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import numpy as np

ROOT = "ags/blobs/runs/certified"
_CTX: "RunRecorder | None" = None


def _git():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return None, None


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "torch", "gurobipy", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            try:
                import gurobipy as gp

                out[m] = ".".join(map(str, gp.gurobi.version())) if m == "gurobipy" else "?"
            except Exception:
                out[m] = None
    return out


def _attr(m, name, default=None):
    try:
        return m.getAttr(name)
    except Exception:
        return default


class RunRecorder:
    def __init__(
        self, run_id, cfg, *, expected_windows_per_k, k_grid, n_param_coords, root=ROOT, heartbeat=30.0
    ):
        self.run_id, self.cfg = run_id, cfg
        self.dir = os.path.join(root, run_id)
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        for sub in ("steps", "models"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.k_grid = list(k_grid)
        self.exp_w = int(expected_windows_per_k)
        self.P = int(n_param_coords)
        self.expected_steps = self.exp_w * len(self.k_grid)
        self.expected_solves = self.expected_steps * 2 * self.P

        self.k = None
        self.window = 0
        self.in_window = 0
        self.n_solves = 0
        self.n_steps = 0
        self.counts = dict(optimal=0, time_limit=0, infeasible=0, other=0, dual_fallback=0, no_solution=0)
        self.k_times = {k: [] for k in self.k_grid}
        self.t0 = time.time()
        self._closed = False

        sha, dirty = _git()
        meta = dict(
            run_id=run_id,
            cfg=cfg,
            k_grid=self.k_grid,
            expected_windows_per_k=self.exp_w,
            n_param_coords=self.P,
            expected_steps=self.expected_steps,
            expected_solves=self.expected_solves,
            git_sha=sha,
            git_dirty=dirty,
            host=platform.node(),
            pid=os.getpid(),
            versions=_versions(),
            env={k: v for k, v in os.environ.items() if k.startswith("AGT_")},
            agt_opt_max_bound_coords=os.environ.get("AGT_OPT_MAX_BOUND_COORDS"),
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
            started_epoch=self.t0,
        )
        self._dump(os.path.join(self.dir, "meta.json"), meta)
        self._solves = open(os.path.join(self.dir, "solves.jsonl"), "a")
        self._prog = open(os.path.join(self.dir, "progress.jsonl"), "a")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._beat, args=(heartbeat,), daemon=True)
        self._hb.start()

    # ---------- io ----------
    @staticmethod
    def _dump(path, obj):
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())

    def _append(self, fh, rec):
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    # ---------- context ----------
    def set_k(self, k):
        self.k, self.window, self.in_window = k, 0, 0

    # ---------- eta ----------
    def eta(self):
        done, tot = self.n_solves, self.expected_solves
        if done < 5 or tot <= 0:
            return None
        el = time.time() - self.t0
        per = el / done
        rem = max(0, tot - done)
        # per-k rate where available, else pooled
        allt = [t for v in self.k_times.values() for t in v]
        if len(allt) >= 20:
            a = np.array(allt)
            lo, hi = float(np.percentile(a, 10)), float(np.percentile(a, 90))
        else:
            lo = hi = per
        return dict(
            done=done,
            total=tot,
            frac=done / tot,
            elapsed_s=el,
            per_solve_s=per,
            eta_s=rem * per,
            eta_lo_s=rem * lo,
            eta_hi_s=rem * hi,
            eta_clock=time.strftime("%H:%M:%S", time.localtime(time.time() + rem * per)),
        )

    def _beat(self, period):
        while not self._stop.wait(period):
            try:
                rec = dict(
                    t=time.strftime("%Y-%m-%d %H:%M:%S"),
                    epoch=time.time(),
                    run_id=self.run_id,
                    k=self.k,
                    window=self.window,
                    steps=self.n_steps,
                    expected_steps=self.expected_steps,
                    solves=self.n_solves,
                    expected_solves=self.expected_solves,
                    counts=dict(self.counts),
                    eta=self.eta(),
                    rss_mb=self._rss(),
                )
                with self._lock:
                    self._append(self._prog, rec)
            except Exception:
                pass

    @staticmethod
    def _rss():
        try:
            with open(f"/proc/{os.getpid()}/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
        except Exception:
            return None

    # ---------- tier 1 ----------
    def record_solve(self, m, wall):
        st = _attr(m, "Status")
        sc = _attr(m, "SolCount", 0) or 0
        maxvio = _attr(m, "MaxVio", 0.0) or 0.0
        try:
            ftol = m.Params.FeasibilityTol
        except Exception:
            ftol = 1e-6
        # bound_objective uses ObjBound both for numerical-violation fallback
        # and whenever optimization stops before OPTIMAL (notably TIME_LIMIT).
        # Record both cases as dual-bound use so the audit metadata matches the
        # value that actually enters the serialized enclosure.
        dual = maxvio >= ftol or st not in (2, 3)
        try:
            obj = m.getObjective()
            objvar = obj.getVar(0).VarName if obj.size() == 1 else None
        except Exception:
            objvar = None
        rec = dict(
            k=self.k,
            window=self.window,
            solve_idx=self.n_solves,
            in_window=self.in_window,
            coord_idx=self.in_window // 2,
            obj_var=objvar,
            sense=("min" if _attr(m, "ModelSense", 1) == 1 else "max"),
            status=st,
            sol_count=sc,
            obj_val=(_attr(m, "ObjVal") if sc > 0 else None),
            obj_bound=_attr(m, "ObjBound"),
            mip_gap=(_attr(m, "MIPGap") if sc > 0 else None),
            runtime_s=_attr(m, "Runtime"),
            wall_s=wall,
            node_count=_attr(m, "NodeCount"),
            iter_count=_attr(m, "IterCount"),
            max_vio=maxvio,
            used_dual_bound=bool(dual),
            hit_time_limit=(st == 9),
        )
        with self._lock:
            self.n_solves += 1
            self.in_window += 1
            if self.k in self.k_times:
                self.k_times[self.k].append(wall)
            if st == 2:
                self.counts["optimal"] += 1
            elif st == 9:
                self.counts["time_limit"] += 1
            elif st == 3:
                self.counts["infeasible"] += 1
            else:
                self.counts["other"] += 1
            if dual:
                self.counts["dual_fallback"] += 1
            if sc == 0:
                self.counts["no_solution"] += 1
            self._append(self._solves, rec)

    # ---------- tiers 0 + 2 ----------
    def record_formulation(self, formulation):
        tag = f"k{self.k}_w{self.window:05d}"
        path = os.path.join(self.dir, "models", f"{tag}.lp")
        try:
            formulation.write(path)
            with open(path, "rb") as fi, gzip.open(path + ".gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(path)
            return dict(
                num_vars=_attr(formulation, "NumVars"),
                num_constrs=_attr(formulation, "NumConstrs"),
                num_bin_vars=_attr(formulation, "NumBinVars"),
                num_q_constrs=_attr(formulation, "NumQConstrs"),
                num_nzs=_attr(formulation, "NumNZs"),
                lp_bytes=os.path.getsize(path + ".gz"),
            )
        except Exception as e:
            return dict(error=str(e))

    def record_step(self, params_l, params_n, params_u, stats, wall):
        flat = lambda ps: np.concatenate([np.asarray(p).ravel() for p in ps])
        l, n, u = flat(params_l), flat(params_n), flat(params_u)
        W = u - l
        tag = f"k{self.k}_w{self.window:05d}"
        np.savez_compressed(
            os.path.join(self.dir, "steps", f"{tag}.npz"),
            param_l=l,
            param_n=n,
            param_u=u,
            width=W,
            shapes=np.array([np.asarray(p).shape for p in params_l], dtype=object),
            k=self.k,
            window=self.window,
            width_linf=float(np.max(np.abs(W))),
            width_l1=float(np.sum(np.abs(W))),
            A_k=float(np.abs(u - n).sum() + np.abs(n - l).sum()) / 2.0,
            wall_s=wall,
            stats=np.array(json.dumps(stats, default=str)),
        )
        with self._lock:
            self.n_steps += 1
            self.window += 1
            self.in_window = 0

    # ---------- close ----------
    def close(self, status="completed", error=None):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._hb.join(timeout=3)
        steps = sorted(os.listdir(os.path.join(self.dir, "steps")))
        models = sorted(os.listdir(os.path.join(self.dir, "models")))
        man = dict(
            run_id=self.run_id,
            status=status,
            error=error,
            finished=time.strftime("%Y-%m-%d %H:%M:%S"),
            wall_s=time.time() - self.t0,
            steps_written=len(steps),
            steps_expected=self.expected_steps,
            models_written=len(models),
            solves_written=self.n_solves,
            solves_expected=self.expected_solves,
            counts=dict(self.counts),
            steps_complete=len(steps) == self.expected_steps,
            solves_complete=self.n_solves == self.expected_solves,
            solves_sha256=self._sha(os.path.join(self.dir, "solves.jsonl")),
        )
        self._dump(os.path.join(self.dir, "MANIFEST.json"), man)
        for fh in (self._solves, self._prog):
            try:
                fh.close()
            except Exception:
                pass
        return man

    @staticmethod
    def _sha(path):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None


# ------------------------- monkeypatching -------------------------
_ORIG = {}


def install(rec: RunRecorder):
    """Wrap Model.optimize and bound_parameters. Idempotent per process."""
    global _CTX
    _CTX = rec
    import gurobipy as gp
    from abstract_gradient_training_opt import certified_training as ct

    if "optimize" not in _ORIG:
        _ORIG["optimize"] = gp.Model.optimize

        def optimize(self, callback=None):
            t = time.time()
            out = _ORIG["optimize"](self, callback)
            if _CTX is not None:
                try:
                    _CTX.record_solve(self, time.time() - t)
                except Exception:
                    pass
            return out

        gp.Model.optimize = optimize

    if "bound_parameters" not in _ORIG:
        _ORIG["bound_parameters"] = ct.bound_parameters

        def bound_parameters(mip_model, callback=None):
            stats = {}
            if _CTX is not None:
                try:
                    stats = _CTX.record_formulation(mip_model.formulation)
                except Exception as e:
                    stats = {"error": str(e)}
            t = time.time()
            l, n, u = _ORIG["bound_parameters"](mip_model, callback)
            if _CTX is not None:
                try:
                    _CTX.record_step(l, n, u, stats, time.time() - t)
                except Exception:
                    pass
            return l, n, u

        ct.bound_parameters = bound_parameters


def uninstall():
    global _CTX
    _CTX = None
    import gurobipy as gp
    from abstract_gradient_training_opt import certified_training as ct

    if "optimize" in _ORIG:
        gp.Model.optimize = _ORIG.pop("optimize")
    if "bound_parameters" in _ORIG:
        ct.bound_parameters = _ORIG.pop("bound_parameters")
