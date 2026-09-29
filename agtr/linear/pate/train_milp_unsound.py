"""Matched PATE-R vs AGT-R benchmark on the linear task -- certified teacher training.

Every arm of the comparison is built from ONE kind of object: a linear teacher y = w x + b trained
by full-batch clipped gradient descent (per-coordinate clip gamma, lr, E steps, init (0,0)) on a
subset of the training data, together with a sound parameter-space bound T_k for every k in a grid,
obtained with the optimisation-based AGT (Gurobi MILP, one window per step = concretisation
frequency 1; see the probe results in RUNLOG.md: IBP saturates on this problem, per-step MILP does
not, multi-step windows are much slower and add nothing for a convex loss).

For every ensemble size T the training set is split into T disjoint shards (one fixed secret
permutation, consecutive chunks), and every shard i gets a teacher with bounds for every k <= |shard|
in the grid.  The analysis script then assembles

  PATE-R        : mean of the T nominal teachers + Laplace(GS/(T eps))            [GS = output range]
  PATE-AGT-R    : same mean + Cauchy calibrated to SS_g = (1/T) max_i SS^beta(f_i, x)
  AGT-R (full)  : the T=1 teacher + Cauchy calibrated to its own SS^beta
  AGT-R (WOR)   : teacher 0 of the T-partition (a uniform without-replacement subsample of size
                  N/T) with the amplified base budget eps_b = log(1 + T (e^eps - 1))

so the four arms share data, model, clipping, optimiser, nominal and certification procedure.

Serialisation (one npz per cell (T, shard, k)):  parameter boxes (l, n, u) after EVERY step,
the final box, wall time, Gurobi solve statuses parsed from the per-cell Gurobi log (logs are kept
next to the cells; *.log is git-ignored), plus config.json / data arrays / manifest.json for the run.
Nothing is ever overwritten: existing cells are skipped, so the run is resumable.

    PYTHONPATH=$PWD python -m agtr.linear.pate.train_milp_unsound --workers 100
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import time

import numpy as np

from common.provenance import git_sha

ROOT = "agtr/linear/values/pate_matched"
DEFAULT_T = [1, 2, 4, 8, 16, 32, 64, 128]
# The k grid MUST be consecutive (1..k_max) for the released smooth-sensitivity bound to be beta-smooth:
# with gaps the back-filled certificate is only (beta * max gap)-smooth (see RUNLOG.md, "Why dense k").
# k_max(m) = min(K_CAP, m // K_FRAC): solves with k a large fraction of the shard are hard MILPs and a
# solve that stops at its time limit is not a monotone function of the data (best bound), which would
# also break smoothness -- so the grid is capped where every solve finishes at optimality.
K_CAP, K_FRAC = 64, 8


def make_data(n_train, n_test, seed, slope, intercept, x_range, noise_std, clip_y):
    """Same generative model as agtr.linear.config.PATE (y = 2x + 1 + N(0, 0.1^2), x ~ U[-2, 2], y clipped)."""
    r = np.random.default_rng(seed)
    X = r.uniform(x_range[0], x_range[1], (n_train, 1))
    y = np.clip(slope * X + intercept + r.normal(0, noise_std, (n_train, 1)), -clip_y, clip_y)
    r = np.random.default_rng(seed + 1000)
    Xt = r.uniform(x_range[0], x_range[1], (n_test, 1))
    yt = np.clip(slope * Xt + intercept + r.normal(0, noise_std, (n_test, 1)), -clip_y, clip_y)
    return X, y, Xt, yt


def shard_indices(n, T, perm):
    """Consecutive chunks of one fixed permutation: shard i of the T-partition."""
    bounds = np.linspace(0, n, T + 1).astype(int)
    return [perm[bounds[i] : bounds[i + 1]] for i in range(T)]


def parse_gurobi_log(path):
    """Counts of solve outcomes in a per-cell Gurobi log (every window solves 4 objectives)."""
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    return {
        "optimal": len(re.findall(r"Optimal solution found", txt)),
        "time_limit": len(re.findall(r"Time limit reached", txt)),
        "infeasible": len(re.findall(r"Model is infeasible", txt)),
        "numerical": len(re.findall(r"Numerical trouble|numerical issues", txt)),
        "solves_started": len(re.findall(r"Optimize a model with", txt)),
    }


def run_cell(job):
    """Train one certified teacher: (T, shard i, k).  Returns a manifest entry."""
    T, i, k, cfg, idx = job
    out = f"{ROOT}/{cfg['tag']}/T{T}"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/shard{i}_k{k}.npz"
    if os.path.exists(path):
        return (f"T{T}/shard{i}_k{k}", {"status": "exists"})
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from abstract_gradient_training_opt.certified_training import certified_training
    from abstract_gradient_training_opt.configuration import OPTConfig

    logging.disable(logging.WARNING)
    X = np.load(f"{ROOT}/{cfg['tag']}/X_train.npy")
    y = np.load(f"{ROOT}/{cfg['tag']}/y_train.npy")
    ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float64), torch.tensor(y[idx], dtype=torch.float64))
    model = nn.Sequential(nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].bias.fill_(0.0)  # public initialisation
    log_path = f"{out}/shard{i}_k{k}.gurobi.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    opt = OPTConfig(
        log_level="ERROR",
        n_epochs=cfg["epochs"],
        batchsize=len(idx),
        learning_rate=cfg["lr"],
        loss="mse",
        concretization_freq=1,
        k_substitution=int(k),
        input_domain=None,
        label_domain=None,
        gradient_clipping=cfg["gamma"],
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={
            "NonConvex": 2,
            "OutputFlag": 1,
            "LogFile": log_path,
            "LogToConsole": 0,
            "Threads": 1,
            "TimeLimit": cfg["time_limit"],
        },
        metadata={"seed": 0, "strict_mode": True},
    )
    t0 = time.time()
    final, concretized = certified_training(model, opt, ds, solve=True)
    secs = time.time() - t0
    flat = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])  # (w, b)
    steps_l = np.stack([flat(m.param_l) for m in concretized])
    steps_n = np.stack([flat(m.param_n) for m in concretized])
    steps_u = np.stack([flat(m.param_u) for m in concretized])
    pl, pn, pu = flat(final.param_l), flat(final.param_n), flat(final.param_u)
    stat = parse_gurobi_log(log_path)
    np.savez_compressed(
        path,
        param_l=pl,
        param_n=pn,
        param_u=pu,
        steps_l=steps_l,
        steps_n=steps_n,
        steps_u=steps_u,
        seconds=secs,
        T=T,
        shard=i,
        k=k,
        m=len(idx),
        shard_idx=idx,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        gurobi_optimal=stat.get("optimal", -1),
        gurobi_time_limit=stat.get("time_limit", -1),
        gurobi_infeasible=stat.get("infeasible", -1),
        gurobi_solves=stat.get("solves_started", -1),
    )
    entry = {
        "status": "ok",
        "seconds": round(secs, 1),
        "m": int(len(idx)),
        "n_steps_serialized": int(len(concretized)),
        "width_w": float(pu[0] - pl[0]),
        "width_b": float(pu[1] - pl[1]),
        "w": float(pn[0]),
        "b": float(pn[1]),
        **{f"gurobi_{a}": b for a, b in stat.items()},
    }
    return (f"T{T}/shard{i}_k{k}", entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n4000_e20_lr0.3_g1")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--T", type=int, nargs="+", default=DEFAULT_T)
    ap.add_argument("--k-cap", type=int, default=K_CAP)
    ap.add_argument("--k-frac", type=int, default=K_FRAC)
    ap.add_argument("--epochs", type=int, default=20, help="full-batch GD steps per teacher")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument(
        "--time-limit",
        type=int,
        default=1800,
        help="Gurobi TimeLimit per solve (s); any solve that hits it is recorded in the cell and manifest",
    )
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="data seed; partition permutation uses seed+1")
    a = ap.parse_args()
    out = f"{ROOT}/{a.tag}"
    os.makedirs(out, exist_ok=True)
    cfg = dict(
        tag=a.tag,
        n_train=a.n_train,
        n_test=a.n_test,
        slope=2.0,
        intercept=1.0,
        x_range=[-2.0, 2.0],
        noise_std=0.1,
        clip_y=6.0,
        gs_output_range=12.0,
        T_values=a.T,
        k_grid=f"dense 1..min({a.k_cap}, m//{a.k_frac})",
        k_cap=a.k_cap,
        k_frac=a.k_frac,
        epochs=a.epochs,
        lr=a.lr,
        gamma=a.gamma,
        clip_method="clamp (per-coordinate)",
        batch="full shard (one step per epoch)",
        init="w=b=0 (public)",
        certification="abstract_gradient_training_opt MILP, concretization_freq=1, best bound on time limit",
        time_limit=a.time_limit,
        seed=a.seed,
        partition="one permutation (seed+1), consecutive chunks; shard 0 is a uniform WOR subsample of size N/T",
        git_sha=git_sha(),
        host=platform.node(),
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not os.path.exists(f"{out}/X_train.npy"):
        X, y, Xt, yt = make_data(a.n_train, a.n_test, a.seed, 2.0, 1.0, (-2.0, 2.0), 0.1, 6.0)
        perm = np.random.default_rng(a.seed + 1).permutation(a.n_train)
        for n_, v in (("X_train", X), ("y_train", y), ("X_test", Xt), ("y_test", yt), ("perm", perm)):
            np.save(f"{out}/{n_}.npy", v)
        json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    else:
        prev = json.load(open(f"{out}/config.json"))
        for key in ("n_train", "epochs", "lr", "gamma", "seed"):
            assert prev[key] == cfg[key], f"{key} differs from the existing run in {out}"
        prev["T_values"] = sorted(set(prev["T_values"]) | set(a.T))
        prev.setdefault("resumed", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(prev, open(f"{out}/config.json", "w"), indent=1)
    perm = np.load(f"{out}/perm.npy")
    jobs = []
    for T in a.T:
        shards = shard_indices(a.n_train, T, perm)
        for i, idx in enumerate(shards):
            for k in range(1, min(a.k_cap, len(idx) // a.k_frac) + 1):
                if not os.path.exists(f"{out}/T{T}/shard{i}_k{k}.npz"):
                    jobs.append((T, i, k, cfg, idx))
    jobs.sort(key=lambda j: -len(j[4]))  # largest shards first (longest cells)
    print(f"{len(jobs)} cells to run with {a.workers} workers -> {out}", flush=True)
    manifest_path = f"{out}/manifest.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for key, entry in pool.imap_unordered(run_cell, jobs, chunksize=1):
            manifest[key] = entry
            done += 1
            if done % 25 == 0 or done == len(jobs):
                json.dump(manifest, open(manifest_path, "w"), indent=1)
                print(f"[{time.time() - t0:7.0f}s] {done}/{len(jobs)}  last {key}: {entry}", flush=True)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
