"""Matched PATE-R vs AGT-R benchmark on the linear task -- certified teachers via the closed-form exact
per-step certifier (exact_linear_certifier.py) instead of the MILP.  Same data model, partition, learner and
serialisation contract as train_milp_unsound.py, but one file per teacher holding every k:

  values/pate_matched/<tag>/T<T>/shard<i>.npz :
      nominal   (steps+1, 2)              trajectory (init point first)
      ks        (K,)                      dense 1..k_max
      boxes     (K, steps+1, 2, 2)        [[wl, wu], [bl, bu]] after every step, for every k
      shard_idx, m, T, shard, epochs, lr, gamma, seconds

k grid: dense 1..k_max(m), k_max(m) = min(k_cap, m).  Teachers are trained in parallel over (T, shard).

    PYTHONPATH=$PWD python -m agtr.linear.pate.train --n-train 4000 --tag exact_n4000_e20_lr0.3_g1
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import time

import numpy as np

from agtr.linear.pate.exact_linear_certifier import certify_all_k
from agtr.linear.pate.train_milp_unsound import ROOT
from agtr.linear.pate.train_milp_unsound import make_data
from agtr.linear.pate.train_milp_unsound import shard_indices
from common.provenance import git_sha


def run_teacher(job):
    T, i, idx, cfg = job
    out = f"{ROOT}/{cfg['tag']}/T{T}"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/shard{i}.npz"
    if os.path.exists(path):
        return f"T{T}/shard{i}", {"status": "exists"}
    X = np.load(f"{ROOT}/{cfg['tag']}/X_train.npy").ravel()
    y = np.load(f"{ROOT}/{cfg['tag']}/y_train.npy").ravel()
    ks = np.arange(1, min(cfg["k_cap"], len(idx)) + 1)
    t0 = time.time()
    nom, boxes = certify_all_k(X[idx], y[idx], ks, cfg["epochs"], cfg["lr"], cfg["gamma"])
    secs = time.time() - t0
    np.savez_compressed(
        path,
        nominal=nom,
        ks=ks,
        boxes=boxes,
        shard_idx=idx,
        m=len(idx),
        T=T,
        shard=i,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        seconds=secs,
        certifier="exact_linear_certifier.certify (closed form, per-step exact)",
    )
    w = boxes[:, -1, :, 1] - boxes[:, -1, :, 0]
    return f"T{T}/shard{i}", {
        "status": "ok",
        "seconds": round(secs, 1),
        "m": int(len(idx)),
        "k_max": int(ks[-1]),
        "w": float(nom[-1, 0]),
        "b": float(nom[-1, 1]),
        "width_k1": w[0].tolist(),
        "width_kmax": w[-1].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--k-cap", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.T = [T for T in a.T if a.n_train // T >= 2]
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
        k_grid=f"dense 1..min({a.k_cap}, m)",
        k_cap=a.k_cap,
        epochs=a.epochs,
        lr=a.lr,
        gamma=a.gamma,
        clip_method="clamp (per-coordinate)",
        batch="full shard (one step per epoch)",
        init="w=b=0 (public)",
        certification="exact_linear_certifier.certify: closed-form exact per-step reachable box (straddling clamps over-approximated); validated vs brute force and substitution attacks",
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
    perm = np.load(f"{out}/perm.npy")
    jobs = [(T, i, idx, cfg) for T in a.T for i, idx in enumerate(shard_indices(a.n_train, T, perm))]
    jobs.sort(key=lambda j: -len(j[2]) * min(a.k_cap, len(j[2])))
    print(f"{len(jobs)} teachers with {a.workers} workers -> {out}", flush=True)
    manifest = {}
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, (key, entry) in enumerate(pool.imap_unordered(run_teacher, jobs, chunksize=1), 1):
            manifest[key] = entry
            if n % 50 == 0 or n == len(jobs):
                json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(jobs)} {key} {entry}", flush=True)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
