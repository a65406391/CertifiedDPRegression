"""PATE-R vs AGT-R on California Housing -- certified shard teachers (genuine AGT, GPU).

Same four-arm design as the linear benchmark (agtr/linear/pate): the training split (16 512
points) is cut into T disjoint shards by one secret permutation; every shard gets an 8-64-1 ReLU teacher
trained by full-shard clipped GD (per-coordinate clip gamma = 0.1, lr 0.01, STEPS steps -- the same
schedule as the full-batch arm of `relu_h64_clamp_gamma0.1`, whose bs16512 cells are the T = 1 arm) with
`privacy_certified_training` bounds for every k in a DENSE grid 1..k_max.  Each cell stores the parameter
box, the output box (y_l, y_u) and the nominal prediction y_n on the full test split, so the analysis needs
no re-propagation.  Cells: values/agt_2026-09-14/pate_relu_h64_clamp_gamma0.1/T<T>/shard<i>_k<k>.npz.

Workers: --worker w --n-workers n takes every n-th cell of the (T, shard, k) list (largest shards first);
run 3 per GPU.  Existing cells are skipped (resumable).  k = 0 is never trained.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m agtr.california.pate.train --worker 0 --n-workers 6
"""

import argparse
import json
import logging
import os
import platform
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

from agtr.california.certify import CFG, OUT_ROOT, STORED, Y_MAX, Y_MIN, git_sha, train_cell
from common import agt_patches
from common.data import load_california_data

logging.disable(logging.WARNING)
PERM_SEED = 123


def shards_for(n, T, perm):
    b = np.linspace(0, n, T + 1).astype(int)
    return [perm[b[i] : b[i + 1]] for i in range(T)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--k-max", type=int, default=28)
    ap.add_argument("--steps", type=int, default=330)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--variant", default="clamp")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64])
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--tag", default="pate_relu_h64_clamp_gamma0.1")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--shuffle",
        type=int,
        default=None,
        help="process this worker's cells in a seeded random order (extra workers collide with in-flight cells only rarely)",
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="process the job list from the T=64 end (extra workers meet the forward ones in the middle)",
    )
    a = ap.parse_args()
    agt_patches.install()
    agt_patches.selftest(a.gamma)
    out = f"{OUT_ROOT}/{a.tag}"
    os.makedirs(out, exist_ok=True)
    train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np = load_california_data()
    assert np.allclose(np.load(f"{STORED}/X_test.npy"), X_test_np), "test split differs from the stored one"
    Xte = X_test_t.double().to(a.device)
    N = len(train_dataset)
    perm = np.random.default_rng(PERM_SEED).permutation(N)
    if a.worker == 0:
        json.dump(
            dict(
                tag=a.tag,
                T_values=a.T,
                k_grid=f"dense 1..{a.k_max}",
                steps=a.steps,
                batch="full shard",
                gamma=a.gamma,
                clip_method=a.variant,
                hidden=a.hidden,
                activation=a.activation,
                lr=CFG["LR"],
                n_train=N,
                perm_seed=PERM_SEED,
                partition="one permutation, consecutive chunks; shard 0 = uniform WOR subsample of N/T",
                t1_arm="relu_h64_clamp_gamma0.1/bs16512_k*.npz (same schedule: 330 full-batch steps)",
                y_min=Y_MIN,
                y_max=Y_MAX,
                git_sha=git_sha(),
                host=platform.node(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
            open(f"{out}/config.json", "w"),
            indent=1,
        )
        np.save(f"{out}/perm.npy", perm)
    X_all, y_all = train_dataset.tensors
    jobs = [(T, i, k) for T in sorted(a.T) for i in range(T) for k in range(1, a.k_max + 1)]
    jobs = jobs[a.worker :: a.n_workers]
    if a.reverse:
        jobs = jobs[::-1]
    if a.shuffle is not None:
        jobs = [jobs[j] for j in np.random.default_rng(a.shuffle).permutation(len(jobs))]
    print(f"worker {a.worker}/{a.n_workers}: {len(jobs)} cells", flush=True)
    for T, i, k in jobs:
        d = f"{out}/T{T}"
        os.makedirs(d, exist_ok=True)
        path = f"{d}/shard{i}_k{k}.npz"
        if os.path.exists(path):
            continue
        idx = shards_for(N, T, perm)[i]
        m = len(idx)
        ds = TensorDataset(X_all[idx], y_all[idx])
        bm, secs = train_cell(
            ds, m, k, a.steps, a.gamma, a.variant, a.device, hidden=tuple(a.hidden), activation=a.activation
        )
        with torch.no_grad():
            yl, yu = bm.bound_forward(Xte, Xte)
            yn = bm.forward(Xte)
        pl = [p.detach().cpu().double().numpy() for p in bm.param_l]
        pn = [p.detach().cpu().double().numpy() for p in bm.param_n]
        pu = [p.detach().cpu().double().numpy() for p in bm.param_u]
        arrays = {f"param_l_{j}": pl[j] for j in range(len(pl))}
        arrays.update({f"param_n_{j}": pn[j] for j in range(len(pn))})
        arrays.update({f"param_u_{j}": pu[j] for j in range(len(pu))})
        np.savez_compressed(
            path,
            **arrays,
            y_l=yl.squeeze().cpu().numpy(),
            y_u=yu.squeeze().cpu().numpy(),
            y_n=yn.squeeze().cpu().numpy(),
            seconds=secs,
            T=T,
            shard=i,
            k=k,
            m=m,
            shard_idx=idx,
            steps=a.steps,
            epochs=a.steps,
            bs=m,
            gamma=a.gamma,
            clip_method=a.variant,
        )
        ylc, yuc = (
            np.clip(yl.squeeze().cpu().numpy(), Y_MIN, Y_MAX),
            np.clip(yu.squeeze().cpu().numpy(), Y_MIN, Y_MAX),
        )
        print(
            f"[w{a.worker}] T{T}/shard{i}_k{k} m={m} {secs:5.0f}s out w {np.mean(yuc - ylc):.3f} sat {np.mean((yuc - ylc) >= (Y_MAX - Y_MIN) - 1e-9):.2f} "
            f"nom MSE {np.mean((yn.squeeze().cpu().numpy() - y_test_t.numpy()) ** 2):.4f}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
