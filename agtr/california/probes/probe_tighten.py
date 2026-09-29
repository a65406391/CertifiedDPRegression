# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Probe: how much tighter does a shard certificate get with fewer steps / different lr, and what does it cost
in nominal accuracy?  Trains shard 0 of the T-partition (same permutation as pate_matched/train.py) for
steps x lr configurations with dense k = 1..K_PROBE, stores cells like the main run under
values/agt_2026-09-14/probe_tighten/T<T>_s<steps>_lr<lr>/shard0_k<k>.npz.

    CUDA_VISIBLE_DEVICES=g PYTHONPATH=$PWD python -m agtr.california.pate.probe_tighten --worker w --n-workers n
"""

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from agtr.california.certify import OUT_ROOT, Y_MAX, Y_MIN, train_cell
from common import agt_patches
from agtr.california.pate.train import shards_for, PERM_SEED
from common.data import load_california_data

logging.disable(logging.WARNING)
K_PROBE = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    agt_patches.install()
    agt_patches.selftest(a.gamma)
    train_dataset, X_test_t, y_test_t, y_mean, y_std, _ = load_california_data()
    Xte = X_test_t.double().to(a.device)
    N = len(train_dataset)
    perm = np.random.default_rng(PERM_SEED).permutation(N)
    X_all, y_all = train_dataset.tensors
    jobs = [(T, s, lr, k) for T in a.T for s in a.steps for lr in a.lrs for k in range(1, K_PROBE + 1)][
        a.worker :: a.n_workers
    ]
    print(f"worker {a.worker}: {len(jobs)} cells", flush=True)
    for T, s, lr, k in jobs:
        d = f"{OUT_ROOT}/probe_tighten/T{T}_s{s}_lr{lr:g}"
        os.makedirs(d, exist_ok=True)
        path = f"{d}/shard0_k{k}.npz"
        if os.path.exists(path):
            continue
        idx = shards_for(N, T, perm)[0]
        ds = TensorDataset(X_all[idx], y_all[idx])
        bm, secs = train_cell(
            ds, len(idx), k, s, a.gamma, "clamp", a.device, hidden=(64,), activation="relu", lr=lr
        )
        with torch.no_grad():
            yl, yu = bm.bound_forward(Xte, Xte)
            yn = bm.forward(Xte)
        np.savez_compressed(
            path,
            y_l=yl.squeeze().cpu().numpy(),
            y_u=yu.squeeze().cpu().numpy(),
            y_n=yn.squeeze().cpu().numpy(),
            seconds=secs,
            T=T,
            shard=0,
            k=k,
            m=len(idx),
            steps=s,
            lr=lr,
            gamma=a.gamma,
            shard_idx=idx,
        )
        w = np.clip(yu.squeeze().cpu().numpy(), Y_MIN, Y_MAX) - np.clip(
            yl.squeeze().cpu().numpy(), Y_MIN, Y_MAX
        )
        print(
            f"[w{a.worker}] T{T} s{s} lr{lr:g} k{k}: {secs:4.0f}s out w {w.mean():.3f} nom MSE {np.mean((yn.squeeze().cpu().numpy() - y_test_t.numpy()) ** 2):.4f}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
