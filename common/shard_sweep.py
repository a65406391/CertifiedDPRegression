"""Find the best number of shards for the shard-and-aggregate baseline, per dataset.

The baseline splits the training set into S disjoint shards, trains S models with no
privacy at all, averages their predictions on the test set and adds Laplace noise to
that average at scale (2*clip_y / S) / eps. S trades two errors against each other:
more shards means less noise but less data per model.

    python -m common.shard_sweep --dataset linear
    python -m common.shard_sweep --dataset california

WHY THIS IS NOT JUST A LOOP OVER baselines.run_sharding
------------------------------------------------------
`run_sharding` retrains every shard for every epsilon, but the trained models do not
depend on epsilon at all -- only the noise added afterwards does. Here each (S, seed)
is trained ONCE and evaluated at every epsilon, which is the same algorithm at a
fraction of the cost. The per-shard construction is otherwise identical to
`baselines.run_sharding`: same permutation, same optimiser, same
`max(1, batch_size // S)` shard batch, same sensitivity.

TWO ACCOUNTINGS, AND THEY DISAGREE BY ORDERS OF MAGNITUDE
---------------------------------------------------------
The baseline releases a noisy prediction for every test point. `run_sharding` gives each
one the FULL epsilon, so the reported MSE is a per-query number: releasing all n_test
predictions that way costs n_test * epsilon under basic composition, not epsilon.

AGT and DP-SGD release a model once and then predict for free, so comparing them against
a per-query baseline is not a comparison of equals. Both are computed here:

    per_query   scale = (2*clip_y / S) / eps                 what run_sharding does
    composed    scale = (2*clip_y / S) * n_test / eps        the whole test set for eps

`composed` is the one that can sit on the same axes as AGT. It is the default for the
arrays the plotters read; `per_query` is kept because it is what the existing baseline
computes and the difference is worth seeing.

TWO NOTIONS OF "BEST", AND ONLY ONE OF THEM IS HONEST
-----------------------------------------------------
`best_fixed` is the single S with the lowest mean MSE across the whole epsilon grid --
one configuration, chosen once, reportable as a baseline.
`envelope` takes the best S separately at each epsilon. It is an upper bound that no
single configuration attains, and it is recorded for reference, never plotted as if it
were an achievable arm. The blobs study uses the same distinction for PATE.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _train_shards(X, y, model_fn, *, n_shards, epochs, lr, batch_size, seed):
    """Train S non-private models on disjoint shards. Mirrors baselines.run_sharding."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(X)
    shard_size = n // n_shards
    idx = np.random.permutation(n)
    models = []
    for s in range(n_shards):
        lo = s * shard_size
        hi = lo + shard_size if s < n_shards - 1 else n
        sl = idx[lo:hi]
        model = model_fn()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        Xs = torch.tensor(X[sl], dtype=torch.float32)
        ys = torch.tensor(y[sl], dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xs, ys), batch_size=max(1, batch_size // n_shards), shuffle=True)
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
        models.append(model)
    return models


def _eval(preds, y_test, *, clip_y, n_shards, eps, rng, composed):
    """Averaged predictions + Laplace noise -> test MSE.

    composed=False: each test point gets the full epsilon (what run_sharding does).
    composed=True:  epsilon is split across the n_test released predictions.
    """
    scale = (2.0 * clip_y / n_shards) / eps
    if composed:
        scale *= len(y_test)
    noisy = preds + rng.laplace(0.0, scale, preds.shape)
    return float(np.mean((noisy.ravel() - y_test.ravel()) ** 2))


def load(dataset):
    """Returns (X_train, y_train, X_test, y_test, model_fn, cfg)."""
    if dataset == "linear":
        from agtr.linear.config import BASELINES as CFG
        from common.data import simulated_linear_data

        X, y = simulated_linear_data(
            n_samples=CFG["N_SAMPLES"],
            slope=CFG["SLOPE"],
            intercept=CFG["INTERCEPT"],
            x_range=CFG["X_RANGE"],
            seed=42,
            dim=1,
        )
        Xte, yte = simulated_linear_data(
            n_samples=2000,
            slope=CFG["SLOPE"],
            intercept=CFG["INTERCEPT"],
            x_range=CFG["X_RANGE"],
            seed=100,
            dim=1,
        )
        X = np.clip(X, -CFG["CLIP_X"], CFG["CLIP_X"])
        y = np.clip(y, -CFG["CLIP_Y"], CFG["CLIP_Y"])
        return X, y, Xte, yte, (lambda: nn.Linear(1, 1)), CFG

    from agtr.california.config import BASELINES as CFG
    from common.data import load_california_data
    from common.models import create_mlp

    tr, Xte_t, yte_t, _, _, _ = load_california_data(clip_x=CFG["CLIP_X"], clip_y=CFG["CLIP_Y"])
    X = tr.tensors[0].numpy()
    y = tr.tensors[1].numpy()
    if y.ndim == 1:
        y = y[:, None]
    yte = yte_t.numpy()
    if yte.ndim == 1:
        yte = yte[:, None]
    return X, y, Xte_t.numpy(), yte, (lambda: create_mlp(8, [64, 64])), CFG


def main(a):
    X, y, Xte, yte, model_fn, CFG = load(a.dataset)
    eps = list(CFG["EPSILONS"])
    grid = [s for s in a.shards if s <= len(X)]
    print(
        f"{a.dataset}: n_train={len(X)}, n_test={len(Xte)}, "
        f"epochs={CFG['EPOCHS']}, lr={CFG['LR']}, clip_y={CFG['CLIP_Y']}"
    )
    print(f"S grid {grid}   eps grid {eps}   seeds {a.seeds}\n", flush=True)

    mse = np.zeros((len(grid), len(eps), a.seeds))  # composed accounting
    mse_pq = np.zeros_like(mse)  # per-query accounting
    nominal = np.zeros((len(grid), a.seeds))  # the average with no noise
    for i, S in enumerate(grid):
        for t in range(a.seeds):
            models = _train_shards(
                X,
                y,
                model_fn,
                n_shards=S,
                epochs=CFG["EPOCHS"],
                lr=CFG["LR"],
                batch_size=CFG["BATCH_SIZE"],
                seed=42 + t,
            )
            with torch.no_grad():
                p = np.mean([m(torch.tensor(Xte, dtype=torch.float32)).numpy() for m in models], axis=0)
            nominal[i, t] = float(np.mean((p.ravel() - yte.ravel()) ** 2))
            for j, e in enumerate(eps):
                kw = dict(clip_y=CFG["CLIP_Y"], n_shards=S, eps=e)
                mse[i, j, t] = _eval(p, yte, **kw, composed=True, rng=np.random.RandomState(1000 * t + j))
                mse_pq[i, j, t] = _eval(
                    p, yte, **kw, composed=False, rng=np.random.RandomState(9000 + 1000 * t + j)
                )
        print(
            f"  S={S:<4} noiseless={nominal[i].mean():.4f}   composed "
            + " ".join(f"{v:.3g}" for v in mse[i].mean(1))
            + "   | per-query "
            + " ".join(f"{v:.3g}" for v in mse_pq[i].mean(1)),
            flush=True,
        )

    mean = mse.mean(2)  # (S, eps), composed
    # Selected on the GEOMETRIC mean, not the arithmetic one: MSE spans four orders of
    # magnitude across this epsilon grid, so an arithmetic mean is decided entirely by
    # the smallest epsilon and picks a shard count that is terrible everywhere else.
    score = np.log(mean).mean(1)
    best_fixed_i = int(np.argmin(score))
    env_i = mean.argmin(0)  # oracle per epsilon

    print(
        f"\nbest single S = {grid[best_fixed_i]} "
        f"(lowest geometric-mean MSE across the epsilon grid; "
        f"arithmetic-mean pick would be S={grid[int(np.argmin(mean.mean(1)))]})"
    )
    print(f"{'eps':>8} {'best-fixed':>12} {'oracle S':>9} {'oracle MSE':>12}")
    for j, e in enumerate(eps):
        print(f"{e:>8} {mean[best_fixed_i, j]:>12.4f} {grid[env_i[j]]:>9} {mean[env_i[j], j]:>12.4f}")

    out = CFG["OUT_DIR"]
    os.makedirs(out, exist_ok=True)

    # The plotters read whatever epsilon grid the stored baseline arrays were computed
    # on, and it is not always the grid in config.py -- California's stored arrays use
    # [0.1, 1, 10] while its config declares seven points. Align to the stored grid when
    # one exists, so the series this writes can actually be plotted beside the others.
    keep = list(range(len(eps)))
    if os.path.exists(f"{out}/epsilons.npy"):
        stored = list(np.load(f"{out}/epsilons.npy"))
        idx = [next((j for j, e in enumerate(eps) if abs(e - se) < 1e-12), None) for se in stored]
        if all(i is not None for i in idx):
            keep = idx
            print(f"\naligning saved arrays to the stored grid {stored} (config grid has {len(eps)} points)")
        else:
            print(
                f"\nWARNING: stored grid {stored} is not a subset of the config grid; "
                f"writing the config grid instead"
            )
    np.savez_compressed(
        f"{out}/shard_sweep.npz",
        shards=np.array(grid),
        epsilons=np.array(eps),
        mse=mse,
        mse_per_query=mse_pq,
        noiseless=nominal,
    )
    np.save(f"{out}/shard_best_mean.npy", mean[best_fixed_i][keep])
    np.save(f"{out}/shard_best_std.npy", mse[best_fixed_i].std(1)[keep])
    np.save(f"{out}/shard_envelope_mean.npy", mean[env_i, np.arange(len(eps))][keep])
    json.dump(
        dict(
            dataset=a.dataset,
            shard_grid=grid,
            epsilons=eps,
            best_fixed_S=grid[best_fixed_i],
            best_fixed_rule="min geometric-mean MSE over the epsilon grid",
            arithmetic_mean_S=grid[int(np.argmin(mean.mean(1)))],
            envelope_S=[grid[i] for i in env_i],
            noiseless_mse={str(s): float(nominal[i].mean()) for i, s in enumerate(grid)},
            seeds=a.seeds,
            n_test=int(len(yte)),
            accounting="composed: epsilon split across the n_test released predictions (basic composition)",
            best_fixed_S_per_query=grid[int(np.argmin(np.log(mse_pq.mean(2)).mean(1)))],
        ),
        open(f"{out}/shard_sweep.json", "w"),
        indent=1,
    )
    if a.per_query:
        pq = mse_pq.mean(2)
        bi = int(np.argmin(np.log(pq).mean(1)))
        np.save(f"{out}/shard_best_mean_per_query.npy", pq[bi][keep])
        np.save(f"{out}/shard_best_std_per_query.npy", mse_pq[bi].std(1)[keep])
    print(
        f"\nwrote {out}/shard_sweep.{{npz,json}} and shard_best_*.npy "
        f"(composed accounting; n_test={len(yte)})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["linear", "california"], required=True)
    ap.add_argument("--shards", type=int, nargs="+", default=[2, 4, 5, 8, 10, 16, 20, 32, 40, 64])
    ap.add_argument(
        "--per-query",
        action="store_true",
        dest="per_query",
        help="also write shard_best_*_per_query.npy for the optimistic accounting",
    )
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args())
