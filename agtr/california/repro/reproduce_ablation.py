"""Independent end-to-end reproduction of the California Housing row of Figure 2.

Unlike ``repro_figure2.py``, which reads the stored arrays, this retrains every
(batch size, k) cell from scratch and recomputes V_k, the smooth sensitivity and
all three panels.  If the result differs from the published figure, the gap is
in the procedure, not in the plotting.

The training loop is transcribed from
``archive/attic/original_scripts/run_ca_ablation.py:73-165`` -- the script that produced
``agtr/california/values/ablation/`` -- rather than from the refactored
``experiments/common/training.py``, because the two differ in the nominal (k=0)
arm: the original uses BoundedMSELoss(reduction='sum') with the Huber clamp and
BoundedSGD, the refactor uses nn.MSELoss with torch.optim.SGD.

V_k is the radius about the nominal, ``max(|y_l - y_nom|, |y_u - y_nom|)``, as
in run_ca_ablation.py:323 and as the paper's A^k requires.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from abstract_gradient_training.bounded_losses import BoundedMSELoss
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD

from common.data import load_california_data

OUT = "agtr/california/repro/runs"
BATCH_SIZES = [250, 500, 1000, 2000, 4000]
K_VALUES = sorted(set([0] + list(range(1, 11)) + list(range(13, 31, 3))))
BASE_BATCH_SIZE, BASE_EPOCHS = 1000, 20
LR, FREQ, SEED = 0.01, 1, 42
CLIP_C, HUBER_DELTA = 1.0, 1.0
Y_MIN, Y_MAX = -5.0, 5.0
GS_VAL = Y_MAX - Y_MIN
BETA = 1.0 / 6.0


def create_model(input_dim=8):
    """run_ca_ablation.py:63-70."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def train_cell(k, train_dataset, epochs, batch_size, lr=LR, seed=SEED, freq=FREQ):
    """run_ca_ablation.py:73-165, transcribed.

    k == 0 accumulates unclipped nominal gradients and steps with a degenerate
    interval; k > 0 clips each sample to joint l2 norm CLIP_C and widens the
    descent direction by the analytic +/- k*2*C/B.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)

    bounded_model = IntervalBoundedModel(create_model(8))
    optimizer = BoundedSGD(bounded_model, learning_rate=lr)
    criterion = BoundedMSELoss(reduction="sum")

    for _epoch in range(epochs):
        accum_n, total_bs = None, 0
        for i, (x_batch, y_batch) in enumerate(loader):
            curr_bs = x_batch.size(0)

            logits_n = bounded_model.forward(x_batch, retain_intermediate=True)
            loss_grad_n = criterion.backward(logits_n.squeeze(), y_batch)
            loss_grad_n = torch.clamp(loss_grad_n, -HUBER_DELTA, HUBER_DELTA).unsqueeze(1)
            grads_n = bounded_model.backward(loss_grad_n)

            if k == 0:
                contrib = [gr.sum(dim=0) for gr in grads_n]
            else:
                total_norm = torch.zeros(curr_bs)
                for gr in grads_n:
                    total_norm += gr.view(curr_bs, -1).norm(2, dim=1) ** 2
                clip_coef = torch.clamp(CLIP_C / (total_norm.sqrt() + 1e-6), max=1.0)
                contrib = [
                    (gr * clip_coef.view(*([curr_bs] + [1] * (gr.dim() - 1)))).sum(dim=0) for gr in grads_n
                ]

            accum_n = contrib if accum_n is None else [a + c for a, c in zip(accum_n, contrib)]
            total_bs += curr_bs

            if (i + 1) % freq == 0 or (i + 1) == len(loader):
                mean_n = [gr / total_bs for gr in accum_n]
                if k == 0:
                    optimizer.step(mean_n, mean_n, mean_n)
                else:
                    delta = k * 2 * CLIP_C / total_bs
                    optimizer.step([gr - delta for gr in mean_n], mean_n, [gr + delta for gr in mean_n])
                accum_n, total_bs = None, 0

    return bounded_model


def run():
    os.makedirs(OUT, exist_ok=True)
    train_dataset, X_test_t, y_test_t, y_mean, y_std, _ = load_california_data()
    y_test_real = y_test_t.numpy() * y_std + y_mean
    x_sample = X_test_t[0].unsqueeze(0)
    n_test = X_test_t.size(0)

    print(
        f"n_train={len(train_dataset)} n_test={n_test} y_std={y_std:.6f} GS_real={GS_VAL * y_std:.6f}",
        flush=True,
    )
    print(f"k grid: {K_VALUES}\n", flush=True)

    results = {}
    for bs in BATCH_SIZES:
        epochs = max(1, int(BASE_EPOCHS * (bs / BASE_BATCH_SIZE)))
        steps = epochs * (len(train_dataset) // bs)
        print(f"=== BS={bs}  epochs={epochs}  steps={steps} ===", flush=True)
        t0 = time.time()

        models = {}
        for k in K_VALUES:
            models[k] = train_cell(k, train_dataset, epochs, bs)
            print(f"    k={k:<3} done ({time.time() - t0:.0f}s)", flush=True)

        with torch.no_grad():
            y_nom = np.clip(models[0].forward(X_test_t).squeeze().numpy(), Y_MIN, Y_MAX)
            nom_mse = float(np.mean((y_nom * y_std + y_mean - y_test_real) ** 2))

            V = np.zeros((n_test, len(K_VALUES)))
            YL = np.zeros_like(V)
            YU = np.zeros_like(V)
            for j, k in enumerate(K_VALUES):
                if k == 0:
                    YL[:, j] = YU[:, j] = y_nom
                    continue
                yl, yu = models[k].bound_forward(X_test_t, X_test_t)
                yl = np.clip(yl.flatten().numpy(), Y_MIN, Y_MAX)
                yu = np.clip(yu.flatten().numpy(), Y_MIN, Y_MAX)
                YL[:, j], YU[:, j] = yl, yu
                V[:, j] = np.maximum(np.abs(yl - y_nom), np.abs(yu - y_nom))

            tail = GS_VAL * np.exp(-BETA * (max(K_VALUES) + 1))
            cand = np.stack([V[:, j] * np.exp(-BETA * k) for j, k in enumerate(K_VALUES) if k != 0])
            S = np.maximum(cand.max(axis=0), tail) * y_std

            # staircase at the single plotted test point, run_ca_ablation.py:167-190
            y_nom_s = float(np.clip(models[0].forward(x_sample).item(), Y_MIN, Y_MAX))
            stair = [(0, 0.0)]
            for k in K_VALUES:
                if k == 0:
                    continue
                yl, yu = models[k].bound_forward(x_sample, x_sample)
                yl = float(np.clip(yl.item(), Y_MIN, Y_MAX))
                yu = float(np.clip(yu.item(), Y_MIN, Y_MAX))
                stair.append((k, max(abs(yl - y_nom_s), abs(yu - y_nom_s))))

        np.savez_compressed(
            f"{OUT}/repro_bs{bs}.npz",
            V=V,
            YL=YL,
            YU=YU,
            S=S,
            y_nom=y_nom,
            k_values=np.array(K_VALUES),
            stair_ks=np.array([a for a, _ in stair]),
            stair_vs=np.array([b for _, b in stair]),
        )
        results[bs] = dict(
            nom_mse=nom_mse,
            avg_S=float(S.mean()),
            frac_Vk1_at_GS=float((V[:, 1] >= GS_VAL * 0.999).mean()),
            avg_V1=float(V[:, 1].mean()),
            seconds=time.time() - t0,
        )
        print(
            f"  nominal MSE = {nom_mse:.5f}   avg S = {S.mean():.6f}   "
            f"avg V^1 = {V[:, 1].mean():.5f}   ({time.time() - t0:.0f}s)\n",
            flush=True,
        )

    with open(f"{OUT}/repro_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
