# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
California Housing Baseline Comparison.

Refactored from run_ca_baselines.py.
"""

import torch
import numpy as np
import os

from agtr.california.config import BASELINES as CFG
from common.data import load_california_data
from common.models import create_mlp
from common.baselines import (
    run_functional_mechanism,
    run_laplace_sgd,
    run_gaussian_sgd,
)


def main():
    print("Running California Baselines...")
    out_dir = CFG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    epsilons = CFG["EPSILONS"]
    n_trials = CFG["N_TRIALS"]

    def model_fn():
        return create_mlp(8, [64, 64])

    # Load data (with clipping for baselines)
    train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np = load_california_data(
        clip_x=CFG["CLIP_X"], clip_y=CFG["CLIP_Y"]
    )

    # Get raw numpy for functional mechanism
    X_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy().ravel()

    # Ensure y shapes for eval
    if y_test_t.ndim == 1:
        y_test_t = y_test_t.unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)

    # 1. Functional Mechanism (Linear Baseline)
    func_means = []
    func_stds = []
    print("Functional Mechanism (Linear)...")
    for eps in epsilons:
        trials = [
            run_functional_mechanism(
                X_train,
                y_train,
                X_test_t.numpy(),
                y_test_t.numpy(),
                eps,
                CFG["CLIP_X"],
                CFG["CLIP_Y"],
                seed=i,
            )
            for i in range(n_trials)
        ]
        func_means.append(np.mean(trials))
        func_stds.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/func_mean.npy", func_means)
    np.save(f"{out_dir}/func_std.npy", func_stds)

    # 2. Laplace DP-SGD (MLP)
    lap_means = []
    lap_stds = []
    print("Laplace DP-SGD (MLP)...")
    for eps in epsilons:
        trials = [
            run_laplace_sgd(model_fn, train_loader, X_test_t, y_test_t, eps, CFG["EPOCHS"], CFG["LR"], seed=i)
            for i in range(n_trials)
        ]
        lap_means.append(np.mean(trials))
        lap_stds.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/lap_mean.npy", lap_means)
    np.save(f"{out_dir}/lap_std.npy", lap_stds)

    # 3. Gaussian DP-SGD (MLP)
    gauss_means = []
    gauss_stds = []
    print("Gaussian DP-SGD (MLP)...")
    for eps in epsilons:
        trials = [
            run_gaussian_sgd(
                model_fn, train_loader, X_test_t, y_test_t, eps, CFG["EPOCHS"], CFG["LR"], seed=i
            )
            for i in range(n_trials)
        ]
        gauss_means.append(np.mean(trials))
        gauss_stds.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/gauss_mean.npy", gauss_means)
    np.save(f"{out_dir}/gauss_std.npy", gauss_stds)

    np.save(f"{out_dir}/epsilons.npy", epsilons)
    print("Done CA Baselines.")


if __name__ == "__main__":
    main()
