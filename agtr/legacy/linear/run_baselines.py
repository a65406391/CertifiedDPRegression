# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
Linear Regression Baseline Comparison.

Refactored from run_lin_baselines.py.

Compares AGT against Functional Mechanism, DP-SGD (Laplace/Gaussian), and Sharding.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from common.baselines import (
    run_functional_mechanism,
    run_gaussian_sgd,
    run_laplace_sgd,
    run_sharding,
)
from common.data import simulated_linear_data
from agtr.linear.config import BASELINES as CFG


def get_dataloader(seed=42):
    """Create training DataLoader with clipped data."""
    X, y = simulated_linear_data(
        n_samples=CFG["N_SAMPLES"],
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        x_range=CFG["X_RANGE"],
        seed=seed,
        dim=1,
    )
    X = np.clip(X, -CFG["CLIP_X"], CFG["CLIP_X"])
    y = np.clip(y, -CFG["CLIP_Y"], CFG["CLIP_Y"])

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)
    return loader, tensor_x, tensor_y


def main():
    print("Running Linear Baselines...")
    out_dir = CFG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    epsilons = CFG["EPSILONS"]
    n_trials = CFG["N_TRIALS"]

    # Test data (shared across all baselines)
    X_test, y_test = simulated_linear_data(
        n_samples=2000,
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        x_range=CFG["X_RANGE"],
        seed=100,
        dim=1,
    )

    # Training data for baselines that need raw numpy
    X_train, y_train = simulated_linear_data(
        n_samples=CFG["N_SAMPLES"],
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        x_range=CFG["X_RANGE"],
        seed=42,
        dim=1,
    )
    X_train = np.clip(X_train, -CFG["CLIP_X"], CFG["CLIP_X"])
    y_train = np.clip(y_train, -CFG["CLIP_Y"], CFG["CLIP_Y"])

    # 1. Functional Mechanism
    func_mses_mean = []
    func_mses_std = []
    print("Functional Mechanism...")
    for eps in epsilons:
        trials = [
            run_functional_mechanism(
                X_train,
                y_train,
                X_test,
                y_test,
                epsilon=eps,
                clip_x=CFG["CLIP_X"],
                clip_y=CFG["CLIP_Y"],
                seed=i,
            )
            for i in range(n_trials)
        ]
        func_mses_mean.append(np.mean(trials))
        func_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/func_mean.npy", func_mses_mean)
    np.save(f"{out_dir}/func_std.npy", func_mses_std)

    # 2. Laplace DP-SGD
    lap_mses_mean = []
    lap_mses_std = []
    print("Laplace DP-SGD...")
    for eps in epsilons:
        trials = []
        for i in range(n_trials):
            loader, _, _ = get_dataloader(seed=i)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32)
            mse = run_laplace_sgd(
                model_fn=lambda: nn.Linear(1, 1),
                train_loader=loader,
                X_test_t=X_test_t,
                y_test_t=y_test_t,
                epsilon=eps,
                epochs=CFG["EPOCHS"],
                lr=CFG["LR"],
                seed=i,
            )
            trials.append(mse)
        lap_mses_mean.append(np.mean(trials))
        lap_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/lap_mean.npy", lap_mses_mean)
    np.save(f"{out_dir}/lap_std.npy", lap_mses_std)

    # 3. Gaussian DP-SGD (Approx DP)
    gauss_mses_mean = []
    gauss_mses_std = []
    print("Gaussian DP-SGD...")
    for eps in epsilons:
        trials = []
        for i in range(n_trials):
            loader, _, _ = get_dataloader(seed=i)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32)
            mse = run_gaussian_sgd(
                model_fn=lambda: nn.Linear(1, 1),
                train_loader=loader,
                X_test_t=X_test_t,
                y_test_t=y_test_t,
                epsilon=eps,
                epochs=CFG["EPOCHS"],
                lr=CFG["LR"],
                seed=i,
            )
            trials.append(mse)
        gauss_mses_mean.append(np.mean(trials))
        gauss_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/gauss_mean.npy", gauss_mses_mean)
    np.save(f"{out_dir}/gauss_std.npy", gauss_mses_std)

    # 4. Sharding S=10
    shard10_mses_mean = []
    shard10_mses_std = []
    print("Sharding (S=10)...")
    for eps in epsilons:
        trials = [
            run_sharding(
                model_fn=lambda: nn.Linear(1, 1),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                epsilon=eps,
                n_shards=10,
                epochs=CFG["EPOCHS"],
                lr=CFG["LR"],
                batch_size=CFG["BATCH_SIZE"],
                clip_y=CFG["CLIP_Y"],
                seed=i,
            )
            for i in range(n_trials)
        ]
        shard10_mses_mean.append(np.mean(trials))
        shard10_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/shard10_mean.npy", shard10_mses_mean)
    np.save(f"{out_dir}/shard10_std.npy", shard10_mses_std)

    # 5. Sharding S=20
    shard20_mses_mean = []
    shard20_mses_std = []
    print("Sharding (S=20)...")
    for eps in epsilons:
        trials = [
            run_sharding(
                model_fn=lambda: nn.Linear(1, 1),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                epsilon=eps,
                n_shards=20,
                epochs=CFG["EPOCHS"],
                lr=CFG["LR"],
                batch_size=CFG["BATCH_SIZE"],
                clip_y=CFG["CLIP_Y"],
                seed=i,
            )
            for i in range(n_trials)
        ]
        shard20_mses_mean.append(np.mean(trials))
        shard20_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")

    np.save(f"{out_dir}/shard20_mean.npy", shard20_mses_mean)
    np.save(f"{out_dir}/shard20_std.npy", shard20_mses_std)

    np.save(f"{out_dir}/epsilons.npy", epsilons)
    print("Done baselines.")


if __name__ == "__main__":
    main()
