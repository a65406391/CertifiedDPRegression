# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
PATE Comparison: Global Sensitivity (Laplace) vs Smooth Sensitivity (Cauchy).

For each ensemble size T, trains two PATE ensembles:
  1. Standard teachers -> private prediction with Lap(GS / (T * eps))
  2. AGT teachers      -> private prediction with Cauchy(6 * SS^beta_PATE / eps)

Then evaluates both across a grid of epsilon values and saves MSE results.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from common.data import make_dataloader, simulated_linear_data
from common.baselines import PateRegressor
from agtr.linear.config import PATE as CFG


def make_datasets(seed=42):
    """Generate clipped train/test data and return as TensorDataset + numpy arrays."""
    X_train, y_train = simulated_linear_data(
        n_samples=CFG["N_TRAIN"],
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        x_range=CFG["X_RANGE"],
        seed=seed,
        dim=1,
    )
    X_test, y_test = simulated_linear_data(
        n_samples=CFG["N_TEST"],
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        x_range=CFG["X_RANGE"],
        seed=seed + 1000,
        dim=1,
    )
    y_train = np.clip(y_train, -CFG["CLIP_Y"], CFG["CLIP_Y"])
    y_test = np.clip(y_test, -CFG["CLIP_Y"], CFG["CLIP_Y"])

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    return train_ds, X_test, y_test


def model_fn():
    return nn.Sequential(nn.Linear(1, 1))


def run_pate_comparison():
    print("Running PATE: Global vs Smooth Sensitivity Comparison")
    out_dir = CFG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    train_ds, X_test, y_test = make_datasets()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = make_dataloader(test_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False)

    epsilons = CFG["EPSILONS"]
    n_trials = CFG["N_TRIALS"]
    gs_val = CFG["GS_VAL"]

    np.save(f"{out_dir}/epsilons.npy", epsilons)
    np.save(f"{out_dir}/y_test.npy", y_test)

    for T in CFG["ENSEMBLE_SIZES"]:
        print(f"\n{'=' * 60}")
        print(f"Ensemble size T={T}")
        print(f"{'=' * 60}")

        pate = PateRegressor(ensemble_size=T, global_sensitivity=gs_val)

        # =================================================================
        # 1. Train standard teachers (for global sensitivity Laplace PATE)
        # =================================================================
        global_dir = f"{out_dir}/T{T}_global"
        print(f"\n--- Training standard teachers (T={T}) ---")
        pate.train_ensemble(
            train_dataset=train_ds,
            model_fn=model_fn,
            epochs=CFG["EPOCHS"],
            lr=CFG["LR"],
            save_path=global_dir,
            with_smooth_sensitivity=False,
            batch_size=CFG["BATCH_SIZE"],
        )

        # =================================================================
        # 2. Train AGT teachers (for smooth sensitivity Cauchy PATE)
        # =================================================================
        smooth_dir = f"{out_dir}/T{T}_smooth"
        print(f"\n--- Training AGT teachers (T={T}, k_values={CFG['K_VALUES']}) ---")
        pate.train_ensemble(
            train_dataset=train_ds,
            model_fn=model_fn,
            epochs=CFG["AGT_EPOCHS"],
            lr=CFG["AGT_LR"],
            save_path=smooth_dir,
            with_smooth_sensitivity=True,
            batch_size=CFG["AGT_BATCH_SIZE"],
            k_values=CFG["K_VALUES"],
            clip_C=CFG["CLIP_C"],
            huber_delta=CFG["HUBER_DELTA"],
        )

        # =================================================================
        # 3. Evaluate: non-private baseline
        # =================================================================
        print(f"\n--- Evaluating non-private mean (T={T}) ---")
        nonprivate_result = pate.evaluate(
            test_loader=test_loader,
            models_dir=global_dir,
            model_fn=model_fn,
            mode="nonprivate",
        )
        print(f"  Non-private MSE: {nonprivate_result['mse_mean']:.6f}")
        np.save(f"{out_dir}/T{T}_nonprivate_mse.npy", nonprivate_result["mse_mean"])

        # =================================================================
        # 4. Evaluate: global sensitivity (Laplace) across epsilons
        # =================================================================
        print(f"\n--- Evaluating PATE-Global (Laplace) across epsilons ---")
        global_mse_mean = []
        global_mse_std = []
        for eps in epsilons:
            result = pate.evaluate(
                test_loader=test_loader,
                models_dir=global_dir,
                epsilon=eps,
                model_fn=model_fn,
                mode="global",
                n_trials=n_trials,
            )
            global_mse_mean.append(result["mse_mean"])
            global_mse_std.append(result["mse_std"])
            print(f"  eps={eps}: MSE={result['mse_mean']:.6f} ± {result['mse_std']:.6f}")

        np.save(f"{out_dir}/T{T}_global_mse_mean.npy", global_mse_mean)
        np.save(f"{out_dir}/T{T}_global_mse_std.npy", global_mse_std)

        # =================================================================
        # 5. Evaluate: smooth sensitivity (Cauchy) across epsilons
        # =================================================================
        print(f"\n--- Evaluating PATE-Smooth (Cauchy) across epsilons ---")
        smooth_mse_mean = []
        smooth_mse_std = []
        for eps in epsilons:
            result = pate.evaluate(
                test_loader=test_loader,
                models_dir=smooth_dir,
                epsilon=eps,
                mode="smooth",
                n_trials=n_trials,
                k_values=CFG["K_VALUES"],
            )
            smooth_mse_mean.append(result["mse_mean"])
            smooth_mse_std.append(result["mse_std"])
            print(f"  eps={eps}: MSE={result['mse_mean']:.6f} ± {result['mse_std']:.6f}")

        np.save(f"{out_dir}/T{T}_smooth_mse_mean.npy", smooth_mse_mean)
        np.save(f"{out_dir}/T{T}_smooth_mse_std.npy", smooth_mse_std)

    print("\nDone. Results saved to:", out_dir)


if __name__ == "__main__":
    run_pate_comparison()
