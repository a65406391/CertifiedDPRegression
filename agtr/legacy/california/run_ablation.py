# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
California Housing Batch Ablation Study.

Refactored from run_ca_ablation.py.

Studies the trade-off between batch size, sensitivity, and privacy
using Fixed Optimization Steps mode.
"""

import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from agtr.california.config import ABLATION as CFG
from common.data import load_california_data
from common.models import create_mlp
from common.training import train_robust_ibp
from common.sensitivity import (
    compute_vk_matrix,
    compute_smooth_sensitivity,
    evaluate_staircase_single,
)
from common.plotting import apply_style, save_figure, ensure_fig_dir, BATCH_COLORS

apply_style("draft")


def run_batch_ablation():
    print("Script Started...", flush=True)
    print("Running California Housing Batch Ablation...", flush=True)

    out_dir = CFG["OUT_DIR"]
    fig_dir = CFG["FIG_DIR"]
    plot_prefix = CFG["PLOT_PREFIX"]
    os.makedirs(out_dir, exist_ok=True)
    ensure_fig_dir(fig_dir)

    batch_sizes = CFG["BATCH_SIZES"]
    k_values = CFG["K_VALUES"]
    Y_MIN, Y_MAX = CFG["Y_MIN"], CFG["Y_MAX"]
    GS_VAL = Y_MAX - Y_MIN
    beta = CFG["BETA"]
    colors = BATCH_COLORS

    # Load Data
    train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np = load_california_data()

    # Save Raw Data
    np.save(f"{out_dir}/X_test.npy", X_test_np)
    np.save(f"{out_dir}/y_test.npy", y_test_t.numpy())
    np.save(f"{out_dir}/y_mean.npy", y_mean)
    np.save(f"{out_dir}/y_std.npy", y_std)
    np.save(f"{out_dir}/k_values.npy", k_values)
    np.save(f"{out_dir}/batch_sizes.npy", batch_sizes)

    x_sample_t = X_test_t[0].unsqueeze(0)

    plot_data_staircase = {}
    plot_data_avg_S = {}
    plot_data_nominal_mse = {}

    for i, bs in enumerate(batch_sizes):
        print(f"Training with Batch Size: {bs}...", flush=True)

        if CFG["FIXED_STEPS_MODE"]:
            scaled_epochs = int(CFG["BASE_EPOCHS"] * (bs / CFG["BASE_BATCH_SIZE"]))
            if scaled_epochs < 1:
                scaled_epochs = 1
            mode_str = "Fixed Steps (Scaled Epochs)"
        else:
            scaled_epochs = CFG["BASE_EPOCHS"]
            mode_str = "Fixed Epochs (Constant Data Passes)"

        print(f"  > BS={bs}, Epochs={scaled_epochs} [{mode_str}]", flush=True)

        t0 = time.time()
        bs_models = {}

        # Model factory
        def model_fn():
            return create_mlp(8, [64, 64])

        # Train nominal
        print("    > Training Nominal...", flush=True)
        nom_model = train_robust_ibp(
            train_dataset,
            model_fn,
            k_val=0,
            batch_size=bs,
            epochs=scaled_epochs,
            lr=CFG["LR"],
            seed=42,
            clip_C=CFG["CLIP_C"],
            huber_delta=CFG["HUBER_DELTA"],
            accumulation_freq=CFG["FREQ"],
        )
        bs_models[0] = nom_model
        torch.save(nom_model, f"{out_dir}/model_bs{bs}.pt")

        # Train robust models
        for k in k_values:
            if k == 0:
                continue
            m = train_robust_ibp(
                train_dataset,
                model_fn,
                k_val=k,
                batch_size=bs,
                epochs=scaled_epochs,
                lr=CFG["LR"],
                seed=42,
                clip_C=CFG["CLIP_C"],
                huber_delta=CFG["HUBER_DELTA"],
                accumulation_freq=CFG["FREQ"],
            )
            bs_models[k] = m

        t1 = time.time()
        print(f"BatchSize={bs} | Total Training Time: {t1 - t0:.2f}s", flush=True)
        np.save(f"{out_dir}/time_bs{bs}.npy", t1 - t0)

        # --- Evaluation ---
        with torch.no_grad():
            logits = nom_model.forward(X_test_t)
            y_pred_cont = logits.squeeze().numpy()
            y_pred_cont = np.clip(y_pred_cont, Y_MIN, Y_MAX)

            np.save(f"{out_dir}/nominal_preds_bs{bs}.npy", y_pred_cont)

            y_pred_real = y_pred_cont * y_std + y_mean
            y_test_real = y_test_t.numpy() * y_std + y_mean
            mse_val = np.mean((y_pred_real - y_test_real) ** 2)
            plot_data_nominal_mse[bs] = mse_val
            print(f"BatchSize={bs} | Nominal MSE: {mse_val:.4f}", flush=True)

            # V_k and sensitivity
            V_k_mat = compute_vk_matrix(
                bs_models, X_test_t, k_values, y_pred_nom=y_pred_cont, y_min=Y_MIN, y_max=Y_MAX
            )

            # Save bounds
            n_test = X_test_t.size(0)
            n_k = len(k_values)
            Y_L_ALL = np.zeros((n_test, n_k))
            Y_U_ALL = np.zeros((n_test, n_k))
            for j, k in enumerate(k_values):
                if k == 0:
                    Y_L_ALL[:, j] = y_pred_cont
                    Y_U_ALL[:, j] = y_pred_cont
                    continue
                model_k = bs_models[k]
                y_l, y_u = model_k.bound_forward(X_test_t, X_test_t)
                Y_L_ALL[:, j] = np.clip(y_l.flatten().numpy(), Y_MIN, Y_MAX)
                Y_U_ALL[:, j] = np.clip(y_u.flatten().numpy(), Y_MIN, Y_MAX)

            np.save(f"{out_dir}/bounds_bs{bs}_yl.npy", Y_L_ALL)
            np.save(f"{out_dir}/bounds_bs{bs}_yu.npy", Y_U_ALL)

            S_vals = compute_smooth_sensitivity(V_k_mat, k_values, beta=beta, gs_val=GS_VAL)
            S_vec_real = S_vals * y_std
            np.save(f"{out_dir}/sensitivity_bs{bs}.npy", S_vec_real)

            plot_data_avg_S[bs] = np.mean(S_vec_real)
            print(f"BatchSize={bs} | Avg Sensitivity (Real): {np.mean(S_vec_real):.4f}", flush=True)

        # Staircase
        res_f = evaluate_staircase_single(bs_models, x_sample_t, k_values, Y_MIN, Y_MAX)
        plot_data_staircase[bs] = list(zip(*res_f))

    # --- PLOTTING ---
    print("Generating Plots...", flush=True)

    # Plot 1: Instability & Smooth Sensitivity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    max_k_val = max(k_values)
    t_grid = np.linspace(0, max_k_val + 5, 500)

    ax1.set_xlabel("Privacy Parameter k")
    ax1.set_ylabel("Instability V_k")
    ax2.set_ylabel("Sensitivity S(t)")
    ax1.set_title("Instability & Smooth Sensitivity")

    for i, bs in enumerate(batch_sizes):
        ks_b, vs_b = plot_data_staircase[bs]
        ks_b = list(ks_b)
        vs_b = list(vs_b)
        c = colors[i % len(colors)]

        np.save(f"{out_dir}/staircase_bs{bs}_ks.npy", ks_b)
        np.save(f"{out_dir}/staircase_bs{bs}_vs.npy", vs_b)

        ax1.step(ks_b, vs_b, where="post", label=f"BS={bs}", color=c, linewidth=2, alpha=0.8)

        indices = np.searchsorted(ks_b, t_grid, side="right") - 1
        indices = np.clip(indices, 0, len(ks_b) - 1)
        v_interp = np.array([vs_b[idx] for idx in indices])
        s_curve = v_interp * np.exp(-beta * t_grid)

        ax2.plot(t_grid, s_curve, "--", color=c, linewidth=1.5, alpha=0.8)

        idx_max = np.argmax(s_curve)
        ax2.plot(t_grid[idx_max], s_curve[idx_max], "*", color=c, markersize=15, markeredgecolor="black")

    ax1.legend(loc="upper right", title="Batch Size")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_staircase.png")
    plt.close()

    # Plot 2: Average Sensitivity Bar Chart
    plt.figure(figsize=(10, 6))
    bar_bs = [str(bs) for bs in batch_sizes if bs in plot_data_avg_S]
    bar_sens = [plot_data_avg_S[bs] for bs in batch_sizes if bs in plot_data_avg_S]
    plt.bar(bar_bs, bar_sens, color=colors[: len(bar_bs)], alpha=0.8, edgecolor="black")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Smooth Sensitivity")
    plt.title("Average Smooth Sensitivity by Batch Size")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_avg_sensitivity_bar.png")
    plt.close()

    # Tradeoff plots
    epsilon_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 500
    delta_dp = 1e-5
    factor = np.sqrt(2 * np.log(1 / delta_dp))
    GS_real = GS_VAL * y_std

    # Plot 3: Approx DP
    plt.figure(figsize=(10, 8))
    gs_sigma_vals = [(GS_real * factor / eps) ** 2 for eps in epsilon_grid]
    plt.plot(epsilon_grid, gs_sigma_vals, "o--", label="Global Sensitivity", color="gray", alpha=0.5)

    for i, bs in enumerate(batch_sizes):
        nom_mse = plot_data_nominal_mse[bs]
        avg_S = plot_data_avg_S[bs]
        c = colors[i % len(colors)]
        mse_mean = []
        mse_std_arr = []
        for eps in epsilon_grid:
            sigma = avg_S * factor / eps
            mse_trials = [
                nom_mse + np.mean(np.random.normal(0, sigma, len(y_test_t)) ** 2) for _ in range(n_trials)
            ]
            mse_mean.append(np.mean(mse_trials))
            mse_std_arr.append(np.std(mse_trials))
        mse_mean = np.array(mse_mean)
        mse_std_arr = np.array(mse_std_arr)
        plt.plot(epsilon_grid, mse_mean, "o-", label=f"BS={bs}", color=c)
        plt.fill_between(
            epsilon_grid, mse_mean - 3 * mse_std_arr, mse_mean + 3 * mse_std_arr, color=c, alpha=0.1
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE (Approx DP)")
    # ylim removed: 0 is invalid on log scale, and MSE often exceeds 1.0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_tradeoff_approx.png")
    plt.close()

    # Plot 4: Pure DP
    plt.figure(figsize=(10, 8))
    gs_mse_median = []
    gs_mse_p25 = []
    gs_mse_p75 = []
    for eps in epsilon_grid:
        scale_gs = GS_real / eps
        mse_trials = [
            np.mean(np.random.standard_cauchy(len(y_test_t)) ** 2 * scale_gs**2) for _ in range(n_trials)
        ]
        gs_mse_median.append(np.median(mse_trials))
        gs_mse_p25.append(np.percentile(mse_trials, 25))
        gs_mse_p75.append(np.percentile(mse_trials, 75))
    plt.plot(epsilon_grid, gs_mse_median, "o--", label="Global Sensitivity", color="gray", alpha=0.5)
    plt.fill_between(epsilon_grid, gs_mse_p25, gs_mse_p75, color="gray", alpha=0.1)

    for i, bs in enumerate(batch_sizes):
        nom_mse = plot_data_nominal_mse[bs]
        avg_S = plot_data_avg_S[bs]
        c = colors[i % len(colors)]
        mse_median = []
        mse_p25 = []
        mse_p75 = []
        for eps in epsilon_grid:
            scale_ss = avg_S / eps
            mse_trials = [
                nom_mse + np.mean(np.random.standard_cauchy(len(y_test_t)) ** 2 * scale_ss**2)
                for _ in range(n_trials)
            ]
            mse_median.append(np.median(mse_trials))
            mse_p25.append(np.percentile(mse_trials, 25))
            mse_p75.append(np.percentile(mse_trials, 75))
        plt.plot(epsilon_grid, mse_median, "o-", label=f"BS={bs}", color=c)
        plt.fill_between(epsilon_grid, mse_p25, mse_p75, color=c, alpha=0.1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE (Pure DP)")
    # ylim removed: 0 is invalid on log scale, and MSE often exceeds 1.0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_tradeoff_pure.png")
    plt.close()

    print("Batch Ablation Complete. All data saved.", flush=True)


if __name__ == "__main__":
    run_batch_ablation()
