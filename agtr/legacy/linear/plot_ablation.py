# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
Linear Ablation Plot (Paper Quality).

Refactored from plot_lin_ablation.py.
"""

import numpy as np

from common import plotting
import torch
import os

from agtr.linear.config import ABLATION as CFG
from common.plotting import (
    apply_style,
    save_figure,
    ensure_fig_dir,
    METHOD_COLORS,
    METHOD_LABELS,
)

import matplotlib.pyplot as plt


#: the trade-off panels are Monte-Carlo over the release noise; seeded so the figure
#: is reproducible and so the combined row can redraw the identical curves
MC_SEED = 0


def release_curve(S_W, epsilons, resid, x_flat, *, mode, beta, gauss_factor, n_trials, seed=MC_SEED):
    """Test MSE against epsilon for one sensitivity S_W, as the panels compute it.

    pure   : Cauchy(scale = S_W / (beta * eps)), median with a 25-75 band
    approx : Gaussian(sigma = S_W * gauss_factor / eps), mean with a +-1 sd band

    Returned as (centre, lo, hi) so both panels and the combined row draw the same
    numbers from the same code rather than from two copies of it.
    """
    rng = np.random.RandomState(seed)
    centre, lo, hi = [], [], []
    for eps in epsilons:
        if mode == "pure":
            eta = rng.standard_cauchy(n_trials) * (S_W / (beta * eps))
        else:
            eta = rng.normal(0, S_W * gauss_factor / eps, n_trials)
        mses = np.array([np.mean((resid + e * x_flat) ** 2) for e in eta])
        if mode == "pure":
            centre.append(np.median(mses))
            lo.append(np.percentile(mses, 25))
            hi.append(np.percentile(mses, 75))
        else:
            m_, sd = mses.mean(), mses.std()
            centre.append(m_)
            # a +-1 sd band runs negative here and the panels are log-scaled, so the
            # lower edge is floored rather than clipped silently to zero by matplotlib
            lo.append(max(m_ - sd, m_ * 1e-2))
            hi.append(m_ + sd)
    return np.array(centre), np.array(lo), np.array(hi)


def method_sensitivity(res, beta):
    """S_W = max_k A_k e^{-beta k}, the smooth sensitivity the release actually uses."""
    return max(res["avg_V_k"][i] * np.exp(-beta * res["k_values"][i]) for i in range(len(res["k_values"])))


def plot_certified_ablation(in_dir=None, out_dir=None, suffix=""):
    print("Plotting Certified Ablation (Paper Quality)...")
    np.random.seed(MC_SEED)
    in_dir = in_dir or CFG["OUT_DIR"]
    out_dir = out_dir or CFG["FIG_DIR"]
    ensure_fig_dir(out_dir)

    apply_style("paper")

    res_file = f"{in_dir}/manual_results.npy"
    if os.path.exists(res_file):
        results = np.load(res_file, allow_pickle=True).item()
    else:
        # Fallback: load individual method_results_{m}.npy files from run_ablation.py
        results = {}
        ordered = ["ibp", "f1", "f2", "f4", "f8", "f16"]
        for m in ordered:
            mf = f"{in_dir}/method_results_{m}.npy"
            if os.path.exists(mf):
                try:
                    results[m] = np.load(mf, allow_pickle=True).item()
                    print(f"  Loaded {m} from {mf}")
                except Exception as e:
                    print(f"  Error loading {mf}: {e}")
        if not results:
            print(f"No results found in {in_dir} (tried manual_results.npy and method_results_*.npy)")
            return

    colors = METHOD_COLORS
    labels = METHOD_LABELS
    ordered_methods = ["ibp", "f1", "f2", "f4", "f8", "f16"]

    # Recover Eval Data
    np.random.seed(999)
    X_sens_eval = np.random.uniform(-2, 2, (50, 1))

    # Re-init manual model to get w0, b0 (Seed 42)
    torch.manual_seed(42)
    mod_dummy = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    w0_nom = mod_dummy[0].weight.item()
    b0_nom = mod_dummy[0].bias.item()

    def get_nominal_pred(x_arr):
        return w0_nom * x_arr + b0_nom

    y_nom_eval = get_nominal_pred(X_sens_eval)

    # Compute Nominal Test MSE
    np.random.seed(12345)
    X_test_mse = np.random.uniform(-2, 2, (1000, 1))
    noise_test = np.random.normal(0, 0.1, (1000, 1))
    y_test_true = 0.5 * X_test_mse + 1.0 + noise_test

    y_test_pred = get_nominal_pred(X_test_mse)
    manual_mse_nom = np.mean((y_test_pred - y_test_true) ** 2)
    print(f"Manual Model Nominal MSE: {manual_mse_nom:.6f}")

    beta = CFG["BETA"]

    # --- Plot 1: Instability Staircase & Smooth Sensitivity ---
    fig, ax1 = plt.subplots(figsize=plotting.FIGSIZE)
    ax2 = ax1.twinx()

    gs_val = CFG["GS_VAL"]

    # First pass: determine max_k_val across all loaded methods
    max_k_val = 6.0
    for m in ordered_methods:
        if m in results and "k_values" in results[m]:
            max_k_val = max(max_k_val, max(results[m]["k_values"]))

    t_grid = np.linspace(0, max_k_val, 1000)

    ax1.set_xlabel("Privacy Parameter k")
    ax1.set_ylabel("Max. Local Sens. Certificate $A^{(k)}$ (dashed)")
    ax2.set_ylabel("Smooth Sensitivity Term (solid)")
    ax1.set_title("Instability & Smooth Sensitivity")

    for m in ordered_methods:
        if m not in results:
            continue
        res = results[m]

        if "avg_V_k" not in res:
            continue

        avg_V_k = res["avg_V_k"]
        k_values = res["k_values"]

        sort_idx = np.argsort(k_values)
        ks = np.array(k_values)[sort_idx]
        vs = np.array(avg_V_k)[sort_idx]

        c = colors[m]
        lbl = labels[m]

        ax1.step(ks, vs, where="post", label=lbl, color=c, linestyle="--", linewidth=3, alpha=0.7)

        indices = np.searchsorted(ks, t_grid, side="right") - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        s_curve = v_interp * np.exp(-beta * t_grid)

        ax2.plot(t_grid, s_curve, "-", color=c, linewidth=4, alpha=0.9)

        idx_max = np.argmax(s_curve)
        ax2.plot(
            t_grid[idx_max], s_curve[idx_max], "*", color=c, markersize=24, markeredgecolor="black", zorder=10
        )

    # Global sensitivity horizontal line on V_k axis
    ax1.axhline(
        y=gs_val,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label=f"Global Sensitivity ({gs_val:.0f})",
        zorder=5,
    )

    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    save_figure(f"{out_dir}/lin_ablation_final_manual{suffix}_staircase.png")
    plt.close()

    # --- Plot 2: Sensitivity Bands ---
    plt.figure(figsize=plotting.FIGSIZE)

    sort_idx = np.argsort(X_sens_eval.flatten())
    X_sorted = X_sens_eval.flatten()[sort_idx]
    y_nom_sorted = y_nom_eval.flatten()[sort_idx]

    for m in ordered_methods:
        if m not in results:
            continue
        res = results[m]
        ks = res["k_values"]
        avg_V_k = res["avg_V_k"]

        S_vals = [avg_V_k[i] * np.exp(-beta * ks[i]) for i in range(len(ks))]
        S_final_W = max(S_vals)

        band_width = S_final_W * np.abs(X_sorted)
        y_low = y_nom_sorted - band_width
        y_high = y_nom_sorted + band_width

        c = colors[m]
        plt.fill_between(X_sorted, y_low, y_high, color=c, alpha=0.1, label=labels[m])
        plt.plot(X_sorted, y_low, color=c, alpha=0.4, linewidth=1)
        plt.plot(X_sorted, y_high, color=c, alpha=0.4, linewidth=1)

    plt.xlabel("Input x")
    plt.ylabel("Prediction $\\pm$ Smooth Sensitivity")
    plt.title("Smooth Sensitivity Bands")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{out_dir}/lin_ablation_final_manual{suffix}_bands.png")
    plt.close()

    print("Saved bands. Computing Tradeoffs...")

    # Load Baselines
    base_dir = "agtr/linear/values/baselines"
    baselines = {}
    # No baseline arms on these panels for now: Functional Mechanism, Gaussian DP-SGD
    # and shard-and-aggregate were all removed (2026-08-25). run_baselines.py and
    # common.shard_sweep still write their arrays, so any of them is a few lines
    # away from coming back.

    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 500
    gs_val = CFG["GS_VAL"]  # = 10.0
    delta_dp = 1e-5
    gauss_factor = np.sqrt(2 * np.log(1 / delta_dp))

    # Precompute residuals and x values for noise injection
    resid = y_test_pred.flatten() - y_test_true.flatten()
    x_flat = X_test_mse.flatten()

    # --- Plot 3: Trade-off (Pure DP) ---
    print("Plotting Pure DP...")
    plt.figure(figsize=plotting.FIGSIZE)

    # Global Sensitivity baseline (Cauchy noise, scale = GS / eps)
    _kw = dict(beta=beta, gauss_factor=gauss_factor, n_trials=n_trials)
    # the GS baseline uses the raw sensitivity, so its Cauchy scale is gs_val/eps: the
    # same call with beta folded out
    gs_mse_median, gs_mse_lo, gs_mse_hi = release_curve(
        gs_val * beta, epsilons, resid, x_flat, mode="pure", **_kw
    )
    plt.plot(
        epsilons,
        gs_mse_median,
        "o--",
        label="Global Sensitivity",
        color="gray",
        alpha=0.6,
        zorder=1,
        **plotting.SERIES,
    )
    plt.fill_between(epsilons, gs_mse_lo, gs_mse_hi, color="gray", alpha=0.15)

    for m in ordered_methods:
        if m not in results:
            continue
        res = results[m]
        avg_V_k = res["avg_V_k"]
        ks = res["k_values"]

        S_W = method_sensitivity(res, beta)
        mse_median_list, mse_lo_list, mse_hi_list = release_curve(
            S_W, epsilons, resid, x_flat, mode="pure", **_kw
        )

        plt.plot(
            epsilons, mse_median_list, "o-", label=labels[m], color=colors[m], zorder=6, **plotting.SERIES
        )
        plt.fill_between(epsilons, mse_lo_list, mse_hi_list, color=colors[m], alpha=0.1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE (Pure DP)")
    plt.title("Freq Trade-off (Pure DP)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{out_dir}/lin_ablation_final_manual{suffix}_tradeoff_pure.png")
    plt.close()

    # --- Plot 4: Trade-off (Approx DP) ---
    print("Plotting Approx DP...")
    plt.figure(figsize=plotting.FIGSIZE)

    # Global Sensitivity baseline (Gaussian noise, sigma = GS * factor / eps)
    gs_mse_mean, gs_lo, gs_hi = release_curve(gs_val, epsilons, resid, x_flat, mode="approx", **_kw)
    plt.plot(
        epsilons,
        gs_mse_mean,
        "o--",
        label="Global Sensitivity",
        color="gray",
        alpha=0.6,
        zorder=1,
        **plotting.SERIES,
    )
    plt.fill_between(epsilons, gs_lo, gs_hi, color="gray", alpha=0.15)

    for m in ordered_methods:
        if m not in results:
            continue
        res = results[m]
        avg_V_k = res["avg_V_k"]
        ks = res["k_values"]

        S_W = method_sensitivity(res, beta)
        mse_mean_list, mse_lo_list, mse_hi_list = release_curve(
            S_W, epsilons, resid, x_flat, mode="approx", **_kw
        )

        plt.plot(epsilons, mse_mean_list, "o-", label=labels[m], color=colors[m], zorder=6, **plotting.SERIES)
        plt.fill_between(epsilons, mse_lo_list, mse_hi_list, color=colors[m], alpha=0.1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE (Approx DP)")
    plt.title("Freq Trade-off (Approx DP)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{out_dir}/lin_ablation_final_manual{suffix}_tradeoff_approx.png")
    plt.close()

    print("All Manual Plots Saved Successfully.")


if __name__ == "__main__":
    import argparse as _ap

    _p = _ap.ArgumentParser()
    _p.add_argument("--values", default=None, help="directory of method_results_*.npy")
    _p.add_argument("--figures", default=None, help="where to write the figures")
    _p.add_argument("--suffix", default="", help="appended to each figure name")
    _a = _p.parse_args()
    plot_certified_ablation(_a.values, _a.figures, _a.suffix)
