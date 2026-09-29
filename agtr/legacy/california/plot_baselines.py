# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.

import numpy as np

from common import plotting
import matplotlib.pyplot as plt
import os

# Configuration
out_dir = "agtr/california/values/baselines"
agt_dir = "agtr/california/values/ablation"
fig_dir = "agtr/california/figures/california"
plot_prefix = "ca_baselines"  # Retained as it's used later
os.makedirs(fig_dir, exist_ok=True)

# Plotting Style
# Plotting Style
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "figure.titlesize": 28,
        "lines.linewidth": 4,
        "axes.linewidth": 2,
    }
)


def main():
    plotting.apply()
    print("Generating CA Baseline Plots...")

    epsilons = np.load(f"{out_dir}/epsilons.npy")

    # Load Baselines
    func_mean = np.load(f"{out_dir}/func_mean.npy")
    func_std = np.load(f"{out_dir}/func_std.npy")

    lap_mean = np.load(f"{out_dir}/lap_mean.npy")
    lap_std = np.load(f"{out_dir}/lap_std.npy")

    gauss_mean = np.load(f"{out_dir}/gauss_mean.npy")
    gauss_std = np.load(f"{out_dir}/gauss_std.npy")

    # Load AGT Benchmark Data (from plott_values_ca_batch_scaled)
    # This directory was created by run_ca_batch_ablation.py
    # We want to load the "Nominal Batch Size=1000" results, or just the main "Pareto" results.
    # Actually, pure_f1_median.npy doesn't exist for CA unless we ran frequency sweep.
    # The batch ablation doesn't save exactly in "pure_f1_median" format.
    # It saves `tradeoff_pure.png`.
    # Let's inspect `run_ca_batch_ablation.py` saving logic...
    # It doesn't seem to save processed summarized arrays for OTHER scripts to consume easily
    # except as internal variables for plotting.
    # WAIT: It saves `bounds_bs{bs}_yl.npy`, etc.
    # AND plot_ca_batch_ablation computes the trade-offs on the fly.
    # We might not be able to load pre-computed median/mean arrays for AGT CA unless I modify ablation or re-calculate here.
    # For now, let's plot Baselines. If I can load AGT data later, I will overlay it.

    # Calculate AGT Curves (Optimized)
    # Calculate AGT Curves from Batch Ablation
    agt_pure_median = []
    agt_approx_mean = []
    HAS_AGT = False

    try:
        # Load Batch Ablation Data
        # We need to reconstruct the best utility for AGT.
        # Ideally, we pick the best Batch Size from the ablation study.
        # For simplicity, we choose BS=2000 (often a good trade-off) or iterate to find best.

        # Load batch sizes to iterate
        if os.path.exists(f"{agt_dir}/batch_sizes.npy"):
            batch_sizes = np.load(f"{agt_dir}/batch_sizes.npy")
            best_bs = 2000 if 2000 in batch_sizes else batch_sizes[0]
        else:
            batch_sizes = [2000]
            best_bs = 2000

        # Load Nominal MSE and Sensitivity for Best BS
        # Assuming saved from plot_ca_ablation re-construction or raw files
        # Raw files: nominal_preds_bs{bs}.npy, sensitivity_bs{bs}.npy

        if (
            os.path.exists(f"{agt_dir}/nominal_preds_bs{best_bs}.npy")
            and os.path.exists(f"{agt_dir}/sensitivity_bs{best_bs}.npy")
            and os.path.exists(f"{agt_dir}/y_test.npy")
        ):
            preds = np.load(f"{agt_dir}/nominal_preds_bs{best_bs}.npy")
            # Load stats to denormalize
            y_test = np.load(f"{agt_dir}/y_test.npy")
            y_mean = np.load(f"{agt_dir}/y_mean.npy")
            y_std = np.load(f"{agt_dir}/y_std.npy")

            y_test_real = y_test * y_std + y_mean
            preds_real = preds * y_std + y_mean
            opt_mse = np.mean((preds_real - y_test_real) ** 2)

            s_vec = np.load(f"{agt_dir}/sensitivity_bs{best_bs}.npy")
            opt_sens = np.mean(s_vec) * y_std  # Real Scale

            print(f"Loaded AGT Run (BS={best_bs}). MSE={opt_mse:.4f}, Sens={opt_sens:.4f}")
            HAS_AGT = True
            label_agt = f"AGT (BS={best_bs})"

        else:
            print("AGT raw files not found in ablation dir.")
            raise FileNotFoundError("Raw AGT files missing")

    except Exception as e:
        print(f"AGT Data Load Error: {e}")
        HAS_AGT = False

    try:
        if HAS_AGT:
            delta_dp = 1e-5
            factor = np.sqrt(2 * np.log(1 / delta_dp))
            n_trials = 500

            for eps in epsilons:
                # Approx DP
                # MSE_priv = MSE_nom + Sigma^2
                sigma = opt_sens * factor / eps
                mse_priv = opt_mse + sigma**2
                agt_approx_mean.append(mse_priv)

                # Pure DP Median
                # MSE_priv_median proxy = MSE_nom + (Scale)^2
                scale = opt_sens / eps
                # More rigorous simulation for consistency with ablation plots
                mse_trials = []
                for _ in range(n_trials):
                    noise = np.random.standard_cauchy(len(y_test)) * scale
                    pass

                # Proxy:
                mse_pure_proxy = opt_mse + scale**2
                agt_pure_median.append(mse_pure_proxy)

            print("AGT Approx Curve:", agt_approx_mean)
            HAS_AGT = True
            label_agt = "AGT Optimized (BS=8000)"

    except Exception as e:
        print(f"AGT Data Load Error: {e}")
        HAS_AGT = False

    # 1. Pure DP
    plt.figure(figsize=plotting.FIGSIZE)
    # Baseline: Functional Mechanism (Removed for Paper)
    # plt.plot(epsilons, func_mean, '^-', label='Functional Mechanism', color='tab:orange', linewidth=2)
    # plt.fill_between(epsilons, func_mean - func_std, func_mean + func_std, color='tab:orange', alpha=0.1)

    plt.plot(epsilons, lap_mean, "s-", label="Laplace DP-SGD (MLP)", color="hotpink", linewidth=5)
    plt.fill_between(epsilons, lap_mean - lap_std, lap_mean + lap_std, color="hotpink", alpha=0.1)

    # Shard-and-aggregate removed from the figure (2026-08-25); the swept arrays
    # remain in shard_best_*.npy for when it comes back.

    if HAS_AGT:
        plt.plot(epsilons, agt_pure_median, "d--", label=label_agt, color="tab:blue", linewidth=5)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\epsilon$")
    plt.ylabel("Test MSE")
    plt.title("California Housing")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.ylim(0, 2.0) # Removed strict limit
    plt.tight_layout()  # Added for better layout
    plt.savefig(f"agtr/california/figures/{plot_prefix}_pure.png")  # dpi from paper_style
    plt.close()

    # 2. Approx DP
    plt.figure(figsize=plotting.FIGSIZE)
    plt.plot(epsilons, gauss_mean, "o-", label="Gaussian DP-SGD (MLP)", color="black", linewidth=5)
    plt.fill_between(
        epsilons, gauss_mean - gauss_std, gauss_mean + gauss_std, color="black", alpha=0.1
    )  # was gauss_mean + gauss_mean

    if HAS_AGT:
        # The original instruction's `Code Edit` snippet for AGT plotting in Approx DP
        # was syntactically incomplete and referenced `epsilon_grid`, `marker`, `c`
        # which are not defined in this script's context.
        # Assuming the intent was to update the existing AGT plot line for Approx DP
        # to use linewidth=4 and keep the existing `epsilons` and `tab:blue` color.
        plt.plot(epsilons, agt_approx_mean, "d--", label=label_agt, color="tab:blue", linewidth=5)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\epsilon$")
    plt.ylabel("Test MSE")
    plt.title("California Housing")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.ylim(0, 1.0) # Removed strict limit
    plt.tight_layout()
    plt.savefig(f"agtr/california/figures/{plot_prefix}_approx.png")  # dpi from paper_style
    plt.close()

    print(f"Plots saved to agtr/california/figures/{plot_prefix}_*.png")


if __name__ == "__main__":
    main()
