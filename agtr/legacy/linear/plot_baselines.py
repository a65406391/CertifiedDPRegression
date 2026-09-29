# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
Linear Baseline Plots.

Refactored from plot_lin_baselines.py.
"""

import numpy as np

from common import plotting
import matplotlib.pyplot as plt

from agtr.linear.config import BASELINES as CFG
from common.plotting import apply_style, save_figure, ensure_fig_dir


def main():
    print("Generating Linear Baseline Plots...")
    apply_style("paper")

    data_dir = CFG["OUT_DIR"]
    agt_dir = "agtr/linear/values/ablation"
    fig_dir = CFG["FIG_DIR"]
    ensure_fig_dir(fig_dir)

    plot_prefix = "lin_baselines"

    epsilons = np.load(f"{data_dir}/epsilons.npy")

    func_mean = np.load(f"{data_dir}/func_mean.npy")
    func_std = np.load(f"{data_dir}/func_std.npy")
    lap_mean = np.load(f"{data_dir}/lap_mean.npy")
    lap_std = np.load(f"{data_dir}/lap_std.npy")
    gauss_mean = np.load(f"{data_dir}/gauss_mean.npy")
    gauss_std = np.load(f"{data_dir}/gauss_std.npy")

    freqs = [1, 3, 5, 10, 20]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    # 1. Pure DP Comparison
    plt.figure(figsize=plotting.FIGSIZE)

    # Functional Mechanism dropped from the figure (2026-08-25): its sufficient-statistic
    # sensitivity uses ||x||_2^2 where a per-record change to X^T X has l1 norm (sum_i |x_i|)^2,
    # so the arm is very likely under-noised. The arrays are still written by run_baselines.py.

    plt.plot(epsilons, lap_mean, "s-", label="Laplace DP-SGD (C=1)", color="hotpink", linewidth=5)
    plt.fill_between(epsilons, lap_mean - lap_std, lap_mean + lap_std, color="hotpink", alpha=0.1)

    # Shard-and-aggregate removed from the figure (2026-08-25); the swept arrays
    # remain in shard_best_*.npy for when it comes back.

    for idx, f in enumerate(freqs):
        try:
            agt_pure_median = np.load(f"{agt_dir}/pure_f{f}_median.npy")
            label_agt = f"AGT (F={f})"
            if len(agt_pure_median) == len(epsilons):
                plt.plot(
                    epsilons,
                    agt_pure_median,
                    marker=".",
                    linestyle="--",
                    label=label_agt,
                    color=colors[idx],
                    linewidth=5,
                )
            else:
                print(f"Warning: AGT F={f} shape mismatch")
        except FileNotFoundError:
            print(f"AGT Freq={f} Pure DP data not found in {agt_dir}.")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE")
    plt.title("Linear")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_pure.png")
    plt.close()

    # 2. Approx DP Comparison
    plt.figure(figsize=plotting.FIGSIZE)

    # Gaussian DP-SGD removed from the figure (2026-08-25); run_baselines.py still
    # writes gauss_mean.npy / gauss_std.npy.

    for idx, f in enumerate(freqs):
        try:
            agt_approx_mean = np.load(f"{agt_dir}/approx_f{f}_mean.npy")
            label_agt = f"AGT (F={f})"
            if len(agt_approx_mean) == len(epsilons):
                plt.plot(
                    epsilons,
                    agt_approx_mean,
                    marker=".",
                    linestyle="--",
                    label=label_agt,
                    color=colors[idx],
                    linewidth=4,
                )
        except FileNotFoundError:
            print(f"AGT Freq={f} Approx DP data not found.")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Privacy Budget $\\epsilon$")
    plt.ylabel("Test MSE")
    plt.title("Linear")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(f"{fig_dir}/{plot_prefix}_approx.png")
    plt.close()

    print(f"Plots saved to {fig_dir}/{plot_prefix}_*.png")


if __name__ == "__main__":
    main()
