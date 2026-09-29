# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.

import numpy as np
import matplotlib.pyplot as plt
import os
import traceback

from common.plotting import apply_style, save_figure, ensure_fig_dir


def replot_ca_ablation():
    print("Replotting CA Batch Ablation from saved data...")
    # The trade-off panels draw Monte-Carlo noise from the global RNG. Without a seed the figure
    # changed on every run, so it was not reproducible; the current release path (agtr/california/
    # release.py) seeds explicitly with SEED = 0 and this matches it.
    np.random.seed(0)
    apply_style("paper")
    data_dir = "agtr/california/values/ablation"  # Fixed dir
    fig_dir = "agtr/california/figures/california"
    ensure_fig_dir(fig_dir)
    plot_prefix = "ca_batch_scaled"

    if not os.path.exists(data_dir):
        print("Data directory not found. Run ablation first.")
        return

    # Load Data (Individual Files)
    # Infer batch sizes from available sensitivity files
    import glob

    sens_files = glob.glob(f"{data_dir}/sensitivity_bs*.npy")
    batch_sizes = []
    for f in sens_files:
        try:
            # Extract BS from filename "sensitivity_bs123.npy"
            bs_str = f.split("sensitivity_bs")[-1].replace(".npy", "")
            batch_sizes.append(int(bs_str))
        except:
            pass
    batch_sizes = sorted(list(set(batch_sizes)))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Pre-load plotting data
    plot_data_staircase = {}  # bs -> (ks, vs)
    plot_data_avg_S = {}  # bs -> avg_S
    plot_data_nominal_mse = {}  # bs -> mse

    if len(batch_sizes) == 0:
        print("No batch sizes found.")
        return

    # Load y_test for MSE calc
    y_test = np.load(f"{data_dir}/y_test.npy")
    y_mean = np.load(f"{data_dir}/y_mean.npy")
    y_std = np.load(f"{data_dir}/y_std.npy")
    y_test_real = y_test * y_std + y_mean

    for bs in batch_sizes:
        # Load Staircase
        ks_file = f"{data_dir}/staircase_bs{bs}_ks.npy"
        vs_file = f"{data_dir}/staircase_bs{bs}_vs.npy"
        if os.path.exists(ks_file) and os.path.exists(vs_file):
            plot_data_staircase[bs] = (np.load(ks_file), np.load(vs_file))

        # Load Sensitivity
        sens_file = f"{data_dir}/sensitivity_bs{bs}.npy"
        if os.path.exists(sens_file):
            s_vec = np.load(sens_file)
            plot_data_avg_S[bs] = np.mean(s_vec)

        # Load MSE
        preds_file = f"{data_dir}/nominal_preds_bs{bs}.npy"
        if os.path.exists(preds_file):
            preds = np.load(preds_file)
            preds_real = preds * y_std + y_mean
            mse = np.mean((preds_real - y_test_real) ** 2)
            plot_data_nominal_mse[bs] = mse

    # --- PLOT 1: Staircase ---
    try:
        fig, ax1 = plt.subplots(figsize=(10, 11))
        ax2 = ax1.twinx()

        beta = 1.0 / 6.0
        gs_val = 10.0  # Y_MAX - Y_MIN = 5.0 - (-5.0)

        # Determine max_k across all loaded batch sizes
        max_k_val = 6.0
        for bs in batch_sizes:
            if bs in plot_data_staircase:
                max_k_val = max(max_k_val, max(plot_data_staircase[bs][0]))
        t_grid = np.linspace(0, max_k_val + 5, 1000)

        if any(bs in plot_data_staircase for bs in batch_sizes):
            ax1.set_xlabel("Privacy Parameter k")
            ax1.set_ylabel("Max. Local Sens. Certificate $A_k$ (dashed)")
            ax2.set_ylabel("Smooth Sensitivity $SS(f, x)$ (solid)")
            ax1.set_title("California Housing")

            for i, bs in enumerate(batch_sizes):
                if bs not in plot_data_staircase:
                    continue
                ks_b, vs_b = plot_data_staircase[bs]
                c = colors[i % len(colors)]

                # Plot Staircase (Dashed, Alpha 0.75)
                ax1.step(
                    ks_b,
                    vs_b,
                    where="post",
                    label=f"BS={bs}",
                    color=c,
                    linestyle="--",
                    linewidth=4,
                    alpha=0.75,
                )

                indices = np.searchsorted(ks_b, t_grid, side="right") - 1
                indices = np.clip(indices, 0, len(ks_b) - 1)
                v_interp = np.array([vs_b[idx] for idx in indices])
                s_curve = v_interp * np.exp(-beta * t_grid)

                # Plot Curve (Solid, Alpha 1.0)
                ax2.plot(t_grid, s_curve, "-", color=c, linewidth=5, alpha=1.0)

                idx_max = np.argmax(s_curve)
                t_max = t_grid[idx_max]
                s_max = s_curve[idx_max]
                ax2.plot(t_max, s_max, "*", color=c, markersize=28, markeredgecolor="black")

            # Global sensitivity horizontal line on V_k axis
            ax1.axhline(
                y=gs_val,
                color="black",
                linestyle=":",
                linewidth=2.5,
                label=f"Global Sensitivity ({gs_val:.0f})",
                zorder=5,
            )

            ax1.legend(loc="upper right", title="Batch Size")
            ax1.grid(True, alpha=0.3)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)
            plt.tight_layout()
            save_figure(f"{fig_dir}/{plot_prefix}_staircase.png")
            plt.close()
    except Exception as e:
        print(f"Staircase Plot Error: {e}")
        traceback.print_exc()

    # --- PLOT 2: Bar Chart ---
    try:
        plt.figure(figsize=(10, 11))
        bar_bs = []
        bar_sens = []
        for bs in batch_sizes:
            if bs in plot_data_avg_S:
                bar_bs.append(str(bs))
                bar_sens.append(plot_data_avg_S[bs])

        plt.bar(bar_bs, bar_sens, color=colors[: len(bar_bs)], alpha=0.7, edgecolor="black")
        plt.xlabel("Batch Size")
        plt.ylabel("Average smooth sensitivity")
        plt.title("California Housing")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_figure(f"{fig_dir}/{plot_prefix}_avg_sensitivity_bar.png")
        plt.close()
    except Exception as e:
        print(f"Bar Plot Error: {e}")

    # --- Comparsion Params ---
    epsilon_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 500
    delta_dp = 1e-5
    factor = np.sqrt(2 * np.log(1 / delta_dp))
    Y_MIN, Y_MAX = -5.0, 5.0
    GS_VAL = Y_MAX - Y_MIN
    GS_real = GS_VAL * y_std

    # --- PLOT 3: Approx DP ---
    try:
        plt.figure(figsize=(10, 11))
        # GS baseline with ±1 std band (matching linear)
        gs_mse_mean = []
        gs_mse_std = []
        ref_mse_approx = (
            plot_data_nominal_mse[batch_sizes[min(2, len(batch_sizes) - 1)]] if plot_data_nominal_mse else 0.5
        )
        for eps in epsilon_grid:
            sigma_gs = GS_real * factor / eps
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.normal(0, sigma_gs, len(y_test_real))
                mse_trials.append(ref_mse_approx + np.mean(noise**2))
            gs_mse_mean.append(np.mean(mse_trials))
            gs_mse_std.append(np.std(mse_trials))
        gs_mse_mean = np.array(gs_mse_mean)
        gs_mse_std = np.array(gs_mse_std)
        plt.plot(
            epsilon_grid,
            gs_mse_mean,
            "o--",
            label="Global Sensitivity",
            color="gray",
            linewidth=3,
            alpha=0.6,
            zorder=1,
        )
        plt.fill_between(
            epsilon_grid, gs_mse_mean - gs_mse_std, gs_mse_mean + gs_mse_std, color="gray", alpha=0.15
        )

        for i, bs in enumerate(batch_sizes):
            if bs not in plot_data_nominal_mse or bs not in plot_data_avg_S:
                continue

            nom_mse = plot_data_nominal_mse[bs]
            avg_S = plot_data_avg_S[bs]
            c = colors[i % len(colors)]
            mse_mean = []
            mse_std = []
            for eps in epsilon_grid:
                sigma = avg_S * factor / eps
                mse_trials = []
                for _ in range(n_trials):
                    noise = np.random.normal(0, sigma, len(y_test_real))
                    mse_trials.append(nom_mse + np.mean(noise**2))
                mse_mean.append(np.mean(mse_trials))
                mse_std.append(np.std(mse_trials))
            mse_mean = np.array(mse_mean)
            mse_std = np.array(mse_std)
            plt.plot(epsilon_grid, mse_mean, "o-", label=f"BS={bs}", color=c, linewidth=4)
            plt.fill_between(epsilon_grid, mse_mean - mse_std, mse_mean + mse_std, color=c, alpha=0.1)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Privacy Budget $\\epsilon$")
        plt.ylabel("Test MSE (Approx DP)")
        plt.title("California Housing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(f"{fig_dir}/{plot_prefix}_tradeoff_approx.png")
        plt.close()
    except Exception as e:
        print(f"Approx DP Plot Error: {e}")
        traceback.print_exc()

    # --- PLOT 4: Pure DP ---
    try:
        plt.figure(figsize=(10, 11))

        # Proper GS Baseline loop
        ref_mse = (
            plot_data_nominal_mse[batch_sizes[min(2, len(batch_sizes) - 1)]] if plot_data_nominal_mse else 0.5
        )
        gs_mse_median = []
        gs_mse_p25 = []
        gs_mse_p75 = []
        for eps in epsilon_grid:
            scale_gs = GS_real / eps
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.standard_cauchy(len(y_test_real)) * scale_gs
                mse_trials.append(ref_mse + np.mean(noise**2))
            gs_mse_median.append(np.median(mse_trials))
            gs_mse_p25.append(np.percentile(mse_trials, 25))
            gs_mse_p75.append(np.percentile(mse_trials, 75))

        plt.plot(
            epsilon_grid,
            gs_mse_median,
            "o--",
            label="Global Sensitivity",
            color="gray",
            linewidth=3,
            alpha=0.6,
            zorder=1,
        )
        plt.fill_between(epsilon_grid, gs_mse_p25, gs_mse_p75, color="gray", alpha=0.15)

        for i, bs in enumerate(batch_sizes):
            if bs not in plot_data_nominal_mse or bs not in plot_data_avg_S:
                continue

            nom_mse = plot_data_nominal_mse[bs]
            avg_S = plot_data_avg_S[bs]
            c = colors[i % len(colors)]
            mse_median = []
            mse_p25 = []
            mse_p75 = []
            for eps in epsilon_grid:
                scale_ss = avg_S / eps
                mse_trials = []
                for _ in range(n_trials):
                    noise = np.random.standard_cauchy(len(y_test_real)) * scale_ss
                    mse_trials.append(nom_mse + np.mean(noise**2))
                mse_median.append(np.median(mse_trials))
                mse_p25.append(np.percentile(mse_trials, 25))
                mse_p75.append(np.percentile(mse_trials, 75))
            mse_median = np.array(mse_median)
            mse_p25 = np.array(mse_p25)
            mse_p75 = np.array(mse_p75)
            plt.plot(epsilon_grid, mse_median, "o-", label=f"BS={bs}", color=c, linewidth=4)
            plt.fill_between(epsilon_grid, mse_p25, mse_p75, color=c, alpha=0.1)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Privacy Budget $\\epsilon$")
        plt.ylabel("Test Median Squared Error (Pure DP)")
        plt.title("California Housing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(f"{fig_dir}/{plot_prefix}_tradeoff_pure.png")
        plt.close()
    except Exception as e:
        print(f"Pure DP Plot Error: {e}")
        traceback.print_exc()

    # --- PLOT 5: Pareto Frontier ---
    try:
        plt.figure(figsize=(10, 11))

        # Collect data
        results = []
        for bs in batch_sizes:
            if bs in plot_data_nominal_mse and bs in plot_data_avg_S:
                results.append({"bs": bs, "mse": plot_data_nominal_mse[bs], "sens": plot_data_avg_S[bs]})

        # Sort
        results.sort(key=lambda x: x["bs"])

        # Plot Scatter
        for i, res in enumerate(results):
            bs = res["bs"]
            c = colors[i % len(colors)]
            plt.scatter(
                res["mse"], res["sens"], s=400, color=c, label=f"BS={bs}", edgecolors="black", zorder=5
            )

        # Draw dashed line
        mse_vals = [r["mse"] for r in results]
        sens_vals = [r["sens"] for r in results]
        plt.plot(mse_vals, sens_vals, "--", color="gray", alpha=0.5, zorder=1)

        plt.xlabel("Nominal Test MSE")
        plt.ylabel("Average Smooth Sensitivity")
        plt.title("California Housing")

        # Reduce x-tick frequency
        plt.locator_params(axis="x", nbins=5)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(f"{fig_dir}/{plot_prefix}_pareto.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting Pareto: {e}")


if __name__ == "__main__":
    replot_ca_ablation()
