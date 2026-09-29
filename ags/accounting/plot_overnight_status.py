"""Render the 2026-08-26 AGS accounting results in the paper figure style."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from common import plotting
from common.plotting import BLUES, C_DPSGD as ORANGE, C_NOMINAL as GREY, C_PATE as GREEN


from ags.accounting.paths import SHARED, artifact

OUT = SHARED / "overnight_status_figures_2026-08-26"
BLUE = BLUES[-1]


def load(name: str):
    with artifact(name).open() as f:
        return json.load(f)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def setup_axis(ax, eps, ylim=None):
    ax.set_xscale("log")
    ax.set_xlim(0.9, 110)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
    if ylim:
        ax.set_ylim(*ylim)


def plot_blobs():
    data = load("blobs_native_pld_envelope_comparison_overnight.json")
    rows = data["rows"]
    eps = np.array([r["target_epsilon"] for r in rows], dtype=float)
    matched = [r["matched_fixed_cyclic_dpsgd"] for r in rows]
    poisson = [r["poisson_dpsgd_stress_test"] for r in rows]
    ags = np.array([r["ags_median"] for r in matched])
    dp = np.array([r["dpsgd_median"] for r in matched])
    pois = np.array([r["dpsgd_median"] for r in poisson])

    size = (13.4, 5.2)
    plotting.apply_combined(2, figsize=size, page_width_in=plotting.A4_TEXT_WIDTH_IN)
    fig, axes = plt.subplots(1, 2, figsize=size, layout="constrained")

    axes[0].plot(
        eps, ags, color=BLUE, marker="o", label="AGS (certified native PLD)", zorder=6, **plotting.SERIES
    )
    axes[0].plot(eps, dp, color=ORANGE, marker="s", label="DP-SGD (matched)", zorder=5, **plotting.SERIES)
    axes[0].plot(
        eps,
        pois,
        color=GREEN,
        marker="^",
        ls="--",
        label="DP-SGD (Poisson stress test)",
        zorder=4,
        **plotting.SERIES,
    )
    axes[0].axhline(0.903, color=GREY, ls=":", lw=2.0, zorder=2)
    axes[0].text(
        1.05,
        0.907,
        "non-private",
        color=GREY,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["legend.fontsize"],
    )
    setup_axis(axes[0], eps, (0.44, 0.93))
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Certified BLOBS utility", pad=8)

    for values, color, marker, label, offset in (
        (matched, ORANGE, "s", "AGS − matched DP-SGD", 0.96),
        (poisson, GREEN, "^", "AGS − Poisson DP-SGD", 1.04),
    ):
        means = np.array([r["mean_difference"] for r in values])
        ci = np.array([r["unpaired_bootstrap_mean_difference_ci95"] for r in values])
        x = eps * offset
        axes[1].errorbar(
            x,
            means,
            yerr=np.vstack((means - ci[:, 0], ci[:, 1] - means)),
            color=color,
            marker=marker,
            ls="-" if marker == "s" else "--",
            capsize=3,
            label=label,
            zorder=5,
            **plotting.SERIES,
        )
    axes[1].axhline(0, color="0.25", ls=":", lw=2.0)
    axes[1].fill_between([0.9, 110], 0, 0.025, color=BLUE, alpha=0.055, zorder=0)
    axes[1].text(
        1.03,
        0.004,
        "AGS better",
        color=BLUE,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["legend.fontsize"],
    )
    setup_axis(axes[1], eps, (-0.37, 0.035))
    axes[1].set_ylabel("Mean Accuracy Difference")
    axes[1].set_title("Descriptive 95% intervals", pad=8)

    plotting.shared_legend(fig, axes, ncol=3)
    save(fig, "blobs_certified_comparison")


def plot_pca():
    data = load("overnight_strong_summary.json")
    long_rows = sorted((r for r in data["pca_mnist"]["rows"] if r["window"] == 1), key=lambda r: r["epsilon"])

    plotting.apply()
    fig, ax = plt.subplots(figsize=plotting.FIGSIZE, layout="constrained")
    eps = np.array([r["epsilon"] for r in long_rows], dtype=float)
    ags = [r["raw"]["ags_median"] for r in long_rows]
    dp = [r["raw"]["dpsgd_median"] for r in long_rows]
    ax.plot(eps, ags, color=BLUE, marker="o", label="AGS (coordinate Laplace)", zorder=6, **plotting.SERIES)
    ax.plot(eps, dp, color=ORANGE, marker="s", label="DP-SGD (Gaussian)", zorder=5, **plotting.SERIES)
    ax.axhline(0.821, color=GREY, ls=":", lw=2.0, zorder=2)
    ax.text(
        1.04,
        0.828,
        "non-private",
        color=GREY,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["legend.fontsize"],
    )
    setup_axis(ax, eps, (0.10, 0.86))
    ax.set_ylabel("Test Accuracy")
    ax.set_title("MNIST", pad=8)
    ax.legend(**plotting.LEGEND_BELOW, ncol=2)
    save(fig, "mnist_fixed_40000_steps")


def write_index():
    html = """<!doctype html>
<meta charset="utf-8">
<title>AGS accounting overnight figures — 26 August 2026</title>
<style>
body { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.5 system-ui, sans-serif; color: #222; }
img { display: block; width: 100%; height: auto; border: 1px solid #ddd; margin: .7rem 0; }
.note { background: #f5f7fa; border-left: 4px solid #08306b; padding: .8rem 1rem; }
a { color: #08306b; }
</style>
<h1>AGS accounting overnight figures</h1>
<p class="note"><strong>Interpretation:</strong> BLOBS is the fully certified headline result. Its intervals are descriptive because the fixed AGS cell was selected during exploratory development. The Poisson curve is an algorithmically unmatched stress test. PCA's long-trajectory sensitivity remains conditional; the one-step panel is exact and reverses the ordering.</p>
<h2>Certified BLOBS comparison</h2>
<img src="blobs_certified_comparison.png" alt="Certified BLOBS AGS and DP-SGD comparison">
<h2>MNIST — fixed 40,000-step trajectory</h2>
<img src="mnist_fixed_40000_steps.png" alt="MNIST fixed 40,000-step trajectory">
<p>Generated from the certified overnight JSON summaries by <code>ags/accounting/plot_overnight_status.py</code>.</p>
"""
    (OUT / "index.html").write_text(html)


if __name__ == "__main__":
    plot_blobs()
    plot_pca()
    write_index()
    print(f"wrote {OUT}")
