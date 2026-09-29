# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""AGT-R linear, combined: [staircase | pure DP | approximate DP] in one row.

A wide row, 15 x 6 in (plotting.FIGSIZE_COMBINED), replacing the three single figures. The panels get narrower and the type comes down with them
(`plotting.apply_combined(3, legend_pt=7.0)`), and one shared legend sits below the row so it costs
no plot area.

The curves are not recomputed here: `plot_ablation.release_curve` and
`plot_ablation.method_sensitivity` are the same functions the single figures call, and
the Monte-Carlo is seeded (`plot_ablation.MC_SEED`), so a panel here is identical to its
single-figure counterpart rather than a second implementation of it.

    python -m agtr.linear.make_combined                      # published k grid
    python -m agtr.linear.make_combined --values agtr/linear/values/ablation_k20 \
        --suffix _k20
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import plotting
from common.plotting import METHOD_COLORS, METHOD_LABELS
from agtr.linear.config import ABLATION as CFG
from agtr.legacy.linear.plot_ablation import MC_SEED, method_sensitivity, release_curve

ORDER = ["ibp", "f1", "f2", "f4", "f8", "f16"]
EPSILONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_TRIALS = 500


def load_results(in_dir):
    out = {}
    for m in ORDER:
        f = f"{in_dir}/method_results_{m}.npy"
        if os.path.exists(f):
            out[m] = np.load(f, allow_pickle=True).item()
    if not out:
        raise SystemExit(f"no method_results_*.npy in {in_dir}")
    return out


def nominal_residuals():
    """The evaluation set and nominal model plot_ablation reconstructs, same seeds."""
    torch.manual_seed(42)
    mod = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    w0, b0 = mod[0].weight.item(), mod[0].bias.item()
    np.random.seed(12345)
    X = np.random.uniform(-2, 2, (1000, 1))
    y_true = 0.5 * X + 1.0 + np.random.normal(0, 0.1, (1000, 1))
    y_pred = w0 * X + b0
    return (y_pred.flatten() - y_true.flatten()), X.flatten()


def panel_staircase(ax, results, beta, gs_val):
    ax2 = ax.twinx()
    max_k = max([6.0] + [max(r["k_values"]) for r in results.values()])
    t = np.linspace(0, max_k, 1000)
    for m in ORDER:
        if m not in results or "avg_V_k" not in results[m]:
            continue
        r = results[m]
        i = np.argsort(r["k_values"])
        ks, vs = np.array(r["k_values"])[i], np.array(r["avg_V_k"])[i]
        c = METHOD_COLORS[m]
        ax.step(
            ks,
            vs,
            where="post",
            label=METHOD_LABELS[m],
            color=c,
            ls="--",
            lw=plt.rcParams["lines.linewidth"],
            alpha=0.7,
        )
        j = np.clip(np.searchsorted(ks, t, side="right") - 1, 0, len(ks) - 1)
        s = vs[j] * np.exp(-beta * t)
        ax2.plot(t, s, "-", color=c, lw=plt.rcParams["lines.linewidth"] + 0.4, alpha=0.9)
        k = int(np.argmax(s))
        ax2.plot(
            t[k],
            s[k],
            "*",
            color=c,
            ms=plt.rcParams["lines.markersize"] * 2.4,
            markeredgecolor="black",
            zorder=10,
        )
    # named apart from the trade-off panels' "Global Sensitivity" series: this is the
    # ceiling on A^(k), that is a release curve, and one shared legend holds both
    ax.axhline(gs_val, color="black", ls=":", lw=2.0, zorder=5, label=f"GS ceiling ({gs_val:.0f})")
    ax.set_xlabel(r"Substitutions $k$")
    ax.set_ylabel(r"$A^{(k)}$ (dashed)")
    ax2.set_ylabel(r"$SS^\beta$ (solid)")
    ax.set_title("Staircase")
    ax.grid(True, alpha=0.3)
    ax.set_zorder(1)
    ax.patch.set_visible(False)


def panel_tradeoff(ax, results, mode, beta, gs_val, resid, x_flat, gauss_factor):
    kw = dict(beta=beta, gauss_factor=gauss_factor, n_trials=N_TRIALS, seed=MC_SEED)
    c, lo, hi = release_curve(gs_val, EPSILONS, resid, x_flat, mode=mode, **kw)
    ax.plot(
        EPSILONS, c, "o--", color="gray", alpha=0.6, zorder=1, label="Global Sensitivity", **plotting.SERIES
    )
    ax.fill_between(EPSILONS, lo, hi, color="gray", alpha=0.15)
    for m in ORDER:
        if m not in results:
            continue
        S_W = method_sensitivity(results[m], beta)
        c, lo, hi = release_curve(S_W, EPSILONS, resid, x_flat, mode=mode, **kw)
        ax.plot(
            EPSILONS, c, "o-", color=METHOD_COLORS[m], label=METHOD_LABELS[m], zorder=6, **plotting.SERIES
        )
        ax.fill_between(EPSILONS, lo, hi, color=METHOD_COLORS[m], alpha=0.1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Privacy Budget $\epsilon$",
        ylabel="Test MSE",
        title=r"Pure $(\epsilon, 0)$-DP" if mode == "pure" else r"Approx. $(\epsilon, \delta)$-DP",
    )
    ax.grid(True, alpha=0.3)


def main(a):
    results = load_results(a.values or CFG["OUT_DIR"])
    out_dir = a.figures or CFG["FIG_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    beta, gs_val = CFG["BETA"], CFG["GS_VAL"]
    gauss_factor = np.sqrt(2 * np.log(1 / 1e-5))
    resid, x_flat = nominal_residuals()

    plotting.apply_combined(3, legend_pt=7.0)
    fig, axes = plt.subplots(1, 3, figsize=plotting.combined_figsize(3), layout="constrained")
    # constrained layout packs the legend right up against the x-labels by default;
    # h_pad opens the gap so the row does not read as one crowded block
    fig.get_layout_engine().set(h_pad=0.20, w_pad=0.06)
    panel_staircase(axes[0], results, beta, gs_val)
    panel_tradeoff(axes[1], results, "pure", beta, gs_val, resid, x_flat, gauss_factor)
    panel_tradeoff(axes[2], results, "approx", beta, gs_val, resid, x_flat, gauss_factor)
    plotting.shared_legend(
        fig, axes, ncol=8, loc="outside upper center", columnspacing=0.8, handlelength=1.5, handletextpad=0.4
    )
    fig.savefig(f"{out_dir}/lin_combined{a.suffix}.png")
    plt.close(fig)
    plotting.apply()
    print(f"wrote {out_dir}/lin_combined{a.suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--suffix", default="")
    main(ap.parse_args())
