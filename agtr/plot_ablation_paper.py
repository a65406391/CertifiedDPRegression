"""The paper's AGT-R ablation row: the two privacy panels of each cross-radius ablation, matched to the
PATE row of agtr/plot_pate_paper.py.

Four panels from the two stored results of the appendix ablations, computing nothing: Linear is the
concretisation ablation (IBP and AGT at F = 1, 2, 4, 8, 16; agtr/linear/values/recompute_2026-09-14),
California the batch-size ablation (BS = 250 ... 16512; agtr/california/values/agt_2026-09-14). The
certificate (staircase) panel of each source figure is left out; the trade-off panels are the same
series as there, pure DP with the tighter Cauchy release at beta = eps / 2 and the mean absolute error,
approximate DP with Laplace(2 SS / eps) at delta = 1e-5 and the test MSE, plus global sensitivity and
the nominal MSE. Same axes, panel titles, group titles and type sizes as the PATE row, one legend under
each dataset's pair of panels.

    PYTHONPATH=. python -m agtr.plot_ablation_paper
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import plotting
from common.plotting import METHOD_COLORS, METHOD_LABELS

LINEAR = "agtr/linear/values/recompute_2026-09-14/results.json"
CALIFORNIA = "agtr/california/values/agt_2026-09-14/{tag}/results_cross_radius.json"
OUT = "agtr/figures/ablation_paper_linear_california.png"
METHODS = ["ibp", "f1", "f2", "f4", "f8", "f16"]
BS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MODES = (  # agt mode, metric, y label, panel title: the same as the PATE row
    ("pure_liang_b2", "mae", "MAE", r"Pure $(\epsilon, 0)$-DP"),
    ("approx", "mse", "MSE", r"Approx. $(\epsilon, 10^{-5})$-DP"),
)


def linear_series(R, mode, metric):
    """[(label, colour, y), ...] for the Linear panels, GS first so it is drawn on top."""
    gs = R["gs_baselines"]["pure" if mode.startswith("pure") else mode]["GS=clipping ceiling (per x)"]
    out = [("Global sensitivity", "black", gs[metric])]
    out += [(METHOD_LABELS[m], METHOD_COLORS[m], R["methods"][m][mode][metric]) for m in METHODS]
    return out, R["nominal_mse"]


def california_series(R, mode, metric):
    """The same for the California panels, one curve per batch size."""
    gs = R["gs_baselines"]["pure" if mode.startswith("pure") else mode]
    out = [("Global sensitivity", "black", gs[metric])]
    out += [
        (f"BS={bs}", BS_COLORS[i], R["batch_sizes"][bs][mode][metric])
        for i, bs in enumerate(R["batch_sizes"])
    ]
    nominal = float(np.mean([e["nominal_mse_real"] for e in R["batch_sizes"].values()]))
    return out, nominal


def draw(runs, out, figsize=(14.0, 3.9), page_width_in=7.16):
    """runs: [(group title, results dict, series function), ...]; two panels per run."""
    plotting.apply_combined(
        2 * len(runs),
        figsize=figsize,
        page_width_in=page_width_in,
        print_target_pt=8,
        label_pt=8,
        tick_pt=7,
        legend_pt=7,
    )
    fig, axes = plt.subplots(1, 2 * len(runs), figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9), wspace=0.06)  # the top strip holds the group titles
    for g, (name, R, series) in enumerate(runs):
        for m, (mode, metric, ylab, title) in enumerate(MODES):
            ax = axes[2 * g + m]
            curves, nominal = series(R, mode, metric)
            for label, col, y in curves:
                gs = label.startswith("Global")
                ax.plot(
                    R["epsilons"],
                    y,
                    "o-",
                    color=col,
                    alpha=0.85 if gs else 1.0,
                    zorder=20 if gs else 6,
                    label=label,
                    **plotting.SERIES,
                )
            if mode == "approx":
                ax.axhline(
                    nominal,
                    color="black",
                    ls="--",
                    lw=1.2,
                    alpha=0.7,
                    label=f"Nominal MSE ({nominal:.2f})",
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=r"Privacy Budget $\epsilon$",
                ylabel=ylab,
                title=title,
            )
            ax.grid(True, alpha=0.3)
    # one legend per group, two rows, under that group's pair of panels
    style = dict(columnspacing=0.6, handlelength=1.4, handletextpad=0.4)
    legends = []
    for g in range(len(runs)):
        handles, labels = [], []
        for ax in axes[2 * g : 2 * g + 2]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        legends.append((handles, labels))
    fig.canvas.draw()  # group titles and legends sit under / over the centre of each pair once placed
    size = plt.rcParams["axes.titlesize"]
    fig.get_layout_engine().set(rect=(0, 0.16, 1, 0.74))  # the bottom strip holds the legends
    fig.canvas.draw()
    for g, (name, _, _) in enumerate(runs):
        left, right = axes[2 * g].get_position(), axes[2 * g + 1].get_position()
        cx = (left.x0 + right.x1) / 2
        fig.text(cx, 0.965, name, ha="center", va="center", fontsize=size * 1.15, fontweight="bold")
        handles, labels = legends[g]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(cx, 0.0),
            ncol=4,
            **{**plotting.LEGEND_BELOW_KW, **style},
        )
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--california-tag", default="relu_h64_clamp_gamma0.1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    runs = [
        ("Linear", json.load(open(LINEAR)), linear_series),
        ("California", json.load(open(CALIFORNIA.format(tag=a.california_tag))), california_series),
    ]
    draw(runs, a.out)
    print(f"wrote {a.out}  [california {a.california_tag}]")


if __name__ == "__main__":
    main()
