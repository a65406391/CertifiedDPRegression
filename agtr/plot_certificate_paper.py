"""The appendix's certificate figure: the staircase panels of the two AGT-R ablations side by side.

Two panels from the same stored results as agtr/plot_ablation_paper.py, computing nothing: the
cross-radius certificate A(x, k) averaged over the test inputs (dashed, left axis) for every arm of the
Linear concretisation ablation (IBP, AGT at F = 1 ... 16) and of the California batch-size ablation
(BS = 250 ... 16512), and the smooth-sensitivity bound A(x, k) e^{-beta k} it induces at beta = 1/2
(solid, right axis) with a star at its maximum SS^beta; the dotted line is the global-sensitivity
ceiling, which is also the tail past the last computed k. Group titles, type sizes and legends match
the paper's Linear / California rows.

    PYTHONPATH=. python -m agtr.plot_certificate_paper
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import plotting
from common.plotting import METHOD_COLORS, METHOD_LABELS
from agtr.plot_ablation_paper import CALIFORNIA, LINEAR, METHODS

OUT = "agtr/figures/certificate_paper_linear_california.png"
BETA = 0.5
#: batch size is ordered, so California takes a sequential palette (plasma, dark to light), which also
#: keeps it apart from the tab10 arms of the Linear panel
BS_COLORS = [plt.cm.plasma(v) for v in np.linspace(0.0, 0.85, 6)]


def linear_curves(R):
    """[(label, colour, A by k), ...], the ceiling, the k range drawn."""
    return (
        [(METHOD_LABELS[m], METHOD_COLORS[m], np.array(R["methods"][m]["A_mean_over_test_by_k"])) for m in METHODS],
        R["ceiling_gs_mean"],
        f"Clipping ceiling ({R['ceiling_gs_mean']:.2f})",
    )


def california_curves(R):
    gs = 10.0  # the output range in std units (Y_MAX - Y_MIN), as in agtr/california/release.py
    return (
        [
            (f"BS={bs}", BS_COLORS[i], np.array(e["A_mean_by_k_std_units"]))
            for i, (bs, e) in enumerate(R["batch_sizes"].items())
        ],
        gs,
        f"Output range GS ({gs:g})",
    )


def draw(runs, out, figsize=(7.0, 5.3), page_width_in=3.5):
    """runs: [(group title, results dict, curves function), ...]; one panel per run, each 1.2 times as
    tall as wide, the type sizes of the paper's rows once the figure sits in one column."""
    plotting.apply_combined(
        2,
        figsize=figsize,
        page_width_in=page_width_in,
        print_target_pt=8,
        label_pt=8,
        tick_pt=7,
        legend_pt=7,
    )
    fig, axes = plt.subplots(1, len(runs), figsize=figsize)
    fig.subplots_adjust(left=0.11, right=0.93, top=0.93, bottom=0.35, wspace=0.62)
    lw = plt.rcParams["lines.linewidth"]
    legends = []
    for ax, (name, R, curves) in zip(axes, runs):
        series, gs, gs_label = curves(R)
        ax2 = ax.twinx()
        for label, col, A in series:
            ks = np.arange(len(A))
            ax.step(ks, A, where="post", color=col, ls="--", lw=lw, alpha=0.75, label=label)
            t = np.linspace(0, len(A) + 4, 600)
            Ai = np.where(t > len(A) - 1, gs, A[np.clip(t.astype(int), 0, len(A) - 1)])
            s = Ai * np.exp(-BETA * t)
            ax2.plot(t, s, "-", color=col, lw=lw + 0.4)
            j = int(np.argmax(s))
            ax2.plot(
                t[j],
                s[j],
                "*",
                color=col,
                ms=plt.rcParams["lines.markersize"] * 2.4,
                markeredgecolor="black",
                zorder=10,
            )
        ax.axhline(gs, color="black", ls=":", lw=2.0, label=gs_label)
        ax.set(xlabel=r"Substitutions $k$", ylabel=r"$\bar{A}(x,k)$ (dashed)", xlim=(0, 20), xticks=range(0, 21, 5))
        ax.set_box_aspect(1.2)
        ax.set_title(name, fontweight="bold", fontsize=plt.rcParams["axes.titlesize"] * 1.15, pad=8)
        ax2.set_ylabel(r"$SS^{\beta}$ at $\beta=1/2$ (solid)")
        ax.grid(True, alpha=0.3)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        legends.append(ax.get_legend_handles_labels())
    # one two-column legend under each panel
    for ax, (h, l) in zip(axes, legends):
        pos = ax.get_position()
        fig.legend(
            h,
            l,
            loc="lower center",
            bbox_to_anchor=((pos.x0 + pos.x1) / 2, 0.01),
            ncol=2,
            **{**plotting.LEGEND_BELOW_KW, "columnspacing": 0.8, "handlelength": 1.4, "handletextpad": 0.4},
        )
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--california-tag", default="relu_h64_clamp_gamma0.1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    runs = [
        ("Linear", json.load(open(LINEAR)), linear_curves),
        ("California", json.load(open(CALIFORNIA.format(tag=a.california_tag))), california_curves),
    ]
    draw(runs, a.out)
    print(f"wrote {a.out}  [california {a.california_tag}]")


if __name__ == "__main__":
    main()
