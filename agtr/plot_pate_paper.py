"""The paper's AGT-R figure: the matched PATE comparison on linear and California in one row.

Four panels, pure and approximate DP for each dataset, from the two stored results.json of the
matched benchmarks (agtr/linear/pate/release.py, agtr/california/pate/release.py; computes nothing).
Same five arms and the same axes as the per-dataset pate_best_vs_eps figures: PATE-R and PATE-AGT-R
at their best T, AGT-R on the full data, AGT-R on one shard with the amplified budget at its best T,
and global sensitivity on the full data; one legend under the row, a group title over each dataset's
pair of panels. Pure DP is the tighter Cauchy release at beta = eps / 2 and the clipped release's mean
absolute error (MAE); approximate DP is Laplace(2 SS / eps) at delta = 1e-5 and the test MSE.

    PYTHONPATH=. python -m agtr.plot_pate_paper
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import plotting

LINEAR = "agtr/linear/values/pate_matched/{tag}/results.json"
CALIFORNIA = "agtr/california/values/agt_2026-09-14/{tag}/results.json"
OUT = "agtr/figures/pate_paper_linear_california.png"
ARMS = (  # label, colour, line style, z-order, line-width factor: on linear PATE-AGT-R's best T is T = 1
    ("PATE-R, best $T$", "black", "o-", 20, 1.0),  # at eps >= 1, i.e. the full-data arm, so that one is
    ("PATE-AGT-R, best $T$", "#d62728", "o-", 10, 2.0),  # drawn on top and the wider red shows beneath it
    ("AGT-R, full data ($T=1$)", "#1f77b4", "o-", 15, 1.0),
    ("AGT-R, one shard + amplification, best $T$", "#2ca02c", "o--", 5, 1.0),
    ("GS on full data ($T=1$)", "gray", "o:", 1, 1.0),
)
MODES = (  # gs mode, agt mode, metric, y label, panel title
    ("gs_pure", "pure_liang_b2", "mae", "MAE", r"Pure $(\epsilon, 0)$-DP"),
    ("gs_approx", "approx", "mse", "MSE", r"Approx. $(\epsilon, 10^{-5})$-DP"),
)


def series(R, gs_mode, agt_mode, metric):
    """The five arms' curves over R['epsilons'], in ARMS order."""
    T = {int(k): v for k, v in R["T"].items()}
    Ts = sorted(T)
    n = len(R["epsilons"])
    best = lambda arm, mode: [min(T[t][arm][mode][metric][j] for t in Ts) for j in range(n)]
    return [
        best("pate", gs_mode),
        best("pate", agt_mode),
        T[1]["full"][agt_mode][metric],
        best("wor", agt_mode),
        T[1]["pate"][gs_mode][metric],
    ]


NOMINAL = {"mae": "nominal_mae_pate", "mse": "nominal_mse_pate"}  # the non-private error, stored per T


def draw(runs, out, *, show_gs=True, excess=(), figsize=(14.0, 3.9), page_width_in=7.16, rc_figsize=None):
    # rc_figsize: the size the fonts and line widths are scaled for, when it differs from figsize (a group
    # drawn alone keeps the combined figure's panel style)
    """runs: [(group title, results dict), ...]; one row of two panels per run. show_gs: True, False, or the
    set of group titles whose panels keep the GS arm. excess: the group titles whose panels draw the error
    above the non-private one (error minus R["T"]["1"][NOMINAL[metric]]) so that arms that all reach the
    floor stay apart on the log axis; a point at the floor is left out."""
    plotting.apply_combined(
        2 * len(runs) if rc_figsize is None else 4,
        figsize=rc_figsize or figsize,
        page_width_in=page_width_in,
        print_target_pt=8,
        label_pt=8,
        tick_pt=7,
        legend_pt=7,
    )
    fig, axes = plt.subplots(1, 2 * len(runs), figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9), wspace=0.06)  # the top strip holds the group titles
    for g, (name, R) in enumerate(runs):
        for m, (gs_mode, agt_mode, metric, ylab, title) in enumerate(MODES):
            ax = axes[2 * g + m]
            lw = plt.rcParams["lines.linewidth"]
            gs_here = show_gs if isinstance(show_gs, bool) else name in show_gs
            floor = R["T"]["1"][NOMINAL[metric]] if name in excess else 0.0
            for (label, col, sty, z, wf), y in zip(ARMS, series(R, gs_mode, agt_mode, metric)):
                if label.startswith("GS") and not gs_here:
                    continue
                pts = [(e, v - floor) for e, v in zip(R["epsilons"], y) if v - floor > 0]
                ax.plot(
                    [e for e, _ in pts],
                    [v for _, v in pts],
                    sty,
                    color=col,
                    zorder=z,
                    lw=lw * wf,
                    label=label,
                    **plotting.SERIES,
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=r"Privacy Budget $\epsilon$",
                ylabel=("Excess " + ylab) if name in excess else ylab,
                title=title,
            )
            ax.grid(True, alpha=0.3)
    style = dict(columnspacing=1.4, handlelength=1.8, handletextpad=0.5)
    if len(runs) > 1:
        plotting.shared_legend(fig, axes, ncol=5, **style)
    else:  # a group alone: three rows, the two PATE arms, the two AGT-R arms, then GS on its own
        h, lab = axes[0].get_legend_handles_labels()
        order = [k for k in (0, 2, 4, 1, 3) if k < len(lab)]  # filled by column: (0, 2, 4), (1, 3)
        fig.legend(
            [h[k] for k in order],
            [lab[k] for k in order],
            loc="outside lower center",
            ncol=2,
            **{**plotting.LEGEND_BELOW_KW, **style},
        )
    fig.canvas.draw()  # the group titles sit over the centre of each pair once the panels are placed
    size = plt.rcParams["axes.titlesize"]
    for g, (name, _) in enumerate(runs):
        left, right = axes[2 * g].get_position(), axes[2 * g + 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            0.965,
            name,
            ha="center",
            va="center",
            fontsize=size * 1.15,
            fontweight="bold",
        )
    fig.savefig(out)
    plt.close(fig)


def draw_two_row(runs, out, figsize=(7.0, 7.2), page_width_in=3.5):
    """The same figure as draw(), every arm and the GS arm drawn, as two rows of two panels (one dataset
    per row) so it fits one column: group title over each row, one legend under the grid."""
    plotting.apply_combined(
        2, figsize=figsize, page_width_in=page_width_in, print_target_pt=8, label_pt=8, tick_pt=7, legend_pt=7
    )
    fig, axes = plt.subplots(len(runs), 2, figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1), wspace=0.08, hspace=0.16)
    lw = plt.rcParams["lines.linewidth"]
    for g, (name, R) in enumerate(runs):
        for m, (gs_mode, agt_mode, metric, ylab, title) in enumerate(MODES):
            ax = axes[g, m]
            for (label, col, sty, z, wf), y in zip(ARMS, series(R, gs_mode, agt_mode, metric)):
                ax.plot(R["epsilons"], y, sty, color=col, zorder=z, lw=lw * wf, label=label, **plotting.SERIES)
            ax.set(xscale="log", yscale="log", xlabel=r"Privacy Budget $\epsilon$", ylabel=ylab, title=title)
            ax.grid(True, alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="outside lower center",
        ncol=2,
        **{**plotting.LEGEND_BELOW_KW, "columnspacing": 1.0, "handlelength": 1.8, "handletextpad": 0.5},
    )
    fig.canvas.draw()
    size = plt.rcParams["axes.titlesize"]
    for g, (name, _) in enumerate(runs):  # a group title centred over each row, above the panel titles
        left, right = axes[g, 0].get_position(), axes[g, 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            left.y1 + 0.055,
            name,
            ha="center",
            va="bottom",
            fontsize=size * 1.15,
            fontweight="bold",
        )
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--linear-tag", default="exact_n40000_e20_lr0.3_g1")
    ap.add_argument("--california-tag", default="pate_relu_h64_clamp_gamma0.1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--no-gs", action="store_true", help="leave out the global-sensitivity arm (stem gets _no_gs)"
    )
    ap.add_argument(
        "--group",
        choices=("Linear", "California"),
        default=None,
        help="draw one group only, as in the combined figure (stem gets _<group>_only)",
    )
    ap.add_argument(
        "--excess-linear",
        action="store_true",
        help="the Linear panels show the error above the non-private one without the GS arm; "
        "California keeps it (stem gets _excess_linear)",
    )
    ap.add_argument(
        "--two-row",
        action="store_true",
        help="the full figure as two rows of two panels, one dataset per row, for one column (stem gets _2_row)",
    )
    ap.add_argument(
        "--excess",
        action="store_true",
        help="every panel shows the error above the non-private one, without the GS arm (stem gets _excess)",
    )
    ap.add_argument(
        "--no-gs-linear",
        action="store_true",
        help="leave the GS arm out of the Linear panels only, so the PATE arms are legible there; "
        "California keeps it (stem gets _no_gs_linear)",
    )
    a = ap.parse_args()
    runs = [
        ("Linear", json.load(open(LINEAR.format(tag=a.linear_tag)))),
        ("California", json.load(open(CALIFORNIA.format(tag=a.california_tag)))),
    ]
    if a.group:
        runs = [r for r in runs if r[0] == a.group]
        draw(
            runs,
            a.out.replace(".png", f"_{a.group.lower()}_only.png"),
            figsize=(7.0, 3.9),
            rc_figsize=(14.0, 3.9),
        )
        return
    if a.two_row:
        draw_two_row(runs, a.out.replace(".png", "_2_row.png"))
        return
    if a.excess:
        draw(runs, a.out.replace(".png", "_excess.png"), excess={"Linear", "California"}, show_gs=False)
        return
    if a.excess_linear:
        draw(runs, a.out.replace(".png", "_excess_linear.png"), excess={"Linear"}, show_gs={"California"})
        return
    if a.no_gs_linear:
        draw(runs, a.out.replace(".png", "_no_gs_linear.png"), show_gs={"California"})
        return
    out = a.out.replace(".png", "_no_gs.png") if a.no_gs else a.out
    draw(runs, out, show_gs=not a.no_gs)
    print(f"wrote {out}  [linear {a.linear_tag}, california {a.california_tag}]")


if __name__ == "__main__":
    main()
