"""The paper's AGS figure: 5-class MNIST and IMDB, pure and approximate DP, in one row.

Four panels from the two dense headlines (ags/mnist/runs/contractive5/headline_dense.json and
ags/imdb/runs/contractive/headline_dense.json, written by the contractive drivers' release stage;
computes nothing). The arms, colours and line styles are those of fig8_paper_contractive_5class and
fig9_paper_contractive_imdb (common.plotting.contractive_paper_figure): AGS under the proven release
rule, matched DP-SGD (basic, and composed in approximate DP), optimized DP-SGD, PATE, the non-private
line as a band, and on IMDB the hidden-layer group's AGS and matched arms. One legend under the row, a
group title over each dataset's pair of panels, budgets up to eps = 100.

    PYTHONPATH=. python -m ags.plot_paper_combined [--no-hidden]
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from common import plotting
from common.plotting import BLUES, C_DPSGD_COMPOSED, C_DPSGD_MATCHED, C_DPSGD_OPT, C_NOMINAL, C_PATE

MNIST = "ags/mnist/runs/contractive5/headline_dense.json"
IMDB = "ags/imdb/runs/contractive/headline_dense.json"
OUT = "ags/figures/ags_paper_mnist_imdb.png"
VARIANT = "tight/cross_radius"
EPS_MAX = 100
TITLES = {"pure": r"Pure $(\varepsilon, 0)$-DP", "approx": r"Approx. $(\varepsilon, 10^{-5})$-DP"}


def arms_of(arms, extra, mode):
    """(label, row getter, style) for each arm drawn in `mode`, from the headline's arms[mode][eps]."""
    spec = [
        ("AGS", arms, lambda r: r["ags"][VARIANT], dict(color=BLUES[-1], marker="o", ls="-", zorder=6)),
        (
            "DP-SGD (matched, basic)",
            arms,
            lambda r: r["dpsgd_matched"]["basic"],
            dict(color=C_DPSGD_MATCHED, marker="s", ls="--"),
        ),
    ]
    if mode == "approx":
        spec.append(
            (
                "DP-SGD (matched, composed)",
                arms,
                lambda r: r["dpsgd_matched"]["composed"],
                dict(color=C_DPSGD_COMPOSED, marker="D", ls=(0, (1, 1.2)), lw=1.6),
            )
        )
    spec += [
        (
            "DP-SGD (optimized)",
            arms,
            lambda r: r["dpsgd_optimized"],
            dict(color=C_DPSGD_OPT, marker="P", ls="-"),
        ),
        ("PATE", arms, lambda r: r["pate"], dict(color=C_PATE, marker="^", ls="-")),
    ]
    if extra is not None:
        spec += [
            ("AGS (hidden)", extra, lambda r: r["ags"][VARIANT], dict(color=BLUES[-3], marker="v", ls="-.")),
            (
                "DP-SGD (matched, hidden)",
                extra,
                lambda r: r["dpsgd_matched"]["basic"],
                dict(color=C_DPSGD_MATCHED, marker="v", ls="-.", alpha=0.75),
            ),
        ]
    return spec


def series(A, mode, getter, xs):
    out = []
    for e in xs:
        try:
            v = getter(A[mode][str(e)])
        except (KeyError, TypeError):
            v = None
        out.append(v["test"] if v else float("nan"))
    return out


def draw(runs, out, *, figsize=(14.0, 3.9), page_width_in=7.16, xticks=(1, 3, 10, 30, 100)):
    """runs: [(group title, arms, extra arms or None, eps, non-private line, ylim), ...]."""
    plotting.apply_combined(
        4, figsize=figsize, page_width_in=page_width_in, print_target_pt=8, label_pt=8, tick_pt=7, legend_pt=7
    )
    fig, axes = plt.subplots(1, 2 * len(runs), figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9), wspace=0.06)
    handles = {}
    lw = plt.rcParams["lines.linewidth"]
    ms = plt.rcParams["lines.markersize"]
    for g, (name, arms, extra, eps, nonpriv, ylim) in enumerate(runs):
        xs = [e for e in eps if e <= EPS_MAX]
        for m, mode in enumerate(("pure", "approx")):
            ax = axes[2 * g + m]
            for label, A, getter, sty in arms_of(arms, extra, mode):
                sty = dict(dict(ms=ms, lw=lw, markeredgecolor="white", markeredgewidth=0.6, zorder=5), **sty)
                (h,) = ax.plot(xs, series(A, mode, getter, xs), **sty)
                handles.setdefault(label, h)
            band = ax.axhline(nonpriv, color=C_NOMINAL, ls="-", lw=lw * 4, alpha=0.28, zorder=1)
            handles.setdefault("non-private", band)
            ax.set_xscale("log")
            ax.set_xlim(0.9, EPS_MAX * 1.1)
            ax.set_ylim(*ylim)
            ax.set_xticks(list(xticks))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set(xlabel=r"Privacy Budget $\varepsilon$", ylabel="Test Accuracy", title=TITLES[mode])
    order = [k for k in plotting.PAPER_ARM_ORDER if k in handles]
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="outside lower center",
        ncol=(len(order) + 1) // 2 if len(order) > 6 else len(order),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.9,
        handletextpad=0.5,
    )
    fig.canvas.draw()
    size = plt.rcParams["axes.titlesize"]
    for g, run in enumerate(runs):
        left, right = axes[2 * g].get_position(), axes[2 * g + 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            0.965,
            run[0],
            ha="center",
            va="center",
            fontsize=size * 1.15,
            fontweight="bold",
        )
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--no-hidden", action="store_true", help="leave out IMDB's hidden-layer arms (stem gets _no_hidden)"
    )
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    H = json.load(open(MNIST))
    Hd = json.load(open(IMDB))
    G = Hd["groups"]["logistic"]
    extra = None if a.no_hidden else Hd["groups"].get("mlp", {}).get("arms")
    runs = [
        ("5-class MNIST", H["arms"], None, H["eps"], H["non_private_line"], (0.15, 0.95)),
        ("IMDB", G["arms"], extra, Hd["eps"], G["non_private_line"], (0.45, 0.85)),
    ]
    import os

    out = a.out.replace(".png", "_no_hidden.png") if a.no_hidden else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw(runs, out)
    print(f"wrote {out}  [mnist5 {MNIST}, imdb {IMDB}, eps<=100]")


if __name__ == "__main__":
    main()
