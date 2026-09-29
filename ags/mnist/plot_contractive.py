"""The contractive-trajectory figure: pure and approximate DP side by side, one shared legend.

Reads runs/<task>/headline.json (written by run_contractive.py release) and computes nothing. Arms:
non-private, PATE, DP-SGD (matched; in approx DP both basic and exact Gaussian composition are drawn
and labelled), DP-SGD (optimized), AGS. The AGS arm is one variant of the release, chosen on the
command line as for plot_optimized.py:

    --certificate cross_radius | width
    --beta-rule aggregate | dimension_free

The default (cross_radius, aggregate) is the sound whole-vector release and writes the stem named in
the task's config; other variants carry a suffix.
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from ags.mnist.run_contractive import TASKS
from common import plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=tuple(TASKS), default="mnist10")
    ap.add_argument("--certificate", choices=("cross_radius", "width"), default="cross_radius")
    ap.add_argument(
        "--beta-rule",
        choices=("tight", "aggregate", "corollary_p", "dimension_free"),
        default="tight",
        dest="beta_rule",
        help="tight (default): the tightest proven release; aggregate: the earlier beta/P rule with the reference constants; corollary_p: Cauchy(6u/eps) at beta = eps/(6P); dimension_free: Corollary 4.1 as written, not established for P > 1",
    )
    ap.add_argument("--width", type=float, default=6.7)
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--page-width", type=float, default=plotting.COLUMN_WIDTH_IN, dest="page_width")
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument(
        "--sweep", action="store_true", help="draw the sweep-scale headline (10k subset) instead of the final"
    )
    a = ap.parse_args()
    C = TASKS[a.task]

    hf = C.get("headline_file", "headline.json")
    H = json.load(open(f"{C['runs_dir']}/{hf.replace('.json', '_sweep.json') if a.sweep else hf}"))
    eps = H["eps"]
    variant = f"{a.beta_rule}/{a.certificate}"
    suffix = ("" if a.certificate == "cross_radius" else "_width") + (
        "" if a.beta_rule == "tight" else f"_{a.beta_rule}"
    )
    stem = C["fig_stem"] + suffix + ("" if H["scale"] == "full" else "_sweep")

    figsize = (a.width, a.height)
    plotting.apply_combined(2, figsize=figsize, page_width_in=a.page_width)
    from common.plotting import (
        BLUES,
        C_DPSGD_COMPOSED,
        C_DPSGD_MATCHED,
        C_DPSGD_OPT,
        C_NOMINAL as C_NOM,
        C_PATE,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    handles = {}

    def series(mode, getter):
        out = []
        for e in eps:
            try:
                v = getter(H["arms"][mode][str(e)])
            except (KeyError, TypeError):
                v = None
            out.append(v["test"] if v else float("nan"))
        return out

    def panel(ax, mode, title):
        arms = [("AGS", series(mode, lambda r: r["ags"][variant]), dict(color=BLUES[-1], marker="o", ls="-"))]
        arms.append(
            (
                "DP-SGD (matched, basic)",
                series(mode, lambda r: r["dpsgd_matched"]["basic"]),
                dict(color=C_DPSGD_MATCHED, marker="s", ls="--"),
            )
        )
        if mode == "approx":
            arms.append(
                (
                    "DP-SGD (matched, composed)",
                    series(mode, lambda r: r["dpsgd_matched"]["composed"]),
                    dict(color=C_DPSGD_COMPOSED, marker="D", ls=(0, (1, 1.2)), lw=2.2),
                )
            )
        arms.append(
            (
                "DP-SGD (optimized)",
                series(mode, lambda r: r["dpsgd_optimized"]),
                dict(color=C_DPSGD_OPT, marker="P", ls="-"),
            )
        )
        arms.append(("PATE", series(mode, lambda r: r["pate"]), dict(color=C_PATE, marker="^", ls="-")))
        for lab, y, sty in arms:
            (h,) = ax.plot(
                eps, y, zorder=6 if lab == "AGS" else 5, markeredgecolor="white", markeredgewidth=1.1, **sty
            )
            handles.setdefault(lab, h)
        band = ax.axhline(H["non_private_line"], color=C_NOM, ls="-", lw=7, alpha=0.28, zorder=1)
        handles.setdefault("non-private", band)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 1100)
        ax.set_ylim(0.05, 1.0)
        ax.set_xticks([1, 10, 100, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, pad=8)

    panel(axes[0], "pure", r"Pure $(\varepsilon, 0)$-DP")
    panel(axes[1], "approx", r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP")
    axes[0].set_ylabel("Test Accuracy")
    order = [
        "AGS",
        "DP-SGD (matched, basic)",
        "DP-SGD (matched, composed)",
        "DP-SGD (optimized)",
        "PATE",
        "non-private",
    ]
    order = [k for k in order if k in handles]
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="outside lower center",
        ncol=a.ncol,
        **plotting.LEGEND_BELOW_KW,
    )
    fig.savefig(f"{C['figures_dir']}/{stem}.png")
    plt.close(fig)
    print(
        f"wrote {C['figures_dir']}/{stem}.png  [task={a.task}, scale={H['scale']}, certificate={a.certificate}, beta rule={a.beta_rule}]"
    )


if __name__ == "__main__":
    main()
