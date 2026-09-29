"""The MNIST optimized figure: pure and approximate DP side by side, one shared legend.

Reads runs/optimized/optimized_headline.json (written by run_optimized.py release) and computes
nothing. Arms: non-private, PATE, DP-SGD (matched), DP-SGD (optimized), AGS (matched), AGS
(optimized). The AGS arms are drawn for one variant of the release, chosen on the command line:

    --certificate cross_radius | width        the paper's certificate, or the reference's width
    --beta-rule aggregate | dimension_free    beta/P (sound) or Corollary 4.1 as written
                | pld_coordinate              per-coordinate Laplace composed by PLD (approx only;
                                              the pure panel then shows the aggregate rule)

The default (cross_radius, aggregate) is the sound whole-vector release and writes
fig5_combined_optimized; the other variants carry a suffix. Same footprint as the blobs combined
figure so the two can be placed the same way.
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from ags.mnist.config import OPTIMIZED, OPTIMIZED_SMOKE
from common import plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--certificate", choices=("cross_radius", "width"), default="cross_radius")
    ap.add_argument(
        "--beta-rule",
        choices=("aggregate", "dimension_free", "pld_coordinate"),
        default="aggregate",
        dest="beta_rule",
    )
    ap.add_argument("--width", type=float, default=6.7)
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--page-width", type=float, default=plotting.COLUMN_WIDTH_IN, dest="page_width")
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C = OPTIMIZED_SMOKE if a.smoke else OPTIMIZED

    H = json.load(open(f"{C['runs_dir']}/optimized_headline.json"))
    eps = H["eps"]
    suffix = ("" if a.certificate == "cross_radius" else "_width") + (
        "" if a.beta_rule == "aggregate" else f"_{a.beta_rule}"
    )
    stem = f"fig5_combined_optimized{suffix}" + ("_smoke" if a.smoke else "")

    figsize = (a.width, a.height)
    plotting.apply_combined(2, figsize=figsize, page_width_in=a.page_width)
    from common.plotting import BLUES, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    handles = {}

    def variant(mode):
        rule = a.beta_rule if not (a.beta_rule == "pld_coordinate" and mode == "pure") else "aggregate"
        return f"{rule}/{a.certificate}"

    def series(mode, arm, key=None):
        out = []
        for e in eps:
            cell = H["arms"][mode][str(e)][arm]
            if key is not None:
                cell = cell.get(key) if cell else None
            out.append(cell["test"] if cell else float("nan"))
        return out

    def panel(ax, mode, title):
        v = variant(mode)
        arms = [
            ("AGS (optimized)", series(mode, "ags_optimized", v), dict(color=BLUES[-1], marker="o", ls="-")),
            ("AGS (matched)", series(mode, "ags_matched", v), dict(color=BLUES[2], marker="o", ls="--")),
            ("DP-SGD (optimized)", series(mode, "dpsgd_optimized"), dict(color=C_DP, marker="s", ls="-")),
            ("DP-SGD (matched)", series(mode, "dpsgd_matched"), dict(color=C_DP, marker="s", ls="--")),
            ("PATE", series(mode, "pate"), dict(color=C_PATE, marker="^", ls="-")),
        ]
        for lab, y, sty in arms:
            (h,) = ax.plot(
                eps, y, zorder=6 if "AGS" in lab else 5, markeredgecolor="white", markeredgewidth=1.1, **sty
            )
            handles.setdefault(lab, h)
        nominal = H["arms"][mode][str(eps[0])]["non_private"]
        if nominal is not None:
            ax.axhline(nominal, color=C_NOM, ls=":", lw=2.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ax.set_ylim(0.05, 0.95)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, pad=8)
        return nominal

    nominal = panel(axes[0], "pure", r"Pure $(\varepsilon, 0)$-DP")
    panel(axes[1], "approx", r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP")
    axes[0].set_ylabel("Test Accuracy")
    if nominal is not None:
        axes[0].text(
            1.05,
            nominal + 0.006,
            "non-private",
            ha="left",
            va="bottom",
            fontsize=plt.rcParams["legend.fontsize"],
            color=C_NOM,
        )
    order = ["AGS (optimized)", "AGS (matched)", "DP-SGD (optimized)", "DP-SGD (matched)", "PATE"]
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="outside lower center",
        ncol=a.ncol,
        **plotting.LEGEND_BELOW_KW,
    )
    fig.savefig(f"{C['figures_dir']}/{stem}.png")
    plt.close(fig)
    print(f"wrote {C['figures_dir']}/{stem}.png  [certificate={a.certificate}, beta rule={a.beta_rule}]")


if __name__ == "__main__":
    main()
