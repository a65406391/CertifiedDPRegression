"""The blobs combined figure: pure and approximate DP side by side, one shared legend.

The panels are SEPARATE MECHANISMS and their curves are not to be read against each
other; they share a y-axis only so the shapes are legible at a glance.

Footprint: 6.7 x 5 in, sized for a SINGLE COLUMN of a two-column page (3.35 in) rather
than the full text block the AGT-R rows use -- so its type is scaled for that placement
and still prints at 9 pt. --width/--height/--page-width override it. The panels get narrower and the type comes down with them
(plotting.apply_combined(2)); the legend sits below both panels so it costs no plot
area. The arms drawn here match the single figures exactly -- tuned DP-SGD and the RDP
accountant are in the JSONs but are not plotted.

--certificate cross_radius reads the _cross_radius JSONs that plot_headline and plot_approx
write under the same flag and saves the figure as fig6_combined_cross_radius; the default reads
the reference JSONs and writes fig4_combined.
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from common import plotting

# The box is a CLI argument so the one-column variants can be generated and compared
# without editing the script. --page-width is what the figure will be SCALED TO on the
# page (A4 text block 6.3 in, a two-column A4 column about 3.35 in); the type is sized
# from it, so a narrow figure meant for a column still prints at 9 pt.


def main():
    _ap = argparse.ArgumentParser()
    # Default: option E -- 6.7 x 5 in, sized for a single column of a two-column page.
    # Chosen from the five compared in the width study; the CLI reproduces any of them.
    _ap.add_argument("--width", type=float, default=6.7)
    _ap.add_argument("--height", type=float, default=5.0)
    _ap.add_argument("--page-width", type=float, default=plotting.COLUMN_WIDTH_IN, dest="page_width")
    _ap.add_argument("--ncol", type=int, default=2)
    _ap.add_argument("--suffix", default="")
    _ap.add_argument(
        "--certificate",
        choices=("width", "cross_radius"),
        default="width",
        help="which certificate's JSONs to read: width (fig4_combined) or cross_radius (fig6_combined_cross_radius)",
    )
    _a, _ = _ap.parse_known_args()

    _FIGSIZE = (_a.width, _a.height)
    plotting.apply_combined(2, figsize=_FIGSIZE, page_width_in=_a.page_width)
    OUT, RUNS = "ags/blobs/figures", "ags/blobs/runs"
    from common.plotting import BLUES as _BLUES, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

    BLUES = _BLUES[1:]  # this figure shows four AGS curves, not five
    FS = [1, 2, 5, 10]

    cert = "_cross_radius" if _a.certificate == "cross_radius" else ""
    stem = "fig6_combined_cross_radius" if cert else "fig4_combined"
    pure = json.load(open(f"{RUNS}/pure_headline{cert}.json"))
    apx = json.load(open(f"{RUNS}/approx_headline_v2{cert}.json"))
    EPS = pure["eps"]
    ags_key = lambda d, F: next(
        v for k, v in d["ags"].items() if f"={F})" in k or f"={F}" == k.split("=")[-1].rstrip(")")
    )

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE, sharey=True, layout="constrained")
    handles = {}

    def panel(ax, ags, dps, title):
        for F, c in zip(FS, BLUES):
            (h,) = ax.plot(
                EPS,
                ags[F],
                color=c,
                marker="o",
                ls="-",
                zorder=6,
                markeredgecolor="white",
                markeredgewidth=1.1,
            )
            handles.setdefault(f"AGS (MILP, $F$={F})", h)
        for lab, y, sty in dps:
            (h,) = ax.plot(EPS, y, zorder=5, **sty)
            handles.setdefault(lab, h)
        ax.axhline(pure["nominal"], color=C_NOM, ls=":", lw=2.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ax.set_ylim(0.44, 0.99)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, pad=8)

    sq = dict(color=C_DP, marker="s", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    di = dict(color=C_DP, marker="D", ls="--", markeredgecolor="white", markeredgewidth=1.1)
    tu = dict(color=C_DP, marker="s", ls=":", lw=2.3, alpha=0.55, markerfacecolor="none")
    pt = dict(color=C_PATE, marker="^", ls="-", markeredgecolor="white", markeredgewidth=1.1)

    panel(
        axes[0],
        {F: ags_key(pure, F) for F in FS},
        [("DP-SGD (matched)", pure["dpsgd_matched"], sq), ("PATE", pure["pate"], pt)],
        r"Pure $(\varepsilon, 0)$-DP",
    )
    panel(
        axes[1],
        {F: ags_key(apx, F) for F in FS},
        [("DP-SGD (matched)", apx["dpsgd_matched_naive"], sq), ("PATE", apx["pate"], pt)],
        r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP",
    )

    axes[0].set_ylabel("Test Accuracy")
    axes[0].text(
        1.05,
        pure["nominal"] + 0.006,
        "non-private",
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["legend.fontsize"],
        color=C_NOM,
    )
    order = [f"AGS (MILP, $F$={F})" for F in FS] + ["DP-SGD (matched)", "PATE"]
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="outside lower center",
        ncol=_a.ncol,
        **plotting.LEGEND_BELOW_KW,
    )
    fig.savefig(f"{OUT}/{stem}{_a.suffix}.png")
    print(
        f"wrote {OUT}/{stem}{_a.suffix}.png  "
        f"[{_FIGSIZE[0]:g} x {_FIGSIZE[1]:g} in, for a {_a.page_width:g} in placement]"
    )
    print("  note: RDP line appears only on the right -- pure DP admits no such composition")


if __name__ == "__main__":
    main()
