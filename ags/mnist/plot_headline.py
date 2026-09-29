"""Pure and approximate DP side by side, in the blobs study's figure style.

Port of AGS_Ablation/blobs/make_sidebyside.py. Reads runs/headline.json and never
recomputes anything -- the envelope, the releases and the baselines are all produced
by run_headline.py.

The panels are SEPARATE MECHANISMS and their curves are not to be read against each
other; they share a y-axis only so the shapes are legible at a glance. The one thing
the pairing does show honestly is structural: the (matched, RDP) line exists only on
the right, because Renyi composition requires delta > 0. Pure DP has no composition
theorem better than basic, which is the whole reason AGS has a case on the left.

    python -m ags.mnist.plot_headline
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from common import plotting


def main():
    plotting.apply()

    OUT = "ags/mnist/figures"
    RUNS = "ags/mnist/runs"
    os.makedirs(OUT, exist_ok=True)

    # blobs palette: AGS blue, DP-SGD vermilion, PATE green, nominal grey
    from common.plotting import C_AGS, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

    R = json.load(open(f"{RUNS}/headline.json"))
    EPS = R["eps"]
    S = R["setting"]
    NOM = R["nominal"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharey=True)
    handles = {}

    ags_sty = dict(color=C_AGS, marker="o", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    sq = dict(color=C_DP, marker="s", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    di = dict(color=C_DP, marker="D", ls="--", markeredgecolor="white", markeredgewidth=1.1)
    tu = dict(color=C_DP, marker="s", ls=":", lw=2.3, alpha=0.55, markerfacecolor="none")
    pt = dict(color=C_PATE, marker="^", ls="-", markeredgecolor="white", markeredgewidth=1.1)

    def panel(ax, series, title):
        for lab, y, sty in series:
            (h,) = ax.plot(EPS, y, zorder=6 if lab.startswith("AGS") else 5, **sty)
            handles.setdefault(lab, h)
        ax.axhline(NOM, color=C_NOM, ls=":", lw=2.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ax.set_ylim(0.04, 0.92)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, fontsize=18, pad=9)

    panel(
        axes[0],
        [
            ("AGS (IBP)", R["ags"]["pure"]["acc"], ags_sty),
            ("DP-SGD (matched, naive)", R["dpsgd"]["matched_pure"], sq),
            ("DP-SGD (tuned, naive)", R["dpsgd"]["tuned_pure"], tu),
            ("PATE", R["pate"]["pure"], pt),
        ],
        r"Pure $(\varepsilon, 0)$-DP",
    )
    panel(
        axes[1],
        [
            ("AGS (IBP)", R["ags"]["approx"]["acc"], ags_sty),
            ("DP-SGD (matched, naive)", R["dpsgd"]["matched_approx"], sq),
            ("DP-SGD (matched, RDP)", R["dpsgd"]["rdp_approx"], di),
            ("DP-SGD (tuned, naive)", R["dpsgd"]["tuned_approx"], tu),
            ("PATE", R["pate"]["approx"], pt),
        ],
        r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP",
    )

    axes[0].set_ylabel("Test Accuracy")
    axes[0].text(1.05, NOM + 0.008, "non-private", ha="left", va="bottom", fontsize=13, color=C_NOM)

    order = ["AGS (IBP)", "DP-SGD (matched, naive)", "DP-SGD (matched, RDP)", "DP-SGD (tuned, naive)", "PATE"]
    fig.subplots_adjust(wspace=0.06, bottom=0.30)
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        columnspacing=1.6,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.5,
        fontsize=14,
        frameon=False,
    )
    fig.savefig(f"{OUT}/fig1_pure_vs_approx.png")
    plt.close(fig)
    print(f"wrote {OUT}/fig1_pure_vs_approx.png")
    print("  note: RDP line appears only on the right -- pure DP admits no such composition")

    # ---------------------------------------------------------------- envelope figure
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    ks = R["k_grid"]
    w = [r["width_mean"] for r in R["envelope"]["rows"]]
    gs = R["envelope"]["gs_width_mean"]

    ax = axes[0]
    ax.plot(
        ks,
        w,
        color=C_AGS,
        marker="o",
        ls="-",
        markeredgecolor="white",
        markeredgewidth=1.1,
        label=r"$A^k$ (IBP width)",
        zorder=6,
    )
    ax.axhline(gs, color=C_NOM, ls="--", lw=2.0, zorder=3)
    ax.text(1.15, gs * 0.93, f"GS $= A^{{k=b}} = {gs:g}$", fontsize=13, color="0.3", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Substitutions  $k$")
    ax.set_ylabel(r"Certified width  $A^k$")
    ax.set_title(r"Envelope saturates at GS by $k=b$", fontsize=17, pad=9)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper left", fontsize=13)

    ax = axes[1]
    for mode, ls, mk in (("pure", "-", "o"), ("approx", "--", "D")):
        ax.plot(
            EPS,
            R["ags"][mode]["ss_mean"],
            color=C_AGS,
            ls=ls,
            marker=mk,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label=f"SS ({mode})",
            zorder=6,
        )
    ax.axhline(gs, color=C_NOM, ls="--", lw=2.0, zorder=3)
    ax.text(1.05, gs * 0.93, "GS", fontsize=13, color="0.3", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
    ax.set_ylabel(r"Smooth sensitivity  $SS^\beta$")
    ax.set_title(r"$SS^\beta$ against GS", fontsize=17, pad=9)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="lower left", fontsize=13)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_envelope.png")
    plt.close(fig)
    print(f"wrote {OUT}/fig2_envelope.png")


if __name__ == "__main__":
    main()
