"""Three-panel figure for the cross-radius recompute (reads values/recompute_2026-09-14/results.json).

Left   : cross-radius certificate A(x,k) averaged over the 2000 test points (dashed) and the
         resulting SS^beta at beta=1/2 (solid, right axis; the beta of the plotted mechanism at
         eps=1), plus the clipping-ceiling width.
Centre : pure (eps,0)-DP, tighter Cauchy (cor:TighterCauchy) with scale SS^beta/(eps-beta) at the
         public beta=eps/2 -- 90% |error| quantile, the object Thm 4.3 bounds (MSE is undefined
         under Cauchy noise; see README). Def 3.7 curves stay in results.json ("pure").
Right  : approx (eps,1e-5)-DP, Laplace(2 SS/eps) with beta=eps/(2 ln(2/delta)) -- test MSE.
Both trade-off panels show the global-sensitivity baseline at the sound clipping ceiling (grey);
the published GS=10 baseline is kept in results.json only.

    PYTHONPATH=$PWD python -m agtr.linear.plot
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import plotting
from common.plotting import METHOD_COLORS, METHOD_LABELS

OUT = "agtr/linear/values/recompute_2026-09-14"
FIG = "agtr/linear/figures/recompute_2026-09-14"
METHODS = ["ibp", "f1", "f2", "f4", "f8", "f16"]


def main():
    os.makedirs(FIG, exist_ok=True)
    R = json.load(open(f"{OUT}/results.json"))
    eps = R["epsilons"]
    plotting.apply_combined(3, legend_pt=7.0)
    fig, axes = plt.subplots(1, 3, figsize=plotting.combined_figsize(3), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.20, w_pad=0.06)

    # --- left: certificate profile and SS at eps=1 (beta = 1/6)
    ax = axes[0]
    ax2 = ax.twinx()
    ks = np.arange(0, 21)
    for m in METHODS:
        A = np.array(R["methods"][m]["A_mean_over_test_by_k"])
        c = METHOD_COLORS[m]
        ax.step(
            ks,
            A,
            where="post",
            color=c,
            ls="--",
            lw=plt.rcParams["lines.linewidth"],
            alpha=0.7,
            label=METHOD_LABELS[m],
        )
        t = np.linspace(0, 26, 600)
        beta = 0.5
        Ai = A[np.clip(t.astype(int), 0, 20)]
        Ai = np.where(t > 20, R["ceiling_gs_mean"], Ai)
        s = Ai * np.exp(-beta * t)
        ax2.plot(t, s, "-", color=c, lw=plt.rcParams["lines.linewidth"] + 0.4, alpha=0.9)
        j = int(np.argmax(s))
        ax2.plot(
            t[j],
            s[j],
            "*",
            color=c,
            ms=plt.rcParams["lines.markersize"] * 2.4,
            markeredgecolor="black",
            zorder=10,
        )
    ax.axhline(
        R["ceiling_gs_mean"],
        color="black",
        ls=":",
        lw=2.0,
        zorder=5,
        label=f"Clipping ceiling ({R['ceiling_gs_mean']:.2f})",
    )
    ax.set_xlabel(r"Substitutions $k$")
    ax.set_ylabel(r"$\bar{A}(x,k)$ cross-radius (dashed)")
    ax2.set_ylabel(r"$SS^{\beta}$ at $\beta=1/2$ (solid)")
    ax.set_title("Certificate")
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # --- centre / right
    for ax, mode, metric, title, ylab in (
        (
            axes[1],
            "pure_liang_b2",
            "mae",
            r"Pure $(\epsilon,0)$-DP, Cauchy $\beta=\epsilon/2$",
            "Mean Abs. Error",
        ),
        (axes[2], "approx", "mse", r"Approx. $(\epsilon,\delta=10^{-5})$-DP", "Test MSE"),
    ):
        gs = R["gs_baselines"]["pure" if mode.startswith("pure") else mode]
        ax.plot(
            eps,
            gs["GS=clipping ceiling (per x)"][metric],
            "o-",
            color="black",
            alpha=0.85,
            zorder=20,
            label="Global sensitivity",
            **plotting.SERIES,
        )
        for m in METHODS:
            r = R["methods"][m][mode]
            ax.plot(
                eps,
                r[metric],
                "o-",
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                zorder=6,
                **plotting.SERIES,
            )
            if metric == "mse":
                ax.fill_between(eps, r["mse_p25"], r["mse_p75"], color=METHOD_COLORS[m], alpha=0.1)
        if mode == "approx":
            ax.axhline(
                R["nominal_mse"],
                color="black",
                ls="--",
                lw=1.2,
                alpha=0.7,
                label=f"Nominal MSE ({R['nominal_mse']:.3f})",
            )
        ax.set(xscale="log", yscale="log", xlabel=r"Privacy Budget $\epsilon$", ylabel=ylab, title=title)
        ax.grid(True, alpha=0.3)
    plotting.shared_legend(
        fig, axes, ncol=6, loc="outside upper center", columnspacing=0.8, handlelength=1.5, handletextpad=0.4
    )
    fig.savefig(f"{FIG}/lin_cross_radius.png")
    plt.close(fig)
    print(f"wrote {FIG}/lin_cross_radius.png")


if __name__ == "__main__":
    main()
