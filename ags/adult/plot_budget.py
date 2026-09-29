"""The per-release budget under composition (fig14): how large each of n releases may be at a fixed total.

For a total (eps, 1e-5) budget and n pure releases, the basic split gives each release eps / n; the
optimal composition theorem (common.accounting.exact_accounting.optimal_pure_budget) gives the largest
eps_0 whose n-fold composition is still (eps, 1e-5)-DP. Left: both against n (dashed basic, solid
optimal), one colour per total budget. Right: their ratio, the factor by which accounting enlarges
each release. Analytic; reads nothing.

    PYTHONPATH=. python -m ags.adult.plot_budget
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from ags.adult.plot_windows import EPS_VIBRANT
from common import plotting
from common.accounting.exact_accounting import optimal_pure_budget

OUT = "ags/adult/figures/fig14_per_release_budget.png"
DELTA = 1e-5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eps", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--n-max-log2", type=int, default=8, dest="n_max")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    ns = [2**j for j in range(a.n_max + 1)]
    figsize = (6.7, 3.6)
    plotting.apply_combined(2, figsize=figsize, page_width_in=plotting.COLUMN_WIDTH_IN)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08)
    handles = {}
    for e, col in zip(a.eps, EPS_VIBRANT):
        basic = [e / n for n in ns]
        opt = [optimal_pure_budget(e, n, DELTA) for n in ns]
        label = rf"$\varepsilon = {e:g}$"
        (h,) = ax1.plot(ns, opt, marker="o", ls="-", color=col, markeredgecolor="white", markeredgewidth=1.1)
        ax1.plot(ns, basic, ls="--", color=col, lw=1.2, alpha=0.8)
        ax2.plot(ns, np.array(opt) / np.array(basic), marker="o", ls="-", color=col, markeredgecolor="white")
        handles.setdefault(label, h)
    handles["optimal composition"] = matplotlib.lines.Line2D([], [], color="0.3", ls="-", marker="o")
    handles["basic composition"] = matplotlib.lines.Line2D([], [], color="0.3", ls="--")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="major", pad=plt.rcParams["xtick.major.pad"] * 2.2)
        ax.set_xlabel("Number of Releases $n$")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"Budget per Release $\varepsilon_0$")
    ax1.set_title(r"Per-release budget at $(\varepsilon, 10^{-5})$", pad=6)
    ax2.axhline(1.0, color="0.35", ls=":", lw=1.0)
    ax2.set_ylabel(r"Gain over $\varepsilon / n$")
    ax2.set_title("Gain from accounting", pad=6)
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.savefig(a.out)
    plt.close(fig)
    print(f"wrote {a.out}  [eps={a.eps}, n up to {ns[-1]}, delta={DELTA:g}]")


if __name__ == "__main__":
    main()
