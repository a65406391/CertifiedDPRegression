"""The sampling-frequency ablation figures: what releasing n times along one trajectory costs, from
runs/<task>/windows.json (written by run_contractive windows). Computes a class proportion and, for the
accounting variant, the optimal composition of the n pure releases; nothing else.

Left (both variants): final test accuracy (tuned threshold; median over seeds, the seeds as faint
markers) against the number of releases n, one line per total budget eps, with the nominal trajectory's
accuracy as a band and the majority-class rate as a dashed line: a released head at that line carries
no information beyond the class prior.

Right, `--variant auc` (fig11): the released head's AUC against n, with the nominal AUC as a band and
chance at 0.5. Right, `--variant accounting` (fig12): the smallest delta at which the chain of n
(eps / n, 0)-DP releases is (eps', delta)-DP, against eps', one curve per n, by the optimal composition
theorem for pure DP (common.accounting.exact_accounting.composed_pure_delta): every curve reaches 0 at
eps' = eps (basic composition), and the more releases, the further below eps the chain is already
(eps', 1e-5)-DP.

Variant `modes` (fig13): pure DP (left) and approximate DP (right), accuracy against n, one line per eps.

    PYTHONPATH=. python -m ags.adult.plot_windows [--task adult] [--variant auc|accounting|modes]
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from ags.adult.run_contractive import TASKS
from ags.adult.shared import raw
from common import plotting
from common.accounting.exact_accounting import composed_pure_delta

EPS_SHADES = ["#c6dbef", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
DELTA_REF = 1e-5
MODE_STYLES = (("pure", "-", "s"), ("approx", "--", "o"))  # Cauchy chains solid, Laplace chains dashed


def mode_of(r):
    return r.get("mode", "pure")


def majority_rate(C):
    """The test split's majority-class rate: the same seeded permutation as shared.load_task."""
    _, _, y = raw()
    te = np.random.RandomState(C["seed"] + 11).permutation(len(y))[: C["n_test"]]
    return float(max(np.mean(y[te] == c) for c in np.unique(y)))


def n_shades(ns):
    cmap = plt.get_cmap("viridis")
    return [cmap(0.05 + 0.85 * i / max(len(ns) - 1, 1)) for i in range(len(ns))]


def accuracy_panel(ax, runs, ns, eps_all, nominal, majority):
    shades = (
        EPS_SHADES[-len(eps_all) :] if len(eps_all) <= len(EPS_SHADES) else [EPS_SHADES[-1]] * len(eps_all)
    )

    def cell(n, e, m):
        return [r["final"]["test"] for r in runs if r["n"] == n and r["eps"] == e and mode_of(r) == m]

    for m, ls, mk in MODE_STYLES:
        for e, col in zip(eps_all, shades):
            xs = [n for n in ns if cell(n, e, m)]
            if not xs:
                continue
            ax.plot(
                xs,
                [float(np.median(cell(n, e, m))) for n in xs],
                marker=mk,
                ls=ls,
                color=col,
                label=rf"$\varepsilon$ = {e}" + ("" if m == "pure" else f" ({m})"),
                zorder=5,
            )
            for n in xs:
                ys = cell(n, e, m)
                if len(ys) > 1:
                    ax.plot([n] * len(ys), ys, ls="none", marker=mk, ms=3, color=col, alpha=0.35, zorder=4)
    if np.isfinite(nominal):
        ax.axhline(nominal, color=plotting.C_NOMINAL, ls="-", lw=6, alpha=0.28, zorder=1, label="non-private")
    ax.axhline(majority, color="0.35", ls="--", lw=1.0, zorder=2, label="majority class")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of Releases $n$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(r"Accuracy, $\varepsilon / n$ per release", pad=6)


def auc_panel(ax, runs, ns, eps_all, nominal_auc):
    shades = (
        EPS_SHADES[-len(eps_all) :] if len(eps_all) <= len(EPS_SHADES) else [EPS_SHADES[-1]] * len(eps_all)
    )

    def cell(n, e, m):
        return [
            r["final"]["auc"]
            for r in runs
            if r["n"] == n and r["eps"] == e and mode_of(r) == m and "auc" in r["final"]
        ]

    for m, ls, mk in MODE_STYLES:
        for e, col in zip(eps_all, shades):
            xs = [n for n in ns if cell(n, e, m)]
            if not xs:
                continue
            ax.plot(
                xs,
                [float(np.median(cell(n, e, m))) for n in xs],
                marker=mk,
                ls=ls,
                color=col,
                label=rf"$\varepsilon$ = {e}" + ("" if m == "pure" else f" ({m})"),
                zorder=5,
            )
    if np.isfinite(nominal_auc):
        ax.axhline(
            nominal_auc, color=plotting.C_NOMINAL, ls="-", lw=6, alpha=0.28, zorder=1, label="non-private"
        )
    ax.axhline(0.5, color="0.35", ls="--", lw=1.0, zorder=2, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of Releases $n$")
    ax.set_ylabel("Test AUC")
    ax.set_title("AUC of the released head", pad=6)


def accounting_panel(ax, runs, ns, eps_all, delta_floor=1e-9):
    """delta(eps') of the composed chain for each n, solid at the largest total budget, dashed at the others."""
    styles = ["-", "--", ":", "-."]
    for e, ls in zip(sorted(eps_all, reverse=True), styles):
        for n, col in zip(ns, n_shades(ns)):
            if not any(r["n"] == n and r["eps"] == e and mode_of(r) == "pure" for r in runs):
                continue  # the pure chains; an approximate chain adds its delta to the same curve
            eps_w = e / n
            grid = np.linspace(0.0, e, 400)
            d = composed_pure_delta(eps_w, n, grid)
            ax.plot(
                grid,
                np.maximum(d, delta_floor),
                color=col,
                ls=ls,
                lw=1.3,
                label=f"$n$ = {n}" if ls == "-" else None,
            )
    ax.axhline(DELTA_REF, color="0.35", ls="--", lw=1.0, zorder=2)
    ax.text(
        0.02,
        DELTA_REF * 1.6,
        r"$\delta = 10^{-5}$",
        color="0.35",
        fontsize=plt.rcParams["legend.fontsize"],
        transform=ax.get_yaxis_transform(),
    )
    ax.set_yscale("log")
    ax.set_ylim(delta_floor, 1.5)
    ax.set_xlim(0, max(eps_all))
    ax.set_xlabel(r"Privacy Budget $\varepsilon'$")
    ax.set_ylabel(r"Minimum $\delta$")
    ax.set_title("Composed guarantee", pad=6)
    ax.set_xticks(range(0, int(max(eps_all)) + 1, 2))
    return ax.get_legend_handles_labels()


EPS_VIBRANT = ["#ff1f5b", "#009ade", "#00cd6c", "#ffc61e", "#af58ba", "#ff7a00"]  # one per total budget


def modes_figure(runs, ns, eps_all, nominal, majority, stem, *, width=6.7, height=3.6, page_width=None):
    """fig13: test accuracy against the number of releases n, pure DP (left) and approximate DP (right),
    one vibrant line per total budget eps, the extra seeds as faint markers, the non-private band and
    the majority-class line; one legend under both panels."""
    plotting.apply_combined(2, figsize=(width, height), page_width_in=page_width or plotting.COLUMN_WIDTH_IN)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08)
    handles = {}
    for ax, mode, title in (
        (axes[0], "pure", r"Pure $(\varepsilon, 0)$-DP"),
        (axes[1], "approx", r"Approx. $(\varepsilon, 10^{-5})$-DP"),
    ):
        for e, col in zip(eps_all, EPS_VIBRANT):
            cell = lambda n: [
                r["final"]["test"] for r in runs if r["n"] == n and r["eps"] == e and mode_of(r) == mode
            ]
            xs = [n for n in ns if cell(n)]
            (h,) = ax.plot(
                xs,
                [float(np.median(cell(n))) for n in xs],
                marker="o",
                ls="-",
                color=col,
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=1.1,
            )
            handles.setdefault(rf"$\varepsilon = {e}$", h)
            for n in xs:
                if len(cell(n)) > 1:
                    ax.plot(
                        [n] * len(cell(n)),
                        cell(n),
                        ls="none",
                        marker="o",
                        ms=4,
                        color=col,
                        alpha=0.35,
                        zorder=4,
                    )
        band = ax.axhline(nominal, color=plotting.C_NOMINAL, ls="-", lw=6, alpha=0.28, zorder=1)
        maj = ax.axhline(majority, color="0.35", ls="--", lw=1.0, zorder=2)
        handles.setdefault("non-private", band)
        handles.setdefault("majority class", maj)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="major", pad=plt.rcParams["xtick.major.pad"] * 2.2)
        ax.set_xlabel("Number of Releases $n$")
        ax.set_title(title, pad=6)
    axes[0].set_ylabel("Test Accuracy")
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="outside lower center",
        ncol=min(len(handles), 4),  # two rows once the budgets exceed the panel width
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.savefig(f"{stem}.png")
    plt.close(fig)


def accounting_approx_figure(
    runs, eps_list, nominal, majority, stem, delta=1e-5, *, n_max_log2=8, width=10.0, height=4.55, twin=False
):
    """fig15: approximate DP with the window budgets set by optimal composition, n = 2^0 .. 2^8.

    Left: test accuracy of the last release against n, one vibrant colour per total budget, solid with the
    optimal-composition budgets and dashed with the basic split (one shared point where the two coincide). Centre: the budget each window got, optimal (solid,
    eps_0 with delta / 2 as the composition slack, as run_windows sets it) against the basic split eps / n
    (dashed). Right: their ratio. One legend under the three panels.

    twin=True (fig16): the budget and the gain share one panel, the budget on the left y axis and the gain
    on a right y axis (dotted, square markers); the accuracy panel is the same."""
    from common.accounting.exact_accounting import optimal_pure_budget

    plotting.apply_combined(2 if twin else 3, figsize=(width, height))
    fig, axes = plt.subplots(1, 2 if twin else 3, figsize=(width, height), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08)
    gain_ax = axes[1].twinx() if twin else axes[2]
    if twin:
        gain_ax.grid(False)  # the budget axis carries the grid; a second one would double the lines
    handles = {}
    grid = [2**j for j in range(n_max_log2 + 1)]  # fixed, so the budget panels span the whole range
    mk = dict(marker="o", markeredgecolor="white", markeredgewidth=1.1)
    for e, col in zip(eps_list, EPS_VIBRANT):
        opt = [optimal_pure_budget(e, n, delta / 2) for n in grid]
        basic = [e / n for n in grid]

        def acc(n, accounting):
            return [
                r["final"]["test"]
                for r in runs
                if r["n"] == n
                and r["eps"] == e
                and mode_of(r) == "approx"
                and r.get("accounting", "basic") == accounting
            ]

        # where optimal composition gains under 1% the two splits give the same eps_0 (they differ only in
        # delta / n against delta / (2n) per window): one point, from whichever chain was run
        same = {n: o / b < 1.01 for n, o, b in zip(grid, opt, basic)}
        for accounting, ls in (("optimal", "-"), ("basic", "--")):
            other = "basic" if accounting == "optimal" else "optimal"
            ys = {n: acc(n, accounting) or (acc(n, other) if same[n] else []) for n in grid}
            xs = [n for n in grid if ys[n]]
            (h,) = axes[0].plot(
                xs,
                [float(np.median(ys[n])) for n in xs],
                ls=ls,
                color=col,
                # basic on top as open rings around the filled optimal markers: where the two coincide
                # (every n here) a filled dot sits inside a ring; where they differ the two separate
                zorder=5 if accounting == "optimal" else 6,
                **(
                    mk
                    if accounting == "optimal"
                    else dict(
                        marker="o",
                        ms=plt.rcParams["lines.markersize"] * 1.9,
                        markerfacecolor="none",
                        markeredgecolor=col,
                        markeredgewidth=1.3,
                        lw=plt.rcParams["lines.linewidth"] * 0.6,
                    )
                ),
            )
            if accounting == "optimal":
                handles.setdefault(rf"$\varepsilon = {e:g}$", h)
        axes[1].plot(grid, opt, ls="-", color=col, **mk)
        axes[1].plot(grid, basic, ls="--", color=col, lw=1.2, alpha=0.8)
        gain_ax.plot(
            grid,
            np.array(opt) / np.array(basic),
            ls=":" if twin else "-",
            color=col,
            **(dict(marker="s", ms=4, markeredgecolor="white", markeredgewidth=0.8) if twin else mk),
        )
    handles["non-private"] = axes[0].axhline(
        nominal, color=plotting.C_NOMINAL, ls="-", lw=6, alpha=0.28, zorder=1
    )
    handles["majority class"] = axes[0].axhline(majority, color="0.35", ls="--", lw=1.0, zorder=2)
    handles["optimal composition"] = matplotlib.lines.Line2D([], [], color="0.3", ls="-", marker="o")
    handles["basic composition"] = matplotlib.lines.Line2D(
        [], [], color="0.3", ls="--", marker="o", markerfacecolor="none", markeredgewidth=1.3
    )
    if twin:
        handles["gain (right axis)"] = matplotlib.lines.Line2D([], [], color="0.3", ls=":", marker="s", ms=4)
    gain_ax.axhline(1.0, color="0.35", ls=":", lw=1.0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(grid)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in grid])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="major", pad=plt.rcParams["xtick.major.pad"] * 2.2)
        ax.set_xlabel("Number of Releases $n$")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title(r"Approx. $(\varepsilon, 10^{-5})$-DP Accuracy", pad=6)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Budget per Release $\varepsilon_0$")
    axes[1].set_title("Budget per Release and Gain" if twin else "Optimal vs. Basic Composition", pad=6)
    gain_ax.set_ylabel(r"Gain over $\varepsilon / n$")
    if not twin:
        gain_ax.set_title("Gain from Accounting", pad=6)
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="outside lower center",
        ncol=len(handles) if twin else -(-len(handles) // 2),  # fig15: two rows
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.savefig(f"{stem}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=tuple(TASKS), default="adult")
    ap.add_argument(
        "--variant",
        choices=("auc", "accounting", "modes", "accounting_approx", "accounting_twin"),
        default="auc",
    )
    ap.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[5, 10, 20, 50, 100, 150],
        help="accounting_approx: the budgets drawn",
    )
    ap.add_argument("--width", type=float, default=6.7)
    ap.add_argument("--height", type=float, default=3.4)
    ap.add_argument("--page-width", type=float, default=plotting.COLUMN_WIDTH_IN, dest="page_width")
    a = ap.parse_args()
    C = TASKS[a.task]
    Wj = json.load(open(f"{C['runs_dir']}/windows.json"))
    runs = list(Wj["runs"].values())
    ns = sorted({r["n"] for r in runs})
    eps_all = sorted({r["eps"] for r in runs})
    nominal = {
        q: float(
            np.median([r["windows"][-1]["nominal"].get(q, np.nan) for r in runs if r["n"] == 1] or [np.nan])
        )
        for q in ("test", "auc")
    }
    majority = majority_rate(C)

    stem = f"{C['figures_dir']}/{C.get('windows_fig_stem', 'fig11_windows_adult')}"
    if a.variant == "accounting_twin":  # fig16: fig15 with the budget and the gain on one twin-axis panel
        stem = stem.replace("fig11_windows", "fig16_windows_accounting_twin")
        accounting_approx_figure(
            runs,
            a.eps,
            nominal["test"],
            majority,
            stem,
            delta=Wj.get("delta", 1e-5),
            width=7.6,
            height=3.5,
            twin=True,
        )
        print(f"wrote {stem}.png  [task={a.task}, variant=accounting_twin, eps={a.eps}]")
        return
    if a.variant == "accounting_approx":  # fig15: approx DP with optimal-composition window budgets
        stem = stem.replace("fig11_windows", "fig15_windows_accounting_approx")
        accounting_approx_figure(runs, a.eps, nominal["test"], majority, stem, delta=Wj.get("delta", 1e-5))
        print(f"wrote {stem}.png  [task={a.task}, variant=accounting_approx, eps={a.eps}]")
        return
    if a.variant == "modes":  # fig13: pure (left) and approximate (right) DP against n
        stem = stem.replace("fig11_windows", "fig13_windows_modes")
        modes_figure(runs, ns, eps_all, nominal["test"], majority, stem, page_width=a.page_width)
        print(f"wrote {stem}.png  [task={a.task}, variant=modes, n={ns}, eps={eps_all}, {len(runs)} runs]")
        return
    plotting.apply_combined(2, figsize=(a.width, a.height), page_width_in=a.page_width)
    # the legend is a band under both panels in either variant: the lines fill their axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(a.width, a.height + 0.9))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.40, wspace=0.32)
    accuracy_panel(ax1, runs, ns, eps_all, nominal["test"], majority)
    if a.variant == "auc":
        auc_panel(ax2, runs, ns, eps_all, nominal["auc"])
        seen = {}
        for ax in (ax1, ax2):
            for h, lab in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(lab, h)
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            fontsize=plt.rcParams["legend.fontsize"],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=4,
            columnspacing=0.9,
            handlelength=1.5,
            handletextpad=0.5,
        )
    else:
        handles, labels = accounting_panel(ax2, runs, ns, eps_all)
        fig.legend(
            handles,
            labels,
            fontsize=plt.rcParams["legend.fontsize"],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=4,
            columnspacing=1.2,
            handlelength=1.4,
            title=rf"solid: $\varepsilon$ = {max(eps_all)}, dashed: $\varepsilon$ = {min(eps_all)}",
            title_fontsize=plt.rcParams["legend.fontsize"],
        )
        stem = stem.replace("fig11_windows", "fig12_windows_accounting")
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    print(
        f"wrote {stem}.png  [task={a.task}, variant={a.variant}, n={ns}, eps={eps_all}, {len(runs)} runs, "
        f"majority {majority:.3f}, nominal {nominal['test']:.3f} / AUC {nominal['auc']:.3f}]"
    )


if __name__ == "__main__":
    main()
