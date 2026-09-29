"""The paper-setting reproduction, in the blobs/mnist_pca figure style.

Reads runs/reproduce_paper_raw.json (written by run_reproduce_paper.py --no-scale) and
recomputes nothing. The header block is the full parameter table for that run, so the
figure carries its own context; the same table is printed to stdout.

Left panel is the like-for-like comparison the paper's run 2 claims: AGS and Laplace
DP-SGD under pure (eps,0)-DP, same algorithm, same trajectory, same clipping. Right
panel is AGS under approximate DP, with the pure curve repeated faded so the gain from
delta > 0 is visible. There is no matched DP-SGD arm on the right because the
reproduction ran only the pure baseline -- said on the figure rather than left to be
inferred.

    python -m ags.mnist.plot_reproduce_paper
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

    from common.plotting import C_AGS, C_CEILING as C_CH, C_DPSGD as C_DP, C_NOMINAL as C_NOM

    R = json.load(open(f"{RUNS}/reproduce_paper_raw.json"))
    # the approximate-DP baseline is a separate run; the figure works with or without it
    try:
        G = json.load(open(f"{RUNS}/dpsgd_approx_matched.json"))
        assert G["eps"] == R["eps"], "the two runs disagree on the epsilon grid"
    except FileNotFoundError:
        G = None
    S, EPS, NOM = R["setting"], R["eps"], R["nominal"]
    slr, cf, P = S["sum_lr"], S["closed_form"], S["P"]
    ceiling = 2.0 * slr * S["gamma"]

    # --------------------------------------------------------------- parameter table
    TABLE = [
        (
            "Data and model",
            [
                ("dataset", "MNIST, PCA-64 (whiten=True)"),
                ("features", "whitened as-is, no unit-norm rescale (--no-scale)"),
                ("split", f"N = {S['N']} train / 2000 test, seed {S['seed']}"),
                ("PCA fit", "private train split (public-preprocessing assumption)"),
                ("model", f"Linear({S['d']} -> 10),  P = {P}"),
                ("loss", "softmax cross-entropy, substitution adjacency"),
            ],
        ),
        (
            "Optimisation (identical for every arm)",
            [
                ("batch", f"b = {S['b']} (full batch), 1 release window"),
                ("steps", f"T = {S['steps']}, sampling_freq = T"),
                ("lr", f"lr_n = {S['lr']} / (1 + {S['eta']} n)   [batchwise decay]"),
                (
                    "sum lr",
                    f"{slr:.4f}  (constant lr = {S['lr'] * S['steps']:g};"
                    f" decay saves {S['lr'] * S['steps'] / slr:.0f}x)",
                ),
                ("clipping", f"per-element gamma = {S['gamma']},  l2_reg = 0"),
                ("nominal", f"{NOM:.4f} test accuracy (same algorithm, no noise)"),
            ],
        ),
        (
            "Sensitivity and release",
            [
                ("closed form", f"cf = sum_lr*2*gamma/b = {cf:.6g}   (cf*P = {cf * P:.4g})"),
                ("local sens", "min(A^k, cf*k); cf*k binds on every coordinate, so the"),
                ("", "release is set by cf and the IBP envelope never enters"),
                (
                    "soundness",
                    f"L = 0.5 lam_max(XtX/b) = {2 / S['lr_limit_2_over_L']:.4f};"
                    f" lr <= 2/L = {S['lr_limit_2_over_L']:.3f}",
                ),
                (
                    "",
                    f"{'HOLDS' if S['cap_sound'] else 'VIOLATED'}"
                    f" with a {S['lr_limit_2_over_L'] / S['lr']:.0f}x margin",
                ),
                ("ceiling", f"clipping ceiling 2*sum_lr*gamma = {ceiling:.4g}"),
                ("k grid", ", ".join(str(k) for k in R["k_grid"])),
                ("pure", "Cauchy, gamma_c = 2, beta = eps/6,"),
                ("", "sigma_j = (2 gamma_c/eps) P ss_j"),
                ("approx", f"Laplace, delta = {S['delta']:g}, beta = min(1, eps/(1+2e ln(P/delta))),"),
                ("", "sigma_j = (2/eps) P ss_j"),
                ("noise alloc", f"{S['noise_mode']} a_j (equals P here: ss is uniform across j)"),
                ("seeds", "41 release draws per point (median); Laplace 11 seeds (median)"),
                ("", "Gaussian: single seed, on GPU (torch_sgd, ~1e-14 from numpy)" if G else ""),
                ("baselines", "matched DP-SGD, same trajectory, basic comp. at eps/T:"),
                ("", "pure = Laplace; approx = Gaussian at delta/T per step, so the"),
                ("", "composed guarantee is (eps, 1e-05) rather than (eps, T*1e-05)"),
                (
                    "wall",
                    f"{R['wall_s'] / 60:.0f} min (AGS + Laplace, CPU)"
                    + (f" + {G['wall_s'] / 60:.0f} min (Gaussian, GPU)" if G else ""),
                ),
            ],
        ),
    ]

    def print_table():
        w = max(len(k) for _, rows in TABLE for k, _ in rows)
        print("=" * 92)
        print(
            "MNIST PCA-64 -- reproduction of the paper's AGS-vs-Laplace setting (run 2 of run_ags_comparison)"
        )
        print("=" * 92)
        for title, rows in TABLE:
            print(f"\n{title}\n" + "-" * len(title))
            for k, v in rows:
                if k or v:
                    print(f"  {k:<{w}}  {v}")
        print("\nResults")
        print("-------")
        ga = G["dpsgd_gaussian"]["strict"]["acc"] if G else None
        print(
            f"  {'eps':>7} {'AGS pure':>9} {'Laplace':>9} {'delta':>8}   "
            f"{'AGS approx':>10} {'Gaussian':>9} {'delta':>8}"
        )
        for i, e in enumerate(EPS):
            ap, lp = R["ags"]["pure"]["acc"][i], R["laplace_matched"][i]
            aa = R["ags"]["approx"]["acc"][i]
            tail = (f" {ga[i]:>9.4f} {aa - ga[i]:>+8.4f}") if ga else ""
            print(f"  {e:>7} {ap:>9.4f} {lp:>9.4f} {ap - lp:>+8.4f}   {aa:>10.4f}{tail}")
        print("=" * 92)

    print_table()

    # --------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15.0, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.74, 1.0], hspace=0.18, wspace=0.06)

    # header: the same table, three columns of monospaced key/value pairs.
    # Column widths and the font size are derived from the longest rendered line, so
    # nothing can silently overlap when a value changes.
    hax = fig.add_subplot(gs[0, :])
    hax.axis("off")
    hax.set_xlim(0, 1)
    hax.set_ylim(0, 1)

    blocks = []
    for title, rows in TABLE:
        kw = max(len(k) for k, _ in rows)
        body = [f"{k:<{kw}}  {v}" if k else f"{'':<{kw}}  {v}" for k, v in rows if k or v]
        blocks.append((title, body, max(len(l) for l in body + [title])))

    def draw_header():
        """Lay the table out in three columns, sized so they cannot collide.

        Must run AFTER the final subplots_adjust: the text is placed in axes
        fractions, so a later layout change would rescale the spacing without
        rescaling the glyphs.
        """
        GUT = 4  # gutter, in characters
        total_ch = sum(b[2] for b in blocks) + GUT * (len(blocks) - 1)
        _p = hax.get_position()
        hax.set_position([0.03, _p.y0, 0.94, _p.height])  # header spans the page

        # character width is MEASURED at a probe size rather than assumed from the font
        # metrics, then the size is solved so the three columns cannot collide
        _probe = hax.text(0, -5, "M" * 100, fontsize=10.0, family="monospace")
        fig.canvas.draw()
        _rend = fig.canvas.get_renderer()
        _bb = _probe.get_window_extent(_rend).transformed(hax.transAxes.inverted())
        _probe.remove()
        char10 = _bb.width / 100.0  # axes fraction per char at fontsize 10
        fs = min(10.0, 0.995 * 10.0 / (total_ch * char10))
        char = char10 * fs / 10.0

        x = 0.0
        for title, body, wch in blocks:
            hax.text(
                x,
                1.02,
                title,
                fontsize=fs + 3.0,
                fontweight="bold",
                va="top",
                family="monospace",
                color="0.15",
            )
            hax.text(
                x,
                0.90,
                "\n".join(body),
                fontsize=fs,
                va="top",
                family="monospace",
                color="0.25",
                linespacing=1.6,
            )
            x += (wch + GUT) * char
        hax.axhline(1.12, color="0.75", lw=1.2)
        hax.axhline(-0.03, color="0.75", lw=1.2)
        hax.text(
            0.0,
            1.22,
            "MNIST PCA-64, 10 classes  --  reproduction of the paper's AGS vs matched Laplace setting",
            fontsize=15,
            fontweight="bold",
            va="baseline",
            color="0.1",
        )

    axes = [fig.add_subplot(gs[1, 0])]
    axes.append(fig.add_subplot(gs[1, 1], sharey=axes[0]))
    handles = {}

    ags_sty = dict(color=C_AGS, marker="o", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    ags_faint = dict(
        color=C_AGS, marker="o", ls="-", alpha=0.30, lw=2.2, markerfacecolor="none", markersize=6
    )
    sq = dict(color=C_DP, marker="s", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    sq_ref = dict(color=C_DP, marker="s", ls=":", lw=2.3, alpha=0.55, markerfacecolor="none", markersize=6)

    def fmt(v, _):
        return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"

    def panel(ax, series, title):
        for lab, y, sty in series:
            (h,) = ax.plot(EPS, y, zorder=6 if lab.startswith("AGS (") else 4, **sty)
            handles.setdefault(lab, h)
        ax.axhline(NOM, color=C_NOM, ls=":", lw=2.0, zorder=2)
        ax.axhline(0.10, color=C_CH, ls="-.", lw=1.5, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(4.2, 26000)
        ax.set_ylim(0.05, 0.92)
        ax.set_xticks([5, 10, 20, 50, 100, 500, 1000, 5000, 20000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, fontsize=18, pad=9)

    panel(
        axes[0],
        [
            ("AGS (certified)", R["ags"]["pure"]["acc"], ags_sty),
            ("DP-SGD (matched Laplace)", R["laplace_matched"], sq),
        ],
        r"Pure $(\varepsilon, 0)$-DP",
    )
    right = [("AGS (certified)", R["ags"]["approx"]["acc"], ags_sty)]
    if G is None:
        right.append(("AGS (pure, repeated for reference)", R["ags"]["pure"]["acc"], ags_faint))
    else:
        right.append(("DP-SGD (matched Gaussian)", G["dpsgd_gaussian"]["strict"]["acc"], sq))
        right.append(
            (r"DP-SGD (reference $\delta$ accounting)", G["dpsgd_gaussian"]["reference"]["acc"], sq_ref)
        )
    panel(axes[1], right, r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP")

    axes[0].set_ylabel("Test Accuracy")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[0].text(4.6, NOM + 0.008, "non-private", ha="left", va="bottom", fontsize=13, color=C_NOM)
    axes[0].text(23000, 0.107, "chance (10 classes)", ha="right", va="bottom", fontsize=12, color=C_CH)
    if G is None:
        axes[1].text(
            0.97,
            0.06,
            "no matched DP-SGD arm was run at $\\delta>0$",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=11.5,
            color="0.45",
            style="italic",
        )

    order = [
        k
        for k in [
            "AGS (certified)",
            "DP-SGD (matched Laplace)",
            "DP-SGD (matched Gaussian)",
            r"DP-SGD (reference $\delta$ accounting)",
            "AGS (pure, repeated for reference)",
        ]
        if k in handles
    ]
    fig.subplots_adjust(bottom=0.135)
    draw_header()
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(order),
        columnspacing=1.6,
        handlelength=2.1,
        handletextpad=0.55,
        fontsize=13.5,
        frameon=False,
    )

    fig.savefig(f"{OUT}/fig3_reproduce_paper.png")
    plt.close(fig)
    print(f"\nwrote {OUT}/fig3_reproduce_paper.png")


if __name__ == "__main__":
    main()
