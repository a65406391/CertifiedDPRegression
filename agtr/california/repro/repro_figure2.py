"""Reproduce the California Housing row of the paper's Figure 2 from stored arrays.

Stage S0. Nothing is re-certified here: this reads
``agtr/california/values/ablation/`` and rebuilds the three published panels
exactly as ``experiments/california/run_ablation.py`` did, so that the plotting
and release path is pinned before any bound is recomputed.

Two conventions are carried side by side.

``V_k`` is the RADIUS about the nominal prediction,
``max(|y_l - y_nom|, |y_u - y_nom|)``, as in
``archive/attic/original_scripts/run_ca_ablation.py:323`` and as the paper's Definition
of ``A^k`` requires.  ``experiments/common/sensitivity.py:57`` instead returns
the WIDTH ``|y_u - y_l|``, which contradicts its own docstring (lines 20-24) and
inflates the sensitivity by 1.4-1.8x here.  The radius reproduces the stored
arrays to 7 significant figures; the width does not.

The smooth sensitivity is likewise computed two ways: the published
aggregation, which evaluates ``V_k e^{-beta k}`` at the computed k and is exact
when the k grid holds every integer up to max(k); and the staircase form used
in ``AGS_Ablation/blobs/ss_reference.py``, which prefills every k with GS and
then takes the smallest sound value at each integer k.  The California grid is
contiguous only to k=10, so the two differ slightly at large batch size where
the maximiser sits near the sparse tail.
"""

import os

import numpy as np

from common import plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "agtr/california/values/ablation"
OUT = "agtr/california/figures/repro"
BATCH_SIZES = [250, 500, 1000, 2000, 4000]
plotting.apply()

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
BETA = 1.0 / 6.0
Y_MIN, Y_MAX = -5.0, 5.0
GS_VAL = Y_MAX - Y_MIN
EPSILONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_TRIALS = 500
DELTA = 1e-5


def ss_published(V, ks):
    """experiments/common/sensitivity.py:compute_smooth_sensitivity.

    max over the *computed* k of V_k e^{-beta k}, plus the GS tail term.
    Exact when the k grid contains every integer up to max(k).
    """
    tail = GS_VAL * np.exp(-BETA * (max(ks) + 1))
    cand = [V[:, j] * np.exp(-BETA * k) for j, k in enumerate(ks) if k != 0]
    return np.maximum(np.max(np.stack(cand), axis=0), tail)


def ss_staircase(V, ks):
    """GS prefill, then the smallest sound value at every integer k.

    A^k is non-decreasing in k, so any computed k_j >= k upper-bounds A^k, as
    does GS.  Evaluating e^{-beta k} against that staircase covers the integers
    between grid points, which matters only where the grid has gaps.
    """
    computed = [k for k in ks if k != 0]
    idx = {k: list(ks).index(k) for k in computed}
    kmax = max(computed)
    best = np.full(V.shape[0], GS_VAL * np.exp(-BETA * (kmax + 1)))
    for k in range(1, kmax + 1):
        sound = np.full(V.shape[0], GS_VAL)
        for kj in computed:
            if kj >= k:
                sound = np.minimum(sound, V[:, idx[kj]])
        best = np.maximum(best, np.exp(-BETA * k) * sound)
    return best


def vk_radius(yl, yu, nominal):
    """A^k as the paper defines it: radius about the nominal prediction.

    run_ca_ablation.py:323 -- v_k = max(|y_l - y_nom|, |y_u - y_nom|)
    """
    return np.maximum(np.abs(yl - nominal[:, None]), np.abs(yu - nominal[:, None]))


def vk_width(yl, yu, nominal):
    """What experiments/common/sensitivity.py:57 returns: the envelope width."""
    return np.abs(yu - yl)


def load():
    ks = list(np.load(f"{DATA}/k_values.npy"))
    y_std = float(np.load(f"{DATA}/y_std.npy"))
    y_mean = float(np.load(f"{DATA}/y_mean.npy"))
    y_test = np.load(f"{DATA}/y_test.npy")
    out = {}
    for bs in BATCH_SIZES:
        yl = np.load(f"{DATA}/bounds_bs{bs}_yl.npy")
        yu = np.load(f"{DATA}/bounds_bs{bs}_yu.npy")
        preds = np.load(f"{DATA}/nominal_preds_bs{bs}.npy")
        out[bs] = dict(
            V=vk_radius(yl, yu, preds),
            V_width=vk_width(yl, yu, preds),
            stored_S=np.load(f"{DATA}/sensitivity_bs{bs}.npy"),
            nom_mse=float(np.mean((preds * y_std + y_mean - (y_test * y_std + y_mean)) ** 2)),
            stair_ks=np.load(f"{DATA}/staircase_bs{bs}_ks.npy"),
            stair_vs=np.load(f"{DATA}/staircase_bs{bs}_vs.npy"),
        )
    return ks, y_std, y_test, out


def panel_staircase(ks, data, y_std, ax=None):
    """Draw into `ax` when given; otherwise own the whole figure and save it."""
    own = ax is None
    fig, ax1 = plt.subplots(figsize=plotting.FIGSIZE) if own else (ax.figure, ax)
    ax2 = ax1.twinx()
    t = np.linspace(0, max(ks) + 5, 500)
    for i, bs in enumerate(BATCH_SIZES):
        kb, vb = list(data[bs]["stair_ks"]), list(data[bs]["stair_vs"])
        c = COLORS[i]
        ax1.step(kb, vb, where="post", color=c, linestyle="--", lw=4, alpha=0.75, label=f"BS={bs}")
        j = np.clip(np.searchsorted(kb, t, side="right") - 1, 0, len(kb) - 1)
        s = np.array([vb[x] for x in j]) * np.exp(-BETA * t)
        ax2.plot(t, s, "-", color=c, lw=5, alpha=1.0)
        m = int(np.argmax(s))
        ax2.plot(t[m], s[m], "*", color=c, ms=15, markeredgecolor="black")
    ax1.set_xlabel("Privacy Parameter k")
    ax1.set_ylabel(r"Max. Local Sens. Certificate $A^{(k)}$ (dashed)")
    ax2.set_ylabel(r"Smooth Sensitivity $SS(f, x)$ (solid)")
    ax1.set_title("California Housing")
    ax1.grid(alpha=0.3)
    if own:
        ax1.legend(loc="upper right", title="Batch Size")
        fig.tight_layout()
        fig.savefig(f"{OUT}/repro_staircase.png")
        plt.close(fig)
    return {
        bs: float(
            np.max(
                np.array(
                    [
                        data[bs]["stair_vs"][x]
                        for x in np.clip(
                            np.searchsorted(list(data[bs]["stair_ks"]), t, side="right") - 1,
                            0,
                            len(data[bs]["stair_vs"]) - 1,
                        )
                    ]
                )
                * np.exp(-BETA * t)
            )
        )
        for bs in BATCH_SIZES
    }


def panel_pure(data, S_by_bs, y_std, n_test, rng, ax=None):
    own = ax is None
    fig, ax = plt.subplots(figsize=plotting.FIGSIZE) if own else (ax.figure, ax)
    gs_real = GS_VAL * y_std
    med = [
        np.median([np.mean(rng.standard_cauchy(n_test) ** 2 * (gs_real / e) ** 2) for _ in range(N_TRIALS)])
        for e in EPSILONS
    ]
    ax.plot(EPSILONS, med, "o--", color="gray", alpha=0.6, label="Global Sensitivity", **plotting.SERIES)
    curves = {}
    for i, bs in enumerate(BATCH_SIZES):
        S = S_by_bs[bs]
        m = [
            np.median(
                [
                    data[bs]["nom_mse"] + np.mean(rng.standard_cauchy(n_test) ** 2 * (S / e) ** 2)
                    for _ in range(N_TRIALS)
                ]
            )
            for e in EPSILONS
        ]
        ax.plot(EPSILONS, m, "o-", color=COLORS[i], label=f"BS={bs}", zorder=6, **plotting.SERIES)
        curves[bs] = m
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Privacy Budget $\epsilon$",
        ylabel="Test Median Squared Error (Pure DP)",
        title="California Housing",
    )
    ax.grid(alpha=0.3)
    if own:
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT}/repro_tradeoff_pure.png")
        plt.close(fig)
    return curves, med


def panel_approx(data, S_by_bs, y_std, n_test, rng, ax=None):
    own = ax is None
    fig, ax = plt.subplots(figsize=plotting.FIGSIZE) if own else (ax.figure, ax)
    factor = np.sqrt(2 * np.log(1 / DELTA))
    gs_real = GS_VAL * y_std
    gsv = [(gs_real * factor / e) ** 2 for e in EPSILONS]
    ax.plot(EPSILONS, gsv, "o--", color="gray", alpha=0.6, label="Global Sensitivity", **plotting.SERIES)
    curves = {}
    for i, bs in enumerate(BATCH_SIZES):
        S = S_by_bs[bs]
        m = [
            data[bs]["nom_mse"]
            + np.mean([np.mean(rng.normal(0, S * factor / e, n_test) ** 2) for _ in range(N_TRIALS)])
            for e in EPSILONS
        ]
        ax.plot(EPSILONS, m, "o-", color=COLORS[i], label=f"BS={bs}", zorder=6, **plotting.SERIES)
        curves[bs] = m
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Privacy Budget $\epsilon$",
        ylabel="Test MSE (Approx DP)",
        title="California Housing",
    )
    ax.grid(alpha=0.3)
    if own:
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT}/repro_tradeoff_approx.png")
        plt.close(fig)
    return curves, gsv


def combined(ks, data, S_by_bs, y_std, y_test, rng):
    """[staircase | pure | approx] in one row, one shared legend, single-figure footprint."""
    plotting.apply_combined(3)
    fig, axes = plt.subplots(1, 3, figsize=plotting.combined_figsize(3), layout="constrained")
    # constrained layout packs the legend right up against the x-labels by default;
    # h_pad opens the gap so the row does not read as one crowded block
    fig.get_layout_engine().set(h_pad=0.20, w_pad=0.06)
    panel_staircase(ks, data, y_std, ax=axes[0])
    panel_pure(data, S_by_bs, y_std, len(y_test), rng, ax=axes[1])
    panel_approx(data, S_by_bs, y_std, len(y_test), rng, ax=axes[2])
    for ax, t in zip(axes, ("Staircase", r"Pure $(\epsilon, 0)$-DP", r"Approx. $(\epsilon, \delta)$-DP")):
        ax.set_title(t)
    # Axis names have to shorten with the panels: the staircase's right-hand label sits
    # directly against the next panel's left-hand label at this width.
    twin = next(
        a for a in fig.axes if a is not axes[0] and a.get_position().bounds == axes[0].get_position().bounds
    )
    axes[0].set_ylabel(r"$A^{(k)}$ (dashed)")
    twin.set_ylabel(r"$SS^\beta$ (solid)")
    axes[0].set_xlabel(r"Substitutions $k$")
    axes[1].set_ylabel("Test MSE")
    axes[2].set_ylabel("Test MSE")
    plotting.shared_legend(fig, axes, ncol=6)
    fig.savefig(f"{OUT}/repro_combined.png")
    plt.close(fig)
    plotting.apply()
    print(f"wrote {OUT}/repro_combined.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    ks, y_std, y_test, data = load()
    n_test = len(y_test)
    rng = np.random.default_rng(0)

    print(f"k grid: {ks}")
    print(f"y_std = {y_std:.6f}   GS (real units) = {GS_VAL * y_std:.6f}   n_test = {n_test}\n")

    print(
        f"{'BS':>6} {'stored S':>11} {'radius':>11} {'stored/rad':>11} "
        f"{'staircase':>11} {'stair/rad':>10} {'width':>11} {'width/rad':>10} {'nom MSE':>9}"
    )
    S_pub, S_stair = {}, {}
    for bs in BATCH_SIZES:
        a = ss_published(data[bs]["V"], ks) * y_std
        b = ss_staircase(data[bs]["V"], ks) * y_std
        w = ss_published(data[bs]["V_width"], ks) * y_std
        S_pub[bs], S_stair[bs] = a.mean(), b.mean()
        stored = data[bs]["stored_S"].mean()
        print(
            f"{bs:>6} {stored:>11.6f} {a.mean():>11.6f} {stored / a.mean():>11.6f} "
            f"{b.mean():>11.6f} {b.mean() / a.mean():>10.4f} {w.mean():>11.6f} "
            f"{w.mean() / a.mean():>10.4f} {data[bs]['nom_mse']:>9.5f}"
        )

    peaks = panel_staircase(ks, data, y_std)
    print("\nstaircase-panel SS peak at the single plotted test point (right axis):")
    for bs in BATCH_SIZES:
        print(f"   BS={bs:<5} peak = {peaks[bs]:.4f}")

    pure, pure_gs = panel_pure(data, S_pub, y_std, n_test, rng)
    approx, approx_gs = panel_approx(data, S_pub, y_std, n_test, rng)
    print(
        f"\npure-DP  MSE at eps=0.1 : GS={pure_gs[0]:.3e}  "
        + "  ".join(f"BS{b}={pure[b][0]:.3e}" for b in BATCH_SIZES)
    )
    print(
        f"pure-DP  MSE at eps=10  : GS={pure_gs[-1]:.3e}  "
        + "  ".join(f"BS{b}={pure[b][-1]:.3e}" for b in BATCH_SIZES)
    )
    print(
        f"approx   MSE at eps=0.1 : GS={approx_gs[0]:.3e}  "
        + "  ".join(f"BS{b}={approx[b][0]:.3e}" for b in BATCH_SIZES)
    )
    print(
        f"approx   MSE at eps=10  : GS={approx_gs[-1]:.3e}  "
        + "  ".join(f"BS{b}={approx[b][-1]:.3e}" for b in BATCH_SIZES)
    )
    combined(ks, data, S_pub, y_std, y_test, np.random.default_rng(0))
    print(f"\nfigures written to {OUT}/")


if __name__ == "__main__":
    main()
