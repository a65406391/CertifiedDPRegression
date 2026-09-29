"""The blobs release under the proven rules (Notes/AGS_RELEASE_ADMISSIBILITY.md), from the stored envelopes.

The blobs figures (fig4_combined, fig6_combined_cross_radius) release through the reference path
(ss_reference: per-coordinate smoothness beta / P and the reference noise scales). This module re-releases
the same stored MILP envelopes (runs/certified/*_F{1,2,5,10}_MILP, nothing re-certified) with the rules the
contractive figures use (common.vector_release.tight_scales): independent Cauchy coordinates in pure DP
(beta_j = eps / (2P), scale 2 a_j S_j / eps) and independent Laplace coordinates in approximate DP (beta_j
from the exact Gamma tail of the dilation loss, alpha = eps / 2), with the l1 common scale as the headline
and the anisotropic allocation stored beside it. There is no public selection set on blobs, so the
allocation is fixed in advance (l1) rather than chosen. The certificate is the cross-radius capped at the
one-substitution GS (ags/blobs/cross_radius.py), with the GS tail as the row after the last stored radius.
The baselines (matched DP-SGD with basic composition, PATE) are read from the stored cross-radius headlines;
they do not depend on the AGS release. Median test accuracy over `--seeds` noise draws.

Writes runs/tight_release.json and figures/fig7_combined_tight.png.

    PYTHONPATH=. python -m ags.blobs.release_tight            # release and figure
    PYTHONPATH=. python -m ags.blobs.release_tight --plot-only
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from ags.blobs import baselines as B
from ags.blobs import cross_radius as CR
from ags.blobs import ss_reference as R
from ags.blobs.shared import load_blobs
from common import plotting
from common import vector_release as V

OUT, RUNS = "ags/blobs/figures", "ags/blobs/runs"
N, d, ANISO, SEP, B_MAIN, EPOCHS, GAM, LR, NP_ = 200, 8, 5.0, 4.0, 20, 20, 0.05, 0.5, 9
DELTA = 1e-5
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
FS = [1, 2, 5, 10]
ALLOCATIONS = {"l1": "l1", "aniso": "anisotropic"}
STEM = "fig7_combined_tight"


def grid(boxes, theta_n, gs_vec):
    """Cross-radius Abar_j(k) for k = 0 .. k_max capped at GS, plus the GS row at k_max + 1.

    Beyond the last stored radius the certificate is GS; with e^(-beta k) decreasing, the supremum of the
    tail is reached at k_max + 1, so one GS row makes the grid saturated for vector_release."""
    A = CR.certificate_grid(boxes, theta_n, gs_vec)
    return np.vstack([A, np.asarray(gs_vec, float)[None, :]])


def release(seeds):
    from ags.blobs.plot_approx import milp_ls

    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    Pte = np.c_[Xte, np.ones(len(Xte))]
    n_s = EPOCHS * (N // B_MAIN)
    gs_vec = R.gs_window(LR * n_s, GAM, B_MAIN, NP_)
    out = {"eps": EPS, "delta": DELTA, "seeds": seeds, "gs_per_coord": float(gs_vec[0]), "ags": {}}
    for F in FS:
        _, _, theta_n, boxes = milp_ls(F, gs_vec)
        A = grid(boxes, theta_n, gs_vec)
        cell = {"radii": sorted(int(k) for k in boxes), "nominal": B.acc(theta_n, Pte, yts)}
        for mode in ("pure", "approx"):
            for tag, alloc in ALLOCATIONS.items():
                rows = []
                for eps in EPS:
                    sigma, S, info = V.tight_scales(A, eps, mode=mode, allocation=alloc, delta=DELTA)
                    accs = [
                        B.acc(R.release(theta_n, sigma, np.random.RandomState(s), mode=mode), Pte, yts)
                        for s in range(seeds)
                    ]
                    rows.append(
                        dict(
                            eps=eps,
                            test=float(np.median(accs)),
                            sigma_mean=float(np.mean(sigma)),
                            S_mean=float(np.mean(S)),
                            **info,
                        )
                    )
                cell[f"{mode}/{tag}"] = rows
                print(
                    f"F={F:2d} {mode:6s} {tag:5s} " + " ".join(f"{r['test']:.3f}" for r in rows), flush=True
                )
        out["ags"][str(F)] = cell
    json.dump(out, open(f"{RUNS}/tight_release.json", "w"), indent=1)
    print(f"wrote {RUNS}/tight_release.json")
    return out


def plot(T, *, allocation="l1", width=6.7, height=5.0, page_width=plotting.COLUMN_WIDTH_IN, ncol=2):
    pure_b = json.load(open(f"{RUNS}/pure_headline_cross_radius.json"))
    apx_b = json.load(open(f"{RUNS}/approx_headline_v2_cross_radius.json"))
    figsize = (width, height)
    plotting.apply_combined(2, figsize=figsize, page_width_in=page_width)
    from common.plotting import BLUES, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

    blues = BLUES[1:]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    handles = {}
    sq = dict(color=C_DP, marker="s", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    pt = dict(color=C_PATE, marker="^", ls="-", markeredgecolor="white", markeredgewidth=1.1)
    for ax, mode, base, dp_key, title in (
        (axes[0], "pure", pure_b, "dpsgd_matched", r"Pure $(\varepsilon, 0)$-DP"),
        (axes[1], "approx", apx_b, "dpsgd_matched_naive", r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP"),
    ):
        for F, c in zip(FS, blues):
            y = [r["test"] for r in T["ags"][str(F)][f"{mode}/{allocation}"]]
            (h,) = ax.plot(
                T["eps"],
                y,
                color=c,
                marker="o",
                ls="-",
                zorder=6,
                markeredgecolor="white",
                markeredgewidth=1.1,
            )
            handles.setdefault(f"AGS (MILP, $F$={F})", h)
        for lab, y, sty in (("DP-SGD (matched)", base[dp_key], sq), ("PATE", base["pate"], pt)):
            (h,) = ax.plot(base["eps"], y, zorder=5, **sty)
            handles.setdefault(lab, h)
        ax.axhline(pure_b["nominal"], color=C_NOM, ls=":", lw=2.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ax.set_ylim(0.44, 0.99)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, pad=8)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].text(
        1.05,
        pure_b["nominal"] + 0.006,
        "non-private",
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["legend.fontsize"],
        color=C_NOM,
    )
    order = [f"AGS (MILP, $F$={F})" for F in FS] + ["DP-SGD (matched)", "PATE"]
    fig.legend(
        [handles[k] for k in order], order, loc="outside lower center", ncol=ncol, **plotting.LEGEND_BELOW_KW
    )
    stem = STEM if allocation == "l1" else f"{STEM}_{allocation}"
    fig.savefig(f"{OUT}/{stem}.png")
    plt.close(fig)
    print(f"wrote {OUT}/{stem}.png  [allocation={allocation}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=101)
    ap.add_argument("--allocation", choices=tuple(ALLOCATIONS), default="l1")
    a = ap.parse_args()
    T = json.load(open(f"{RUNS}/tight_release.json")) if a.plot_only else release(a.seeds)
    plot(T, allocation=a.allocation)


if __name__ == "__main__":
    main()
