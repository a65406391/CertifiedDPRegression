"""Approximate-DP counterpart of the headline figure. LINEAR model, b = 20.

Every mechanism switches to its (eps, delta) form, per experiments/common/ags.py:

  AGS      beta = min(1, eps / (1 + 2e ln(d/delta)))   (L320)
           sigma_j = (2/eps) * ss_j * a_j              (L537)
           Laplace(0, sigma_j) per coordinate          (L549)
  DP-SGD   three variants, all on the SAME fixed cyclic batching AGS certifies:
             (matched, naive) basic composition, Delta_2 = sqrt(d)*2*gamma/b  (L757-770)
             (matched, RDP)   Renyi composition, sample_rate=1 -- no amplification claim
             (tuned,   naive) best single config over the 90-point grid, basic comp.
           Poisson-sampled DP-SGD is deliberately NOT shown: amplification needs a
           sampling scheme we do not use, so it is not a matched algorithm.
  PATE     Gaussian on the vote counts, l2 sensitivity sqrt(2)

--certificate cross_radius swaps the AGS certificate for the cross-radius bound
(ags/blobs/cross_radius.py) and writes the figure and JSON with a _cross_radius suffix; the
default is the reference width and leaves the stored outputs untouched.
"""

import argparse
import glob
import json
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from common import plotting
from ags.blobs.shared import load_blobs
from ags.blobs import ss_reference as R
from ags.blobs import cross_radius as CR
from ags.blobs import baselines as B

plotting.apply()
OUT, RUNS = "ags/blobs/figures", "ags/blobs/runs"
N, d, ANISO, SEP, B_MAIN, EPOCHS, GAM, LR, NP_ = 200, 8, 5.0, 4.0, 20, 20, 0.05, 0.5, 9
DELTA = 1e-5
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
FS = [1, 2, 5, 10]
CERT, SUFFIX = "width", ""  # set from --certificate in __main__
from common.plotting import BLUES, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

MS, QS = (2, 3, 5, 10, 20, 40), (10, 20, 40, 80)
BS, EPOCH_G, LRS = (200, 100, 50, 25, 20, 10), (1, 2, 5, 10, 20), (0.1, 0.25, 0.5)


def milp_ls(F, gs_vec, n_s=200):
    runs = [f"a5_b20_ns200_F{F}_MILP", f"a5tl_linear_b20_ns200_F{F}_MILP", f"a5k3_linear_b20_ns200_F{F}_MILP"]
    ls, boxes, th = {}, {}, None
    for rid in runs:
        for kf in glob.glob(f"{RUNS}/certified/{rid}/steps/k*_w*.npz"):
            k = int(re.match(r"k(\d+)_w", os.path.basename(kf)).group(1))
            w = sorted(glob.glob(f"{RUNS}/certified/{rid}/steps/k{k}_w*.npz"))
            if len(w) < n_s // F:
                continue
            z = np.load(w[-1])
            cand = R.local_sens(z["param_l"].ravel(), z["param_u"].ravel(), gs_vec, k)
            if k not in ls or cand.mean() < ls[k].mean():
                ls[k] = cand
                boxes[k] = (z["param_l"].ravel(), z["param_u"].ravel())
            if k == 1 and th is None:
                th = z["param_n"].ravel()
    return ls, sorted(ls), th, boxes


def ags_curve(theta_n, ls, ks, gs_vec, seeds=41, boxes=None):
    out = []
    for eps in EPS:
        beta = R.beta_for(eps, mode="approx", n_params=NP_, delta=DELTA)
        if CERT == "cross_radius":
            ss = CR.smooth_sensitivity(boxes, theta_n, gs_vec, beta, NP_)
        else:
            ss = R.smooth_sensitivity(ls, ks, gs_vec, beta, NP_)
        sg = R.sigma(ss, eps, NP_, noise_mode="isotropic", mode="approx")
        out.append(
            float(
                np.median(
                    [
                        B.acc(R.release(theta_n, sg, np.random.RandomState(s), mode="approx"), PTE, YTS)
                        for s in range(seeds)
                    ]
                )
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The blobs approximate-DP headline figure.")
    ap.add_argument(
        "--certificate",
        choices=("width", "cross_radius"),
        default="width",
        help="width: the reference envelope width (default); cross_radius: ags/blobs/cross_radius.py",
    )
    a = ap.parse_args()
    CERT, SUFFIX = a.certificate, "_cross_radius" if a.certificate == "cross_radius" else ""
    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    P, PTE, YTS = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))], yts
    n_s = EPOCHS * (N // B_MAIN)
    gs_vec = R.gs_window(LR * n_s, GAM, B_MAIN, NP_)
    print(f"approx DP, delta={DELTA:g}, b={B_MAIN}, n_s={n_s}, GS/coord={gs_vec[0]:g}")
    print(
        f"  beta(eps=10) approx = {R.beta_for(10, mode='approx', n_params=NP_, delta=DELTA):.5f}"
        f"   vs pure {R.beta_for(10):.5f}\n",
        flush=True,
    )

    rungs = []
    ib = json.load(open(f"{RUNS}/ibp_sweep_ref.json"))
    cell = next(c for c in ib["cells"] if c["b"] == B_MAIN)  # non-private reference only
    for F in FS:
        ls, ks, th, boxes = milp_ls(F, gs_vec)
        rungs.append((f"AGS (MILP, $F$={F})", ags_curve(th, ls, ks, gs_vec, boxes=boxes)))
        if CERT == "cross_radius":
            print(f"  F={F}\n" + CR.report(boxes, th, gs_vec), flush=True)
    for lab, y in rungs:
        print(f"  {lab:18s}" + "".join(f"{v:8.4f}" for v in y), flush=True)

    dpm_naive = [
        float(
            np.median(
                [
                    B.acc(
                        B.dpsgd_gaussian(
                            P, ys, e, DELTA, n_s=n_s, lr=LR, gamma=GAM, lam=0.0, seed=s, b=B_MAIN
                        )[0],
                        PTE,
                        YTS,
                    )
                    for s in range(41)
                ]
            )
        )
        for e in EPS
    ]
    print(f"  {'DP-SGD (m, naive)':20s}" + "".join(f"{v:8.4f}" for v in dpm_naive), flush=True)
    dpm_rdp = [
        float(
            np.median(
                [
                    B.acc(
                        B.dpsgd_rdp_fixed(
                            P, ys, e, DELTA, n_s=n_s, lr=LR, gamma=GAM, lam=0.0, b=B_MAIN, seed=s
                        )[0],
                        PTE,
                        YTS,
                    )
                    for s in range(41)
                ]
            )
        )
        for e in EPS
    ]
    print(f"  {'DP-SGD (m, RDP)':20s}" + "".join(f"{v:8.4f}" for v in dpm_rdp), flush=True)

    cached = json.load(open(f"{RUNS}/approx_headline.json"))
    dpt, (bb, bep, blr) = cached["dpsgd_tuned"], cached["dpsgd_tuned_cfg"]
    print(
        f"  {'DP-SGD (t, naive)':20s}"
        + "".join(f"{v:8.4f}" for v in dpt)
        + f"  [b={bb}, ep={bep}, lr={blr}]",
        flush=True,
    )
    pat, (pm, pq) = cached["pate"], cached["pate_cfg"]
    print(f"  {'PATE':20s}" + "".join(f"{v:8.4f}" for v in pat) + f"  [m={pm}, q={pq}]", flush=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    for (lab, y), c, mk in zip(rungs, BLUES[1:], ["o", "o", "o", "o"]):
        ax.plot(
            EPS,
            y,
            color=c,
            marker=mk,
            ls="-",
            label=lab,
            zorder=6,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
    ax.plot(
        EPS,
        dpm_naive,
        color=C_DP,
        marker="s",
        ls="-",
        zorder=5,
        label="DP-SGD (matched, naive)",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    # RDP and tuned arms are still computed, printed and serialised to
    # approx_headline_v2.json, but no longer drawn: RDP is a different accountant on the
    # same trajectory and the tuned arm is a per-epsilon hyperparameter search, so neither
    # is the matched comparison this panel is making.
    ax.plot(
        EPS,
        pat,
        color=C_PATE,
        marker="^",
        ls="-",
        zorder=5,
        label="PATE",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.axhline(cell["nominal"], color=C_NOM, ls=":", lw=2.2, zorder=2)
    ax.text(1.05, cell["nominal"] + 0.006, "non-private", ha="left", va="bottom", fontsize=14, color=C_NOM)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 110)
    ax.set_ylim(0.44, 0.99)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP", fontsize=20, pad=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.165),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.9,
        handletextpad=0.5,
        labelspacing=0.4,
        fontsize=13.5,
        frameon=False,
    )
    fig.savefig(f"{OUT}/fig3_headline_approx{SUFFIX}.png")
    json.dump(
        dict(
            eps=EPS,
            delta=DELTA,
            ags={l: y for l, y in rungs},
            dpsgd_matched_naive=dpm_naive,
            dpsgd_matched_rdp=dpm_rdp,
            dpsgd_tuned_naive=dpt,
            dpsgd_tuned_cfg=[bb, bep, blr],
            pate=pat,
            pate_cfg=[pm, pq],
            nominal=cell["nominal"],
        ),
        open(f"{RUNS}/approx_headline_v2{SUFFIX}.json", "w"),
        indent=1,
    )
    print(f"\n  wrote {OUT}/fig3_headline_approx{SUFFIX}.png")
