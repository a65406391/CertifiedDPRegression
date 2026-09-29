"""The two main-paper figures for the Blobs AGS study. LINEAR model, b = 20 throughout.

Inference is the reference procedure (ss_reference, a port of experiments/common/ags.py):
per-coordinate envelope WIDTH capped at the one-substitution GS, beta_j = beta/n_params,
isotropic a_j = n_params, sigma_j = (2*gamma_c/eps)*ss_j*a_j, Cauchy per coordinate.

  fig1_headline    privacy-utility at b=20: the AGS MILP F-ladder against DP-SGD
                   (matched algorithm, and best single tuned config) and PATE.
  fig2_batchsize   accuracy vs batch size at fixed eps, AGS-IBP vs matched DP-SGD.

Every AGS number is recomputed here from the stored A_grid and the analytic global
sensitivity, so the figures use the SOUND smooth sensitivity (GS tail) regardless of
what the originating run wrote to disk.

--certificate cross_radius swaps the certificate for the paper's cross-radius bound
(ags/blobs/cross_radius.py) and writes every output with a _cross_radius suffix, so the stored
figures and JSON are never overwritten. fig2 is not drawn under it: the batch-size sweep stored
only envelope widths, and the cross-radius needs the boxes. The default is the reference above.
"""

import argparse
import glob
import gzip
import json
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from common import plotting
from ags.blobs import ss_reference as R
from ags.blobs import cross_radius as CR
from ags.blobs.shared import load_blobs
from ags.blobs import baselines as B

plotting.apply()
OUT, RUNS = "ags/blobs/figures", "ags/blobs/runs"
os.makedirs(OUT, exist_ok=True)

EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
KG = [1, 2, 4, 8]
FS = [1, 2, 5, 10]
# AGS: one blue per rung, light (loosest bound) to dark (tightest)
from common.plotting import BLUES, C_DPSGD as C_DP, C_NOMINAL as C_NOM, C_PATE

N, d, ANISO, SEP, B_MAIN, EPOCHS, GAM, LR, NP_ = 200, 8, 5.0, 4.0, 20, 20, 0.05, 0.5, 9
# Which certificate the release uses, set from --certificate in __main__. "width" is the reference.
CERT, SUFFIX = "width", ""
J = lambda p: json.load(open(f"{RUNS}/{p}"))


def setup():
    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    P, Pte = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / N))
    return P, Pte, ys, yts, lr


def ags_curve(theta_n, ls_by_k, k_subset, n_s, b, seeds=41, boxes=None):
    """Reference inference. ls_by_k must hold per-coordinate min(width, GS*k).

    Under CERT == "cross_radius" the staircase runs on Abar(k) from `boxes` (k -> (param_l, param_u))
    instead; everything after the smooth sensitivity is the same reference code.
    """
    gs_vec = R.gs_window(LR * n_s, GAM, b, NP_)
    out = []
    for eps in EPS:
        if CERT == "cross_radius":
            ss = CR.smooth_sensitivity(boxes, theta_n, gs_vec, R.beta_for(eps), NP_)
        else:
            ss = R.smooth_sensitivity(ls_by_k, k_subset, gs_vec, R.beta_for(eps), NP_)
        sg = R.sigma(ss, eps, NP_, noise_mode="isotropic")
        out.append(
            float(
                np.median(
                    [B.acc(R.release(theta_n, sg, np.random.RandomState(s)), PTE, YTS) for s in range(seeds)]
                )
            )
        )
    return out, gs_vec[0]


def milp_bounds(F, b=20, n_s=200):
    """Per-coordinate local sensitivities for b=20 at this F, tightest available.

    k=1,2 solved to optimality on the first pass; k=4,8 re-solved at 600 s (tag a5tl);
    k=3 solved separately (tag a5k3). Prefer whichever gives the smaller envelope --
    a looser bound is always sound, so min() over sources is safe.
    """
    gs_vec = R.gs_window(LR * n_s, GAM, b, NP_)
    runs = [f"a5_b20_ns200_F{F}_MILP", f"a5tl_linear_b20_ns200_F{F}_MILP", f"a5k3_linear_b20_ns200_F{F}_MILP"]
    ls, boxes, th, src = {}, {}, None, {}
    for rid in runs:
        for kf in sorted(glob.glob(f"{RUNS}/certified/{rid}/steps/k*_w*.npz")):
            k = int(re.match(r"k(\d+)_w(\d+)", os.path.basename(kf)).group(1))
            w = sorted(glob.glob(f"{RUNS}/certified/{rid}/steps/k{k}_w*.npz"))
            if len(w) < n_s // F:  # incomplete sweep: not usable
                continue
            z = np.load(w[-1])
            cand = R.local_sens(z["param_l"].ravel(), z["param_u"].ravel(), gs_vec, k)
            if k not in ls or cand.mean() < ls[k].mean():
                ls[k], src[k] = cand, rid.split("_")[0]
                boxes[k] = (z["param_l"].ravel(), z["param_u"].ravel())
            if k == 1 and th is None:
                th = z["param_n"].ravel()
    return ls, sorted(ls), th, src, boxes


def theta_of(path):
    return np.load(path)["param_n"].ravel()


def tl_rate(run_id):
    """TimeLimit hits / total solves. The serializer gzips solves.jsonl when a run
    finishes, so accept either name rather than silently returning None."""
    base = f"{RUNS}/certified/{run_id}/solves.jsonl"
    if os.path.exists(base + ".gz"):
        opener, p = lambda q: gzip.open(q, "rt"), base + ".gz"
    elif os.path.exists(base):
        opener, p = open, base
    else:
        return None
    n = tl = 0
    for l in opener(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        n += 1
        tl += r.get("status") in (9, "TIME_LIMIT", "time_limit")
    return tl, n


def dpsgd_tuned():
    """Best SINGLE configuration over the 90-point grid, chosen by mean across eps.

    Selecting per eps would let the baseline see the privacy budget before choosing
    its hyperparameters, which is not a comparison anyone can make in practice.
    """
    rows = J("dpsgd_tuned_a5.json")["rows"]
    sc = lambda r: np.mean([r["acc"][str(e)] for e in EPS if str(e) in r["acc"]])
    best = max(rows, key=sc)
    return [best["acc"][str(e)] for e in EPS], best


def pate_curve():
    """One PATE configuration held fixed across eps, chosen under the same rule."""
    g = J("pate_grid_full.json")
    idx = [g["eps"].index(e) for e in EPS]
    key = max(g["grid"], key=lambda k: np.mean([g["grid"][k][i] for i in idx]))
    m, q = key.split("_")
    env = [max(v[i] for v in g["grid"].values()) for i in idx]
    return [g["grid"][key][i] for i in idx], f"m={m}, q={q}", env


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png", flush=True)


def fig1():
    n_s = EPOCHS * (N // B_MAIN)
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    rungs, report = [], []

    hd = J("ibp_sweep_ref.json")
    cell = next(c for c in hd["cells"] if c["b"] == B_MAIN)  # non-private reference only
    for F in FS:
        ls, ks, th, src, boxes = milp_bounds(F)
        y, gs = ags_curve(th, ls, ks, n_s, B_MAIN, boxes=boxes)
        if CERT == "cross_radius":
            print(f"\n  F={F}\n" + CR.report(boxes, th, R.gs_window(LR * n_s, GAM, B_MAIN, NP_)))
        rungs.append((f"AGS (MILP, $F$={F})", y, ls[1].mean()))
        report.append((f"F={F}", ls[1].mean(), gs, {**src, "k": ks}))

    for (lab, y, _), c, mk in zip(rungs, BLUES[1:], ["o", "o", "o", "o"]):
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

    dp_m = [r["dpsgd"] for r in cell["rows"]]
    ax.plot(
        EPS,
        dp_m,
        color=C_DP,
        marker="s",
        ls="-",
        zorder=5,
        label="DP-SGD (matched)",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    dp_t, cfg = dpsgd_tuned()
    # The tuned arm is still computed, printed in the table below and written to
    # pure_headline.json, but it is no longer drawn: a per-epsilon hyperparameter search
    # is not an arm AGS can match, and on the figure it read as a comparable baseline.
    pa, pcfg, penv = pate_curve()
    ax.plot(
        EPS,
        pa,
        color=C_PATE,
        marker="^",
        ls="-",
        zorder=5,
        label="PATE",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )

    # IBP, F=1 and F=2 all clamp to the one-substitution GS and plot on top of one
    # another; say so rather than letting the reader assume three distinct curves.
    ax.axhline(cell["nominal"], color=C_NOM, ls=":", lw=2.2, zorder=2)
    ax.text(1.05, cell["nominal"] + 0.006, "non-private", ha="left", va="bottom", fontsize=14, color=C_NOM)
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(r"Pure $(\varepsilon, 0)$-DP", fontsize=20, pad=10)
    ax.set_xlim(0.9, 110)
    ax.set_ylim(0.44, 0.96)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # eight entries will cover data wherever they sit inside the axes
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.155),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.9,
        handletextpad=0.5,
        labelspacing=0.4,
        fontsize=13.5,
        frameon=False,
    )
    save(fig, "fig1_headline" + SUFFIX)
    json.dump(
        dict(
            eps=EPS,
            ags={lab: y for lab, y, _ in rungs},
            dpsgd_matched=dp_m,
            dpsgd_tuned=dp_t,
            dpsgd_tuned_cfg=[cfg["b"], cfg["epochs"], cfg["lr"]],
            pate=pa,
            pate_cfg=pcfg,
            nominal=cell["nominal"],
        ),
        open(f"{RUNS}/pure_headline{SUFFIX}.json", "w"),
        indent=1,
    )

    print(f"\n  b={B_MAIN}  n_s={n_s}  nominal={cell['nominal']:.4f}")
    print(f"  {'rung':10s} {'w(k=1)':>10s} {'GS':>9s}  acc@eps=" + " ".join(f"{e:>7}" for e in EPS))
    for (lab, y, a1), (_, _, gs, src) in zip(rungs, report):
        s = "" if not src else "  k=" + str(src.get("k"))
        print(
            f"  {lab.replace('AGS ', ''):10s} {a1:10.4f} {gs:9.3f}  " + " ".join(f"{v:7.4f}" for v in y) + s
        )
    print(f"  {'DP-SGD-m':8s} {'':>10s} {'':>9s}  " + " ".join(f"{v:7.4f}" for v in dp_m))
    print(
        f"  {'DP-SGD-t':8s} {'':>10s} {'':>9s}  "
        + " ".join(f"{v:7.4f}" for v in dp_t)
        + f"  [b={cfg['b']}, n_s={cfg['n_s']}, lr={cfg['lr']}]"
    )
    print(
        f"  {'PATE':8s} {'':>10s} {'':>9s}  "
        + " ".join(f"{v:7.4f}" for v in pa)
        + f"  [{pcfg}, best mean across eps]"
    )
    print(
        f"  {'PATE-env':8s} {'':>10s} {'':>9s}  "
        + " ".join(f"{v:7.4f}" for v in penv)
        + "  [per-eps envelope: an UPPER bound no single config attains]"
    )
    for F in FS:
        r = tl_rate(f"a5_b20_ns200_F{F}_MILP")
        rr = tl_rate(f"a5tl_linear_b20_ns200_F{F}_MILP")
        print(
            f"  TimeLimit F={F:<3} first pass {r[0]}/{r[1]}"
            + (f"   re-run {rr[0]}/{rr[1]}" if rr else "   re-run pending")
        )


def fig2(eps_fix=20):
    if CERT == "cross_radius":
        print("\n  fig2 not drawn under --certificate cross_radius: ibp_sweep_ref.json stores only widths")
        return
    i = EPS.index(eps_fix)
    cells = J("ibp_sweep_ref.json")["cells"]
    pts = []
    for c in cells:
        b = c["b"]
        gsv = R.gs_window(LR * c["n_s"], GAM, b, NP_)
        ls = {int(k): R.local_sens(np.zeros(NP_), np.array(v), gsv, int(k)) for k, v in c["widths"].items()}
        y, _ = ags_curve(np.array(c["theta_n"]), ls, sorted(ls), c["n_s"], b)
        pts.append((b, y[i], c["rows"][i]["dpsgd"], c["nominal"], c["n_s"]))
    pts.sort()
    bs = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot(bs, [p[3] for p in pts], color=C_NOM, ls=":", lw=2.2, zorder=2)
    ax.text(bs[-1], pts[-1][3] + 0.008, "non-private", ha="right", va="bottom", fontsize=14, color=C_NOM)
    ax.margins(y=0.08)
    ax.plot(
        bs,
        [p[1] for p in pts],
        color=BLUES[3],
        marker="o",
        ls="-",
        zorder=6,
        label="AGS (IBP)",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.plot(
        bs,
        [p[2] for p in pts],
        color=C_DP,
        marker="s",
        ls="-",
        zorder=5,
        label="DP-SGD (matched)",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    # the curves cross, so a single-signed fill would misrepresent the comparison
    ax.set_xscale("log")
    ax.set_xlabel("Batch Size  $b$")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(bs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(rf"Pure $(\varepsilon, 0)$-DP,  $\varepsilon = {eps_fix}$", fontsize=19, pad=10)
    ax.legend(loc="lower right", fontsize=13.5)
    save(fig, f"fig2_batchsize_eps{eps_fix}")
    print(f"\n  eps={eps_fix}")
    print(f"  {'b':>5} {'n_s':>6} {'AGS':>8} {'DP-SGD':>8} {'gap':>8} {'nominal':>8}")
    for b, a, s, nm, ns in pts:
        print(f"  {b:>5} {ns:>6} {a:8.4f} {s:8.4f} {a - s:+8.4f} {nm:8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The blobs headline figures.")
    ap.add_argument(
        "--certificate",
        choices=("width", "cross_radius"),
        default="width",
        help="width: the reference envelope width (default); cross_radius: ags/blobs/cross_radius.py",
    )
    a = ap.parse_args()
    CERT, SUFFIX = a.certificate, "_cross_radius" if a.certificate == "cross_radius" else ""
    P, PTE, YS, YTS, LR = setup()
    print("FIG 1  headline\n" + "-" * 60)
    fig1()
    print("\nFIG 2  batch size\n" + "-" * 60)
    for e in (10, 20):
        fig2(e)
