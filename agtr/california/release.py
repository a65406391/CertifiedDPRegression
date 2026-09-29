"""Cross-radius release for the genuine-AGT California run (reads values/agt_2026-09-14/<variant>/).

Per test point x and every integer k:
  A(x,k) = max(y_u^k - y_l^{k+1}, y_u^{k+1} - y_l^k)
with boxes made nested by the running hull and gaps in the k grid back-filled by the
smallest computed k' >= k (sound: T_k is contained in T_k'). Output boxes are clipped to
[Y_MIN, Y_MAX]; k = 0 is the certified nominal; beyond the largest computed k the
certificate is the output range GS = Y_MAX - Y_MIN (the released prediction is clipped).

  SS_eps(x) = max( max_k e^{-beta k} A(x,k), e^{-beta (kmax+1)} GS ),
  beta = eps/6 (Cauchy, Def 3.7) or eps/(2 ln(2/delta)) (Laplace, Def 3.8),

noise calibrated per point, everything reported in real target units (x y_std).

    PYTHONPATH=$PWD python -m agtr.california.release --variant norm_gamma1
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

from common import certificate, mechanisms, plotting
from common.metrics import release_metrics
from agtr.california.config import ABLATION as CFG

ROOT = "agtr/california/values/agt_2026-09-14"
STORED = "agtr/california/values/ablation"
FIG = "agtr/california/figures/agt_2026-09-14"
BATCH_SIZES = CFG["BATCH_SIZES"]  # fallback; batch sizes are discovered from the cells at run time
Y_MIN, Y_MAX = CFG["Y_MIN"], CFG["Y_MAX"]
GS = Y_MAX - Y_MIN
EPS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA = 1e-5
N_TRIALS = 500
SEED = 0
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_variant(vdir):
    cells = {}
    for f in glob.glob(f"{vdir}/bs*_k*.npz"):
        m = re.search(r"bs(\d+)_k(\d+)\.npz", f)
        cells[(int(m.group(1)), int(m.group(2)))] = np.load(f)
    return cells


def boxes_for_bs(cells, bs):
    ks = sorted(k for (b, k) in cells if b == bs)
    kpos = [k for k in ks if k > 0]
    if not kpos:
        raise RuntimeError(f"bs={bs}: no k >= 1 cell")
    # the certified nominal is param_n of any k >= 1 cell (a k = 0 cell, if present, is redundant)
    yn = cells[(bs, kpos[0])]["y_n"]
    for k in ks:
        if np.abs(cells[(bs, k)]["y_n"] - yn).max() > 1e-9:
            raise RuntimeError(f"bs={bs} k={k}: nominal prediction differs across k (AGT semantics violated)")
    ync = np.clip(yn, Y_MIN, Y_MAX)
    ks = [0] + kpos  # k = 0 box is the nominal itself
    yl, yu, prev = {}, {}, None
    for k in ks:
        if k == 0:
            l, u = ync.copy(), ync.copy()
        else:
            l = np.clip(cells[(bs, k)]["y_l"], Y_MIN, Y_MAX)
            u = np.clip(cells[(bs, k)]["y_u"], Y_MIN, Y_MAX)
            if not (np.all(l <= ync + 1e-9) and np.all(u >= ync - 1e-9)):
                raise RuntimeError(f"bs={bs} k={k}: certified nominal outside its own box")
            if prev is not None:
                l, u = np.minimum(l, yl[prev]), np.maximum(u, yu[prev])  # running hull (nested)
        yl[k], yu[k] = l, u
        prev = k
    return ks, ync, yl, yu


def cross_radius(ks, yl, yu):
    """A(x, k) for k = 0..k_max, capped at the global sensitivity.

    The box beyond the last computed radius is the public output range: there is no radius beyond
    the battery, so no analytical tail is invented. `warn_on_gaps` is off because the caller already
    refuses a gapped grid unless --allow-gaps was passed for a saturated battery.
    """
    boxes = {k: (yl[k], yu[k]) for k in ks if k > 0}
    tail = (np.full_like(yl[0], Y_MIN), np.full_like(yu[0], Y_MAX))
    return certificate.cross_radius(yl[0], boxes, tail=tail, cap=GS, warn_on_gaps=False)


def smooth_sensitivity(A, beta):
    return certificate.smooth_sensitivity(A, beta, GS)


def metrics(resid, draw, rng, clip=None):
    """clip = (lo - y, hi - y): the release is clipped to the output range before the mean |error|."""
    return release_metrics(resid, draw, rng, n_trials=N_TRIALS, clip=clip)


def curves(resid, scale_by_eps, family, rng, clip=None):
    out = {k: [] for k in ("mse", "mse_p25", "mse_p75", "med_se", "q90_abs", "mae")}
    for eps in EPS:
        s = scale_by_eps[eps]
        draw = {
            "cauchy": lambda r, s=s: r.standard_cauchy(len(resid)) * s,
            "laplace": lambda r, s=s: r.laplace(0, s, len(resid)),
            "gauss": lambda r, s=s: r.normal(0, s, len(resid)),
        }[family]
        for k, v in metrics(resid, draw, rng, clip).items():
            out[k].append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="re-draw the figure from results_cross_radius.json without recomputing",
    )
    ap.add_argument(
        "--allow-gaps",
        action="store_true",
        help="accept a k grid with gaps (ONLY for saturated batteries where A(k)=GS for all k>=1; "
        "otherwise the back-filled bound is only (beta*gap)-smooth, see linear/values/pate_matched/RUNLOG.md)",
    )
    a = ap.parse_args()
    vdir = f"{ROOT}/{a.variant}"
    if a.plot_only:
        make_figure(json.load(open(f"{vdir}/results_cross_radius.json")), a.variant)
        return
    cells = load_variant(vdir)
    y_std = float(np.load(f"{STORED}/y_std.npy"))
    y_mean = float(np.load(f"{STORED}/y_mean.npy"))
    y_test = np.load(f"{STORED}/y_test.npy")
    y_real = y_test * y_std + y_mean
    R = {
        "variant": a.variant,
        "epsilons": EPS,
        "delta": DELTA,
        "GS_real": GS * y_std,
        "batch_sizes": {},
        "gs_baselines": {},
    }
    resid_by_bs, A_by_bs = {}, {}
    batch_sizes = sorted(
        {b for (b, k) in cells}
    )  # discovered from the serialized cells (may include 16512 = full batch)
    KMAX_EXPECTED = max(CFG["K_VALUES"])
    for bs in batch_sizes:
        ks_have = sorted(k for (b, k) in cells if b == bs and k > 0)
        if ks_have and ks_have != list(range(1, max(ks_have) + 1)) and not a.allow_gaps:
            raise SystemExit(
                f"bs={bs}: k grid {ks_have} has gaps; back-filling is not beta-smooth. Run the missing k or pass --allow-gaps for a saturated battery."
            )
        if not ks_have or max(ks_have) < KMAX_EXPECTED:
            # an incomplete battery would be back-filled with GS beyond its last k and misrepresent the curve
            print(
                f"bs={bs}: battery incomplete (k up to {max(ks_have) if ks_have else None} of {KMAX_EXPECTED}); skipped"
            )
            continue
        ks, ync, yl, yu = boxes_for_bs(cells, bs)
        A = cross_radius(ks, yl, yu)
        A_by_bs[bs] = A
        resid = (ync * y_std + y_mean) - y_real
        resid_by_bs[bs] = resid
        clip = (
            Y_MIN * y_std + y_mean - y_real,
            Y_MAX * y_std + y_mean - y_real,
        )  # released prediction clipped to the output range
        entry = {
            "k_computed": ks,
            "nominal_mse_real": float(np.mean(resid**2)),
            "A_mean_by_k_std_units": A.mean(axis=1).tolist(),
            "frac_saturated_k1": float(np.mean((yu[ks[1]] - yl[ks[1]]) >= GS - 1e-9))
            if len(ks) > 1
            else None,
            "published_avg_S_real": (
                float(np.load(f"{STORED}/sensitivity_bs{bs}.npy").mean())
                if os.path.exists(f"{STORED}/sensitivity_bs{bs}.npy")
                else None
            ),
        }
        # pure: Def 3.7 Cauchy(6 SS/eps), beta = eps/6.  pure_liang_*: cor:TighterCauchy, Cauchy(SS^beta/(eps-beta)) with a
        # public beta = eps/6 or eps/2.  approx: Def 3.8 Laplace(2 SS/eps).
        modes = [
            (name, lambda e, m=m: m["beta"](e, DELTA), m["family"], m["multiplier"])
            for name, m in mechanisms.RELEASE_MODES.items()
        ]
        for mode, beta_of, fam, mult in modes:
            ss = {e: smooth_sensitivity(A, beta_of(e)) * y_std for e in EPS}  # real units
            entry[mode] = {
                "beta": [beta_of(e) for e in EPS],
                "SS_mean_real": [float(ss[e].mean()) for e in EPS],
                **curves(
                    resid,
                    {e: mult(e, beta_of(e)) * ss[e] for e in EPS},
                    fam,
                    np.random.default_rng(SEED),
                    clip,
                ),
            }
        R["batch_sizes"][bs] = entry
    if resid_by_bs:
        bs0 = min(resid_by_bs)
        resid = resid_by_bs[bs0]
        clip = (Y_MIN * y_std + y_mean - y_real, Y_MAX * y_std + y_mean - y_real)
        R["gs_baselines"] = {
            "pure": curves(
                resid,
                {e: mechanisms.laplace_gs_scale(GS * y_std, e) for e in EPS},
                "laplace",
                np.random.default_rng(SEED),
                clip,
            ),
            "approx": curves(
                resid,
                {e: mechanisms.gaussian_gs_sigma(GS * y_std, e, DELTA) for e in EPS},
                "gauss",
                np.random.default_rng(SEED),
                clip,
            ),
            "note": f"residuals of bs={bs0} nominal; GS = output range {GS} x y_std",
        }
    os.makedirs(vdir, exist_ok=True)
    json.dump(R, open(f"{vdir}/results_cross_radius.json", "w"), indent=1)

    # ---- console
    print(f"variant {a.variant}: GS (real) = {GS * y_std:.3f}")
    for bs, e in R["batch_sizes"].items():
        print(
            f"bs={bs}: k={e['k_computed'][:4]}..{e['k_computed'][-1]}  nominal MSE {e['nominal_mse_real']:.4f}  A(k=1) mean {e['A_mean_by_k_std_units'][1] if len(e['A_mean_by_k_std_units']) > 1 else float('nan'):.3f} (std units)  "
            f"saturated@k1 {e['frac_saturated_k1']}  published avg S {e['published_avg_S_real'] if e['published_avg_S_real'] is None else round(e['published_avg_S_real'], 3)}"
        )
    for mode, metric in (
        ("pure", "mae"),
        ("pure_liang_b6", "mae"),
        ("pure_liang_b2", "mae"),
        ("approx", "mse"),
    ):
        print(f"\n== {mode} | {metric} | eps = {EPS}")
        for bs, e in R["batch_sizes"].items():
            print(
                f"  bs={bs:<5} SS "
                + " ".join(f"{v:8.3f}" for v in e[mode]["SS_mean_real"])
                + "   | "
                + " ".join(f"{v:9.3g}" for v in e[mode][metric])
            )
        if R["gs_baselines"]:
            print(
                f"  GS      "
                + " " * 72
                + "| "
                + " ".join(
                    f"{v:9.3g}"
                    for v in R["gs_baselines"]["pure" if mode.startswith("pure") else mode][metric]
                )
            )

    make_figure(R, a.variant)


def make_figure(R, variant):
    """Three-panel figure from ``results_cross_radius.json`` only (so it can be re-drawn with --plot-only).

    Left  : mean cross-radius certificate per batch size and SS^beta at beta = 1/2 (the beta of the plotted
            mechanism at eps = 1); dotted = output-range GS, which is also the k > k_max tail.
    Centre: pure (eps,0)-DP with the paper's tighter Cauchy (cor:TighterCauchy), scale SS^beta/(eps-beta) at the
            public beta = eps/2 -- 90% |error| quantile.  Def 3.7 and beta = eps/6 stay in the json.
    Right : approx (eps,1e-5)-DP, Laplace(2 SS/eps) -- test MSE; dashed black = nominal MSE.
    Grey  : Laplace / Gaussian on the output-range global sensitivity.
    """
    if not R["batch_sizes"]:
        return
    os.makedirs(FIG, exist_ok=True)
    EPS_ = R["epsilons"]
    plotting.apply_combined(3, legend_pt=7.0)
    fig, axes = plt.subplots(1, 3, figsize=plotting.combined_figsize(3), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.20, w_pad=0.06)
    ax = axes[0]
    ax2 = ax.twinx()
    for i, bs in enumerate(R["batch_sizes"]):
        A = np.array(R["batch_sizes"][bs]["A_mean_by_k_std_units"])
        ks = np.arange(len(A))
        c = COLORS[i]
        ax.step(
            ks,
            A,
            where="post",
            color=c,
            ls="--",
            lw=plt.rcParams["lines.linewidth"],
            alpha=0.75,
            label=f"BS={bs}",
        )
        t = np.linspace(0, len(A) + 4, 600)
        Ai = np.where(t > len(A) - 1, GS, A[np.clip(t.astype(int), 0, len(A) - 1)])
        s = Ai * np.exp(-0.5 * t)
        ax2.plot(t, s, "-", color=c, lw=plt.rcParams["lines.linewidth"] + 0.4)
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
    ax.axhline(GS, color="black", ls=":", lw=2.0, label=f"Output range GS ({GS:g})")
    ax.set_xlabel(r"Substitutions $k$")
    ax.set_ylabel(r"$\bar{A}(x,k)$ (dashed, std units)")
    ax2.set_ylabel(r"$SS^{\beta}$ at $\beta=1/2$ (solid)")
    ax.set_title("Certificate (genuine AGT)")
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
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
        gsm = "pure" if mode.startswith("pure") else mode
        ax.plot(
            EPS_,
            R["gs_baselines"][gsm][metric],
            "o-",
            color="black",
            alpha=0.85,
            zorder=20,
            label="Global sensitivity",
            **plotting.SERIES,
        )
        for i, bs in enumerate(R["batch_sizes"]):
            ax.plot(
                EPS_,
                R["batch_sizes"][bs][mode][metric],
                "o-",
                color=COLORS[i],
                label=f"BS={bs}",
                zorder=6,
                **plotting.SERIES,
            )
        if mode == "approx":
            nom = float(np.mean([e["nominal_mse_real"] for e in R["batch_sizes"].values()]))
            ax.axhline(nom, color="black", ls="--", lw=1.2, alpha=0.7, label=f"Nominal MSE ({nom:.2f})")
        ax.set(xscale="log", yscale="log", xlabel=r"Privacy Budget $\epsilon$", ylabel=ylab, title=title)
        ax.grid(True, alpha=0.3)
    plotting.shared_legend(
        fig, axes, ncol=6, loc="outside upper center", columnspacing=0.8, handlelength=1.5, handletextpad=0.4
    )
    fig.savefig(f"{FIG}/ca_cross_radius_{variant}.png")
    plt.close(fig)
    print(f"wrote {FIG}/ca_cross_radius_{variant}.png")


if __name__ == "__main__":
    main()
