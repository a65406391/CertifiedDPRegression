"""Release + comparison for the matched PATE-R vs AGT-R benchmark (reads the cells written by train_milp_unsound.py).

Arms, all (eps, 0)-DP under substitution adjacency on the full N-point dataset (approx-DP versions too):

  PATE-R (GS)      g(x) = mean_i f_i(x) + Lap(GS / (T eps))                       [Gaussian for approx]
  PATE-AGT-R       g(x) + Cauchy calibrated to SS_g(x) = (1/T) max_i SS_i(x)      [Laplace(2 SS_g/eps) for approx]
  AGT-R (full)     f_0(x) of the T = 1 partition + Cauchy calibrated to SS_0(x)
  AGT-R (WOR, T)   f_0(x) of the T-partition (a uniform WOR subsample of size N/T) with the amplified base
                   budget eps_b = log(1 + T (e^eps - 1)) (Lemma agtr_wor_amplification), delta_b = T delta.

Per teacher: nominal (w, b) = param_n (identical across k, asserted), boxes B_k (running hull for nesting),
exact output interval of w x + b over the box (predictions and interval ends clipped to [-6, 6]),
cross-radius certificate A(x, k) = max(u_k - l_{k+1}, u_{k+1} - l_k) for k = 0..k_max-1 (k = 0 pairs the
nominal with B_1), A(x, k_max) pairs B_{k_max} with the GS box, tail k > k_max = GS.  The k grid of every
teacher is dense (asserted), so SS(x) = max_k e^{-beta k} A(x, k) is beta-smooth (see RUNLOG.md).
The aggregate's local sensitivity at distance k is at most (1/T) max_i A_i(x, k) (the k substitutions
land in the shards in some way; each shard's A_i is nondecreasing in k), hence SS_g = (1/T) max_i SS_i.

Mechanisms: pure DP -- Def 3.7 Cauchy(6 SS/eps) with beta = eps/6, and cor:TighterCauchy
Cauchy(SS^beta/(eps-beta)) at public beta = eps/6 and eps/2 (the plotted one);
approx DP -- Laplace(2 SS/eps) with beta = eps/(2 ln(2/delta)).  Errors summarised by the 90 %
quantile of |error| (pure; MSE is undefined under Cauchy) and MSE (approx), N_TRIALS noise draws.
Solver hygiene: any cell whose Gurobi log recorded a time-limited solve is reported and excluded from
the release (a time-limited best bound is not a monotone function of the data).

    PYTHONPATH=$PWD python -m agtr.linear.pate.release --tag n4000_e20_lr0.3_g1
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

from common import certificate as cert
from common import mechanisms
from common import plotting
from common.metrics import release_metrics

ROOT = "agtr/linear/values/pate_matched"
FIG = "agtr/linear/figures/pate_matched"
EPS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA = 1e-5
N_TRIALS = 200
SEED = 0
# Defined once in common.mechanisms; kept as module names because the results JSON keys use them.
PURE_MODES = {
    m: (mechanisms.RELEASE_MODES[m]["beta"], mechanisms.RELEASE_MODES[m]["multiplier"])
    for m in mechanisms.PURE_MODES
}


def APPROX_BETA(e):
    return mechanisms.RELEASE_MODES["approx"]["beta"](e, DELTA)


# ----------------------------------------------------------------------------- cells
def load_run(tag):
    """cells: (T, shard, k) -> npz path for the MILP layout (shard{i}_k{k}.npz), or (T, shard, None) -> path for the
    exact-certifier layout (shard{i}.npz holding every k)."""
    d = f"{ROOT}/{tag}"
    cfg = json.load(open(f"{d}/config.json"))
    X_test = np.load(f"{d}/X_test.npy").ravel()
    y_test = np.load(f"{d}/y_test.npy").ravel()
    cells = {}
    for f in glob.glob(f"{d}/T*/shard*.npz"):
        m = re.search(r"T(\d+)/shard(\d+)(?:_k(\d+))?\.npz", f)
        cells[(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)] = f
    return cfg, X_test, y_test, cells


def teacher(cells, T, i, y_clip):
    """(nominal (w,b), ks, nested final boxes [(l, u)] per k, solver flags) of teacher i of the T-partition."""
    if (T, i, None) in cells:  # exact-certifier layout
        z = np.load(cells[(T, i, None)])
        ks = [int(k) for k in z["ks"]]
        nom = z["nominal"][-1].copy()
        assert ks == list(range(1, ks[-1] + 1)), f"T={T} shard {i}: k grid not dense"
        fin = z["boxes"][:, -1]  # (K, 2, 2) -> [[wl, wu], [bl, bu]]
        raw = [(fin[j, :, 0].copy(), fin[j, :, 1].copy()) for j in range(len(ks))]
        bad = []
    else:  # MILP layout
        ks = sorted(k for (t, s, k) in cells if t == T and s == i and k is not None)
        assert ks and ks == list(range(1, ks[-1] + 1)), f"T={T} shard {i}: k grid {ks} is not dense 1..k_max"
        nom, raw, bad = None, [], []
        for k in ks:
            z = np.load(cells[(T, i, k)])
            if nom is None:
                nom = z["param_n"].copy()
            assert np.allclose(z["param_n"], nom, atol=1e-12), (
                f"T={T} shard {i}: nominal differs between k cells"
            )
            if int(z["gurobi_time_limit"]) > 0 or int(z["gurobi_optimal"]) < int(z["gurobi_solves"]):
                bad.append(
                    (k, int(z["gurobi_optimal"]), int(z["gurobi_time_limit"]), int(z["gurobi_solves"]))
                )
            raw.append((z["param_l"].copy(), z["param_u"].copy()))
    boxes = []
    for l, u in raw:  # running hull: boxes must be nested in k (tiny tolerance violations are enlarged)
        if boxes:
            l = np.minimum(l, boxes[-1][0])
            u = np.maximum(u, boxes[-1][1])
        assert np.all(l <= nom + 1e-9) and np.all(u >= nom - 1e-9)
        boxes.append((l, u))
    return nom, ks, boxes, bad


def out_interval(box, x, y_clip):
    """Exact interval of w x + b over a parameter box, ends clipped to the output range."""
    (wl, bl), (wu, bu) = box
    lo = np.where(x >= 0, wl * x, wu * x) + bl
    hi = np.where(x >= 0, wu * x, wl * x) + bu
    return np.clip(lo, -y_clip, y_clip), np.clip(hi, -y_clip, y_clip)


def certificate(nom, boxes, x, y_clip, gs):
    """A(x, k) for k = 0..k_max, shape (n_test, k_max + 1). The box beyond the last computed radius
    is the public output range, and every entry is capped at the global sensitivity."""
    yn = np.clip(nom[0] * x + nom[1], -y_clip, y_clip)
    per_k = {k + 1: out_interval(b, x, y_clip) for k, b in enumerate(boxes)}
    tail = (np.full_like(x, -y_clip), np.full_like(x, y_clip))
    return yn, cert.cross_radius(yn, per_k, tail=tail, cap=gs).T


def smooth_sensitivity(A, gs, beta):
    """max_k e^(-beta k) A(x, k) over k = 0..k_max, plus the GS tail e^(-beta (k_max+1)) gs."""
    return cert.smooth_sensitivity(A.T, beta, gs)


def metrics(resid, draw, rng, clip=None, n_trials=None):
    """clip = (lo - y, hi - y): the release is clipped to the output range before the mean |error|.

    This analysis aggregates trials with the MEAN, where the linear and California releases use the
    median. Under Cauchy noise the mean of the per-trial MSE is dominated by the occasional enormous
    draw, so `mse`, `med_se` and `q90_abs` here are not comparable with those reported elsewhere;
    `mae`, the headline pure-DP metric, uses the mean everywhere and is comparable. Kept as it was
    so the published PATE numbers are unchanged.
    """
    return release_metrics(
        resid, draw, rng, n_trials=N_TRIALS if n_trials is None else n_trials, clip=clip, aggregate="mean"
    )


def release_curves(resid, ss_by_eps_beta, kind, clip=None, n_trials=None):
    """ss_by_eps_beta: eps -> (scale array per test point).  kind: 'cauchy' | 'laplace' | 'gauss'."""
    out = {m: [] for m in ("q90_abs", "med_se", "mse", "mse_p25", "mse_p75", "mae")}
    rng = np.random.default_rng(SEED)
    for e in EPS:
        s = ss_by_eps_beta[e]
        draw = {
            "cauchy": lambda r, s=s: s * r.standard_cauchy(size=resid.shape),
            "laplace": lambda r, s=s: r.laplace(0.0, s, size=resid.shape),
            "gauss": lambda r, s=s: r.normal(0.0, s, size=resid.shape),
        }[kind]
        for m_, v in metrics(resid, draw, rng, clip, n_trials).items():
            out[m_].append(v)
    return out


def arms_from_ss(resid, ss_fn, gs_scale_T, eps_map=None, delta_scale=1.0, clip=None, n_trials=None):
    """All mechanisms for one predictor.  ss_fn(beta) -> SS(x) array (already divided by T for PATE);
    gs_scale_T = GS / T.  eps_map maps the outer eps to the base eps used for calibration (amplification)."""
    eps_map = eps_map or (lambda e: e)
    res = {}
    for mode, (beta_of, scale_of) in PURE_MODES.items():
        d = {}
        for e in EPS:
            eb = eps_map(e)
            b = beta_of(eb)
            d[e] = ss_fn(b) * scale_of(eb, b)
        res[mode] = release_curves(resid, d, "cauchy", clip, n_trials)
        res[mode]["beta"] = [beta_of(eps_map(e)) for e in EPS]
        res[mode]["SS_mean"] = [float(np.mean(ss_fn(beta_of(eps_map(e))))) for e in EPS]
    d = {}
    for e in EPS:
        eb = eps_map(e)
        db = DELTA * delta_scale
        b = eb / (2.0 * np.log(2.0 / db))
        d[e] = 2.0 * ss_fn(b) / eb
    res["approx"] = release_curves(resid, d, "laplace", clip, n_trials)
    res["approx"]["beta"] = [eps_map(e) / (2.0 * np.log(2.0 / (DELTA * delta_scale))) for e in EPS]
    res["gs_pure"] = release_curves(
        resid, {e: np.full(resid.shape, gs_scale_T / eps_map(e)) for e in EPS}, "laplace", clip
    )
    res["gs_approx"] = release_curves(
        resid,
        {
            e: np.full(
                resid.shape, gs_scale_T * np.sqrt(2.0 * np.log(1.25 / (DELTA * delta_scale))) / eps_map(e)
            )
            for e in EPS
        },
        "gauss",
        clip,
        n_trials,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n4000_e20_lr0.3_g1")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument(
        "--no-gs", action="store_true", help="omit the GS-on-full-data curve from the best-vs-eps figure"
    )
    a = ap.parse_args()
    d = f"{ROOT}/{a.tag}"
    os.makedirs(FIG, exist_ok=True)
    if a.plot_only:
        make_figures(json.load(open(f"{d}/results.json")), a.tag, show_gs=not a.no_gs)
        return
    cfg, x, y, cells = load_run(a.tag)
    y_clip, gs = cfg["clip_y"], cfg["gs_output_range"]
    Ts = sorted({t for (t, _, _) in cells})
    R = {
        "tag": a.tag,
        "epsilons": EPS,
        "delta": DELTA,
        "n_trials": N_TRIALS,  # the module default; arms_from_ss callers may override
        "gs": gs,
        "T": {},
        "excluded_cells": [],
    }
    print(f"{a.tag}: {len(cells)} cells, T = {Ts}")
    for T in Ts:
        shards = sorted({s for (t, s, _) in cells if t == T})
        assert shards == list(range(T)), f"T={T}: shards present {shards}"
        preds, SS = [], {}  # SS[beta] built lazily per teacher -> we store A and compute on demand
        A_list, bad_all = [], []
        for i in shards:
            nom, ks, boxes, bad = teacher(cells, T, i, y_clip)
            if bad:
                bad_all.append((i, bad))
            yn, A = certificate(nom, boxes, x, y_clip, gs)
            preds.append(yn)
            A_list.append(A)
        if bad_all:
            R["excluded_cells"].append({"T": T, "teachers_with_time_limited_solves": bad_all})
            print(f"  T={T}: {len(bad_all)} teachers had time-limited / non-optimal solves -> arm excluded")
            continue
        preds = np.stack(preds)
        kmax = min(A.shape[1] for A in A_list) - 1
        mean_pred = preds.mean(axis=0)
        resid_pate = mean_pred - y
        resid_t0 = preds[0] - y
        clip = (-y_clip - y, y_clip - y)  # released prediction clipped to the output range (post-processing)
        # per-teacher SS -> aggregate: (1/T) max_i SS_i  (each SS_i uses its own dense grid + GS tail)
        ss_pate = lambda beta: np.max(np.stack([smooth_sensitivity(A, gs, beta) for A in A_list]), axis=0) / T
        ss_t0 = lambda beta: smooth_sensitivity(A_list[0], gs, beta)
        m_T = int(
            len(np.load(cells[(T, 0, None)] if (T, 0, None) in cells else cells[(T, 0, 1)])["shard_idx"])
        )
        entry = {
            "m": m_T,
            "k_max": int(kmax),
            "nominal_mse_pate": float(np.mean(resid_pate**2)),
            "nominal_q90_pate": float(np.quantile(np.abs(resid_pate), 0.9)),
            "nominal_mae_pate": float(np.mean(np.abs(resid_pate))),
            "nominal_mse_teacher0": float(np.mean(resid_t0**2)),
            "A_mean_by_k_teacher_max": [
                float(v) for v in np.max(np.stack([A[:, : kmax + 1].mean(axis=0) for A in A_list]), axis=0)
            ],
            "pate": arms_from_ss(resid_pate, ss_pate, gs / T, clip=clip),
            "wor": arms_from_ss(
                resid_t0,
                ss_t0,
                gs,
                eps_map=lambda e: float(np.log1p(T * np.expm1(e))),
                delta_scale=T,
                clip=clip,
            ),
        }
        if T == 1:
            entry["full"] = entry["wor"]  # eps_map is the identity at T = 1: AGT-R on the full data
        R["T"][T] = entry
        p = entry["pate"]
        print(
            f"  T={T:3d} m={entry['m']:5d} k_max={kmax:2d} nominal MSE (pate) {entry['nominal_mse_pate']:.4f} teacher0 {entry['nominal_mse_teacher0']:.4f} | "
            f"pure b2 MAE @eps=1: PATE-R {p['gs_pure']['mae'][3]:.3f} PATE-AGT-R {p['pure_liang_b2']['mae'][3]:.3f} "
            f"WOR {entry['wor']['pure_liang_b2']['mae'][3]:.3f} | approx mse @eps=1: GS {p['gs_approx']['mse'][3]:.3f} AGT {p['approx']['mse'][3]:.3f}"
        )
    json.dump(R, open(f"{d}/results.json", "w"), indent=1)
    summarise(R)
    make_figures(R, a.tag)


def summarise(R):
    R["T"] = {int(k): v for k, v in R["T"].items()}
    Ts = sorted(R["T"])
    e_idx = {e: i for i, e in enumerate(EPS)}
    print(
        "\nBest T per eps (pure DP, tighter Cauchy beta=eps/2, mean |error| of the clipped release; approx DP, MSE):"
    )
    print(
        f"{'eps':>5} | {'PATE-R best T':>14} {'value':>8} | {'PATE-AGT-R best T':>18} {'value':>8} | {'AGT-R full':>10} | {'WOR best T':>10} {'value':>8} || approx: {'PATE-R T':>8} {'mse':>8} | {'PATE-AGT-R T':>12} {'mse':>8}"
    )
    for e in EPS:
        j = e_idx[e]

        def best(arm, mode, metric):
            v, T = min((R["T"][T][arm][mode][metric][j], T) for T in Ts)
            return T, v

        Tg, vg = best("pate", "gs_pure", "mae")
        Ta, va = best("pate", "pure_liang_b2", "mae")
        Tw, vw = best("wor", "pure_liang_b2", "mae")
        full = R["T"][1]["full"]["pure_liang_b2"]["mae"][j]
        Tg2, vg2 = best("pate", "gs_approx", "mse")
        Ta2, va2 = best("pate", "approx", "mse")
        print(
            f"{e:5g} | {Tg:14d} {vg:8.3g} | {Ta:18d} {va:8.3g} | {full:10.3g} | {Tw:10d} {vw:8.3g} || {Tg2:8d} {vg2:8.3g} | {Ta2:12d} {va2:8.3g}"
        )


def make_figures(R, tag, show_gs=True):
    R["T"] = {int(k): v for k, v in R["T"].items()}
    Ts = sorted(R["T"])
    eps = R["epsilons"]
    plotting.apply_combined(3, legend_pt=7.0)
    # ---- Figure 1: value vs T for each eps (pure, tighter Cauchy beta = eps/2), PATE-R vs PATE-AGT-R vs WOR
    fig, axes = plt.subplots(1, 3, figsize=plotting.combined_figsize(3), layout="constrained")
    cmap = plt.get_cmap("viridis")
    cols = [cmap(i / (len(eps) - 1)) for i in range(len(eps))]
    for ax, arm, mode, metric, title in (
        (axes[0], "pate", "gs_pure", "mae", "PATE-R (GS, Laplace)"),
        (axes[1], "pate", "pure_liang_b2", "mae", r"PATE-AGT-R (SS, Cauchy $\beta=\epsilon/2$)"),
        (axes[2], "wor", "pure_liang_b2", "mae", r"AGT-R on one shard, amplified $\epsilon_b$"),
    ):
        for j, e in enumerate(eps):
            ax.plot(
                Ts,
                [R["T"][T][arm][mode][metric][j] for T in Ts],
                "o-",
                color=cols[j],
                label=rf"$\epsilon={e:g}$",
                **plotting.SERIES,
            )
        ax.plot(
            Ts,
            [R["T"][T]["nominal_mae_pate"] for T in Ts],
            "--",
            color="black",
            lw=1.0,
            alpha=0.7,
            label="noise-free (PATE mean)" if ax is axes[0] else None,
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Number of shards $T$",
            ylabel="Mean Absolute Error",
            title=title,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xticks(Ts)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(t))}}}$" for t in Ts])  # powers of two: 1024 does not fit
        ax.tick_params(
            axis="x", which="major", pad=plt.rcParams["xtick.major.pad"] * 2.2
        )  # clear of the exponents
    plotting.shared_legend(  # one line under the row: seven budgets and the noise-free line
        fig,
        axes,
        ncol=8,
        loc="outside lower center",
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
        fontsize=plt.rcParams["legend.fontsize"] * 1.2,
    )
    fig.get_layout_engine().set(h_pad=0.12)  # a strip of white between the panels and the legend
    fig.savefig(f"{FIG}/pate_vs_T_{tag}.png")
    plt.close(fig)
    # ---- Figure 2: best-over-T of each arm vs eps (pure and approx) + AGT-R full
    fig, axes = plt.subplots(1, 2, figsize=plotting.combined_figsize(2), layout="constrained")
    for ax, (gs_mode, agt_mode, metric, ylab, title) in zip(
        axes,
        (
            (
                "gs_pure",
                "pure_liang_b2",
                "mae",
                "Mean Absolute Error",
                r"Pure $(\epsilon,0)$-DP, Cauchy $\beta=\epsilon/2$",
            ),
            ("gs_approx", "approx", "mse", "Test MSE", r"Approx. $(\epsilon,\delta=10^{-5})$-DP"),
        ),
    ):
        best = lambda arm, mode: [min(R["T"][T][arm][mode][metric][j] for T in Ts) for j in range(len(eps))]
        bestT = lambda arm, mode: [
            min((R["T"][T][arm][mode][metric][j], T) for T in Ts)[1] for j in range(len(eps))
        ]
        ax.plot(
            eps,
            best("pate", gs_mode),
            "o-",
            color="black",
            zorder=20,
            label="PATE-R, best $T$",
            **plotting.SERIES,
        )
        ax.plot(
            eps,
            best("pate", agt_mode),
            "o-",
            color="#d62728",
            zorder=10,
            label="PATE-AGT-R, best $T$",
            **plotting.SERIES,
        )
        ax.plot(
            eps,
            R["T"][1]["full"][agt_mode][metric],
            "o-",
            color="#1f77b4",
            label="AGT-R, full data ($T=1$)",
            **plotting.SERIES,
        )
        ax.plot(
            eps,
            best("wor", agt_mode),
            "o--",
            color="#2ca02c",
            label="AGT-R, one shard + amplification, best $T$",
            **plotting.SERIES,
        )
        if show_gs:
            ax.plot(
                eps,
                R["T"][1]["pate"][gs_mode][metric],
                "o:",
                color="gray",
                label="GS on full data ($T=1$)",
                **plotting.SERIES,
            )
        for j, e in enumerate(eps):
            ax.annotate(
                str(bestT("pate", gs_mode)[j]),
                (e, best("pate", gs_mode)[j]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6,
                color="black",
            )
            ax.annotate(
                str(bestT("pate", agt_mode)[j]),
                (e, best("pate", agt_mode)[j]),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                fontsize=6,
                color="#d62728",
            )
        ax.set(xscale="log", yscale="log", xlabel=r"Privacy Budget $\epsilon$", ylabel=ylab, title=title)
        ax.grid(True, alpha=0.3)
    plotting.shared_legend(
        fig, axes, ncol=5, loc="outside upper center", columnspacing=0.8, handlelength=1.5, handletextpad=0.4
    )
    fig.savefig(f"{FIG}/pate_best_vs_eps_{tag}.png")
    plt.close(fig)
    print(f"wrote {FIG}/pate_vs_T_{tag}.png and {FIG}/pate_best_vs_eps_{tag}.png")


if __name__ == "__main__":
    main()
