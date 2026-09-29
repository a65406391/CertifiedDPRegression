"""Linear AGT-R: recompute the release under the cross-radius convention.

Reuses the *published* parameter bounds (no new certification):

  * MIP cells (F1/F2/F4/F8/F16, k <= 5F): the final window's four Gurobi solves
    recorded in ``values/ablation_k20_run.log`` (order Min w, Max w, Min b, Max b).
    The *best bound* of each solve is used, which is a sound endpoint whatever the
    incumbent was; validated in ``provenance.json`` against the stored 50-point widths
    (``method_results_*.npy``): incumbent-based widths match to ~1e-14 in 72/75 cells,
    bound-based widths are never tighter and at most 3.1e-4 wider.
  * IBP cells (method ``ibp``, and the k > 5F fallbacks of F1/F2): regenerated with
    ``privacy_certified_training`` (deterministic, seconds) and validated to 3e-16
    against the stored widths.
  * k >= b = 5 under IBP equals the data-independent clipping ceiling
    ``theta_init +- T*lr*gamma``; it is used for every k > 20 (the battery endpoint,
    equivalent to k = N) and as the sound global-sensitivity baseline.

Then, per query x, with exact interval evaluation of the linear model:

  A(x,k)  = max(y_u^k - y_l^{k+1}, y_u^{k+1} - y_l^k)     k = 0..20  (k=0: box {nominal})
  SS_eps  = max( max_k e^{-beta_eps k} A(x,k),  e^{-21 beta_eps} * width_ceiling(x) )
  beta_eps = eps/6 (Cauchy, Def 3.7)   or   eps/(2 ln(2/delta)) (Laplace, Def 3.8)

Boxes are made nested by the running hull (enlarging is always sound); the nominal is
the certified per-sample-clamp trajectory (``param_n``), not the separately trained
k=0 model. Noise is calibrated per test point. Outputs go to
``values/recompute_2026-09-14/``; nothing under ``values/ablation_k20`` is touched.

    PYTHONPATH=$PWD python -m agtr.linear.release
"""

import json
import os
import re

import numpy as np
import torch

from agtr.linear.config import ABLATION as CFG
from common.data import simulated_linear_data, get_or_create_w_true
from agtr.linear.certify import train_certified_model
from common import certificate
from common.metrics import release_metrics

LOG = "agtr/linear/values/ablation_k20_run.log"
RES = "agtr/linear/values/ablation_k20"
OUT = "agtr/linear/values/recompute_2026-09-14"
METHODS = ["ibp", "f1", "f2", "f4", "f8", "f16"]
FREQ = {"f1": 1, "f2": 2, "f4": 4, "f8": 8, "f16": 16}
KMAX = 20
EPS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA = 1e-5
MAE_DRAWS = 5
N_TRIALS = 500
SEED = 0
LR, GAMMA, BS, EPOCHS, N = CFG["LR"], CFG["CLIP_GAMMA"], CFG["BATCH_SIZE"], CFG["EPOCHS"], CFG["N_SAMPLES"]
T_STEPS = EPOCHS * (N // BS)  # 16


# ----------------------------------------------------------------------------- bounds
def parse_log():
    """Per (method, k): the last window's (w_min, w_max, b_min, b_max) best bounds."""
    lines = open(LOG, errors="replace").read().split("\n")
    seen = {}
    for i, l in enumerate(lines):
        m = re.search(r"Method (F\d+) \(k=(\d+)\):", l)
        if m:
            seen.setdefault((m.group(1).lower(), int(m.group(2))), i)
    keys = sorted(seen, key=lambda t: (METHODS.index(t[0]), t[1]))
    out, prov = {}, {}
    for idx, (m, k) in enumerate(keys):
        if k == 0 or k > FREQ[m] * BS:
            continue
        a = seen[(m, k)]
        b = seen[keys[idx + 1]] if idx + 1 < len(keys) else len(lines)
        sols, sense = [], None
        for j in range(a, b):
            l = lines[j]
            if l.startswith("INFO"):
                continue
            s = re.search(r"Optimize a model with .*\((Max|Min)\)", l)
            if s:
                sense = s.group(1)
                continue
            o = re.search(r"Best objective ([-\d.eE+]+), best bound ([-\d.eE+]+)", l)
            if o:
                sols.append((sense, float(o.group(1)), float(o.group(2)), j + 1))
                sense = None
                continue
            o = re.search(r"Optimal objective\s+([-\d.eE+]+)", l)
            if o:
                sols.append((sense, float(o.group(1)), float(o.group(1)), j + 1))
                sense = None
        last = sols[-4:]
        if [s[0] for s in last] != ["Min", "Max", "Min", "Max"]:
            raise RuntimeError(f"{m} k={k}: unexpected solve order {[s[0] for s in last]}")
        out[(m, k)] = np.array([s[2] for s in last])  # best bounds
        prov[(m, k)] = dict(
            source="gurobi-log-best-bound",
            log_lines=[s[3] for s in last],
            incumbent=[s[1] for s in last],
            best_bound=[s[2] for s in last],
            n_solves_in_cell=len(sols),
        )
    return out, prov


def regenerate_ibp(X, y):
    boxes, nominal = {}, None
    for k in range(1, KMAX + 1):
        m = train_certified_model(X, y, epochs=EPOCHS, lr=LR, seed=42, method="ibp", k=k)
        boxes[k] = np.array(
            [float(m.param_l[0]), float(m.param_u[0]), float(m.param_l[1]), float(m.param_u[1])]
        )
        nom = np.array([float(m.param_n[0]), float(m.param_n[1])])
        if nominal is None:
            nominal = nom
        elif np.abs(nom - nominal).max() > 1e-12:
            raise RuntimeError("IBP nominal differs across k")
    return boxes, nominal


def ceiling_box():
    init = torch.load(CFG["INIT_STATE_PATH"], map_location="cpu")
    w0 = float(init["0.weight"].flatten()[0])
    b0 = float(init["0.bias"].flatten()[0])
    r = T_STEPS * LR * GAMMA
    return np.array([w0 - r, w0 + r, b0 - r, b0 + r]), (w0, b0)


def hull(prev, cur):
    return np.array([min(prev[0], cur[0]), max(prev[1], cur[1]), min(prev[2], cur[2]), max(prev[3], cur[3])])


# ---------------------------------------------------------------------------- release
def out_box(box, x):
    """Exact output interval of y = w x + b over the parameter box, for x (n,)."""
    wl, wu, bl, bu = box
    yl = np.where(x >= 0, wl * x, wu * x) + bl
    yu = np.where(x >= 0, wu * x, wl * x) + bu
    return yl, yu


def cross_radius_profile(boxes, ceil, nominal, x):
    """A(x, k) for k = 0..KMAX plus the ceiling width; boxes must be nested (hull applied)."""
    yn = nominal[0] * x + nominal[1]
    per_k = {k: out_box(boxes[k], x) for k in range(1, KMAX + 1)}
    ceil_box = out_box(ceil, x)
    A = certificate.cross_radius(yn, per_k, tail=ceil_box)
    return A, (ceil_box[1] - ceil_box[0])


def smooth_sensitivity(A, ceil_width, beta):
    return certificate.smooth_sensitivity(A, beta, ceil_width)


Y_OUT = 5.0  # public output domain [-5, 5] (the published GS = 10); the released prediction is clipped to it


def _metrics(resid, draw, rng, clip=None):
    """Three error summaries of the released predictions over N_TRIALS draws.

    mse      : mean over points of squared error (published metric; infinite
               expectation under Cauchy noise, so its Monte-Carlo value is a tail artefact)
    med_se   : median over points of squared error (finite for every mechanism)
    q90_abs  : 90th percentile over points of |error| (the quantile object of Thms 4.3/4.4)
    mae      : mean over points of |error| of the RELEASED prediction clipped to the public output
               domain (post-processing, no privacy cost), averaged over all trials; finite for every
               mechanism and the plotted pure-DP metric.  clip = (lo - y, hi - y) per point.
    Each is summarised by its median over trials, with the 25-75 band for mse.
    """
    m = release_metrics(resid, draw, rng, n_trials=N_TRIALS, clip=clip)
    return (m["mse"], m["mse_p25"], m["mse_p75"], m["med_se"], m["q90_abs"], m["mae"])


def mse_curves(nominal, x, y, ss_by_eps, mode, rng, beta_by_eps=None):
    """Per-point calibrated noise.

    pure        : Cauchy(6 SS/eps), Def 3.7 (beta = eps/6)
    pure_liang  : Cauchy(SS^beta/(eps - beta)), cor:TighterCauchy (Liang et al. 2024), any public beta < eps;
                  beta_by_eps supplies the public beta used for the staircase
    approx      : Laplace(2 SS/eps), Def 3.8
    """
    yn = nominal[0] * x + nominal[1]
    resid = yn - y
    clip = (-Y_OUT - y, Y_OUT - y)
    out = {k: [] for k in ("mse", "mse_p25", "mse_p75", "med_se", "q90_abs", "mae")}
    for eps in EPS:
        ss = ss_by_eps[eps]
        if mode == "pure":
            scale = 6.0 * ss / eps
            draw = lambda r, s=scale: r.standard_cauchy(len(x)) * s
        elif mode == "pure_liang":
            scale = ss / (eps - beta_by_eps[eps])
            draw = lambda r, s=scale: r.standard_cauchy(len(x)) * s
        else:
            scale = 2.0 * ss / eps
            draw = lambda r, s=scale: r.laplace(0, s, len(x))
        for k, v in zip(out, _metrics(resid, draw, rng, clip)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def gs_curves(nominal, x, y, gs, mode, rng):
    """Global-sensitivity baselines: Laplace(gs/eps) (pure) and Gaussian sigma = gs sqrt(2 ln(1.25/delta))/eps (Def 3.4)."""
    yn = nominal[0] * x + nominal[1]
    resid = yn - y
    clip = (-Y_OUT - y, Y_OUT - y)
    out = {k: [] for k in ("mse", "mse_p25", "mse_p75", "med_se", "q90_abs", "mae")}
    for eps in EPS:
        if mode == "pure":
            draw = lambda r, s=gs / eps: r.laplace(0, s, len(x))
        else:
            draw = lambda r, s=gs * np.sqrt(2 * np.log(1.25 / DELTA)) / eps: r.normal(0, s, len(x))
        for k, v in zip(out, _metrics(resid, draw, rng, clip)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    w_true = get_or_create_w_true(1, 1, path=CFG["W_TRUE_PATH"])
    X, y = simulated_linear_data(
        n_samples=N, dim=1, sparsity=1, seed=42, w_true=w_true, w_true_path=CFG["W_TRUE_PATH"]
    )
    Xte, yte = simulated_linear_data(
        n_samples=2000, dim=1, sparsity=1, seed=100, w_true=w_true, w_true_path=CFG["W_TRUE_PATH"]
    )
    xte, yte = Xte.flatten(), yte.flatten()
    np.random.seed(999)
    xs = np.random.uniform(-2, 2, (CFG["N_EVAL"], 1)).flatten()  # the 50 published eval points
    A50 = np.c_[np.abs(xs), np.ones(len(xs))]

    mip, prov = parse_log()
    ibp, nominal = regenerate_ibp(X, y)
    ceil, init = ceiling_box()
    stored = {m: np.load(f"{RES}/method_results_{m}.npy", allow_pickle=True).item() for m in METHODS}

    # assemble raw boxes per method/k with provenance, validate against stored widths
    raw, provenance = (
        {},
        {
            "nominal": dict(
                w=nominal[0], b=nominal[1], source="privacy_certified_training param_n (identical for all k)"
            ),
            "ceiling": dict(box=ceil.tolist(), init=init, radius=T_STEPS * LR * GAMMA),
            "cells": {},
        },
    )
    worst = {}
    for m in METHODS:
        raw[m] = {}
        for k in range(1, KMAX + 1):
            if m != "ibp" and (m, k) in mip:
                box, p = mip[(m, k)], dict(prov[(m, k)])
            else:
                box, p = (
                    ibp[k],
                    dict(source="ibp-regenerated" if m == "ibp" else "ibp-fallback (k > 5F), regenerated"),
                )
            V = stored[m]["V_k_mat"][:, list(stored[m]["k_values"]).index(k)]
            width_diff = A50 @ [box[1] - box[0], box[3] - box[2]] - V
            p.update(
                box=box.tolist(),
                max_width_excess_vs_stored=float(width_diff.max()),
                min_width_excess_vs_stored=float(width_diff.min()),
            )
            if width_diff.min() < -1e-9:
                raise RuntimeError(
                    f"{m} k={k}: reconstructed box narrower than stored widths ({width_diff.min():.2e})"
                )
            provenance["cells"][f"{m}_k{k}"] = p
            worst[m] = max(worst.get(m, 0.0), float(width_diff.max()))
            raw[m][k] = box
    # nominal inside every k=1 box, boxes inside ceiling
    for m in METHODS:
        b1 = raw[m][1]
        assert (
            b1[0] - 1e-12 <= nominal[0] <= b1[1] + 1e-12 and b1[2] - 1e-12 <= nominal[1] <= b1[3] + 1e-12
        ), m
    # running hull -> nested, intersect nothing (hull only enlarges)
    boxes = {}
    for m in METHODS:
        boxes[m] = {}
        prev = None
        for k in range(1, KMAX + 1):
            cur = raw[m][k] if prev is None else hull(prev, raw[m][k])
            cur = np.array(
                [max(cur[0], ceil[0]), min(cur[1], ceil[1]), max(cur[2], ceil[2]), min(cur[3], ceil[3])]
            )
            boxes[m][k] = cur
            prev = cur

    # --- published-protocol regression gate: S_W = max_k mean_50 width_k e^{-k/6} (no tail)
    gate = {}
    for m in METHODS:
        widths = np.array(
            [
                np.mean(A50 @ [raw[m][k][1] - raw[m][k][0], raw[m][k][3] - raw[m][k][2]])
                for k in range(1, KMAX + 1)
            ]
        )
        s_w = float(np.max(widths * np.exp(-np.arange(1, KMAX + 1) / 6.0)))
        s_pub = float(
            max(
                stored[m]["avg_V_k"][i] * np.exp(-stored[m]["k_values"][i] / 6.0)
                for i in range(len(stored[m]["k_values"]))
            )
        )
        gate[m] = dict(S_W_reconstructed=s_w, S_W_published=s_pub, ratio=s_w / s_pub)

    # --- cross-radius release, per-epsilon beta, per-point noise on the 2000-pt test set
    rng = np.random.default_rng(SEED)
    _, ceil_w = cross_radius_profile(boxes["ibp"], ceil, nominal, xte)
    results = {
        "epsilons": EPS,
        "delta": DELTA,
        "nominal_mse": float(np.mean((nominal[0] * xte + nominal[1] - yte) ** 2)),
        "ceiling_gs_mean": float(ceil_w.mean()),
        "methods": {},
        "gs_baselines": {},
    }
    for mode in ("pure", "approx"):
        results["gs_baselines"][mode] = {
            "GS=10 (published)": {
                k: v.tolist()
                for k, v in gs_curves(nominal, xte, yte, 10.0, mode, np.random.default_rng(SEED)).items()
            },
            "GS=clipping ceiling (per x)": {
                k: v.tolist()
                for k, v in gs_curves(nominal, xte, yte, ceil_w, mode, np.random.default_rng(SEED)).items()
            },
        }
    for m in METHODS:
        A, ceil_w = cross_radius_profile(boxes[m], ceil, nominal, xte)
        entry = {"A_mean_over_test_by_k": A.mean(axis=1).tolist()}
        modes = (
            ("pure", "pure", lambda e: e / 6.0),
            ("approx", "approx", lambda e: e / (2 * np.log(2 / DELTA))),
            ("pure_liang_b6", "pure_liang", lambda e: e / 6.0),
            ("pure_liang_b2", "pure_liang", lambda e: e / 2.0),
        )
        for name, mode, beta_of in modes:
            ss_by_eps = {e: smooth_sensitivity(A, ceil_w, beta_of(e)) for e in EPS}
            curves = mse_curves(
                nominal,
                xte,
                yte,
                ss_by_eps,
                mode,
                np.random.default_rng(SEED),
                beta_by_eps={e: beta_of(e) for e in EPS},
            )
            entry[name] = {
                "beta": [beta_of(e) for e in EPS],
                "SS_mean": [float(ss_by_eps[e].mean()) for e in EPS],
                "SS_max": [float(ss_by_eps[e].max()) for e in EPS],
                **{k: v.tolist() for k, v in curves.items()},
            }
        results["methods"][m] = entry

    json.dump(
        {k: v for k, v in provenance.items()}, open(f"{OUT}/provenance.json", "w"), indent=1, default=float
    )
    json.dump(
        {m: {k: v.tolist() for k, v in boxes[m].items()} for m in METHODS},
        open(f"{OUT}/boxes_hull.json", "w"),
        indent=1,
    )
    json.dump(
        {m: {k: v.tolist() for k, v in raw[m].items()} for m in METHODS},
        open(f"{OUT}/boxes_raw.json", "w"),
        indent=1,
    )
    json.dump(gate, open(f"{OUT}/regression_gate.json", "w"), indent=1)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

    # --- console summary
    print(
        f"nominal (certified): w={nominal[0]:.12f} b={nominal[1]:.12f}   test MSE {results['nominal_mse']:.5f}"
    )
    print(f"ceiling box {ceil.round(6).tolist()}   mean ceiling output width on test set {ceil_w.mean():.4f}")
    print(
        "validation: max width excess of reconstructed vs stored, per method:",
        {m: f"{v:.1e}" for m, v in worst.items()},
    )
    print("regression gate (published S_W, beta=1/6, no tail, 50-pt mean widths):")
    for m in METHODS:
        print(
            f"   {m:4s} reconstructed {gate[m]['S_W_reconstructed']:.5f}  published {gate[m]['S_W_published']:.5f}  ratio {gate[m]['ratio']:.6f}"
        )
    for mode in ("pure", "pure_liang_b6", "pure_liang_b2", "approx"):
        for metric, label in (
            ("mse", "MSE (median over trials)"),
            ("med_se", "median-over-points SE"),
            ("q90_abs", "90% |error| quantile"),
        ):
            if mode.startswith("pure_liang") and metric != "q90_abs":
                continue
            print(f"\n== {mode} | {label} ==")
            print("eps          " + " ".join(f"{e:>9g}" for e in EPS))
            for m in METHODS:
                r = results["methods"][m][mode]
                print(f"{m:12s} " + " ".join(f"{v:9.3g}" for v in r[metric]))
            for name, c in results["gs_baselines"]["pure" if mode.startswith("pure") else mode].items():
                print(f"{name[:12]:12s} " + " ".join(f"{v:9.3g}" for v in c[metric]))
        print(f"\n   {mode}: mean SS over test points")
        for m in METHODS:
            print(f"{m:12s} " + " ".join(f"{v:9.4f}" for v in results["methods"][m][mode]["SS_mean"]))
    print(f"\nwritten to {OUT}/")


if __name__ == "__main__":
    main()
