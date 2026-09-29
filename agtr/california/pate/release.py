"""Four-arm PATE-R vs AGT-R release on California Housing from the shard cells of train.py.

Arms (all (eps,0)-DP under substitution on the 16 512-point training split; approx-DP versions too):
  PATE-R        mean of the T nominal teachers + Laplace(GS / (T eps))            [Gaussian for approx]
  PATE-AGT-R    same mean + Cauchy calibrated to SS_g(x) = (1/T) max_i SS_i(x)    [Laplace(2 SS_g / eps) approx]
  AGT-R full    the full-batch teacher of relu_h64_clamp_gamma0.1 (bs = 16 512, same schedule) = T = 1
  AGT-R WOR     shard 0 of the T-partition with the amplified base budget eps_b = log(1 + T (e^eps - 1)), delta_b = T delta

Per teacher the cells already hold the output boxes (y_l, y_u) and the nominal y_n on the test split for every k,
so the certificate is the cross-radius on those (as in release.py): A(x,k) = max(u_k - l_{k+1}, u_{k+1} - l_k)
with k = 0 the nominal, boxes clipped to [Y_MIN, Y_MAX] and hulled for nesting, dense k asserted, GS tail beyond
k_max.  Everything in real target units (x y_std); the released prediction is clipped to the output range before
the mean absolute error (post-processing).  Pure DP = tighter Cauchy at public beta = eps/2 (Def 3.7 and eps/6 kept).

    PYTHONPATH=$PWD python -m agtr.california.pate.release [--tag pate_relu_h64_clamp_gamma0.1] [--plot-only] [--no-gs]
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import numpy as np

from common import certificate as cert

from agtr.california.release import ROOT, STORED, GS, EPS, DELTA, Y_MIN, Y_MAX
from agtr.linear.pate.release import (
    arms_from_ss,
)

T1_DIR = f"{ROOT}/relu_h64_clamp_gamma0.1"  # full-batch cells bs16512_k*.npz are the T = 1 arm
FIG = "agtr/california/figures/pate_matched"
N_TRIALS = 200


def load_teacher_cells(files):
    """files: {k: path}. Returns (y_n, ks, boxes[(l,u)]) with nesting hull, all in standardised units, clipped."""
    ks = sorted(files)
    assert ks == list(range(1, ks[-1] + 1)), f"k grid {ks} not dense"
    yn = None
    boxes = []
    for k in ks:
        z = np.load(files[k])
        if yn is None:
            yn = np.clip(z["y_n"], Y_MIN, Y_MAX)
        else:
            assert np.allclose(z["y_n"], np.load(files[ks[0]])["y_n"], atol=1e-10), "nominal differs across k"
        l, u = np.clip(z["y_l"], Y_MIN, Y_MAX), np.clip(z["y_u"], Y_MIN, Y_MAX)
        if boxes:
            l = np.minimum(l, boxes[-1][0])
            u = np.maximum(u, boxes[-1][1])
        boxes.append((l, u))
    return yn, ks, boxes


def certificate(yn, boxes):
    """A(x, k), k = 0..k_max, shape (n_test, k_max + 1), standardised units, capped at GS."""
    tail = (np.full_like(yn, Y_MIN), np.full_like(yn, Y_MAX))
    return cert.cross_radius(yn, {k + 1: b for k, b in enumerate(boxes)}, tail=tail, cap=GS).T


def smooth_sensitivity(A, beta):
    return cert.smooth_sensitivity(A.T, beta, GS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pate_relu_h64_clamp_gamma0.1")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-gs", action="store_true")
    a = ap.parse_args()
    d = f"{ROOT}/{a.tag}"
    os.makedirs(FIG, exist_ok=True)
    if a.plot_only:
        make_figures(json.load(open(f"{d}/results.json")), a.tag, show_gs=not a.no_gs)
        return
    y_std = float(np.load(f"{STORED}/y_std.npy"))
    y_mean = float(np.load(f"{STORED}/y_mean.npy"))
    y_test = np.load(f"{STORED}/y_test.npy").ravel()
    y_real = y_test * y_std + y_mean
    gs_real = GS * y_std
    clip = (Y_MIN * y_std + y_mean - y_real, Y_MAX * y_std + y_mean - y_real)
    # T = 1 arm from the full-batch cells
    t1 = {
        int(re.search(r"_k(\d+)\.npz", f).group(1)): f
        for f in glob.glob(f"{T1_DIR}/bs16512_k*.npz")
        if not f.endswith("_k0.npz")
    }
    teachers = {1: [load_teacher_cells(t1)]}
    for Td in sorted(glob.glob(f"{d}/T*"), key=lambda p: int(p.split("/T")[-1])):
        T = int(Td.split("/T")[-1])
        per = {}
        for f in glob.glob(f"{Td}/shard*_k*.npz"):
            m = re.search(r"shard(\d+)_k(\d+)\.npz", f)
            per.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
        if sorted(per) != list(range(T)) or any(max(v) < 28 or len(v) < 28 for v in per.values()):
            print(
                f"T={T}: battery incomplete ({len(per)} shards, min k_max {min((max(v) for v in per.values()), default=0)}); skipped"
            )
            continue
        teachers[T] = [load_teacher_cells(per[i]) for i in range(T)]
    R = {"tag": a.tag, "epsilons": EPS, "delta": DELTA, "n_trials": N_TRIALS, "gs": gs_real, "T": {}}
    import agtr.linear.pate.release as L

    for T, tl in sorted(teachers.items()):
        preds = np.stack([t[0] for t in tl]) * y_std + y_mean  # real units, clipped nominal per teacher
        A_list = [certificate(t[0], t[2]) for t in tl]
        mean_pred = preds.mean(axis=0)
        resid_pate = mean_pred - y_real
        resid_t0 = preds[0] - y_real
        ss_pate = lambda beta: (
            np.max(np.stack([smooth_sensitivity(A, beta) for A in A_list]), axis=0) * y_std / T
        )
        ss_t0 = lambda beta: smooth_sensitivity(A_list[0], beta) * y_std
        entry = {
            "m": int(16512 // T),
            "k_max": int(min(A.shape[1] for A in A_list) - 1),
            "nominal_mse_pate": float(np.mean(resid_pate**2)),
            "nominal_mae_pate": float(np.mean(np.abs(resid_pate))),
            "nominal_q90_pate": float(np.quantile(np.abs(resid_pate), 0.9)),
            "nominal_mse_teacher0": float(np.mean(resid_t0**2)),
            "A_mean_by_k_teacher_max": [
                float(v) for v in np.max(np.stack([A[:, :29].mean(axis=0) for A in A_list]), axis=0)
            ],
            "pate": arms_from_ss(resid_pate, ss_pate, gs_real / T, clip=clip, n_trials=N_TRIALS),
            "wor": arms_from_ss(
                resid_t0,
                ss_t0,
                gs_real,
                eps_map=lambda e: float(np.log1p(T * np.expm1(e))),
                delta_scale=T,
                clip=clip,
                n_trials=N_TRIALS,
            ),
        }
        if T == 1:
            entry["full"] = entry["wor"]
        R["T"][T] = entry
        p = entry["pate"]
        print(
            f"T={T:3d} m={entry['m']:5d} nominal MSE {entry['nominal_mse_pate']:.3f} MAE {entry['nominal_mae_pate']:.3f} | pure b2 MAE @eps=1: "
            f"PATE-R {p['gs_pure']['mae'][3]:.3f} PATE-AGT-R {p['pure_liang_b2']['mae'][3]:.3f} WOR {entry['wor']['pure_liang_b2']['mae'][3]:.3f} | "
            f"approx MSE @eps=1: GS {p['gs_approx']['mse'][3]:.3f} AGT {p['approx']['mse'][3]:.3f}",
            flush=True,
        )
    json.dump(R, open(f"{d}/results.json", "w"), indent=1)
    L.summarise(json.loads(json.dumps(R)))
    make_figures(R, a.tag, show_gs=not a.no_gs)


def make_figures(R, tag, show_gs=True):
    """Reuse the linear figure code (same JSON layout); write under the California figure directory."""
    import agtr.linear.pate.release as L

    old = L.FIG
    L.FIG = FIG
    try:
        L.make_figures(R, tag, show_gs=show_gs)
    finally:
        L.FIG = old


if __name__ == "__main__":
    main()
