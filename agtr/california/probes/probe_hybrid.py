# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Probe: PATE with a random subset of T' of the T teachers (amplified aggregate), from the existing shard cells.

Mechanism M_S(D): draw S, |S| = T', uniformly (secret); release mean_{i in S} f_i(x) + noise.  For fixed S the
record's shard is in S with probability q = T'/T; if not, M_S(D) = M_S(D'); if yes, the released statistic is
the PATE-(AGT-)R mechanism on the selected shards, made eps_b-DP with its own (smooth) sensitivity
GS/T' resp. max_{i in S} SS_i / T'.  Fixed-size sampling without replacement of shards then gives
eps = log(1 + q (e^{eps_b} - 1)) (Balle et al. Thm 9), i.e. eps_b = log(1 + (e^eps - 1)/q).
Both PATE-R and PATE-AGT-R get the same treatment; T' = T is the un-amplified mechanism.

    PYTHONPATH=$PWD python -m agtr.california.pate.probe_hybrid
"""

import glob
import json
import re
import numpy as np
from agtr.california.release import ROOT, STORED, GS, EPS, Y_MIN, Y_MAX
from agtr.california.pate.release import (
    load_teacher_cells,
    certificate,
    smooth_sensitivity,
)
import agtr.linear.pate.release as L

N_DRAWS, N_TRIALS = 6, 60
TAG = "pate_relu_h64_clamp_gamma0.1"


def main():
    y_std = float(np.load(f"{STORED}/y_std.npy"))
    y_mean = float(np.load(f"{STORED}/y_mean.npy"))
    y_real = np.load(f"{STORED}/y_test.npy").ravel() * y_std + y_mean
    gs_real = GS * y_std
    clip = (Y_MIN * y_std + y_mean - y_real, Y_MAX * y_std + y_mean - y_real)
    out = {"epsilons": EPS, "n_draws": N_DRAWS, "n_trials": N_TRIALS, "T": {}}
    rng = np.random.default_rng(0)
    for T in (8, 16, 32, 64):
        per = {}
        for f in glob.glob(f"{ROOT}/{TAG}/T{T}/shard*_k*.npz"):
            m = re.search(r"shard(\d+)_k(\d+)\.npz", f)
            per.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
        tl = [load_teacher_cells(per[i]) for i in range(T)]
        preds = np.stack([t[0] for t in tl]) * y_std + y_mean
        A = [certificate(t[0], t[2]) for t in tl]
        res = {}
        for Tp in [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= T]:
            q = Tp / T
            eps_map = lambda e, q=q: float(np.log1p(np.expm1(e) / q))
            draws = [np.sort(rng.choice(T, Tp, replace=False)) for _ in range(N_DRAWS if Tp < T else 1)]
            acc = {}
            for S in draws:
                resid = preds[S].mean(axis=0) - y_real
                ss = lambda beta, S=S: (
                    np.max(np.stack([smooth_sensitivity(A[i], beta) for i in S]), axis=0) * y_std / Tp
                )
                r = L.arms_from_ss(
                    resid,
                    ss,
                    gs_real / Tp,
                    n_trials=N_TRIALS,
                    eps_map=eps_map,
                    delta_scale=1.0 / q,
                    clip=clip,
                )
                for arm, key in (
                    ("agt", "pure_liang_b2"),
                    ("gs", "gs_pure"),
                    ("agt_approx", "approx"),
                    ("gs_approx", "gs_approx"),
                ):
                    met = "mse" if "approx" in arm else "mae"
                    acc.setdefault(arm, []).append(r[key][met])
            res[Tp] = {arm: np.mean(np.array(v), axis=0).tolist() for arm, v in acc.items()}
            res[Tp]["eps_b"] = [eps_map(e) for e in EPS]
            res[Tp]["nominal_mae"] = float(
                np.mean([np.mean(np.abs(preds[S].mean(axis=0) - y_real)) for S in draws])
            )
            print(
                f"T={T:2d} T'={Tp:2d} q={q:.3f} eps_b@0.5={eps_map(0.5):.2f} | pure MAE eps=0.2/0.5/1: AGT {res[Tp]['agt'][1]:.3f}/{res[Tp]['agt'][2]:.3f}/{res[Tp]['agt'][3]:.3f}  "
                f"PATE-R {res[Tp]['gs'][1]:.3f}/{res[Tp]['gs'][2]:.3f}/{res[Tp]['gs'][3]:.3f} | approx MSE eps=0.5: AGT {res[Tp]['agt_approx'][2]:.2f} GS {res[Tp]['gs_approx'][2]:.2f}",
                flush=True,
            )
        out["T"][T] = res
    json.dump(out, open(f"{ROOT}/{TAG}/probe_hybrid.json", "w"), indent=1)
    print(
        "\nBest over (T, T') per eps -- pure MAE:  hybrid PATE-AGT-R | hybrid PATE-R | un-amplified PATE-R (T'=T)"
    )
    for j, e in enumerate(EPS):
        ba = min((out["T"][T][Tp]["agt"][j], T, Tp) for T in out["T"] for Tp in out["T"][T])
        bg = min((out["T"][T][Tp]["gs"][j], T, Tp) for T in out["T"] for Tp in out["T"][T])
        bu = min((out["T"][T][T]["gs"][j], T) for T in out["T"])
        print(
            f"  eps={e:4g}: {ba[0]:.3f} (T={ba[1]},T'={ba[2]}) | {bg[0]:.3f} (T={bg[1]},T'={bg[2]}) | {bu[0]:.3f} (T={bu[1]})"
        )


if __name__ == "__main__":
    main()
