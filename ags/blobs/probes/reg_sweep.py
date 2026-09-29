# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Does L2 regularisation tighten AGS-IBP enough to reach matched-RDP DP-SGD?

AGS at lambda = 0 is clamped at the global sensitivity, so the only lever on the
clamped arms is GS itself. Under L2 the contraction bound replaces a GS that grows
linearly in n_s with one that saturates -- see ss_reference.gs_window_l2.

IBP only: the MILP path has no regularisation field in OPTConfig.
"""

import json
import time
import numpy as np
from ags.blobs.shared import load_blobs, linear_model
from ags.blobs.ags import agt_bound
from ags.blobs import ss_reference as R
from ags.blobs import baselines as B

N, d, ANISO, SEP, EPOCHS, GAM, LR, SEEDS, B_ = 200, 8, 5.0, 4.0, 20, 0.05, 0.5, 41, 20
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
KG = [1, 2, 3, 4, 8]
LAMS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
DELTA, NP_ = 1e-5, d + 1

if __name__ == "__main__":
    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    P, Pte = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))]
    n_s = EPOCHS * (N // B_)
    print(f"b={B_} n_s={n_s} lr={LR} gamma={GAM}  IBP, k={KG}\n", flush=True)
    print(
        f"{'lam':>6}{'GS_lam':>10}{'w(k=1)':>9}{'clamp':>7}{'nom':>8}  "
        f"{'PURE  ':>6}" + "".join(f"{e:>7}" for e in EPS),
        flush=True,
    )
    cells = []
    for lam in LAMS:
        t0 = time.time()
        gs = R.gs_window_l2(LR, n_s, GAM, B_, lam, NP_)
        ls, th = {}, None
        for k in KG:
            pl, pn, pu = agt_bound(
                linear_model(d), X, ys, k=k, b=B_, n_epochs=EPOCHS, lr=LR, gamma=GAM, l2_reg=lam
            )
            ls[k] = R.local_sens(pl, pu, gs, k)
            if k == 1:
                th, w1 = pn, float(np.abs(pu - pl).mean())
        nom = B.acc(th, Pte, yts)
        rows = {}
        for mode in ("pure", "approx"):
            acc = []
            for e in EPS:
                beta = (
                    R.beta_for(e)
                    if mode == "pure"
                    else R.beta_for(e, mode="approx", n_params=NP_, delta=DELTA)
                )
                ss = R.smooth_sensitivity(ls, KG, gs, beta, NP_)
                sg = R.sigma(ss, e, NP_, noise_mode="isotropic", mode=mode)
                acc.append(
                    float(
                        np.median(
                            [
                                B.acc(R.release(th, sg, np.random.RandomState(s), mode=mode), Pte, yts)
                                for s in range(SEEDS)
                            ]
                        )
                    )
                )
            rows[mode] = acc
        cl = "yes" if w1 >= gs[0] - 1e-12 else "no"
        print(
            f"{lam:>6}{gs[0]:10.5f}{w1:9.5f}{cl:>7}{nom:8.4f}  {'':>6}"
            + "".join(f"{v:7.4f}" for v in rows["pure"])
            + f"   [{time.time() - t0:.0f}s]",
            flush=True,
        )
        print(f"{'':>32}{'':>8}  {'APPROX':>6}" + "".join(f"{v:7.4f}" for v in rows["approx"]), flush=True)
        cells.append(
            dict(
                lam=lam,
                gs=float(gs[0]),
                w1=w1,
                clamped=(cl == "yes"),
                nominal=nom,
                pure=rows["pure"],
                approx=rows["approx"],
            )
        )
    json.dump(
        dict(eps=EPS, b=B_, n_s=n_s, lr=LR, gamma=GAM, delta=DELTA, k_grid=KG, cells=cells),
        open("ags/blobs/runs/reg_sweep.json", "w"),
        indent=1,
    )
    print("\nwrote ags/blobs/runs/reg_sweep.json")
