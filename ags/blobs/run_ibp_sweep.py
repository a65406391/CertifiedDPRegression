"""AGS-IBP + matched DP-SGD across batch size at the headline config (lambda = 0).

Inference is the reference procedure (ss_reference, a port of experiments/common/ags.py).
Stores per-coordinate param_l/param_u for every k so downstream consumers never have to
reconstruct the envelope -- only the WIDTH matters to the reference, and the l1 radius
about the nominal is a different quantity that must not be substituted for it.

AGS and DP-SGD privatize THE SAME algorithm at each b: identical b, epochs, lr, clip,
lambda. Sweeping b sweeps the algorithm, not the privacy mechanism.
"""

import json
import time
import numpy as np
from ags.blobs.shared import load_blobs, linear_model
from ags.blobs.ags import agt_bound, A_of_k
from ags.blobs import ss_reference as R
from ags.blobs import baselines as B

N, d, ANISO, SEP, EPOCHS, GAM, LAM, SEEDS = 200, 8, 5.0, 4.0, 20, 0.05, 0.0, 41
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
KG = [1, 2, 3, 4, 8]
BATCHES = [5, 10, 20, 25, 50, 100, 200]
NP_ = d + 1

if __name__ == "__main__":
    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    P, Pte = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / N))
    print(f"N={N} d={d} lr={lr:g} gamma={GAM} lam={LAM} epochs={EPOCHS} k={KG}\n", flush=True)
    cells = []
    for b in BATCHES:
        ns = EPOCHS * max(1, N // b)
        gs_vec = R.gs_window(lr * ns, GAM, b, NP_)
        t0, ls_by_k, widths, Ag, th = time.time(), {}, {}, [], None
        for k in KG:
            pl, pn, pu = agt_bound(
                linear_model(d), X, ys, k=k, b=b, n_epochs=EPOCHS, lr=lr, gamma=GAM, l2_reg=LAM
            )
            ls_by_k[k] = R.local_sens(pl, pu, gs_vec, k)
            widths[k] = np.abs(pu - pl).tolist()
            Ag.append(A_of_k(pl, pn, pu))
            if k == 1:
                th = pn
        nom = B.acc(th, Pte, yts)
        rec = dict(
            b=b,
            n_s=ns,
            k_grid=KG,
            A_grid_l1_radius=Ag,
            gs_per_coord=float(gs_vec[0]),
            widths=widths,
            nominal=nom,
            theta_n=th.tolist(),
            rows=[],
        )
        print(
            f"b={b:<4} n_s={ns:<5} GS/coord={gs_vec[0]:.4g}  width(k=1)={ls_by_k[1].mean():.4g}"
            f"  nominal={nom:.4f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        for eps in EPS:
            ss = R.smooth_sensitivity(ls_by_k, KG, gs_vec, R.beta_for(eps), NP_)
            sg = R.sigma(ss, eps, NP_, noise_mode="isotropic")
            ags = float(
                np.median(
                    [B.acc(R.release(th, sg, np.random.RandomState(s)), Pte, yts) for s in range(SEEDS)]
                )
            )
            sgd = float(
                np.median(
                    [
                        B.acc(
                            B.dpsgd_pure(P, ys, eps, n_s=ns, lr=lr, gamma=GAM, lam=LAM, seed=s, b=b)[0],
                            Pte,
                            yts,
                        )
                        for s in range(SEEDS)
                    ]
                )
            )
            rec["rows"].append(
                dict(eps=eps, ags=ags, dpsgd=sgd, ss_mean=float(ss.mean()), sigma_mean=float(sg.mean()))
            )
            print(f"    eps={eps:<4g} AGS={ags:.4f}  DP-SGD={sgd:.4f}", flush=True)
        cells.append(rec)
    json.dump(
        dict(
            N=N,
            d=d,
            aniso=ANISO,
            sep=SEP,
            epochs=EPOCHS,
            gamma=GAM,
            lam=LAM,
            lr=lr,
            seeds=SEEDS,
            eps=EPS,
            k_grid=KG,
            inference="ss_reference",
            cells=cells,
        ),
        open("ags/blobs/runs/ibp_sweep_ref.json", "w"),
        indent=1,
    )
    print("\nwrote ags/blobs/runs/ibp_sweep_ref.json")
