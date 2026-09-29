"""AGS-IBP + matched DP-SGD across batch size at the headline config (lambda = 0).

Port of AGS_Ablation/blobs/ibp_sweep.py. Data and model change; nothing else does.

Inference is the reference procedure (ss_reference, a port of experiments/common/ags.py).
Stores per-coordinate param_l/param_u for every k so downstream consumers never have to
reconstruct the envelope -- only the WIDTH matters to the reference, and the l1 radius
about the nominal is a different quantity that must not be substituted for it.

GS is COMPUTED, at k = b, and recorded per cell as `gs_computed_*`. The reference's
closed-form `gs_window` expression is recorded separately as `closed_form_per_coord`
and is never called GS. The two differ by a factor of b here.

AGS and DP-SGD privatize THE SAME algorithm at each b: identical b, epochs, lr, clip,
lambda. Sweeping b sweeps the algorithm, not the privacy mechanism.
"""

import json
import time
import numpy as np
from ags.mnist.shared import load_mnist_pca, logistic_model, reference_accuracy
from ags.mnist.ags import agt_bound, A_of_k, global_sensitivity, clipping_ceiling
from ags.mnist import ss_reference as R
from ags.mnist import baselines as B

N, d, DIGITS, EPOCHS, GAM, LAM, SEEDS = 200, 64, (3, 8), 20, 0.2, 0.0, 41
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
KG = [1, 2, 3, 4, 8]
BATCHES = [5, 10, 20, 25, 50, 100, 200]
NP_ = d + 1

if __name__ == "__main__":
    X, ys, Xte, yts = load_mnist_pca(N, d, digits=DIGITS, seed=0)
    P, Pte = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / N))
    ceiling = reference_accuracy(X, ys, Xte, yts)
    print(
        f"N={N} d={d} digits={DIGITS} lr={lr:g} gamma={GAM} lam={LAM} "
        f"epochs={EPOCHS} k={KG}\nunclipped logistic ceiling = {ceiling:.4f}\n",
        flush=True,
    )
    cells = []
    for b in BATCHES:
        ns = EPOCHS * max(1, N // b)
        closed = R.gs_window(lr * ns, GAM, b, NP_)  # reference expression only
        gs_w, gs_r, *_ = global_sensitivity(
            X, ys, b=b, n_epochs=EPOCHS, lr=lr, gamma=GAM, d=d
        )  # GS: AGT at k = b
        t0, ls_by_k, widths, Ag, th = time.time(), {}, {}, [], None
        for k in KG:
            pl, pn, pu = agt_bound(
                logistic_model(d), X, ys, k=k, b=b, n_epochs=EPOCHS, lr=lr, gamma=GAM, l2_reg=LAM
            )
            ls_by_k[k] = R.local_sens(pl, pu, closed, k)
            widths[k] = np.abs(pu - pl).tolist()
            Ag.append(A_of_k(pl, pn, pu))
            if k == 1:
                th = pn
        nom = B.acc(th, Pte, yts)
        w1 = np.mean(widths[1])
        rec = dict(
            b=b,
            n_s=ns,
            k_grid=KG,
            A_grid_l1_radius=Ag,
            gs_computed_width_mean=float(gs_w.mean()),
            gs_computed_width=gs_w.tolist(),
            gs_computed_radius_mean=float(gs_r.mean()),
            closed_form_per_coord=float(closed[0]),
            clipping_ceiling=clipping_ceiling(ns, lr, GAM),
            widths=widths,
            nominal=nom,
            theta_n=th.tolist(),
            rows=[],
        )
        print(
            f"b={b:<4} n_s={ns:<5} GS(k=b)={gs_w.mean():.4g}  width(k=1)={w1:.4g}"
            f"  w1/GS={w1 / gs_w.mean():.3f}  closed_form={closed[0]:.4g}"
            f"  nominal={nom:.4f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        for eps in EPS:
            ss = R.smooth_sensitivity(ls_by_k, KG, closed, R.beta_for(eps), NP_)
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
            digits=list(DIGITS),
            epochs=EPOCHS,
            gamma=GAM,
            lam=LAM,
            lr=lr,
            seeds=SEEDS,
            eps=EPS,
            k_grid=KG,
            inference="ss_reference",
            loss="binary_cross_entropy",
            ceiling=ceiling,
            cells=cells,
        ),
        open("ags/mnist/runs/ibp_sweep_ref.json", "w"),
        indent=1,
    )
    print("\nwrote ags/mnist/runs/ibp_sweep_ref.json")
