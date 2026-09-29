# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""The loss choice is a trade-off, so measure both halves of it.

nominal_sweep.py answers "which loss trains best in few steps". That is only half the
question: AGS's cost is the certified envelope, and the two losses do not produce the
same one. Hinge has EXACTLY zero per-sample gradient variation wherever every margin
stays clear across the parameter box (ABLATION_blobs.md 1(iii)), so its envelope can
sit exactly on the quiescent bound; cross-entropy gradients vary everywhere, so its
envelope is strictly wider. The same exact-quiescence property is why hinge barely
moves in the first few steps from a zero init -- at theta = 0 every logit is 0, every
margin is active, and the gradient is constant until the trajectory clears the margin.

This module measures, per (loss, n_s), at near-full batch:

    nominal test accuracy      what the trajectory buys
    IBP width at k = 1         what the certificate costs
    GS = IBP width at k = b    the computed global sensitivity (ags.global_sensitivity)
    width(k=1) / GS            the fraction of GS the certificate saves

    python -m ags.mnist.loss_tradeoff
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import load_mnist_pca, logistic_model, accuracy, N_CLASSES
from ags.mnist.ags import agt_bound, global_sensitivity, clipping_ceiling, A_of_k
from ags.mnist import ss_reference as R

CERT_LOSSES = ["cross_entropy", "hinge_ova"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--b", type=int, default=0, help="0 = full batch")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--lr", type=float, default=25.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--out", default="ags/mnist/runs/loss_tradeoff.json")
    a = ap.parse_args()

    X, y, Xte, yte = load_mnist_pca(a.N, a.d, seed=0)
    b = a.b or a.N
    P = N_CLASSES * a.d + N_CLASSES
    print(f"MNIST PCA-{a.d}, 10 classes.  N={a.N} b={b} P={P} lr={a.lr} gamma={a.gamma}")
    print("GS is COMPUTED as the AGT envelope at k = b, never as a formula.\n")
    print(
        f"{'loss':<14}{'n_s':>5}{'test':>8}{'width(k=1)':>13}{'GS(k=b)':>12}"
        f"{'w1/GS':>9}{'ceiling':>10}{'GS/ceil':>9}{'A^1':>11}{'t(s)':>8}"
    )

    rows = []
    for loss in CERT_LOSSES:
        for n_s in a.steps:
            t0 = time.time()
            # n_epochs == n_s at full batch (one step per epoch)
            ep = max(1, n_s // max(1, a.N // b))
            pl, pn, pu = agt_bound(
                logistic_model(a.d),
                X,
                y,
                k=1,
                b=b,
                n_epochs=ep,
                lr=a.lr,
                gamma=a.gamma,
                l2_reg=0.0,
                loss=loss,
            )
            gs_w, gs_r, *_ = global_sensitivity(
                X, y, b=b, n_epochs=ep, lr=a.lr, gamma=a.gamma, loss=loss, d=a.d
            )
            w1 = np.abs(pu - pl).mean()
            gs = gs_w.mean()
            ceil = clipping_ceiling(n_s, a.lr, a.gamma)
            acc = accuracy(pn, Xte, yte)
            rows.append(
                dict(
                    loss=loss,
                    n_s=n_s,
                    n_epochs=ep,
                    b=b,
                    lr=a.lr,
                    gamma=a.gamma,
                    test_acc=acc,
                    width_k1_mean=float(w1),
                    gs_computed_mean=float(gs),
                    w1_over_gs=float(w1 / gs),
                    clipping_ceiling=ceil,
                    gs_over_ceiling=float(gs / ceil),
                    A1=A_of_k(pl, pn, pu),
                    closed_form_per_coord=float(R.gs_window(a.lr * n_s, a.gamma, b, P)[0]),
                )
            )
            r = rows[-1]
            print(
                f"{loss:<14}{n_s:>5}{acc:>8.4f}{w1:>13.5g}{gs:>12.5g}"
                f"{r['w1_over_gs']:>9.4f}{ceil:>10.4g}{r['gs_over_ceiling']:>9.4f}"
                f"{r['A1']:>11.4g}{time.time() - t0:>8.0f}",
                flush=True,
            )

    json.dump(
        dict(N=a.N, d=a.d, b=b, P=P, lr=a.lr, gamma=a.gamma, steps=a.steps, losses=CERT_LOSSES, rows=rows),
        open(a.out, "w"),
        indent=1,
    )
    print(f"\nwrote {a.out}")
