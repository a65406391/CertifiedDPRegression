# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Which loss trains best on PCA-64 MNIST in FEW STEPS at NEAR-FULL BATCH?

This is the setting-selection step: AGS releases once at the end of training, and its
certified width grows per step, so the regime that matters is a short trajectory over
a large batch. This sweep measures nominal (non-private, clipped) accuracy for the
three 10-class arms in losses.py across that regime, and reports which are
certifiable by abstract_gradient_training as they stand.

Everything here is element-wise clipped SGD from a zero init -- the exact algorithm
AGS certifies, with no noise. Nothing about privacy enters.

    python -m ags.mnist.nominal_sweep
    python -m ags.mnist.nominal_sweep --N 5000 --quick
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import load_mnist_pca, reference_accuracy, accuracy, logits, N_CLASSES
from ags.mnist.losses import GRADS, CERTIFIABLE

LOSSES = ["cross_entropy", "hinge_ova", "hinge_ww"]


def train(
    X,
    y,
    *,
    loss,
    n_s,
    lr,
    gamma,
    b,
    d=None,
    n_classes=N_CLASSES,
    margin=1.0,
    lam=0.0,
    theta0=None,
    track=None,
):
    """Element-wise clipped SGD, matching AGT's clip_method='clamp'.

    Fixed cyclic batching, exactly as blobs/baselines.py does, so the algorithm AGS
    certifies and the one measured here are the same object.
    """
    n, dd = X.shape
    b = b or n
    P = n_classes * dd + n_classes
    theta = np.zeros(P) if theta0 is None else theta0.copy()
    grad_fn = GRADS[loss]
    spe = max(1, n // b)
    for t in range(n_s):
        sl = slice((t % spe) * b, ((t % spe) + 1) * b)
        Xb, yb = X[sl], y[sl]
        g = grad_fn(logits(theta, Xb, n_classes), yb, Xb, n_classes, margin)
        g = np.clip(g, -gamma, gamma)
        theta = (1.0 - lr * lam) * theta - lr * (g.sum(0) / len(Xb))
        if track is not None:
            track.append(theta.copy())
    return theta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000, help="train size; paper uses 2000")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="coarse grid")
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    ap.add_argument("--gammas", type=float, nargs="+", default=None)
    ap.add_argument("--out", default="ags/mnist/runs/nominal_sweep.json")
    a = ap.parse_args()

    X, y, Xte, yte = load_mnist_pca(a.N, a.d, seed=0, n_test=a.n_test)
    P = N_CLASSES * a.d + N_CLASSES
    reference = reference_accuracy(X, y, Xte, yte)

    # "close to full batch": the whole set, and two divisions of it.
    BS = [a.N, a.N // 2, a.N // 4]
    # "few steps": AGS pays certified width per step, so this is the axis that matters.
    STEPS = a.steps or ([1, 2, 5, 10, 20] if a.quick else [1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    LRS = a.lrs or ([0.5, 2.0, 10.0] if a.quick else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0])
    GAMS = a.gammas or ([0.1, 1.0] if a.quick else [0.01, 0.05, 0.1, 0.5, 1.0])

    print(f"MNIST PCA-{a.d}, 10 classes.  N={a.N} n_test={len(yte)}  P={P}")
    print(
        f"unregularised logistic reference = {reference:.4f}  "
        f"(a reference point, NOT a ceiling -- clipped SGD can beat it)"
    )
    print(
        f"grid: b={BS}  steps={STEPS}  lr={LRS}  gamma={GAMS}  "
        f"= {len(BS) * len(STEPS) * len(LRS) * len(GAMS) * len(LOSSES)} cells\n",
        flush=True,
    )

    rows, t0 = [], time.time()
    for loss in LOSSES:
        cert = CERTIFIABLE.get(loss)
        for b in BS:
            for n_s in STEPS:
                for lr in LRS:
                    for gamma in GAMS:
                        th = train(X, y, loss=loss, n_s=n_s, lr=lr, gamma=gamma, b=b)
                        rows.append(
                            dict(
                                loss=loss,
                                certifiable_as=cert,
                                b=b,
                                n_s=n_s,
                                lr=lr,
                                gamma=gamma,
                                train_acc=accuracy(th, X, y),
                                test_acc=accuracy(th, Xte, yte),
                            )
                        )
        best = max((r for r in rows if r["loss"] == loss), key=lambda r: r["test_acc"])
        print(
            f"{loss:<14} certifiable_as={str(cert):<14} best test={best['test_acc']:.4f}"
            f"  at b={best['b']} n_s={best['n_s']} lr={best['lr']} gamma={best['gamma']}",
            flush=True,
        )

    json.dump(
        dict(
            N=a.N,
            d=a.d,
            n_classes=N_CLASSES,
            P=P,
            n_test=len(yte),
            reference_unregularised=reference,
            batch_sizes=BS,
            steps=STEPS,
            lrs=LRS,
            gammas=GAMS,
            losses=LOSSES,
            certifiable=CERTIFIABLE,
            wall_s=time.time() - t0,
            rows=rows,
        ),
        open(a.out, "w"),
        indent=1,
    )

    print(f"\n--- best test accuracy per (loss, n_s), maximised over b, lr, gamma ---")
    hdr = "  ".join(f"{s:>6}" for s in STEPS)
    print(f"{'loss':<14}{'':<2}{hdr}")
    for loss in LOSSES:
        cells = []
        for n_s in STEPS:
            c = [r for r in rows if r["loss"] == loss and r["n_s"] == n_s]
            cells.append(f"{max(r['test_acc'] for r in c):>6.4f}")
        print(f"{loss:<14}{'':<2}" + "  ".join(cells))

    print(f"\n--- best test accuracy per (loss, b) at n_s <= 10 ---")
    print(f"{'loss':<14}{'':<2}" + "  ".join(f"{b:>7}" for b in BS))
    for loss in LOSSES:
        cells = []
        for b in BS:
            c = [r for r in rows if r["loss"] == loss and r["b"] == b and r["n_s"] <= 10]
            cells.append(f"{max(r['test_acc'] for r in c):>7.4f}")
        print(f"{loss:<14}{'':<2}" + "  ".join(cells))

    print(f"\nwrote {a.out}   [{time.time() - t0:.0f}s]")
