# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Pick the CE setting for a FIVE-STEP run at b = 2000.

Why five steps at b = 2000. AGS's concretization frequency F divides the trajectory,
so n_s = 5 admits F in {1, 5}: concretize after every step, or wrap the whole run in
one MIP window. Those are the two rungs the blobs study reports, so a 5-step setting
lines this study's F-ladder up with that one exactly.

What "works" means here, and it is two conditions, not one:

  1. acc(5) is the maximum over n_s = 1..5. If the trajectory has already turned over
     by step 3, the extra steps buy nothing while still widening the envelope, and
     F = 5 is then certifying a run nobody would choose to make.
  2. acc(5) is as high as possible subject to (1).

The earlier wide-lr sweep peaked at n_s = 2 with lr = 25 and then FELL at n_s = 3.
That is an overshooting trajectory, not a property of the loss: two steps at lr = 25
land near the optimum and the third jumps past it. Condition (1) rules such settings
out no matter how good their peak looks.

N is swept alongside (lr, gamma) because at fixed b it decides what a step IS: at
N = 2000 every step is a full-batch pass, while at N = 10000 five steps is exactly one
epoch. Those are different algorithms and they favour different step counts.

    python -m ags.mnist.ce_five_step
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import load_mnist_pca, accuracy, N_CLASSES
from ags.mnist.probes.nominal_sweep import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--loss", default="cross_entropy")
    ap.add_argument("--Ns", type=int, nargs="+", default=[2000, 4000, 6000, 10000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="ags/mnist/runs/ce_five_step.json")
    a = ap.parse_args()

    LRS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0]
    GAMS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.4]
    target = a.steps[-1]

    print(
        f"MNIST PCA-{a.d}, 10 classes, loss={a.loss}, b={a.b}, "
        f"P={N_CLASSES * a.d + N_CLASSES}, seeds={a.seeds}"
    )
    print(f"condition: acc({target}) must be the max over n_s={a.steps}\n", flush=True)

    t0, out = time.time(), []
    for N in a.Ns:
        data = [load_mnist_pca(N, a.d, seed=s) for s in range(a.seeds)]
        cfgs = []
        for lr in LRS:
            for gamma in GAMS:
                curve, sd = [], []
                for n_s in a.steps:
                    accs = [
                        accuracy(train(X, y, loss=a.loss, n_s=n_s, lr=lr, gamma=gamma, b=a.b), Xte, yte)
                        for X, y, Xte, yte in data
                    ]
                    curve.append(float(np.mean(accs)))
                    sd.append(float(np.std(accs)))
                cfgs.append(
                    dict(
                        N=N,
                        lr=lr,
                        gamma=gamma,
                        curve=curve,
                        sd=sd,
                        acc_target=curve[-1],
                        sd_target=sd[-1],
                        best_is_target=bool(curve[-1] >= max(curve) - 1e-12),
                        monotone=bool(all(curve[i] <= curve[i + 1] + 1e-12 for i in range(len(curve) - 1))),
                    )
                )
        out.extend(cfgs)
        ok = [c for c in cfgs if c["best_is_target"]]
        best = max(ok, key=lambda c: c["acc_target"]) if ok else None
        peak = max(cfgs, key=lambda c: max(c["curve"]))
        print(
            f"N={N:<6} best satisfying the condition: "
            + (
                f"acc({target})={best['acc_target']:.4f} +-{best['sd_target']:.4f} "
                f"at lr={best['lr']:g} gamma={best['gamma']:g}   "
                f"curve=" + " ".join(f"{v:.4f}" for v in best["curve"])
                if best
                else "NONE"
            ),
            flush=True,
        )
        print(
            f"{'':<8}unconstrained peak anywhere in 1..{target}: "
            f"{max(peak['curve']):.4f} at n_s={a.steps[int(np.argmax(peak['curve']))]}, "
            f"lr={peak['lr']:g} gamma={peak['gamma']:g}\n",
            flush=True,
        )

    json.dump(
        dict(
            d=a.d,
            b=a.b,
            loss=a.loss,
            Ns=a.Ns,
            seeds=a.seeds,
            steps=a.steps,
            lrs=LRS,
            gammas=GAMS,
            wall_s=time.time() - t0,
            configs=out,
        ),
        open(a.out, "w"),
        indent=1,
    )

    ok = [c for c in out if c["best_is_target"]]
    print(f"--- top 12 overall, across N, with acc({target}) maximal in 1..{target} ---")
    print(
        f"{'N':>7}{'lr':>7}{'gamma':>7}"
        + "".join(f"{s:>9}" for s in a.steps)
        + f"{'  sd(5)':>9}{'  mono':>7}"
    )
    for c in sorted(ok, key=lambda c: -c["acc_target"])[:12]:
        print(
            f"{c['N']:>7}{c['lr']:>7g}{c['gamma']:>7g}"
            + "".join(f"{v:>9.4f}" for v in c["curve"])
            + f"{c['sd_target']:>9.4f}"
            + f"{'  yes' if c['monotone'] else '   no':>7}"
        )
    print(f"\nwrote {a.out}   [{time.time() - t0:.0f}s]")
