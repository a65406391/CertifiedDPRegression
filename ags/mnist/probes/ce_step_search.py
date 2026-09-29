# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Find a CE setting at b = 2000 whose accuracy still climbs at n_s = 3, 4, 5.

Motivation. The wide-lr sweep found cross-entropy peaking at n_s = 2 (lr = 25) and
then FALLING at n_s = 3 before recovering at 5. That non-monotonicity is an
overshooting trajectory, not a property of the loss: at lr = 25 two steps land near
the optimum and the third jumps past it. A lower learning rate should trade peak
two-step accuracy for a trajectory that is still improving at five steps.

Why five matters. AGS's concretization frequency F divides the trajectory, so a
5-step run admits F in {1, 5} -- the two rungs the blobs study reports -- with F = 5
wrapping the whole trajectory in a single MIP. A setting whose best accuracy sits at
n_s = 5 therefore lines this study's F-ladder up with the blobs one exactly.

Scoring. A config is only useful here if it is still climbing, so configs are ranked
by accuracy at n_s = 5 and separately reported on whether acc(5) is their own
maximum. Every cell is averaged over `--seeds` data draws, because a single draw at
N = 2000 is exactly how the California setting ended up seed-fragile (status.md 3).

    python -m ags.mnist.ce_step_search
    python -m ags.mnist.ce_step_search --N 10000 --b 2000
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import load_mnist_pca, accuracy, N_CLASSES
from ags.mnist.probes.nominal_sweep import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--loss", default="cross_entropy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--target", type=int, default=5, help="step count to optimise for")
    ap.add_argument("--out", default="ags/mnist/runs/ce_step_search.json")
    a = ap.parse_args()

    LRS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0]
    GAMS = [0.05, 0.1, 0.15, 0.25, 0.5, 1.0]

    data = [load_mnist_pca(a.N, a.d, seed=s) for s in range(a.seeds)]
    print(
        f"MNIST PCA-{a.d}, 10 classes.  loss={a.loss}  N={a.N} b={a.b} "
        f"P={N_CLASSES * a.d + N_CLASSES}  seeds={a.seeds}"
    )
    print(
        f"grid: lr={LRS}\n      gamma={GAMS}  steps={a.steps}  "
        f"= {len(LRS) * len(GAMS) * len(a.steps) * a.seeds} trainings\n",
        flush=True,
    )

    t0, cfgs = time.time(), []
    for lr in LRS:
        for gamma in GAMS:
            curve = []
            for n_s in a.steps:
                accs = [
                    accuracy(train(X, y, loss=a.loss, n_s=n_s, lr=lr, gamma=gamma, b=a.b), Xte, yte)
                    for X, y, Xte, yte in data
                ]
                curve.append(float(np.mean(accs)))
            i_t = a.steps.index(a.target)
            cfgs.append(
                dict(
                    lr=lr,
                    gamma=gamma,
                    steps=a.steps,
                    curve=curve,
                    acc_target=curve[i_t],
                    argmax_step=a.steps[int(np.argmax(curve))],
                    climbing=bool(curve[i_t] >= max(curve) - 1e-12),
                    monotone_to_target=bool(all(curve[i] <= curve[i + 1] + 1e-12 for i in range(i_t))),
                )
            )
    json.dump(
        dict(
            N=a.N,
            d=a.d,
            b=a.b,
            loss=a.loss,
            seeds=a.seeds,
            steps=a.steps,
            target=a.target,
            lrs=LRS,
            gammas=GAMS,
            wall_s=time.time() - t0,
            configs=cfgs,
        ),
        open(a.out, "w"),
        indent=1,
    )

    hdr = "".join(f"{s:>8}" for s in a.steps)
    print(f"--- top 12 by accuracy at n_s = {a.target} ---")
    print(f"{'lr':>7}{'gamma':>7}{hdr}{'  argmax':>9}{'  mono':>7}")
    for c in sorted(cfgs, key=lambda c: -c["acc_target"])[:12]:
        print(
            f"{c['lr']:>7g}{c['gamma']:>7g}"
            + "".join(f"{v:>8.4f}" for v in c["curve"])
            + f"{c['argmax_step']:>9}{'  yes' if c['monotone_to_target'] else '   no':>7}"
        )

    climb = [c for c in cfgs if c["argmax_step"] >= a.target and c["monotone_to_target"]]
    print(f"\n--- top 8 that peak at n_s >= {a.target} AND climb monotonically to it ---")
    print(f"{'lr':>7}{'gamma':>7}{hdr}{'  argmax':>9}")
    for c in sorted(climb, key=lambda c: -c["acc_target"])[:8]:
        print(
            f"{c['lr']:>7g}{c['gamma']:>7g}"
            + "".join(f"{v:>8.4f}" for v in c["curve"])
            + f"{c['argmax_step']:>9}"
        )
    if not climb:
        print("  none -- every config has already turned over by the target")

    print(f"\n--- best accuracy attainable at each n_s, over the whole grid ---")
    print(f"{'n_s':>5}{'acc':>9}{'lr':>8}{'gamma':>8}")
    for i, n_s in enumerate(a.steps):
        b = max(cfgs, key=lambda c: c["curve"][i])
        print(f"{n_s:>5}{b['curve'][i]:>9.4f}{b['lr']:>8g}{b['gamma']:>8g}")
    print(f"\nwrote {a.out}   [{time.time() - t0:.0f}s]")
