"""Full PATE (m, q) x eps table at the headline config.

Stores every cell so a single configuration can be selected under ANY fixed rule --
tuned at eps=1, or best mean across eps (the rule DP-SGD's single config was chosen
under). Per-eps selection is recorded only as an upper envelope, never as the baseline.
"""

import json
import numpy as np
from ags.blobs.shared import load_blobs
from ags.blobs import baselines as B

N, d, b, EPOCHS, GAM, LAM, SEEDS = 200, 8, 20, 20, 0.05, 0.0, 41
ANISO, SEP = 5.0, 4.0
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100, 200]
M = [2, 3, 5, 10, 20, 40]
Q = [10, 20, 40, 80]

if __name__ == "__main__":
    X, ys, Xte, yts = load_blobs(N, d, sep=SEP, seed=0, aniso=ANISO)
    P, Pte = np.c_[X, np.ones(N)], np.c_[Xte, np.ones(len(Xte))]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / N))
    ns = EPOCHS * (N // b)
    grid = {}
    for m in M:
        for q in Q:
            accs = [
                float(
                    np.median(
                        [
                            B.acc(
                                B.pate(
                                    P,
                                    ys,
                                    Pte,
                                    e,
                                    n_teachers=m,
                                    n_queries=q,
                                    n_s=ns,
                                    lr=lr,
                                    gamma=GAM,
                                    lam=LAM,
                                    seed=s,
                                )[0],
                                Pte,
                                yts,
                            )
                            for s in range(SEEDS)
                        ]
                    )
                )
                for e in EPS
            ]
            grid[f"{m}_{q}"] = accs
            print(
                f"m={m:>3} q={q:>3}  mean={np.mean(accs):.4f}  " + " ".join(f"{a:.3f}" for a in accs),
                flush=True,
            )
    json.dump(
        dict(
            N=N,
            d=d,
            b=b,
            n_s=ns,
            lr=lr,
            gamma=GAM,
            lam=LAM,
            aniso=ANISO,
            sep=SEP,
            seeds=SEEDS,
            eps=EPS,
            M=M,
            Q=Q,
            grid=grid,
        ),
        open("ags/blobs/runs/pate_grid_full.json", "w"),
        indent=1,
    )
    best_mean = max(grid, key=lambda k: np.mean(grid[k]))
    best_e1 = max(grid, key=lambda k: grid[k][0])
    print(f"\nbest by mean-across-eps: m,q = {best_mean}  " + " ".join(f"{a:.3f}" for a in grid[best_mean]))
    print(f"best at eps=1        : m,q = {best_e1}  " + " ".join(f"{a:.3f}" for a in grid[best_e1]))
    print(
        f"per-eps envelope     : "
        + " ".join(f"{max(v[i] for v in grid.values()):.3f}" for i in range(len(EPS)))
    )
