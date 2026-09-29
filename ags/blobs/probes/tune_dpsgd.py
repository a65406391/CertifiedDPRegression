# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""DP-SGD at its own optimum, over the shared grid in tune_grid.py.

Reports DP-SGD only. The comparison against AGS is made afterwards, from tune_ags_a5.py
run over the SAME grid -- crediting AGS with cells it happens to have already run while
DP-SGD searches freely is not a comparison.
"""

import argparse
import json
import numpy as np
from ags.blobs.shared import load_blobs, bayes_accuracy
from ags.blobs.tune_grid import EPS, GAMMA, LAM, grid
from ags.blobs import baselines as B

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--aniso", type=float, default=5.0)
    ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--out", default="ags/blobs/runs/dpsgd_tuned_a5.json")
    a = ap.parse_args()

    X, ys, Xte, yts = load_blobs(a.N, a.d, sep=a.sep, seed=0, aniso=a.aniso)
    P, Pte = np.c_[X, np.ones(a.N)], np.c_[Xte, np.ones(len(Xte))]
    g = grid()
    print(f"N={a.N} d={a.d} aniso={a.aniso:g} Bayes={bayes_accuracy(a.sep, 1.0, a.d):.4f}", flush=True)
    print(
        f"DP-SGD over {len(g)} configs (b x epochs x lr), gamma={GAMMA} lambda={LAM}, {a.seeds} seeds\n",
        flush=True,
    )
    # Full config x eps table: a single config must be chosen for ALL eps, so the
    # per-eps maximum is not a legitimate arm -- it is only an upper bound on one.
    rows = []
    for j, (b, ep, lr) in enumerate(g):
        ns = ep * max(1, a.N // b)
        acc = {}
        for eps in EPS:
            acc[str(eps)] = float(
                np.median(
                    [
                        B.acc(
                            B.dpsgd_pure(P, ys, eps, n_s=ns, lr=lr, gamma=GAMMA, lam=LAM, seed=s, b=b)[0],
                            Pte,
                            yts,
                        )
                        for s in range(a.seeds)
                    ]
                )
            )
        rows.append(dict(b=b, epochs=ep, n_s=ns, lr=lr, acc=acc))
        print(
            f"[{j + 1}/{len(g)}] b={b} ep={ep} ns={ns} lr={lr}  "
            + " ".join(f"{e}:{acc[str(e)]:.3f}" for e in EPS),
            flush=True,
        )
        json.dump(
            dict(
                N=a.N,
                d=a.d,
                aniso=a.aniso,
                sep=a.sep,
                seeds=a.seeds,
                gamma=GAMMA,
                lam=LAM,
                n_configs=len(g),
                rows=rows,
            ),
            open(a.out, "w"),
            indent=1,
        )
    mean = lambda r: sum(r["acc"][str(e)] for e in EPS) / len(EPS)
    bestcfg = max(rows, key=mean)
    print(
        f"\nsingle best config by mean acc over eps: b={bestcfg['b']} "
        f"epochs={bestcfg['epochs']} n_s={bestcfg['n_s']} lr={bestcfg['lr']}",
        flush=True,
    )
    print(f"\nwrote {a.out}", flush=True)
