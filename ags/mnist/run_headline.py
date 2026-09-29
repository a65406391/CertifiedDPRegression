"""AGS-IBP and every baseline at the headline setting, pure and approximate DP.

    MNIST PCA-64, 10 classes, multinomial logistic head, P = 650
    N = 10000, b = 2000, 1 epoch = 5 steps, lr = 15, gamma = 0.12, lambda = 0
    cross-entropy, substitution adjacency, gamma_c = 2

n_s = 5 admits F in {1, 5}, the two concretization rungs the blobs study reports,
which is why the step count is 5 and the batch is 2000.

The k grid REACHES the batch size (k = 2000 = b), so A^k has saturated at GS by the
end of the grid and the smooth-sensitivity staircase needs no tail term. See
release.py for why that is sound rather than an omission.

Arms, all privatizing the SAME algorithm unless labelled "tuned":

  AGS (IBP)              certified envelope -> smooth sensitivity -> Cauchy / Laplace
  DP-SGD (matched)       identical b, n_s, lr, clip; Laplace (pure) / Gaussian (approx),
                         basic composition at eps/n_s
  DP-SGD (matched, RDP)  approx only; same fixed batching, Renyi composition, no
                         subsampling amplification claimed
  DP-SGD (tuned)         best SINGLE config over a (b, n_s, lr) grid, chosen by mean
                         accuracy across eps -- never per-eps
  PATE                   disjoint teachers -> noisy argmax on a public pool -> student;
                         best single (m, q) under the same rule
  non-private            the same clipped SGD with no noise

    python -m ags.mnist.run_headline
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import (
    load_mnist_pca,
    logistic_model,
    accuracy,
    reference_accuracy,
    N_CLASSES,
    save_bound,
)
from ags.mnist.ags import agt_bound, A_of_k, clipping_ceiling
from ags.mnist import release as REL
from ags.mnist import baselines as B

K_GRID = [1, 2, 3, 4, 8, 16, 32, 128, 1024, 2000]
EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100]
DELTA = 1e-5
LOSS = "cross_entropy"

# DP-SGD tuning grid. b and n_s are the free axes; gamma is pinned at the AGS value
# because the clip is not independently free -- it sets how far a run can move.
TUNE_B = [10000, 5000, 2000, 1000, 500]
TUNE_STEPS = [1, 2, 5, 10, 20, 50]
TUNE_LR = [1.0, 5.0, 15.0, 50.0]
PATE_M = [2, 3, 5, 10, 20, 50]
PATE_Q = [50, 100, 250, 500, 1000]


def main(a):
    X, y, Xte, yte = load_mnist_pca(a.N, a.d, seed=a.seed, n_test=a.n_test)
    P = N_CLASSES * a.d + N_CLASSES
    spe = a.N // a.b
    n_s = a.epochs * spe
    ev = lambda th: accuracy(th, Xte, yte)
    t0 = time.time()

    print(f"MNIST PCA-{a.d}, {N_CLASSES} classes, loss={LOSS}, P={P}")
    print(f"N={a.N} b={a.b} epochs={a.epochs} -> n_s={n_s}  lr={a.lr} gamma={a.gamma} lambda={a.lam}")
    print(f"k grid = {K_GRID}   (reaches b={a.b}, so no tail term)")
    print(
        f"clipping ceiling (width) = 2*n_s*lr*gamma = {clipping_ceiling(n_s, a.lr, a.gamma):g}\n", flush=True
    )

    # ---------------------------------------------------------------- envelopes
    A_by_k, radius_by_k, theta_n, rows_k = {}, {}, None, []
    for k in K_GRID:
        tk = time.time()
        pl, pn, pu = agt_bound(
            logistic_model(a.d),
            X,
            y,
            k=k,
            b=a.b,
            n_epochs=a.epochs,
            lr=a.lr,
            gamma=a.gamma,
            l2_reg=a.lam,
            loss=LOSS,
        )
        A_by_k[k] = np.abs(pu - pl)  # WIDTH, per release.py
        radius_by_k[k] = np.maximum(pu - pn, pn - pl)
        if theta_n is None:
            theta_n = pn
        rows_k.append(
            dict(
                k=k,
                width_mean=float(A_by_k[k].mean()),
                width_max=float(A_by_k[k].max()),
                radius_mean=float(radius_by_k[k].mean()),
                A1_l1_radius=A_of_k(pl, pn, pu),
                secs=time.time() - tk,
            )
        )
        save_bound(
            f"ibp_N{a.N}_b{a.b}_ns{n_s}_k{k}",
            [pl],
            [pn],
            [pu],
            dict(
                N=a.N,
                d=a.d,
                b=a.b,
                epochs=a.epochs,
                n_s=n_s,
                lr=a.lr,
                gamma=a.gamma,
                lam=a.lam,
                loss=LOSS,
                k=k,
                bound="IBP",
            ),
        )
        print(
            f"  k={k:<5} width_mean={A_by_k[k].mean():.6g}  "
            f"radius_mean={radius_by_k[k].mean():.6g}  [{time.time() - tk:.0f}s]",
            flush=True,
        )

    GS = A_by_k[a.b]
    nominal = ev(theta_n)
    ceil = clipping_ceiling(n_s, a.lr, a.gamma)
    print(f"\nGS = A^(k=b) width mean = {GS.mean():.6g}   GS/ceiling = {GS.mean() / ceil:.4f}")
    print(f"A^1 / GS = {A_by_k[1].mean() / GS.mean():.6f}   nominal acc = {nominal:.4f}")
    print(f"unregularised logistic reference = {reference_accuracy(X, y, Xte, yte):.4f}\n", flush=True)

    # ---------------------------------------------------------------- baselines
    print("baselines", flush=True)
    nonpriv = ev(B.non_private(X, y, n_s=n_s, lr=a.lr, gamma=a.gamma, lam=a.lam, b=a.b, loss=LOSS))
    print(f"  non-private (matched algorithm) = {nonpriv:.4f}", flush=True)

    def med(fn):
        return float(np.median([ev(fn(s)) for s in range(a.seeds)]))

    dp_matched_pure = [
        med(
            lambda s, e=e: B.dpsgd_pure(
                X, y, e, n_s=n_s, lr=a.lr, gamma=a.gamma, lam=a.lam, b=a.b, seed=s, loss=LOSS
            )[0]
        )
        for e in EPS
    ]
    print(f"  DP-SGD matched  (pure)   " + " ".join(f"{v:.4f}" for v in dp_matched_pure), flush=True)
    dp_matched_apx = [
        med(
            lambda s, e=e: B.dpsgd_gaussian(
                X, y, e, DELTA, n_s=n_s, lr=a.lr, gamma=a.gamma, lam=a.lam, b=a.b, seed=s, loss=LOSS
            )[0]
        )
        for e in EPS
    ]
    print(f"  DP-SGD matched  (approx) " + " ".join(f"{v:.4f}" for v in dp_matched_apx), flush=True)
    dp_rdp_apx = [
        med(
            lambda s, e=e: B.dpsgd_rdp_fixed(
                X, y, e, DELTA, n_s=n_s, lr=a.lr, gamma=a.gamma, lam=a.lam, b=a.b, seed=s, loss=LOSS
            )[0]
        )
        for e in EPS
    ]
    print(f"  DP-SGD RDP      (approx) " + " ".join(f"{v:.4f}" for v in dp_rdp_apx), flush=True)

    def tune_dpsgd(mode):
        grid, best, bcfg = [], None, None
        for bb in TUNE_B:
            for ns in TUNE_STEPS:
                for lr in TUNE_LR:
                    cur = []
                    for e in EPS:
                        if mode == "pure":
                            f = lambda s, e=e, bb=bb, ns=ns, lr=lr: B.dpsgd_pure(
                                X, y, e, n_s=ns, lr=lr, gamma=a.gamma, lam=a.lam, b=bb, seed=s, loss=LOSS
                            )[0]
                        else:
                            f = lambda s, e=e, bb=bb, ns=ns, lr=lr: B.dpsgd_gaussian(
                                X,
                                y,
                                e,
                                DELTA,
                                n_s=ns,
                                lr=lr,
                                gamma=a.gamma,
                                lam=a.lam,
                                b=bb,
                                seed=s,
                                loss=LOSS,
                            )[0]
                        cur.append(float(np.median([ev(f(s)) for s in range(a.tune_seeds)])))
                    grid.append(dict(b=bb, n_s=ns, lr=lr, acc=cur))
                    if best is None or np.mean(cur) > np.mean(best):
                        best, bcfg = cur, dict(b=bb, n_s=ns, lr=lr)
        return best, bcfg, grid

    dp_tuned_pure, cfg_tp, grid_tp = tune_dpsgd("pure")
    print(
        f"  DP-SGD tuned    (pure)   " + " ".join(f"{v:.4f}" for v in dp_tuned_pure) + f"   {cfg_tp}",
        flush=True,
    )
    dp_tuned_apx, cfg_ta, grid_ta = tune_dpsgd("approx")
    print(
        f"  DP-SGD tuned    (approx) " + " ".join(f"{v:.4f}" for v in dp_tuned_apx) + f"   {cfg_ta}",
        flush=True,
    )

    def tune_pate(mode):
        grid, best, bcfg = [], None, None
        for m in PATE_M:
            for q in PATE_Q:
                cur = []
                for e in EPS:
                    if mode == "pure":
                        f = lambda s, e=e, m=m, q=q: B.pate(
                            X,
                            y,
                            Xte,
                            e,
                            n_teachers=m,
                            n_queries=q,
                            n_s=n_s,
                            lr=a.lr,
                            gamma=a.gamma,
                            lam=a.lam,
                            seed=s,
                            loss=LOSS,
                        )[0]
                    else:
                        f = lambda s, e=e, m=m, q=q: B.pate_gaussian(
                            X,
                            y,
                            Xte,
                            e,
                            DELTA,
                            n_teachers=m,
                            n_queries=q,
                            n_s=n_s,
                            lr=a.lr,
                            gamma=a.gamma,
                            lam=a.lam,
                            seed=s,
                            loss=LOSS,
                        )[0]
                    cur.append(float(np.median([ev(f(s)) for s in range(a.tune_seeds)])))
                grid.append(dict(m=m, q=q, acc=cur))
                if best is None or np.mean(cur) > np.mean(best):
                    best, bcfg = cur, dict(m=m, q=q)
        return best, bcfg, grid

    pate_pure, cfg_pp, grid_pp = tune_pate("pure")
    print(
        f"  PATE            (pure)   " + " ".join(f"{v:.4f}" for v in pate_pure) + f"   {cfg_pp}", flush=True
    )
    pate_apx, cfg_pa, grid_pa = tune_pate("approx")
    print(
        f"  PATE            (approx) " + " ".join(f"{v:.4f}" for v in pate_apx) + f"   {cfg_pa}", flush=True
    )

    # ---------------------------------------------------------------- AGS
    print("\nAGS (IBP)", flush=True)
    ags = {}
    for mode in ("pure", "approx"):
        accs, ss_m, sg_m = [], [], []
        for e in EPS:
            acc, ss, sg = REL.ags_release(
                theta_n,
                A_by_k,
                K_GRID,
                eps=e,
                n_params=P,
                batchsize=a.b,
                mode=mode,
                delta=DELTA,
                seeds=a.seeds,
                evaluate=ev,
            )
            accs.append(acc)
            ss_m.append(float(ss.mean()))
            sg_m.append(float(sg.mean()))
        ags[mode] = dict(acc=accs, ss_mean=ss_m, sigma_mean=sg_m)
        print(f"  {mode:<7} acc " + " ".join(f"{v:.4f}" for v in accs), flush=True)
        print(f"  {'':<7} SS  " + " ".join(f"{v:.3g}" for v in ss_m), flush=True)

    out = dict(
        setting=dict(
            dataset="mnist_pca64_10class",
            N=a.N,
            d=a.d,
            n_classes=N_CLASSES,
            P=P,
            b=a.b,
            epochs=a.epochs,
            n_s=n_s,
            steps_per_epoch=spe,
            lr=a.lr,
            gamma=a.gamma,
            lam=a.lam,
            loss=LOSS,
            seed=a.seed,
            n_test=len(yte),
            adjacency="substitution",
            gamma_cauchy=2,
            delta=DELTA,
            F_available=[1, 5],
        ),
        k_grid=K_GRID,
        eps=EPS,
        seeds=a.seeds,
        tune_seeds=a.tune_seeds,
        envelope=dict(
            rows=rows_k,
            gs_width_mean=float(GS.mean()),
            gs_width=GS.tolist(),
            clipping_ceiling=ceil,
            gs_over_ceiling=float(GS.mean() / ceil),
            A1_over_gs=float(A_by_k[1].mean() / GS.mean()),
            width_by_k={str(k): A_by_k[k].tolist() for k in K_GRID},
            no_tail_justification="k grid reaches b, so A^k = GS for all k >= b",
        ),
        nominal=nominal,
        non_private=nonpriv,
        reference_unregularised=reference_accuracy(X, y, Xte, yte),
        theta_n=theta_n.tolist(),
        ags=ags,
        dpsgd=dict(
            matched_pure=dp_matched_pure,
            matched_approx=dp_matched_apx,
            rdp_approx=dp_rdp_apx,
            tuned_pure=dp_tuned_pure,
            tuned_pure_cfg=cfg_tp,
            tuned_approx=dp_tuned_apx,
            tuned_approx_cfg=cfg_ta,
            grid_pure=grid_tp,
            grid_approx=grid_ta,
        ),
        pate=dict(
            pure=pate_pure,
            pure_cfg=cfg_pp,
            approx=pate_apx,
            approx_cfg=cfg_pa,
            grid_pure=grid_pp,
            grid_approx=grid_pa,
        ),
        wall_s=time.time() - t0,
    )
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=15.0)
    ap.add_argument("--gamma", type=float, default=0.12)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--tune_seeds", type=int, default=11)
    ap.add_argument("--out", default="ags/mnist/runs/headline.json")
    main(ap.parse_args())
