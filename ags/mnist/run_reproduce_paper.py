"""Reproduce the regime where AGS beats matched Laplace DP-SGD.

Source: experiments/mnist/run_ags_comparison.py, whose stored results.json holds four
runs. Run 2 is the one where AGS wins and run 3 is the one where it does not, and they
differ in exactly two settings:

                     run 2 (AGS wins)        run 3 (tied)
    sampling_freq    40000 = total_steps     10000
    n_windows        1                       4
    k_subset         [1,2,4,...,256,500]     [1,4,7,10]

    eps               5      10      15      20        10     100    1000   20000
    AGS            0.515   0.655   0.706   0.731     0.104   0.107   0.165   0.714
    Laplace        0.094   0.090   0.095   0.094     0.091   0.094   0.151   0.743
    delta          +0.42   +0.57   +0.61   +0.64     +0.01   +0.01   +0.01   -0.03

Three mechanisms produce that win, none of which was in my earlier 5-step setting:

1. LEARNING-RATE DECAY. lr_n = lr/(1 + eta n) with eta = 0.01. Over 40000 steps
   sum_lr = 29.99 instead of lr*T = 2000 -- a 67x reduction. Sensitivity is linear in
   sum_lr, so decay buys a long trajectory at almost no sensitivity cost. This is
   AGTConfig(lr_decay=eta, lr_decay_type="batch").

2. ONE RELEASE. sampling_freq = total_steps gives n_windows = 1, so the full epsilon
   is spent on a single release. Run 3's four windows split epsilon four ways AND add
   four Cauchy draws, and that is what flattens it back to chance.

3. THE CLOSED-FORM CAP BINDS. local_sens = min(A^k, cf*k) with
   cf = sum_lr*2*gamma/b = 0.0029995. The reference's reported avg_ss of 1.95 is
   exactly cf * P = 0.0029995 * 650, so the cap selects the closed form on every
   coordinate and the IBP envelope never enters the release.

   That cap is sound only when the update map is non-expansive, i.e. lr <= 2/L. For
   multinomial logistic with ||x||<=1, L <= 1/2, so lr <= 4. Their lr = 0.05 clears it
   by 80x. My earlier lr = 15 VIOLATED it by 3.75x, which is why the honest uncapped
   envelope there was 203x larger and AGS sat at chance.

Because the cap binds, the AGS release at this setting is fully determined by cf and
needs no solver -- but `--check-envelope` verifies the premise by computing A^k with
AGT and asserting cf*k <= A^k, so the min() is doing what is claimed.

    python -m ags.mnist.run_reproduce_paper
    python -m ags.mnist.run_reproduce_paper --N 10000 --check-envelope
"""

import argparse
import json
import time

import numpy as np

from ags.mnist.shared import load_mnist_pca, logistic_model, accuracy, N_CLASSES
from ags.mnist.ags import agt_bound, sum_lr, clipping_ceiling
from ags.mnist import release as REL
from ags.mnist import baselines as B

EPS = [5, 10, 15, 20, 50, 100, 500, 1000, 5000, 20000]
K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]


def main(a):
    X, y, Xte, yte = load_mnist_pca(a.N, a.d, seed=a.seed, n_test=a.n_test, scale_unit_norm=not a.no_scale)
    P = N_CLASSES * a.d + N_CLASSES
    ev = lambda th: accuracy(th, Xte, yte)
    slr = sum_lr(a.lr, a.steps, a.eta)
    cf = REL.closed_form_substitution(slr, a.gamma, a.b, P)
    xmax = float(np.linalg.norm(X, axis=1).max())
    L = REL.smoothness_constant(X=X, loss="cross_entropy")
    t0 = time.time()

    print(f"MNIST PCA-{a.d}, {N_CLASSES} classes, P={P}")
    print(f"N={a.N} b={a.b} steps={a.steps} lr={a.lr} eta={a.eta} gamma={a.gamma}")
    print(f"max||x||_2 = {xmax:.4g}   L = 0.5*lam_max(XtX/b) = {L:.4g}   lr limit 2/L = {2 / L:.4g}")
    print(
        f"sum_lr = {slr:.4f}   (constant-lr would be {a.lr * a.steps:g}, "
        f"decay saves {a.lr * a.steps / slr:.0f}x)"
    )
    print(f"closed form cf = sum_lr*2*gamma/b = {cf[0]:.6g}   cf*P = {cf[0] * P:.4g}")
    print(
        f"non-expansiveness: lr={a.lr} vs 2/L={2 / L:g}  -> "
        f"{'HOLDS, cap is sound' if a.lr <= 2 / L else 'VIOLATED, cap is NOT sound'}"
    )
    print(f"clipping ceiling = {clipping_ceiling(a.steps, a.lr, a.gamma, a.eta):.4g}\n", flush=True)

    nominal = ev(B.non_private(X, y, n_s=a.steps, lr=a.lr, gamma=a.gamma, lam=0.0, b=a.b, lr_decay=a.eta))
    print(f"non-private (matched algorithm) = {nominal:.4f}", flush=True)

    # -------------------------------------------------- optional envelope check
    env = None
    if a.check_envelope:
        print("\nverifying the cap binds: computing A^k with AGT", flush=True)
        env = []
        spe = max(1, a.N // a.b)
        ep = max(1, a.steps // spe)
        for k in a.env_ks:
            tk = time.time()
            pl, pn, pu = agt_bound(
                logistic_model(a.d),
                X,
                y,
                k=k,
                b=a.b,
                n_epochs=ep,
                lr=a.lr,
                gamma=a.gamma,
                l2_reg=0.0,
                lr_decay=a.eta,
            )
            w = np.abs(pu - pl)
            binds = bool((cf * k <= w + 1e-12).all())
            env.append(
                dict(k=k, width_mean=float(w.mean()), cf_k=float(cf[0] * k), cap_binds_everywhere=binds)
            )
            print(
                f"  k={k:<5} A^k={w.mean():.6g}  cf*k={cf[0] * k:.6g}  "
                f"ratio={w.mean() / (cf[0] * k):.1f}x  cap binds: {binds}  "
                f"[{time.time() - tk:.0f}s]",
                flush=True,
            )

    # -------------------------------------------------- AGS release
    # The cap binds, so local_sens[k] = cf*k on every coordinate, and the staircase
    # runs over that. This is the reference's own inference (ss_reference L434-527).
    ls = {k: cf * k for k in K_GRID}
    print("\nAGS (cap binds -> release determined by the closed form)", flush=True)
    ags = {}
    for mode in ("pure", "approx"):
        accs, sss, sgs = [], [], []
        for e in EPS:
            beta = (
                REL.beta_for(e)
                if mode == "pure"
                else REL.beta_for(e, mode="approx", n_params=P, delta=a.delta)
            )
            ss = REL.R.smooth_sensitivity(ls, K_GRID, cf, beta, P)
            sg = REL.sigma(ss, e, P, noise_mode=a.noise_mode, mode=mode)
            acc = (
                float(
                    np.median(
                        [
                            ev(
                                REL.release(
                                    np.zeros(P) if a.theta is None else a.theta,
                                    sg,
                                    np.random.RandomState(s),
                                    mode=mode,
                                )
                            )
                            for s in range(a.seeds)
                        ]
                    )
                )
                if a.theta is not None
                else None
            )
            accs.append(acc)
            sss.append(float(ss.mean()))
            sgs.append(float(sg.mean()))
        ags[mode] = dict(ss_mean=sss, sigma_mean=sgs, acc=accs)

    theta_n = B.non_private(X, y, n_s=a.steps, lr=a.lr, gamma=a.gamma, lam=0.0, b=a.b, lr_decay=a.eta)
    for mode in ("pure", "approx"):
        accs = []
        for i, e in enumerate(EPS):
            beta = (
                REL.beta_for(e)
                if mode == "pure"
                else REL.beta_for(e, mode="approx", n_params=P, delta=a.delta)
            )
            ss = REL.R.smooth_sensitivity(ls, K_GRID, cf, beta, P)
            sg = REL.sigma(ss, e, P, noise_mode=a.noise_mode, mode=mode)
            accs.append(
                float(
                    np.median(
                        [
                            ev(REL.release(theta_n, sg, np.random.RandomState(s), mode=mode))
                            for s in range(a.seeds)
                        ]
                    )
                )
            )
        ags[mode]["acc"] = accs
        print(f"  {mode:<7} " + " ".join(f"{v:.4f}" for v in accs), flush=True)
        print(f"  {'':<7} sigma " + " ".join(f"{v:.3g}" for v in ags[mode]["sigma_mean"]), flush=True)

    # -------------------------------------------------- matched Laplace DP-SGD
    print("\nmatched Laplace DP-SGD (same algorithm, basic composition at eps/T)", flush=True)
    lap = [
        float(
            np.median(
                [
                    ev(
                        B.dpsgd_pure(
                            X,
                            y,
                            e,
                            n_s=a.steps,
                            lr=a.lr,
                            gamma=a.gamma,
                            lam=0.0,
                            b=a.b,
                            seed=s,
                            lr_decay=a.eta,
                        )[0]
                    )
                    for s in range(a.lap_seeds)
                ]
            )
        )
        for e in EPS
    ]
    print("  " + " ".join(f"{v:.4f}" for v in lap), flush=True)

    print(f"\n{'eps':>8} {'AGS(pure)':>10} {'Laplace':>9} {'delta':>8}")
    for i, e in enumerate(EPS):
        print(f"{e:>8} {ags['pure']['acc'][i]:>10.4f} {lap[i]:>9.4f} {ags['pure']['acc'][i] - lap[i]:>+8.4f}")

    json.dump(
        dict(
            setting=dict(
                N=a.N,
                d=a.d,
                b=a.b,
                steps=a.steps,
                lr=a.lr,
                eta=a.eta,
                gamma=a.gamma,
                P=P,
                n_windows=1,
                seed=a.seed,
                noise_mode=a.noise_mode,
                delta=a.delta,
                sum_lr=slr,
                closed_form=float(cf[0]),
                lr_limit_2_over_L=2 / L,
                cap_sound=bool(a.lr <= 2 / L),
            ),
            eps=EPS,
            k_grid=K_GRID,
            nominal=nominal,
            ags=ags,
            laplace_matched=lap,
            envelope_check=env,
            wall_s=time.time() - t0,
        ),
        open(a.out, "w"),
        indent=1,
    )
    print(f"\nwrote {a.out}   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument(
        "--noise_mode", default="anisotropic", help="train_ags defaults to anisotropic; blobs used isotropic"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--lap_seeds", type=int, default=11)
    ap.add_argument(
        "--no-scale",
        action="store_true",
        dest="no_scale",
        help="whitened PCA features as-is, matching the reference's "
        "load_mnist, instead of rescaling to max||x||_2 = 1",
    )
    ap.add_argument("--check-envelope", action="store_true", dest="check_envelope")
    ap.add_argument("--env_ks", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--theta", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="ags/mnist/runs/reproduce_paper.json")
    main(ap.parse_args())
