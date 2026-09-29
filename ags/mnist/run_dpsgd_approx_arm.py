"""Matched Gaussian DP-SGD at delta = 1e-5, the arm the reproduction was missing.

Same setting as run_reproduce_paper.py --no-scale, same trajectory, same clipping, no
tuning of anything: the only difference from the nominal run is the noise added to
each step's mean gradient.

TWO ACCOUNTINGS, because the reference's own is optimistic
----------------------------------------------------------
`baselines.dpsgd_gaussian` ports experiments/common/ags.py:train_gaussian_dpsgd, which
splits epsilon over the steps (eps/T) but passes the SAME delta to every step. Under
basic composition that composes to (eps, T*delta) = (eps, 0.4) here, not (eps, 1e-5) --
the mechanism is under-noised relative to what it claims.

    reference   delta_step = delta        -> sigma ∝ sqrt(2 ln(1.25/1e-5))    = 4.85
    strict      delta_step = delta / T    -> sigma ∝ sqrt(2 ln(1.25/2.5e-10)) = 6.68

Only a 1.38x difference in noise, so both are run and both are reported. The strict row
is the one that is actually (eps, 1e-5)-DP and therefore the one to compare AGS against;
the reference row is there because it is what the paper's pipeline computes.

WHERE IT RUNS
-------------
On CUDA by default, through torch_sgd.train, which reproduces baselines.train to
float64 round-off (max|d theta| ~ 1e-14, checked by torch_sgd.validate) and is ~35x
faster: ~12 s per 40000-step training against ~200-380 s of numpy on a loaded CPU box.
--device cpu falls back to the numpy path across a process pool.

    PYTHONPATH=$PWD ${PYTHON:-python} -m ags.mnist.run_dpsgd_approx_arm
"""

import argparse
import json
import os
import time

# one BLAS thread per worker: each training is a stream of 2000x64 @ 64x10 products,
# far too small to thread, and on a shared box the threads only fight each other.
# Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import numpy as np

from ags.mnist.shared import load_mnist_pca, accuracy, N_CLASSES
from ags.mnist import baselines as B
from ags.mnist import torch_sgd as T

EPS = [5, 10, 15, 20, 50, 100, 500, 1000, 5000, 20000]

_W = {}


def _init(a, X, y, Xte, yte):
    _W.update(a=a, X=X, y=y, Xte=Xte, yte=yte)


def _one(task):
    """One (accounting, eps, seed) training. ~200 s at T=40000 on this box."""
    tag, dstep, e, seed = task
    a = _W["a"]
    th, sc = B.dpsgd_gaussian(
        _W["X"],
        _W["y"],
        e,
        dstep,
        n_s=a.steps,
        lr=a.lr,
        gamma=a.gamma,
        lam=0.0,
        b=a.b,
        seed=seed,
        lr_decay=a.eta,
    )
    return tag, e, seed, float(accuracy(th, _W["Xte"], _W["yte"], N_CLASSES)), float(sc)


def main(a):
    X, y, Xte, yte = load_mnist_pca(a.N, a.d, seed=a.seed, n_test=a.n_test, scale_unit_norm=not a.no_scale)
    P = N_CLASSES * a.d + N_CLASSES
    ev = lambda th: accuracy(th, Xte, yte)
    t0 = time.time()

    print(f"matched Gaussian DP-SGD, delta = {a.delta:g}")
    print("single seed (seed 0), no median" if a.seeds == 1 else f"median over {a.seeds} seeds")
    print(f"N={a.N} b={a.b} steps={a.steps} lr={a.lr} eta={a.eta} gamma={a.gamma} P={P}")
    print(f"l2 sensitivity of the mean grad = sqrt(P)*2*gamma/b = {np.sqrt(P) * 2 * a.gamma / a.b:.6g}")
    print("", flush=True)

    nominal = ev(B.non_private(X, y, n_s=a.steps, lr=a.lr, gamma=a.gamma, lam=0.0, b=a.b, lr_decay=a.eta))
    print(f"non-private (matched algorithm) = {nominal:.4f}\n", flush=True)

    dev = T.pick_device() if a.device == "auto" else __import__("torch").device(a.device)
    accountings = (("reference", a.delta), ("strict", a.delta / a.steps))

    if dev.type == "cuda":
        import torch

        print(f"device: {dev} ({torch.cuda.get_device_name(dev)})", flush=True)
        out, t1 = {}, time.time()
        for tag, dstep in accountings:
            accs, sigmas = [], []
            print(f"\n{tag}: delta_step = {dstep:.4g}", flush=True)
            for e in EPS:
                te = time.time()
                per_seed = []
                for seed in range(a.seeds):
                    nz, sc = T.gaussian_noise(e, dstep, n_s=a.steps, gamma=a.gamma, b=a.b, P=P, seed=seed)
                    th = T.train(
                        X,
                        y,
                        n_s=a.steps,
                        lr=a.lr,
                        gamma=a.gamma,
                        lam=0.0,
                        b=a.b,
                        noise=nz,
                        lr_decay=a.eta,
                        device=dev,
                    )
                    per_seed.append(ev(th))
                accs.append(float(np.median(per_seed)))
                sigmas.append(float(sc))
                print(
                    f"  eps={e:<6} acc={accs[-1]:.4f}  sigma={sigmas[-1]:.4g}  [{time.time() - te:.0f}s]",
                    flush=True,
                )
            out[tag] = dict(delta_step=dstep, acc=accs, sigma=sigmas)
        print(f"\nall arms done in {(time.time() - t1) / 60:.1f} min", flush=True)
        return finish(a, out, nominal, P, t0)

    tasks = [(tag, dstep, e, seed) for tag, dstep in accountings for e in EPS for seed in range(a.seeds)]
    print(
        f"{len(tasks)} trainings = 2 accountings x {len(EPS)} eps x {a.seeds} seeds, {a.workers} at a time",
        flush=True,
    )

    got, t1 = {}, time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(a, X, y, Xte, yte)) as pool:
        for i, (tag, e, seed, acc, sc) in enumerate(pool.imap_unordered(_one, tasks, chunksize=1), 1):
            got.setdefault((tag, e), []).append((acc, sc))
            if i % a.workers == 0 or i == len(tasks):
                el = time.time() - t1
                print(
                    f"  {i:>3}/{len(tasks)}  [{el / 60:.1f} min elapsed, "
                    f"~{el / i * (len(tasks) - i) / 60:.0f} min left]",
                    flush=True,
                )

    out = {}
    for tag, dstep in accountings:
        accs = [float(np.median([v[0] for v in got[(tag, e)]])) for e in EPS]
        sigmas = [got[(tag, e)][0][1] for e in EPS]
        out[tag] = dict(delta_step=dstep, acc=accs, sigma=sigmas)
        print(f"\n{tag}: delta_step = {dstep:.4g}")
        print("  acc   " + " ".join(f"{v:.4f}" for v in accs))
        print("  sigma " + " ".join(f"{v:.3g}" for v in sigmas), flush=True)

    return finish(a, out, nominal, P, t0)


def finish(a, out, nominal, P, t0):
    ref = json.load(open(a.ags))
    ags = ref["ags"]["approx"]["acc"]
    print(f"\n{'eps':>8} {'AGS approx':>11} {'DP-SGD strict':>14} {'delta':>8}   {'DP-SGD ref':>11}")
    for i, e in enumerate(EPS):
        print(
            f"{e:>8} {ags[i]:>11.4f} {out['strict']['acc'][i]:>14.4f} "
            f"{ags[i] - out['strict']['acc'][i]:>+8.4f}   "
            f"{out['reference']['acc'][i]:>11.4f}"
        )

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
                delta=a.delta,
                seed=a.seed,
                seeds=a.seeds,
                no_scale=a.no_scale,
                device=a.device,
            ),
            eps=EPS,
            nominal=nominal,
            dpsgd_gaussian=out,
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1, help="1 = the single seed-0 run; >1 takes the median")
    ap.add_argument("--workers", type=int, default=16, help="CPU fallback only")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    ap.add_argument("--no-scale", action="store_true", dest="no_scale", default=True)
    ap.add_argument("--scale", action="store_false", dest="no_scale")
    ap.add_argument("--ags", default="ags/mnist/runs/reproduce_paper_raw.json")
    ap.add_argument("--out", default="ags/mnist/runs/dpsgd_approx_matched.json")
    main(ap.parse_args())
