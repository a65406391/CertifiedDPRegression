"""MILP certified runs with total serialization. One run directory per (cell, F).

Every Gurobi solve, every concretization step, every formulation, and the full
provenance are written to ags/blobs/runs/certified/<run_id>/. See serialize.py
for the layout. Nothing computed is discarded.
"""

import argparse
import json
import os
import time
import numpy as np
from ags.blobs.shared import load_blobs, save_bound, relu_model, random_relu_features
from ags.blobs.ags import A_of_k
from ags.blobs import ss_reference as R
from ags.blobs.serialize import RunRecorder, install, uninstall
from ags.blobs.opt_bound import opt_bound
from ags.blobs import baselines as B

EPS = [1, 2, 5, 10, 15, 20, 30, 50, 100, 200]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--aniso", type=float, default=5.0)
    ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--F", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--semantics",
        default="MILP",
        choices=["LP", "QP", "MILP"],
        help="LP/QP are SOUND relaxations of MILP: relaxing the "
        "feasible set can only widen the envelope, never invalidate it",
    )
    ap.add_argument(
        "--obbt",
        action="store_true",
        help="optimization-based bound tightening: narrows the variable\n                          ranges feeding the spatial relaxation, which is the half of\n                          tractability no solver flag touches",
    )
    ap.add_argument("--obbt-timeout", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--tag", default="a5")
    ap.add_argument("--arch", default="linear", choices=["linear", "relu"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument(
        "--rf",
        type=int,
        default=0,
        help="frozen random ReLU features: train a head on phi(x) "
        "(the only non-linear model the MILP supports today)",
    )
    a = ap.parse_args()
    assert a.lam == 0.0, "lambda must be 0 for this study"

    X, ys, Xte, yts = load_blobs(a.N, a.d, sep=a.sep, seed=0, aniso=a.aniso)
    y01 = (ys > 0).astype(float)
    P, Pte = np.c_[X, np.ones(a.N)], np.c_[Xte, np.ones(len(Xte))]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / a.N))
    ns = a.epochs * (a.N // a.b)
    if a.rf:
        X, (Wf, bf) = random_relu_features(X, a.rf, seed=0)
        Xte, _ = random_relu_features(Xte, a.rf, seed=0)
        model_fn, n_coords, arch_tag = None, a.rf + 1, f"rf{a.rf}"
        a.d = a.rf
    elif a.arch == "relu":
        model_fn = lambda d: relu_model(d, a.width, depth=1)
        n_coords = a.d * a.width + a.width + a.width + 1
        arch_tag = f"relu{a.width}"
    else:
        model_fn, n_coords, arch_tag = None, a.d + 1, "linear"
    run_id = f"{a.tag}_{arch_tag}_b{a.b}_ns{ns}_F{a.F}_{a.semantics}" + ("_OBBT" if a.obbt else "")

    cfg = dict(
        N=a.N,
        d=a.d,
        sep=a.sep,
        aniso=a.aniso,
        batch_size=a.b,
        n_epochs=a.epochs,
        n_s=ns,
        gamma=a.gamma,
        lam=a.lam,
        lr=lr,
        loss="hinge",
        F=a.F,
        b_eff=min(a.b * a.F, a.N),
        semantics=a.semantics,
        k_grid=list(a.ks),
        timeout=a.timeout,
        threads=a.threads,
        adjacency="substitution",
        obbt=a.obbt,
        obbt_timeout=a.obbt_timeout,
        arch=arch_tag,
        width=(a.width if a.arch == "relu" else None),
        n_param_coords=n_coords,
        data_sha256=__import__("hashlib").sha256(X.tobytes()).hexdigest(),
    )
    rec = RunRecorder(run_id, cfg, expected_windows_per_k=ns // a.F, k_grid=a.ks, n_param_coords=n_coords)
    print(
        f"run_id={run_id}\n  arch={arch_tag} coords={n_coords} "
        f"n_s={ns} b_eff={cfg['b_eff']} windows/k={ns // a.F} "
        f"k={a.ks}\n  expected: {rec.expected_steps} steps, {rec.expected_solves} solves"
        f"\n  dir={rec.dir}\n",
        flush=True,
    )
    install(rec)
    # reference: true_gs_window = sum_lr_window * 2 * clip_gamma / batchsize (one release
    # over the whole trajectory, so sum_lr_window = lr * n_s)
    gs_vec = R.gs_window(lr * ns, a.gamma, a.b, n_coords)
    print(f"  GS/coord (reference) = {gs_vec[0]:.6g}", flush=True)
    Ag, ls_by_k, theta_n, t0 = [], {}, None, time.time()
    try:
        for k in a.ks:
            rec.set_k(k)
            tk = time.time()
            l, n, u, _ = opt_bound(
                X,
                y01,
                k=k,
                b=a.b,
                n_epochs=a.epochs,
                lr=lr,
                gamma=a.gamma,
                F=a.F,
                semantics=a.semantics,
                timeout=a.timeout,
                threads=a.threads,
                model_fn=model_fn,
                obbt=a.obbt,
                obbt_timeout=a.obbt_timeout,
            )
            Ag.append(A_of_k(l, n, u))
            # reference inference needs the per-coordinate WIDTH, capped at GS*k
            ls_by_k[k] = R.local_sens(l, u, gs_vec, k)
            if k == a.ks[0]:
                theta_n = n
            save_bound(
                f"{run_id}_k{k}",
                [l],
                [n],
                [u],
                dict(cfg, k=k, bound=f"{a.semantics} F={a.F}"),
                extra=dict(A_k=Ag[-1]),
            )
            print(
                f"  k={k:<3} A^k={Ag[-1]:.6e}  [{time.time() - tk:.0f}s]  "
                f"solves={rec.n_solves}/{rec.expected_solves}",
                flush=True,
            )
        nom = B.acc(theta_n, Pte, yts)
        rows = []
        for eps in EPS:
            # experiments/common/ags.py inference, ported line-by-line in ss_reference
            ss = R.smooth_sensitivity(ls_by_k, a.ks, gs_vec, R.beta_for(eps), n_coords)
            sg = R.sigma(ss, eps, n_coords, noise_mode="isotropic")
            acc = float(
                np.median(
                    [
                        B.acc(R.release(theta_n, sg, np.random.RandomState(s)), Pte, yts)
                        for s in range(a.seeds)
                    ]
                )
            )
            rows.append(dict(eps=eps, ags=acc, ss_mean=float(ss.mean()), sigma_mean=float(sg.mean())))
        res = dict(
            cfg=cfg,
            run_id=run_id,
            k_grid=list(a.ks),
            A_grid=Ag,
            gs_per_coord=float(gs_vec[0]),
            nominal=nom,
            rows=rows,
            wall_s=time.time() - t0,
        )
        with open(os.path.join(rec.dir, "results.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        man = rec.close("completed")
        print(f"\nnominal={nom:.4f}  wall={time.time() - t0:.0f}s")
        print(
            f"MANIFEST: steps {man['steps_written']}/{man['steps_expected']} "
            f"solves {man['solves_written']}/{man['solves_expected']} "
            f"complete={man['steps_complete'] and man['solves_complete']}"
        )
        print(f"  counts: {man['counts']}", flush=True)
    except Exception as e:
        rec.close("failed", str(e))
        raise
    finally:
        uninstall()
