"""P1: does a window of F updates tighten the bound over IBP, at lambda = 0?

OPTConfig has no regularisation field, so the MILP path cannot apply L2. Running both
arms at lambda = 0 keeps the comparison honest -- same algorithm, two bounds -- at the
cost of sitting in the regime where the envelope grows rather than contracting. That is
also where F should matter most: growth is exponential in n_s/F, so a window divides the
exponent, and F = n_s wraps the whole trajectory in one problem.

Semantics:
  IBP   AGT interval propagation (F = 1 by construction)
  LP    binaries and bilinears relaxed
  QP    binaries relaxed, bilinear terms kept
  MILP  exact
"""

import argparse
import json
import time
import warnings
import numpy as np
import torch

from ags.blobs.shared import load_blobs, linear_model
from ags.blobs.ags import agt_bound, A_of_k

SEMANTICS = {
    "LP": dict(relax_binaries=True, relax_bilinear=True),
    "QP": dict(relax_binaries=True, relax_bilinear=False),
    "MILP": dict(relax_binaries=False, relax_bilinear=False),
}


def opt_bound(
    X,
    y01,
    *,
    k,
    b,
    n_epochs,
    lr,
    gamma,
    F,
    semantics,
    timeout,
    threads,
    input_domain=None,
    model_fn=None,
    obbt=False,
    obbt_timeout=60,
):
    from abstract_gradient_training_opt.certified_training import certified_training
    from abstract_gradient_training_opt.configuration import OPTConfig

    d = X.shape[1]
    ds = torch.utils.data.TensorDataset(torch.tensor(X).double(), torch.tensor(y01[:, None]).double())
    cfg = OPTConfig(
        log_level="ERROR",
        n_epochs=n_epochs,
        batchsize=b,
        learning_rate=lr,
        loss="hinge",
        concretization_freq=F,
        k_substitution=k,
        input_domain=input_domain,
        label_domain=None,
        gradient_clipping=gamma,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 0, "Threads": threads, "TimeLimit": timeout},
        obbt=obbt,
        obbt_timeout=obbt_timeout,
        metadata={"seed": 0, "strict_mode": True},
        **SEMANTICS[semantics],
    )
    # model_fn lets a ReLU net (or any nn.Sequential the formulation supports) be
    # bounded here; default stays the zero-init linear head so existing runs are
    # bit-identical.
    m = model_fn(d) if model_fn is not None else linear_model(d)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm, _ = certified_training(m, cfg, ds, solve=True)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(mm.param_l), f(mm.param_n), f(mm.param_u), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--n_s", type=int, default=4)
    ap.add_argument("--b", type=int, default=0, help="0 = full batch")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--domain", action="store_true", help="constrain the substituted record to the data box (realizable)"
    )
    a = ap.parse_args()
    b = a.b if a.b > 0 else a.N
    spe = max(1, a.N // b)
    n_ep = max(1, a.n_s // spe)
    X, y01, _, _ = load_blobs(a.N, a.d, sep=4.0, seed=0)
    ys = y01.copy()
    y01 = (y01 > 0).astype(float)  # opt path wants {0,1}
    P = np.c_[X, np.ones(a.N)]
    dom = (X.min(0) - 0.02, X.max(0) + 0.02) if a.domain else None
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(P.T @ P).max()) / a.N))
    print(
        f"N={a.N} b={b} ({'full' if b == a.N else 'mini'}) d={a.d} n_s={a.n_s} "
        f"epochs={n_ep} steps/epoch={spe} lr={lr:.3f} gamma={a.gamma} lambda=0 k=1 "
        f"substitute={'REALIZABLE (input_domain)' if a.domain else 'free box'}"
    )

    pl, pn, pu = agt_bound(
        linear_model(a.d), X, ys, k=1, b=b, n_epochs=n_ep, lr=lr, gamma=a.gamma, l2_reg=0.0
    )
    ibp = A_of_k(pl, pn, pu)
    print(f"\n{'semantics':>10}{'F':>5}{'F.b':>7}{'A^1 (l1)':>13}{'vs IBP':>9}{'t(s)':>9}")
    print(f"{'IBP':>10}{1:>5}{b:>7}{ibp:>13.5e}{1.0:>9.2f}{'-':>9}", flush=True)
    rows = [dict(semantics="IBP", F=1, A1=ibp, ratio=1.0)]
    F = 1
    while F <= a.n_s:
        for sem in ("MILP",) if F > 1 else ("MILP",):
            try:
                l, n, u, t = opt_bound(
                    X,
                    y01,
                    k=1,
                    b=b,
                    n_epochs=n_ep,
                    lr=lr,
                    gamma=a.gamma,
                    F=F,
                    semantics=sem,
                    timeout=a.timeout,
                    threads=a.threads,
                    input_domain=dom,
                )
                A = A_of_k(l, n, u)
                rows.append(dict(semantics=sem, F=F, A1=A, ratio=A / ibp, t=t))
                print(f"{sem:>10}{F:>5}{F * b:>7}{A:>13.5e}{A / ibp:>9.3f}{t:>9.1f}", flush=True)
            except Exception as e:
                print(f"{sem:>10}{F:>5}{F * b:>7}   FAILED {type(e).__name__}: {str(e)[:44]}", flush=True)
        F *= 2
    json.dump(
        dict(N=a.N, d=a.d, n_s=a.n_s, gamma=a.gamma, lam=0.0, rows=rows),
        open(f"ags/blobs/runs/p1_N{a.N}_b{b}_ns{a.n_s}{'_dom' if a.domain else ''}.json", "w"),
        indent=1,
    )
    print(f"\nwrote ags/blobs/runs/p1_N{a.N}_b{b}_ns{a.n_s}.json")
