"""AGS on a contractive trajectory against matched and optimized DP-SGD and PATE, budgets to 1000.

The lever this driver pulls is L2 regularization: with lambda above the interval propagation's own
growth constant (about 0.2 on PCA-16), the certified envelope contracts and saturates at
2 gamma k / (b (lambda - c)) whatever the trajectory length, while matched DP-SGD keeps paying the
step count under composition. The release is one multivariate draw on the whole parameter vector
(common/vector_release.py): the l1 cross-radius certificate over all P coordinates, one smooth
sensitivity with the sound beta/P rule, one epsilon; no per-parameter budgets, no anisotropic split.
Pure DP releases Cauchy; approx DP takes the better of Laplace and the Cauchy release, both of which
are (epsilon, delta)-DP, and stores both.

Stages, each resumable (envelopes per configuration and radius, JSON per stage):

    sweep      certify a grid on a 10k subset of the private split, release every variant, rank
               configurations by their margin over quick matched pure DP-SGD at large epsilon
    baselines  matched DP-SGD (pure: basic composition; approx: basic and exact Gaussian
               composition, both drawn), optimized DP-SGD (Poisson + PLD / amplified basic), PATE
    final      re-certify the winning configurations on the full private split
    windows    Algorithm 2: n releases with envelope collapse at each released point, chained from
               the noisy release, budgets composed; pure DP, a few seeds, the best configuration
    release    the headline JSON the figure reads (plot_contractive.py)

    PYTHONPATH=. python -m ags.mnist.run_contractive sweep --task mnist10 --device cuda:0
    PYTHONPATH=. python -m ags.mnist.run_contractive all --task smoke          # every path, minutes

run_contractive.sh runs the two tasks on the GPU with the baselines on the CPU under one deadline.
Settings: ags/mnist/config.py::CONTRACTIVE (10 classes), CONTRACTIVE_5 (digits 0-4).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from ags.mnist import baselines as B
from ags.mnist import ss_reference as R
from ags.mnist.ags import agt_bound
from ags.mnist.config import (
    CONTRACTIVE,
    CONTRACTIVE_5,
    CONTRACTIVE_5_DENSE,
    CONTRACTIVE_5X,
    CONTRACTIVE_5Y,
    CONTRACTIVE_SMOKE,
    CONTRACTIVE_X,
    CONTRACTIVE_Y,
)
from ags.mnist.shared import MNIST_DIR, _idx, accuracy, logistic_model, run_id
from common import vector_release as V
from common.accounting.accounting import generic_pld
from common.provenance import stamp

TASKS = {
    "mnist10": CONTRACTIVE,
    "mnist5": CONTRACTIVE_5,
    "mnist5_dense": CONTRACTIVE_5_DENSE,
    "mnist10x": CONTRACTIVE_X,
    "mnist5x": CONTRACTIVE_5X,
    "mnist10y": CONTRACTIVE_Y,
    "mnist5y": CONTRACTIVE_5Y,
    "smoke": CONTRACTIVE_SMOKE,
}
MODES = ("pure", "approx")
CERTS = ("cross_radius", "width")
RULES = ("aggregate", "dimension_free")
MECHS = {"pure": ("cauchy",), "approx": ("laplace", "cauchy")}
DEFAULT_VARIANT = (
    "tight/cross_radius"  # the tightest proven release: best of the l1 and anisotropic allocations
)
TABLE_VARIANTS = (
    ("tight", "tight/cross_radius"),
    ("aggr", "aggregate/cross_radius"),
    ("cor/p", "corollary_p/cross_radius"),
    ("dim-free", "dimension_free/cross_radius"),
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def _dump(path, obj):
    """Atomic: a reader never sees a half-written file, even with two stages running side by side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------
# data: the task's digits, PCA-d, a public pool and selection set, the sweep subset, the full split
# ------------------------------------------------------------------------------------------------

_DATA = {}


def load_task(C, d):
    key = (C["task"], int(d))
    if key in _DATA:
        return _DATA[key]
    from sklearn.decomposition import PCA

    xtr, ytr = (
        _idx(f"{MNIST_DIR}/train-images-idx3-ubyte.gz", False),
        _idx(f"{MNIST_DIR}/train-labels-idx1-ubyte.gz", True),
    )
    xte, yte = (
        _idx(f"{MNIST_DIR}/t10k-images-idx3-ubyte.gz", False),
        _idx(f"{MNIST_DIR}/t10k-labels-idx1-ubyte.gz", True),
    )
    lut = {c: i for i, c in enumerate(C["classes"])}
    keep_tr, keep_te = np.isin(ytr, C["classes"]), np.isin(yte, C["classes"])
    xtr, ytr = xtr[keep_tr], np.array([lut[v] for v in ytr[keep_tr]], dtype=np.int64)
    xte, yte = xte[keep_te], np.array([lut[v] for v in yte[keep_te]], dtype=np.int64)
    X = (xtr.astype(np.float64) / 255.0 - 0.1307) / 0.3081
    Xte = (xte.astype(np.float64) / 255.0 - 0.1307) / 0.3081
    pca = PCA(n_components=d, whiten=True, random_state=C["seed"])
    X, Xte = pca.fit_transform(X), pca.transform(Xte)
    r = np.linalg.norm(X, axis=1).max()
    X, Xte = X / r, Xte / r
    perm = np.random.RandomState(C["seed"] + 11).permutation(len(X))
    val, rest = perm[: C["n_val"]], perm[C["n_val"] :]
    pool, sel = val[: C["n_pool"]], val[C["n_pool"] :]
    sweep = rest[: C["n_sweep"]]
    D = dict(
        C=len(C["classes"]),
        d=int(d),
        Xfull=X[rest],
        yfull=ytr[rest],
        Xsweep=X[sweep],
        ysweep=ytr[sweep],
        Xpool=X[pool],
        ypool=ytr[pool],
        Xsel=X[sel],
        ysel=ytr[sel],
        Xte=Xte,
        yte=yte,
    )
    _DATA[key] = D
    log(
        f"task {C['task']} d={d}: private {len(rest)} (sweep {len(sweep)}), pool {len(pool)}, select {len(sel)}, test {len(yte)}"
    )
    return D


def train_set(D, scale):
    return (D["Xsweep"], D["ysweep"]) if scale == "sweep" else (D["Xfull"], D["yfull"])


def n_params(D):
    return D["C"] * (D["d"] + 1)


def scores(theta, D):
    return dict(
        select=float(accuracy(theta, D["Xsel"], D["ysel"], D["C"])),
        test=float(accuracy(theta, D["Xte"], D["yte"], D["C"])),
    )


def cfg_key(cfg, scale):
    return run_id(f"con_{scale}", cfg)


# ------------------------------------------------------------------------------------------------
# certification: one file per configuration and radius, a manifest per configuration
# ------------------------------------------------------------------------------------------------


def certify(C, D, cfg, scale, device, *, key=None, theta0=None, T=None, radii=None):
    key = key or cfg_key(cfg, scale)
    man_path = f"{C['runs_dir']}/envelopes/{key}.json"
    man = _load(man_path, None)
    if man and man.get("complete"):
        return man
    os.makedirs(f"{C['runs_dir']}/bounds", exist_ok=True)
    X, y = train_set(D, scale)
    b, T = len(y), int(T or cfg["T"])
    radii = [k for k in (radii or C["k_grid"]) if k < b] + [b]
    paths, theta_n, t0 = {}, None, time.time()
    for k in radii:
        path = f"{C['runs_dir']}/bounds/{key}_k{k}.npz"
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            pl, pn, pu = z["param_l"], z["param_n"], z["param_u"]
        else:
            tk = time.time()
            pl, pn, pu = agt_bound(
                logistic_model(D["d"], D["C"]),
                X,
                y,
                k=k,
                b=b,
                n_epochs=T,
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                l2_reg=cfg["lam"],
                lr_decay=cfg["decay"],
                n_classes=D["C"],
                device=device,
                bound_model=C["bound_model"],
                theta0=theta0,
            )
            tmp = f"{path}.tmp.{os.getpid()}.npz"  # atomic: two processes may certify the same finalist
            np.savez_compressed(
                tmp,
                param_l=pl,
                param_n=pn,
                param_u=pu,
                cfg=np.array(json.dumps({**cfg, "k": k, "T": T, "b": b})),
                seconds=time.time() - tk,
            )
            os.replace(tmp, path)
            log(f"  {key} k={k:<6} width mean {np.abs(pu - pl).mean():.3e}  {time.time() - tk:.1f}s")
        if theta_n is None:
            theta_n = pn
        elif not np.allclose(theta_n, pn):
            raise RuntimeError(f"{key}: nominal differs across radii at k={k}")
        paths[str(k)] = path
    man = dict(
        cfg=cfg,
        key=key,
        scale=scale,
        b=b,
        T=T,
        radii=radii,
        paths=paths,
        theta_n=theta_n.tolist(),
        nominal=scores(theta_n, D),
        from_theta0=theta0 is not None,
        seconds=time.time() - t0,
        provenance=stamp({**cfg, "T": T, "b": b, "bound_model": C["bound_model"], "task": C["task"]}),
        complete=True,
    )
    _dump(man_path, man)
    log(f"{key}: {cfg} T={T} b={b}  nominal {man['nominal']}  {man['seconds']:.0f}s")
    return man


def load_boxes(man):
    boxes = {}
    for k, path in man["paths"].items():
        z = np.load(path, allow_pickle=True)
        boxes[int(k)] = (z["param_l"], z["param_u"])
    return boxes, np.asarray(man["theta_n"], float)


# ------------------------------------------------------------------------------------------------
# the release: whole-vector, l1 cross-radius, beta/P; every variant stored
# ------------------------------------------------------------------------------------------------


def release_stats(theta_n, sigma, draw, seeds, D):
    sel, te = [], []
    for s in range(seeds):
        th = R.release(theta_n, sigma, np.random.RandomState(s), mode=draw)
        sel.append(accuracy(th, D["Xsel"], D["ysel"], D["C"]))
        te.append(accuracy(th, D["Xte"], D["yte"], D["C"]))
    return dict(select=float(np.median(sel)), test=float(np.median(te)))


def mechanism(mech, eps, P, C):
    """(beta, draw mode) for a mechanism: the reference's constants."""
    if mech == "laplace":
        return R.beta_for(eps, mode="approx", n_params=P, delta=C["delta"]), "approx"
    return R.beta_for(eps), "pure"


def evaluate(C, D, man, seeds):
    P = n_params(D)
    boxes, theta_n = load_boxes(man)
    grids = {
        "cross_radius": V.cross_radius_grid(theta_n, boxes, man["b"]),
        "width": V.width_grid(boxes, man["b"]),
    }
    out = {}
    for mode in MODES:
        out[mode] = {}
        for eps in C["eps"]:
            row = {}
            for mech in MECHS[mode]:
                beta, draw = mechanism(mech, eps, P, C)
                for rule in RULES:
                    beta_eff = V.effective_beta(beta, P, rule)
                    for cert, G in grids.items():
                        sigma, S = V.noise_scales(
                            G, beta_eff, eps, mode=draw, allocation="l1", gamma_cauchy=C["gamma_cauchy"]
                        )
                        st = release_stats(theta_n, sigma, draw, seeds, D)
                        row.setdefault(rule, {}).setdefault(cert, {})[mech] = dict(
                            st, sigma=float(sigma[0]), SS=float(S[0]), beta=float(beta)
                        )
            out[mode][str(eps)] = row
    return out


def best_mech(cell):
    """cell: {mech: stats} -> (mech, stats) with the best selection accuracy."""
    m = max(cell, key=lambda k: cell[k]["select"])
    return m, cell[m]


def evaluate_tight(C, D, man, seeds):
    """The proven rules (common/vector_release.tight_variants) on a stored envelope, computed once into
    runs/<task>/releases_tight/<key>.json; the sweep and final files are left as they were written."""
    path = f"{C['runs_dir']}/{C.get('tight_dir', 'releases_tight')}/{man['key']}.json"
    T = _load(path, None)
    if T:
        return T
    boxes, theta_n = load_boxes(man)
    T = V.tight_variants(
        theta_n,
        boxes,
        man["b"],
        C["eps"],
        delta=C["delta"],
        seeds=seeds,
        score=lambda th: scores(th, D),
        draw=lambda th, sigma, rng, mode: R.release(th, sigma, rng, mode=mode),
    )
    T["provenance"] = stamp({"key": man["key"], "cfg": man["cfg"], "seeds": seeds})
    _dump(path, T)
    log(f"  tight release {man['key']}: {path}")
    return T


def variants_tight(row):
    """row = tight releases[mode][eps] -> {"rule/cert": {mech: stats}}, plus "tight/cert": the union of the
    two allocations' mechanisms (keys "mech@l1", "mech@aniso") so the selection set picks the allocation."""
    out = {}
    for rule, certs in row.items():
        for cert, cell in certs.items():
            out[f"{rule}/{cert}"] = cell
            if rule.startswith("tight_"):
                tag = "l1" if rule == "tight_l1" else "aniso"
                out.setdefault(f"tight/{cert}", {}).update({f"{m}@{tag}": st for m, st in cell.items()})
    return out


# ------------------------------------------------------------------------------------------------
# sweep: rank configurations by their margin over quick matched pure DP-SGD at large epsilon
# ------------------------------------------------------------------------------------------------


def quick_matched_pure(C, D, cfg, scale, eps_list, seeds):
    X, y = train_set(D, scale)
    out = {}
    for eps in eps_list:
        accs = []
        for s in range(seeds):
            th, _ = B.dpsgd_pure(
                X,
                y,
                eps,
                n_s=cfg["T"],
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                lam=cfg["lam"],
                b=len(y),
                seed=s,
                n_classes=D["C"],
                lr_decay=cfg["decay"],
            )
            accs.append(accuracy(th, D["Xsel"], D["ysel"], D["C"]))
        out[str(eps)] = float(np.median(accs))
    return out


def score(C, entry):
    ags = [
        entry["releases"]["pure"][str(e)]["aggregate"]["cross_radius"]["cauchy"]["select"]
        for e in C["select_eps"]
    ]
    dp = [entry["quick_matched"][str(e)] for e in C["select_eps"]]
    return dict(margin=float(np.mean(ags) - np.mean(dp)), ags=float(np.mean(ags)), dpsgd=float(np.mean(dp)))


def sweep_path(C):
    """This process's sweep file. An extra sweep (another grid, another GPU) writes its own file, and
    every later stage reads the union, so two sweeps never write the same JSON."""
    return f"{C['runs_dir']}/{C.get('sweep_file', 'sweep.json')}"


def load_sweeps(C):
    import glob

    S = {}
    for f in sorted(glob.glob(f"{C['runs_dir']}/sweep*.json")):
        S.update(_load(f, {}))
    return S


def ranked(C, S, keys=None):
    keys = list(S) if keys is None else keys
    return sorted(
        (k for k in keys if S[k].get("complete")),
        key=lambda k: (S[k]["score"]["margin"], S[k]["score"]["ags"]),
        reverse=True,
    )


def run_sweep(C, a):
    S = _load(sweep_path(C), {})

    def do(cfg, stage):
        key = cfg_key(cfg, "sweep")
        if key in S and S[key].get("complete"):
            return key
        if time.time() > a.deadline:
            log("deadline reached; sweep stops here")
            return None
        D = load_task(C, cfg["d"])
        man = certify(C, D, cfg, "sweep", a.device)
        t0 = time.time()
        rel = evaluate(C, D, man, C["seeds"])
        qm = quick_matched_pure(C, D, cfg, "sweep", C["select_eps"], C["select_seeds"])
        S[key] = dict(
            cfg=cfg,
            key=key,
            stage=stage,
            nominal=man["nominal"],
            b=man["b"],
            releases=rel,
            quick_matched=qm,
            complete=True,
        )
        S[key]["score"] = score(C, S[key])
        _dump(sweep_path(C), S)
        sc = S[key]["score"]
        peek = {
            e: round(rel["pure"][str(e)]["aggregate"]["cross_radius"]["cauchy"]["test"], 3)
            for e in C["select_eps"]
        }
        log(
            f"  released {key} in {time.time() - t0:.0f}s: nominal {man['nominal']['test']:.3f}; AGS pure test {peek}; margin over matched {sc['margin']:+.3f} (AGS {sc['ags']:.3f} vs DP {sc['dpsgd']:.3f})"
        )
        return key

    s1 = C["stage1"]
    stage1_keys = []
    configs = C.get("stage1_list") or [
        dict(d=d, lam=lam, gamma=gamma, lr=s1["lr"], decay=s1["decay"], T=s1["T"])
        for d, lam, gamma in itertools.product(s1["d"], s1["lam"], s1["gamma"])
    ]
    for cfg in configs:
        k = do(cfg, 1)
        if k:
            stage1_keys.append(k)
    for base in ranked(C, S, stage1_keys)[: C["stage2"]["top"]]:
        for var in C["stage2"]["variants"]:
            do({**S[base]["cfg"], **var}, 2)
    for k in ranked(C, S)[:5]:
        log(
            f"rank: {k} {S[k]['cfg']} margin {S[k]['score']['margin']:+.3f} AGS {S[k]['score']['ags']:.3f} DP {S[k]['score']['dpsgd']:.3f} nominal {S[k]['nominal']['test']:.3f}"
        )
    open(f"{C['runs_dir']}/SWEEP_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def finalists(C, S):
    if C["final"].get("list"):  # chosen by hand from the sweep's per-epsilon numbers
        return list(C["final"]["list"])
    out, seen = [], set()
    for k in ranked(C, S):
        cfg = S[k]["cfg"]
        ck = json.dumps(cfg, sort_keys=True)
        if ck not in seen:
            seen.add(ck)
            out.append(cfg)
        if len(out) == C["final"]["top"]:
            break
    return out


def final_path(C):
    return f"{C['runs_dir']}/final.json"


def run_final(C, a):
    S = load_sweeps(C)
    F = _load(final_path(C), {})
    for cfg in finalists(C, S):
        key = cfg_key(cfg, "full")
        if key in F:
            continue
        if time.time() > a.deadline:
            log("deadline reached; final stops here")
            break
        D = load_task(C, cfg["d"])
        man = certify(C, D, cfg, "full", a.device, radii=C["final"].get("k_grid"))
        rel = evaluate(C, D, man, C["seeds"])
        F[key] = dict(cfg=cfg, key=key, nominal=man["nominal"], b=man["b"], releases=rel, complete=True)
        _dump(final_path(C), F)
        peek = {
            e: round(rel["pure"][str(e)]["aggregate"]["cross_radius"]["cauchy"]["test"], 3)
            for e in C["select_eps"]
        }
        log(f"final {key}: nominal {man['nominal']['test']:.3f}; AGS pure test {peek}")
    open(f"{C['runs_dir']}/FINAL_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------------------------------------
# Algorithm 2: windows with collapse, chained from the noisy release, budgets composed
# ------------------------------------------------------------------------------------------------


def window_budget(C, eps, n, mode):
    """(eps_w, delta_w) per window so that n releases compose to (eps, delta); pure: eps/n exactly."""
    if mode == "pure":
        return eps / n, 0.0
    delta_w = C["delta"] / (2 * n)
    lo, hi = 0.0, eps

    def total(e):
        return generic_pld(e, delta_w, C["pld_interval"]).self_compose(n).get_epsilon_for_delta(C["delta"])

    for _ in range(40):
        mid = (lo + hi) / 2
        if total(mid) <= eps:
            lo = mid
        else:
            hi = mid
    return lo, delta_w


def run_windows(C, a):
    S = load_sweeps(C)
    if not S:
        log("no sweep; windows skipped")
        return
    cfg = finalists(C, S)[0]  # the figure's trajectory
    D = load_task(C, cfg["d"])
    P = n_params(D)
    Wj = _load(f"{C['runs_dir']}/windows.json", {"cfg": cfg, "runs": {}})
    single = S[cfg_key(cfg, "sweep")]["releases"]["pure"]
    for n in C["windows"]["n"]:
        for eps in C["windows"]["eps"]:
            tag = f"n{n}_eps{eps}"
            if tag in Wj["runs"] and len(Wj["runs"][tag]["test"]) >= C["windows"]["seeds"]:
                continue
            if time.time() > a.deadline:
                log("deadline reached; windows stop here")
                _dump(f"{C['runs_dir']}/windows.json", Wj)
                return
            eps_w, _ = window_budget(C, eps, n, "pure")
            beta = R.beta_for(eps_w)
            tests, sels, sigmas = [], [], []
            for seed in range(C["windows"]["seeds"]):
                theta = None
                for w in range(n):
                    key = f"{cfg_key(cfg, 'sweep')}_win{n}_eps{eps}_s{seed}_w{w}"
                    man = certify(
                        C,
                        D,
                        cfg,
                        "sweep",
                        a.device,
                        key=key,
                        theta0=theta,
                        T=max(1, cfg["T"] // n),
                        radii=C["windows"]["k_grid"],
                    )
                    boxes, theta_n = load_boxes(man)
                    G = V.cross_radius_grid(theta_n, boxes, man["b"])
                    sigma, _ = V.noise_scales(
                        G, beta / P, eps_w, mode="pure", allocation="l1", gamma_cauchy=C["gamma_cauchy"]
                    )
                    theta = R.release(theta_n, sigma, np.random.RandomState(1000 * seed + w), mode="pure")
                    sigmas.append(float(sigma[0]))
                sc = scores(theta, D)
                tests.append(sc["test"])
                sels.append(sc["select"])
            Wj["runs"][tag] = dict(
                n=n,
                eps=eps,
                eps_w=eps_w,
                test=tests,
                select=sels,
                sigma_mean=float(np.mean(sigmas)),
                single_release_test=single[str(eps)]["aggregate"]["cross_radius"]["cauchy"]["test"],
            )
            _dump(f"{C['runs_dir']}/windows.json", Wj)
            log(
                f"windows n={n} eps={eps}: test median {np.median(tests):.3f} (seeds {tests}) vs single release {Wj['runs'][tag]['single_release_test']:.3f}"
            )


# ------------------------------------------------------------------------------------------------
# baselines: matched on the finalists' trajectories, optimized DP-SGD, PATE
# ------------------------------------------------------------------------------------------------


def _median_runs(fn, seeds, D):
    sel, te, extra = [], [], None
    for s in range(seeds):
        th, extra = fn(s)
        sel.append(accuracy(th, D["Xsel"], D["ysel"], D["C"]))
        te.append(accuracy(th, D["Xte"], D["yte"], D["C"]))
    return dict(select=float(np.median(sel)), test=float(np.median(te)), mechanism=float(extra))


def run_baselines(C, a, scale):
    S = load_sweeps(C)
    if not S:
        log("no sweep yet; baselines skipped")
        return
    cfgs = finalists(C, S)
    seeds = C["tune_seeds"] if scale == "sweep" else C["final_train_seeds"]
    path = f"{C['runs_dir']}/baselines_{scale}.json"
    Bl = _load(
        path,
        {
            "matched": {},
            "dpsgd_optimized": {m: {"grid": {}, "selected": {}} for m in MODES},
            "pate": {m: {"grid": {}, "selected": {}} for m in MODES},
            "seeds": seeds,
        },
    )

    # ---- matched DP-SGD on each finalist's trajectory
    for cfg in cfgs:
        key = cfg_key(cfg, scale)
        D = load_task(C, cfg["d"])
        X, y = train_set(D, scale)
        M = Bl["matched"].setdefault(key, {"cfg": cfg, "pure": {}, "approx": {}})
        for mode in MODES:
            for eps in C["eps"]:
                if str(eps) in M[mode] or time.time() > a.deadline:
                    continue
                kw = dict(
                    n_s=cfg["T"],
                    lr=cfg["lr"],
                    gamma=cfg["gamma"],
                    lam=cfg["lam"],
                    b=len(y),
                    n_classes=D["C"],
                    lr_decay=cfg["decay"],
                )
                if mode == "pure":
                    M[mode][str(eps)] = dict(
                        basic=_median_runs(lambda s: B.dpsgd_pure(X, y, eps, seed=s, **kw), seeds, D)
                    )
                else:
                    M[mode][str(eps)] = dict(
                        basic=_median_runs(
                            lambda s: B.dpsgd_gaussian(X, y, eps, C["delta"], seed=s, **kw), seeds, D
                        ),
                        composed=_median_runs(
                            lambda s: B.dpsgd_gaussian_composed(X, y, eps, C["delta"], seed=s, **kw), seeds, D
                        ),
                    )
                _dump(path, Bl)
                log(f"matched {scale} {key} {mode} eps={eps}: {M[mode][str(eps)]}")

    # ---- optimized DP-SGD and PATE, on the top finalist's features
    cfg = cfgs[0]
    D = load_task(C, cfg["d"])
    X, y = train_set(D, scale)
    O = Bl["dpsgd_optimized"]
    for mode in MODES:
        grid = C["dpsgd_opt_approx"] if mode == "approx" else C["dpsgd_opt_pure"]
        keys = ["q", "steps", "lr", "clip" if mode == "approx" else "gamma"]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

        def runner(g, eps):
            if mode == "approx":
                return lambda s: B.dpsgd_poisson_gaussian(
                    X,
                    y,
                    eps,
                    C["delta"],
                    steps=g["steps"],
                    lr=g["lr"],
                    clip=g["clip"],
                    q=g["q"],
                    seed=s,
                    n_classes=D["C"],
                    interval=C["pld_interval"],
                )
            return lambda s: B.dpsgd_poisson_laplace(
                X, y, eps, steps=g["steps"], lr=g["lr"], gamma=g["gamma"], q=g["q"], seed=s, n_classes=D["C"]
            )

        for eps in C["eps"]:
            if str(eps) in O[mode]["selected"] or time.time() > a.deadline:
                continue
            cells = O[mode]["grid"].setdefault(str(eps), {})
            for g in combos:
                gk = json.dumps(g, sort_keys=True)
                if gk not in cells:
                    cells[gk] = _median_runs(runner(g, eps), C["tune_seeds"], D)
                    _dump(path, Bl)
            g = json.loads(max(cells, key=lambda k: cells[k]["select"]))
            O[mode]["selected"][str(eps)] = dict(_median_runs(runner(g, eps), seeds, D), config=g)
            _dump(path, Bl)
            log(f"dpsgd optimized {scale} {mode} eps={eps}: {O[mode]['selected'][str(eps)]}")

    pc, Xpool, votes = C["pate"], D["Xpool"], {}

    def counts_for(m, seed):
        if (m, seed) not in votes:
            _, counts, _ = B._teachers_and_votes(
                X,
                y,
                Xpool,
                n_teachers=m,
                n_queries=len(Xpool),
                n_s=pc["n_s"],
                lr=pc["lr"],
                gamma=pc["gamma"],
                lam=0.0,
                loss=B.DEFAULT_LOSS,
                rng=np.random.RandomState(seed),
                n_classes=D["C"],
            )
            votes[(m, seed)] = counts
        return votes[(m, seed)]

    def pate_run(mode, eps, m, Q, seed):
        counts = counts_for(m, seed)[:, :Q]
        rng = np.random.RandomState(10_000 + seed)
        if mode == "pure":
            eps_q = eps / Q
            noisy, mech = counts + rng.laplace(0.0, 2.0 / eps_q, size=counts.shape), eps_q
        else:
            eps_q, delta_q = eps / Q, C["delta"] / Q
            mech = (np.sqrt(2.0) / eps_q) * np.sqrt(2.0 * np.log(1.25 / delta_q))
            noisy = counts + rng.normal(0.0, mech, size=counts.shape)
        student = B.train(
            Xpool[:Q],
            noisy.argmax(0),
            n_s=pc["student_n_s"],
            lr=pc["lr"],
            gamma=pc["gamma"],
            lam=0.0,
            n_classes=D["C"],
        )
        return student, mech

    T = Bl["pate"]
    for mode in MODES:
        for eps in C["eps"]:
            if str(eps) in T[mode]["selected"] or time.time() > a.deadline:
                continue
            cells = T[mode]["grid"].setdefault(str(eps), {})
            for m in pc["teachers"]:
                for Q in pc["queries"]:
                    if Q > len(Xpool):
                        continue
                    gk = json.dumps(dict(teachers=m, queries=Q))
                    if gk not in cells:
                        cells[gk] = _median_runs(lambda s: pate_run(mode, eps, m, Q, s), C["tune_seeds"], D)
            g = json.loads(max(cells, key=lambda k: cells[k]["select"]))
            T[mode]["selected"][str(eps)] = dict(
                _median_runs(lambda s: pate_run(mode, eps, g["teachers"], g["queries"], s), seeds, D),
                config=g,
            )
            _dump(path, Bl)
            log(f"pate {scale} {mode} eps={eps}: {T[mode]['selected'][str(eps)]}")
    _dump(path, Bl)


# ------------------------------------------------------------------------------------------------
# the headline JSON the figure reads
# ------------------------------------------------------------------------------------------------


def variants(row):
    """row = releases[mode][eps] -> {"rule/cert": {mech: stats}}; {} for a budget the sweep did not evaluate."""
    return {f"{rule}/{cert}": row[rule][cert] for rule in RULES for cert in CERTS if rule in row}


def run_release(C, a):
    S = load_sweeps(C)
    F = _load(final_path(C), {})
    scale = "full" if (F and a.scale != "sweep") else "sweep"
    if scale == "full":
        entries = F
    else:  # the sweep-scale headline: one trajectory, C["sweep_figure"] (the long one) or the finalist
        cfgs = C.get("sweep_figure") or finalists(C, S)
        entries = {
            cfg_key(cfg, "sweep"): S[cfg_key(cfg, "sweep")] for cfg in cfgs if cfg_key(cfg, "sweep") in S
        }
    Bl = _load(
        f"{C['runs_dir']}/baselines_{scale}.json",
        {
            "matched": {},
            "dpsgd_optimized": {m: {"selected": {}} for m in MODES},
            "pate": {m: {"selected": {}} for m in MODES},
        },
    )
    if not entries:
        log("nothing to release")
        return
    tight = {}
    for key, entry in entries.items():
        man = _load(f"{C['runs_dir']}/envelopes/{key}.json", None)
        if man is None:
            raise FileNotFoundError(f"no envelope manifest for {key}")
        tight[key] = evaluate_tight(C, load_task(C, entry["cfg"]["d"]), man, C["seeds"])
    H = dict(
        task=C["task"],
        scale=scale,
        eps=C["eps"],
        delta=C["delta"],
        configs={k: dict(cfg=e["cfg"], nominal=e["nominal"], b=e["b"]) for k, e in entries.items()},
        arms={},
        windows=_load(f"{C['runs_dir']}/windows.json", None),
        provenance=stamp(),
    )
    chosen_counts = {}
    for mode in MODES:
        H["arms"][mode] = {}
        for eps in C["eps"]:
            e = str(eps)
            ags = {}
            for key, entry in entries.items():
                T = tight[key]
                allv = {**variants(entry["releases"][mode].get(e, {})), **variants_tight(T[mode][e])}
                for name, cell in allv.items():
                    mech, st = best_mech(cell)
                    cur = ags.get(name)
                    if cur is None or st["select"] > cur["select"]:
                        ags[name] = dict(
                            st, mechanism=mech, key=key, cfg=entry["cfg"], nominal=entry["nominal"]
                        )
            chosen = ags[DEFAULT_VARIANT]["key"]
            chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
            matched = Bl["matched"].get(chosen, {}).get(mode, {}).get(e)
            H["arms"][mode][e] = dict(
                ags=ags,
                dpsgd_matched=matched,
                dpsgd_optimized=Bl["dpsgd_optimized"][mode]["selected"].get(e),
                pate=Bl["pate"][mode]["selected"].get(e),
                non_private=entries[chosen]["nominal"]["test"],
            )
    ref = max(chosen_counts, key=chosen_counts.get)
    H["non_private_line"] = entries[ref]["nominal"]["test"]
    H["reference_config"] = dict(key=ref, cfg=entries[ref]["cfg"])
    out = (
        f"{C['runs_dir']}/{C.get('headline_file', 'headline.json')}"
        if scale == "full" or not F
        else f"{C['runs_dir']}/{C.get('headline_file', 'headline.json').replace('.json', '_sweep.json')}"
    )
    _dump(out, H)
    log(f"wrote {out} ({scale}; {len(entries)} configurations; reference {ref} {entries[ref]['cfg']})")
    g = lambda x: f"{x['test']:.3f}" if x else "  -  "
    for mode in MODES:
        heads = " ".join(f"{name:>8}" for name, _ in TABLE_VARIANTS)
        log(
            f"  {mode:6s} {'eps':>5} {'nonpriv':>7} {'PATE':>6} {'DP-m':>6} {'DP-m(c)':>7} {'DP-opt':>6} {heads} {'mech':>12}"
        )
        for eps in C["eps"]:
            r = H["arms"][mode][str(eps)]
            m = r["dpsgd_matched"] or {}
            cols = " ".join(f"{g(r['ags'].get(v)):>8}" for _, v in TABLE_VARIANTS)
            log(
                f"  {'':6s} {eps:>5} {r['non_private']:7.3f} {g(r['pate']):>6} {g(m.get('basic')):>6} {g(m.get('composed')):>7} {g(r['dpsgd_optimized']):>6} {cols} {r['ags'][DEFAULT_VARIANT]['mechanism']:>12}"
            )


# ------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("sweep", "baselines", "final", "windows", "release", "all"))
    ap.add_argument("--task", choices=tuple(TASKS), default="mnist10")
    ap.add_argument(
        "--scale",
        choices=("sweep", "full"),
        default=None,
        help="baselines: which split; release: force the sweep-scale headline",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deadline", type=float, default=float("inf"))
    a = ap.parse_args()
    C = TASKS[a.task]
    os.makedirs(C["runs_dir"], exist_ok=True)
    if a.command in ("sweep", "all"):
        run_sweep(C, a)
    if a.command == "baselines":
        run_baselines(C, a, a.scale or "sweep")
    if a.command == "all":
        run_baselines(C, a, "sweep")
    if a.command in ("final", "all"):
        run_final(C, a)
    if a.command == "all":
        run_baselines(C, a, "full")
    if a.command in ("windows", "all"):
        run_windows(C, a)
    if a.command in ("release", "all"):
        run_release(C, a)


if __name__ == "__main__":
    main()
