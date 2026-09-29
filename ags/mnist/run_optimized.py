"""Optimized AGS against matched and optimized DP-SGD and PATE on the MNIST PCA-64 logistic head.

The figure this produces (plot_optimized.py, fig5_combined_optimized) has, per panel (pure, approx):
non-private, PATE, DP-SGD (matched), DP-SGD (optimized), AGS (matched), AGS (optimized). Every tuned
arm picks its hyperparameters per epsilon on a public selection set and reports on the test set.

The AGS release is one multivariate draw on the final parameter vector, calibrated for the WHOLE
vector (common/vector_release.py): the l1 cross-radius certificate over all P coordinates, one smooth
sensitivity, one epsilon. Never noise per parameter with its own epsilon. Envelopes are genuine AGT
bound propagation on a dense-then-geometric radius grid up to the batch size, saved per configuration
and radius so the sweep resumes where it stopped. Amplification by one persistent without-replacement
subsample (paper Lemma 4.1) is a sweep dimension through q.

Every variant of the release is stored beside the primary one so the paper's choices can be compared
from the same envelopes: certificate (cross_radius | width, the reference's), beta rule (aggregate
beta/P, the sound one | dimension_free, the paper's Corollary 4.1 as written), allocation (l1 |
anisotropic | isotropic), and for approx DP the per-coordinate Laplace release composed by PLD.

    PYTHONPATH=. python -m ags.mnist.run_optimized envelopes --stage all --device cuda:0
    PYTHONPATH=. python -m ags.mnist.run_optimized baselines
    PYTHONPATH=. python -m ags.mnist.run_optimized release
    PYTHONPATH=. python -m ags.mnist.run_optimized all --smoke      # every code path in minutes

run_optimized.sh runs the first two in parallel under a wall-clock deadline, then the third and the
plot. Writes under ags/mnist/runs/optimized/ (settings in ags/mnist/config.py::OPTIMIZED).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time

import numpy as np

from ags.mnist import baselines as B
from ags.mnist import ss_reference as R
from ags.mnist.ags import agt_bound
from ags.mnist.config import OPTIMIZED, OPTIMIZED_SMOKE
from ags.mnist.shared import accuracy, load_mnist_pca, logistic_model, run_id
from common import vector_release as V
from common.accounting.accounting import base_budget_for_outer, calibrate_ags
from common.provenance import stamp

MODES = ("pure", "approx")
CERTS = ("cross_radius", "width")
SELECT_ALLOCATIONS = ("l1", "anisotropic")  # the optimized arm may choose between these two


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def n_params(C):
    return C["d"] * 10 + 10


# ------------------------------------------------------------------------------------------------
# data: private training split, public pool and selection set, test set
# ------------------------------------------------------------------------------------------------


def load_split(C):
    X, y, Xte, yte = load_mnist_pca(C["N"], C["d"], seed=C["seed"], n_test=C["n_test"])
    perm = np.random.RandomState(C["seed"] + 11).permutation(len(X))
    val, tr = perm[: C["n_val"]], perm[C["n_val"] :]
    pool, sel = val[: C["n_pool"]], val[C["n_pool"] :]
    return dict(
        Xtr=X[tr], ytr=y[tr], Xpool=X[pool], ypool=y[pool], Xsel=X[sel], ysel=y[sel], Xte=Xte, yte=yte
    )


def resolve(cfg, n_train):
    """Batch size, steps and epochs a configuration means on (a fraction q of) the training split."""
    q = float(cfg.get("q", 1.0))
    m = int(round(q * n_train))
    b = m if cfg["b"] == "full" else m // 5
    spe = m // b
    n_epochs = math.ceil(cfg["n_s"] / spe)
    return dict(q=q, m=m, b=b, spe=spe, n_epochs=n_epochs, steps=n_epochs * spe, rows=spe * b)


def cfg_key(cfg):
    return run_id("opt", cfg)


def subsample(D, C, res):
    """The persistent without-replacement subsample the amplification lemma needs (q < 1)."""
    X, y = D["Xtr"], D["ytr"]
    if res["q"] < 1:
        idx = np.random.RandomState(C["seed"] + 7).choice(len(X), res["m"], replace=False)
        X, y = X[idx], y[idx]
    return X[: res["rows"]], y[: res["rows"]]


def scores(theta, D):
    return dict(
        select=float(accuracy(theta, D["Xsel"], D["ysel"])), test=float(accuracy(theta, D["Xte"], D["yte"]))
    )


# ------------------------------------------------------------------------------------------------
# stage A: envelopes, one file per configuration and radius
# ------------------------------------------------------------------------------------------------


def certify(D, C, cfg, device):
    """Envelopes for one configuration on every radius of the grid; resumable."""
    res = resolve(cfg, len(D["Xtr"]))
    key = cfg_key(cfg)
    man_path = f"{C['runs_dir']}/envelopes/{key}.json"
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if man.get("complete"):
            return man
    os.makedirs(f"{C['runs_dir']}/envelopes", exist_ok=True)
    os.makedirs(f"{C['runs_dir']}/bounds", exist_ok=True)
    Xq, yq = subsample(D, C, res)
    radii = [k for k in C["k_grid"] if k < res["b"]] + [res["b"]]
    paths, theta_n, t0 = {}, None, time.time()
    for k in radii:
        rid = run_id("opt", {**cfg, "k": k, "N": C["N"], "seed": C["seed"]})
        path = f"{C['runs_dir']}/bounds/{rid}.npz"
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            pl, pn, pu = z["param_l"], z["param_n"], z["param_u"]
        else:
            tk = time.time()
            pl, pn, pu = agt_bound(
                logistic_model(C["d"]),
                Xq,
                yq,
                k=k,
                b=res["b"],
                n_epochs=res["n_epochs"],
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                l2_reg=0.0,
                lr_decay=cfg["decay"],
                device=device,
                bound_model=C["bound_model"],
            )
            np.savez_compressed(
                path,
                param_l=pl,
                param_n=pn,
                param_u=pu,
                cfg=np.array(json.dumps({**cfg, "k": k, **res}, default=str)),
                seconds=time.time() - tk,
            )
            log(f"  {key} k={k:<6} width mean {np.abs(pu - pl).mean():.3e}  {time.time() - tk:.1f}s")
        if theta_n is None:
            theta_n = pn
        elif not np.allclose(theta_n, pn):
            raise RuntimeError(f"{key}: nominal differs across radii at k={k}")
        paths[str(k)] = path
    man = dict(
        cfg=cfg,
        key=key,
        **res,
        radii=radii,
        paths=paths,
        theta_n=theta_n.tolist(),
        nominal=scores(theta_n, D),
        seconds=time.time() - t0,
        provenance=stamp({**cfg, **res, "bound_model": C["bound_model"]}),
        complete=True,
    )
    json.dump(man, open(man_path, "w"), indent=1)
    log(
        f"{key}: {cfg}  steps={res['steps']} b={res['b']} q={res['q']}  nominal {man['nominal']}  {man['seconds']:.0f}s"
    )
    return man


def load_boxes(man):
    boxes = {}
    for k, path in man["paths"].items():
        z = np.load(path, allow_pickle=True)
        boxes[int(k)] = (z["param_l"], z["param_u"])
    return boxes, np.asarray(man["theta_n"], float)


# ------------------------------------------------------------------------------------------------
# stage B: the AGS release from stored envelopes, every variant
# ------------------------------------------------------------------------------------------------


def release_stats(theta_n, sigma, mode, seeds, D):
    sel, te = [], []
    for s in range(seeds):
        th = R.release(theta_n, sigma, np.random.RandomState(s), mode=mode)
        sel.append(accuracy(th, D["Xsel"], D["ysel"]))
        te.append(accuracy(th, D["Xte"], D["yte"]))
    return dict(select=float(np.median(sel)), test=float(np.median(te)))


def pld_coordinate_budget(eps, delta, q, P, C, cache):
    """Per-coordinate (eps0, delta0) such that P Laplace releases composed by PLD meet (eps, delta)."""
    k = (eps, q)
    if k not in cache:
        interval = C["pld_interval"]
        for _ in range(3):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cal = calibrate_ags(
                        eps,
                        delta,
                        coordinates=P,
                        steps=1,
                        q=q,
                        mechanism="laplace",
                        interval=interval,
                        delta_fraction=0.5,
                    )
                break
            except RuntimeError:
                interval /= 10.0
        else:
            raise
        base_step_delta = delta * 0.5 / (q * 1)
        cache[k] = (cal.mechanism_parameter, base_step_delta / (2 * P))
    return cache[k]


def evaluate_config(D, C, man, pld_cache):
    """Release accuracies for one configuration under every (mode, eps, beta rule, certificate, allocation)."""
    P = n_params(C)
    boxes, theta_n = load_boxes(man)
    b, q = man["b"], man["q"]
    grids = {"cross_radius": V.cross_radius_grid(theta_n, boxes, b), "width": V.width_grid(boxes, b)}
    out = {}
    for mode in MODES:
        out[mode] = {}
        for eps in C["eps"]:
            delta = C["delta"] if mode == "approx" else 0.0
            eps_b, delta_b = base_budget_for_outer(eps, delta, q)
            beta = (
                R.beta_for(eps_b, mode="approx", n_params=P, delta=delta_b)
                if mode == "approx"
                else R.beta_for(eps_b)
            )
            row = {"eps_base": eps_b, "beta": beta}
            for rule in V.BETA_RULES:
                beta_eff = V.effective_beta(beta, P, rule)
                row[rule] = {}
                for cert, G in grids.items():
                    row[rule][cert] = {}
                    for alloc in V.ALLOCATIONS:
                        sigma, S = V.noise_scales(
                            G, beta_eff, eps_b, mode=mode, allocation=alloc, gamma_cauchy=C["gamma_cauchy"]
                        )
                        stats = release_stats(theta_n, sigma, mode, C["seeds"], D)
                        row[rule][cert][alloc] = dict(
                            stats, sigma_mean=float(sigma.mean()), S_mean=float(S.mean())
                        )
            if mode == "approx":
                eps0, delta0 = pld_coordinate_budget(eps, delta, q, P, C, pld_cache)
                beta_j = eps0 / (2.0 * math.log(2.0 / delta0))
                row["pld_coordinate"] = {"eps0": eps0, "delta0": delta0, "beta_j": beta_j}
                for cert, G in grids.items():
                    ssj = V.smooth_sensitivity(G, beta_j)
                    sigma = 2.0 * ssj / eps0
                    stats = release_stats(theta_n, sigma, mode, C["seeds"], D)
                    row["pld_coordinate"][cert] = {
                        "coordinate": dict(stats, sigma_mean=float(sigma.mean()), S_mean=float(ssj.mean()))
                    }
            out[mode][str(eps)] = row
    return out


def releases_path(C):
    return f"{C['runs_dir']}/ags_releases.json"


def evaluate_all(D, C, manifests):
    """Evaluate every complete manifest not yet in ags_releases.json; incremental."""
    path = releases_path(C)
    done = json.load(open(path)) if os.path.exists(path) else {}
    cache = {}
    for man in manifests:
        if man["key"] in done:
            continue
        t0 = time.time()
        done[man["key"]] = dict(
            cfg=man["cfg"],
            q=man["q"],
            b=man["b"],
            steps=man["steps"],
            nominal=man["nominal"],
            results=evaluate_config(D, C, man, cache),
        )
        json.dump(done, open(path, "w"), indent=1)
        r = done[man["key"]]["results"]
        peek = {m: r[m][str(C["eps"][-1])]["aggregate"]["cross_radius"]["l1"]["test"] for m in MODES}
        log(
            f"released {man['key']} in {time.time() - t0:.0f}s; l1/aggregate/cross-radius test acc at eps={C['eps'][-1]}: {peek}"
        )
    return done


def selection_score(entry, C):
    """Selection-set accuracy at eps = 20 (or the largest available), sound rule, cross-radius, best allocation."""
    eps = str(20 if 20 in C["eps"] else C["eps"][-1])
    vals = []
    for mode in MODES:
        cell = entry["results"][mode][eps]["aggregate"]["cross_radius"]
        vals.append(max(cell[a]["select"] for a in SELECT_ALLOCATIONS))
    return float(np.mean(vals))


def stage2_configs(C, done):
    ranked = sorted(
        (k for k, e in done.items() if e["q"] >= 1 and e["cfg"] in C["ags_stage1"]),
        key=lambda k: selection_score(done[k], C),
        reverse=True,
    )
    out = []
    for key in ranked[: C["ags_stage2_top"]]:
        cfg = done[key]["cfg"]
        for q in C["ags_stage2_q"]:
            out.append({**cfg, "q": q})
        for f in C["ags_stage2_lr_factors"]:
            out.append({**cfg, "lr": round(cfg["lr"] * f, 6)})
    return out


def run_envelopes(C, a):
    D = load_split(C)
    log(
        f"train {len(D['ytr'])}, pool {len(D['ypool'])}, select {len(D['ysel'])}, test {len(D['yte'])}; device {a.device}"
    )
    manifests = []
    if a.stage in ("1", "all"):
        for cfg in C["ags_stage1"]:
            if time.time() > a.deadline:
                log("deadline reached before stage 1 finished")
                break
            manifests.append(certify(D, C, cfg, a.device))
        evaluate_all(D, C, manifests)
    if a.stage in ("2", "all"):
        done = json.load(open(releases_path(C)))
        todo = stage2_configs(C, done)
        log(f"stage 2: {len(todo)} configurations")
        manifests = []
        for cfg in todo:
            if time.time() > a.deadline:
                log("deadline reached during stage 2")
                break
            manifests.append(certify(D, C, cfg, a.device))
        evaluate_all(D, C, manifests)


# ------------------------------------------------------------------------------------------------
# stage C: the baselines, on the CPU, in parallel with the envelopes
# ------------------------------------------------------------------------------------------------


def _median_runs(fn, seeds, D):
    sel, te, extra = [], [], None
    for s in range(seeds):
        th, extra = fn(s)
        sel.append(accuracy(th, D["Xsel"], D["ysel"]))
        te.append(accuracy(th, D["Xte"], D["yte"]))
    return dict(select=float(np.median(sel)), test=float(np.median(te)), mechanism=float(extra))


def _load_or(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def _dump(path, obj):
    json.dump(obj, open(path, "w"), indent=1, default=str)


def matched_configuration(D, C):
    """The trajectory the matched pair shares: the stage-1 configuration with the best non-private selection accuracy."""
    best = None
    for cfg in C["ags_stage1"]:
        res = resolve(cfg, len(D["Xtr"]))
        Xq, yq = subsample(D, C, res)
        th = B.non_private(
            Xq,
            yq,
            n_s=res["steps"],
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            lam=0.0,
            b=res["b"],
            lr_decay=cfg["decay"],
        )
        sc = scores(th, D)
        if best is None or sc["select"] > best["nominal"]["select"]:
            best = dict(cfg=cfg, key=cfg_key(cfg), **res, nominal=sc)
    return best


def run_baselines(C, a):
    D = load_split(C)
    out_dir = C["runs_dir"]
    os.makedirs(out_dir, exist_ok=True)
    P = n_params(C)
    Xtr, ytr = D["Xtr"], D["ytr"]

    # ---- matched pair: the trajectory, and DP-SGD on it
    path = f"{out_dir}/baselines_matched.json"
    M = _load_or(path, None)
    if M is None:
        M = matched_configuration(D, C)
        M["dpsgd"] = {m: {} for m in MODES}
        _dump(path, M)
    log(f"matched configuration {M['cfg']} steps={M['steps']} b={M['b']} nominal {M['nominal']}")
    cfg, res = M["cfg"], M
    Xq, yq = subsample(D, C, res)
    for mode in MODES:
        for eps in C["eps"]:
            if str(eps) in M["dpsgd"][mode] or time.time() > a.deadline:
                continue
            if mode == "pure":
                fn = lambda s: B.dpsgd_pure(
                    Xq,
                    yq,
                    eps,
                    n_s=res["steps"],
                    lr=cfg["lr"],
                    gamma=cfg["gamma"],
                    lam=0.0,
                    b=res["b"],
                    seed=s,
                    lr_decay=cfg["decay"],
                )
            else:
                fn = lambda s: B.dpsgd_gaussian_composed(
                    Xq,
                    yq,
                    eps,
                    C["delta"],
                    n_s=res["steps"],
                    lr=cfg["lr"],
                    gamma=cfg["gamma"],
                    lam=0.0,
                    b=res["b"],
                    seed=s,
                    lr_decay=cfg["decay"],
                )
            M["dpsgd"][mode][str(eps)] = _median_runs(fn, C["seeds"], D)
            _dump(path, M)
            log(f"dpsgd matched {mode} eps={eps}: {M['dpsgd'][mode][str(eps)]}")

    # ---- DP-SGD optimized: a grid per mode, selected per eps, the winner re-run with the full seeds
    path = f"{out_dir}/baselines_dpsgd_optimized.json"
    O = _load_or(path, {m: {"grid": {}, "selected": {}} for m in MODES})
    for mode in MODES:
        grid = C["dpsgd_opt_approx"] if mode == "approx" else C["dpsgd_opt_pure"]
        keys = ["q", "steps", "lr", "clip" if mode == "approx" else "gamma"]
        combos = [dict(zip(keys, vals)) for vals in __import__("itertools").product(*(grid[k] for k in keys))]
        for eps in C["eps"]:
            if str(eps) in O[mode]["selected"] or time.time() > a.deadline:
                continue
            cells = O[mode]["grid"].setdefault(str(eps), {})
            for g in combos:
                gk = json.dumps(g, sort_keys=True)
                if gk in cells:
                    continue
                if mode == "approx":
                    fn = lambda s: B.dpsgd_poisson_gaussian(
                        Xtr,
                        ytr,
                        eps,
                        C["delta"],
                        steps=g["steps"],
                        lr=g["lr"],
                        clip=g["clip"],
                        q=g["q"],
                        seed=s,
                        interval=C["pld_interval"],
                    )
                else:
                    fn = lambda s: B.dpsgd_poisson_laplace(
                        Xtr, ytr, eps, steps=g["steps"], lr=g["lr"], gamma=g["gamma"], q=g["q"], seed=s
                    )
                cells[gk] = _median_runs(fn, C["tune_seeds"], D)
                _dump(path, O)
            best_key = max(cells, key=lambda k: cells[k]["select"])
            g = json.loads(best_key)
            if mode == "approx":
                fn = lambda s: B.dpsgd_poisson_gaussian(
                    Xtr,
                    ytr,
                    eps,
                    C["delta"],
                    steps=g["steps"],
                    lr=g["lr"],
                    clip=g["clip"],
                    q=g["q"],
                    seed=s,
                    interval=C["pld_interval"],
                )
            else:
                fn = lambda s: B.dpsgd_poisson_laplace(
                    Xtr, ytr, eps, steps=g["steps"], lr=g["lr"], gamma=g["gamma"], q=g["q"], seed=s
                )
            O[mode]["selected"][str(eps)] = dict(_median_runs(fn, C["seeds"], D), config=g)
            _dump(path, O)
            log(f"dpsgd optimized {mode} eps={eps}: {O[mode]['selected'][str(eps)]}")

    # ---- PATE: teachers depend only on (m, seed); votes are cached and reused across eps and queries
    path = f"{out_dir}/baselines_pate.json"
    T = _load_or(path, {m: {"grid": {}, "selected": {}} for m in MODES})
    pc = C["pate"]
    Xpool = D["Xpool"]
    votes = {}

    def counts_for(m, seed):
        if (m, seed) not in votes:
            _, counts, _ = B._teachers_and_votes(
                Xtr,
                ytr,
                Xpool,
                n_teachers=m,
                n_queries=len(Xpool),
                n_s=pc["n_s"],
                lr=pc["lr"],
                gamma=pc["gamma"],
                lam=0.0,
                loss=B.DEFAULT_LOSS,
                rng=np.random.RandomState(seed),
            )
            votes[(m, seed)] = counts
        return votes[(m, seed)]

    def pate_run(mode, eps, m, Q, seed):
        counts = counts_for(m, seed)[:, :Q]
        rng = np.random.RandomState(10_000 + seed)
        if mode == "pure":
            eps_q = eps / Q
            noisy = counts + rng.laplace(0.0, 2.0 / eps_q, size=counts.shape)
            mech = eps_q
        else:
            eps_q, delta_q = eps / Q, C["delta"] / Q
            mech = (np.sqrt(2.0) / eps_q) * np.sqrt(2.0 * np.log(1.25 / delta_q))
            noisy = counts + rng.normal(0.0, mech, size=counts.shape)
        labels = noisy.argmax(0)
        student = B.train(Xpool[:Q], labels, n_s=pc["student_n_s"], lr=pc["lr"], gamma=pc["gamma"], lam=0.0)
        return student, mech

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
                    if gk in cells:
                        continue
                    cells[gk] = _median_runs(lambda s: pate_run(mode, eps, m, Q, s), C["tune_seeds"], D)
            best_key = max(cells, key=lambda k: cells[k]["select"])
            g = json.loads(best_key)
            T[mode]["selected"][str(eps)] = dict(
                _median_runs(lambda s: pate_run(mode, eps, g["teachers"], g["queries"], s), C["seeds"], D),
                config=g,
            )
            _dump(path, T)
            log(f"pate {mode} eps={eps}: {T[mode]['selected'][str(eps)]}")


# ------------------------------------------------------------------------------------------------
# stage D: the headline JSON the figure reads
# ------------------------------------------------------------------------------------------------


def run_release(C, a):
    D = load_split(C)
    man_dir = f"{C['runs_dir']}/envelopes"
    manifests = (
        [json.load(open(f"{man_dir}/{f}")) for f in sorted(os.listdir(man_dir))]
        if os.path.isdir(man_dir)
        else []
    )
    manifests = [m for m in manifests if m.get("complete")]
    done = evaluate_all(D, C, manifests)
    M = _load_or(f"{C['runs_dir']}/baselines_matched.json", None)
    O = _load_or(f"{C['runs_dir']}/baselines_dpsgd_optimized.json", {m: {"selected": {}} for m in MODES})
    T = _load_or(f"{C['runs_dir']}/baselines_pate.json", {m: {"selected": {}} for m in MODES})
    matched_key = M["key"] if M else None
    if matched_key not in done:
        log(
            f"matched configuration {matched_key} has no envelopes yet; AGS (matched) falls back to the best stage-1 configuration"
        )
        matched_key = max(
            (k for k in done if done[k]["q"] >= 1), key=lambda k: done[k]["nominal"]["select"], default=None
        )

    def variants(cell):
        """cell = results[mode][eps] -> {variant_name: {alloc: stats}}"""
        out = {}
        for rule in V.BETA_RULES:
            for cert in CERTS:
                out[f"{rule}/{cert}"] = cell[rule][cert]
        if "pld_coordinate" in cell:
            for cert in CERTS:
                out[f"pld_coordinate/{cert}"] = cell["pld_coordinate"][cert]
        return out

    headline = dict(
        setting={
            k: C[k]
            for k in (
                "N",
                "d",
                "n_val",
                "n_pool",
                "n_test",
                "seed",
                "delta",
                "seeds",
                "tune_seeds",
                "bound_model",
            )
        },
        eps=C["eps"],
        matched=dict(
            key=matched_key,
            cfg=done[matched_key]["cfg"] if matched_key else None,
            nominal=done[matched_key]["nominal"] if matched_key else (M["nominal"] if M else None),
        ),
        n_configs=len(done),
        arms={},
        provenance=stamp(),
    )
    for mode in MODES:
        headline["arms"][mode] = {}
        for eps in C["eps"]:
            e = str(eps)
            row = dict(
                non_private=headline["matched"]["nominal"]["test"]
                if headline["matched"]["nominal"]
                else None,
                pate=T[mode]["selected"].get(e),
                dpsgd_matched=(M or {}).get("dpsgd", {}).get(mode, {}).get(e),
                dpsgd_optimized=O[mode]["selected"].get(e),
                ags_matched={},
                ags_optimized={},
            )
            if matched_key:
                for name, allocs in variants(done[matched_key]["results"][mode][e]).items():
                    alloc = max(
                        (x for x in allocs if x in SELECT_ALLOCATIONS or x == "coordinate"),
                        key=lambda x: allocs[x]["select"],
                    )
                    row["ags_matched"][name] = dict(allocs[alloc], allocation=alloc, key=matched_key)
            for name in variants(done[matched_key]["results"][mode][e]) if matched_key else {}:
                best = None
                for key, entry in done.items():
                    allocs = variants(entry["results"][mode][e])[name]
                    for alloc, st in allocs.items():
                        if alloc not in SELECT_ALLOCATIONS and alloc != "coordinate":
                            continue
                        if best is None or st["select"] > best["select"]:
                            best = dict(st, allocation=alloc, key=key, cfg=entry["cfg"], q=entry["q"])
                row["ags_optimized"][name] = best
            headline["arms"][mode][e] = row
    out = f"{C['runs_dir']}/optimized_headline.json"
    _dump(out, headline)
    log(f"wrote {out}: {len(done)} AGS configurations, matched {matched_key}")
    for mode in MODES:
        log(
            f"  {mode:6s} eps      non-priv  PATE   DP-m   DP-opt  AGS-m(l1,sound)  AGS-opt(sound)  AGS-opt(dim-free)"
        )
        for eps in C["eps"]:
            r = headline["arms"][mode][str(eps)]
            g = lambda x: f"{x['test']:.3f}" if x else "  -  "
            am = r["ags_matched"].get("aggregate/cross_radius")
            ao = r["ags_optimized"].get("aggregate/cross_radius")
            ad = r["ags_optimized"].get("dimension_free/cross_radius")
            log(
                f"  {'':6s} {eps:<8} {r['non_private'] or 0:.3f}    {g(r['pate'])}  {g(r['dpsgd_matched'])}  {g(r['dpsgd_optimized'])}   {g(am)}            {g(ao)}           {g(ad)}"
            )


# ------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("envelopes", "baselines", "release", "all"))
    ap.add_argument("--stage", choices=("1", "2", "all"), default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--deadline",
        type=float,
        default=float("inf"),
        help="unix time after which no new configuration starts",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="the tiny setting in ags/mnist/config.py::OPTIMIZED_SMOKE"
    )
    a = ap.parse_args()
    C = OPTIMIZED_SMOKE if a.smoke else OPTIMIZED
    os.makedirs(C["runs_dir"], exist_ok=True)
    if a.command in ("envelopes", "all"):
        run_envelopes(C, a)
    if a.command in ("baselines", "all"):
        run_baselines(C, a)
    if a.command in ("release", "all"):
        run_release(C, a)


if __name__ == "__main__":
    main()
