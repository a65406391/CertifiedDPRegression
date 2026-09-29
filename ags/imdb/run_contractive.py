"""AGS on IMDB sentence-encoder features against matched and optimized DP-SGD and PATE, budgets to 1000.

The MNIST contractive comparison (ags/mnist/run_contractive.py) moved to a frozen foundation-model
backbone: all-mpnet-base-v2 embeddings of the 25k IMDB training reviews, PCA-d on a public split, a
head certified by AGT on the 20k private reviews. The release is one multivariate draw on the whole
parameter vector (common/vector_release.py): the l1 cross-radius certificate over all P coordinates,
one smooth sensitivity with the sound beta/P rule, one epsilon. Pure DP releases Cauchy; approx DP
takes the better of Laplace and the Cauchy release and stores both.

Four groups, each a trajectory with its own matched, optimized and PATE baselines:

    logistic        the paper's setting: Linear(d, 2) from zero, contraction toward zero
    logistic_warm   start at the public head theta_pub (trained on the 2.5k public pool) and contract
                    toward it (ags/agt_ext/centered_l2.py); the certificate is unchanged, the fixed
                    point moves from chance to the public head
    mlp             Linear(d, h) -> ReLU -> Linear(h, 2) from torch's initialisation
    mlp_warm        the same from, and toward, its public warm start
    logistic_long   the logistic finalist on the long trajectory (config `long`, T = 3000; `sweep --long`):
                    AGS's release is T-invariant under contraction, the matched arm pays T

Stages, each resumable (envelopes per configuration and radius, JSON per stage):

    embed              the encoder's embeddings, cached (ags/imdb/shared.py)
    sweep              stage 1 grid (d, lambda, gamma) at T = 300 on the whole private split, stage 2
                       longer trajectories for the top configurations, then the three other groups at
                       the finalist's hyperparameters; writes <GROUP>_DONE as each group's finalist is
                       certified so its baselines can start
    baselines --group  matched DP-SGD (pure: basic composition; approx: basic and exact Gaussian
                       composition, both drawn), optimized DP-SGD (Poisson + PLD / amplified basic), PATE
    release            the headline JSON the figure reads (plot_contractive.py)

    PYTHONPATH=. python -m ags.imdb.run_contractive sweep --device cuda:0
    PYTHONPATH=. python -m ags.imdb.run_contractive all --task smoke          # every path, minutes

run_contractive.sh runs the whole thing under one deadline. Settings: ags/imdb/config.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from ags.imdb import heads as H
from ags.imdb.certify import agt_bound
from ags.imdb.config import CONTRACTIVE, CONTRACTIVE_DENSE, CONTRACTIVE_N2K, CONTRACTIVE_SMOKE
from ags.imdb.shared import embeddings, log
from ags.imdb.shared import load_task as imdb_load_task
from ags.mnist import ss_reference as R
from ags.mnist.baselines import poisson_gaussian_multiplier
from ags.mnist.shared import run_id
from common import vector_release as V
from common.provenance import stamp

TASKS = {
    "imdb": CONTRACTIVE,
    "imdb_dense": CONTRACTIVE_DENSE,
    "imdb_n2k": CONTRACTIVE_N2K,
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
GROUPS = ("logistic", "logistic_warm", "mlp", "mlp_warm", "logistic_long")


def _load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def _dump(path, obj):
    """Atomic: a reader never sees a half-written file, even with two stages running side by side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)


def marker(C, name):
    return f"{C['runs_dir']}/{name.upper()}_DONE"


def mark(C, name):
    open(marker(C, name), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


LOCK_STALE = 3600.0  # a lock older than this with no result behind it belonged to a crashed worker


def claim(path):
    """Claim the computation of `path` for this process through `path`.lock; False if another worker
    holds it. Several sweep workers (run_contractive.sh starts three) share one grid this way: each
    takes the radii nobody has claimed and waits for the rest, so a configuration certifies about three
    times faster than on one process."""
    lock = f"{path}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > LOCK_STALE:
                os.remove(lock)
                return claim(path)
        except FileNotFoundError:
            return claim(path)
        return False


def release_claim(path):
    try:
        os.remove(f"{path}.lock")
    except FileNotFoundError:
        pass


def compute_or_wait(path, compute, what):
    """Produce `path` once across workers: compute it if this process gets the claim, otherwise wait
    for the worker that has it (or take over if that worker died). Returns once the file exists."""
    while not os.path.exists(path):
        if claim(path):
            try:
                if not os.path.exists(path):
                    compute()
            finally:
                release_claim(path)
        else:
            log(f"  waiting for another worker: {what}")
            while not os.path.exists(path) and os.path.exists(f"{path}.lock"):
                time.sleep(5)


# ------------------------------------------------------------------------------------------------
# configurations: head, start, key
# ------------------------------------------------------------------------------------------------


def load_task(C, d, device):
    """The task's data. Another dataset reuses this driver by putting its loader (same signature and
    returned keys as ags.imdb.shared.load_task) under C["loader"]; see ags/adult/."""
    return (C.get("loader") or imdb_load_task)(C, d, device)


def group_of(cfg):
    return cfg["head"] + ("_warm" if cfg.get("warm") else "")


def head_of(cfg, D):
    return H.make_head(cfg["head"], D["d"], D["C"], cfg.get("h"))


def cfg_key(cfg):
    return run_id("con_imdb", cfg)


def scores(head, theta, D, device):
    """Selection-set and test accuracy of a flat parameter vector. When the task sets
    D["tune_threshold"] (binary, imbalanced: Adult), the decision threshold on the logit margin is
    chosen on the public selection set and applied to the test set, for every arm alike: a
    post-processing of the released head with public data, so it costs no privacy. The test AUC is
    stored too."""
    if not D.get("tune_threshold"):
        return dict(
            select=head.accuracy(theta, D["Xsel"], D["ysel"], device),
            test=head.accuracy(theta, D["Xte"], D["yte"], device),
        )
    zs, zt = head.logits(theta, D["Xsel"], device), head.logits(theta, D["Xte"], device)
    ms, mt = zs[:, 1] - zs[:, 0], zt[:, 1] - zt[:, 0]
    ts = np.quantile(ms, np.linspace(0.01, 0.99, 99))
    ys, yt = np.asarray(D["ysel"]), np.asarray(D["yte"])
    accs = np.array([((ms > t) == ys).mean() for t in ts])
    t = float(ts[int(accs.argmax())])
    order = np.argsort(mt)
    ranks = np.empty(len(mt))
    ranks[order] = np.arange(1, len(mt) + 1)
    n_pos = yt.sum()
    auc = (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(yt) - n_pos))
    return dict(
        select=float(accs.max()),
        test=float(((mt > t) == yt).mean()),
        auc=float(auc),
        argmax_test=head.accuracy(theta, D["Xte"], D["yte"], device),
    )


def warm_start(C, D, head, a):
    """theta_pub: the head trained without clipping on the public pool, cached under runs/warm/."""
    w = C["warm"]
    tag = f"{head.name}_d{D['d']}" + (f"_h{head.h}" if head.name == "mlp" else "") + f"_n{w['n']}"
    path = f"{C['runs_dir']}/warm/{tag}.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def compute():
        theta = H.train(
            head,
            D["Xpool"][: w["n"]],
            D["ypool"][: w["n"]],
            theta0=head.init(0),
            n_s=w["T"],
            lr=w["lr"],
            gamma=None,
            lam=w["lam"],
            device=a.device,
        )
        sc = scores(head, theta, D, a.device)
        tmp = f"{path}.tmp.{os.getpid()}.npz"
        np.savez(tmp, theta=theta, select=sc["select"], test=sc["test"], cfg=np.array(json.dumps(w)))
        os.replace(tmp, path)
        log(f"public head {tag}: {sc} ({w['n']} labelled public reviews, {w})")

    compute_or_wait(path, compute, f"public head {tag}")
    return np.load(path)["theta"]


def start_and_center(C, D, head, cfg, a):
    """(theta0, center): the public head twice for a warm group, (init, None) otherwise."""
    if cfg.get("warm"):
        th = warm_start(C, D, head, a)
        return th, th
    return head.init(0), None


# ------------------------------------------------------------------------------------------------
# certification: one file per configuration and radius, a manifest per configuration
# ------------------------------------------------------------------------------------------------


def certify(C, D, head, cfg, a, theta0, center, *, key=None, radii=None):
    key = key or cfg_key(cfg)
    man_path = f"{C['runs_dir']}/envelopes/{key}.json"
    man = _load(man_path, None)
    if man and man.get("complete"):
        return man
    os.makedirs(f"{C['runs_dir']}/bounds", exist_ok=True)
    X, y = D["Xfull"], D["yfull"]
    b, T = len(y), int(cfg["T"])
    radii = [k for k in (radii or C["k_grid"]) if k < b] + [b]
    paths, theta_n, t0 = {}, None, time.time()
    cells = {}
    for k in radii:
        path = f"{C['runs_dir']}/bounds/{key}_k{k}.npz"

        def compute(k=k, path=path):
            tk = time.time()
            pl, pn, pu = agt_bound(
                head,
                X,
                y,
                k=k,
                b=b,
                T=T,
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                lam=cfg["lam"],
                decay=cfg["decay"],
                theta0=theta0,
                center=center,
                device=a.device,
                bound_model=C["bound_model"],
            )
            tmp = f"{path}.tmp.{os.getpid()}.npz"
            np.savez_compressed(
                tmp,
                param_l=pl,
                param_n=pn,
                param_u=pu,
                cfg=np.array(json.dumps({**cfg, "k": k, "b": b})),
                seconds=time.time() - tk,
            )
            os.replace(tmp, path)
            log(f"  {key} k={k:<6} width mean {np.abs(pu - pl).mean():.3e}  {time.time() - tk:.1f}s")

        cells[k] = (path, compute)
    # first pass: every radius nobody holds; second pass: wait for the other workers' radii (or take
    # over one whose worker died)
    for k, (path, compute) in cells.items():
        if not os.path.exists(path) and claim(path):
            try:
                if not os.path.exists(path):
                    compute()
            finally:
                release_claim(path)
    for k, (path, compute) in cells.items():
        compute_or_wait(path, compute, f"{key} k={k}")
        z = np.load(path, allow_pickle=True)
        pl, pn, pu = z["param_l"], z["param_n"], z["param_u"]
        if theta_n is None:
            theta_n = pn
        elif not np.allclose(theta_n, pn):
            raise RuntimeError(f"{key}: nominal differs across radii at k={k}")
        if not (
            np.all(np.isfinite(pl))
            and np.all(np.isfinite(pu))
            and np.all(pl <= pn + 1e-9)
            and np.all(pu >= pn - 1e-9)
        ):
            log(
                f"  {key} k={k}: the package's box is not a valid interval; the clipping ceiling replaces it at load time"
            )
        paths[str(k)] = path
    man = dict(
        cfg=cfg,
        key=key,
        group=group_of(cfg),
        P=head.P,
        b=b,
        T=T,
        radii=radii,
        paths=paths,
        theta0=np.asarray(theta0, float).tolist(),
        centered=center is not None,
        theta_n=theta_n.tolist(),
        nominal=scores(head, theta_n, D, a.device),
        seconds=time.time() - t0,
        provenance=stamp(
            {**cfg, "b": b, "bound_model": C["bound_model"], "task": C["task"], "encoder": C.get("encoder")}
        ),
        complete=True,
    )
    _dump(man_path, man)
    log(f"{key}: {cfg} b={b} P={head.P}  nominal {man['nominal']}  {man['seconds']:.0f}s")
    return man


def clipping_ceiling(cfg, T):
    """2 gamma sum_t lr_t: a sound per-coordinate bound on |theta_T(D) - theta_T(D')| at any radius, from the
    element-wise clip alone. Two clipped trajectories from the same start differ by at most lr_t 2 gamma per
    step in every coordinate, and the L2 step multiplies the difference by |1 - lr_t lambda| <= 1 whenever
    lr_t lambda <= 2, which every configuration here satisfies."""
    lrs = [cfg["lr"] / (1.0 + cfg["decay"] * t) for t in range(int(T))]
    if max(lrs) * cfg["lam"] > 2.0:
        raise ValueError("the clipping ceiling needs lr_t * lambda <= 2 at every step")
    return 2.0 * cfg["gamma"] * float(sum(lrs))


def sound_box(pl, pn, pu, ceiling):
    """The certified box made sound against the package's numerical failures. A box whose bounds are finite
    and contain the nominal is intersected with the clipping-ceiling box (both sound, so is the
    intersection); a box that is not (non-finite, or inverted after an overflow in the propagation, as at
    the saturation radius k = b on long trajectories) is replaced by the ceiling box. Returns (lo, hi, how)."""
    pl, pn, pu = (np.asarray(a, float) for a in (pl, pn, pu))
    valid = bool(
        np.all(np.isfinite(pl))
        and np.all(np.isfinite(pu))
        and np.all(pl <= pn + 1e-9)
        and np.all(pu >= pn - 1e-9)
    )
    lo_c, hi_c = pn - ceiling, pn + ceiling
    if not valid:
        return lo_c, hi_c, "ceiling"
    lo, hi = np.maximum(pl, lo_c), np.minimum(pu, hi_c)
    return lo, hi, ("intersected" if (np.any(lo > pl) or np.any(hi < pu)) else "agt")


def load_boxes(man):
    boxes = {}
    ceiling = clipping_ceiling(man["cfg"], man["T"])
    repaired = []
    for k, path in man["paths"].items():
        z = np.load(path, allow_pickle=True)
        lo, hi, how = sound_box(z["param_l"], z["param_n"], z["param_u"], ceiling)
        boxes[int(k)] = (lo, hi)
        if how != "agt":
            repaired.append(f"k={k}:{how}")
    if repaired:
        log(
            f"  {man['key']}: boxes made sound against the clipping ceiling {ceiling:.3g}: {', '.join(repaired)}"
        )
    return boxes, np.asarray(man["theta_n"], float)


# ------------------------------------------------------------------------------------------------
# the release: whole-vector, l1 cross-radius, beta/P; every variant stored
# ------------------------------------------------------------------------------------------------


def release_stats(head, theta_n, sigma, draw, seeds, D, device):
    sel, te = [], []
    for s in range(seeds):
        th = R.release(theta_n, sigma, np.random.RandomState(s), mode=draw)
        sc = scores(head, th, D, device)
        sel.append(sc["select"])
        te.append(sc["test"])
    return dict(select=float(np.median(sel)), test=float(np.median(te)))


def mechanism(mech, eps, P, C):
    """(beta, draw mode) for a mechanism: the reference's constants."""
    if mech == "laplace":
        return R.beta_for(eps, mode="approx", n_params=P, delta=C["delta"]), "approx"
    return R.beta_for(eps), "pure"


def evaluate(C, D, head, man, seeds, device):
    P = head.P
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
                        st = release_stats(head, theta_n, sigma, draw, seeds, D, device)
                        row.setdefault(rule, {}).setdefault(cert, {})[mech] = dict(
                            st, sigma=float(sigma[0]), SS=float(S[0]), beta=float(beta)
                        )
            out[mode][str(eps)] = row
    return out


def best_mech(cell):
    m = max(cell, key=lambda k: cell[k]["select"])
    return m, cell[m]


def evaluate_tight(C, D, head, man, seeds, device):
    """The proven rules (common/vector_release.tight_variants) on a stored envelope, computed once into
    runs/<task>/releases_tight/<key>.json; sweep.json is left as it was written."""
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
        score=lambda th: scores(head, th, D, device),
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
# matched DP-SGD: the trajectory the certificate describes, plus noise
# ------------------------------------------------------------------------------------------------


def matched(head, D, cfg, theta0, center, eps, mode, kind, C, seed, device):
    """One matched DP-SGD run. pure: Laplace, l1 sensitivity 2 P gamma / b, basic composition over T.
    approx basic: Gaussian, l2 sensitivity sqrt(P) 2 gamma / b, eps / T per step; approx composed: the
    exact Gaussian composition multiplier. Returns (theta, mechanism parameter)."""
    X, y = D["Xfull"], D["yfull"]
    b, T, P, gamma = len(y), cfg["T"], head.P, cfg["gamma"]
    if mode == "pure":
        scale, kind_ = 2.0 * P * gamma / b * T / eps, "laplace"
    elif kind == "basic":
        scale = np.sqrt(P) * 2.0 * gamma / b / (eps / T) * np.sqrt(2.0 * np.log(1.25 / C["delta"]))
        kind_ = "gauss"
    else:
        from common.accounting.exact_accounting import calibrate_composed_gaussian

        multiplier, _ = calibrate_composed_gaussian(eps, C["delta"], T)
        scale, kind_ = multiplier * gamma * np.sqrt(P) / b, "gauss"
    th = H.train(
        head,
        X,
        y,
        theta0=theta0,
        center=center,
        n_s=T,
        lr=cfg["lr"],
        gamma=gamma,
        lam=cfg["lam"],
        noise_scale=scale,
        noise_kind=kind_,
        rng=np.random.RandomState(seed),
        lr_decay=cfg["decay"],
        device=device,
    )
    return th, scale


def quick_matched_pure(C, D, head, cfg, theta0, center, a):
    out = {}
    for eps in C["select_eps"]:
        accs = [
            scores(
                head,
                matched(head, D, cfg, theta0, center, eps, "pure", "basic", C, s, a.device)[0],
                D,
                a.device,
            )["select"]
            for s in range(C["select_seeds"])
        ]
        out[str(eps)] = float(np.median(accs))
    return out


def score(C, entry):
    """Margin of the AGS release over quick matched pure DP-SGD at the selection budgets, on the public
    selection set. The release ranked is the proven rule (tight, l1, Cauchy) when the sweep evaluated it,
    else the earlier aggregate rule (sweeps written before 2026-09-20)."""
    tight = entry.get("tight_select")
    ags = (
        [tight[str(e)] for e in C["select_eps"]]
        if tight
        else [
            entry["releases"]["pure"][str(e)]["aggregate"]["cross_radius"]["cauchy"]["select"]
            for e in C["select_eps"]
        ]
    )
    dp = [entry["quick_matched"][str(e)] for e in C["select_eps"]]
    return dict(margin=float(np.mean(ags) - np.mean(dp)), ags=float(np.mean(ags)), dpsgd=float(np.mean(dp)))


# ------------------------------------------------------------------------------------------------
# sweep
# ------------------------------------------------------------------------------------------------


def sweep_path(C):
    return f"{C['runs_dir']}/sweep.json"


def ranked(C, S, group, keys=None):
    keys = list(S) if keys is None else keys
    return sorted(
        (
            k
            for k in keys
            if k in S and S[k].get("complete") and S[k]["group"] == group and S[k].get("stage") != "long"
        ),
        key=lambda k: (S[k]["score"]["margin"], S[k]["score"]["ags"]),
        reverse=True,
    )


def finalist(C, S, group):
    """The group's trajectory for the figure and the baselines: chosen by hand in config, else the
    sweep's best margin over matched pure DP-SGD at the selection budgets."""
    if group == "logistic_long":  # the finalist on the long trajectory (config `long`)
        fin = finalist(C, S, "logistic")
        if fin is None or not C.get("long"):
            return None
        cfg = {**fin, **C["long"]}
        return cfg if cfg_key(cfg) in S else None
    base = group.replace("_warm", "")
    if C["final"].get(base):
        cfg = {"head": base, "h": None, **C["final"][base]}
    else:
        keys = None
        if base == "mlp":  # only the arms this config generates from the logistic finalist
            fin = finalist(C, S, "logistic")
            keys = [cfg_key(c) for c in mlp_arms(C, fin)] if fin else []
        r = ranked(C, S, base, keys)
        if not r:
            return None
        cfg = dict(S[r[0]]["cfg"])
    cfg["warm"] = group.endswith("_warm")
    return cfg if cfg_key(cfg) in S else (None if cfg["warm"] else cfg)


def mlp_arms(C, fin):
    """The hidden-layer configurations: the logistic finalist with the overrides of config arms.mlp."""
    m = C["arms"]["mlp"]
    out = []
    for h, d, gamma, lam, T in itertools.product(
        m["h"], m["d"], m.get("gamma", [None]), m.get("lam", [None]), m.get("T", [None])
    ):
        out.append(
            {
                **fin,
                "head": "mlp",
                "h": h,
                "d": d or fin["d"],
                "gamma": gamma or fin["gamma"],
                "lam": lam or fin["lam"],
                "T": T or fin["T"],
                "warm": False,
            }
        )
    return out


def run_sweep(C, a):
    S = _load(sweep_path(C), {})

    def do(cfg, stage):
        cfg = {"head": "logistic", "h": None, "warm": False, **cfg}
        key = cfg_key(cfg)
        S.update(_load(sweep_path(C), {}))  # entries other workers wrote
        if key in S and S[key].get("complete"):
            return key
        if time.time() > a.deadline:
            log("deadline reached; sweep stops here")
            return None
        D = load_task(C, cfg["d"], a.device)
        head = head_of(cfg, D)
        theta0, center = start_and_center(C, D, head, cfg, a)
        man = certify(C, D, head, cfg, a, theta0, center)
        t0 = time.time()
        rel = evaluate(C, D, head, man, C["seeds"], a.device)
        T = evaluate_tight(C, D, head, man, C["seeds"], a.device)
        qm = quick_matched_pure(C, D, head, cfg, theta0, center, a)
        S[key] = dict(
            cfg=cfg,
            key=key,
            stage=stage,
            group=group_of(cfg),
            P=head.P,
            nominal=man["nominal"],
            b=man["b"],
            releases=rel,
            quick_matched=qm,
            tight_select={
                str(e): T["pure"][str(e)]["tight_l1"]["cross_radius"]["cauchy"]["select"]
                for e in C["select_eps"]
            },
            tight_test={
                str(e): T["pure"][str(e)]["tight_l1"]["cross_radius"]["cauchy"]["test"]
                for e in C["select_eps"]
            },
            public_head=scores(head, theta0, D, a.device) if cfg["warm"] else None,
            complete=True,
        )
        S[key]["score"] = score(C, S[key])
        _dump(sweep_path(C), {**_load(sweep_path(C), {}), key: S[key]})  # merge: workers share the file
        sc = S[key]["score"]
        peek = {e: round(S[key]["tight_test"][str(e)], 3) for e in C["select_eps"]}
        log(
            f"  released {key} [{S[key]['group']}] in {time.time() - t0:.0f}s: nominal {man['nominal']['test']:.3f}; "
            f"AGS pure test {peek}; margin over matched {sc['margin']:+.3f} (AGS {sc['ags']:.3f} vs DP {sc['dpsgd']:.3f})"
        )
        return key

    s1 = C["stage1"]
    keys1 = []
    for d, lam, gamma in itertools.product(s1["d"], s1["lam"], s1["gamma"]):
        k = do(dict(d=d, lam=lam, gamma=gamma, lr=s1["lr"], decay=s1["decay"], T=s1["T"]), 1)
        if k:
            keys1.append(k)
    for base in ranked(C, S, "logistic", keys1)[: C["stage2"]["top"]]:
        for var in C["stage2"]["variants"]:
            do({**S[base]["cfg"], **var}, 2)
    if C["final"].get(
        "logistic"
    ):  # a finalist pinned by hand is certified whether or not the ranking picked it
        do(dict(C["final"]["logistic"]), 2)
    fin = finalist(C, S, "logistic")
    if fin is None:
        return
    mark(C, "logistic")
    log(f"finalist logistic: {fin}")

    groups = C.get("groups", GROUPS)  # a task may run the logistic group alone
    if "logistic_warm" in groups and do({**fin, "warm": True}, "warm"):
        mark(C, "logistic_warm")
    if C.get("long") and a.long and do({**fin, **C["long"]}, "long"):
        mark(C, "logistic_long")

    if "mlp" not in groups:
        mark(C, "sweep")
        return
    for cfg in mlp_arms(C, fin):
        do(cfg, "mlp")
    fin_mlp = finalist(C, S, "mlp")
    if fin_mlp is None:
        return
    mark(C, "mlp")
    log(f"finalist mlp: {fin_mlp}")
    if do({**fin_mlp, "warm": True}, "mlp_warm"):
        mark(C, "mlp_warm")

    for g in GROUPS:
        for k in ranked(C, S, g)[:3]:
            log(
                f"rank {g}: {k} {S[k]['cfg']} margin {S[k]['score']['margin']:+.3f} AGS {S[k]['score']['ags']:.3f} "
                f"DP {S[k]['score']['dpsgd']:.3f} nominal {S[k]['nominal']['test']:.3f}"
            )
    mark(C, "sweep")


# ------------------------------------------------------------------------------------------------
# baselines for one group: matched on its trajectory, optimized DP-SGD, PATE
# ------------------------------------------------------------------------------------------------


def _median_runs(head, fn, seeds, D, device):
    sel, te, extra = [], [], None
    for s in range(seeds):
        th, extra = fn(s)
        sc = scores(head, th, D, device)
        sel.append(sc["select"])
        te.append(sc["test"])
    return dict(select=float(np.median(sel)), test=float(np.median(te)), mechanism=float(extra))


def run_baselines(C, a, group):
    S = _load(sweep_path(C), {})
    cfg = finalist(C, S, group)
    if cfg is None or not os.path.exists(marker(C, group)):
        log(f"no finalist for {group} yet; baselines skipped")
        return
    D = load_task(C, cfg["d"], a.device)
    head = head_of(cfg, D)
    theta0, center = start_and_center(C, D, head, cfg, a)
    X, y, dev = D["Xfull"], D["yfull"], a.device
    seeds = C["train_seeds"]
    path = f"{C['runs_dir']}/baselines_{group}.json"
    old = _load(path, None)
    if old and old.get("key") != cfg_key(cfg):  # the group's finalist changed: keep the old file, start anew
        kept = f"{C['runs_dir']}/baselines_{group}_{old.get('key')}.json"
        if not os.path.exists(kept):
            os.replace(path, kept)
        else:
            os.remove(path)
        log(
            f"baselines {group}: finalist changed from {old.get('key')} to {cfg_key(cfg)}; the old cells stay in {kept}"
        )
    Bl = _load(
        path,
        {
            "cfg": cfg,
            "key": cfg_key(cfg),
            "matched": {m: {} for m in MODES},
            "dpsgd_optimized": {m: {"grid": {}, "selected": {}} for m in MODES},
            "pate": {m: {"grid": {}, "selected": {}} for m in MODES},
            "seeds": seeds,
        },
    )
    M = Bl["matched"]
    for mode in MODES:
        for eps in C["eps"]:
            if str(eps) in M[mode] or time.time() > a.deadline:
                continue
            kinds = ("basic",) if mode == "pure" else ("basic", "composed")
            M[mode][str(eps)] = {
                kind: _median_runs(
                    head,
                    lambda s, kind=kind: matched(head, D, cfg, theta0, center, eps, mode, kind, C, s, dev),
                    seeds,
                    D,
                    dev,
                )
                for kind in kinds
            }
            _dump(path, Bl)
            log(f"matched {group} {mode} eps={eps}: {M[mode][str(eps)]}")

    O = Bl["dpsgd_optimized"]
    for mode in MODES:
        grid = C["dpsgd_opt_approx"] if mode == "approx" else C["dpsgd_opt_pure"]
        keys = ["q", "steps", "lr", "clip" if mode == "approx" else "gamma"]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

        def runner(g, eps):
            if mode == "approx":
                m = poisson_gaussian_multiplier(
                    eps, C["delta"], q=g["q"], steps=g["steps"], interval=C["pld_interval"]
                )
                return lambda s: (
                    H.poisson_steps(
                        head,
                        X,
                        y,
                        theta0=theta0,
                        steps=g["steps"],
                        lr=g["lr"],
                        q=g["q"],
                        rng=np.random.RandomState(s),
                        per_step_noise=lambda r, P: r.normal(0.0, m * g["clip"], size=P),
                        clip_rows=H.clip_l2(g["clip"]),
                        device=dev,
                    ),
                    m,
                )
            eps_step = eps / g["steps"]
            eps_s = np.log1p(np.expm1(eps_step) / g["q"]) if g["q"] < 1 else eps_step
            scale = 2.0 * head.P * g["gamma"] / eps_s
            return lambda s: (
                H.poisson_steps(
                    head,
                    X,
                    y,
                    theta0=theta0,
                    steps=g["steps"],
                    lr=g["lr"],
                    q=g["q"],
                    rng=np.random.RandomState(s),
                    per_step_noise=lambda r, P: r.laplace(0.0, scale, size=P),
                    clip_rows=H.clip_elementwise(g["gamma"]),
                    device=dev,
                ),
                eps_s,
            )

        for eps in C["eps"]:
            if str(eps) in O[mode]["selected"] or time.time() > a.deadline:
                continue
            cells = O[mode]["grid"].setdefault(str(eps), {})
            for g in combos:
                gk = json.dumps(g, sort_keys=True)
                if gk not in cells:
                    cells[gk] = _median_runs(head, runner(g, eps), C["tune_seeds"], D, dev)
                    _dump(path, Bl)
            g = json.loads(max(cells, key=lambda k: cells[k]["select"]))
            O[mode]["selected"][str(eps)] = dict(_median_runs(head, runner(g, eps), seeds, D, dev), config=g)
            _dump(path, Bl)
            log(f"dpsgd optimized {group} {mode} eps={eps}: {O[mode]['selected'][str(eps)]}")

    pc, Xpool, votes = C.get("pate"), D["Xpool"], {}
    if (
        not pc
    ):  # a task without a teacher pool (SST-2: the public set is the 872 dev sentences) has no PATE arm
        _dump(path, Bl)
        mark(C, f"BASELINES_{group.upper()}_DONE")
        return

    def counts_for(m, seed):
        if (m, seed) not in votes:
            votes[(m, seed)] = H.teachers_and_votes(
                head,
                X,
                y,
                Xpool,
                n_teachers=m,
                n_s=pc["n_s"],
                lr=pc["lr"],
                lam=0.0,
                rng=np.random.RandomState(seed),
                device=dev,
                seed=100 * seed,
            )
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
        student = H.train(
            head,
            Xpool[:Q],
            noisy.argmax(0),
            theta0=theta0,
            n_s=pc["student_n_s"],
            lr=pc["lr"],
            gamma=None,
            lam=0.0,
            device=dev,
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
                        cells[gk] = _median_runs(
                            head, lambda s: pate_run(mode, eps, m, Q, s), C["tune_seeds"], D, dev
                        )
            g = json.loads(max(cells, key=lambda k: cells[k]["select"]))
            T[mode]["selected"][str(eps)] = dict(
                _median_runs(
                    head, lambda s: pate_run(mode, eps, g["teachers"], g["queries"], s), seeds, D, dev
                ),
                config=g,
            )
            _dump(path, Bl)
            log(f"pate {group} {mode} eps={eps}: {T[mode]['selected'][str(eps)]}")
    _dump(path, Bl)
    mark(C, f"baselines_{group}")


# ------------------------------------------------------------------------------------------------
# the headline JSON the figure reads
# ------------------------------------------------------------------------------------------------


def variants(row):
    """{} for a budget the sweep did not evaluate (dense-eps tasks add budgets at release time)."""
    return {f"{rule}/{cert}": row[rule][cert] for rule in RULES for cert in CERTS if rule in row}


# ------------------------------------------------------------------------------------------------
# the sampling-frequency ablation: Algorithm 2, n releases along one trajectory, each window certified
# from the previous noisy release (envelope collapse), budgets composed (pure: eps / n per window)
# ------------------------------------------------------------------------------------------------


def run_windows(C, a):
    """AGS on its own with n releases: the trajectory of the logistic finalist is cut into n windows of
    T / n steps; window w is certified from the parameters released at the end of window w - 1 (the
    noisy point, so the envelope collapses to it), released with the proven l1 rule at budget eps / n
    (basic composition: pure DP with Cauchy noise, or approximate DP with Laplace noise at (eps / n,
    delta / n) per window; `windows.modes` in the config, `--modes` on the command line), and the last
    release is scored. Every worker runs the same loop in the same order and the noise is seeded, so
    workers share radii through the usual locks and agree on the released points. Results in
    runs/<task>/windows.json: per (mode, n, eps, seed) the per-window nominal and released scores and
    noise scales; the pure chains keep their original keys, the approximate ones carry `_approx`.

    `accounting` (config `windows.accounting` or `--accounting`): "basic" splits the budget as above;
    "optimal" gives each window the largest eps_0 for which the n windows compose to (eps, delta) by the
    optimal composition theorem (common.accounting.exact_accounting.optimal_pure_budget): pure windows
    then compose to (eps, delta)-DP, and approximate windows run at (eps_0, delta / (2n)) with the other
    delta / 2 as the composition slack (Kairouz et al. 2015, Thm 3.5: 1 - (1 - delta_0)^n (1 - slack)
    <= delta). Keys carry `_opt`. `--windows-eps` and `--windows-n` override the config's grids."""
    W = C["windows"]
    S = _load(sweep_path(C), {})
    cfg = finalist(C, S, "logistic")
    if cfg is None:
        log("no logistic finalist; windows skipped")
        return
    cfg = {**cfg, "T": W.get("T", cfg["T"])}
    D = load_task(C, cfg["d"], a.device)
    head = head_of(cfg, D)
    theta_init = head.init(0)
    path = f"{C['runs_dir']}/windows.json"
    modes = tuple(a.modes) if getattr(a, "modes", None) else tuple(W.get("modes", ("pure",)))
    accounting = getattr(a, "accounting", None) or W.get("accounting", "basic")
    ns = list(getattr(a, "windows_n", None) or W["n"])
    eps_grid = [int(e) if float(e).is_integer() else e for e in (getattr(a, "windows_eps", None) or W["eps"])]
    from common.accounting.exact_accounting import optimal_pure_budget

    for mode, n, eps in itertools.product(modes, ns, eps_grid):
        if True:
            for seed in range(W["seeds"]):
                sfx = ("" if mode == "pure" else f"_{mode}") + ("_opt" if accounting == "optimal" else "")
                tag = f"n{n}_eps{eps}_s{seed}{sfx}"
                Wj = _load(path, {"cfg": cfg, "runs": {}})
                if tag in Wj["runs"]:
                    continue
                if time.time() > a.deadline:
                    log("deadline reached; windows stop here")
                    return
                T_w = max(1, cfg["T"] // n)
                if accounting == "optimal":
                    slack = C["delta"] if mode == "pure" else C["delta"] / 2
                    eps_w = optimal_pure_budget(eps, n, slack)
                    delta_w = C["delta"] / (2 * n) if mode == "approx" else None
                else:
                    eps_w, delta_w = eps / n, (C["delta"] / n if mode == "approx" else None)
                theta, per_window = theta_init, []
                for w in range(n):
                    key = f"{cfg_key(cfg)}_win{n}_eps{eps}_s{seed}{sfx}_w{w}"
                    man = certify(
                        C, D, head, {**cfg, "T": T_w}, a, theta, None, key=key, radii=W.get("k_grid")
                    )
                    boxes, theta_n = load_boxes(man)
                    G = V.cross_radius_grid(theta_n, boxes, man["b"])
                    sigma, S_l1, info = V.tight_scales(G, eps_w, mode=mode, allocation="l1", delta=delta_w)
                    theta = R.release(theta_n, sigma, np.random.RandomState(1000 * seed + w), mode=mode)
                    per_window.append(
                        dict(
                            window=w,
                            nominal=scores(head, theta_n, D, a.device),
                            released=scores(head, theta, D, a.device),
                            sigma=float(sigma[0]),
                            SS=float(S_l1[0]),
                            beta_j=info["beta_j"],
                            alpha=info["alpha"],
                        )
                    )
                Wj = _load(path, {"cfg": cfg, "runs": {}})
                Wj["runs"][tag] = dict(
                    mode=mode,
                    accounting=accounting,
                    n=n,
                    eps=eps,
                    seed=seed,
                    eps_w=eps_w,
                    delta_w=delta_w,
                    T_w=T_w,
                    windows=per_window,
                    final=per_window[-1]["released"],
                )
                _dump(path, Wj)
                log(
                    f"windows {mode}/{accounting} n={n} eps={eps} (eps_w {eps_w:.4g}) seed={seed}: final test {per_window[-1]['released']['test']:.3f} "
                    f"(nominal {per_window[-1]['nominal']['test']:.3f}); per-window released test "
                    f"{[round(p['released']['test'], 3) for p in per_window]}"
                )


def run_release(C, a):
    S = _load(sweep_path(C), {})
    Hd = dict(
        task=C["task"],
        encoder=C.get("encoder"),
        eps=C["eps"],
        delta=C["delta"],
        groups={},
        provenance=stamp(),
    )
    for group in GROUPS:
        cfg = finalist(C, S, group)
        if cfg is None or cfg_key(cfg) not in S:
            continue
        entry = S[cfg_key(cfg)]
        man = _load(f"{C['runs_dir']}/envelopes/{cfg_key(cfg)}.json", None)
        if man is None:
            raise FileNotFoundError(f"no envelope manifest for {cfg_key(cfg)}")
        D = load_task(C, cfg["d"], a.device)
        T = evaluate_tight(C, D, head_of(cfg, D), man, C["seeds"], a.device)
        Bl = _load(
            f"{C['runs_dir']}/baselines_{group}.json",
            {
                "matched": {m: {} for m in MODES},
                "dpsgd_optimized": {m: {"selected": {}} for m in MODES},
                "pate": {m: {"selected": {}} for m in MODES},
            },
        )
        arms = {}
        for mode in MODES:
            arms[mode] = {}
            for eps in C["eps"]:
                e = str(eps)
                ags = {}
                allv = {**variants(entry["releases"][mode].get(e, {})), **variants_tight(T[mode][e])}
                for name, cell in allv.items():
                    mech, st = best_mech(cell)
                    ags[name] = dict(st, mechanism=mech)
                arms[mode][e] = dict(
                    ags=ags,
                    dpsgd_matched=Bl["matched"][mode].get(e),
                    dpsgd_optimized=Bl["dpsgd_optimized"][mode]["selected"].get(e),
                    pate=Bl["pate"][mode]["selected"].get(e),
                    non_private=entry["nominal"]["test"],
                )
        Hd["groups"][group] = dict(
            cfg=cfg,
            key=cfg_key(cfg),
            P=entry["P"],
            b=entry["b"],
            nominal=entry["nominal"],
            non_private_line=entry["nominal"]["test"],
            public_head=entry.get("public_head"),
            arms=arms,
        )
    out = f"{C['runs_dir']}/{C.get('headline_file', 'headline.json')}"
    _dump(out, Hd)
    log(f"wrote {out} ({list(Hd['groups'])})")
    g = lambda x: f"{x['test']:.3f}" if x else "  -  "
    for group, G in Hd["groups"].items():
        log(
            f"== {group}: {G['cfg']} P={G['P']} nominal {G['nominal']['test']:.3f}"
            + (f" public head {G['public_head']['test']:.3f}" if G["public_head"] else "")
        )
        for mode in MODES:
            heads = " ".join(f"{name:>8}" for name, _ in TABLE_VARIANTS)
            log(
                f"  {mode:6s} {'eps':>5} {'PATE':>6} {'DP-m':>6} {'DP-m(c)':>7} {'DP-opt':>6} {heads} {'mech':>12}"
            )
            for eps in C["eps"]:
                r = G["arms"][mode][str(eps)]
                m = r["dpsgd_matched"] or {}
                cols = " ".join(f"{g(r['ags'].get(v)):>8}" for _, v in TABLE_VARIANTS)
                log(
                    f"  {'':6s} {eps:>5} {g(r['pate']):>6} {g(m.get('basic')):>6} {g(m.get('composed')):>7} {g(r['dpsgd_optimized']):>6} {cols} {r['ags'][DEFAULT_VARIANT]['mechanism']:>12}"
                )


# ------------------------------------------------------------------------------------------------


def main(tasks=None, default_task="imdb"):
    tasks = TASKS if tasks is None else tasks
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("embed", "sweep", "baselines", "windows", "release", "all"))
    ap.add_argument("--task", choices=tuple(tasks), default=default_task)
    ap.add_argument(
        "--group",
        choices=GROUPS,
        default=None,
        help="baselines: which group (default: every group with a finalist)",
    )
    ap.add_argument(
        "--long",
        action="store_true",
        help="sweep: also certify the logistic finalist on the long trajectory (config `long`)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deadline", type=float, default=float("inf"))
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("pure", "approx"),
        default=None,
        help="windows: which release modes to run (default: the config's `windows.modes`, else pure)",
    )
    ap.add_argument(
        "--accounting",
        choices=("basic", "optimal"),
        default=None,
        help="windows: how the budget is split over the n windows (default: the config's, else basic)",
    )
    ap.add_argument("--windows-eps", type=float, nargs="+", default=None, dest="windows_eps")
    ap.add_argument("--windows-n", type=int, nargs="+", default=None, dest="windows_n")
    a = ap.parse_args()
    C = tasks[a.task]
    os.makedirs(C["runs_dir"], exist_ok=True)
    if a.command in ("embed", "all") and C.get("encoder"):
        for split in ("train", "test"):
            embeddings(C["encoder"], split, a.device)
    if a.command in ("sweep", "all"):
        run_sweep(C, a)
    if a.command in ("baselines", "all"):
        for group in [a.group] if a.group else C.get("groups", GROUPS):
            run_baselines(C, a, group)
    if a.command == "windows":
        run_windows(C, a)
    if a.command in ("release", "all"):
        run_release(C, a)


if __name__ == "__main__":
    main()
