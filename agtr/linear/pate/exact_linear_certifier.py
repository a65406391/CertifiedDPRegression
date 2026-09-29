"""Exact per-step parameter-space bounds for full-batch clipped GD on a 1-D linear model (no solver).

Setting (identical to what the MILP path certifies with concretisation after every step): the learner
y = w x + b, MSE, per-sample gradients g_j(θ) = c·(w x_j + b − y_j)·(x_j, 1) clamped per coordinate to
[−γ, γ], full-batch step θ ← θ − (lr/m) Σ_j clamp(g_j(θ)); adjacency = substitution of k of the m samples,
a substituted sample's clipped gradient being any point of [−γ, γ]².  Given the box B_t reached at step t,
the exact reachable box at step t+1 is computed in closed form:

  for a coordinate q ∈ {w, b},  s_j(θ) := clamp(g_j(θ)_q)
  max q'  =  max_{θ∈B_t} [ θ_q − (lr/m) Σ_j s_j(θ) + (lr/m) TopK_k(s(θ)) ] + (lr/m) k γ
  min q'  =  min_{θ∈B_t} [ θ_q − (lr/m) Σ_j s_j(θ) + (lr/m) BottomK_k(s(θ)) ] − (lr/m) k γ

Over the box, every s_j is affine in θ (clamp inactive), constant (clamp saturated) or "straddling".
Straddling samples are replaced by affine under/over-estimators of the clamp (DeepPoly-style chords and
tangent-mixtures; the only over-approximation); with that, the max-objective is convex and the
min-objective concave in θ, so both are attained at the 4 vertices of B_t and are evaluated exactly.

Why this is admissible as the certificate procedure: it is deterministic, it computes the exact (or an
inclusion-monotone over-approximate) one-step image, and for neighbouring datasets the (k+1)-instance on
D' can emulate every k-instance on D by substituting the differing sample -- so box(D, k) ⊆ box(D', k+1)
at every step, which is the monotonicity that makes max_k e^{−βk} Ā(x,k) β-smooth (RUNLOG.md).
The loss scale c is 2 (torch MSELoss); it is validated against the MILP cells in validate_exact_vs_milp().
"""

import numpy as np

C_LOSS = 2.0  # d/dpred (pred - y)^2


def _affine_range(a_w, a_b, a_0, wl, wu, bl, bu):
    """Interval of a_w w + a_b b + a_0 over the box."""
    lo = np.where(a_w >= 0, a_w * wl, a_w * wu) + np.where(a_b >= 0, a_b * bl, a_b * bu) + a_0
    hi = np.where(a_w >= 0, a_w * wu, a_w * wl) + np.where(a_b >= 0, a_b * bu, a_b * bl) + a_0
    return lo, hi


def _clamp_relaxation(a_w, a_b, a_0, lo, hi, gamma, mode="mono"):
    """Affine under- and over-estimators of clamp(a(θ), ±γ) valid on the box, where a(θ) = a_w w + a_b b + a_0
    ranges over [lo, hi].  Returns (under, over), each a (coef_w, coef_b, const) triple per sample.
      inactive  (lo >= -γ, hi <= γ): clamp = a                      -> both = a
      saturated (lo >= γ | hi <= -γ): clamp = ±γ                    -> both = const
      upper kink only (lo >= -γ, hi > γ): clamp concave in a       -> over: λ a + (1-λ) γ, λ = (γ-lo)/(hi-lo);  under: chord (lo,lo)->(hi,γ)
      lower kink only (hi <= γ, lo < -γ): clamp convex in a        -> under: λ a - (1-λ) γ, λ = (hi+γ)/(hi-lo);  over: chord (lo,-γ)->(hi,hi)
      both kinks: constants -γ / γ.
    Every case is a pointwise bound for all a in [lo, hi], hence for all θ in the box.

    mode="mono" (the RELEASE procedure) replaces the tangent-mixture over-estimator of the concave case by the
    constant γ and the tangent-mixture under-estimator of the convex case by the constant -γ.  With that every
    estimator is inclusion-monotone: enlarging the box can only lower the under-estimator and raise the
    over-estimator pointwise on the smaller box (chords of a concave/convex function move outward when the
    interval widens; constants and the exact affine piece never move inward).  Inclusion-monotonicity of every
    per-sample estimator is what makes the whole step monotone, box(D,k) ⊆ box(D',k+1), which the smooth
    sensitivity proof needs.  mode="affine" (tighter, diagnostics only) is NOT inclusion-monotone."""
    n = len(lo)
    z = np.zeros(n)
    uw, ub, u0 = a_w.copy(), a_b.copy(), a_0.copy()
    ow, ob, o0 = a_w.copy(), a_b.copy(), a_0.copy()
    sat_hi = lo >= gamma
    sat_lo = hi <= -gamma
    for m_, c in ((sat_hi, gamma), (sat_lo, -gamma)):
        uw[m_] = ub[m_] = ow[m_] = ob[m_] = 0.0
        u0[m_] = o0[m_] = c
    up = (lo >= -gamma) & (hi > gamma) & ~sat_hi
    if up.any():
        lam = (gamma - lo[up]) / (hi[up] - lo[up])
        sl = (gamma - lo[up]) / (hi[up] - lo[up])  # chord slope
        if mode == "mono":
            ow[up] = ob[up] = 0.0
            o0[up] = gamma
        else:
            ow[up], ob[up], o0[up] = lam * a_w[up], lam * a_b[up], lam * a_0[up] + (1 - lam) * gamma
        uw[up], ub[up], u0[up] = sl * a_w[up], sl * a_b[up], sl * a_0[up] + lo[up] - sl * lo[up]
    low = (hi <= gamma) & (lo < -gamma) & ~sat_lo
    if low.any():
        lam = (hi[low] + gamma) / (hi[low] - lo[low])
        sl = (hi[low] + gamma) / (hi[low] - lo[low])
        if mode == "mono":
            uw[low] = ub[low] = 0.0
            u0[low] = -gamma
        else:
            uw[low], ub[low], u0[low] = lam * a_w[low], lam * a_b[low], lam * a_0[low] - (1 - lam) * gamma
        ow[low], ob[low], o0[low] = sl * a_w[low], sl * a_b[low], sl * a_0[low] - gamma - sl * lo[low]
    both = (lo < -gamma) & (hi > gamma)
    uw[both] = ub[both] = ow[both] = ob[both] = 0.0
    u0[both] = -gamma
    o0[both] = gamma
    return (uw, ub, u0), (ow, ob, o0)


def step_box(box, X, y, k, lr, gamma, mode="mono"):
    """One exact step of the box.  box = (wl, wu, bl, bu) -> new box.

    For coordinate q:  q' = q − (lr/m)[Σ_{j∉S} s_j(θ) + Σ_{j∈S} u_j],  s_j = clamp(g_j(θ)_q),  u_j ∈ [−γ, γ].
      max q' = max_θ [ q − (lr/m) Σ_j s_j^under(θ) + (lr/m) TopK(s^over(θ)) ] + (lr/m) k γ
      min q' = min_θ [ q − (lr/m) Σ_j s_j^over(θ)  + (lr/m) BottomK(s^under(θ)) ] − (lr/m) k γ
    with affine under/over-estimators of the clamps (exact where the clamp is inactive or saturated), so the
    max-objective is convex and the min-objective concave in θ: both are attained at a vertex of the box."""
    wl, wu, bl, bu = box
    m = len(X)
    verts = np.array([(wl, bl), (wl, bu), (wu, bl), (wu, bu)])
    out = []
    for q in (0, 1):
        mult = X if q == 0 else np.ones_like(X)
        a_w, a_b, a_0 = C_LOSS * X * mult, C_LOSS * mult, -C_LOSS * y * mult
        lo, hi = _affine_range(a_w, a_b, a_0, wl, wu, bl, bu)
        (uw, ub, u0), (ow, ob, o0) = _clamp_relaxation(a_w, a_b, a_0, lo, hi, gamma, mode)
        best_max, best_min = -np.inf, np.inf
        for w, b in verts:
            s_under = uw * w + ub * b + u0
            s_over = ow * w + ob * b + o0
            base = w if q == 0 else b
            topk = np.sort(s_over)[-k:].sum() if k > 0 else 0.0
            best_max = max(best_max, base - lr / m * s_under.sum() + lr / m * topk)
            botk = np.sort(s_under)[:k].sum() if k > 0 else 0.0
            best_min = min(best_min, base - lr / m * s_over.sum() + lr / m * botk)
        out.append((best_min - lr / m * k * gamma, best_max + lr / m * k * gamma))
    return (out[0][0], out[0][1], out[1][0], out[1][1])


def nominal_step(theta, X, y, lr, gamma):
    w, b = theta
    r = w * X + b - y
    gw = np.clip(C_LOSS * r * X, -gamma, gamma)
    gb = np.clip(C_LOSS * r, -gamma, gamma)
    return np.array([w - lr * gw.mean(), b - lr * gb.mean()])


def step_box_split(box, X, y, k, lr, gamma, n_split=1, mode="mono"):
    """Image of the box as the union of the images of an n_split x n_split grid of sub-boxes (sound: the exact
    image of a union is the union of the images; each sub-box has fewer straddling clamps, so the affine
    relaxations are tighter).  n_split = 1 is step_box.
    DIAGNOSTICS ONLY: a box-relative grid is not inclusion-monotone (a larger box is cut differently), so
    n_split > 1 must not be used for a release; a fixed public lattice would be needed instead."""
    if n_split <= 1:
        return step_box(box, X, y, k, lr, gamma, mode)
    wl, wu, bl, bu = box
    ws = np.linspace(wl, wu, n_split + 1)
    bs = np.linspace(bl, bu, n_split + 1)
    out = np.array([np.inf, -np.inf, np.inf, -np.inf])
    for i in range(n_split):
        for j in range(n_split):
            r = step_box((ws[i], ws[i + 1], bs[j], bs[j + 1]), X, y, k, lr, gamma, mode)
            out = [min(out[0], r[0]), max(out[1], r[1]), min(out[2], r[2]), max(out[3], r[3])]
    return tuple(out)


def certify(X, y, k, steps, lr, gamma, init=(0.0, 0.0), n_split=1, mode="mono"):
    """Returns nominal trajectory (steps+1, 2) and boxes (steps+1, 2, 2) as [[wl, wu], [bl, bu]] (step 0 = init point)."""
    X = np.asarray(X, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    theta = np.array(init, dtype=np.float64)
    box = (theta[0], theta[0], theta[1], theta[1])
    noms, boxes = [theta.copy()], [np.array([[box[0], box[1]], [box[2], box[3]]])]
    for t in range(1, steps + 1):
        box = step_box_split(box, X, y, k, lr, gamma, n_split, mode)
        theta = nominal_step(theta, X, y, lr, gamma)
        # clipping ceiling: no coordinate can move more than lr*gamma per step (sound, data-independent)
        c = t * lr * gamma
        box = (
            max(box[0], init[0] - c),
            min(box[1], init[0] + c),
            max(box[2], init[1] - c),
            min(box[3], init[1] + c),
        )
        assert (
            box[0] - 1e-12 <= theta[0] <= box[1] + 1e-12 and box[2] - 1e-12 <= theta[1] <= box[3] + 1e-12
        ), "nominal left the box"
        noms.append(theta.copy())
        boxes.append(np.array([[box[0], box[1]], [box[2], box[3]]]))
    return np.stack(noms), np.stack(boxes)


def certify_all_k(X, y, ks, steps, lr, gamma, init=(0.0, 0.0), n_split=1, mode="mono"):
    """Nominal (steps+1, 2) and boxes (len(ks), steps+1, 2, 2); the nominal is k-independent."""
    nom = None
    allb = []
    for k in ks:
        n, b = certify(X, y, k, steps, lr, gamma, init, n_split, mode)
        nom = n if nom is None else nom
        allb.append(b)
    return nom, np.stack(allb)


# ----------------------------------------------------------------------------------------- validation
def validate_exact_vs_milp(tag="n4000_e20_lr0.3_g1", max_cells=None):
    """Compare with every finished MILP cell (all solves optimal): nominal trajectory, and the box
    reached from the MILP's own previous-step box (one-step exactness) as well as the full trajectory."""
    import glob
    import json

    root = f"agtr/linear/values/pate_matched/{tag}"
    cfg = json.load(open(f"{root}/config.json"))
    X = np.load(f"{root}/X_train.npy").ravel()
    y = np.load(f"{root}/y_train.npy").ravel()
    files = sorted(glob.glob(f"{root}/T*/shard*_k*.npz"))[:max_cells]
    worst = dict(
        nominal=0.0,
        one_step_rel=(-np.inf, None),
        traj_final_rel=(-np.inf, None),
        unsound=0,
        n=0,
        skipped_tl=0,
    )
    for f in files:
        z = np.load(f)
        if int(z["gurobi_time_limit"]) > 0 or int(z["gurobi_optimal"]) < int(z["gurobi_solves"]):
            worst["skipped_tl"] += 1
            continue
        idx = z["shard_idx"]
        k = int(z["k"])
        Xs, ys = X[idx], y[idx]
        nom, boxes = certify(Xs, ys, k, int(z["epochs"]), float(z["lr"]), float(z["gamma"]))
        ml, mu, mn = z["steps_l"], z["steps_u"], z["steps_n"]  # (steps, 2) after each step
        worst["nominal"] = max(worst["nominal"], float(np.abs(mn - nom[1:]).max()))
        # one-step: from the MILP box at step t, our step vs MILP box at step t+1 (relative to MILP width)
        for t in range(len(ml) - 1):
            b = step_box(
                (ml[t][0], mu[t][0], ml[t][1], mu[t][1]), Xs, ys, k, float(z["lr"]), float(z["gamma"])
            )
            ours = np.array([b[1] - b[0], b[3] - b[2]])
            theirs = mu[t + 1] - ml[t + 1]
            rel = float(((ours - theirs) / np.maximum(theirs, 1e-12)).max())
            if rel > worst["one_step_rel"][0]:
                worst["one_step_rel"] = (rel, f"{f.split('/')[-2]}/{f.split('/')[-1]} step {t + 1}")
            # unsound would be: our box strictly inside the (exact) MILP box beyond tolerance
            if (
                (b[0] > ml[t + 1][0] + 1e-7)
                or (b[1] < mu[t + 1][0] - 1e-7)
                or (b[2] > ml[t + 1][1] + 1e-7)
                or (b[3] < mu[t + 1][1] - 1e-7)
            ):
                worst["unsound"] += 1
        ours_f = boxes[-1][:, 1] - boxes[-1][:, 0]
        theirs_f = mu[-1] - ml[-1]
        rel = float(((ours_f - theirs_f) / np.maximum(theirs_f, 1e-12)).max())
        if rel > worst["traj_final_rel"][0]:
            worst["traj_final_rel"] = (rel, f"{f.split('/')[-2]}/{f.split('/')[-1]}")
        worst["n"] += 1
    return worst


if __name__ == "__main__":
    import sys
    import time

    t = time.time()
    print(validate_exact_vs_milp(max_cells=int(sys.argv[1]) if len(sys.argv) > 1 else None))
    print(f"{time.time() - t:.0f}s")
