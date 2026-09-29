"""The AGS release of a whole parameter vector: certificate, smooth sensitivity, noise scales.

An AGS release is one multivariate draw added to the final parameter vector, calibrated so that the
*whole* vector is (epsilon, delta)-DP. Never noise per parameter with its own epsilon: that is a
P * epsilon guarantee, the error an earlier MNIST run made.

Three steps, each a function here:

1. The certificate on the stored envelopes. `cross_radius_grid` gives the coordinate-wise cross-radius
   Abar_j(k) for every radius k = 0 .. K on a dense grid (gaps back-filled, boxes nested, T_(K+1) = T_K
   at the saturation radius, normally the batch size). `width_grid` gives the reference's certificate,
   the envelope width, on the same grid, so the two can be swapped and nothing else changes.
2. The smooth sensitivity, `smooth_sensitivity(A, beta)` = max_k e^(-beta k) A[k], for a per-coordinate
   grid (K+1, P) or for the l1 vector certificate sum_j Abar_j(k) (K+1,).
3. The scale of the product noise, `noise_scales`. The mechanism is theta + S * Z with Z a product of
   independent Cauchy (pure) or Laplace (approx) draws and S a scale vector. Its joint admissibility
   rests on two aggregate conditions over the P coordinates (the paper's per-coordinate appendix): a
   sliding condition sum_j |Delta_j| / S_j <= 1, met by any of the three allocations below because
   |Delta_j| <= Abar_j, and a dilation condition sum_j |ln S_j(D) - ln S_j(D')| <= beta, which for a
   data-dependent scale on P coordinates means each S_j must be (beta / P)-smooth. That is the
   `aggregate` beta rule and it is the sound one. The `dimension_free` rule keeps beta itself, which is
   what the paper's Corollary 4.1 states for the vector release; it is not established for P > 1 and is
   carried only so the two can be compared.

   Allocations: `l1`, one common scale from the smooth sensitivity of the l1 certificate sum_j Abar_j
   (the whole-space calibration); `anisotropic`, S_j = SS_j * sum_i sqrt(SS_i) / sqrt(SS_j), the
   reference's option, the least total noise under the sliding condition; `isotropic`, S_j = P * SS_j,
   the reference's default. All three are one epsilon for the whole vector.

Constants follow the pinned reference (ags/*/ss_reference.py): Cauchy scale (2 gamma_c / eps) S with
gamma_c = 2, Laplace scale (2 / eps) S.
"""

from __future__ import annotations

import numpy as np

from common import certificate

ALLOCATIONS = ("l1", "anisotropic", "isotropic")
BETA_RULES = ("aggregate", "dimension_free")


def _clean(boxes):
    return {
        int(k): (np.asarray(lo, float).ravel(), np.asarray(hi, float).ravel())
        for k, (lo, hi) in boxes.items()
    }


def cross_radius_grid(theta_n, boxes, saturation_k):
    """Dense Abar_j(k), k = 0 .. saturation_k, shape (K + 1, P); T_(K+1) := T_K."""
    boxes = _clean(boxes)
    if saturation_k not in boxes:
        raise ValueError(f"the saturation radius {saturation_k} must be a computed envelope")
    return certificate.cross_radius(
        np.asarray(theta_n, float).ravel(), boxes, tail=boxes[saturation_k], warn_on_gaps=False
    )


def width_grid(boxes, saturation_k):
    """Dense reference certificate W_j(k) = U_j^k - L_j^k on the same grid, back-filled and nested.

    W(0) is taken as W(1), which is what the reference staircase reads at its check point 0.
    """
    boxes = _clean(boxes)
    full = certificate.nest(certificate.backfill(boxes, saturation_k))
    W = np.stack([full[k][1] - full[k][0] for k in range(1, saturation_k + 1)])
    return np.vstack([W[:1], W])


def smooth_sensitivity(A, beta):
    """max_k e^(-beta k) A[k] over a dense saturated grid; per coordinate for 2-D A, scalar for 1-D."""
    A = np.asarray(A, float)
    if beta < 0:
        raise ValueError("beta must be non-negative")
    decay = np.exp(-beta * np.arange(A.shape[0]))
    return (decay[:, None] * A).max(axis=0) if A.ndim == 2 else float((decay * A).max())


def effective_beta(beta, n_params, rule):
    if rule == "aggregate":
        return beta / float(n_params)
    if rule == "dimension_free":
        return beta
    raise ValueError(f"unknown beta rule {rule!r}; one of {BETA_RULES}")


def noise_scales(A, beta_eff, eps, *, mode, allocation, gamma_cauchy=2):
    """Per-coordinate noise scales sigma_j for the product mechanism, and the S_j they came from.

    A is the dense per-coordinate certificate grid (K + 1, P); beta_eff the smoothing parameter after
    the beta rule; eps the budget the release is calibrated at (the base budget when amplified).
    """
    A = np.asarray(A, float)
    P = A.shape[1]
    if allocation == "l1":
        S = np.full(P, smooth_sensitivity(A.sum(axis=1), beta_eff))
    else:
        ss = smooth_sensitivity(A, beta_eff)
        if allocation == "anisotropic":
            root = np.sqrt(np.clip(ss, 1e-300, None))
            S = ss * root.sum() / root
        elif allocation == "isotropic":
            S = P * ss
        else:
            raise ValueError(f"unknown allocation {allocation!r}; one of {ALLOCATIONS}")
    multiplier = 2.0 * float(gamma_cauchy) / eps if mode == "pure" else 2.0 / eps
    return multiplier * S, S


# ------------------------------------------------------------------------------------------------
# The proven rules (Notes/AGS_RELEASE_ADMISSIBILITY.md). Every coordinate gets the same smoothness
# beta_j; a fixed share alpha of the budget pays for sliding (the shift of the centre) and the rest for
# dilation (the change of scale between neighbouring datasets); the scale of coordinate j is
# a_j S_j / alpha with sum_j 1/a_j <= 1 (l1: one common scale, a_j = 1, S_j = SS of the l1 certificate;
# anisotropic: a_j = sum_i sqrt(SS_i) / sqrt(SS_j)). The budget split alpha = eps/2 is a fixed rule, never
# chosen on the private certificate.
# ------------------------------------------------------------------------------------------------

TIGHT_ALLOCATIONS = ("l1", "anisotropic")


def laplace_dilation_beta(p, budget, delta, *, split):
    """The largest per-coordinate smoothness beta_j such that the dilation loss of p independent Laplace
    coordinates exceeds `budget` with probability at most delta.

    split="common"      one common scale: the loss is exactly D = p lam + (e^-lam - 1) G with G ~ Gamma(p, 1)
                        under the mechanism's own randomness, checked for both signs of lam.
    split="coordinate"  per-coordinate scales whose smoothness may move in any sign pattern: the bound
                        D <= p beta + (e^beta - 1) G, which gives up the cancellation of the Jacobian.
    """
    from scipy.optimize import brentq
    from scipy.stats import gamma

    if budget <= 0:
        raise ValueError("the dilation budget must be positive")
    p = int(p)

    def tail_common(lam):
        if lam > 0:  # D = p lam - (1 - e^-lam) G exceeds the budget when G is small
            g = (p * lam - budget) / (1.0 - np.exp(-lam))
            return gamma.cdf(g, p) if g > 0 else 0.0
        lam = -lam  # D = -p lam + (e^lam - 1) G exceeds the budget when G is large
        return gamma.sf((budget + p * lam) / (np.exp(lam) - 1.0), p)

    def tail_coordinate(beta):
        if p * beta >= budget:
            return 1.0
        return gamma.sf((budget - p * beta) / (np.exp(beta) - 1.0), p)

    if split == "common":
        # for lam > 0 the loss is at most p lam, so beyond budget/p the tail is 1 and below it a root exists;
        # for lam < 0 the loss is unbounded in G and a root always exists on a wide enough bracket
        hi = 2.0 * budget / p + 5.0

        def root(f):
            return brentq(f, 1e-12, hi) if f(hi) > 0 else hi

        pos = root(lambda l: tail_common(l) - delta)
        neg = root(lambda l: tail_common(-l) - delta)
        return float(min(pos, neg))
    if split == "coordinate":
        return float(brentq(lambda b: tail_coordinate(b) - delta, 1e-12, budget / p - 1e-15))
    raise ValueError(f"unknown split {split!r}")


def tight_scales(A, eps, *, mode, allocation, delta=None, alpha_frac=0.5):
    """Per-coordinate noise scales of the tightest proven release for the dense certificate grid A (K+1, P).

    pure   : Cauchy, beta_j = (eps - alpha) / P, scale a_j S_j / alpha           (Theorem 1 of the notes)
    approx : Laplace, beta_j from laplace_dilation_beta(P, eps - alpha, delta), scale a_j S_j / alpha
             (Theorem 2; "common" for the l1 allocation, "coordinate" for the anisotropic one)
    Returns (sigma, S, info) with info = {beta_j, alpha}.
    """
    A = np.asarray(A, float)
    P = A.shape[1]
    alpha = alpha_frac * eps
    if mode == "pure":
        beta_j = (eps - alpha) / P
    elif mode == "approx":
        if delta is None:
            raise ValueError("approx mode needs delta")
        beta_j = laplace_dilation_beta(
            P, eps - alpha, delta, split="common" if allocation == "l1" else "coordinate"
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if allocation == "l1":
        S = np.full(P, smooth_sensitivity(A.sum(axis=1), beta_j))
    else:
        ss = smooth_sensitivity(A, beta_j)
        if allocation == "anisotropic":
            root = np.sqrt(np.clip(ss, 1e-300, None))
            S = ss * root.sum() / root
        elif allocation == "isotropic":
            S = P * ss
        else:
            raise ValueError(f"unknown allocation {allocation!r}")
    return S / alpha, S, dict(beta_j=float(beta_j), alpha=float(alpha))


def corollary_scales(A, eps, *, split_beta=True):
    """The paper's single-release corollary: Cauchy(6 u / eps) on every coordinate with u the smooth
    sensitivity of the l1 certificate, at beta = eps / (6 P) (split_beta=True, the dilation factor P
    accounted for) or at beta = eps / 6 as written (split_beta=False, not established for P > 1)."""
    A = np.asarray(A, float)
    P = A.shape[1]
    beta = eps / 6.0 / (P if split_beta else 1.0)
    S = np.full(P, smooth_sensitivity(A.sum(axis=1), beta))
    return 6.0 * S / eps, S, dict(beta_j=float(beta), alpha=eps / 6.0)


TIGHT_RULES = ("tight_l1", "tight_aniso", "corollary_p")


def tight_variants(
    theta_n, boxes, b, eps_list, *, delta, seeds, score, draw, certs=("cross_radius", "width")
):
    """Every proven rule at every budget on stored envelopes: {mode: {eps: {rule: {cert: {mech: stats}}}}}.

    score(theta) -> dict of accuracies (select, test, ...); draw(theta_n, sigma, rng, mode) adds the noise
    ("pure" Cauchy, "approx" Laplace). In approx DP the pure Cauchy release is also (eps, delta)-DP, so
    both mechanisms are stored and the caller picks on the public selection set, as for the other rules.
    Median over `seeds` draws. Rules: tight_l1, tight_aniso (Theorems 1 and 2), corollary_p (pure only).
    """
    theta_n = np.asarray(theta_n, float).ravel()
    grids = {}
    if "cross_radius" in certs:
        grids["cross_radius"] = cross_radius_grid(theta_n, boxes, b)
    if "width" in certs:
        grids["width"] = width_grid(boxes, b)
    out = {}
    for mode in ("pure", "approx"):
        out[mode] = {}
        for eps in eps_list:
            row = {}
            for rule in TIGHT_RULES:
                for cert, G in grids.items():
                    cell = {}
                    mechs = ("cauchy",) if mode == "pure" else ("laplace", "cauchy")
                    for mech in mechs:
                        draw_mode = "pure" if mech == "cauchy" else "approx"
                        if rule == "corollary_p":
                            if mech != "cauchy":
                                continue
                            sigma, S, info = corollary_scales(G, eps)
                        else:
                            sigma, S, info = tight_scales(
                                G,
                                eps,
                                mode=draw_mode,
                                allocation="l1" if rule == "tight_l1" else "anisotropic",
                                delta=delta,
                            )
                        stats = [
                            score(draw(theta_n, sigma, np.random.RandomState(s), draw_mode))
                            for s in range(seeds)
                        ]
                        cell[mech] = {
                            **{k: float(np.median([st[k] for st in stats])) for k in stats[0]},
                            "sigma_mean": float(np.mean(sigma)),
                            "S_mean": float(np.mean(S)),
                            **info,
                        }
                    if cell:
                        row.setdefault(rule, {})[cert] = cell
            out[mode][str(eps)] = row
    return out
