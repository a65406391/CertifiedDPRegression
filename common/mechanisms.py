"""Noise mechanisms and their smoothing limits, in one place.

Before this module every analysis wrote these formulas out by hand, ten times over, and two
different Gaussian constants were in live use. Definition 3.6 of the paper is the `1.25/delta` form;
`sqrt(2 ln(1/delta))` appeared in several legacy paths and understates the noise.

| regime          | ours, on smooth sensitivity      | baseline, on global sensitivity | beta limit             |
|-----------------|----------------------------------|---------------------------------|------------------------|
| pure (eps, 0)   | Cauchy(6 SS / eps)               | Laplace(GS / eps)               | eps / 6                |
| pure, tighter   | Cauchy(SS / (eps - beta))        | Laplace(GS / eps)               | any 0 < beta < eps     |
| approx (eps, d) | Laplace(2 SS / eps)              | Gaussian(GS_2 sqrt(2 ln(1.25/d)) / eps) | eps / (2 ln(2/d)) |

The tighter Cauchy is Liang et al. (2024), the paper's `cor:TighterCauchy`. At beta = eps/6 it is
five times less noise than Definition 3.7, so only use it where that corollary is part of the claim.
"""

from __future__ import annotations

import math

import numpy as np


def beta_max(epsilon: float, delta: float | None = None, mechanism: str = "cauchy") -> float:
    """Largest smoothing parameter the mechanism admits at this budget.

    A smaller beta gives a larger smooth sensitivity, so beta must be set per epsilon; a fixed
    beta = 1/6 is only valid for epsilon >= 1 (pure) and epsilon >= 4.07 at delta = 1e-5 (approx).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if mechanism in ("cauchy", "liang"):
        return epsilon / 6.0
    if mechanism == "laplace":
        if delta is None or not 0 < delta < 1:
            raise ValueError("the Laplace mechanism on smooth sensitivity needs delta in (0, 1)")
        return epsilon / (2.0 * math.log(2.0 / delta))
    raise ValueError(f"unknown mechanism {mechanism!r}")


def cauchy_scale(smooth_sensitivity, epsilon: float):
    """Definition 3.7: Cauchy(6 SS / eps) is (eps, 0)-DP when beta <= eps / 6."""
    return 6.0 * np.asarray(smooth_sensitivity, dtype=float) / epsilon


def liang_cauchy_scale(smooth_sensitivity, epsilon: float, beta: float):
    """cor:TighterCauchy (Liang et al. 2024): Cauchy(SS / (eps - beta)) for any 0 < beta < eps."""
    if not 0 < beta < epsilon:
        raise ValueError("need 0 < beta < epsilon")
    return np.asarray(smooth_sensitivity, dtype=float) / (epsilon - beta)


def laplace_ss_scale(smooth_sensitivity, epsilon: float):
    """Definition 3.8: Laplace(2 SS / eps) is (eps, delta)-DP when beta <= eps / (2 ln(2/delta))."""
    return 2.0 * np.asarray(smooth_sensitivity, dtype=float) / epsilon


def laplace_gs_scale(global_sensitivity, epsilon: float):
    """Definition 3.5: Laplace(GS / eps), the pure-DP baseline."""
    return np.asarray(global_sensitivity, dtype=float) / epsilon


def gaussian_gs_sigma(global_l2_sensitivity, epsilon: float, delta: float):
    """Definition 3.6: sigma = GS_2 sqrt(2 ln(1.25/delta)) / eps, the approximate-DP baseline."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return np.asarray(global_l2_sensitivity, dtype=float) * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# The release modes every AGT-R analysis reports, defined once. Each entry gives the smoothing
# parameter for a budget and the multiplier applied to the smooth sensitivity to get the noise
# scale, so `scale = multiplier(eps, beta) * SS^beta`.
#
#   pure           Definition 3.7, Cauchy(6 SS / eps) at beta = eps/6
#   pure_liang_b6  cor:TighterCauchy at beta = eps/6, five times less noise than Definition 3.7
#   pure_liang_b2  the same corollary at beta = eps/2, which trades a larger beta for a smaller scale
#   approx         Definition 3.8, Laplace(2 SS / eps) at the largest admissible beta
#
# The Liang modes are only reportable where that corollary is part of the paper's claims.
RELEASE_MODES = {
    "pure": dict(
        family="cauchy", beta=lambda eps, delta=None: eps / 6.0, multiplier=lambda eps, beta: 6.0 / eps
    ),
    "pure_liang_b6": dict(
        family="cauchy",
        beta=lambda eps, delta=None: eps / 6.0,
        multiplier=lambda eps, beta: 1.0 / (eps - beta),
    ),
    "pure_liang_b2": dict(
        family="cauchy",
        beta=lambda eps, delta=None: eps / 2.0,
        multiplier=lambda eps, beta: 1.0 / (eps - beta),
    ),
    "approx": dict(
        family="laplace",
        beta=lambda eps, delta: eps / (2.0 * math.log(2.0 / delta)),
        multiplier=lambda eps, beta: 2.0 / eps,
    ),
}

PURE_MODES = tuple(m for m, v in RELEASE_MODES.items() if v["family"] == "cauchy")
