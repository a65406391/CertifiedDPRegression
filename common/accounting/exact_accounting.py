"""Closed-form accountants, where an exact answer beats a numerical one.

    composed_gaussian_delta / calibrate_composed_gaussian
        the exact tradeoff function of a composed Gaussian mechanism, mu = 2 sqrt(steps) / multiplier
    composed_laplace_epsilon / calibrate_composed_laplace
        a pessimistic native Laplace privacy-loss composition

Both round the mechanism parameter away from the budget, so a floating-point step can only make the
guarantee stronger. Used for the strongest available comparison in the PCA-MNIST panel, where the
generic privacy-loss grid would only add slack.
"""

from __future__ import annotations

import math

import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from scipy.special import log_ndtr


def composed_gaussian_delta(epsilon: float, mu: float) -> float:
    """Exact hockey-stick divergence for a composed Gaussian tradeoff.

    ``mu`` is the l2 norm of all normalized mean shifts.  Thus ``steps``
    Gaussian mechanisms with replace-one normalized shift ``2 / multiplier``
    have ``mu = 2 * sqrt(steps) / multiplier``.
    """
    if epsilon < 0 or mu <= 0:
        raise ValueError("epsilon must be nonnegative and mu must be positive")
    log_first = float(log_ndtr(mu / 2.0 - epsilon / mu))
    log_second = float(epsilon + log_ndtr(-mu / 2.0 - epsilon / mu))
    # Analytically the second term is no larger.  Guard the last ulp near zero.
    difference = min(0.0, log_second - log_first)
    return float(math.exp(log_first) * (-math.expm1(difference)))


def calibrate_composed_gaussian(target_epsilon: float, delta: float, steps: int) -> tuple[float, float]:
    """Return the smallest safe noise multiplier and its exact delta."""
    if target_epsilon <= 0 or not 0 < delta < 1 or steps <= 0:
        raise ValueError("invalid Gaussian calibration arguments")
    low_mu, high_mu = 1e-12, 1.0
    while composed_gaussian_delta(target_epsilon, high_mu) <= delta:
        high_mu *= 2.0
    for _ in range(100):
        mid = (low_mu + high_mu) / 2.0
        if composed_gaussian_delta(target_epsilon, mid) <= delta:
            low_mu = mid
        else:
            high_mu = mid
    # Use the feasible endpoint and round the resulting multiplier upward.
    multiplier = math.nextafter(2.0 * math.sqrt(steps) / low_mu, math.inf)
    achieved_delta = composed_gaussian_delta(target_epsilon, 2.0 * math.sqrt(steps) / multiplier)
    return multiplier, achieved_delta


def composed_laplace_epsilon(
    epsilon_coordinate: float,
    coordinates: int,
    delta: float,
    *,
    interval: float = 1e-4,
) -> float:
    """Pessimistic discretized PLD epsilon for coordinate Laplace releases."""
    if epsilon_coordinate <= 0 or coordinates <= 0 or not 0 < delta < 1:
        raise ValueError("invalid Laplace accounting arguments")
    coordinate = privacy_loss_distribution.from_laplace_mechanism(
        parameter=1.0 / epsilon_coordinate,
        sensitivity=1.0,
        pessimistic_estimate=True,
        value_discretization_interval=interval,
    )
    return float(coordinate.self_compose(coordinates).get_epsilon_for_delta(delta))


def calibrate_composed_laplace(
    target_epsilon: float,
    delta: float,
    coordinates: int,
    *,
    interval: float = 1e-4,
) -> tuple[float, float]:
    """Largest coordinate parameter certified by the pessimistic PLD grid."""
    low, high = 1e-10, 1.0
    while composed_laplace_epsilon(high, coordinates, delta, interval=interval) <= target_epsilon:
        high *= 2.0
    for _ in range(36):
        mid = (low + high) / 2.0
        if composed_laplace_epsilon(mid, coordinates, delta, interval=interval) <= target_epsilon:
            low = mid
        else:
            high = mid
    # FFT/discretization evaluations can vary by a few ulps when reconstructed.
    # A two-ppm parameter margin is negligible for utility and makes the final
    # independently reconstructed pessimistic PLD strictly feasible.
    low = math.nextafter(low * (1.0 - 2e-6), 0.0)
    achieved = composed_laplace_epsilon(low, coordinates, delta, interval=interval)
    return low, achieved


def composed_pure_delta(epsilon_step: float, steps: int, epsilon_total) -> "float | np.ndarray":
    """The smallest delta at which `steps` adaptively composed (epsilon_step, 0)-DP mechanisms are
    (epsilon_total, delta)-DP for every such sequence: the optimal composition theorem of Kairouz, Oh and
    Viswanath (2015, Theorem 3.3, homogeneous case),

        delta = (1 + e^eps)^-k  sum_{i=0}^{k} C(k, i) max(e^{(k - i) eps} - e^{eps_g} e^{i eps}, 0),

    which is the hockey-stick divergence of the k-fold product of the worst-case pair (randomized
    response at eps). It is 0 at eps_g >= k eps (basic composition) and (e^eps - e^{eps_g}) / (1 + e^eps)
    at k = 1. `epsilon_total` may be an array. Evaluated in log space, so k in the hundreds is fine."""
    from scipy.special import gammaln, logsumexp

    eps, k = float(epsilon_step), int(steps)
    if eps <= 0 or k <= 0:
        raise ValueError("invalid pure composition arguments")
    eg = np.atleast_1d(np.asarray(epsilon_total, dtype=float))
    i = np.arange(k + 1)
    log_binom = gammaln(k + 1) - gammaln(i + 1) - gammaln(k - i + 1)
    loss = (k - 2 * i) * eps  # the privacy loss of an outcome with i "wrong" coordinates
    out = np.zeros_like(eg)
    for j, e in enumerate(eg):
        pos = loss > e
        if not pos.any():
            continue
        terms = log_binom[pos] + (k - i[pos]) * eps + np.log1p(-np.exp(e - loss[pos]))
        out[j] = float(np.exp(logsumexp(terms) - k * np.log1p(np.exp(eps))))
    return out if np.ndim(epsilon_total) else float(out[0])


def optimal_pure_budget(epsilon_total: float, steps: int, delta: float) -> float:
    """The largest per-step epsilon_0 such that `steps` adaptively composed (epsilon_0, 0)-DP mechanisms are
    (epsilon_total, delta)-DP, by composed_pure_delta (the optimal composition theorem). Never below the
    basic split epsilon_total / steps, where the composed delta is exactly 0; equal to it when the optimal
    theorem gains nothing (few steps, large per-step budgets). Rounded down, so the guarantee holds."""
    from scipy.optimize import brentq

    k = int(steps)
    if k <= 0 or epsilon_total <= 0 or not 0 < delta < 1:
        raise ValueError("invalid budget arguments")
    basic = float(epsilon_total) / k
    if k == 1:
        return basic
    gap = lambda e0: composed_pure_delta(e0, k, epsilon_total) - delta
    hi = float(epsilon_total)
    if gap(basic * (1 + 1e-9)) > 0:
        return basic
    if gap(hi) <= 0:
        return hi
    e0 = brentq(gap, basic * (1 + 1e-9), hi, xtol=1e-12)
    while gap(e0) > 0:  # brentq's root may sit a hair above the level set
        e0 = math.nextafter(e0, 0.0)
    return float(e0)
