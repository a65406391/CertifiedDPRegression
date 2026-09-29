"""Renyi accounting for scalar Laplace laws with data-dependent scales.

The normalized neighboring pair is

    P = Laplace(0, 1), Q = Laplace(shift, scale_ratio).

This module evaluates both privacy directions analytically. It also supplies a
numerical grid maximizer over the smooth-Laplace feasible set used by the AGS
experiments. The grid routine rounds each sampled value upward but does not bound
gaps between grid points; a result is labelled fully certified only after an
interval/monotonicity validation of the continuous maximization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import mpmath as mp
from scipy.special import logsumexp


def laplace_renyi(alpha: float, shift: float, scale_ratio: float) -> float:
    """Return D_alpha(Lap(0,1) || Lap(shift, scale_ratio)).

    The integral is split at zero and ``abs(shift)``.  ``inf`` is returned when
    the tails are not integrable.
    """
    if alpha <= 1:
        raise ValueError("alpha must exceed one")
    if scale_ratio <= 0:
        raise ValueError("scale_ratio must be positive")
    a = abs(float(shift))
    r = float(scale_ratio)
    c = (alpha - 1.0) / r
    tail_rate = alpha - c
    if tail_rate <= 0:
        return math.inf

    left = math.exp(c * a) / tail_rate
    if a == 0:
        middle = 0.0
    else:
        rate = alpha + c
        # exp(c*a) * (1-exp(-rate*a)) / rate, evaluated stably.
        middle = math.exp(c * a) * (-math.expm1(-rate * a)) / rate
    right = math.exp(-alpha * a) / tail_rate
    integral = 0.5 * (r ** (alpha - 1.0)) * (left + middle + right)
    return math.log(integral) / (alpha - 1.0)


def bidirectional_laplace_renyi(alpha: float, shift: float, scale_ratio: float) -> float:
    """Maximum Renyi divergence in both neighboring directions."""
    forward = laplace_renyi(alpha, shift, scale_ratio)
    reverse = laplace_renyi(alpha, abs(shift) / scale_ratio, 1.0 / scale_ratio)
    return max(forward, reverse)


def _laplace_renyi_array(alpha: float, shifts, ratios) -> np.ndarray:
    """Vectorized internal form of :func:`laplace_renyi`."""
    a = np.abs(np.asarray(shifts, dtype=float))
    r = np.asarray(ratios, dtype=float)
    c = (alpha - 1.0) / r
    tail = alpha - c
    valid = tail > 0
    safe_tail = np.where(valid, tail, 1.0)
    rate = alpha + c
    log_left = c * a - np.log(safe_tail)
    middle_factor = -np.expm1(-rate * a)
    log_middle = np.full(np.broadcast(a, r).shape, -np.inf, dtype=float)
    positive = middle_factor > 0
    log_middle[positive] = (c * a)[positive] + np.log(middle_factor[positive]) - np.log(rate[positive])
    log_right = -alpha * a - np.log(safe_tail)
    stacked = np.stack(np.broadcast_arrays(log_left, log_middle, log_right), axis=0)
    log_integral = -math.log(2.0) + (alpha - 1.0) * np.log(r) + logsumexp(stacked, axis=0)
    result = log_integral / (alpha - 1.0)
    return np.where(valid, result, np.inf)


def bidirectional_laplace_renyi_array(alpha: float, shifts, ratios) -> np.ndarray:
    """Vectorized bidirectional divergence for matching arrays."""
    a = np.asarray(shifts, dtype=float)
    r = np.asarray(ratios, dtype=float)
    forward = _laplace_renyi_array(alpha, a, r)
    reverse = _laplace_renyi_array(alpha, a / r, 1.0 / r)
    return np.maximum(forward, reverse)


def joint_common_scale_laplace_renyi_array(alpha: float, shifts, ratios, coordinates: int) -> np.ndarray:
    """Bidirectional RDP for an iid Laplace vector with one shared scale.

    ``shifts`` is the normalized joint l1 displacement.  The implementation
    places it in one coordinate.  This is the extremal allocation when the
    scalar log integral is convex in absolute displacement; the experiment
    driver separately checks this against dense simplex allocations.
    """
    if coordinates < 1:
        raise ValueError("coordinates must be positive")
    a = np.asarray(shifts, dtype=float)
    r = np.asarray(ratios, dtype=float)
    forward = _laplace_renyi_array(alpha, a, r) + (coordinates - 1) * _laplace_renyi_array(
        alpha, np.zeros_like(a), r
    )
    reverse = _laplace_renyi_array(alpha, a / r, 1 / r) + (coordinates - 1) * _laplace_renyi_array(
        alpha, np.zeros_like(a), 1 / r
    )
    return np.maximum(forward, reverse)


def worst_case_joint_smooth_laplace_rdp_grid(
    alpha: float,
    epsilon_joint: float,
    beta: float,
    coordinates: int,
    *,
    ratio_points: int = 2001,
    upward_slack: float = 1e-12,
) -> SmoothLaplaceRdpPoint:
    """Grid maximum for a joint-l1 smooth Laplace release."""
    ratios = np.exp(np.linspace(-beta, beta, ratio_points))
    shifts = 0.5 * epsilon_joint * np.minimum(1.0, ratios)
    values = joint_common_scale_laplace_renyi_array(alpha, shifts, ratios, coordinates)
    index = int(np.argmax(values))
    return SmoothLaplaceRdpPoint(
        float(alpha), float(values[index] + upward_slack), float(ratios[index]), float(shifts[index])
    )


@dataclass(frozen=True)
class SmoothLaplaceRdpPoint:
    alpha: float
    rdp: float
    scale_ratio: float
    shift: float


@dataclass(frozen=True)
class SmoothLaplaceRdpInterval:
    alpha: float
    lower: float
    upper: float
    boxes: int


def _laplace_renyi_interval(alpha: float, shift, ratio):
    """Interval extension of the analytic scalar Renyi expression."""
    iv = mp.iv
    c = (alpha - 1.0) / ratio
    tail = alpha - c
    if float(tail.a) <= 0:
        return iv.mpf([0, mp.inf])
    rate = alpha + c
    left = iv.exp(c * shift) / tail
    middle = iv.exp(c * shift) * (1 - iv.exp(-rate * shift)) / rate
    right = iv.exp(-alpha * shift) / tail
    integral = iv.mpf("0.5") * iv.power(ratio, alpha - 1.0) * (left + middle + right)
    return iv.log(integral) / (alpha - 1.0)


def certified_smooth_laplace_rdp(
    alpha: float,
    epsilon_coordinate: float,
    beta: float,
    *,
    tolerance: float = 1e-8,
    max_boxes: int = 200000,
) -> SmoothLaplaceRdpInterval:
    """Interval-certify the continuous scalar smooth-Laplace RDP maximum.

    For fixed scales, differentiating the unlogged Renyi integral with respect
    to absolute shift gives a positive factor times
    ``exp(c*a)-exp(-alpha*a)``. Thus the divergence is nondecreasing in shift,
    and it suffices to set ``a=epsilon/2*min(1,r)``. The remaining one-dimensional
    log-scale domain is enclosed by interval branch-and-bound, separately on
    either side of ``r=1``.
    """
    if alpha <= 1 or epsilon_coordinate <= 0 or beta < 0:
        raise ValueError("invalid smooth-Laplace parameters")
    if tolerance <= 0 or max_boxes < 2:
        raise ValueError("invalid interval search controls")

    def point_value(t: float) -> float:
        r = math.exp(t)
        shift = 0.5 * epsilon_coordinate * min(1.0, r)
        return bidirectional_laplace_renyi(alpha, shift, r)

    def interval_value(lo: float, hi: float) -> float:
        iv = mp.iv
        t = iv.mpf([lo, hi])
        r = iv.exp(t)
        # Boxes never cross zero, so min(1,r) has one analytic branch.
        shift = 0.5 * epsilon_coordinate * (r if hi <= 0 else iv.mpf(1))
        forward = _laplace_renyi_interval(alpha, shift, r)
        reverse = _laplace_renyi_interval(alpha, shift / r, 1 / r)
        return max(float(forward.b), float(reverse.b))

    intervals = [(-beta, 0.0)] if beta == 0 else [(-beta, 0.0), (0.0, beta)]
    lower = max(point_value(-beta), point_value(0.0), point_value(beta))
    # Max-heap via negative upper endpoint.
    import heapq

    heap = []
    for lo, hi in intervals:
        heapq.heappush(heap, (-interval_value(lo, hi), lo, hi))
    boxes = len(heap)
    while heap and -heap[0][0] - lower > tolerance:
        if boxes >= max_boxes:
            raise RuntimeError("interval RDP maximization did not converge")
        _, lo, hi = heapq.heappop(heap)
        mid = (lo + hi) / 2
        lower = max(lower, point_value(mid))
        for left, right in ((lo, mid), (mid, hi)):
            upper = interval_value(left, right)
            if upper > lower:
                heapq.heappush(heap, (-upper, left, right))
            boxes += 1
    upper = max(lower, -heap[0][0] if heap else lower)
    return SmoothLaplaceRdpInterval(alpha, lower, upper, boxes)


def worst_case_smooth_laplace_rdp_grid(
    alpha: float,
    epsilon_coordinate: float,
    beta: float,
    *,
    ratio_points: int = 2001,
    shift_points: int = 65,
    upward_slack: float = 1e-12,
) -> SmoothLaplaceRdpPoint:
    """Grid-search the complete theorem-feasible scalar pair.

    For scale ratio ``r``, smoothness and local sensitivity give

        r in [exp(-beta), exp(beta)]
        |shift| <= epsilon_coordinate/2 * min(1, r).

    Both the scale and shift dimensions are searched, although the maximum is
    normally attained at maximal shift.  ``upward_slack`` prevents reporting a
    value below the evaluated floating-point maximum.
    """
    if epsilon_coordinate <= 0 or beta < 0:
        raise ValueError("epsilon_coordinate must be positive and beta nonnegative")
    ratios = np.exp(np.linspace(-beta, beta, ratio_points))
    fractions = np.linspace(0.0, 1.0, shift_points)
    ratio_grid, fraction_grid = np.meshgrid(ratios, fractions, indexing="ij")
    shifts = 0.5 * epsilon_coordinate * np.minimum(1.0, ratio_grid) * fraction_grid
    values = bidirectional_laplace_renyi_array(alpha, shifts, ratio_grid)
    flat_index = int(np.argmax(values))
    row, column = np.unravel_index(flat_index, values.shape)
    best = (float(values[row, column]), float(ratio_grid[row, column]), float(shifts[row, column]))
    return SmoothLaplaceRdpPoint(float(alpha), float(best[0] + upward_slack), best[1], best[2])


def epsilon_from_rdp(orders, rdp_values, delta: float) -> tuple[float, float]:
    """Tight standard RDP-to-DP conversion and minimizing order.

    This matches ``dp_accounting.rdp.compute_epsilon`` (the improved conversion
    of Canonne et al.) so every mechanism in the matched panel uses the same
    conversion rather than the older basic bound.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    candidates = []
    for order, rdp in zip(orders, rdp_values):
        order, rdp = float(order), float(rdp)
        if order <= 1:
            raise ValueError("RDP orders must exceed one")
        if rdp < 0:
            raise ValueError("RDP values must be nonnegative")
        if delta * delta + math.expm1(-rdp) > 0:
            candidate = 0.0
        elif order > 1.01:
            candidate = rdp + math.log1p(-1.0 / order) - math.log(delta * order) / (order - 1.0)
        else:
            candidate = math.inf
        candidates.append(candidate)
    index = int(np.argmin(candidates))
    return max(0.0, candidates[index]), float(list(orders)[index])
