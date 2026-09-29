"""Fixed-pair PLDs for Laplace laws with unequal locations and scales.

This module is exact for a specified pair.  It does not maximize over the
continuous smooth-sensitivity scale-ratio envelope; callers must not treat an
endpoint search as a certified accountant for data-dependent scales.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution


def _standard_laplace_cdf(value: float) -> float:
    if value < 0:
        return 0.5 * math.exp(value)
    return 1.0 - 0.5 * math.exp(-value)


def _interval_probability(lower: float, upper: float) -> float:
    lower_cdf = 0.0 if lower == -math.inf else _standard_laplace_cdf(lower)
    upper_cdf = 1.0 if upper == math.inf else _standard_laplace_cdf(upper)
    return max(0.0, upper_cdf - lower_cdf)


def laplace_pair_privacy_loss_cdf(shift: float, scale_ratio: float) -> Callable[[float], float]:
    """CDF of log(p/q) under p for p=Lap(0,1), q=Lap(shift,r)."""
    if shift < 0 or scale_ratio <= 0:
        raise ValueError("shift must be nonnegative and scale_ratio positive")
    a, r = float(shift), float(scale_ratio)
    log_r = math.log(r)
    segments = (
        (-math.inf, 0.0, log_r + a / r, 1.0 - 1.0 / r),
        (0.0, a, log_r + a / r, -1.0 - 1.0 / r),
        (a, math.inf, log_r - a / r, -1.0 + 1.0 / r),
    )

    def cdf(value: float) -> float:
        probability = 0.0
        for lower, upper, intercept, slope in segments:
            if abs(slope) < 1e-15:
                if intercept <= value:
                    probability += _interval_probability(lower, upper)
            elif slope > 0:
                endpoint = min(upper, (value - intercept) / slope)
                if endpoint > lower:
                    probability += _interval_probability(lower, endpoint)
            else:
                endpoint = max(lower, (value - intercept) / slope)
                if upper > endpoint:
                    probability += _interval_probability(endpoint, upper)
        return min(1.0, max(0.0, probability))

    return cdf


def fixed_pair_laplace_pld(
    shift: float,
    scale_ratio: float,
    *,
    interval: float = 0.002,
    tail_mass_truncation: float = 1e-12,
):
    """Return a pessimistically discretized PLD for one specified direction."""
    return privacy_loss_distribution.create_from_cdf(
        laplace_pair_privacy_loss_cdf(shift, scale_ratio),
        pessimistic_estimate=True,
        value_discretization_interval=interval,
        tail_mass_truncation=tail_mass_truncation,
    )


def continuous_ratio_laplace_pld_envelope(
    epsilon_coordinate: float,
    beta: float,
    *,
    privacy_loss_interval: float = 0.001,
    output_interval: float = 0.002,
    ratio_intervals: int = 1000,
    tail_mass_truncation: float = 1e-14,
):
    """Dominating PLD for the complete smooth-Laplace neighboring envelope.

    For normalized scale ratio ``r``, the feasible absolute shift is at most
    ``epsilon_coordinate / 2 * min(1, r)``.  Output and log-ratio space are
    partitioned into rectangles.  On each rectangle the privacy loss

        log(r) - abs(x) + abs(x-a) / r

    is rounded up using elementary interval bounds, and the exact standard-
    Laplace probability of the output interval is assigned to that upper
    value.  Truncated output tails are assigned infinite privacy loss.  The
    resulting random variable pointwise dominates every fixed neighboring-pair
    privacy-loss variable.  The ratio domain is symmetric, so it also covers
    the reverse privacy direction.
    """
    if epsilon_coordinate <= 0 or beta < 0:
        raise ValueError("epsilon_coordinate must be positive and beta nonnegative")
    if privacy_loss_interval <= 0 or output_interval <= 0:
        raise ValueError("discretization intervals must be positive")
    if ratio_intervals < 2 or not 0 < tail_mass_truncation < 1:
        raise ValueError("invalid envelope discretization")

    max_output = -math.log(tail_mass_truncation)
    edges = np.arange(
        -max_output,
        max_output + output_interval,
        output_interval,
        dtype=float,
    )
    lower_x, upper_x = edges[:-1], edges[1:]
    minimum_abs_x = np.where(
        (lower_x <= 0) & (upper_x >= 0),
        0.0,
        np.minimum(np.abs(lower_x), np.abs(upper_x)),
    )
    upper_loss = np.full(len(lower_x), -np.inf, dtype=float)
    if beta == 0:
        log_ratio_cells = [(0.0, 0.0)]
    else:
        log_ratio_edges = np.unique(
            np.r_[
                np.linspace(-beta, beta, ratio_intervals + 1),
                0.0,
            ]
        )
        log_ratio_cells = list(zip(log_ratio_edges[:-1], log_ratio_edges[1:]))
    max_shift = 0.5 * epsilon_coordinate
    for lower_log_ratio, upper_log_ratio in log_ratio_cells:
        lower_ratio = math.exp(lower_log_ratio)
        upper_ratio = math.exp(upper_log_ratio)
        if upper_ratio <= 1.0:
            upper_shift = max_shift * upper_ratio
        elif lower_ratio >= 1.0:
            upper_shift = max_shift
        else:  # The explicit zero edge means a cell cannot cross ratio one.
            raise AssertionError("log-ratio interval crossed zero")
        # Shift sign is immaterial in distribution by symmetry.  Cover every
        # nonnegative magnitude from zero to the feasible upper endpoint.
        maximum_difference = np.maximum.reduce(
            (
                np.abs(lower_x),
                np.abs(upper_x),
                np.abs(lower_x - upper_shift),
                np.abs(upper_x - upper_shift),
            )
        )
        rectangle_upper = upper_log_ratio - minimum_abs_x + maximum_difference / lower_ratio
        upper_loss = np.maximum(upper_loss, rectangle_upper)

    lower_mass = np.where(
        lower_x < 0,
        0.5 * np.exp(lower_x),
        1.0 - 0.5 * np.exp(-lower_x),
    )
    upper_mass = np.where(
        upper_x < 0,
        0.5 * np.exp(upper_x),
        1.0 - 0.5 * np.exp(-upper_x),
    )
    probabilities = upper_mass - lower_mass
    rounded = np.ceil(upper_loss / privacy_loss_interval).astype(int)
    probability_mass = collections.defaultdict(float)
    for value, probability in zip(rounded, probabilities):
        probability_mass[int(value)] += float(probability)
    infinity_mass = max(0.0, 1.0 - sum(probability_mass.values()))
    return privacy_loss_distribution.PrivacyLossDistribution.create_from_rounded_probability(
        probability_mass,
        infinity_mass,
        privacy_loss_interval,
        pessimistic_estimate=True,
    )
