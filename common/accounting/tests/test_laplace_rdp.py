import math

import numpy as np
from scipy.integrate import quad

from common.accounting.laplace_rdp import (
    bidirectional_laplace_renyi,
    certified_smooth_laplace_rdp,
    epsilon_from_rdp,
    laplace_renyi,
    joint_common_scale_laplace_renyi_array,
    worst_case_smooth_laplace_rdp_grid,
)

from dp_accounting.rdp import rdp_privacy_accountant


def _quadrature(alpha, shift, ratio):
    def integrand(x):
        log_p = -math.log(2) - abs(x)
        log_q = -math.log(2 * ratio) - abs(x - shift) / ratio
        return math.exp(alpha * log_p + (1 - alpha) * log_q)

    value = quad(integrand, -np.inf, np.inf, points=None, epsabs=1e-11)[0]
    return math.log(value) / (alpha - 1)


def test_identity_is_zero():
    for alpha in (1.1, 2, 5, 10):
        assert abs(laplace_renyi(alpha, 0, 1)) < 1e-13


def test_fixed_scale_matches_known_formula():
    # Mironov's closed form for equal-scale Laplace laws.
    for alpha in (1.1, 2, 3.5, 10):
        for shift in (0.01, 0.2, 1.0):
            integral = alpha / (2 * alpha - 1) * math.exp((alpha - 1) * shift) + (alpha - 1) / (
                2 * alpha - 1
            ) * math.exp(-alpha * shift)
            expected = math.log(integral) / (alpha - 1)
            assert laplace_renyi(alpha, shift, 1) == pytest.approx(expected, rel=1e-12)


def test_matches_numerical_quadrature():
    for alpha, shift, ratio in ((1.2, 0.1, 0.98), (2, 0.4, 1.1), (5, 0.2, 1.03)):
        assert laplace_renyi(alpha, shift, ratio) == pytest.approx(
            _quadrature(alpha, shift, ratio), rel=2e-9, abs=2e-10
        )


def test_reverse_direction_transformation():
    alpha, shift, ratio = 3.0, 0.2, 1.1
    reverse = laplace_renyi(alpha, shift / ratio, 1 / ratio)
    assert bidirectional_laplace_renyi(alpha, shift, ratio) == max(
        laplace_renyi(alpha, shift, ratio), reverse
    )


def test_grid_search_includes_boundaries_and_max_shift():
    point = worst_case_smooth_laplace_rdp_grid(2.0, 0.5, 0.02, ratio_points=101, shift_points=33)
    assert math.exp(-0.02) <= point.scale_ratio <= math.exp(0.02)
    assert 0 <= point.shift <= 0.25 * min(1, point.scale_ratio) + 1e-15
    # A nonzero admissible shift must be worse than the identical pair.
    assert point.rdp > 0


def test_joint_common_scale_reduces_to_scalar():
    alpha = 2.5
    got = joint_common_scale_laplace_renyi_array(alpha, np.array([0.2]), np.array([1.03]), 1)[0]
    assert got == pytest.approx(bidirectional_laplace_renyi(alpha, 0.2, 1.03))


def test_concentrated_joint_shift_dominates_even_split():
    # Numerically validate the extremal allocation used by the joint accountant.
    for alpha in (1.2, 2, 5, 10):
        for ratio in (0.98, 1.0, 1.02):
            total, coordinates = 0.4, 9
            concentrated_f = laplace_renyi(alpha, total, ratio) + (coordinates - 1) * laplace_renyi(
                alpha, 0, ratio
            )
            split_f = coordinates * laplace_renyi(alpha, total / coordinates, ratio)
            concentrated_r = laplace_renyi(alpha, total / ratio, 1 / ratio) + (
                coordinates - 1
            ) * laplace_renyi(alpha, 0, 1 / ratio)
            split_r = coordinates * laplace_renyi(alpha, total / (coordinates * ratio), 1 / ratio)
            assert concentrated_f + 1e-12 >= split_f
            assert concentrated_r + 1e-12 >= split_r


def test_rdp_conversion_matches_dp_accounting():
    orders = np.array([1.5, 2.0, 4.0, 8.0, 16.0])
    values = np.array([0.1, 0.15, 0.3, 0.7, 1.5])
    expected = rdp_privacy_accountant.compute_epsilon(orders, values, 1e-5)
    got = epsilon_from_rdp(orders, values, 1e-5)
    assert got == pytest.approx(expected)


def test_interval_max_encloses_dense_grid():
    alpha, epsilon, beta = 3.0, 2.0, 0.2
    certified = certified_smooth_laplace_rdp(alpha, epsilon, beta, tolerance=1e-7)
    grid = worst_case_smooth_laplace_rdp_grid(
        alpha,
        epsilon,
        beta,
        ratio_points=20001,
        shift_points=2,
        upward_slack=0.0,
    )
    assert certified.lower <= grid.rdp + 1e-12
    assert grid.rdp <= certified.upper + 1e-12
    assert certified.upper - certified.lower <= 1e-7


import pytest  # noqa: E402  (keeps the mathematical imports grouped above)
