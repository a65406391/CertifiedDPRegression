import numpy as np

from common.accounting.accounting import dpsgd_epsilon
from common.accounting.exact_accounting import (
    calibrate_composed_gaussian,
    calibrate_composed_laplace,
    composed_gaussian_delta,
    composed_laplace_epsilon,
)
from ags.mnist.accounting.pca.run_strong_accounting import (
    public_class_bias_recalibration,
)


def test_exact_gaussian_calibration_hits_delta_and_matches_fine_pld():
    multiplier, achieved_delta = calibrate_composed_gaussian(20.0, 1e-5, 20)
    assert achieved_delta <= 1e-5
    assert np.isclose(achieved_delta, 1e-5, rtol=1e-10)
    pld_epsilon = dpsgd_epsilon(multiplier, q=1.0, steps=20, delta=1e-5, interval=2e-4)
    assert abs(pld_epsilon - 20.0) < 2e-4


def test_gaussian_delta_increases_with_composed_shift():
    assert composed_gaussian_delta(4.0, 0.5) < composed_gaussian_delta(4.0, 1.0)


def test_laplace_pld_calibration_is_feasible_and_grid_converged():
    parameter, achieved = calibrate_composed_laplace(4.0, 1e-5, 9, interval=2e-4)
    assert achieved <= 4.0
    fine = composed_laplace_epsilon(parameter * (1 - 2e-6), 9, 1e-5, interval=1e-4)
    assert fine <= 4.0


def test_public_class_bias_recalibration_preserves_majority_direction():
    # Ten-dimensional features make the flat 10-class parameter vector length 110.
    theta = np.zeros(110)
    X = np.zeros((19, 10))
    y = np.array([0] * 10 + list(range(1, 10)))
    adjusted = public_class_bias_recalibration(theta, X, y)
    assert np.argmax(adjusted[-10:]) == 0


def test_composed_pure_delta_matches_enumeration_and_limits():
    import itertools

    import numpy as np

    from common.accounting.exact_accounting import composed_pure_delta

    eps, k = 0.7, 3
    p1, p0 = np.exp(eps) / (1 + np.exp(eps)), 1 / (1 + np.exp(eps))
    for eg in (0.0, 0.5, 1.2, 2.0, 2.1):
        brute = 0.0
        for outcome in itertools.product((0, 1), repeat=k):
            p = np.prod([p1 if o else p0 for o in outcome])
            q = np.prod([p0 if o else p1 for o in outcome])
            brute += max(p - np.exp(eg) * q, 0.0)
        assert abs(composed_pure_delta(eps, k, eg) - brute) < 1e-12
    assert composed_pure_delta(eps, k, k * eps) == 0.0
    assert abs(composed_pure_delta(eps, 1, 0.3) - (np.exp(eps) - np.exp(0.3)) / (1 + np.exp(eps))) < 1e-12
    # advanced composition (Dwork, Rothblum and Vadhan 2010) is a valid but looser bound
    delta = 1e-5
    adv = eps * np.sqrt(2 * 64 * np.log(1 / delta)) + 64 * eps * (np.exp(eps) - 1)
    assert composed_pure_delta(eps, 64, adv) <= delta
    curve = composed_pure_delta(eps, 64, np.array([1.0, 5.0, 20.0]))
    assert curve.shape == (3,) and np.all(np.diff(curve) < 0)


def test_optimal_pure_budget():
    from common.accounting.exact_accounting import composed_pure_delta, optimal_pure_budget

    assert optimal_pure_budget(10.0, 1, 1e-5) == 10.0
    assert abs(optimal_pure_budget(10.0, 4, 1e-5) / 2.5 - 1) < 1e-4  # few large steps: nothing to gain
    e0 = optimal_pure_budget(10.0, 64, 1e-5)
    assert e0 > 10.0 / 64 * 1.5  # many small steps: about 1.66 x the basic split
    assert composed_pure_delta(e0, 64, 10.0) <= 1e-5
    assert composed_pure_delta(e0 * 1.01, 64, 10.0) > 1e-5
