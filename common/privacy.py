"""
Privacy trade-off computation (Approx DP and Pure DP noise addition).

Extracted from the repeated trade-off plotting logic in:
- run_ca_ablation.py (lines 417-511)
- plot_lin_ablation.py (lines 223-303)
- plot_ca_baselines.py
"""

import numpy as np


def compute_approx_dp_mse(
    nom_mse, avg_sensitivity, epsilons, delta=1e-5, n_test=None, n_trials=500, seed=None
):
    """Compute expected MSE under approximate DP (Gaussian mechanism).

    MSE = nom_mse + E[noise^2], where noise ~ N(0, sigma^2)
    and sigma = avg_sensitivity * sqrt(2*ln(1/delta)) / epsilon.

    Args:
        nom_mse: Nominal (noise-free) MSE.
        avg_sensitivity: Average smooth sensitivity.
        epsilons: List of privacy budgets.
        delta: Privacy delta parameter.
        n_test: Number of test points (for noise simulation). If None, uses analytic formula.
        n_trials: Number of Monte Carlo trials.
        seed: Random seed.

    Returns:
        (mse_mean, mse_std): Arrays of shape (len(epsilons),).
    """
    if seed is not None:
        np.random.seed(seed)

    factor = np.sqrt(2 * np.log(1 / delta))

    mse_mean = []
    mse_std = []

    for eps in epsilons:
        sigma = avg_sensitivity * factor / eps

        if n_test is not None:
            # Monte Carlo simulation
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.normal(0, sigma, n_test)
                mse_trials.append(nom_mse + np.mean(noise**2))
            mse_mean.append(np.mean(mse_trials))
            mse_std.append(np.std(mse_trials))
        else:
            # Analytic: E[noise^2] = sigma^2
            mse_mean.append(nom_mse + sigma**2)
            mse_std.append(0.0)

    return np.array(mse_mean), np.array(mse_std)


def compute_pure_dp_mse(
    nom_mse, avg_sensitivity, epsilons, beta=1.0 / 6.0, n_test=None, n_trials=500, seed=None
):
    """Compute expected MSE under pure DP (Cauchy mechanism).

    Noise ~ Cauchy(0, scale), where scale = avg_sensitivity / (beta * epsilon).
    Reports median MSE (Cauchy has no finite variance).

    Args:
        nom_mse: Nominal (noise-free) MSE.
        avg_sensitivity: Average smooth sensitivity.
        epsilons: List of privacy budgets.
        beta: Smooth sensitivity decay parameter.
        n_test: Number of test points for noise simulation.
        n_trials: Number of Monte Carlo trials.
        seed: Random seed.

    Returns:
        (mse_median, mse_p25, mse_p75): Arrays of shape (len(epsilons),).
    """
    if seed is not None:
        np.random.seed(seed)

    if n_test is None:
        n_test = 1000  # Default for simulation

    mse_median = []
    mse_p25 = []
    mse_p75 = []

    for eps in epsilons:
        scale = avg_sensitivity / (beta * eps)
        mse_trials = []
        for _ in range(n_trials):
            noise = np.random.standard_cauchy(n_test) * scale
            mse_trials.append(nom_mse + np.mean(noise**2))
        mse_median.append(np.median(mse_trials))
        mse_p25.append(np.percentile(mse_trials, 25))
        mse_p75.append(np.percentile(mse_trials, 75))

    return np.array(mse_median), np.array(mse_p25), np.array(mse_p75)


def compute_global_sensitivity_mse(gs_val, epsilons, delta=1e-5, n_test=1000, n_trials=500, mode="approx"):
    """Compute MSE using global sensitivity (for comparison baseline).

    Args:
        gs_val: Global sensitivity value.
        epsilons: List of privacy budgets.
        delta: Privacy delta (approx DP only).
        n_test: Number of test points.
        n_trials: Number of trials.
        mode: 'approx' (Gaussian) or 'pure' (Cauchy).

    Returns:
        Depends on mode:
        - approx: (sigma_squared_values,) list
        - pure: (median, p25, p75) arrays
    """
    if mode == "approx":
        factor = np.sqrt(2 * np.log(1 / delta))
        sigma_sq_vals = []
        for eps in epsilons:
            sigma_sq_vals.append((gs_val * factor / eps) ** 2)
        return np.array(sigma_sq_vals)
    elif mode == "pure":
        mse_median = []
        mse_p25 = []
        mse_p75 = []
        for eps in epsilons:
            scale_gs = gs_val / eps
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.standard_cauchy(n_test) * scale_gs
                mse_trials.append(np.mean(noise**2))
            mse_median.append(np.median(mse_trials))
            mse_p25.append(np.percentile(mse_trials, 25))
            mse_p75.append(np.percentile(mse_trials, 75))
        return np.array(mse_median), np.array(mse_p25), np.array(mse_p75)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Mechanism calibration per the paper (Defs 3.5-3.8, cor:TighterCauchy). Added alongside the
# older MSE-simulation helpers above; those use Gaussian noise on smooth sensitivity and a
# fixed beta, which no theorem in the paper covers, and are kept for legacy plots only.
# =============================================================================
import math as _math


def beta_max(epsilon, delta=None, mechanism="cauchy"):
    """Largest admissible smoothing parameter beta for a given budget.

    cauchy  (pure DP, Def 3.7):    beta <= eps / 6
    laplace (approx DP, Def 3.8):  beta <= eps / (2 ln(2/delta))
    liang   (pure DP, cor:TighterCauchy): any 0 < beta < eps; returns eps / 6 as a default.
    """
    if mechanism == "cauchy" or mechanism == "liang":
        return epsilon / 6.0
    if mechanism == "laplace":
        if delta is None or not 0 < delta < 1:
            raise ValueError("laplace on smooth sensitivity needs delta in (0, 1)")
        return epsilon / (2.0 * _math.log(2.0 / delta))
    raise ValueError(mechanism)


def cauchy_scale(ss_beta, epsilon):
    """Def 3.7: Cauchy(6 SS^beta / eps) is (eps, 0)-DP for beta <= eps/6."""
    return 6.0 * np.asarray(ss_beta, float) / epsilon


def liang_cauchy_scale(ss_beta, epsilon, beta):
    """cor:TighterCauchy (Liang et al. 2024): Cauchy(SS^beta / (eps - beta)), 0 < beta < eps.
    Only valid if that corollary is part of the paper's claims."""
    if not 0 < beta < epsilon:
        raise ValueError("need 0 < beta < epsilon")
    return np.asarray(ss_beta, float) / (epsilon - beta)


def laplace_ss_scale(ss_beta, epsilon):
    """Def 3.8: Laplace(2 SS^beta / eps) is (eps, delta)-DP for beta <= eps / (2 ln(2/delta))."""
    return 2.0 * np.asarray(ss_beta, float) / epsilon


def laplace_gs_scale(global_sensitivity, epsilon):
    """Def 3.5: Laplace(Delta f / eps), pure DP baseline."""
    return global_sensitivity / epsilon


def gaussian_gs_sigma(global_l2_sensitivity, epsilon, delta):
    """Def 3.6: sigma = Delta_2 f sqrt(2 ln(1.25/delta)) / eps, approx DP baseline."""
    return global_l2_sensitivity * _math.sqrt(2.0 * _math.log(1.25 / delta)) / epsilon


def abs_error_quantile(errors, q=0.9):
    """Quantile of |error|: the utility object of Thms 4.3/4.4. Use this for pure-DP (Cauchy)
    panels, whose MSE has no finite expectation."""
    return float(np.percentile(np.abs(np.asarray(errors, float)), 100.0 * q))
