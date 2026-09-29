"""Line-by-line port of the AGS inference procedure in experiments/common/ags.py.

This module exists because re-deriving that procedure produced six separate
deviations from it. Nothing here is re-derived: every function names the reference
lines it ports, keeps the reference's variable names, and performs the operations
in the reference's order. If the reference changes, change this and re-run the
end-to-end differential test in test_ss_reference.py -- never one alone.

Reference: experiments/common/ags.py, train_ags(), lines 335-560.
"""
import numpy as np

DEFAULT_K_SUBSET = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]   # reference L183


def gs_window(sum_lr, clip_gamma, batchsize, n_coords):
    """Reference L335. Per-coordinate GLOBAL sensitivity for ONE substitution.

        true_gs_window = sum_lr_window * 2 * clip_gamma / batchsize

    Substituting one record moves the summed gradient by at most 2*clip_gamma per
    coordinate, hence the batch mean by 2*clip_gamma/batchsize, hence theta by
    sum_lr times that. This is the k=1 quantity, and it caps the staircase at
    EVERY k -- smooth sensitivity is defined over the local sensitivity at a
    distance-k neighbour, which is a one-substitution quantity, not the
    reachable-set radius under k substitutions.
    """
    return np.full(int(n_coords), float(sum_lr) * 2.0 * float(clip_gamma) / float(batchsize))


def gs_window_l2(lr, n_s, clip_gamma, batchsize, l2_reg, n_coords):
    """Per-coordinate one-substitution GS under L2 regularisation (contraction bound).

    The reference's gs_window() assumes theta_{t+1} = theta_t - lr*g_t, giving a
    sensitivity that grows linearly in the number of steps. With an L2 term the update
    is theta_{t+1} = (1 - lr*lam) theta_t - lr*g_t, so a one-record perturbation obeys

        delta_{t+1} = (1 - lr*lam) delta_t - lr (gbar - gbar')  ,  |gbar - gbar'| <= 2*gamma/b

    and summing the geometric series over T steps gives

        GS_lam = (2*gamma / (lam*b)) * (1 - (1 - lr*lam)^T)

    which tends to lr*T*2*gamma/b as lam -> 0 (recovering gs_window) and to the FINITE
    limit 2*gamma/(lam*b) as T -> infinity. This is the contraction bound from
    AGS_Ablation/attic/baselines/ags.py:52; it is an extension of the reference's
    formula to lam > 0, not a deviation from it -- at lam = 0 the two agree exactly.

    Requires 0 < lr*lam < 2 for the recursion to contract.
    """
    if l2_reg <= 0:
        return gs_window(lr * n_s, clip_gamma, batchsize, n_coords)
    rho = 1.0 - lr * l2_reg
    assert abs(rho) < 1.0, f"no contraction: lr*lam = {lr*l2_reg} must lie in (0, 2)"
    val = (2.0 * clip_gamma / (l2_reg * batchsize)) * (1.0 - rho ** n_s)
    return np.full(int(n_coords), float(val))


def local_sens(param_l, param_u, true_gs, k):
    """Reference L400-402:  local_sens[k] = min(|theta_U - theta_L|, GS * k).

    Note the WIDTH of the envelope, not its radius about the nominal.
    """
    width = np.abs(np.asarray(param_u, float).ravel() - np.asarray(param_l, float).ravel())
    return np.minimum(width, true_gs * k)


def beta_for(eps, gamma_cauchy=2, mode="pure", n_params=None, delta=None):
    """Reference L316-320.

        pure  : beta = eps / (2 * (gamma_cauchy + 1)) - 1e-7
        approx: beta = min(1, eps / (1 + 2*e*ln(n_params / delta)))
    """
    if mode == "pure":
        return eps / (2.0 * (gamma_cauchy + 1.0)) - 1e-7
    if n_params is None or delta is None:
        raise ValueError("approx mode needs n_params and delta")
    return min(1.0, eps / (1.0 + 2.0 * np.e * float(np.log(n_params / delta))))


def smooth_sensitivity(ls_by_k, k_subset, true_gs, beta, n_params):
    """Reference L434-527. Per-coordinate SS over the staircase. Returns ss_j.

    beta_js = beta / n_params  (L437) -- the decay budget is SPLIT across
    coordinates, so that sum_j beta_j = beta. Using the undivided beta makes the
    decay n_params times too fast and the noise far too small.
    """
    k_subset = sorted(set(int(k) for k in k_subset))
    beta_js = beta / float(n_params)                                   # L437

    check_points = {0}                                                 # L473-478
    check_points.update(k_subset)
    check_points.update(k_subset[j] + 1 for j in range(len(k_subset) - 1))
    check_points = sorted(check_points)

    def _get_staircase_ls(k_val):                                      # L481-489
        for kv in k_subset:
            if kv >= k_val:
                return ls_by_k[kv]
        return true_gs * k_val

    ss = np.zeros_like(true_gs)                                        # L494
    for cp in check_points:                                            # L499-507
        decay = np.exp(-beta_js * cp)
        capped = np.minimum(_get_staircase_ls(cp), true_gs)
        ss = np.maximum(ss, decay * capped)

    max_k_val = k_subset[-1]                                           # L512-517
    ss = np.maximum(ss, np.exp(-beta_js * (max_k_val + 1)) * true_gs)
    return ss


def sigma(ss, eps, n_params, gamma_cauchy=2, noise_mode="isotropic", mode="pure"):
    """Reference L529-539.

        pure  : sigma_j = (2 * gamma_cauchy / eps) * s_j      (Cauchy)
        approx: sigma_j = (2 / eps) * s_j                     (Laplace)
    """
    if noise_mode == "anisotropic":                                    # L531-532
        a_js = np.sqrt(ss).sum() / np.clip(np.sqrt(ss), 1e-30, None)
    else:                                                              # L534
        a_js = np.full_like(ss, float(n_params))
    s_js = ss * a_js                                                   # L535
    if mode == "approx":
        return (2.0 / eps) * s_js                                      # L537
    return (2.0 * float(gamma_cauchy) / eps) * s_js                    # L539


def release(theta_n, sigma_js, rng, mode="pure"):
    """Reference L546-562: independent noise per coordinate.

    pure -> Cauchy(0, sigma_j);  approx -> Laplace(0, sigma_j).
    """
    sigma_js = np.clip(sigma_js, 1e-8, None)                           # L548
    theta_n = np.asarray(theta_n, float).ravel()
    if mode == "approx":
        return theta_n + rng.laplace(0.0, sigma_js)                    # L549-551
    return theta_n + rng.standard_cauchy(len(sigma_js)) * sigma_js     # L554-556
