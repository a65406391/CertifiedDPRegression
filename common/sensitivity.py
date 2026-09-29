"""
Sensitivity evaluation utilities (V_k, smooth sensitivity, staircase).

Consolidates the identical sensitivity computation repeated in:
- run_lin_ablation.py (V_k_mat, S_vals, avg_V_k)
- run_ca_ablation.py (V_all, S_list, evaluate_staircase)
- run_syn_ablation.py (v_k_sums, S_list)
- run_syn_context_ablation.py (v_k_vals, S_list, evaluate_staircase)
- run_lin_dim_ablation.py (V_k_mat, S_vals)
"""

import numpy as np
import torch


def compute_vk_matrix(models_dict, X_eval_t, k_values, y_pred_nom=None, y_min=None, y_max=None, gs_cap=None):
    """Compute V_k instability matrix across eval points and k values.

    V_k(x) = max(|y_l(x) - y_nom(x)|, |y_u(x) - y_nom(x)|)

    Args:
        models_dict: Dict mapping k -> IntervalBoundedModel.
        X_eval_t: Evaluation points tensor (N, D).
        k_values: List of k values (must include 0).
        y_pred_nom: Nominal predictions (N,). If None, computed from models_dict[0].
        y_min: Optional lower clamp for predictions.
        y_max: Optional upper clamp for predictions.
        gs_cap: Optional global sensitivity cap for V_k values.

    Returns:
        V_k matrix of shape (N_eval, len(k_values)).
    """
    n_eval = X_eval_t.size(0) if isinstance(X_eval_t, torch.Tensor) else len(X_eval_t)
    n_k = len(k_values)

    # Get nominal predictions if not provided
    if y_pred_nom is None:
        nom_model = models_dict[0]
        with torch.no_grad():
            y_pred_nom = nom_model.forward(X_eval_t).squeeze().numpy()
        if y_min is not None and y_max is not None:
            y_pred_nom = np.clip(y_pred_nom, y_min, y_max)

    V_k_mat = np.zeros((n_eval, n_k))

    with torch.no_grad():
        for j, k in enumerate(k_values):
            if k == 0:
                continue
            if models_dict.get(k) is None:
                continue

            mk = models_dict[k]
            yl, yu = mk.bound_forward(X_eval_t, X_eval_t, retain_intermediate=False)
            yl = yl.squeeze().numpy()
            yu = yu.squeeze().numpy()

            if y_min is not None and y_max is not None:
                yl = np.clip(yl, y_min, y_max)
                yu = np.clip(yu, y_min, y_max)

            vk = np.abs(yu - yl)

            if gs_cap is not None:
                vk = np.minimum(vk, gs_cap)

            V_k_mat[:, j] = vk

    return V_k_mat


def compute_vk_matrix_per_model_nominal(models_dict, X_eval_t, k_values, y_min=None, y_max=None):
    """Compute V_k using each model's own nominal prediction as reference.

    Used by run_syn_ablation.py and run_syn_context_ablation.py where
    the nominal prediction is taken from the k-model itself (not k=0).

    V_k(x) = max(|y_l_k(x) - y_nom_k(x)|, |y_u_k(x) - y_nom_k(x)|)
    """
    n_eval = X_eval_t.size(0)
    n_k = len(k_values)
    V_k_mat = np.zeros((n_eval, n_k))

    with torch.no_grad():
        for j, k in enumerate(k_values):
            if k == 0:
                continue
            model_k = models_dict[k]

            for idx in range(n_eval):
                x_in = X_eval_t[idx : idx + 1]

                y_l, y_u = model_k.bound_forward(x_in, x_in)
                y_l_val = y_l.item()
                y_u_val = y_u.item()

                y_n = model_k.forward(x_in).item()

                if y_min is not None and y_max is not None:
                    y_l_val = np.clip(y_l_val, y_min, y_max)
                    y_u_val = np.clip(y_u_val, y_min, y_max)
                    y_n = np.clip(y_n, y_min, y_max)

                v = abs(y_u_val - y_l_val)
                V_k_mat[idx, j] = v

    return V_k_mat


def compute_smooth_sensitivity(V_k_mat, k_values, beta=1.0 / 6.0, gs_val=10.0):
    """Compute smooth sensitivity S(x) for each eval point.

    S(x) = max_k{ V_k(x) * exp(-beta * k) }  (plus tail term)

    Args:
        V_k_mat: Matrix of shape (N_eval, len(k_values)).
        k_values: List of k values.
        beta: Decay parameter.
        gs_val: Global sensitivity value for the tail term.

    Returns:
        S_vals: Array of smooth sensitivity values, shape (N_eval,).
    """
    max_k = max(k_values)
    tail_S = gs_val * np.exp(-beta * (max_k + 1))

    n_eval = V_k_mat.shape[0]
    S_vals = np.zeros(n_eval)

    for idx in range(n_eval):
        ss = []
        for j, k in enumerate(k_values):
            if k == 0:
                continue
            ss.append(V_k_mat[idx, j] * np.exp(-beta * k))
        ss.append(tail_S)
        S_vals[idx] = max(ss)

    return S_vals


def evaluate_staircase_single(models_dict, x_eval_t, k_values_sorted, y_min=None, y_max=None):
    """Compute staircase V_k for a single eval point.

    Extracted from run_ca_ablation.py:evaluate_staircase().

    Args:
        models_dict: Dict mapping k -> IntervalBoundedModel.
        x_eval_t: Single evaluation point tensor (1, D).
        k_values_sorted: Sorted list of k values.
        y_min, y_max: Optional output clamping bounds.

    Returns:
        List of (k, v_k) tuples.
    """
    nom_model = models_dict[0]
    with torch.no_grad():
        y_nom = nom_model.forward(x_eval_t).item()
    if y_min is not None and y_max is not None:
        y_nom = np.clip(y_nom, y_min, y_max)

    results = [(0, 0.0)]

    for k in k_values_sorted:
        if k == 0:
            continue
        model = models_dict[k]
        with torch.no_grad():
            y_l, y_u = model.bound_forward(x_eval_t, x_eval_t)
        y_l = y_l.item()
        y_u = y_u.item()

        if y_min is not None and y_max is not None:
            y_l = np.clip(y_l, y_min, y_max)
            y_u = np.clip(y_u, y_min, y_max)

        v_k = abs(y_u - y_l)
        results.append((k, v_k))

    return results


def compute_sensitivity_full_pipeline(
    models_dict,
    X_eval_t,
    k_values,
    beta=1.0 / 6.0,
    gs_val=10.0,
    y_min=None,
    y_max=None,
    use_per_model_nominal=False,
):
    """Full sensitivity pipeline: V_k matrix -> smooth sensitivity -> average.

    Convenience function combining compute_vk_matrix + compute_smooth_sensitivity.

    Returns:
        (V_k_mat, S_vals, avg_S, avg_V_k)
    """
    if use_per_model_nominal:
        V_k_mat = compute_vk_matrix_per_model_nominal(
            models_dict, X_eval_t, k_values, y_min=y_min, y_max=y_max
        )
    else:
        V_k_mat = compute_vk_matrix(models_dict, X_eval_t, k_values, y_min=y_min, y_max=y_max, gs_cap=gs_val)

    S_vals = compute_smooth_sensitivity(V_k_mat, k_values, beta=beta, gs_val=gs_val)
    avg_S = np.mean(S_vals)
    avg_V_k = np.mean(V_k_mat, axis=0)

    return V_k_mat, S_vals, avg_S, avg_V_k


# =============================================================================
# Cross-radius certificate (the paper's Thm 4.2 convention). Added alongside the
# width/radius routines above, which are kept as diagnostics only.
#
#   d(x, k) = max( y_U^k - y_L^{k+1},  y_U^{k+1} - y_L^k ),   k = 0 is the nominal point,
#   missing radii are back-filled by the next available (larger) envelope, and the
#   largest available radius is crossed with the data-independent ceiling box when one
#   is given (otherwise with itself: T_{m+1} := T_m).
# =============================================================================


def cross_radius_certificate(y_nom, yl_by_k, yu_by_k, ceiling=None):
    """Per-point sound-bounding function d(x, k) for every k in 0..max(k).

    Args:
        y_nom: (N,) nominal predictions (the k = 0 envelope).
        yl_by_k, yu_by_k: dict k -> (N,) lower / upper output bounds for k >= 1 (any subset of
            radii; gaps are back-filled by the next larger available envelope).
        ceiling: optional (ceil_l, ceil_u), each (N,), the data-independent box (clipping
            ceiling propagated through the model) used as the k_max + 1 envelope.

    Returns:
        A: (kmax + 1, N) array with A[k] = d(x, k) for k = 0..kmax, where kmax = max(yl_by_k).
        tail_width: (N,) width of the ceiling box, or None if no ceiling was given.
    """
    y_nom = np.asarray(y_nom, dtype=float)
    ks = sorted(int(k) for k in yl_by_k)
    assert ks and ks[0] >= 1 and set(ks) == set(yu_by_k), "yl_by_k/yu_by_k must share radii >= 1"
    kmax = ks[-1]
    L = {0: y_nom}
    U = {0: y_nom}
    nxt = None
    for k in range(kmax, 0, -1):  # back-fill gaps with the next larger available envelope
        if k in yl_by_k:
            nxt = (np.asarray(yl_by_k[k], float), np.asarray(yu_by_k[k], float))
        L[k], U[k] = nxt
    if ceiling is not None:
        cl, cu = (np.asarray(ceiling[0], float), np.asarray(ceiling[1], float))
        assert np.all(cl <= L[kmax] + 1e-12) and np.all(cu >= U[kmax] - 1e-12), "ceiling must contain T_kmax"
    A = np.zeros((kmax + 1, len(y_nom)))
    for k in range(kmax + 1):
        nl, nu = (L[k + 1], U[k + 1]) if k < kmax else ((cl, cu) if ceiling is not None else (L[k], U[k]))
        A[k] = np.maximum(U[k] - nl, nu - L[k])
    return A, (cu - cl if ceiling is not None else None)


def smooth_sensitivity_cross_radius(A, beta, tail_width=None):
    """max_k e^{-beta k} A[k] per point, plus the ceiling tail e^{-beta (kmax+1)} * tail_width.

    ``beta`` must already respect the mechanism's constraint (see common.privacy.beta_max).
    """
    A = np.asarray(A, float)
    ks = np.arange(A.shape[0])[:, None]
    ss = (np.exp(-beta * ks) * A).max(axis=0)
    if tail_width is not None:
        ss = np.maximum(ss, np.exp(-beta * A.shape[0]) * np.asarray(tail_width, float))
    return ss
