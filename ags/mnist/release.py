"""The AGS release: smooth sensitivity from a computed A^k grid, then noise.

This module differs from `ss_reference.py` in exactly one place, deliberately, and
the reason is the definition of GS. Everything downstream of the staircase --
beta_for, sigma, release -- is reused from ss_reference unchanged, so the mechanism
constants are still the reference's.

--------------------------------------------------------------------------------
GS is computed, not assumed
--------------------------------------------------------------------------------
GS is the AGT envelope with every record in the bound's scope substitutable, i.e.
k = b for the per-batch IBP recursion. It is measured by running the bound at k = b,
not read off a formula. `ss_reference.gs_window()` computes

    sum_lr * 2 * clip_gamma / batchsize

which is the reference implementation's `true_gs_window`: a closed-form substitution
quantity that assumes the non-substituted gradients of the two trajectories stay
identical. It is not GS and is never used as GS here.

Because A^k is computed at EVERY k in the grid, including k = b, there is nothing
left for a formula to cap: the grid already contains the largest value the local
sensitivity can take.

--------------------------------------------------------------------------------
Why no tail term
--------------------------------------------------------------------------------
The reference staircase ends with

    ss = max(ss, exp(-beta_j * (k_max + 1)) * true_gs)

a tail covering every k beyond the computed grid. That term exists only because the
grid stops short of the point where A^k saturates. Here the grid REACHES the batch
size: AGT caps the per-batch substitution count at b (k_priv = min(k, b)), so

    A^k = A^b = GS      for every k >= b

and the decay e^{-beta_j k} is strictly decreasing, so

    max_{k >= b} e^{-beta_j k} A^k  =  e^{-beta_j b} A^b

which is already one of the grid's own check-points. The tail is therefore subsumed,
not dropped, and `staircase()` asserts that k_max >= batchsize so this can never be
silently violated.

--------------------------------------------------------------------------------
Width, not radius
--------------------------------------------------------------------------------
A^k is taken as the per-coordinate WIDTH |theta_U - theta_L|, matching
ss_reference.local_sens and experiments/common/sensitivity.py. The paper's A^k is a
one-sided radius about the nominal, which is up to 2x smaller. Using the width is
therefore CONSERVATIVE -- sound, never tight-in-the-wrong-direction -- and both are
recorded by the runners so the convention can be changed in one place.
"""
import numpy as np

from ags.mnist import ss_reference as R

# reused verbatim from the reference port
beta_for = R.beta_for
sigma = R.sigma
release = R.release


def check_points(k_grid):
    """Reference L473-478: every computed k, each k+1, and 0.

    A^k is a step function between computed points: for k between grid entries the
    sound value is the next computed k_j >= k. Over such a plateau e^{-beta k} A^{k_j}
    is largest at the plateau's SMALLEST k, which is the previous grid point + 1 --
    hence the k_j + 1 entries. Omitting them can understate the maximum.
    """
    ks = sorted(set(int(k) for k in k_grid))
    cps = {0} | set(ks) | {ks[j] + 1 for j in range(len(ks) - 1)}
    return sorted(c for c in cps if c <= ks[-1])


def staircase(A_by_k, k_grid, beta, n_params, batchsize):
    """Per-coordinate smooth sensitivity SS^beta from a computed A^k grid.

        SS_j = max_{k in check_points} e^{-(beta/P) k} * A^k_j

    beta_js = beta / n_params (reference L437): the decay budget is SPLIT across
    coordinates so that sum_j beta_j = beta. Using the undivided beta makes the decay
    n_params times too fast and the noise far too small.

    A_by_k maps k -> per-coordinate A^k (width). Requires max(k_grid) >= batchsize so
    that the tail beyond the grid is subsumed; see the module docstring.
    """
    ks = sorted(set(int(k) for k in k_grid))
    if ks[-1] < batchsize:
        raise ValueError(
            f"k grid stops at {ks[-1]} < batchsize {batchsize}, so A^k has not "
            "saturated and a tail term is still required. Either extend the grid to "
            "k = b or add the tail explicitly -- do not drop it."
        )
    beta_js = beta / float(n_params)

    def A_at(kv):
        for k in ks:
            if k >= kv:
                return A_by_k[k]
        raise AssertionError("unreachable: check_points are clipped to max(ks)")

    ss = np.zeros_like(np.asarray(A_by_k[ks[0]], float))
    for cp in check_points(ks):
        ss = np.maximum(ss, np.exp(-beta_js * cp) * A_at(cp))
    return ss


def ags_release(theta_n, A_by_k, k_grid, *, eps, n_params, batchsize, mode="pure",
                gamma_cauchy=2, delta=None, seeds=41, evaluate=None):
    """One AGS release at a given epsilon; returns (median metric, ss, sigma).

    pure   : beta = eps / (2(gamma_c+1)),  sigma_j = (2 gamma_c/eps) P ss_j, Cauchy
    approx : beta = min(1, eps/(1+2e ln(P/delta))), sigma_j = (2/eps) P ss_j, Laplace
    """
    if mode == "pure":
        beta = beta_for(eps, gamma_cauchy=gamma_cauchy)
    else:
        beta = beta_for(eps, mode="approx", n_params=n_params, delta=delta)
    ss = staircase(A_by_k, k_grid, beta, n_params, batchsize)
    sg = sigma(ss, eps, n_params, gamma_cauchy=gamma_cauchy,
               noise_mode="isotropic", mode=mode)
    if evaluate is None:
        return None, ss, sg
    vals = [evaluate(release(theta_n, sg, np.random.RandomState(s), mode=mode))
            for s in range(seeds)]
    return float(np.median(vals)), ss, sg


# --------------------------------------------------------------------------------
# The closed-form substitution bound, and the condition under which it may be used
# --------------------------------------------------------------------------------
def smoothness_constant(X=None, n_classes=None, max_x_norm=None, loss="cross_entropy"):
    """Lipschitz constant L of the BATCH-MEAN loss gradient, for the lr <= 2/L check.

    Multinomial logistic on a linear head. The Hessian of the batch mean wrt vec(W) is

        (1/b) sum_i (diag(p_i) - p_i p_i^T) kron x_i x_i^T

    and ||diag(p) - p p^T|| <= 1/2, so

        L <= 0.5 * lambda_max( X^T X / b )

    NOT 0.5 * max_i ||x_i||^2. That per-sample form bounds the Hessian of a SINGLE
    sample's loss and is wildly pessimistic for the mean: on whitened PCA features
    lambda_max(X^T X / b) = 1 exactly, while max||x||^2 is ~155. Using the per-sample
    form here declares sound settings unsound -- it wrongly rejected lr = 15 on
    unit-norm features, whose actual limit is 664.

    Pass X (the training features) for the correct constant. max_x_norm is retained
    only for the loose per-sample fallback and warns when used.
    """
    if loss != "cross_entropy":
        raise NotImplementedError(
            f"no smoothness constant established for '{loss}'; the closed-form "
            "substitution bound may not be applied to it."
        )
    if X is not None:
        X = np.asarray(X, float)
        return 0.5 * float(np.linalg.eigvalsh(X.T @ X / len(X)).max())
    if max_x_norm is None:
        raise ValueError("pass X (preferred) or max_x_norm")
    print("WARNING: using the per-sample bound 0.5*max||x||^2, which is far looser "
          "than 0.5*lambda_max(X^T X/b) for the batch mean. Pass X instead.")
    return 0.5 * float(max_x_norm) ** 2


def closed_form_substitution(sum_lr_window, clip_gamma, batchsize, n_params):
    """ss_reference.gs_window: sum_lr * 2 * clip_gamma / batchsize, per coordinate.

    NOT global sensitivity. This is the movement of theta when ONE record is
    substituted AND the two trajectories' remaining gradients are assumed identical
    -- i.e. assuming no feedback through the parameter box. It is a valid bound on
    the local sensitivity only when the update map is non-expansive; see
    `local_sens_capped`.
    """
    return R.gs_window(sum_lr_window, clip_gamma, batchsize, n_params)


def local_sens_capped(A_by_k, k_grid, closed_form, *, lr, L, strict=True):
    """min(computed A^k, closed_form * k), the reference's local_sens (L400-402).

    Two valid bounds on the local sensitivity at distance k, intersected:

      A^k          computed, assumes nothing beyond clipping, includes the feedback
                   term, and is therefore larger
      cf * k       closed form, assumes the non-substituted gradients of the two
                   trajectories coincide. That holds when the update map
                   theta -> theta - lr grad f(theta) is NON-EXPANSIVE, which for a
                   convex L-smooth objective requires lr <= 2/L.

    The intersection is sound ONLY if that condition holds, so it is CHECKED rather
    than assumed. On this model L <= 1/2 (see smoothness_constant), giving lr <= 4.
    A run at lr = 15 is 3.75x over the limit and the cap would be invalid there --
    which is exactly the regime where the uncapped envelope is 200x larger, so the
    difference is not academic.

    Pass strict=False only to measure what the cap WOULD give; never to release.
    """
    limit = 2.0 / L
    if lr > limit:
        msg = (f"lr = {lr} exceeds 2/L = {limit:g}, so the update map is not "
               f"certified non-expansive and the closed-form substitution bound is "
               f"NOT a valid local-sensitivity bound at this setting.")
        if strict:
            raise ValueError(msg + " Pass strict=False to measure it anyway.")
        print("WARNING (unsound, measurement only): " + msg)
    cf = np.asarray(closed_form, float)
    return {int(k): np.minimum(np.asarray(A_by_k[k], float), cf * int(k))
            for k in k_grid}
