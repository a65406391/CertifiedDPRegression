"""The whole-vector release on synthetic nested envelopes."""

import numpy as np

from common import vector_release as V

P = 6
RNG = np.random.RandomState(1)
THETA = RNG.normal(size=P)


def boxes(k_max=8, computed=(1, 2, 4, 8)):
    out = {}
    for k in computed:
        w = 0.05 * k * (1 + RNG.uniform(0, 0.5, size=P))
        out[k] = (THETA - 0.4 * w, THETA + 0.6 * w)
    return out


def test_grids_are_dense_and_saturate():
    b = boxes()
    A, W = V.cross_radius_grid(THETA, b, 8), V.width_grid(b, 8)
    assert A.shape == W.shape == (9, P)
    np.testing.assert_allclose(A[8], W[8])  # T_9 := T_8, so the last cross-radius is the last width
    np.testing.assert_allclose(W[0], W[1])
    assert np.all(np.diff(W, axis=0) >= -1e-12)  # nested


def test_cross_radius_sandwich():
    b = boxes()
    A, W = V.cross_radius_grid(THETA, b, 8), V.width_grid(b, 8)
    for k in range(1, 8):
        assert np.all(A[k] >= W[k] - 1e-12) and np.all(A[k] <= W[k + 1] + 1e-12)


def test_l1_never_exceeds_the_sum_of_coordinates():
    A = V.cross_radius_grid(THETA, boxes(), 8)
    for beta in (0.01, 0.3, 2.0):
        ss_l1 = V.smooth_sensitivity(A.sum(axis=1), beta)
        assert ss_l1 <= V.smooth_sensitivity(A, beta).sum() + 1e-12


def test_allocations_share_one_budget():
    A = V.cross_radius_grid(THETA, boxes(), 8)
    ss = V.smooth_sensitivity(A, 0.1)
    _, s_iso = V.noise_scales(A, 0.1, 1.0, mode="pure", allocation="isotropic")
    _, s_ani = V.noise_scales(A, 0.1, 1.0, mode="pure", allocation="anisotropic")
    _, s_l1 = V.noise_scales(A, 0.1, 1.0, mode="pure", allocation="l1")
    # sliding condition sum_j SS_j / S_j <= 1 for every allocation
    for s in (s_iso, s_ani, s_l1):
        assert (ss / s).sum() <= 1.0 + 1e-9
    # anisotropic is the least total noise of the three per-coordinate splits
    assert s_ani.sum() <= s_iso.sum() + 1e-12


def test_constants_follow_the_reference():
    A = V.cross_radius_grid(THETA, boxes(), 8)
    sig_pure, S = V.noise_scales(A, 0.1, 2.0, mode="pure", allocation="isotropic")
    sig_apx, _ = V.noise_scales(A, 0.1, 2.0, mode="approx", allocation="isotropic")
    np.testing.assert_allclose(sig_pure, 2.0 * 2 / 2.0 * S)
    np.testing.assert_allclose(sig_apx, 2.0 / 2.0 * S)


def test_beta_rules():
    assert V.effective_beta(1.0, 650, "aggregate") == 1.0 / 650
    assert V.effective_beta(1.0, 650, "dimension_free") == 1.0


# ---- the proven rules (Notes/AGS_RELEASE_ADMISSIBILITY.md)


def test_laplace_dilation_beta_matches_monte_carlo():
    """Common scale: the loss D = p lam + (e^-lam - 1) G at the admissible lam exceeds the budget with
    probability about delta (Gamma tail), and clearly more at 1.5 lam."""
    from scipy.stats import gamma

    p, budget, delta = 12, 0.5, 1e-3
    lam = V.laplace_dilation_beta(p, budget, delta, split="common")
    rng = np.random.RandomState(0)
    G = gamma.rvs(p, size=400_000, random_state=rng)
    for sign in (+1, -1):
        D = p * sign * lam + (np.exp(-sign * lam) - 1.0) * G
        assert np.mean(D > budget) <= 1.5 * delta + 5e-4
    D = -p * 1.5 * lam + (np.exp(1.5 * lam) - 1.0) * G
    assert np.mean(D > budget) > delta
    # the sign-pattern-safe bound is more conservative than the exact common-scale value
    assert V.laplace_dilation_beta(p, budget, delta, split="coordinate") < lam


def test_tight_scales_spend_the_budget_exactly():
    A = V.cross_radius_grid(THETA, boxes(), 8)
    eps = 2.0
    sigma, S, info = V.tight_scales(A, eps, mode="pure", allocation="l1")
    assert abs(P * info["beta_j"] + info["alpha"] - eps) < 1e-12  # p beta + alpha = eps
    np.testing.assert_allclose(sigma, S / info["alpha"])
    np.testing.assert_allclose(S, V.smooth_sensitivity(A.sum(axis=1), info["beta_j"]))
    sig_a, S_a, _ = V.tight_scales(A, eps, mode="pure", allocation="anisotropic")
    ss = V.smooth_sensitivity(A, info["beta_j"])
    assert (ss / S_a).sum() <= 1.0 + 1e-9  # sliding: sum_j |Delta_j| / (a_j S_j) <= sum_j 1/a_j = 1
    sig_l, _, info_l = V.tight_scales(A, eps, mode="approx", allocation="l1", delta=1e-5)
    assert info_l["beta_j"] == V.laplace_dilation_beta(P, eps / 2, 1e-5, split="common")


def test_corollary_scales():
    A = V.cross_radius_grid(THETA, boxes(), 8)
    sigma, S, info = V.corollary_scales(A, 3.0)
    assert abs(info["beta_j"] - 3.0 / 6.0 / P) < 1e-12
    np.testing.assert_allclose(sigma, 6.0 * S / 3.0)
    _, _, info_as_written = V.corollary_scales(A, 3.0, split_beta=False)
    assert abs(info_as_written["beta_j"] - 0.5) < 1e-12


def test_tight_variants_schema():
    b = boxes()
    score = lambda th: {"select": float(np.abs(th - THETA).mean()), "test": 0.0}
    draw = lambda th, sigma, rng, mode: (
        th + (rng.standard_cauchy(P) if mode == "pure" else rng.laplace(size=P)) * sigma
    )
    out = V.tight_variants(THETA, b, 8, [1.0, 10.0], delta=1e-5, seeds=3, score=score, draw=draw)
    assert set(out) == {"pure", "approx"} and set(out["pure"]) == {"1.0", "10.0"}
    assert set(out["pure"]["1.0"]) == {"tight_l1", "tight_aniso", "corollary_p"}
    assert set(out["approx"]["1.0"]) == {
        "tight_l1",
        "tight_aniso",
        "corollary_p",
    }  # a pure release is also (eps, delta)-DP
    assert set(out["approx"]["1.0"]["tight_l1"]["cross_radius"]) == {"laplace", "cauchy"}
    cell = out["pure"]["10.0"]["tight_l1"]["cross_radius"]["cauchy"]
    assert cell["beta_j"] == 5.0 / P and cell["alpha"] == 5.0
