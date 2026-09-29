"""Tests pinning ss_reference to experiments/common/ags.py.

History this exists to prevent: the blobs path needed the reference's inference and
it was re-derived rather than ported, drifting in six ways at once and under-noising
by ~71x. Abstract property tests did not catch that. Tests that compare against the
reference's own logic do.

`test_matches_reference_transcription` is the load-bearing one: it re-implements the
reference block inline, straight from experiments/common/ags.py L434-539, and asserts
bit-equality. If ss_reference and that transcription ever disagree, one of them has
drifted from the reference and BOTH should be re-read against it.
"""
import numpy as np
import pytest

from ags.blobs import ss_reference as R

BETAS = [R.beta_for(e) for e in (0.5, 1, 2, 5, 10, 20, 100)]
GRIDS = [[1, 2, 4, 8], [1, 2, 3, 4, 8], [1, 2, 4, 8, 16, 32], [2, 5, 9]]


def _mk(grid, n_params=9, gs_val=0.5, slope=0.12, seed=0):
    rng = np.random.RandomState(seed)
    gs = np.full(n_params, gs_val)
    widths = {k: np.abs(slope * k * (1.0 + 0.3 * rng.randn(n_params))) for k in grid}
    ls = {k: np.minimum(widths[k], gs * k) for k in grid}
    return ls, gs


def _reference_transcription(ls_by_k, k_subset, true_gs, beta, n_params):
    """experiments/common/ags.py L434-539, transcribed independently of ss_reference."""
    beta_js = beta / n_params                                          # L437
    check_points = set()                                               # L473-478
    check_points.add(0)
    for kv in k_subset:
        check_points.add(kv)
    for j in range(len(k_subset) - 1):
        check_points.add(k_subset[j] + 1)
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


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("beta", BETAS)
def test_matches_reference_transcription(grid, beta):
    ls, gs = _mk(grid)
    got = R.smooth_sensitivity(ls, grid, gs, beta, len(gs))
    want = _reference_transcription(ls, sorted(grid), gs, beta, len(gs))
    assert np.allclose(got, want, rtol=0, atol=1e-15)


@pytest.mark.parametrize("beta", BETAS)
def test_beta_is_divided_by_n_params(beta):
    """The deviation that under-noised by 71x. beta_j must be beta / d, not beta."""
    grid = [1, 2, 4, 8]
    ls, gs = _mk(grid, gs_val=1e9)          # cap inert, so decay alone is visible
    d = len(gs)
    got = R.smooth_sensitivity(ls, grid, gs, beta, d)
    assert np.allclose(got, np.maximum.reduce(
        [np.exp(-(beta / d) * cp) * ls[min(k for k in grid if k >= cp)]
         for cp in (1, 2, 3, 4, 5, 8)] +
        [np.exp(-(beta / d) * 9) * gs]), rtol=0, atol=1e-12)
    undivided = np.exp(-beta * 1) * ls[1]
    assert got.mean() > undivided.mean()    # dividing by d gives MORE noise, never less


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("beta", BETAS)
def test_k0_checkpoint_floors_ss(grid, beta):
    """SS >= min(A^{k_min}, GS) with no decay -- the k=0 term. Was missing entirely."""
    ls, gs = _mk(grid)
    got = R.smooth_sensitivity(ls, grid, gs, beta, len(gs))
    assert np.all(got >= np.minimum(ls[sorted(grid)[0]], gs) - 1e-15)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("beta", BETAS)
def test_never_exceeds_gs(grid, beta):
    ls, gs = _mk(grid)
    assert np.all(R.smooth_sensitivity(ls, grid, gs, beta, len(gs)) <= gs + 1e-15)


def test_local_sens_uses_width_not_radius():
    """The reference takes |theta_U - theta_L|; the radius about the nominal is a
    different quantity and is roughly half the size."""
    l, n, u = np.array([-1.0, -2.0]), np.array([0.0, 0.5]), np.array([1.0, 2.0])
    gs = np.full(2, 1e9)
    assert np.allclose(R.local_sens(l, u, gs, 1), np.abs(u - l))
    radius = np.maximum(u - n, n - l)
    assert not np.allclose(R.local_sens(l, u, gs, 1), radius)


def test_local_sens_capped_at_gs_times_k():
    gs = np.full(3, 0.5)
    l, u = np.zeros(3), np.full(3, 100.0)
    assert np.allclose(R.local_sens(l, u, gs, 4), gs * 4)


@pytest.mark.parametrize("eps", [0.5, 1, 5, 20, 100])
def test_pure_and_approx_constants(eps):
    """pure: beta = eps/6, sigma = (2*gamma_c/eps)*s_j.  approx: L320 / L537."""
    d, delta = 9, 1e-5
    assert R.beta_for(eps) == pytest.approx(eps / 6.0 - 1e-7, rel=0, abs=1e-15)
    assert R.beta_for(eps, mode="approx", n_params=d, delta=delta) == pytest.approx(
        min(1.0, eps / (1.0 + 2 * np.e * np.log(d / delta))), rel=0, abs=1e-15)
    ss = np.full(d, 0.3)
    assert np.allclose(R.sigma(ss, eps, d), (4.0 / eps) * ss * d)
    assert np.allclose(R.sigma(ss, eps, d, mode="approx"), (2.0 / eps) * ss * d)


def test_gs_window_l2_reduces_to_gs_window_as_lambda_goes_to_zero():
    lr, n_s, gamma, b, d = 0.5, 200, 0.05, 20, 9
    base = R.gs_window(lr * n_s, gamma, b, d)
    assert np.allclose(R.gs_window_l2(lr, n_s, gamma, b, 0.0, d), base)
    assert np.allclose(R.gs_window_l2(lr, n_s, gamma, b, 1e-9, d), base, rtol=1e-5)
    # and saturates rather than growing with n_s
    long = R.gs_window_l2(lr, 10_000, gamma, b, 0.1, d)
    assert long[0] == pytest.approx(2 * gamma / (0.1 * b), rel=1e-9)


def test_release_noise_families():
    rng = np.random.RandomState(0)
    th, sig = np.zeros(5000), np.full(5000, 1.0)
    cauchy = R.release(th, sig, np.random.RandomState(1))
    laplace = R.release(th, sig, np.random.RandomState(1), mode="approx")
    # Cauchy is heavy-tailed: its sample max should dwarf Laplace's
    assert np.abs(cauchy).max() > 10 * np.abs(laplace).max()
    # numpy's second arg is the Laplace scale b, matching torch.distributions.Laplace
    # in the reference (L549); std of Laplace(0, b) is b*sqrt(2).
    assert laplace.std() == pytest.approx(np.sqrt(2), rel=0.1)
