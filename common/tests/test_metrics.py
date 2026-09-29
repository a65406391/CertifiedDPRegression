"""Pin common.metrics to the implementations it replaces, including their disagreement."""

import numpy as np

from common.metrics import release_metrics


def _original_median(resid, draw, rng, n_trials, clip):
    """agtr/linear/release.py and agtr/california/release.py: median over trials."""
    mse, med, q90, mae = [], [], [], []
    for _ in range(n_trials):
        e = resid + draw(rng)
        se = e**2
        mse.append(se.mean())
        med.append(np.median(se))
        q90.append(np.percentile(np.abs(e), 90))
        ec = np.clip(e, clip[0], clip[1]) if clip is not None else e
        mae.append(np.abs(ec).mean())
    return {
        "mse": float(np.median(mse)),
        "mse_p25": float(np.percentile(mse, 25)),
        "mse_p75": float(np.percentile(mse, 75)),
        "med_se": float(np.median(med)),
        "q90_abs": float(np.median(q90)),
        "mae": float(np.mean(mae)),
    }


def _original_mean(resid, draw, rng, n_trials, clip):
    """agtr/linear/pate/release.py: mean over trials."""
    q90, med, mse, mae = [], [], [], []
    for _ in range(n_trials):
        e = resid + draw(rng)
        q90.append(np.quantile(np.abs(e), 0.9))
        med.append(np.median(e**2))
        mse.append(np.mean(e**2))
        ec = np.clip(e, clip[0], clip[1]) if clip is not None else e
        mae.append(np.abs(ec).mean())
    return {
        "q90_abs": float(np.mean(q90)),
        "med_se": float(np.mean(med)),
        "mse": float(np.mean(mse)),
        "mae": float(np.mean(mae)),
        "mse_p25": float(np.quantile(mse, 0.25)),
        "mse_p75": float(np.quantile(mse, 0.75)),
    }


def _setup(seed=0, n=40):
    rng = np.random.default_rng(seed)
    resid = rng.normal(0, 0.3, n)
    clip = (np.full(n, -5.0) - resid, np.full(n, 5.0) - resid)
    return resid, clip


def test_matches_the_median_original():
    resid, clip = _setup()

    def draw(r):
        return r.laplace(0, 0.5, len(resid))

    got = release_metrics(resid, draw, np.random.default_rng(7), n_trials=50, clip=clip)
    want = _original_median(resid, draw, np.random.default_rng(7), 50, clip)
    assert got == want


def test_matches_the_mean_original():
    """np.quantile and np.percentile agree, so the two originals differ only in the aggregation."""
    resid, clip = _setup(seed=1)

    def draw(r):
        return r.standard_cauchy(len(resid)) * 0.2

    got = release_metrics(resid, draw, np.random.default_rng(3), n_trials=50, clip=clip, aggregate="mean")
    want = _original_mean(resid, draw, np.random.default_rng(3), 50, clip)
    for k in want:
        assert np.isclose(got[k], want[k], rtol=0, atol=0), k


def test_the_two_aggregations_differ_under_heavy_tails():
    """Why the choice is explicit: under Cauchy noise the mean MSE is a tail artefact."""
    resid, clip = _setup(seed=2)

    def draw(r):
        return r.standard_cauchy(len(resid))

    med = release_metrics(resid, draw, np.random.default_rng(11), n_trials=200, clip=clip)
    mean = release_metrics(resid, draw, np.random.default_rng(11), n_trials=200, clip=clip, aggregate="mean")
    assert mean["mse"] > 10 * med["mse"]
    assert np.isclose(mean["mae"], med["mae"])  # mae always uses the mean
