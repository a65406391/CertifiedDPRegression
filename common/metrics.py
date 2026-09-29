"""Error summaries for a privatised release, in one place.

Three copies of this existed and they disagreed on how to combine trials: two took the median over
noise draws, the third the mean. Under Cauchy noise the mean is dominated by the occasional enormous
draw, so the two were not comparable. `aggregate` now makes the choice explicit at every call site.

    mse      mean over points of the squared error. Has no finite expectation under Cauchy noise,
             so its value is a tail artefact; kept because it is the published quantity in the
             approximate-DP panels, where the noise is light-tailed.
    med_se   median over points of the squared error. Finite for every mechanism.
    q90_abs  90th percentile over points of |error|. The object of Theorems 4.3 and 4.4.
    mae      mean over points of |error| of the released prediction clipped to the public output
             range. Clipping is post-processing, so it costs nothing, it is what a deployed
             predictor does, and it makes the mean finite under Cauchy noise. The headline
             pure-DP metric.
"""

from __future__ import annotations

import numpy as np

KEYS = ("mse", "mse_p25", "mse_p75", "med_se", "q90_abs", "mae")


def release_metrics(
    resid: np.ndarray,
    draw,
    rng,
    *,
    n_trials: int,
    clip: tuple[np.ndarray, np.ndarray] | None = None,
    aggregate: str = "median",
) -> dict[str, float]:
    """Summarise the error of a noisy release over `n_trials` draws.

    Args:
        resid: (n,) nominal prediction minus truth, before noise.
        draw: callable taking `rng` and returning one (n,) noise vector.
        rng: numpy Generator, seeded by the caller so the result is reproducible.
        n_trials: number of noise draws.
        clip: (lo - y, hi - y) per point; the released prediction is clipped to the public output
            range before `mae`. None disables clipping.
        aggregate: "median" or "mean", how mse, med_se and q90_abs combine across trials. `mae`
            always uses the mean, and the interquartile band of mse always uses quantiles.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError("aggregate must be 'median' or 'mean'")
    combine = np.median if aggregate == "median" else np.mean

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
        "mse": float(combine(mse)),
        "mse_p25": float(np.percentile(mse, 25)),
        "mse_p75": float(np.percentile(mse, 75)),
        "med_se": float(combine(med)),
        "q90_abs": float(combine(q90)),
        "mae": float(np.mean(mae)),
    }
