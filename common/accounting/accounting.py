"""Replace-one privacy accounting and calibration for the AGS releases.

Turns a target budget into a mechanism parameter, and a mechanism parameter back into a budget, for
the coordinate-wise AGS release and for the DP-SGD arms it is compared against.

    amplify_pair            one persistent without-replacement subsample: (eps, delta) -> amplified
    base_budget_for_outer   its inverse, the paper's amplification lemma: what the base mechanism
                            must satisfy so the subsampled one meets the target
    ags_epsilon             compose the coordinates, amplify the vector once, compose the steps
    calibrate_ags           bisection to a target, failing closed when the privacy-loss grid cannot
                            resolve the coordinate budget
    calibrate_dpsgd         the matched baseline, through dp_accounting's PLD accountant

Calibration keeps a small safety margin, so an audited quantity always meets its target rather than
sitting a rounding error the wrong side of it. Amplification applies to the whole parameter vector
once, never per coordinate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from dp_accounting import dp_event
from dp_accounting import pld
from dp_accounting import privacy_accountant
from dp_accounting.pld import common
from dp_accounting.pld import privacy_loss_distribution


@dataclass(frozen=True)
class Calibration:
    mechanism_parameter: float
    epsilon: float
    delta: float
    steps: int
    q: float
    accountant: str
    accounting_runtime_seconds: float


def amplify_pair(epsilon: float, delta: float, q: float) -> tuple[float, float]:
    if q == 1:
        return epsilon, delta
    if epsilon < 40:
        amplified = math.log1p(q * math.expm1(epsilon))
    else:
        amplified = epsilon + math.log(q + (1 - q) * math.exp(-epsilon))
    return amplified, q * delta


def base_budget_for_outer(epsilon: float, delta: float, q: float) -> tuple[float, float]:
    """Inverse of ``amplify_pair``: the (eps_b, delta_b) a base mechanism on one persistent
    without-replacement subsample of fraction q must satisfy so that the subsampled mechanism is
    (epsilon, delta)-DP on the full dataset (paper Lemma "Amplification of AGT mechanisms",
    eq. AmplificationEQ):  eps_b = log(1 + (e^eps - 1)/q),  delta_b = delta / q  (needs delta <= q).
    """
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    if q == 1:
        return epsilon, delta
    if delta > q:
        raise ValueError("delta_b = delta/q exceeds 1; the lemma needs delta <= q")
    if epsilon < 40:
        eps_b = math.log1p(math.expm1(epsilon) / q)
    else:
        eps_b = epsilon - math.log(q) + math.log1p((1 - q) * math.exp(-epsilon))
    return eps_b, delta / q


def generic_pld(epsilon: float, delta: float, interval: float):
    params = common.DifferentialPrivacyParameters(epsilon=epsilon, delta=delta)
    return privacy_loss_distribution.from_privacy_parameters(params, interval)


def ags_epsilon(
    eps_coord: float,
    delta_coord: float,
    *,
    coordinates: int,
    steps: int,
    q: float,
    target_delta: float,
    base_step_delta: float,
    interval: float,
) -> float:
    """Conservative coordinate-compose, vector-amplify-once AGS accountant."""
    coordinate = generic_pld(eps_coord, delta_coord, interval)
    base_vector = coordinate.self_compose(coordinates)
    eps_base = base_vector.get_epsilon_for_delta(base_step_delta)
    eps_amplified, delta_amplified = amplify_pair(eps_base, base_step_delta, q)
    step = generic_pld(eps_amplified, delta_amplified, interval)
    return float(step.self_compose(steps).get_epsilon_for_delta(target_delta))


def calibrate_ags(
    target_epsilon: float,
    target_delta: float,
    *,
    coordinates: int,
    steps: int,
    q: float,
    mechanism: str,
    interval: float,
    delta_fraction: float,
) -> Calibration:
    started = time.perf_counter()
    base_step_delta = target_delta * delta_fraction / (q * steps)

    def evaluate(value: float) -> float:
        delta_coord = 0.0 if mechanism in {"cauchy", "laplace_pure"} else base_step_delta / (2 * coordinates)
        return ags_epsilon(
            value,
            delta_coord,
            coordinates=coordinates,
            steps=steps,
            q=q,
            target_delta=target_delta,
            base_step_delta=base_step_delta,
            interval=interval,
        )

    # A coordinate budget is tiny when tens of thousands of coordinates are
    # composed. Starting at one needlessly constructs an enormous PLD support.
    # Generic PLD discretization is not numerically monotone when epsilon lies
    # below one grid cell. Restrict calibration to the auditable grid-resolved
    # regime; convergence tests compare this result across smaller intervals.
    low_index, high_index = 1, 2
    if evaluate(interval) > target_epsilon:
        raise RuntimeError("PLD interval is too coarse to resolve the coordinate budget")
    evaluations = 1
    while evaluate(high_index * interval) < target_epsilon:
        evaluations += 1
        elapsed = time.perf_counter() - started
        print(
            f"[accounting:ags] evaluations={evaluations} elapsed={elapsed:.1f}s",
            flush=True,
        )
        high_index *= 2
    while high_index - low_index > 1:
        mid_index = (low_index + high_index) // 2
        epsilon_mid = evaluate(mid_index * interval)
        evaluations += 1
        elapsed = time.perf_counter() - started
        print(
            f"[accounting:ags] evaluations={evaluations} elapsed={elapsed:.1f}s "
            f"candidate_epsilon={epsilon_mid:.6g}",
            flush=True,
        )
        if epsilon_mid <= target_epsilon:
            low_index = mid_index
        else:
            high_index = mid_index
    parameter = low_index * interval
    achieved = evaluate(parameter)
    return Calibration(
        parameter,
        achieved,
        target_delta,
        steps,
        q,
        "generic-pld-conservative",
        time.perf_counter() - started,
    )


def dpsgd_epsilon(noise_multiplier: float, *, q: float, steps: int, delta: float, interval: float) -> float:
    accountant = pld.PLDAccountant(
        neighboring_relation=privacy_accountant.NeighboringRelation.REPLACE_ONE,
        value_discretization_interval=interval,
    )
    event = dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier))
    accountant.compose(event, steps)
    return float(accountant.get_epsilon(delta))


def calibrate_dpsgd(
    target_epsilon: float,
    target_delta: float,
    *,
    q: float,
    steps: int,
    interval: float,
    search_iterations: int = 24,
) -> Calibration:
    started = time.perf_counter()
    evaluations = 0

    def evaluate(value: float, expected_total: int | None = None) -> float:
        nonlocal evaluations
        value_epsilon = dpsgd_epsilon(value, q=q, steps=steps, delta=target_delta, interval=interval)
        evaluations += 1
        elapsed = time.perf_counter() - started
        eta_text = "estimating"
        if expected_total is not None:
            eta = elapsed / evaluations * max(0, expected_total - evaluations)
            eta_text = f"{eta:.1f}s"
        print(
            f"[accounting:dpsgd] evaluation={evaluations} "
            f"noise={value:.8g} epsilon={value_epsilon:.6g} "
            f"elapsed={elapsed:.1f}s eta={eta_text}",
            flush=True,
        )
        return value_epsilon

    low, high = 1e-3, 1.0
    while evaluate(high) > target_epsilon:
        high *= 2
    max_binary_evaluations = search_iterations
    for _ in range(max_binary_evaluations):
        mid = (low + high) / 2
        epsilon_mid = evaluate(mid, evaluations + max_binary_evaluations + 1)
        if epsilon_mid > target_epsilon:
            low = mid
        else:
            high = mid
        if (high - low) / high < 1e-5:
            break
    achieved = evaluate(high, evaluations + 1)
    if achieved > target_epsilon:
        raise RuntimeError("DP-SGD calibration did not meet the target epsilon")
    print(
        f"[accounting:dpsgd] final_noise_bracket=[{low:.8g}, {high:.8g}] width={high - low:.8g}",
        flush=True,
    )
    return Calibration(
        high,
        achieved,
        target_delta,
        steps,
        q,
        "dp-accounting-pld-replace-one",
        time.perf_counter() - started,
    )


def dpsgd_generic_epsilon(
    noise_multiplier: float,
    *,
    steps: int,
    target_delta: float,
    interval: float,
    delta_fraction: float,
) -> float:
    """Gaussian DP-SGD reduced to generic DP pairs before composition.

    This deliberately discards the Gaussian privacy-loss shape.  It is the
    accountant-strength control for AGS's generic-PLD path, not the preferred
    Gaussian accountant.  Full participation and replace-one adjacency give a
    normalized shift of two relative to the per-record norm bound.
    """
    delta_step = target_delta * delta_fraction / steps
    eps_step = (2.0 / noise_multiplier) * math.sqrt(2.0 * math.log(1.25 / delta_step))
    step = generic_pld(eps_step, delta_step, interval)
    return float(step.self_compose(steps).get_epsilon_for_delta(target_delta))


def calibrate_dpsgd_generic(
    target_epsilon: float,
    target_delta: float,
    *,
    steps: int,
    interval: float,
    delta_fraction: float = 0.5,
) -> Calibration:
    """Calibrate DP-SGD using exactly the generic-PLD composition class."""
    started = time.perf_counter()
    low, high = 1e-3, 1.0
    while (
        dpsgd_generic_epsilon(
            high,
            steps=steps,
            target_delta=target_delta,
            interval=interval,
            delta_fraction=delta_fraction,
        )
        > target_epsilon
    ):
        high *= 2
    for _ in range(32):
        mid = (low + high) / 2
        value = dpsgd_generic_epsilon(
            mid,
            steps=steps,
            target_delta=target_delta,
            interval=interval,
            delta_fraction=delta_fraction,
        )
        if value > target_epsilon:
            low = mid
        else:
            high = mid
    achieved = dpsgd_generic_epsilon(
        high,
        steps=steps,
        target_delta=target_delta,
        interval=interval,
        delta_fraction=delta_fraction,
    )
    return Calibration(
        high,
        achieved,
        target_delta,
        steps,
        1.0,
        "generic-pld-gaussian-pair-control",
        time.perf_counter() - started,
    )
