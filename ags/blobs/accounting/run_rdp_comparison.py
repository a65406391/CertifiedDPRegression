"""Matched-RDP analysis for the one-release linear-Blobs experiment.

AGS uses the analytic scalar Laplace Renyi integral, maximized numerically over
the complete smooth-Laplace feasible set in both privacy directions.  DP-SGD
uses the exact full-participation Gaussian RDP expression under replace-one
adjacency.  Both arms compose in RDP and convert once to (epsilon, delta).

Coordinate AGS uses analytic mean-shift monotonicity and interval branch-and-bound
for the continuous scale-ratio maximum. The optional joint-l1 panel remains a
numerical diagnostic.
"""

from __future__ import annotations

import argparse
import functools
import math
import time
from pathlib import Path

import numpy as np

from ags.blobs import baselines as B
from ags.blobs.shared import load_blobs
from common.accounting.artifacts import write_immutable_json
from common.accounting.laplace_rdp import (
    certified_smooth_laplace_rdp,
    epsilon_from_rdp,
    worst_case_joint_smooth_laplace_rdp_grid,
    worst_case_smooth_laplace_rdp_grid,
)
from ags.blobs.accounting.run_cross_radius_comparison import (
    BATCH_SIZE,
    D,
    GAMMA,
    LR,
    N,
    PARAMETERS,
    PRIVACY_STEPS,
    SEP,
    STEPS,
    complete_profile,
    cross_radius_profile,
    load_intersected_boxes,
    quantiles,
    smooth_profile,
)


ORDERS = np.r_[np.linspace(1.05, 2.0, 20), np.arange(2.5, 64.5, 0.5)]


def public_bias_recalibration(theta, public_design, public_labels):
    """Optimize only the intercept on labeled public validation data."""
    theta = np.asarray(theta, dtype=float)
    scores = public_design[:, :-1] @ theta[:-1]
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    positive = np.asarray(public_labels)[order] > 0
    # A cut predicts negative below it and positive at/above it.
    positive_prefix = np.r_[0, np.cumsum(positive)]
    negative_prefix = np.r_[0, np.cumsum(~positive)]
    n = len(scores)
    errors = positive_prefix + (negative_prefix[-1] - negative_prefix)
    valid = np.r_[True, sorted_scores[:-1] < sorted_scores[1:], True]
    cut = int(np.argmin(np.where(valid, errors, np.inf)))
    if cut == 0:
        threshold = sorted_scores[0] - 1.0
    elif cut == n:
        threshold = sorted_scores[-1] + 1.0
    else:
        threshold = (sorted_scores[cut - 1] + sorted_scores[cut]) / 2.0
    result = theta.copy()
    result[-1] = -threshold
    return result


def release_accuracies(nominal, scale, seeds, test_design, test_labels, public_design, public_labels):
    raw, recalibrated = [], []
    for seed in range(seeds):
        released = nominal + np.random.RandomState(seed).laplace(0, scale, PARAMETERS)
        raw.append(B.acc(released, test_design, test_labels))
        adjusted = public_bias_recalibration(released, public_design, public_labels)
        recalibrated.append(B.acc(adjusted, test_design, test_labels))
    return raw, recalibrated


def smooth_laplace_privacy(eps_coord, target_delta, beta_fraction, *, points=2001):
    delta_coordinate = target_delta / (2 * PARAMETERS)
    beta_limit = eps_coord / (2 * math.log(2 / delta_coordinate))
    beta = beta_fraction * beta_limit
    per_coordinate = []
    maximizers = []
    for order in ORDERS:
        point = worst_case_smooth_laplace_rdp_grid(
            float(order), eps_coord, beta, ratio_points=points, shift_points=2
        )
        per_coordinate.append(point.rdp)
        maximizers.append(point)
    epsilon, order = epsilon_from_rdp(ORDERS, PARAMETERS * np.asarray(per_coordinate), target_delta)
    return epsilon, order, beta, beta_limit, per_coordinate, maximizers


def calibrate_smooth_laplace(target_epsilon, delta, beta_fraction):
    low, high = 1e-5, 1.0
    while smooth_laplace_privacy(high, delta, beta_fraction, points=1001)[0] < target_epsilon:
        high *= 2
    for _ in range(24):
        mid = (low + high) / 2
        if smooth_laplace_privacy(mid, delta, beta_fraction, points=1001)[0] <= target_epsilon:
            low = mid
        else:
            high = mid
    low *= 1 - 2e-6
    result = smooth_laplace_privacy(low, delta, beta_fraction, points=8001)
    # Recheck the shift interior densely at the RDP-conversion minimizing order.
    dense = worst_case_smooth_laplace_rdp_grid(result[1], low, result[2], ratio_points=8001, shift_points=129)
    return low, result, dense


def allocated_coordinate_privacy(base, bias_ratio, target_delta, beta_fraction, *, points=1001):
    epsilons = [base] * D + [base * bias_ratio]
    rdp_total = np.zeros(len(ORDERS))
    betas = []
    for group_size, eps_coord in ((D, base), (1, base * bias_ratio)):
        delta_coordinate = target_delta / (2 * PARAMETERS)
        beta_limit = eps_coord / (2 * math.log(2 / delta_coordinate))
        beta = beta_fraction * beta_limit
        betas.extend([beta] * group_size)
        rdp_total += group_size * np.asarray(
            [
                worst_case_smooth_laplace_rdp_grid(
                    float(order), eps_coord, beta, ratio_points=points, shift_points=2
                ).rdp
                for order in ORDERS
            ]
        )
    epsilon, order = epsilon_from_rdp(ORDERS, rdp_total, target_delta)
    return epsilon, order, np.asarray(epsilons), np.asarray(betas)


def calibrate_allocated_coordinates(target_epsilon, delta, beta_fraction, bias_ratio):
    low, high = 1e-5, 1.0
    while (
        allocated_coordinate_privacy(high, bias_ratio, delta, beta_fraction, points=501)[0] < target_epsilon
    ):
        high *= 2
    for _ in range(22):
        mid = (low + high) / 2
        if (
            allocated_coordinate_privacy(mid, bias_ratio, delta, beta_fraction, points=501)[0]
            <= target_epsilon
        ):
            low = mid
        else:
            high = mid
    low *= 1 - 2e-6
    return low, allocated_coordinate_privacy(low, bias_ratio, delta, beta_fraction, points=4001)


def vector_allocated_privacy(base, ratios, target_delta, beta_fraction, *, points=1001):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (PARAMETERS,) or np.any(ratios <= 0):
        raise ValueError("allocation ratios must be one positive value per coordinate")
    epsilons = base * ratios
    delta_coordinate = target_delta / (2 * PARAMETERS)
    betas = beta_fraction * epsilons / (2 * math.log(2 / delta_coordinate))
    total = np.zeros(len(ORDERS))
    for eps_coord, beta in zip(epsilons, betas):
        total += np.asarray(
            [
                worst_case_smooth_laplace_rdp_grid(
                    float(order),
                    float(eps_coord),
                    float(beta),
                    ratio_points=points,
                    shift_points=2,
                ).rdp
                for order in ORDERS
            ]
        )
    epsilon, order = epsilon_from_rdp(ORDERS, total, target_delta)
    return epsilon, order, epsilons, betas


def calibrate_vector_allocation(target_epsilon, delta, beta_fraction, ratios):
    low, high = 1e-5, 1.0
    while vector_allocated_privacy(high, ratios, delta, beta_fraction, points=501)[0] < target_epsilon:
        high *= 2
    for _ in range(22):
        mid = (low + high) / 2
        if vector_allocated_privacy(mid, ratios, delta, beta_fraction, points=501)[0] <= target_epsilon:
            low = mid
        else:
            high = mid
    low *= 1 - 2e-6
    return low, vector_allocated_privacy(low, ratios, delta, beta_fraction, points=4001)


def certify_coordinate_accounting(epsilons, betas, order, delta):
    groups = {}
    for eps_coord, beta in zip(epsilons, betas):
        key = (float(order), float(eps_coord), float(beta))
        groups[key] = groups.get(key, 0) + 1
    intervals = [(_certified_coordinate_law(*key), count) for key, count in groups.items()]
    rdp_lower = sum(count * item.lower for item, count in intervals)
    rdp_upper = sum(count * item.upper for item, count in intervals)
    epsilon_lower = epsilon_from_rdp([order], [rdp_lower], delta)[0]
    epsilon_upper = epsilon_from_rdp([order], [rdp_upper], delta)[0]
    return {
        "order": float(order),
        "rdp_lower": rdp_lower,
        "rdp_upper": rdp_upper,
        "epsilon_lower": epsilon_lower,
        "epsilon_upper": epsilon_upper,
        "interval_boxes_evaluated": sum(item.boxes for item, _ in intervals),
        "unique_coordinate_laws": len(intervals),
        "coordinate_laws_composed": int(sum(count for _, count in intervals)),
    }


@functools.lru_cache(maxsize=None)
def _certified_coordinate_law(order, eps_coord, beta):
    # The two-ppm calibration safety margin is at least 2e-6 at the
    # smallest target epsilon. Even after composing nine coordinates, this
    # 5e-8 per-law enclosure is comfortably below that margin.
    return certified_smooth_laplace_rdp(order, eps_coord, beta, tolerance=5e-8)


def joint_smooth_laplace_privacy(epsilon_joint, target_delta, beta_fraction, *, points=2001):
    beta_limit = epsilon_joint / (2 * math.log(2 / target_delta))
    beta = beta_fraction * beta_limit
    profile = [
        worst_case_joint_smooth_laplace_rdp_grid(
            float(order), epsilon_joint, beta, PARAMETERS, ratio_points=points
        )
        for order in ORDERS
    ]
    epsilon, order = epsilon_from_rdp(ORDERS, [point.rdp for point in profile], target_delta)
    return epsilon, order, beta, beta_limit, profile


def calibrate_joint_smooth_laplace(target_epsilon, delta, beta_fraction):
    low, high = 1e-5, 1.0
    while joint_smooth_laplace_privacy(high, delta, beta_fraction, points=1001)[0] < target_epsilon:
        high *= 2
    for _ in range(24):
        mid = (low + high) / 2
        if joint_smooth_laplace_privacy(mid, delta, beta_fraction, points=1001)[0] <= target_epsilon:
            low = mid
        else:
            high = mid
    low *= 1 - 2e-6
    return low, joint_smooth_laplace_privacy(low, delta, beta_fraction, points=8001)


def calibrate_gaussian_rdp(target_epsilon, delta):
    def privacy(multiplier):
        # Replace-one changes the normalized mean by two record bounds.
        rdp = 2.0 * ORDERS * PRIVACY_STEPS / (multiplier * multiplier)
        return epsilon_from_rdp(ORDERS, rdp, delta)

    low, high = 1e-4, 1.0
    while privacy(high)[0] > target_epsilon:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if privacy(mid)[0] > target_epsilon:
            low = mid
        else:
            high = mid
    return high, privacy(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds-root", type=Path, default=Path("ags/blobs/runs/certified"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=101)
    ap.add_argument("--beta-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--bias-allocation-ratios", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100])
    args = ap.parse_args()
    started = time.perf_counter()

    boxes, nominal, sources, data_hash = load_intersected_boxes(args.bounds_root, 10)
    certificates, diagnostics = cross_radius_profile(boxes, N)
    unconditional = np.full(PARAMETERS, 2.0 * STEPS * LR * GAMMA)
    certified, provenance = complete_profile(boxes, certificates, unconditional, N)
    clipping_cap = np.full(PARAMETERS, LR * STEPS * 2.0 * GAMMA / BATCH_SIZE)
    capped = {k: np.minimum(v, clipping_cap) for k, v in certified.items()}

    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    P = np.c_[X, np.ones(N)]
    validation_size = 1000
    P_validation = np.c_[Xte[:validation_size], np.ones(validation_size)]
    y_validation = yte[:validation_size]
    Pte = np.c_[Xte[validation_size:], np.ones(len(Xte) - validation_size)]
    yte = yte[validation_size:]
    public_rms = np.sqrt(np.mean(P_validation * P_validation, axis=0))
    # Allocation ratios must be public.  In particular, do not use Abar(D): a
    # certificate can be sound while still being private-data dependent.
    public_scores = np.maximum(public_rms, np.finfo(float).tiny)
    coordinate_allocation_ratios = {}
    for power in (0.25, 0.5, 0.75):
        ratios = public_scores**power
        ratios /= np.exp(np.mean(np.log(ratios)))
        coordinate_allocation_ratios[f"public_rms_power_{power:g}"] = np.clip(ratios, 0.25, 4.0)
    rows = []
    all_coordinate_cells_certified = True
    completed_fraction_cells = 0
    total_fraction_cells = len(args.epsilons) * len(args.beta_fractions)
    for target in args.epsilons:
        dp_multiplier, (dp_epsilon, dp_order) = calibrate_gaussian_rdp(target, 1e-5)
        dp_std = dp_multiplier * GAMMA * math.sqrt(PARAMETERS) / BATCH_SIZE
        dp_acc = []
        dp_recalibrated = []
        for seed in range(args.seeds):
            released = B.train_hinge(
                P,
                y,
                n_s=STEPS,
                lr=LR,
                gamma=GAMMA,
                lam=0.0,
                b=BATCH_SIZE,
                noise_scale=dp_std,
                rng=np.random.RandomState(seed),
                noise_kind="gauss",
            )
            dp_acc.append(B.acc(released, Pte, yte))
            adjusted = public_bias_recalibration(released, P_validation, y_validation)
            dp_recalibrated.append(B.acc(adjusted, Pte, yte))
        variants = {}
        allocated_variants = {}
        coordinate_allocated_variants = {}
        joint_variants = {}
        for fraction in args.beta_fractions:
            eps_coord, privacy, dense = calibrate_smooth_laplace(target, 1e-5, fraction)
            achieved, order, beta, beta_limit, _, maximizers = privacy
            panels = {}
            for name, profile in (("certified_tail", certified), ("clipping_cap", capped)):
                sensitivity = smooth_profile(profile, beta)
                scale = 2.0 * sensitivity / eps_coord
                acc, recalibrated = release_accuracies(
                    nominal,
                    scale,
                    args.seeds,
                    Pte,
                    yte,
                    P_validation,
                    y_validation,
                )
                panels[name] = {
                    "accuracies": acc,
                    "median_accuracy": float(np.median(acc)),
                    "public_bias_recalibrated_accuracies": recalibrated,
                    "public_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
                    "sensitivity_quantiles": quantiles(sensitivity),
                    "scale_quantiles": quantiles(scale),
                }
            variants[str(fraction)] = {
                "epsilon_coordinate": eps_coord,
                "achieved_epsilon": achieved,
                "minimizing_order": order,
                "beta": beta,
                "beta_limit": beta_limit,
                "dense_shift_check": dense.__dict__,
                "order_maximizers": [p.__dict__ for p in maximizers if math.isfinite(p.rdp)],
                "panels": panels,
                "interval_certification": certify_coordinate_accounting(
                    np.full(PARAMETERS, eps_coord),
                    np.full(PARAMETERS, beta),
                    order,
                    1e-5,
                ),
            }
            all_coordinate_cells_certified &= (
                variants[str(fraction)]["interval_certification"]["epsilon_upper"] <= target
            )
            joint_parameter, joint_privacy = calibrate_joint_smooth_laplace(target, 1e-5, fraction)
            joint_epsilon, joint_order, joint_beta, joint_beta_limit, joint_points = joint_privacy
            joint_panels = {}
            for name, profile in (("certified_tail", certified), ("clipping_cap", capped)):
                joint_sensitivity = max(
                    math.exp(-joint_beta * k) * float(np.asarray(value).sum()) for k, value in profile.items()
                )
                joint_scale = 2 * joint_sensitivity / joint_parameter
                joint_acc, joint_recalibrated = release_accuracies(
                    nominal,
                    joint_scale,
                    args.seeds,
                    Pte,
                    yte,
                    P_validation,
                    y_validation,
                )
                joint_panels[name] = {
                    "accuracies": joint_acc,
                    "median_accuracy": float(np.median(joint_acc)),
                    "public_bias_recalibrated_accuracies": joint_recalibrated,
                    "public_bias_recalibrated_median_accuracy": float(np.median(joint_recalibrated)),
                    "joint_smooth_sensitivity": joint_sensitivity,
                    "shared_coordinate_scale": joint_scale,
                }
            joint_variants[str(fraction)] = {
                "epsilon_joint_parameter": joint_parameter,
                "achieved_epsilon": joint_epsilon,
                "minimizing_order": joint_order,
                "beta": joint_beta,
                "beta_limit": joint_beta_limit,
                "order_maximizers": [p.__dict__ for p in joint_points if math.isfinite(p.rdp)],
                "panels": joint_panels,
            }
            allocations = {}
            for bias_ratio in args.bias_allocation_ratios:
                _, allocated = calibrate_allocated_coordinates(target, 1e-5, fraction, bias_ratio)
                allocated_epsilon, allocated_order, eps_vector, beta_vector = allocated
                allocation_panels = {}
                for name, profile in (("certified_tail", certified), ("clipping_cap", capped)):
                    allocated_sensitivity = smooth_profile(profile, beta_vector)
                    allocated_scale = 2 * allocated_sensitivity / eps_vector
                    allocation_acc, allocation_recalibrated = release_accuracies(
                        nominal,
                        allocated_scale,
                        args.seeds,
                        Pte,
                        yte,
                        P_validation,
                        y_validation,
                    )
                    validation_acc = [
                        B.acc(
                            nominal + np.random.RandomState(seed).laplace(0, allocated_scale),
                            P_validation,
                            y_validation,
                        )
                        for seed in range(args.seeds)
                    ]
                    allocation_panels[name] = {
                        "accuracies": allocation_acc,
                        "median_accuracy": float(np.median(allocation_acc)),
                        "public_bias_recalibrated_accuracies": allocation_recalibrated,
                        "public_bias_recalibrated_median_accuracy": float(np.median(allocation_recalibrated)),
                        "validation_accuracies": validation_acc,
                        "validation_median_accuracy": float(np.median(validation_acc)),
                        "sensitivity_quantiles": quantiles(allocated_sensitivity),
                        "scale_quantiles": quantiles(allocated_scale),
                    }
                allocations[str(bias_ratio)] = {
                    "achieved_epsilon": allocated_epsilon,
                    "minimizing_order": allocated_order,
                    "epsilon_coordinates": eps_vector.tolist(),
                    "beta_coordinates": beta_vector.tolist(),
                    "panels": allocation_panels,
                    "interval_certification": certify_coordinate_accounting(
                        eps_vector, beta_vector, allocated_order, 1e-5
                    ),
                }
                all_coordinate_cells_certified &= (
                    allocations[str(bias_ratio)]["interval_certification"]["epsilon_upper"] <= target
                )
            for panel_name in ("certified_tail", "clipping_cap"):
                selected_ratio = max(
                    args.bias_allocation_ratios,
                    key=lambda ratio: allocations[str(ratio)]["panels"][panel_name][
                        "validation_median_accuracy"
                    ],
                )
                allocations[f"selected_by_public_validation_{panel_name}"] = {
                    "bias_ratio": selected_ratio,
                    "validation_median_accuracy": allocations[str(selected_ratio)]["panels"][panel_name][
                        "validation_median_accuracy"
                    ],
                    "heldout_test_median_accuracy": allocations[str(selected_ratio)]["panels"][panel_name][
                        "median_accuracy"
                    ],
                }
            allocated_variants[str(fraction)] = allocations
            coordinate_allocations = {}
            for allocation_name, ratios in coordinate_allocation_ratios.items():
                _, allocated = calibrate_vector_allocation(target, 1e-5, fraction, ratios)
                allocated_epsilon, allocated_order, eps_vector, beta_vector = allocated
                sensitivity = smooth_profile(certified, beta_vector)
                scale = 2 * sensitivity / eps_vector
                test_accuracies, test_recalibrated = release_accuracies(
                    nominal,
                    scale,
                    args.seeds,
                    Pte,
                    yte,
                    P_validation,
                    y_validation,
                )
                validation_accuracies = [
                    B.acc(nominal + np.random.RandomState(seed).laplace(0, scale), P_validation, y_validation)
                    for seed in range(args.seeds)
                ]
                coordinate_allocations[allocation_name] = {
                    "ratios": ratios.tolist(),
                    "achieved_epsilon": allocated_epsilon,
                    "minimizing_order": allocated_order,
                    "epsilon_coordinates": eps_vector.tolist(),
                    "beta_coordinates": beta_vector.tolist(),
                    "sensitivity_quantiles": quantiles(sensitivity),
                    "scale_quantiles": quantiles(scale),
                    "accuracies": test_accuracies,
                    "median_accuracy": float(np.median(test_accuracies)),
                    "public_bias_recalibrated_accuracies": test_recalibrated,
                    "public_bias_recalibrated_median_accuracy": float(np.median(test_recalibrated)),
                    "validation_accuracies": validation_accuracies,
                    "validation_median_accuracy": float(np.median(validation_accuracies)),
                    "interval_certification": certify_coordinate_accounting(
                        eps_vector, beta_vector, allocated_order, 1e-5
                    ),
                }
                all_coordinate_cells_certified &= (
                    coordinate_allocations[allocation_name]["interval_certification"]["epsilon_upper"]
                    <= target
                )
            selected_name = max(
                coordinate_allocations,
                key=lambda name: coordinate_allocations[name]["validation_median_accuracy"],
            )
            coordinate_allocations["selected_by_public_validation"] = {
                "allocation": selected_name,
                "validation_median_accuracy": coordinate_allocations[selected_name][
                    "validation_median_accuracy"
                ],
                "heldout_test_median_accuracy": coordinate_allocations[selected_name]["median_accuracy"],
            }
            coordinate_allocated_variants[str(fraction)] = coordinate_allocations
            completed_fraction_cells += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_fraction_cells * (total_fraction_cells - completed_fraction_cells)
            print(
                f"[blob-rdp-accounting] epsilon={target:g} beta_fraction={fraction:g} "
                f"cells={completed_fraction_cells}/{total_fraction_cells} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        rows.append(
            {
                "target_epsilon": target,
                "ags_coordinate": variants,
                "ags_public_weight_bias_allocations": allocated_variants,
                "ags_public_coordinate_allocations": coordinate_allocated_variants,
                "ags_joint_l1": joint_variants,
                "dpsgd": {
                    "noise_multiplier": dp_multiplier,
                    "achieved_epsilon": dp_epsilon,
                    "minimizing_order": dp_order,
                    "gradient_noise_std": dp_std,
                    "accuracies": dp_acc,
                    "median_accuracy": float(np.median(dp_acc)),
                    "public_bias_recalibrated_accuracies": dp_recalibrated,
                    "public_bias_recalibrated_median_accuracy": float(np.median(dp_recalibrated)),
                },
            }
        )
        print(
            f"[blob-rdp] eps={target:g} DP={np.median(dp_acc):.4f} "
            + " ".join(
                f"beta={f:g}:coord={variants[str(f)]['panels']['clipping_cap']['median_accuracy']:.4f}"
                f"/joint={joint_variants[str(f)]['panels']['certified_tail']['median_accuracy']:.4f}"
                for f in args.beta_fractions
            ),
            flush=True,
        )

    payload = {
        "schema": 1,
        "status": (
            "coordinate_rdp_interval_certified_using_analytic_envelope_smoothness"
            if all_coordinate_cells_certified
            else "coordinate_rdp_interval_certification_failed"
        ),
        "experiment": "one_release_blobs_matched_rdp",
        "privacy": {"delta": 1e-5, "adjacency": "replace-one", "q": 1.0},
        "setting": {
            "N": N,
            "parameters": PARAMETERS,
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "record_touching_privacy_steps": PRIVACY_STEPS,
            "learning_rate": LR,
            "coordinate_clip": GAMMA,
            "frequency": 10,
            "seeds": args.seeds,
            "public_validation_size": validation_size,
            "heldout_test_size": len(yte),
        },
        "accounting": (
            "common RDP composition/conversion; analytic Laplace pair integral and "
            "exact Gaussian RDP. Fixed cyclic batches use adaptive parallel composition "
            "within each epoch, so each record incurs 20 Gaussian releases."
        ),
        "ags_caveat": (
            "coordinate releases use analytic shift monotonicity plus interval "
            "branch-and-bound over the continuous scale ratio; the separate joint-l1 "
            "panel remains numerical. Beta fractions above one exceed the black-box "
            "theorem's sufficient beta limit and are justified by the certified direct "
            "coordinate RDP accounting."
        ),
        "envelope_smoothness_basis": (
            "uses the stated analytic beta-smoothness result for the corrected "
            "cross-radius envelope construction; finite-domain tests independently "
            "exercise exact envelopes, while relaxed/time-limited solver boxes are "
            "not exhaustively neighbor-enumerated"
        ),
        "clipping_cap_caveat": "reported separately from the unconditional MILP/clipping-diameter tail",
        "selection_caveat": (
            "all cells are retained and individually accounted; validation-selected "
            "summaries are exploratory because evaluating private-trained candidate "
            "releases and selecting among them is not privacy-accounted"
        ),
        "public_postprocessing": (
            "Every arm also reports deterministic intercept recalibration on the "
            "labeled public validation set. This is postprocessing of each released "
            "model and consumes no additional privacy budget; raw results remain "
            "alongside it."
        ),
        "box_tightening": (
            "all sound MILP/LP boxes are intersected with the public clipped-output "
            "range and, by nested dataset balls, with every stored enclosing-radius box"
        ),
        "available_radii": sorted(boxes),
        "cross_radius_radii": sorted(certificates),
        "public_coordinate_allocation_rms_scores": public_scores.tolist(),
        "profile_provenance": {str(k): v for k, v in provenance.items()},
        "box_sources": sources,
        "data_sha256": data_hash,
        "cross_radius_diagnostics": diagnostics,
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    digest = write_immutable_json(args.output, payload)
    print(f"[blob-rdp] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
