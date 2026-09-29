"""Certified native-Laplace PLD for the fixed blobs AGS cell: the headline accounting.

The generic privacy-loss grid charges AGS more than its mechanism actually costs. This builds a
certified envelope instead: it partitions the output space and the log scale-ratio range, bounds the
privacy loss on each rectangle by interval arithmetic, and assigns the exact Laplace mass to the
rounded-up loss. The result dominates every admissible continuous shift and scale ratio under a
coupling, so it stays sound under coordinate composition.

The envelope is looser than RDP at small epsilon, so the driver keeps whichever is stronger per row
and records which in `selected_accountant`. Reads the serialized MILP boxes, writes an immutable
artifact under `ags/accounting/artifacts/`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ags.blobs import baselines as B
from ags.blobs.shared import load_blobs
from common.accounting.artifacts import write_immutable_json
from common.accounting.laplace_pld import (
    continuous_ratio_laplace_pld_envelope,
)
from ags.blobs.accounting.run_cross_radius_comparison import (
    BATCH_SIZE,
    D,
    GAMMA,
    LR,
    N,
    PARAMETERS,
    SEP,
    STEPS,
    complete_profile,
    cross_radius_profile,
    load_intersected_boxes,
    smooth_profile,
)
from ags.blobs.accounting.run_public_projection import projected_accuracies


COARSE = {
    "privacy_loss_interval": 0.001,
    "output_interval": 0.002,
    "ratio_intervals": 1000,
    "tail_mass_truncation": 1e-14,
}
FINE = {
    "privacy_loss_interval": 0.0005,
    "output_interval": 0.001,
    "ratio_intervals": 2000,
    "tail_mass_truncation": 1e-15,
}


def envelope_epsilon(epsilon_coordinates, beta_coordinates, delta: float, settings: dict) -> float:
    weight = continuous_ratio_laplace_pld_envelope(
        float(epsilon_coordinates[0]), float(beta_coordinates[0]), **settings
    )
    bias = continuous_ratio_laplace_pld_envelope(
        float(epsilon_coordinates[-1]), float(beta_coordinates[-1]), **settings
    )
    return float(weight.self_compose(8).compose(bias).get_epsilon_for_delta(delta))


def calibrate(target, base_epsilon, base_beta, delta, settings):
    low, high = 0.5, 1.0
    while envelope_epsilon(low * base_epsilon, low * base_beta, delta, settings) > target:
        high = low
        low *= 0.5
    while envelope_epsilon(high * base_epsilon, high * base_beta, delta, settings) <= target:
        low = high
        high *= 1.25
    for _ in range(20):
        middle = 0.5 * (low + high)
        if envelope_epsilon(middle * base_epsilon, middle * base_beta, delta, settings) <= target:
            low = middle
        else:
            high = middle
    factor = low * (1.0 - 2e-5)
    achieved = envelope_epsilon(factor * base_epsilon, factor * base_beta, delta, settings)
    return factor, achieved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, default=101)
    args = parser.parse_args()
    started = time.perf_counter()
    source = json.loads(args.input.read_text())
    source_rows = {row["target_epsilon"]: row for row in source["rows"]}
    targets = args.epsilons or sorted(source_rows)
    delta = float(source["privacy"]["delta"])

    boxes, nominal, box_sources, data_hash = load_intersected_boxes(Path("ags/blobs/runs/certified"), 10)
    certificates, _ = cross_radius_profile(boxes, N)
    profile, provenance = complete_profile(
        boxes,
        certificates,
        np.full(PARAMETERS, 2.0 * STEPS * LR * GAMMA),
        N,
    )
    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    design = np.c_[X, np.ones(N)]
    nominal_check = B.train_hinge(
        design,
        y,
        n_s=STEPS,
        lr=LR,
        gamma=GAMMA,
        lam=0.0,
        b=BATCH_SIZE,
    )
    np.testing.assert_allclose(nominal, nominal_check, rtol=0, atol=1e-10)
    public_size = int(source["setting"]["public_validation_size"])
    public_design = np.c_[Xte[:public_size], np.ones(public_size)]
    public_labels = yte[:public_size]
    test_design = np.c_[Xte[public_size:], np.ones(len(Xte) - public_size)]
    test_labels = yte[public_size:]
    radius = STEPS * LR * GAMMA

    rows = []
    for target in targets:
        cell = source_rows[target]["ags_public_weight_bias_allocations"]["4.0"]["2.0"]
        base_epsilon = np.asarray(cell["epsilon_coordinates"], dtype=float)
        base_beta = np.asarray(cell["beta_coordinates"], dtype=float)
        envelope_factor, achieved = calibrate(target, base_epsilon, base_beta, delta, COARSE)
        fine_epsilon = envelope_epsilon(
            envelope_factor * base_epsilon,
            envelope_factor * base_beta,
            delta,
            FINE,
        )
        if fine_epsilon > target:
            envelope_factor, fine_epsilon = calibrate(target, base_epsilon, base_beta, delta, FINE)
            achieved = envelope_epsilon(
                envelope_factor * base_epsilon,
                envelope_factor * base_beta,
                delta,
                COARSE,
            )
        rdp_epsilon = cell["interval_certification"]["epsilon_upper"]
        if envelope_factor > 1.0:
            factor = envelope_factor
            selected_accountant = "continuous_pointwise_envelope_native_pld"
            selected_achieved_epsilon = max(achieved, fine_epsilon)
        else:
            factor = 1.0
            selected_accountant = "analytic_interval_rdp"
            selected_achieved_epsilon = rdp_epsilon
        epsilon = factor * base_epsilon
        beta = factor * base_beta
        sensitivity = smooth_profile(profile, beta)
        scales = 2.0 * sensitivity / epsilon
        utility = projected_accuracies(
            nominal,
            scales,
            args.seeds,
            test_design,
            test_labels,
            public_design,
            public_labels,
            radius,
        )
        rows.append(
            {
                "target_epsilon": target,
                "rdp_base_epsilon_upper": rdp_epsilon,
                "envelope_calibrated_factor": envelope_factor,
                "selected_parameter_factor_over_rdp": factor,
                "selected_accountant": selected_accountant,
                "selected_achieved_epsilon_upper": selected_achieved_epsilon,
                "achieved_epsilon_coarse_envelope": achieved,
                "achieved_epsilon_fine_envelope": fine_epsilon,
                "epsilon_coordinates": epsilon.tolist(),
                "beta_coordinates": beta.tolist(),
                "sensitivity": sensitivity.tolist(),
                "noise_scales": scales.tolist(),
                "utility": utility,
            }
        )
        print(
            f"[blob-envelope-pld] epsilon={target:g} factor={factor:.5f} "
            f"accountant={selected_accountant} "
            f"coarse={achieved:.6f} fine={fine_epsilon:.6f} "
            f"median={utility['latest_box_unprojected_median_accuracy']:.4f}",
            flush=True,
        )

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "coordinate_native_pld_certified_by_continuous_pointwise_envelope",
            "proof_basis": (
                "For every coordinate, normalized output x, admissible shift, and "
                "continuous scale ratio, rectangle arithmetic upper-bounds the privacy "
                "loss. Exact standard-Laplace bin mass is assigned to the rounded-up "
                "bound and truncated tails to infinite privacy loss. Independent sums "
                "of these coupled upper variables dominate coordinate composition. "
                "The symmetric log-ratio domain also covers the reverse direction."
            ),
            "source": str(args.input),
            "setting": source["setting"],
            "privacy": source["privacy"],
            "ags_fixed_reference": {
                "beta_fraction": 4.0,
                "bias_epsilon_ratio": 2.0,
                "selection_status": "chosen during exploratory method development",
            },
            "coarse_envelope": COARSE,
            "fine_envelope_audit": FINE,
            "available_radii": sorted(boxes),
            "profile_provenance": {str(k): value for k, value in provenance.items()},
            "box_sources": box_sources,
            "data_sha256": data_hash,
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[blob-envelope-pld] wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
