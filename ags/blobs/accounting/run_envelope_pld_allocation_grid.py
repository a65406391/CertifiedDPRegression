"""The same certified envelope, across every retained budget-allocation rule.

The headline cell fixes one way of splitting epsilon across the nine coordinates. That cell was
chosen during exploratory development, not preregistered, so this runs all nine retained rules
(uniform, three public root-mean-square weightings, five weight-to-bias ratios) and reports each.

Reporting all nine is the point: selecting the best rule after the fact and quoting it alone would
be selection on the test set. `summarize_allocation_robustness.py` reads this.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from ags.blobs.accounting.run_envelope_pld_comparison import COARSE, FINE
from ags.blobs.accounting.run_public_projection import projected_accuracies


def compose_envelopes(epsilon, beta, delta, settings):
    grouped = Counter(zip(np.asarray(epsilon, float), np.asarray(beta, float)))
    result = None
    for (epsilon_coordinate, beta_coordinate), count in grouped.items():
        mechanism = continuous_ratio_laplace_pld_envelope(
            epsilon_coordinate, beta_coordinate, **settings
        ).self_compose(count)
        result = mechanism if result is None else result.compose(mechanism)
    return float(result.get_epsilon_for_delta(delta))


def calibrate(target, epsilon, beta, delta, settings):
    low, high = 0.5, 1.0
    while compose_envelopes(low * epsilon, low * beta, delta, settings) > target:
        high, low = low, low * 0.5
    while compose_envelopes(high * epsilon, high * beta, delta, settings) <= target:
        low, high = high, high * 1.25
    for _ in range(18):
        middle = 0.5 * (low + high)
        if compose_envelopes(middle * epsilon, middle * beta, delta, settings) <= target:
            low = middle
        else:
            high = middle
    factor = low * (1 - 2e-5)
    return factor, compose_envelopes(factor * epsilon, factor * beta, delta, settings)


def retained_rules(row):
    uniform = row["ags_coordinate"]["4.0"]
    yield (
        "uniform",
        uniform,
        np.full(PARAMETERS, uniform["epsilon_coordinate"]),
        np.full(PARAMETERS, uniform["beta"]),
    )
    for name, cell in row["ags_public_coordinate_allocations"]["4.0"].items():
        if name.startswith("public_rms_power_"):
            yield name, cell, np.asarray(cell["epsilon_coordinates"]), np.asarray(cell["beta_coordinates"])
    for ratio, cell in row["ags_public_weight_bias_allocations"]["4.0"].items():
        if ratio.replace(".", "", 1).isdigit():
            yield (
                f"weight_bias_ratio_{ratio}",
                cell,
                np.asarray(cell["epsilon_coordinates"]),
                np.asarray(cell["beta_coordinates"]),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[15, 20, 25, 30])
    parser.add_argument("--seeds", type=int, default=101)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    source_rows = {row["target_epsilon"]: row for row in source["rows"]}
    delta = float(source["privacy"]["delta"])
    boxes, nominal, box_sources, data_hash = load_intersected_boxes(Path("ags/blobs/runs/certified"), 10)
    certificates, _ = cross_radius_profile(boxes, N)
    profile, provenance = complete_profile(
        boxes,
        certificates,
        np.full(PARAMETERS, 2 * STEPS * LR * GAMMA),
        N,
    )
    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    design = np.c_[X, np.ones(N)]
    nominal_check = B.train_hinge(design, y, n_s=STEPS, lr=LR, gamma=GAMMA, lam=0, b=BATCH_SIZE)
    np.testing.assert_allclose(nominal, nominal_check, rtol=0, atol=1e-10)
    public_size = int(source["setting"]["public_validation_size"])
    public_design = np.c_[Xte[:public_size], np.ones(public_size)]
    public_labels = yte[:public_size]
    test_design = np.c_[Xte[public_size:], np.ones(len(Xte) - public_size)]
    test_labels = yte[public_size:]
    radius = STEPS * LR * GAMMA

    rows = []
    for target in args.epsilons:
        rules = []
        for name, cell, base_epsilon, base_beta in retained_rules(source_rows[target]):
            envelope_factor, coarse = calibrate(target, base_epsilon, base_beta, delta, COARSE)
            fine = compose_envelopes(
                envelope_factor * base_epsilon,
                envelope_factor * base_beta,
                delta,
                FINE,
            )
            if fine > target:
                envelope_factor, fine = calibrate(target, base_epsilon, base_beta, delta, FINE)
                coarse = compose_envelopes(
                    envelope_factor * base_epsilon,
                    envelope_factor * base_beta,
                    delta,
                    COARSE,
                )
            rdp_upper = float(cell["interval_certification"]["epsilon_upper"])
            factor = max(1.0, envelope_factor)
            accountant = (
                "continuous_pointwise_envelope_native_pld" if envelope_factor > 1 else "analytic_interval_rdp"
            )
            selected_upper = max(coarse, fine) if envelope_factor > 1 else rdp_upper
            epsilon, beta = factor * base_epsilon, factor * base_beta
            sensitivity = smooth_profile(profile, beta)
            scale = 2 * sensitivity / epsilon
            utility = projected_accuracies(
                nominal,
                scale,
                args.seeds,
                test_design,
                test_labels,
                public_design,
                public_labels,
                radius,
            )
            rules.append(
                {
                    "rule": name,
                    "selected_accountant": accountant,
                    "selected_parameter_factor_over_rdp": factor,
                    "selected_achieved_epsilon_upper": selected_upper,
                    "envelope_calibrated_factor": envelope_factor,
                    "achieved_epsilon_coarse_envelope": coarse,
                    "achieved_epsilon_fine_envelope": fine,
                    "utility": utility,
                }
            )
            print(
                f"[blob-envelope-grid] epsilon={target:g} rule={name} "
                f"factor={factor:.5f} median="
                f"{utility['latest_box_unprojected_median_accuracy']:.4f}",
                flush=True,
            )
        rows.append({"target_epsilon": target, "rules": rules})
    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "certified_native_pld_envelope_all_retained_beta4_rules",
            "selection_status": (
                "Every retained beta=4 rule is reported; no rule is selected using test accuracy."
            ),
            "proof_basis": (
                "The continuous pointwise privacy-loss envelope described in "
                "blobs_native_pld_envelope_overnight.json is independently composed "
                "for every coordinate law."
            ),
            "source": str(args.input),
            "privacy": source["privacy"],
            "coarse_envelope": COARSE,
            "fine_envelope_audit": FINE,
            "available_radii": sorted(boxes),
            "profile_provenance": {str(k): value for k, value in provenance.items()},
            "box_sources": box_sources,
            "data_sha256": data_hash,
            "rows": rows,
        },
    )
    print(f"[blob-envelope-grid] wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
