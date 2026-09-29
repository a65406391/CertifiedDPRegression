"""Project each release onto the public trajectory cube, symmetrically.

Every parameter of the clean run lies in a box fixed by the clipping ceiling, which is public, so
clipping a noisy release into it is post-processing and costs nothing. It is applied to AGS and to
DP-SGD alike, because a projection given to one arm and not the other is not a comparison.

Raw unprojected accuracy stays the primary number; this is reported beside it.
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
from ags.blobs.accounting.run_rdp_comparison import public_bias_recalibration


def projected_accuracies(
    nominal, scale, seeds, test_design, test_labels, public_design, public_labels, radius
):
    raw, recalibrated = [], []
    unprojected_raw, unprojected_recalibrated = [], []
    for seed in range(seeds):
        released = nominal + np.random.RandomState(seed).laplace(0, scale, PARAMETERS)
        unprojected_raw.append(B.acc(released, test_design, test_labels))
        unprojected_adjusted = public_bias_recalibration(released, public_design, public_labels)
        unprojected_recalibrated.append(B.acc(unprojected_adjusted, test_design, test_labels))
        released = np.clip(released, -radius, radius)
        raw.append(B.acc(released, test_design, test_labels))
        adjusted = public_bias_recalibration(released, public_design, public_labels)
        recalibrated.append(B.acc(adjusted, test_design, test_labels))
    return {
        "latest_box_unprojected_accuracies": unprojected_raw,
        "latest_box_unprojected_median_accuracy": float(np.median(unprojected_raw)),
        "latest_box_unprojected_public_bias_recalibrated_accuracies": unprojected_recalibrated,
        "latest_box_unprojected_public_bias_recalibrated_median_accuracy": float(
            np.median(unprojected_recalibrated)
        ),
        "accuracies": raw,
        "median_accuracy": float(np.median(raw)),
        "public_bias_recalibrated_accuracies": recalibrated,
        "public_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
    }


def projected_dpsgd_accuracies(
    design, labels, gradient_std, seeds, test_design, test_labels, public_design, public_labels, radius
):
    raw, recalibrated = [], []
    for seed in range(seeds):
        released = B.train_hinge(
            design,
            labels,
            n_s=STEPS,
            lr=LR,
            gamma=GAMMA,
            lam=0.0,
            b=BATCH_SIZE,
            noise_scale=gradient_std,
            rng=np.random.RandomState(seed),
            noise_kind="gauss",
        )
        released = np.clip(released, -radius, radius)
        raw.append(B.acc(released, test_design, test_labels))
        adjusted = public_bias_recalibration(released, public_design, public_labels)
        recalibrated.append(B.acc(adjusted, test_design, test_labels))
    return {
        "accuracies": raw,
        "median_accuracy": float(np.median(raw)),
        "public_bias_recalibrated_accuracies": recalibrated,
        "public_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dpsgd-pld-input",
        type=Path,
        default=Path("ags/blobs/accounting/artifacts/blobs_dpsgd_pld_reference_overnight.json"),
    )
    parser.add_argument("--bounds-root", type=Path, default=Path("ags/blobs/runs/certified"))
    args = parser.parse_args()
    started = time.perf_counter()
    source = json.loads(args.input.read_text())
    pld_source = json.loads(args.dpsgd_pld_input.read_text()) if args.dpsgd_pld_input.exists() else None
    pld_rows = {row["target_epsilon"]: row for row in pld_source["rows"]} if pld_source is not None else {}
    seeds = int(source["setting"]["seeds"])

    boxes, nominal, box_sources, data_hash = load_intersected_boxes(args.bounds_root, 10)
    certificates, _ = cross_radius_profile(boxes, N)
    profile, provenance = complete_profile(
        boxes,
        certificates,
        np.full(PARAMETERS, 2.0 * STEPS * LR * GAMMA),
        N,
    )
    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    # Recompute rather than loading a hidden clean model from an artifact.
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
    for row in source["rows"]:
        uniform, bias_allocations, coordinate_allocations = {}, {}, {}
        for fraction, cell in row["ags_coordinate"].items():
            epsilon = np.full(PARAMETERS, cell["epsilon_coordinate"])
            beta = np.full(PARAMETERS, cell["beta"])
            scale = 2.0 * smooth_profile(profile, beta) / epsilon
            uniform[fraction] = projected_accuracies(
                nominal,
                scale,
                seeds,
                test_design,
                test_labels,
                public_design,
                public_labels,
                radius,
            )
        for fraction, allocations in row["ags_public_weight_bias_allocations"].items():
            bias_allocations[fraction] = {}
            for ratio, cell in allocations.items():
                if not ratio.replace(".", "", 1).isdigit():
                    continue
                epsilon = np.asarray(cell["epsilon_coordinates"])
                beta = np.asarray(cell["beta_coordinates"])
                scale = 2.0 * smooth_profile(profile, beta) / epsilon
                bias_allocations[fraction][ratio] = projected_accuracies(
                    nominal,
                    scale,
                    seeds,
                    test_design,
                    test_labels,
                    public_design,
                    public_labels,
                    radius,
                )
        for fraction, allocations in row["ags_public_coordinate_allocations"].items():
            coordinate_allocations[fraction] = {}
            for name, cell in allocations.items():
                if not name.startswith("public_rms_power_"):
                    continue
                epsilon = np.asarray(cell["epsilon_coordinates"])
                beta = np.asarray(cell["beta_coordinates"])
                scale = 2.0 * smooth_profile(profile, beta) / epsilon
                coordinate_allocations[fraction][name] = projected_accuracies(
                    nominal,
                    scale,
                    seeds,
                    test_design,
                    test_labels,
                    public_design,
                    public_labels,
                    radius,
                )
        rows.append(
            {
                "target_epsilon": row["target_epsilon"],
                "ags_uniform": uniform,
                "ags_public_weight_bias_allocations": bias_allocations,
                "ags_public_coordinate_allocations": coordinate_allocations,
                "dpsgd_rdp": projected_dpsgd_accuracies(
                    design,
                    y,
                    row["dpsgd"]["gradient_noise_std"],
                    seeds,
                    test_design,
                    test_labels,
                    public_design,
                    public_labels,
                    radius,
                ),
                "dpsgd_native_pld": (
                    projected_dpsgd_accuracies(
                        design,
                        y,
                        pld_rows[row["target_epsilon"]]["gradient_noise_std"],
                        seeds,
                        test_design,
                        test_labels,
                        public_design,
                        public_labels,
                        radius,
                    )
                    if row["target_epsilon"] in pld_rows
                    else None
                ),
            }
        )
        print(f"[blob-project] epsilon={row['target_epsilon']:g}", flush=True)

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "complete",
            "experiment": "blobs_public_cube_projection",
            "privacy": source["privacy"],
            "setting": source["setting"],
            "postprocessing": (
                "Every coordinate of the already-private release is projected onto "
                f"the public clean-trajectory range [-{radius:g},{radius:g}]."
            ),
            "sensitivity_source_note": (
                "Noise scales are recomputed from the latest complete sound boxes; "
                "they may be tighter than the input artifact if an exact radius "
                "completed after that artifact loaded its boxes."
            ),
            "available_radii": sorted(boxes),
            "profile_provenance": {str(k): value for k, value in provenance.items()},
            "box_sources": box_sources,
            "data_sha256": data_hash,
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[blob-project] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
