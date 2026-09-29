"""Standard Poisson-sampled DP-SGD stress test for the Blobs comparison.

This is deliberately separate from the exactly matched fixed-cyclic baseline:
subsampling changes the training algorithm as well as its accountant.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ags.blobs import baselines as B
from ags.blobs.shared import load_blobs
from common.accounting.accounting import calibrate_dpsgd
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
)
from ags.blobs.accounting.run_rdp_comparison import public_bias_recalibration


def train_poisson(design, labels, noise_multiplier, seed):
    q = BATCH_SIZE / N
    record_l2_bound = GAMMA * math.sqrt(PARAMETERS)
    sum_noise_std = noise_multiplier * record_l2_bound
    rng = np.random.RandomState(seed)
    theta = np.zeros(PARAMETERS)
    for _ in range(STEPS):
        indices = np.flatnonzero(rng.random_sample(N) < q)
        gradients = (
            np.clip(B.hinge_grads(theta, design[indices], labels[indices]), -GAMMA, GAMMA)
            if len(indices)
            else np.zeros((0, PARAMETERS))
        )
        noisy_sum = gradients.sum(axis=0) + rng.normal(0.0, sum_noise_std, PARAMETERS)
        theta -= LR * noisy_sum / BATCH_SIZE
    return theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    )
    parser.add_argument("--seeds", type=int, default=101)
    parser.add_argument("--interval", type=float, default=1e-3)
    args = parser.parse_args()
    started = time.perf_counter()
    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    design = np.c_[X, np.ones(N)]
    public_size = 1000
    public_design = np.c_[Xte[:public_size], np.ones(public_size)]
    public_labels = yte[:public_size]
    test_design = np.c_[Xte[public_size:], np.ones(len(Xte) - public_size)]
    test_labels = yte[public_size:]
    radius = STEPS * LR * GAMMA

    rows = []
    q = BATCH_SIZE / N
    for target in args.epsilons:
        calibration = calibrate_dpsgd(
            target,
            1e-5,
            q=q,
            steps=STEPS,
            interval=args.interval,
            search_iterations=18,
        )
        raw, recalibrated, projected, projected_recalibrated = [], [], [], []
        for seed in range(args.seeds):
            released = train_poisson(design, y, calibration.mechanism_parameter, seed)
            raw.append(B.acc(released, test_design, test_labels))
            recalibrated.append(
                B.acc(
                    public_bias_recalibration(released, public_design, public_labels),
                    test_design,
                    test_labels,
                )
            )
            clipped = np.clip(released, -radius, radius)
            projected.append(B.acc(clipped, test_design, test_labels))
            projected_recalibrated.append(
                B.acc(
                    public_bias_recalibration(clipped, public_design, public_labels), test_design, test_labels
                )
            )
        rows.append(
            {
                "target_epsilon": target,
                "calibration": calibration.__dict__,
                "accuracies": raw,
                "median_accuracy": float(np.median(raw)),
                "public_bias_recalibrated_accuracies": recalibrated,
                "public_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
                "public_cube_projected_accuracies": projected,
                "public_cube_projected_median_accuracy": float(np.median(projected)),
                "projected_and_bias_recalibrated_accuracies": projected_recalibrated,
                "projected_and_bias_recalibrated_median_accuracy": float(np.median(projected_recalibrated)),
            }
        )
        print(
            f"[blob-poisson] epsilon={target:g} raw={np.median(raw):.4f} "
            f"public-bias={np.median(recalibrated):.4f}",
            flush=True,
        )

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "complete",
            "experiment": "blobs_poisson_dpsgd_native_pld_stress_test",
            "privacy": {"delta": 1e-5, "adjacency": "replace-one", "q": q},
            "setting": {
                "N": N,
                "parameters": PARAMETERS,
                "expected_batch_size": BATCH_SIZE,
                "steps": STEPS,
                "learning_rate": LR,
                "coordinate_clip": GAMMA,
                "seeds": args.seeds,
                "public_validation_size": public_size,
                "heldout_test_size": len(test_labels),
            },
            "accounting": "native replace-one Poisson-sampled Gaussian PLD",
            "comparison_caveat": (
                "This is the standard amplified DP-SGD stress test, not the exactly "
                "matched fixed-cyclic optimizer used by AGS and the primary DP arm."
            ),
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[blob-poisson] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
