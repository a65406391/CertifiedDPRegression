"""Strong Gaussian-PLD DP-SGD reference for the matched Blobs panel.

This small companion keeps the mechanism and data split identical to
``run_rdp_comparison.py`` while giving Gaussian DP-SGD its native PLD
accountant.  Public intercept recalibration is deterministic postprocessing of
the one released model and is applied exactly as in the AGS panel.
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
    PRIVACY_STEPS,
    SEP,
    STEPS,
)
from ags.blobs.accounting.run_rdp_comparison import public_bias_recalibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=101)
    parser.add_argument("--interval", type=float, default=1e-3)
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    )
    args = parser.parse_args()
    started = time.perf_counter()

    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    design = np.c_[X, np.ones(N)]
    public_size = 1000
    public_design = np.c_[Xte[:public_size], np.ones(public_size)]
    public_labels = yte[:public_size]
    test_design = np.c_[Xte[public_size:], np.ones(len(Xte) - public_size)]
    test_labels = yte[public_size:]

    rows = []
    for target in args.epsilons:
        calibration = calibrate_dpsgd(
            target,
            1e-5,
            q=1.0,
            steps=PRIVACY_STEPS,
            interval=args.interval,
            search_iterations=18,
        )
        gradient_std = calibration.mechanism_parameter * GAMMA * math.sqrt(PARAMETERS) / BATCH_SIZE
        raw, recalibrated = [], []
        for seed in range(args.seeds):
            released = B.train_hinge(
                design,
                y,
                n_s=STEPS,
                lr=LR,
                gamma=GAMMA,
                lam=0.0,
                b=BATCH_SIZE,
                noise_scale=gradient_std,
                rng=np.random.RandomState(seed),
                noise_kind="gauss",
            )
            raw.append(B.acc(released, test_design, test_labels))
            adjusted = public_bias_recalibration(released, public_design, public_labels)
            recalibrated.append(B.acc(adjusted, test_design, test_labels))
        rows.append(
            {
                "target_epsilon": target,
                "calibration": calibration.__dict__,
                "gradient_noise_std": gradient_std,
                "accuracies": raw,
                "median_accuracy": float(np.median(raw)),
                "public_bias_recalibrated_accuracies": recalibrated,
                "public_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
            }
        )
        print(
            f"[blob-dp-pld] epsilon={target:g} raw={np.median(raw):.4f} "
            f"public-bias={np.median(recalibrated):.4f}",
            flush=True,
        )

    write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "complete",
            "experiment": "blobs_dpsgd_native_pld_reference",
            "privacy": {"delta": 1e-5, "adjacency": "replace-one", "q": 1.0},
            "setting": {
                "N": N,
                "parameters": PARAMETERS,
                "batch_size": BATCH_SIZE,
                "steps": STEPS,
                "record_touching_privacy_steps": PRIVACY_STEPS,
                "learning_rate": LR,
                "coordinate_clip": GAMMA,
                "seeds": args.seeds,
                "public_validation_size": public_size,
                "heldout_test_size": len(test_labels),
            },
            "accounting": (
                "Native Gaussian PLD under replace-one adjacency. The deterministic "
                "cyclic batches form disjoint groups within each epoch, so adaptive "
                "parallel composition gives one record-touching release per epoch."
            ),
            "postprocessing": (
                "The intercept-only recalibration uses a disjoint labeled public "
                "validation set and is privacy-free deterministic postprocessing."
            ),
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
