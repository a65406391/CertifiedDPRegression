"""PCA-MNIST over short trajectories, where the sensitivity is exact.

One to fifty steps, where the one-step sensitivity needs no non-expansiveness assumption, so the
comparison is unconditional. It reverses the long-trajectory result: DP-SGD wins at every epsilon
tested.

That reversal is why the long-trajectory panel is labelled conditional. Both belong in any honest
account of the PCA-MNIST arm.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from ags.mnist.ags import sum_lr
from ags.mnist.shared import N_CLASSES, accuracy, load_mnist_pca
from ags.mnist.torch_sgd import train
from common.accounting.artifacts import write_immutable_json
from common.accounting.exact_accounting import (
    calibrate_composed_gaussian,
    calibrate_composed_laplace,
    composed_laplace_epsilon,
)
from ags.mnist.accounting.pca.run_strong_accounting import (
    public_class_bias_recalibration,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 4, 8, 10, 20])
    parser.add_argument("--seeds", type=int, default=101)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--pld-interval", type=float, default=1e-4)
    args = parser.parse_args()
    started = time.perf_counter()
    n, dimensions, ntest, batch = 2000, 64, 2000, 2000
    lr, decay, gamma = 0.05, 0.01, 0.1
    parameters = N_CLASSES * dimensions + N_CLASSES
    X, y, Xte, yte = load_mnist_pca(
        n,
        dimensions,
        seed=42,
        n_test=ntest,
        root=Path("ags/mnist/data/mnist"),
        scale_unit_norm=False,
    )
    public_size = 1000
    X_public, y_public = Xte[:public_size], yte[:public_size]
    X_heldout, y_heldout = Xte[public_size:], yte[public_size:]
    device = torch.device("cpu")

    laplace_calibrations = {}
    for target in args.epsilons:
        parameter, achieved = calibrate_composed_laplace(
            target, args.delta, parameters, interval=args.pld_interval
        )
        audit = composed_laplace_epsilon(
            parameter * (1 - 2e-6),
            parameters,
            args.delta,
            interval=args.pld_interval / 2,
        )
        if audit > target:
            parameter, achieved = calibrate_composed_laplace(
                target,
                args.delta,
                parameters,
                interval=args.pld_interval / 2,
            )
            audit = achieved
        laplace_calibrations[target] = (parameter, achieved, audit)

    rows = []
    for steps in args.steps:
        nominal = train(
            X,
            y,
            n_s=steps,
            lr=lr,
            gamma=gamma,
            lam=0,
            b=batch,
            lr_decay=decay,
            device=device,
        )
        sensitivity = 2 * gamma * sum_lr(lr, steps, decay) / batch
        epsilon_rows = []
        for target in args.epsilons:
            eps_coord, ags_epsilon, half_grid = laplace_calibrations[target]
            scale = sensitivity / eps_coord
            ags_releases = [
                nominal + np.random.RandomState(seed).laplace(0, scale, parameters)
                for seed in range(args.seeds)
            ]
            ags_raw = [accuracy(value, X_heldout, y_heldout) for value in ags_releases]
            ags_recalibrated = [
                accuracy(
                    public_class_bias_recalibration(value, X_public, y_public),
                    X_heldout,
                    y_heldout,
                )
                for value in ags_releases
            ]

            multiplier, achieved_delta = calibrate_composed_gaussian(target, args.delta, steps)
            gradient_std = multiplier * gamma * math.sqrt(parameters) / batch
            dp_raw, dp_recalibrated = [], []
            for seed in range(args.seeds):
                noise = np.random.RandomState(seed).normal(0, gradient_std, size=(steps, parameters))
                released = train(
                    X,
                    y,
                    n_s=steps,
                    lr=lr,
                    gamma=gamma,
                    lam=0,
                    b=batch,
                    noise=noise,
                    lr_decay=decay,
                    device=device,
                )
                dp_raw.append(accuracy(released, X_heldout, y_heldout))
                dp_recalibrated.append(
                    accuracy(
                        public_class_bias_recalibration(released, X_public, y_public),
                        X_heldout,
                        y_heldout,
                    )
                )
            epsilon_rows.append(
                {
                    "target_epsilon": target,
                    "ags": {
                        "epsilon_coordinate": eps_coord,
                        "achieved_epsilon_pessimistic_pld": ags_epsilon,
                        "half_grid_achieved_epsilon": half_grid,
                        "sensitivity": sensitivity,
                        "noise_scale": scale,
                        "accuracies": ags_raw,
                        "median_accuracy": float(np.median(ags_raw)),
                        "public_class_bias_recalibrated_accuracies": ags_recalibrated,
                        "public_class_bias_recalibrated_median_accuracy": float(np.median(ags_recalibrated)),
                    },
                    "dpsgd": {
                        "noise_multiplier": multiplier,
                        "achieved_delta_at_target_epsilon": achieved_delta,
                        "gradient_noise_std": gradient_std,
                        "accuracies": dp_raw,
                        "median_accuracy": float(np.median(dp_raw)),
                        "public_class_bias_recalibrated_accuracies": dp_recalibrated,
                        "public_class_bias_recalibrated_median_accuracy": float(np.median(dp_recalibrated)),
                    },
                }
            )
            print(
                f"[pca-early] steps={steps} epsilon={target:g} "
                f"AGS={np.median(ags_raw):.3f} DP={np.median(dp_raw):.3f}",
                flush=True,
            )
        rows.append(
            {
                "steps": steps,
                "sensitivity_status": (
                    "exact_unconditional_one_step_replacement_sensitivity"
                    if steps == 1
                    else "conditional_nonexpansive_sensitivity"
                ),
                "nominal_accuracy": accuracy(nominal, X_heldout, y_heldout),
                "nominal_public_class_bias_recalibrated_accuracy": accuracy(
                    public_class_bias_recalibration(nominal, X_public, y_public),
                    X_heldout,
                    y_heldout,
                ),
                "privacy_rows": epsilon_rows,
            }
        )

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "complete_with_per_step_sensitivity_status",
            "experiment": "pca_mnist_early_stop_strong_accounting",
            "privacy": {"delta": args.delta, "adjacency": "replace-one", "q": 1.0},
            "setting": {
                "N": n,
                "test_N": len(y_heldout),
                "public_validation_N": public_size,
                "pca_dimensions": dimensions,
                "parameters": parameters,
                "batch_size": batch,
                "learning_rate": lr,
                "lr_decay": decay,
                "coordinate_clip": gamma,
                "scale_unit_norm": False,
                "steps_grid": args.steps,
                "seeds": args.seeds,
                "pca_private_fit_public_preprocessing_assumption": True,
            },
            "accounting": {
                "ags": "native coordinate-Laplace pessimistic PLD composition",
                "dpsgd": "exact composed Gaussian privacy-loss tradeoff",
                "laplace_pld_interval": args.pld_interval,
            },
            "public_postprocessing": (
                "A positive logit temperature and additive class intercepts are fit "
                "on the disjoint labeled public validation split. Legacy JSON field "
                "names retain the shorter public_class_bias_recalibrated label."
            ),
            "selection_caveat": (
                "The step grid follows an exploratory audit of existing results. "
                "Every step count is retained and must not be presented as a "
                "confirmatory test-selected configuration."
            ),
            "sensitivity_caveat": (
                "The one-step certificate is exact without a non-expansiveness "
                "assumption. Multi-step rows retain the clipped-update "
                "non-expansiveness caveat from the long PCA panel."
            ),
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[pca-early] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
