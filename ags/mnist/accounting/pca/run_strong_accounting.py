"""PCA-MNIST comparison with mechanism-specific strong accountants.

Coordinate Laplace releases use a pessimistically discretized native PLD.
Gaussian DP-SGD uses the closed-form Gaussian privacy-loss composition, which
is the zero-grid-error limit of its native PLD accountant.  The same public
tail-averaging windows are applied to both mechanisms.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from ags.mnist.shared import N_CLASSES, accuracy, load_mnist_pca, logits
from ags.mnist.torch_sgd import train
from common.accounting.artifacts import write_immutable_json
from common.accounting.exact_accounting import (
    calibrate_composed_gaussian,
    calibrate_composed_laplace,
    composed_laplace_epsilon,
)
from ags.mnist.accounting.pca.run_tail_average_rdp import tail_sensitivities


def public_class_bias_recalibration(theta, X, y):
    """Fit a positive logit temperature and class intercepts on public data."""
    base = logits(theta, X)
    rows = np.arange(len(y))

    def objective(parameters):
        temperature = np.exp(parameters[0])
        bias = parameters[1:]
        shifted = temperature * base + bias
        normalizer = logsumexp(shifted, axis=1)
        loss = float(np.mean(normalizer - shifted[rows, y]))
        probabilities = np.exp(shifted - normalizer[:, None])
        probabilities[rows, y] -= 1.0
        gradient_bias = probabilities.mean(axis=0)
        gradient_log_temperature = float(np.mean(np.sum(probabilities * (temperature * base), axis=1)))
        return loss, np.r_[gradient_log_temperature, gradient_bias]

    result = minimize(
        objective,
        np.zeros(N_CLASSES + 1),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0)] + [(None, None)] * N_CLASSES,
        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"public class-bias calibration failed: {result.message}")
    adjusted = np.asarray(theta).copy() * np.exp(result.x[0])
    adjusted[-N_CLASSES:] += result.x[1:]
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 4, 5, 8, 10, 20, 50, 100])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 1000, 5000, 10000, 20000, 40000])
    parser.add_argument("--seeds", type=int, default=41)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--pld-interval", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    started = time.perf_counter()

    n, dimensions, ntest, batch, steps = 2000, 64, 2000, 2000, 40000
    lr, decay, gamma = 0.05, 0.01, 0.1
    parameters = N_CLASSES * dimensions + N_CLASSES
    windows = sorted(set(args.windows))
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
    device = torch.device(args.device)
    final, clean_averages = train(
        X,
        y,
        n_s=steps,
        lr=lr,
        gamma=gamma,
        lam=0,
        b=batch,
        lr_decay=decay,
        device=device,
        tail_average_windows=windows,
    )
    sensitivities = tail_sensitivities(lr, decay, steps, windows, gamma, batch)
    nominal = {
        str(window): {
            "raw": accuracy(clean_averages[window], X_heldout, y_heldout),
            "public_class_bias_recalibrated": accuracy(
                public_class_bias_recalibration(clean_averages[window], X_public, y_public),
                X_heldout,
                y_heldout,
            ),
        }
        for window in windows
    }

    rows = []
    completed_seeds = 0
    total_seeds = len(args.epsilons) * args.seeds
    for target in args.epsilons:
        epsilon_coordinate, ags_epsilon = calibrate_composed_laplace(
            target, args.delta, parameters, interval=args.pld_interval
        )
        # A half-grid convergence audit is retained independently.  Shrinking
        # by two ppm makes the audit a feasibility check, not a retuning step.
        audit_parameter = epsilon_coordinate * (1.0 - 2e-6)
        half_grid_epsilon = composed_laplace_epsilon(
            audit_parameter,
            parameters,
            args.delta,
            interval=args.pld_interval / 2.0,
        )
        if half_grid_epsilon > target:
            epsilon_coordinate, ags_epsilon = calibrate_composed_laplace(
                target,
                args.delta,
                parameters,
                interval=args.pld_interval / 2.0,
            )
            audit_parameter = epsilon_coordinate
            half_grid_epsilon = ags_epsilon
        ags_windows = {}
        for window in windows:
            scale = sensitivities[window] / epsilon_coordinate
            released = [
                clean_averages[window] + np.random.RandomState(seed).laplace(0, scale, parameters)
                for seed in range(args.seeds)
            ]
            values = [accuracy(value, X_heldout, y_heldout) for value in released]
            recalibrated = [
                accuracy(
                    public_class_bias_recalibration(value, X_public, y_public),
                    X_heldout,
                    y_heldout,
                )
                for value in released
            ]
            ags_windows[str(window)] = {
                "sensitivity": sensitivities[window],
                "noise_scale": scale,
                "accuracies": values,
                "median_accuracy": float(np.median(values)),
                "public_class_bias_recalibrated_accuracies": recalibrated,
                "public_class_bias_recalibrated_median_accuracy": float(np.median(recalibrated)),
            }

        multiplier, achieved_delta = calibrate_composed_gaussian(target, args.delta, steps)
        std = multiplier * gamma * math.sqrt(parameters) / batch
        dp_values = {str(window): [] for window in windows}
        dp_recalibrated = {str(window): [] for window in windows}
        seed_times = []
        for seed in range(args.seeds):
            tick = time.perf_counter()
            noise = np.random.RandomState(seed).normal(0, std, size=(steps, parameters))
            _, noisy_averages = train(
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
                tail_average_windows=windows,
            )
            for window in windows:
                dp_values[str(window)].append(accuracy(noisy_averages[window], X_heldout, y_heldout))
                dp_recalibrated[str(window)].append(
                    accuracy(
                        public_class_bias_recalibration(noisy_averages[window], X_public, y_public),
                        X_heldout,
                        y_heldout,
                    )
                )
            seed_times.append(time.perf_counter() - tick)
            completed_seeds += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_seeds * (total_seeds - completed_seeds)
            print(
                f"[pca-strong] epsilon={target:g} seed={seed}/{args.seeds - 1} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        rows.append(
            {
                "target_epsilon": target,
                "ags": {
                    "epsilon_coordinate": epsilon_coordinate,
                    "achieved_epsilon_pessimistic_pld": ags_epsilon,
                    "half_grid_audit_parameter": audit_parameter,
                    "half_grid_achieved_epsilon": half_grid_epsilon,
                    "windows": ags_windows,
                },
                "dpsgd": {
                    "noise_multiplier": multiplier,
                    "achieved_delta_at_target_epsilon": achieved_delta,
                    "gradient_noise_std": std,
                    "windows": {
                        key: {
                            "accuracies": values,
                            "median_accuracy": float(np.median(values)),
                            "public_class_bias_recalibrated_accuracies": dp_recalibrated[key],
                            "public_class_bias_recalibrated_median_accuracy": float(
                                np.median(dp_recalibrated[key])
                            ),
                        }
                        for key, values in dp_values.items()
                    },
                    "seed_runtime_seconds": seed_times,
                },
            }
        )
        print(
            f"[pca-strong-result] epsilon={target:g} "
            + " ".join(
                f"w{window}:AGS={ags_windows[str(window)]['median_accuracy']:.3f}"
                f"/DP={np.median(dp_values[str(window)]):.3f}"
                for window in windows
            ),
            flush=True,
        )

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "strong_accounting_conditional_sensitivity_certificate",
            "experiment": "pca_mnist_tail_average_strong_accounting",
            "privacy": {"delta": args.delta, "adjacency": "replace-one", "q": 1.0},
            "setting": {
                "N": n,
                "test_N": len(y_heldout),
                "public_validation_N": public_size,
                "pca_dimensions": dimensions,
                "parameters": parameters,
                "batch_size": batch,
                "steps": steps,
                "learning_rate": lr,
                "lr_decay": decay,
                "coordinate_clip": gamma,
                "scale_unit_norm": False,
                "windows": windows,
                "seeds": args.seeds,
                "pca_private_fit_public_preprocessing_assumption": True,
            },
            "accounting": {
                "ags": "native coordinate-Laplace pessimistic PLD composition",
                "dpsgd": "exact composed Gaussian privacy-loss tradeoff",
                "laplace_pld_interval": args.pld_interval,
            },
            "sensitivity_caveat": (
                "The closed-form final/tail-average sensitivity assumes the "
                "elementwise-clipped update is non-expansive. The repository does "
                "not yet contain a complete proof for that clipped map."
            ),
            "selection_caveat": (
                "Every public window is retained; choosing a window from heldout "
                "test accuracy is exploratory, not a confirmatory protocol."
            ),
            "public_postprocessing": (
                "A positive logit temperature and additive class intercepts are fit "
                "on the disjoint labeled public validation split and applied "
                "identically to each released model. Legacy JSON field names retain "
                "the shorter public_class_bias_recalibrated label."
            ),
            "nominal_final_accuracy": accuracy(final, X_heldout, y_heldout),
            "nominal_tail_average_accuracies": nominal,
            "tail_sensitivities": {str(k): v for k, v in sensitivities.items()},
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[pca-strong] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
