"""Exploratory tail-averaging improvement for matched PCA-MNIST mechanisms.

Both arms use the same public averaging windows.  AGS releases only the clean
tail average and calibrates to its weighted non-expansive sensitivity bound;
DP-SGD averages its noisy internal iterates as privacy-free postprocessing.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from ags.mnist.shared import N_CLASSES, accuracy, load_mnist_pca
from ags.mnist.torch_sgd import train
from common.accounting.artifacts import write_immutable_json
from ags.mnist.accounting.pca.run_matched_rdp import (
    calibrate_gaussian,
    calibrate_laplace,
)


def tail_sensitivities(lr, decay, steps, windows, gamma, batch):
    cumulative = np.cumsum([lr / (1.0 + decay * step) for step in range(steps)])
    return {int(window): float(2 * gamma / batch * cumulative[-int(window) :].mean()) for window in windows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 4, 8, 10, 20])
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 1000, 5000, 10000, 20000, 40000])
    ap.add_argument("--seeds", type=int, default=21)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    started = time.perf_counter()
    N, d, ntest, batch, steps = 2000, 64, 1000, 2000, 40000
    lr, decay, gamma = 0.05, 0.01, 0.1
    p = N_CLASSES * d + N_CLASSES
    windows = sorted(set(args.windows))
    root = Path("ags/mnist/data/mnist")
    X, y, Xte, yte = load_mnist_pca(N, d, seed=42, n_test=ntest, root=root, scale_unit_norm=False)
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
    nominal = {str(window): accuracy(clean_averages[window], Xte, yte) for window in windows}
    rows = []
    for target in args.epsilons:
        eps_coord, (ags_eps, ags_order) = calibrate_laplace(target, args.delta, p)
        ags_windows = {}
        for window in windows:
            scale = sensitivities[window] / eps_coord
            values = [
                accuracy(
                    clean_averages[window] + np.random.RandomState(seed).laplace(0, scale, p),
                    Xte,
                    yte,
                )
                for seed in range(args.seeds)
            ]
            ags_windows[str(window)] = {
                "sensitivity": sensitivities[window],
                "noise_scale": scale,
                "accuracies": values,
                "median_accuracy": float(np.median(values)),
            }

        multiplier, (dp_eps, dp_order) = calibrate_gaussian(target, args.delta, steps)
        std = multiplier * gamma * math.sqrt(p) / batch
        dp_values = {str(window): [] for window in windows}
        seed_times = []
        for seed in range(args.seeds):
            tick = time.perf_counter()
            noise = np.random.RandomState(seed).normal(0, std, size=(steps, p))
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
                dp_values[str(window)].append(accuracy(noisy_averages[window], Xte, yte))
            seed_times.append(time.perf_counter() - tick)
            print(
                f"[pca-tail] eps={target:g} seed={seed} "
                + " ".join(f"w{window}={dp_values[str(window)][-1]:.4f}" for window in windows),
                flush=True,
            )
        dp_windows = {
            key: {"accuracies": values, "median_accuracy": float(np.median(values))}
            for key, values in dp_values.items()
        }
        rows.append(
            {
                "target_epsilon": target,
                "ags": {
                    "epsilon_coordinate": eps_coord,
                    "achieved_epsilon": ags_eps,
                    "minimizing_order": ags_order,
                    "windows": ags_windows,
                },
                "dpsgd": {
                    "noise_multiplier": multiplier,
                    "achieved_epsilon": dp_eps,
                    "minimizing_order": dp_order,
                    "gradient_noise_std": std,
                    "windows": dp_windows,
                    "seed_runtime_seconds": seed_times,
                },
            }
        )
    payload = {
        "schema": 1,
        "status": "exploratory_conditional_nonexpansive_sensitivity",
        "experiment": "pca_mnist_tail_average_matched_rdp",
        "privacy": {"delta": args.delta, "adjacency": "replace-one", "q": 1.0},
        "setting": {
            "N": N,
            "test_N": ntest,
            "pca_dimensions": d,
            "parameters": p,
            "batch_size": batch,
            "steps": steps,
            "learning_rate": lr,
            "lr_decay": decay,
            "coordinate_clip": gamma,
            "scale_unit_norm": False,
            "windows": windows,
            "seeds": args.seeds,
        },
        "accounting": "common analytic RDP composition and common conversion",
        "sensitivity_caveat": (
            "Each AGS tail-average certificate is the average of the per-iterate "
            "closed-form bounds and assumes the elementwise-clipped update map is "
            "non-expansive, the same unresolved repository-specific condition as "
            "the final-iterate PCA panel."
        ),
        "selection_caveat": (
            "All public averaging windows are retained; choosing the best window "
            "from these test results is exploratory, not a confirmatory protocol."
        ),
        "nominal_final_accuracy": accuracy(final, Xte, yte),
        "nominal_tail_average_accuracies": nominal,
        "tail_sensitivities": {str(k): v for k, v in sensitivities.items()},
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    digest = write_immutable_json(args.output, payload)
    print(f"[pca-tail] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
