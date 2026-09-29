"""PCA-MNIST: one AGS release against per-step DP-SGD, in a common RDP accounting.

Both arms over the same 40,000-step full-batch trajectory, so the comparison isolates the mechanism:
AGS pays once for the final parameter vector, DP-SGD pays at every step.

The sensitivity here is the closed form for the clipped mean update, which is sound only if that
update is non-expansive, and the repository does not yet prove it for the elementwise-clipped map.
This panel is therefore a conditional study, and `run_early_stop_strong.py` is the unconditional one to
read beside it.
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
from ags.mnist.ags import sum_lr
from common.accounting.artifacts import write_immutable_json
from common.accounting.laplace_rdp import epsilon_from_rdp, laplace_renyi


ORDERS = np.r_[np.linspace(1.01, 2.0, 40), np.arange(2.25, 128.25, 0.25)]


def calibrate_laplace(target, delta, coordinates):
    def privacy(eps_coord):
        rdp = coordinates * np.asarray([laplace_renyi(float(order), eps_coord, 1.0) for order in ORDERS])
        return epsilon_from_rdp(ORDERS, rdp, delta)

    low, high = 1e-7, 1.0
    while privacy(high)[0] < target:
        high *= 2
    for _ in range(50):
        mid = (low + high) / 2
        if privacy(mid)[0] <= target:
            low = mid
        else:
            high = mid
    return low, privacy(low)


def calibrate_gaussian(target, delta, steps):
    def privacy(multiplier):
        # Full participation and replace-one: normalized shift is two.
        return epsilon_from_rdp(ORDERS, 2.0 * ORDERS * steps / (multiplier**2), delta)

    low, high = 1e-4, 1.0
    while privacy(high)[0] > target:
        high *= 2
    for _ in range(50):
        mid = (low + high) / 2
        if privacy(mid)[0] > target:
            low = mid
        else:
            high = mid
    return high, privacy(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 4, 5, 8, 10, 20, 50, 100])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--scale-unit-norm", action="store_true", help="Rescale PCA features to max training l2 norm one"
    )
    args = ap.parse_args()
    started = time.perf_counter()
    N, d, ntest, batch, steps = 2000, 64, 1000, 2000, 40000
    lr, eta, gamma = 0.05, 0.01, 0.1
    p = N_CLASSES * d + N_CLASSES
    root = Path("ags/mnist/data/mnist")
    # Keep this explicit: the historical PCA-MNIST win uses whitened PCA
    # coordinates without the optional unit-norm rescaling.  Relying on the
    # loader default previously made this RDP panel an unmatched experiment.
    X, y, Xte, yte = load_mnist_pca(
        N,
        d,
        seed=42,
        n_test=ntest,
        root=root,
        scale_unit_norm=args.scale_unit_norm,
    )
    device = torch.device(args.device)
    theta = train(X, y, n_s=steps, lr=lr, gamma=gamma, lam=0, b=batch, lr_decay=eta, device=device)
    nominal_accuracy = accuracy(theta, Xte, yte)
    sensitivity = 2 * gamma * sum_lr(lr, steps, eta) / batch

    rows = []
    for target in args.epsilons:
        eps_coord, (ags_eps, ags_order) = calibrate_laplace(target, args.delta, p)
        scale = sensitivity / eps_coord
        ags_acc = [
            accuracy(theta + np.random.RandomState(seed).laplace(0, scale, p), Xte, yte)
            for seed in range(args.seeds)
        ]

        multiplier, (dp_eps, dp_order) = calibrate_gaussian(target, args.delta, steps)
        std = multiplier * gamma * math.sqrt(p) / batch
        dp_acc = []
        seed_times = []
        for seed in range(args.seeds):
            tick = time.perf_counter()
            noise = np.random.RandomState(seed).normal(0, std, size=(steps, p))
            released = train(
                X, y, n_s=steps, lr=lr, gamma=gamma, lam=0, b=batch, noise=noise, lr_decay=eta, device=device
            )
            dp_acc.append(accuracy(released, Xte, yte))
            seed_times.append(time.perf_counter() - tick)
            print(f"[pca-rdp] eps={target:g} seed={seed} DP={dp_acc[-1]:.4f}", flush=True)
        rows.append(
            {
                "target_epsilon": target,
                "ags": {
                    "epsilon_coordinate": eps_coord,
                    "achieved_epsilon": ags_eps,
                    "minimizing_order": ags_order,
                    "sensitivity": sensitivity,
                    "noise_scale": scale,
                    "accuracies": ags_acc,
                    "median_accuracy": float(np.median(ags_acc)),
                },
                "dpsgd": {
                    "noise_multiplier": multiplier,
                    "achieved_epsilon": dp_eps,
                    "minimizing_order": dp_order,
                    "gradient_noise_std": std,
                    "accuracies": dp_acc,
                    "median_accuracy": float(np.median(dp_acc)),
                    "seed_runtime_seconds": seed_times,
                },
            }
        )
        print(
            f"[pca-rdp-result] eps={target:g} AGS={np.median(ags_acc):.4f} DP={np.median(dp_acc):.4f}",
            flush=True,
        )

    payload = {
        "schema": 1,
        "status": "analytic_rdp_conditional_sensitivity_certificate",
        "experiment": "pca_mnist_one_release_ags_vs_perstep_dpsgd_matched_rdp",
        "privacy": {"delta": args.delta, "adjacency": "replace-one", "q": 1.0},
        "setting": {
            "N": N,
            "test_N": ntest,
            "pca_dimensions": d,
            "parameters": p,
            "batch_size": batch,
            "steps": steps,
            "learning_rate": lr,
            "lr_decay": eta,
            "coordinate_clip": gamma,
            "seeds": args.seeds,
            "scale_unit_norm": args.scale_unit_norm,
            "pca_private_fit_public_preprocessing_assumption": True,
        },
        "accounting": "common analytic RDP composition and common conversion",
        "sensitivity_caveat": (
            "The closed-form final-coordinate sensitivity assumes the clipped update "
            "map is non-expansive. The repository checks the unclipped softmax "
            "smoothness condition, but a proof covering elementwise per-example "
            "gradient clamping is still required before treating this PCA panel as "
            "fully certified."
        ),
        "ags_release": "fixed-scale coordinate Laplace after final training step",
        "dpsgd_release": "Gaussian gradient release at every step",
        "nominal_accuracy": nominal_accuracy,
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    digest = write_immutable_json(args.output, payload)
    print(f"[pca-rdp] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
