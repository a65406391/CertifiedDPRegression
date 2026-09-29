# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Corrected one-shot PCA-64 AGS and tightly-accounted matched DP-SGD.

Each invocation writes one immutable JSON artifact so epsilon arms can run in
parallel and interrupted work remains usable.  The PCA fit intentionally matches
the historical experiment and therefore retains its documented public-
preprocessing assumption.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ags.mnist import release as REL
from ags.mnist.ags import agt_bound, sum_lr
from ags.mnist.shared import N_CLASSES, accuracy, load_mnist_pca, logistic_model
from ags.mnist.torch_sgd import train
from common.accounting.accounting import calibrate_ags, calibrate_dpsgd


def immutable_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def data(args):
    data_root = Path(__file__).resolve().parent / "data" / "mnist"
    return load_mnist_pca(
        args.N,
        args.d,
        seed=args.data_seed,
        n_test=args.n_test,
        root=data_root,
        scale_unit_norm=args.scale_unit_norm,
    )


def nominal(args, X, y, device):
    return train(
        X,
        y,
        n_s=args.steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=0.0,
        b=args.batch_size,
        lr_decay=args.eta,
        device=device,
    )


def run_release(args) -> None:
    started = time.perf_counter()
    device = torch.device(args.device)
    X, y, Xte, yte = data(args)
    p = N_CLASSES * args.d + N_CLASSES
    theta = nominal(args, X, y, device)
    nominal_accuracy = accuracy(theta, Xte, yte)
    cumulative_lr = sum_lr(args.lr, args.steps, args.eta)
    certificate = 2.0 * args.gamma * cumulative_lr / args.batch_size
    local = {k: np.full(p, certificate * k) for k in args.k_grid}

    historical = {}
    for mode in ("pure", "approx"):
        beta = (
            REL.beta_for(args.epsilon)
            if mode == "pure"
            else REL.beta_for(args.epsilon, mode="approx", n_params=p, delta=args.delta)
        )
        ss = REL.R.smooth_sensitivity(local, args.k_grid, np.full(p, certificate), beta, p)
        scale = REL.sigma(ss, args.epsilon, p, noise_mode="anisotropic", mode=mode)
        values = [
            accuracy(REL.release(theta, scale, np.random.RandomState(seed), mode=mode), Xte, yte)
            for seed in range(args.seeds)
        ]
        historical[mode] = {
            "accuracies": values,
            "median_accuracy": float(np.median(values)),
            "ss_quantiles": np.quantile(ss, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "scale_quantiles": np.quantile(scale, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        }

    calibration = calibrate_ags(
        args.epsilon,
        args.delta,
        coordinates=p,
        steps=1,
        q=1.0,
        mechanism="laplace",
        interval=args.ags_interval,
        delta_fraction=args.ags_delta_fraction,
    )
    # The non-expansive certificate bounds every neighboring final-model pair,
    # independently of radius. Consequently max_k exp(-beta*k) Abar^(k) is k=0.
    corrected_ss = np.full(p, certificate)
    corrected_scale = corrected_ss / calibration.mechanism_parameter
    corrected_values = [
        accuracy(theta + np.random.RandomState(seed).laplace(0.0, corrected_scale), Xte, yte)
        for seed in range(args.seeds)
    ]
    immutable_json(
        Path(args.output),
        {
            "schema": 1,
            "arm": "one_shot_ags",
            "setting": vars(args),
            "pca_private_fit_public_preprocessing_assumption": True,
            "parameters": p,
            "sum_learning_rates": cumulative_lr,
            "nonexpansive_neighbor_certificate": certificate,
            "nominal_accuracy": nominal_accuracy,
            "historical": historical,
            "corrected_generic_pld": {
                "calibration": calibration.__dict__,
                "accuracies": corrected_values,
                "median_accuracy": float(np.median(corrected_values)),
                "ss_quantiles": np.quantile(corrected_ss, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                "scale_quantiles": np.quantile(corrected_scale, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            },
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def run_dpsgd(args) -> None:
    started = time.perf_counter()
    device = torch.device(args.device)
    X, y, Xte, yte = data(args)
    p = N_CLASSES * args.d + N_CLASSES
    theta = nominal(args, X, y, device)
    calibration = calibrate_dpsgd(
        args.epsilon,
        args.delta,
        q=1.0,
        steps=args.steps,
        interval=args.dp_interval,
        search_iterations=args.dp_search_iterations,
    )
    # The replace-one PLD already models the two-record shift.  Its Gaussian
    # noise multiplier is relative to the per-record norm bound C.
    sensitivity = args.gamma * math.sqrt(p) / args.batch_size
    std = calibration.mechanism_parameter * sensitivity
    values = []
    run_times = []
    for seed in range(args.seeds):
        tick = time.perf_counter()
        noise = np.random.RandomState(seed).normal(0.0, std, size=(args.steps, p))
        released = train(
            X,
            y,
            n_s=args.steps,
            lr=args.lr,
            gamma=args.gamma,
            lam=0.0,
            b=args.batch_size,
            noise=noise,
            lr_decay=args.eta,
            device=device,
        )
        values.append(accuracy(released, Xte, yte))
        run_times.append(time.perf_counter() - tick)
        print(
            f"[pca-dpsgd] epsilon={args.epsilon} seed={seed} "
            f"accuracy={values[-1]:.4f} elapsed={run_times[-1]:.1f}s",
            flush=True,
        )
    immutable_json(
        Path(args.output),
        {
            "schema": 1,
            "arm": "matched_coordinate_clipped_gaussian_dpsgd",
            "setting": vars(args),
            "pca_private_fit_public_preprocessing_assumption": True,
            "parameters": p,
            "nominal_accuracy": accuracy(theta, Xte, yte),
            "calibration": calibration.__dict__,
            "mean_gradient_l2_sensitivity": sensitivity,
            "mean_gradient_noise_std": std,
            "accuracies": values,
            "median_accuracy": float(np.median(values)),
            "seed_runtime_seconds": run_times,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def run_certificate(args) -> None:
    started = time.perf_counter()
    X, y, _, _ = data(args)
    lower, center, upper = agt_bound(
        logistic_model(args.d),
        X,
        y,
        k=args.radius,
        b=args.batch_size,
        n_epochs=args.steps,
        lr=args.lr,
        gamma=args.gamma,
        l2_reg=0.0,
        loss="cross_entropy",
        lr_decay=args.eta,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"immutable artifact already exists: {output}")
    np.savez_compressed(
        output,
        lower=lower,
        center=center,
        upper=upper,
        metadata=json.dumps({"radius": args.radius, "runtime_seconds": time.perf_counter() - started}),
    )
    print(
        f"[pca-certificate] k={args.radius} width_mean={(upper - lower).mean():.6g} "
        f"runtime={time.perf_counter() - started:.1f}s",
        flush=True,
    )


def _per_step_train(args, X, y, Xte, yte, calibration, *, arm, seed, device):
    """Fast analytic per-example gradients for the PCA multinomial head."""
    p = N_CLASSES * args.d + N_CLASSES
    xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    xte = torch.as_tensor(Xte, dtype=torch.float64, device=device)
    yte_t = torch.as_tensor(yte, dtype=torch.long, device=device)
    theta = torch.zeros(p, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + (1000 if arm == "ags" else 2000))
    beta = calibration.mechanism_parameter * args.beta_epsilon_ratio
    # As above, REPLACE_ONE is handled by the accountant, not by doubling C.
    gaussian_sensitivity = args.gamma * math.sqrt(p) / args.batch_size
    started = time.perf_counter()
    report_every = max(1, args.steps // 20)
    for step in range(args.steps):
        lr_t = args.lr / (1.0 + args.eta * step)
        weight = theta[: N_CLASSES * args.d].view(N_CLASSES, args.d)
        bias = theta[N_CLASSES * args.d :]
        dz = torch.softmax(xt @ weight.T + bias, dim=1)
        dz[torch.arange(len(yt), device=device), yt] -= 1.0
        gradients = torch.cat([(dz.unsqueeze(2) * xt.unsqueeze(1)).reshape(len(yt), -1), dz], dim=1).clamp_(
            -args.gamma, args.gamma
        )
        nominal = theta - lr_t * gradients.mean(0)
        if arm == "ags":
            local = lr_t * (args.gamma + gradients.abs().amax(0)) / args.batch_size
            gs = 2.0 * lr_t * args.gamma / args.batch_size
            sensitivity = torch.clamp_min(local, math.exp(-beta) * gs)
            scale = sensitivity / calibration.mechanism_parameter
            uniform = torch.rand(p, dtype=torch.float64, device=device, generator=generator)
            uniform = uniform.clamp(torch.finfo(uniform.dtype).eps, 1 - torch.finfo(uniform.dtype).eps)
            centered = uniform - 0.5
            noise = -scale * torch.sign(centered) * torch.log1p(-2 * centered.abs())
        else:
            std = lr_t * calibration.mechanism_parameter * gaussian_sensitivity
            noise = torch.randn(p, dtype=torch.float64, device=device, generator=generator) * std
        theta = nominal + noise
        if (step + 1) % report_every == 0 or step + 1 == args.steps:
            elapsed = time.perf_counter() - started
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            print(
                f"[pca-perstep-{arm}] seed={seed} step={step + 1}/{args.steps} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    with torch.no_grad():
        weight = theta[: N_CLASSES * args.d].view(N_CLASSES, args.d)
        bias = theta[N_CLASSES * args.d :]
        logits = xte @ weight.T + bias
        acc = float((logits.argmax(1) == yte_t).double().mean().item())
        loss = float(torch.nn.functional.cross_entropy(logits, yte_t).item())
    return acc, loss, time.perf_counter() - started


def run_per_step(args, arm: str) -> None:
    started = time.perf_counter()
    device = torch.device(args.device)
    X, y, Xte, yte = data(args)
    p = N_CLASSES * args.d + N_CLASSES
    if arm == "ags":
        calibration = calibrate_ags(
            args.epsilon,
            args.delta,
            coordinates=p,
            steps=args.steps,
            q=1.0,
            mechanism="laplace",
            interval=args.per_step_ags_interval,
            delta_fraction=args.ags_delta_fraction,
        )
    else:
        calibration = calibrate_dpsgd(
            args.epsilon,
            args.delta,
            q=1.0,
            steps=args.steps,
            interval=args.dp_interval,
            search_iterations=args.dp_search_iterations,
        )
    results = []
    for seed in range(args.seeds):
        acc, loss, runtime = _per_step_train(
            args, X, y, Xte, yte, calibration, arm=arm, seed=seed, device=device
        )
        results.append({"seed": seed, "accuracy": acc, "loss": loss, "runtime_seconds": runtime})
        print(f"[pca-perstep-{arm}-result] seed={seed} accuracy={acc:.4f} loss={loss:.5g}", flush=True)
    immutable_json(
        Path(args.output),
        {
            "schema": 1,
            "arm": f"per_step_{arm}",
            "setting": vars(args),
            "pca_private_fit_public_preprocessing_assumption": True,
            "parameters": p,
            "calibration": calibration.__dict__,
            "results": results,
            "median_accuracy": float(np.median([r["accuracy"] for r in results])),
            "median_loss": float(np.median([r["loss"] for r in results])),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("release", "dpsgd", "certificate", "perstep_ags", "perstep_dpsgd"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--epsilon", type=float, default=10.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--scale-unit-norm", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ags-interval", type=float, default=1e-4)
    ap.add_argument("--ags-delta-fraction", type=float, default=0.9)
    ap.add_argument("--beta-epsilon-ratio", type=float, default=0.001)
    ap.add_argument("--per-step-ags-interval", type=float, default=1e-5)
    ap.add_argument("--dp-interval", type=float, default=1e-3)
    ap.add_argument("--dp-search-iterations", type=int, default=16)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 500])
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if args.arm == "release":
        run_release(args)
    elif args.arm == "dpsgd":
        run_dpsgd(args)
    elif args.arm == "certificate":
        run_certificate(args)
    elif args.arm == "perstep_ags":
        run_per_step(args, "ags")
    else:
        run_per_step(args, "dpsgd")
