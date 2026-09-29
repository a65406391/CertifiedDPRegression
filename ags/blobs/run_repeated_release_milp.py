"""Repeated-release AGS with online, fully serialized MILP certificates.

Each block contains ``F`` consecutive minibatch updates.  At its boundary we
compute T_1,...,T_(K+1) from the current *public* parameter vector, form the
corrected cross-radius certificates, sample one coordinate-wise Laplace AGS
release, and overwrite the model state with that release.  No clean trajectory
is retained across a release boundary.

The radii above K are handled by the public coordinate global-sensitivity tail,
so truncating the MILP radius grid does not silently truncate smooth sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from ags.blobs.shared import load_blobs, linear_model
from ags.blobs.opt_bound import opt_bound
from ags.blobs.serialize import RunRecorder, install, uninstall
from common.accounting.accounting import calibrate_ags
from common.accounting.laplace_rdp import (
    certified_smooth_laplace_rdp,
    epsilon_from_rdp,
    worst_case_smooth_laplace_rdp_grid,
)


RDP_ORDERS = np.r_[np.linspace(1.05, 2.0, 20), np.arange(2.5, 64.5, 0.5)]


def cross_radius_box(l_k, u_k, l_next, u_next):
    """Elementwise XDiam(T_k,T_(k+1))."""
    return np.maximum(
        np.abs(np.asarray(l_k) - np.asarray(u_next)), np.abs(np.asarray(u_k) - np.asarray(l_next))
    )


def certified_smooth_sensitivity(boxes, nominal, gs, beta, max_k):
    """Cross-radius SS through max_k, completed by a sound GS tail."""
    candidates = []
    l_prev = u_prev = np.asarray(nominal, dtype=float)
    for k in range(max_k + 1):
        l_next, u_next = boxes[k + 1]
        abar = np.minimum(cross_radius_box(l_prev, u_prev, l_next, u_next), gs)
        candidates.append(np.exp(-beta * k) * abar)
        l_prev, u_prev = l_next, u_next
    # Every uncomputed Abar^(k) is at most GS; decay is monotone, hence the
    # largest omitted contribution is the first omitted radius.
    candidates.append(np.exp(-beta * (max_k + 1)) * gs)
    return np.maximum.reduce(candidates), candidates[:-1]


def record_touching_releases(epochs):
    """Maximum releases involving one record under the fixed cyclic schedule.

    Every release block contains distinct records (checked in ``main``), and a
    record occurs once per epoch.  Conditional mechanisms on disjoint blocks
    compose in parallel even when each block starts from the preceding public
    release, so only these record-touching releases compose sequentially.
    """
    return int(epochs)


def clipping_tail(block_steps, lr, gamma, batch_size):
    """Public coordinate GS tail available from clipping alone.

    For one update, only the substituted example changes and the exact clipped
    mean bound retains ``1/b``.  Across multiple updates, absent a proved
    non-expansive update map, subsequent shared-example gradients may diverge;
    the unconditional output diameter is therefore ``2*lr*gamma*steps``.
    """
    if block_steps == 1:
        return 2.0 * lr * gamma / batch_size
    return 2.0 * lr * gamma * block_steps


def calibrate_repeated_rdp(target_epsilon, delta, coordinates, privacy_steps, beta_fraction):
    """Calibrate one conditional smooth-Laplace release in common RDP."""
    delta_coordinate = delta / (2 * coordinates * privacy_steps)

    def privacy(epsilon_coordinate, points):
        beta_limit = epsilon_coordinate / (2 * math.log(2 / delta_coordinate))
        beta = beta_fraction * beta_limit
        per_coordinate = np.asarray(
            [
                worst_case_smooth_laplace_rdp_grid(
                    float(order),
                    epsilon_coordinate,
                    beta,
                    ratio_points=points,
                    shift_points=2,
                ).rdp
                for order in RDP_ORDERS
            ]
        )
        return (
            *epsilon_from_rdp(RDP_ORDERS, coordinates * privacy_steps * per_coordinate, delta),
            beta,
            beta_limit,
        )

    low, high = 1e-6, 1.0
    while privacy(high, 1001)[0] < target_epsilon:
        high *= 2
    for _ in range(26):
        mid = (low + high) / 2
        if privacy(mid, 1001)[0] <= target_epsilon:
            low = mid
        else:
            high = mid
    low *= 1 - 2e-6
    achieved, order, beta, beta_limit = privacy(low, 8001)
    interval = certified_smooth_laplace_rdp(order, low, beta, tolerance=1e-8)
    epsilon_upper = epsilon_from_rdp([order], [coordinates * privacy_steps * interval.upper], delta)[0]
    return {
        "mechanism_parameter": low,
        "epsilon": achieved,
        "delta": delta,
        "steps": privacy_steps,
        "q": 1.0,
        "accountant": "analytic-rdp-interval-certified",
        "minimizing_order": order,
        "beta": beta,
        "beta_limit": beta_limit,
        "beta_fraction": beta_fraction,
        "interval_epsilon_upper": epsilon_upper,
        "interval_rdp_lower": interval.lower,
        "interval_rdp_upper": interval.upper,
        "interval_boxes_per_coordinate_release": interval.boxes,
    }


def model_at(theta):
    theta = np.asarray(theta, dtype=float)

    def make(d):
        model = linear_model(d)
        vector_to_parameters(torch.tensor(theta, dtype=torch.float64), model.parameters())
        return model

    return make


def accuracy(theta, X, y):
    theta = np.asarray(theta)
    return float((np.sign(X @ theta[:-1] + theta[-1]) == np.sign(y)).mean())


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_public_checkpoint(
    path: Path, *, theta, rng, release_idx, next_step, learning_rate, sample_indices, accounting
):
    """Write the complete state allowed to initialize the next block.

    The clean nominal and interval endpoints deliberately are not included.
    They live only in the non-public certificate audit tree.
    """
    state = rng.get_state()
    np.savez_compressed(
        path,
        released_theta=np.asarray(theta),
        release_idx=np.int64(release_idx),
        next_step=np.int64(next_step),
        learning_rate=np.float64(learning_rate),
        momentum=np.array([], dtype=float),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        rng_algorithm=np.asarray(state[0]),
        rng_keys=state[1],
        rng_position=np.int64(state[2]),
        rng_has_gauss=np.int64(state[3]),
        rng_cached_gaussian=np.float64(state[4]),
        accounting=np.asarray(json.dumps(accounting, sort_keys=True)),
        persistent_state_policy=np.asarray("released-vector-only"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--b", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--F", type=int, required=True)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--accountant", choices=("generic", "rdp"), default="rdp")
    parser.add_argument(
        "--beta-fraction", type=float, default=1.0, help="multiple of the classic smooth-Laplace beta limit"
    )
    parser.add_argument("--beta-ratio", type=float, default=0.02)
    parser.add_argument("--pld-interval", type=float, default=1e-4)
    parser.add_argument("--delta-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--max-releases", type=int, default=0, help="diagnostic truncation; zero runs every release"
    )
    parser.add_argument("--tag", default="repeated")
    args = parser.parse_args()

    if not 1 <= args.F <= args.N // args.b:
        raise ValueError("F must be between 1 and the number of disjoint batches per epoch")
    if args.max_k + 1 > args.F * args.b:
        raise ValueError("max-k+1 exceeds the block dataset diameter")

    X, y, X_test, y_test = load_blobs(args.N, args.d, sep=4.0, seed=0, aniso=5.0)
    y01 = (y > 0).astype(float)
    design = np.c_[X, np.ones(args.N)]
    lr = min(0.5, 1.0 / (float(np.linalg.eigvalsh(design.T @ design).max()) / args.N))
    total_steps = args.epochs * (args.N // args.b)
    releases = math.ceil(total_steps / args.F)
    if args.max_releases:
        releases = min(releases, args.max_releases)
    coords = args.d + 1
    privacy_steps = record_touching_releases(args.epochs)

    if args.accountant == "generic":
        calibration_obj = calibrate_ags(
            args.epsilon,
            args.delta,
            coordinates=coords,
            steps=privacy_steps,
            q=1.0,
            mechanism="laplace",
            interval=args.pld_interval,
            delta_fraction=args.delta_fraction,
        )
        calibration = calibration_obj.__dict__
        eps_coord = calibration_obj.mechanism_parameter
        beta = eps_coord * args.beta_ratio
        delta_coord = args.delta * args.delta_fraction / (privacy_steps * 2 * coords)
        beta_limit = eps_coord / (2 * math.log(2 / delta_coord))
        if beta > beta_limit:
            raise ValueError(f"beta={beta} violates smooth-Laplace limit {beta_limit}")
        calibration.update(beta=beta, beta_limit=beta_limit)
    else:
        calibration = calibrate_repeated_rdp(
            args.epsilon, args.delta, coords, privacy_steps, args.beta_fraction
        )
        eps_coord = calibration["mechanism_parameter"]
        beta = calibration["beta"]
        beta_limit = calibration["beta_limit"]
        if calibration["interval_epsilon_upper"] > args.epsilon:
            raise RuntimeError("continuous RDP interval exceeds target epsilon")

    beta_tag = f"bf{args.beta_fraction:g}" if args.accountant == "rdp" else f"br{args.beta_ratio:g}"
    run_id = (
        f"{args.tag}_{args.accountant}_{beta_tag}_F{args.F}"
        f"_eps{args.epsilon:g}_d{args.delta:g}_K{args.max_k}"
        f"_seed{args.seed}"
    )
    root = Path(__file__).resolve().parent / "runs" / "repeated_release" / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "public_checkpoints").mkdir()
    (root / "nonpublic_certificate_audit").mkdir()
    cfg = vars(args) | {
        "run_id": run_id,
        "lr": lr,
        "total_steps": total_steps,
        "release_count": releases,
        "coordinates": coords,
        "record_touching_privacy_steps": privacy_steps,
        "eps_coord": eps_coord,
        "beta": beta,
        "beta_limit": beta_limit,
        "accounting": calibration,
        "sampling_q": 1.0,
        "amplification": "none",
        "composition": "adaptive parallel within epoch; sequential across epochs",
        "state_boundary": "released-vector-only",
        "artifact_policy": {
            "restartable_state": "public_checkpoints/release_*.npz",
            "certificate_trace": "nonpublic_certificate_audit/",
            "certificate_trace_may_be_published": False,
            "next_block_reads_certificate_trace": False,
        },
        "radius_semantics": "cross-radius T_k versus T_(k+1), GS tail",
        "data_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
    }
    atomic_json(root / "meta.json", cfg)

    theta = np.zeros(coords)
    rng = np.random.RandomState(args.seed)
    records = []
    started = time.time()
    try:
        for release_idx in range(releases):
            start_step = release_idx * args.F
            block_steps = min(args.F, total_steps - start_step)
            sample_idx = np.arange(start_step * args.b, (start_step + block_steps) * args.b) % args.N
            if len(np.unique(sample_idx)) != len(sample_idx):
                raise RuntimeError("a block contains a repeated source record")
            xb, yb = X[sample_idx], y01[sample_idx]
            child_id = f"{run_id}_release{release_idx:04d}"
            child_cfg = cfg | {
                "release_idx": release_idx,
                "block_steps": block_steps,
                "sample_indices": sample_idx.tolist(),
            }
            recorder = RunRecorder(
                child_id,
                child_cfg,
                expected_windows_per_k=1,
                k_grid=list(range(1, args.max_k + 2)),
                n_param_coords=coords,
                root=str(root / "nonpublic_certificate_audit"),
                heartbeat=30.0,
            )
            boxes = {}
            nominal = None
            install(recorder)
            try:
                for radius in range(1, args.max_k + 2):
                    recorder.set_k(radius)
                    l, n, u, _ = opt_bound(
                        xb,
                        yb,
                        k=radius,
                        b=args.b,
                        n_epochs=1,
                        lr=lr,
                        gamma=args.gamma,
                        F=block_steps,
                        semantics="MILP",
                        timeout=args.timeout,
                        threads=args.threads,
                        model_fn=model_at(theta),
                    )
                    boxes[radius] = (l, u)
                    if nominal is None:
                        nominal = n
                    elif not np.allclose(nominal, n, rtol=1e-9, atol=1e-10):
                        raise RuntimeError("nominal trajectory changed across radius solves")
                manifest = recorder.close("completed")
            except Exception as error:
                recorder.close("failed", str(error))
                raise
            finally:
                uninstall()

            gs = np.full(coords, clipping_tail(block_steps, lr, args.gamma, args.b))
            sensitivity, contributions = certified_smooth_sensitivity(
                boxes,
                nominal,
                gs,
                beta,
                args.max_k,
            )
            scale = 2 * sensitivity / eps_coord
            released = nominal + rng.laplace(0.0, scale)
            record = {
                "release_idx": release_idx,
                "start_step": start_step,
                "block_steps": block_steps,
                "sample_indices": sample_idx.tolist(),
                "public_start": theta.tolist(),
                "released": released.tolist(),
                "gs": gs.tolist(),
                "smooth_sensitivity": sensitivity.tolist(),
                "noise_scale": scale.tolist(),
                "cross_radius_weighted": [v.tolist() for v in contributions],
                "selected_candidate": np.argmax(
                    np.stack(contributions + [np.exp(-beta * (args.max_k + 1)) * gs]), axis=0
                ).tolist(),
                "test_accuracy": accuracy(released, X_test, y_test),
                "certificate_manifest": manifest,
            }
            records.append(record)
            # The clean nominal is retained only in the explicitly non-public
            # audit tree, never in the public/restartable release stream.
            np.savez_compressed(
                root / "nonpublic_certificate_audit" / f"release_{release_idx:04d}_clean_nominal.npz",
                nominal_discarded=nominal,
            )
            with open(root / "releases.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            theta = released.copy()  # the only state crossing the privacy boundary
            save_public_checkpoint(
                root / "public_checkpoints" / f"release_{release_idx:04d}.npz",
                theta=theta,
                rng=rng,
                release_idx=release_idx,
                next_step=start_step + block_steps,
                learning_rate=lr,
                sample_indices=sample_idx,
                accounting=calibration,
            )
            elapsed = time.time() - started
            eta = elapsed / (release_idx + 1) * (releases - release_idx - 1)
            print(
                f"[F={args.F}] release={release_idx + 1}/{releases} "
                f"acc={record['test_accuracy']:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

        released_path = np.asarray([record["released"] for record in records])
        averaging_windows = sorted(set(min(len(released_path), value) for value in (1, 10, 20, 50, 100, 200)))
        tail_averages = {
            str(window): {
                "theta": released_path[-window:].mean(axis=0).tolist(),
                "accuracy": accuracy(released_path[-window:].mean(axis=0), X_test, y_test),
            }
            for window in averaging_windows
        }
        result = {
            "status": "completed",
            "run_id": run_id,
            "config": cfg,
            "final_theta": theta.tolist(),
            "final_accuracy": accuracy(theta, X_test, y_test),
            "tail_averages_postprocessing": tail_averages,
            "wall_s": time.time() - started,
            "release_records": len(records),
        }
        atomic_json(root / "results.json", result)
        atomic_json(
            root / "MANIFEST.json",
            {
                "status": "completed",
                "release_records": len(records),
                "release_records_expected": releases,
                "clean_state_retained": False,
            },
        )
        print(json.dumps(result, indent=1), flush=True)
    except Exception as error:
        atomic_json(
            root / "MANIFEST.json",
            {
                "status": "failed",
                "error": str(error),
                "release_records": len(records),
                "release_records_expected": releases,
            },
        )
        raise


if __name__ == "__main__":
    main()
