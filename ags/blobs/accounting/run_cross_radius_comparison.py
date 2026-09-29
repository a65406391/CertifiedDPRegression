"""Re-account the completed linear-blob MILP runs with cross-radius semantics.

This script never recomputes an envelope.  It intersects compatible, completed
serialized boxes at each radius, constructs

    Abar_j^(k) = max(|L_j^k-U_j^(k+1)|, |U_j^k-L_j^(k+1)|),

and uses wider stored boxes or the unconditional clipping diameter where an
adjacent pair is unavailable.  A separate diagnostic records what would happen
under the repository's unproved non-expansive hinge-SGD cap.  AGS makes one
coordinate-wise Laplace release after training; DP-SGD injects Gaussian noise on
200 updates, while a fixed record participates in 20 of them. Both arms are
calibrated to replace-one (epsilon, delta) with PLD accounting.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from ags.blobs import baselines as B
from ags.blobs.shared import load_blobs
from common.accounting.accounting import (
    calibrate_ags,
    calibrate_dpsgd,
    calibrate_dpsgd_generic,
)
from common.accounting.artifacts import write_immutable_json


# The experiment is defined in config.py and re-exported here, because seven sibling drivers and a
# test already import these names from this module.
from ags.blobs.accounting.config import (  # noqa: E402
    ANISO,
    BATCH_SIZE,
    D,
    EPSILONS,
    FREQUENCIES,
    GAMMA,
    LAM,
    LR,
    N,
    PARAMETERS,
    PRIVACY_STEPS,
    SEP,
    STEPS,
)


def intersect_public_and_nested(boxes, absolute_radius: float, context: str = ""):
    """Intersect sound boxes with a public range and all enclosing radii."""
    tightened = {}
    for k, (lower, upper) in boxes.items():
        lower = np.maximum(np.asarray(lower, dtype=float), -absolute_radius)
        upper = np.minimum(np.asarray(upper, dtype=float), absolute_radius)
        if np.any(lower > upper):
            raise RuntimeError(f"empty public-range intersection {context}, k={k}")
        tightened[k] = lower, upper
    suffix_lower = None
    suffix_upper = None
    for k in sorted(tightened, reverse=True):
        lower, upper = tightened[k]
        suffix_lower = lower.copy() if suffix_lower is None else np.maximum(lower, suffix_lower)
        suffix_upper = upper.copy() if suffix_upper is None else np.minimum(upper, suffix_upper)
        if np.any(suffix_lower > suffix_upper):
            raise RuntimeError(f"empty nested-radius intersection {context}, k={k}")
        tightened[k] = suffix_lower.copy(), suffix_upper.copy()
    return tightened


def _run_directories(root: Path, frequency: int) -> list[Path]:
    # MILP envelopes and LP relaxations are both sound. Intersecting either kind
    # can only tighten the reachable box; QP is excluded until its solver-status
    # serialization receives the same audit coverage.
    return sorted(
        list(root.glob(f"*b20_ns200_F{frequency}_MILP")) + list(root.glob(f"*b20_ns200_F{frequency}_LP"))
    )


def load_intersected_boxes(root: Path, frequency: int):
    """Load only complete radius sweeps and intersect their valid final boxes."""
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sources: dict[int, list[str]] = {}
    nominal = None
    data_hash = None
    for directory in _run_directories(root, frequency):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cfg = meta["cfg"]
        if cfg.get("data_sha256") != data_hash and data_hash is not None:
            raise RuntimeError(f"incompatible data hashes in F={frequency}")
        data_hash = cfg.get("data_sha256")
        expected = int(meta["expected_windows_per_k"])
        expected_solves_per_radius = expected * 2 * int(meta["n_param_coords"])
        solve_counts: dict[int, int] = {}
        solves_path = directory / "solves.jsonl"
        solves_gzip_path = directory / "solves.jsonl.gz"
        if solves_path.exists() or solves_gzip_path.exists():
            if solves_path.exists():
                solve_lines = solves_path.read_text().splitlines()
            else:
                with gzip.open(solves_gzip_path, "rt") as handle:
                    solve_lines = handle.read().splitlines()
            for line in solve_lines:
                try:
                    radius = int(json.loads(line)["k"])
                    solve_counts[radius] = solve_counts.get(radius, 0) + 1
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        grouped: dict[int, list[tuple[int, Path]]] = {}
        for path in (directory / "steps").glob("k*_w*.npz"):
            match = re.fullmatch(r"k(\d+)_w(\d+)", path.stem)
            if match:
                k, window = map(int, match.groups())
                grouped.setdefault(k, []).append((window, path))
        for k, paths in grouped.items():
            # A multi-radius process can still be running after one radius has
            # completed. Reuse that radius only when both its step trace and all
            # coordinate solves are individually complete.
            if len(paths) != expected or solve_counts.get(k, 0) != expected_solves_per_radius:
                continue
            final_path = max(paths)[1]
            with np.load(final_path) as z:
                lower = np.asarray(z["param_l"], dtype=float).ravel()
                center = np.asarray(z["param_n"], dtype=float).ravel()
                upper = np.asarray(z["param_u"], dtype=float).ravel()
            if not (np.all(lower <= center) and np.all(center <= upper)):
                raise RuntimeError(f"invalid envelope ordering in {final_path}")
            if nominal is None:
                nominal = center
            elif not np.allclose(nominal, center, rtol=0, atol=1e-12):
                raise RuntimeError(f"nominal trajectory mismatch in {final_path}")
            if k in boxes:
                old_lower, old_upper = boxes[k]
                lower, upper = np.maximum(old_lower, lower), np.minimum(old_upper, upper)
                if np.any(lower > upper):
                    raise RuntimeError(f"empty envelope intersection at F={frequency}, k={k}")
            boxes[k] = lower, upper
            sources.setdefault(k, []).append(str(directory))
    if nominal is None or 1 not in boxes:
        raise RuntimeError(f"no complete radius-one box for F={frequency}")

    # Every coordinate starts at zero and each of the STEPS updates has
    # magnitude at most LR*GAMMA after elementwise clipping.  This is a public,
    # dataset-independent output range, so intersecting it with every AGT box is
    # sound (and prevents a timed-out relaxation from being worse than the
    # elementary global range).
    boxes = intersect_public_and_nested(boxes, STEPS * LR * GAMMA, context=f"at F={frequency}")
    boxes[0] = nominal.copy(), nominal.copy()
    return boxes, nominal, sources, data_hash


def cross_radius_profile(boxes, diameter=N):
    """Return every certificate for which T_k and T_(k+1) are both stored."""
    # At the full dataset diameter the ball has saturated: T_(N+1) := T_N.
    if diameter in boxes and diameter + 1 not in boxes:
        boxes = dict(boxes)
        boxes[diameter + 1] = boxes[diameter]
    certificates: dict[int, np.ndarray] = {}
    diagnostics = []
    for k in sorted(radius for radius in boxes if radius <= diameter):
        if k + 1 not in boxes:
            continue
        lower, upper = boxes[k]
        next_lower, next_upper = boxes[k + 1]
        nested = bool(np.all(next_lower <= lower + 1e-10) and np.all(next_upper >= upper - 1e-10))
        cross = np.maximum(np.abs(lower - next_upper), np.abs(upper - next_lower))
        next_width = next_upper - next_lower
        if nested and np.any(cross > next_width + 1e-9):
            raise RuntimeError(f"cross-radius/diameter check failed at k={k}")
        certificates[k] = cross
        diagnostics.append(
            {
                "k": k,
                "nested": nested,
                "cross_mean": float(cross.mean()),
                "cross_max": float(cross.max()),
                "next_width_mean": float(next_width.mean()),
                "next_width_max": float(next_width.max()),
            }
        )
    return certificates, diagnostics


def complete_profile(boxes, certificates, unconditional_diameter, diameter=N):
    """Fill missing radii without assuming non-expansive hinge updates."""
    profile = {}
    provenance = {}
    available = sorted(boxes)
    # This is a final release of training on the N-record dataset. Although each
    # optimizer batch has size 20, the replace-one dataset space has diameter N.
    for k in range(diameter + 1):
        if k in certificates:
            profile[k] = certificates[k]
            provenance[k] = "cross_radius"
            continue
        enclosing = [radius for radius in available if radius >= k + 1]
        if enclosing:
            radius = min(enclosing)
            lower, upper = boxes[radius]
            profile[k] = upper - lower
            provenance[k] = f"diameter_T_{radius}"
        else:
            profile[k] = unconditional_diameter.copy()
            provenance[k] = "unconditional_clipping_diameter"
    return profile, provenance


def smooth_profile(profile, beta: float):
    return np.maximum.reduce([np.exp(-beta * k) * value for k, value in profile.items()])


def quantiles(values):
    return np.quantile(np.asarray(values), [0, 0.25, 0.5, 0.75, 1]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounds-root", type=Path, default=Path("ags/blobs/runs/certified"))
    parser.add_argument(
        "--output", type=Path, default=Path("ags/blobs/accounting/artifacts/blobs_cross_radius_pld_v3.json")
    )
    parser.add_argument("--seeds", type=int, default=41)
    parser.add_argument("--ags-interval", type=float, default=1e-4)
    parser.add_argument("--dp-interval", type=float, default=1e-3)
    args = parser.parse_args()
    started = time.perf_counter()

    X, y, X_test, y_test = load_blobs(N, D, sep=SEP, seed=0, aniso=ANISO)
    P = np.c_[X, np.ones(N)]
    P_test = np.c_[X_test, np.ones(len(X_test))]
    conditional_nonexpansive_gs = np.full(PARAMETERS, LR * STEPS * 2.0 * GAMMA / BATCH_SIZE)
    unconditional_diameter = np.full(PARAMETERS, 2.0 * STEPS * LR * GAMMA)

    box_data = {}
    for frequency in FREQUENCIES:
        boxes, nominal, sources, data_hash = load_intersected_boxes(args.bounds_root, frequency)
        # F is a concretization frequency, not a sampling event.  The released
        # model is a function of the full N-record training dataset, so the
        # replace-one metric saturates at N for every F.
        diameter = N
        certs, diagnostics = cross_radius_profile(boxes, diameter)
        certified_profile, provenance = complete_profile(boxes, certs, unconditional_diameter, diameter)
        conditional_profile = {
            k: np.minimum(value, conditional_nonexpansive_gs) for k, value in certified_profile.items()
        }
        box_data[frequency] = (nominal, certified_profile, conditional_profile)
        box_data[(frequency, "audit")] = {
            "available_radii": sorted(boxes),
            "cross_radius_radii": sorted(certs),
            "profile_provenance": {str(k): value for k, value in provenance.items()},
            "sources": sources,
            "data_sha256": data_hash,
            "diagnostics": diagnostics,
        }

    rows = []
    for epsilon in EPSILONS:
        print(f"[blobs] calibrating epsilon={epsilon}", flush=True)
        ags_cal = calibrate_ags(
            epsilon,
            1e-5,
            coordinates=PARAMETERS,
            steps=1,
            q=1.0,
            mechanism="laplace",
            interval=args.ags_interval,
            delta_fraction=0.9,
        )
        dp_cal = calibrate_dpsgd(
            epsilon,
            1e-5,
            q=1.0,
            steps=PRIVACY_STEPS,
            interval=args.dp_interval,
            search_iterations=16,
        )
        dp_generic_cal = calibrate_dpsgd_generic(
            epsilon,
            1e-5,
            steps=PRIVACY_STEPS,
            interval=args.dp_interval,
            delta_fraction=0.99,
        )
        delta_coord = 1e-5 * 0.9 / (2 * PARAMETERS)
        beta_coord = ags_cal.mechanism_parameter / (2.0 * math.log(2.0 / delta_coord))

        ags_results = {}
        for frequency in FREQUENCIES:
            nominal, certified_profile, conditional_profile = box_data[frequency]
            variants = {}
            for name, profile in (
                ("certified", certified_profile),
                ("conditional_nonexpansive_cap", conditional_profile),
            ):
                smooth = smooth_profile(profile, beta_coord)
                scale = 2.0 * smooth / ags_cal.mechanism_parameter
                accuracies = [
                    B.acc(nominal + np.random.RandomState(seed).laplace(0.0, scale), P_test, y_test)
                    for seed in range(args.seeds)
                ]
                variants[name] = {
                    "accuracies": accuracies,
                    "median_accuracy": float(np.median(accuracies)),
                    "accuracy_quantiles": quantiles(accuracies),
                    "smooth_sensitivity_quantiles": quantiles(smooth),
                    "noise_scale_quantiles": quantiles(scale),
                }
            ags_results[str(frequency)] = variants

        # GaussianDpEvent defines the multiplier relative to the per-record norm
        # bound C.  Its REPLACE_ONE PLD already models the two-record shift, so
        # multiplying C by two here would count replacement adjacency twice.
        per_record_l2_bound = GAMMA * math.sqrt(PARAMETERS)
        dp_noise_std = dp_cal.mechanism_parameter * per_record_l2_bound / BATCH_SIZE
        dp_accuracies = []
        dp_generic_accuracies = []
        for seed in range(args.seeds):
            theta = B.train_hinge(
                P,
                y,
                n_s=STEPS,
                lr=LR,
                gamma=GAMMA,
                lam=LAM,
                b=BATCH_SIZE,
                noise_scale=dp_noise_std,
                rng=np.random.RandomState(seed),
                noise_kind="gauss",
            )
            dp_accuracies.append(B.acc(theta, P_test, y_test))
            theta_generic = B.train_hinge(
                P,
                y,
                n_s=STEPS,
                lr=LR,
                gamma=GAMMA,
                lam=LAM,
                b=BATCH_SIZE,
                noise_scale=(dp_generic_cal.mechanism_parameter * per_record_l2_bound / BATCH_SIZE),
                rng=np.random.RandomState(seed),
                noise_kind="gauss",
            )
            dp_generic_accuracies.append(B.acc(theta_generic, P_test, y_test))
        rows.append(
            {
                "target_epsilon": epsilon,
                "ags_calibration": ags_cal.__dict__,
                "ags_delta_coordinate": delta_coord,
                "ags_beta_coordinate": beta_coord,
                "ags": ags_results,
                "dpsgd_calibration": dp_cal.__dict__,
                "dpsgd_per_record_l2_bound": per_record_l2_bound,
                "dpsgd_gradient_noise_std": dp_noise_std,
                "dpsgd_accuracies": dp_accuracies,
                "dpsgd_median_accuracy": float(np.median(dp_accuracies)),
                "dpsgd_accuracy_quantiles": quantiles(dp_accuracies),
                "dpsgd_matched_generic_calibration": dp_generic_cal.__dict__,
                "dpsgd_matched_generic_gradient_noise_std": (
                    dp_generic_cal.mechanism_parameter * per_record_l2_bound / BATCH_SIZE
                ),
                "dpsgd_matched_generic_accuracies": dp_generic_accuracies,
                "dpsgd_matched_generic_median_accuracy": float(np.median(dp_generic_accuracies)),
                "dpsgd_matched_generic_accuracy_quantiles": quantiles(dp_generic_accuracies),
            }
        )
        print(
            f"[blobs-result] eps={epsilon} "
            + " ".join(f"F{f}={ags_results[str(f)]['certified']['median_accuracy']:.4f}" for f in FREQUENCIES)
            + f" conditional-F10={ags_results['10']['conditional_nonexpansive_cap']['median_accuracy']:.4f}"
            + f" DP={np.median(dp_accuracies):.4f}",
            flush=True,
        )

    payload = {
        "schema": 1,
        "experiment": "linear_blobs_cross_radius_accountant_controlled_v4",
        "supersedes": [
            "ags/blobs/accounting/artifacts/blobs_cross_radius_pld.json",
            "ags/blobs/accounting/artifacts/blobs_cross_radius_pld_v2.json",
        ],
        "supersession_reason": (
            "The first pilot counted replace-one twice for DP-SGD. V2 corrected that "
            "but treated the unproved 0.5 hinge-SGD non-expansive cap as unconditional. "
            "V3 separates fully certified and conditional results. V4 fixes two "
            "radius-bookkeeping defects: F is a concretization frequency rather than "
            "a dataset diameter, and adjacent certificates after a sparse-radius gap "
            "must still be retained."
        ),
        "privacy": {"delta": 1e-5, "adjacency": "replace-one", "q": 1.0},
        "setting": {
            "N": N,
            "dimensions": D,
            "parameters": PARAMETERS,
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "epochs": PRIVACY_STEPS,
            "learning_rate": LR,
            "coordinate_clip": GAMMA,
            "seeds": args.seeds,
            "frequencies": list(FREQUENCIES),
            "epsilons": list(EPSILONS),
        },
        "accounting_note": (
            "AGS uses conservative generic PLDs composed across the nine coordinates; "
            "the matched-generic DP-SGD control reduces every Gaussian step to a generic "
            "DP pair before using the same composition class. A separate diagnostic gives "
            "DP-SGD its replace-one Gaussian PLD across the 20 releases touching any "
            "fixed record. The ten disjoint deterministic minibatches within an epoch "
            "use adaptive parallel composition, not 200-fold sequential composition."
        ),
        "sensitivity_note": (
            "Adjacent-radius MILP/LP boxes define Abar. Sound boxes are intersected with "
            "the public clipped-output range and with every enclosing larger-radius box. "
            "The certified variant fills missing radii using a wider stored box or the "
            "unconditional clipping diameter. The "
            "conditional variant additionally assumes a 0.5 non-expansive hinge-SGD cap; "
            "that assumption is not established and must not be presented as certified."
        ),
        "conditional_nonexpansive_cap_per_coordinate": float(conditional_nonexpansive_gs[0]),
        "unconditional_clipping_diameter_per_coordinate": float(unconditional_diameter[0]),
        "bound_audits": {str(f): box_data[(f, "audit")] for f in FREQUENCIES},
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    digest = write_immutable_json(args.output, payload)
    print(f"[blobs] wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
