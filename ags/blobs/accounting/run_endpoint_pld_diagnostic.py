"""Non-certified native-PLD headroom diagnostic for the fixed Blobs cell.

The fixed-pair PLDs are pessimistic, but the search covers only mixed endpoint
scale ratios.  A continuous-envelope proof is required before these scales can
replace the interval-certified RDP scales in a privacy claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ags.blobs import baselines as B
from ags.blobs.shared import load_blobs
from common.accounting.artifacts import write_immutable_json
from common.accounting.laplace_pld import fixed_pair_laplace_pld
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
from ags.blobs.accounting.run_public_projection import projected_accuracies


def compose(parts):
    result = None
    for mechanism, count in parts:
        if count:
            component = mechanism.self_compose(count)
            result = component if result is None else result.compose(component)
    if result is None:
        raise ValueError("at least one mechanism must be composed")
    return result


def mixed_endpoint_epsilon(
    epsilon_weight: float,
    beta_weight: float,
    epsilon_bias: float,
    beta_bias: float,
    delta: float,
    interval: float,
) -> tuple[float, dict]:
    ratios_weight = (math.exp(-beta_weight), math.exp(beta_weight))
    ratios_bias = (math.exp(-beta_bias), math.exp(beta_bias))

    def directions(epsilon, ratio):
        shift = 0.5 * epsilon * min(1.0, ratio)
        return (
            fixed_pair_laplace_pld(shift, ratio, interval=interval),
            fixed_pair_laplace_pld(shift / ratio, 1.0 / ratio, interval=interval),
        )

    weights = {ratio: directions(epsilon_weight, ratio) for ratio in ratios_weight}
    biases = {ratio: directions(epsilon_bias, ratio) for ratio in ratios_bias}
    worst = (-math.inf, None)
    for low_count in range(9):
        for bias_ratio in ratios_bias:
            for direction in (0, 1):
                law = compose(
                    (
                        (weights[ratios_weight[0]][direction], low_count),
                        (weights[ratios_weight[1]][direction], 8 - low_count),
                        (biases[bias_ratio][direction], 1),
                    )
                )
                epsilon = law.get_epsilon_for_delta(delta)
                if epsilon > worst[0]:
                    worst = (
                        epsilon,
                        {
                            "low_scale_ratio_weight_coordinates": low_count,
                            "bias_scale_ratio": bias_ratio,
                            "direction": "forward" if direction == 0 else "reverse",
                        },
                    )
    return worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[15, 20, 25, 30, 40, 50])
    parser.add_argument("--interval", type=float, default=0.002)
    parser.add_argument("--seeds", type=int, default=101)
    args = parser.parse_args()
    started = time.perf_counter()
    source = json.loads(args.input.read_text())
    source_rows = {row["target_epsilon"]: row for row in source["rows"]}
    boxes, nominal, box_sources, data_hash = load_intersected_boxes(Path("ags/blobs/runs/certified"), 10)
    certificates, _ = cross_radius_profile(boxes, N)
    profile, provenance = complete_profile(
        boxes,
        certificates,
        np.full(PARAMETERS, 2.0 * STEPS * LR * GAMMA),
        N,
    )
    X, y, Xte, yte = load_blobs(N, D, sep=SEP, seed=0, aniso=5.0)
    design = np.c_[X, np.ones(N)]
    nominal_check = B.train_hinge(design, y, n_s=STEPS, lr=LR, gamma=GAMMA, lam=0.0, b=BATCH_SIZE)
    np.testing.assert_allclose(nominal, nominal_check, rtol=0, atol=1e-10)
    public_size = int(source["setting"]["public_validation_size"])
    public_design = np.c_[Xte[:public_size], np.ones(public_size)]
    public_labels = yte[:public_size]
    test_design = np.c_[Xte[public_size:], np.ones(len(Xte) - public_size)]
    test_labels = yte[public_size:]
    radius = STEPS * LR * GAMMA

    rows = []
    for target in args.epsilons:
        cell = source_rows[target]["ags_public_weight_bias_allocations"]["4.0"]["2.0"]
        base_epsilon = np.asarray(cell["epsilon_coordinates"], dtype=float)
        base_beta = np.asarray(cell["beta_coordinates"], dtype=float)

        def evaluate(factor):
            return mixed_endpoint_epsilon(
                factor * base_epsilon[0],
                factor * base_beta[0],
                factor * base_epsilon[-1],
                factor * base_beta[-1],
                1e-5,
                args.interval,
            )

        low, high = 1.0, 1.5
        while evaluate(high)[0] <= target:
            high *= 1.25
        for _ in range(16):
            middle = 0.5 * (low + high)
            if evaluate(middle)[0] <= target:
                low = middle
            else:
                high = middle
        factor = low * (1.0 - 2e-4)
        achieved, worst = evaluate(factor)
        epsilon = factor * base_epsilon
        beta = factor * base_beta
        sensitivity = smooth_profile(profile, beta)
        scales = 2.0 * sensitivity / epsilon
        utility = projected_accuracies(
            nominal,
            scales,
            args.seeds,
            test_design,
            test_labels,
            public_design,
            public_labels,
            radius,
        )
        rows.append(
            {
                "target_epsilon": target,
                "rdp_base_epsilon_upper": cell["interval_certification"]["epsilon_upper"],
                "endpoint_pld_factor": factor,
                "endpoint_pld_achieved_epsilon": achieved,
                "endpoint_pld_worst_configuration": worst,
                "epsilon_coordinates": epsilon.tolist(),
                "beta_coordinates": beta.tolist(),
                "sensitivity": sensitivity.tolist(),
                "noise_scales": scales.tolist(),
                "utility": utility,
            }
        )
        print(
            f"[blob-endpoint-pld] epsilon={target:g} factor={factor:.5f} "
            f"achieved={achieved:.6f} median={utility['latest_box_unprojected_median_accuracy']:.4f}",
            flush=True,
        )

    digest = write_immutable_json(
        args.output,
        {
            "schema": 1,
            "status": "noncertified_endpoint_native_pld_headroom_diagnostic",
            "certification_blocker": (
                "Fixed neighboring-pair PLDs are pessimistically discretized and mixed "
                "endpoint assignments are searched, but no proof currently excludes a "
                "larger composed hockey-stick divergence at an interior continuous "
                "scale ratio. These scales must not be reported as a DP guarantee."
            ),
            "source": str(args.input),
            "setting": source["setting"],
            "pld_interval": args.interval,
            "available_radii": sorted(boxes),
            "profile_provenance": {str(k): value for k, value in provenance.items()},
            "box_sources": box_sources,
            "data_sha256": data_hash,
            "rows": rows,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"[blob-endpoint-pld] wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
