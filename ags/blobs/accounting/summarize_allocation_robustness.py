"""Audit the allocation rules without selecting on test accuracy.

Reports every retained rule rather than the best one, and labels the post-hoc finding as post-hoc:
one rule looks better than the fixed cell at a low epsilon, but that is one of nine inspected
without multiplicity adjustment, so it is a preregistration target rather than a claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from common.accounting.artifacts import write_immutable_json


ROOT = Path("ags/accounting/artifacts")
SOURCES = {
    "ags": ROOT / "blobs_public_projection_overnight.json",
    "fixed_dp": ROOT / "blobs_dpsgd_pld_reference_overnight.json",
    "poisson_dp": ROOT / "blobs_poisson_dpsgd_reference_overnight.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(ags, dp, seed: int) -> dict:
    ags = np.asarray(ags, dtype=float)
    dp = np.asarray(dp, dtype=float)
    rng = np.random.RandomState(seed)
    trials = 20_000
    ags_indices = rng.randint(0, len(ags), size=(trials, len(ags)))
    dp_indices = rng.randint(0, len(dp), size=(trials, len(dp)))
    differences = ags[ags_indices].mean(1) - dp[dp_indices].mean(1)
    return {
        "ags_median": float(np.median(ags)),
        "dpsgd_median": float(np.median(dp)),
        "mean_difference": float(ags.mean() - dp.mean()),
        "unpaired_bootstrap_mean_difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "ags_trials": len(ags),
        "dpsgd_trials": len(dp),
    }


def allocation_rules(row: dict):
    yield "uniform", row["ags_uniform"]["4.0"]
    for name, cell in row["ags_public_coordinate_allocations"]["4.0"].items():
        yield name, cell
    for ratio, cell in row["ags_public_weight_bias_allocations"]["4.0"].items():
        yield f"weight_bias_ratio_{ratio}", cell


def main() -> None:
    ags = json.loads(SOURCES["ags"].read_text())
    fixed = {row["target_epsilon"]: row for row in json.loads(SOURCES["fixed_dp"].read_text())["rows"]}
    poisson = {row["target_epsilon"]: row for row in json.loads(SOURCES["poisson_dp"].read_text())["rows"]}
    rows = []
    for row_index, row in enumerate(ags["rows"]):
        epsilon = row["target_epsilon"]
        rules = []
        for rule_index, (name, cell) in enumerate(allocation_rules(row)):
            releases = cell["latest_box_unprojected_accuracies"]
            matched = comparison(
                releases,
                fixed[epsilon]["accuracies"],
                seed=20260826 + 100 * row_index + 2 * rule_index,
            )
            stress = comparison(
                releases,
                poisson[epsilon]["accuracies"],
                seed=20260827 + 100 * row_index + 2 * rule_index,
            )
            rules.append(
                {
                    "rule": name,
                    "matched_fixed_cyclic_dpsgd": matched,
                    "poisson_dpsgd_stress_test": stress,
                }
            )
        rows.append(
            {
                "target_epsilon": epsilon,
                "rules": rules,
                "rules_with_positive_matched_ci": sum(
                    rule["matched_fixed_cyclic_dpsgd"]["unpaired_bootstrap_mean_difference_ci95"][0] > 0
                    for rule in rules
                ),
                "rules_with_positive_poisson_ci": sum(
                    rule["poisson_dpsgd_stress_test"]["unpaired_bootstrap_mean_difference_ci95"][0] > 0
                    for rule in rules
                ),
            }
        )
    payload = {
        "schema": 1,
        "status": "complete_exploratory_all_rule_robustness_audit",
        "beta_fraction": 4.0,
        "selection_status": (
            "No rule is selected in this artifact. Every retained beta=4 rule is "
            "reported. This is a post-hoc descriptive robustness audit, not a "
            "confirmatory comparison."
        ),
        "accounting": {
            "ags": "direct interval-certified coordinate Laplace RDP",
            "matched_dpsgd": "native replace-one Gaussian PLD; 20 fixed-cyclic releases",
            "poisson_stress_test": "native amplified Gaussian PLD; q=0.1, 200 releases",
        },
        "source_sha256": {name: digest(path) for name, path in SOURCES.items()},
        "rows": rows,
    }
    output = ROOT / "blobs_allocation_robustness_overnight.json"
    output_hash = write_immutable_json(output, payload)

    lines = [
        "# Blobs beta=4 allocation robustness audit",
        "",
        payload["selection_status"],
        "",
        "| epsilon | retained rules | positive vs matched DP | positive vs Poisson DP |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_epsilon']:g} | {len(row['rules'])} | "
            f"{row['rules_with_positive_matched_ci']} | "
            f"{row['rules_with_positive_poisson_ci']} |"
        )
    report = ROOT / "blobs_allocation_robustness_overnight.md"
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} sha256={output_hash}")


if __name__ == "__main__":
    main()
