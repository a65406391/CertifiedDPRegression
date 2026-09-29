"""Summarise the certified envelope across all nine retained allocation rules.

Companion to `summarize_native_pld_envelope.py`, which reports the single fixed cell. Reports how
many rules show a positive interval at each epsilon, so a reader can see whether the headline cell
is representative or lucky.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.accounting.artifacts import write_immutable_json


ROOT = Path("ags/accounting/artifacts")


def comparison(first, second, seed):
    first, second = np.asarray(first, float), np.asarray(second, float)
    rng = np.random.RandomState(seed)
    trials = 20_000
    differences = first[rng.randint(len(first), size=(trials, len(first)))].mean(1) - second[
        rng.randint(len(second), size=(trials, len(second)))
    ].mean(1)
    return {
        "ags_median": float(np.median(first)),
        "dpsgd_median": float(np.median(second)),
        "mean_difference": float(first.mean() - second.mean()),
        "unpaired_bootstrap_mean_difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def main():
    source = json.loads((ROOT / "blobs_native_pld_envelope_allocation_grid_overnight.json").read_text())
    matched = {
        row["target_epsilon"]: row
        for row in json.loads((ROOT / "blobs_dpsgd_pld_reference_overnight.json").read_text())["rows"]
    }
    poisson = {
        row["target_epsilon"]: row
        for row in json.loads((ROOT / "blobs_poisson_dpsgd_reference_overnight.json").read_text())["rows"]
    }
    rows = []
    for row_index, row in enumerate(source["rows"]):
        epsilon = row["target_epsilon"]
        rules = []
        for rule_index, rule in enumerate(row["rules"]):
            releases = rule["utility"]["latest_box_unprojected_accuracies"]
            rules.append(
                {
                    "rule": rule["rule"],
                    "selected_accountant": rule["selected_accountant"],
                    "selected_parameter_factor_over_rdp": rule["selected_parameter_factor_over_rdp"],
                    "matched_fixed_cyclic_dpsgd": comparison(
                        releases,
                        matched[epsilon]["accuracies"],
                        20260826 + 100 * row_index + rule_index,
                    ),
                    "poisson_dpsgd_stress_test": comparison(
                        releases,
                        poisson[epsilon]["accuracies"],
                        20261826 + 100 * row_index + rule_index,
                    ),
                }
            )
        rows.append(
            {
                "target_epsilon": epsilon,
                "rules": rules,
                "positive_matched_ci_count": sum(
                    x["matched_fixed_cyclic_dpsgd"]["unpaired_bootstrap_mean_difference_ci95"][0] > 0
                    for x in rules
                ),
                "positive_poisson_ci_count": sum(
                    x["poisson_dpsgd_stress_test"]["unpaired_bootstrap_mean_difference_ci95"][0] > 0
                    for x in rules
                ),
            }
        )
    payload = {
        "schema": 1,
        "status": "complete_certified_native_pld_all_rule_summary",
        "selection_status": source["selection_status"],
        "statistical_note": (
            "All intervals are descriptive and unadjusted for multiplicity. "
            "Every retained rule is included; no test-best rule is selected."
        ),
        "rows": rows,
    }
    output = ROOT / "blobs_native_pld_envelope_allocation_summary_overnight.json"
    digest = write_immutable_json(output, payload)
    lines = [
        "# Certified native-PLD all-allocation Blobs summary",
        "",
        payload["statistical_note"],
        "",
        "| epsilon | rules | positive vs matched DP | positive vs Poisson DP |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_epsilon']:g} | {len(row['rules'])} | "
            f"{row['positive_matched_ci_count']} | "
            f"{row['positive_poisson_ci_count']} |"
        )
    lines += [
        "",
        "## Epsilon 15 details",
        "",
        "| rule | AGS | matched DP | difference | 95% CI |",
        "|:---|---:|---:|---:|---:|",
    ]
    for rule in rows[0]["rules"]:
        value = rule["matched_fixed_cyclic_dpsgd"]
        low, high = value["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {rule['rule']} | {value['ags_median']:.4f} | "
            f"{value['dpsgd_median']:.4f} | {value['mean_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] |"
        )
    report = ROOT / "blobs_native_pld_envelope_allocation_summary_overnight.md"
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
