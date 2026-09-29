"""The headline blobs table: certified AGS against both DP-SGD references.

Reads the envelope comparison and the two baselines, and reports per epsilon the medians over 101
release seeds and the mean difference with a bootstrap interval.

Two things the table says in its own labels. The intervals are descriptive, not confirmatory,
because the AGS cell was chosen during exploratory development. And the matched baseline is
fixed-cyclic DP-SGD with its native Gaussian accountant; the Poisson arm is a separate stress test
that is not algorithmically matched, so it is never folded into the same claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from common.accounting.artifacts import write_immutable_json


ROOT = Path("ags/accounting/artifacts")
SOURCES = {
    "ags": ROOT / "blobs_native_pld_envelope_overnight.json",
    "matched_dp": ROOT / "blobs_dpsgd_pld_reference_overnight.json",
    "poisson_dp": ROOT / "blobs_poisson_dpsgd_reference_overnight.json",
}


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
        "ags_trials": len(first),
        "dpsgd_trials": len(second),
    }


def main():
    ags = json.loads(SOURCES["ags"].read_text())
    matched = {row["target_epsilon"]: row for row in json.loads(SOURCES["matched_dp"].read_text())["rows"]}
    poisson = {row["target_epsilon"]: row for row in json.loads(SOURCES["poisson_dp"].read_text())["rows"]}
    rows = []
    for index, row in enumerate(ags["rows"]):
        epsilon = row["target_epsilon"]
        releases = row["utility"]["latest_box_unprojected_accuracies"]
        rows.append(
            {
                "target_epsilon": epsilon,
                "selected_accountant": row["selected_accountant"],
                "selected_parameter_factor_over_rdp": row["selected_parameter_factor_over_rdp"],
                "selected_achieved_epsilon_upper": row["selected_achieved_epsilon_upper"],
                "matched_fixed_cyclic_dpsgd": comparison(
                    releases, matched[epsilon]["accuracies"], 20260826 + index
                ),
                "poisson_dpsgd_stress_test": comparison(
                    releases, poisson[epsilon]["accuracies"], 20260926 + index
                ),
            }
        )
    payload = {
        "schema": 1,
        "status": "complete_certified_native_pld_envelope_comparison",
        "proof_basis": ags["proof_basis"],
        "selection_caveat": ags["ags_fixed_reference"]["selection_status"],
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in SOURCES.items()
        },
        "statistical_note": (
            "Intervals independently bootstrap mechanism trials and are descriptive, "
            "not confirmatory. The Poisson arm is an algorithmically unmatched stress test."
        ),
        "rows": rows,
    }
    output = ROOT / "blobs_native_pld_envelope_comparison_overnight.json"
    digest = write_immutable_json(output, payload)
    lines = [
        "# Certified native-PLD-envelope Blobs comparison",
        "",
        payload["statistical_note"],
        "",
        "| epsilon | accountant | factor | AGS | matched DP | difference | 95% CI | Poisson DP | difference | 95% CI |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fixed = row["matched_fixed_cyclic_dpsgd"]
        stress = row["poisson_dpsgd_stress_test"]
        flo, fhi = fixed["unpaired_bootstrap_mean_difference_ci95"]
        slo, shi = stress["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {row['target_epsilon']:g} | {row['selected_accountant']} | "
            f"{row['selected_parameter_factor_over_rdp']:.4f} | "
            f"{fixed['ags_median']:.4f} | {fixed['dpsgd_median']:.4f} | "
            f"{fixed['mean_difference']:+.4f} | [{flo:+.4f}, {fhi:+.4f}] | "
            f"{stress['dpsgd_median']:.4f} | {stress['mean_difference']:+.4f} | "
            f"[{slo:+.4f}, {shi:+.4f}] |"
        )
    report = ROOT / "blobs_native_pld_envelope_comparison_overnight.md"
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
