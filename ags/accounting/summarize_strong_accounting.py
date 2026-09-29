"""Summarize the strongest native-accountant and public-postprocessing panels."""

from __future__ import annotations

import json

import numpy as np

from ags.accounting.paths import artifact
from common.accounting.artifacts import write_immutable_json


def comparison(ags, dp):
    ags, dp = np.asarray(ags, float), np.asarray(dp, float)
    rng = np.random.RandomState(20260826)
    boots = np.asarray(
        [
            ags[rng.randint(0, len(ags), len(ags))].mean() - dp[rng.randint(0, len(dp), len(dp))].mean()
            for _ in range(20000)
        ]
    )
    return {
        "ags_median": float(np.median(ags)),
        "dpsgd_median": float(np.median(dp)),
        "mean_difference": float(ags.mean() - dp.mean()),
        "unpaired_bootstrap_mean_difference_ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
        "ags_trials": len(ags),
        "dpsgd_trials": len(dp),
    }


def main():
    blobs = json.loads((artifact("blobs_matched_rdp_lp_complete_overnight.json")).read_text())
    dp_pld = json.loads((artifact("blobs_dpsgd_pld_reference_overnight.json")).read_text())
    dp_poisson = json.loads((artifact("blobs_poisson_dpsgd_reference_overnight.json")).read_text())
    projection = json.loads((artifact("blobs_public_projection_overnight.json")).read_text())
    pca = json.loads((artifact("pca_strong_accounting_overnight.json")).read_text())
    pca_early = json.loads((artifact("pca_early_stop_strong_corrected_overnight.json")).read_text())
    dp_rows = {row["target_epsilon"]: row for row in dp_pld["rows"]}
    poisson_rows = {row["target_epsilon"]: row for row in dp_poisson["rows"]}
    projected_rows = {row["target_epsilon"]: row for row in projection["rows"]}

    blob_rows = []
    for row in blobs["rows"]:
        target = row["target_epsilon"]
        ags = row["ags_public_weight_bias_allocations"]["4.0"]["2.0"]
        panel = ags["panels"]["certified_tail"]
        dp = dp_rows[target]
        poisson = poisson_rows[target]
        projected = projected_rows[target]
        projected_ags = projected["ags_public_weight_bias_allocations"]["4.0"]["2.0"]
        projected_dp = projected["dpsgd_native_pld"]
        blob_rows.append(
            {
                "epsilon": target,
                "privacy_interval": ags["interval_certification"],
                "raw": comparison(panel["accuracies"], dp["accuracies"]),
                "public_bias_recalibrated": comparison(
                    panel["public_bias_recalibrated_accuracies"],
                    dp["public_bias_recalibrated_accuracies"],
                ),
                "public_cube_projected_raw": comparison(
                    projected_ags["accuracies"], projected_dp["accuracies"]
                ),
                "latest_completed_boxes_unprojected_raw": comparison(
                    projected_ags["latest_box_unprojected_accuracies"],
                    dp["accuracies"],
                ),
                "public_cube_projected_and_bias_recalibrated": comparison(
                    projected_ags["public_bias_recalibrated_accuracies"],
                    projected_dp["public_bias_recalibrated_accuracies"],
                ),
                "poisson_dpsgd_stress_test_raw": comparison(
                    projected_ags["latest_box_unprojected_accuracies"],
                    poisson["accuracies"],
                ),
                "poisson_dpsgd_stress_test_public_bias_recalibrated": comparison(
                    projected_ags["latest_box_unprojected_public_bias_recalibrated_accuracies"],
                    poisson["public_bias_recalibrated_accuracies"],
                ),
            }
        )

    pca_rows = []
    for row in pca["rows"]:
        for window, ags in row["ags"]["windows"].items():
            dp = row["dpsgd"]["windows"][window]
            pca_rows.append(
                {
                    "epsilon": row["target_epsilon"],
                    "window": int(window),
                    "raw": comparison(ags["accuracies"], dp["accuracies"]),
                    "public_class_bias_recalibrated": comparison(
                        ags["public_class_bias_recalibrated_accuracies"],
                        dp["public_class_bias_recalibrated_accuracies"],
                    ),
                }
            )

    pca_early_rows = []
    for step_row in pca_early["rows"]:
        for row in step_row["privacy_rows"]:
            pca_early_rows.append(
                {
                    "steps": step_row["steps"],
                    "sensitivity_status": step_row["sensitivity_status"],
                    "nominal_accuracy": step_row["nominal_accuracy"],
                    "epsilon": row["target_epsilon"],
                    "raw": comparison(row["ags"]["accuracies"], row["dpsgd"]["accuracies"]),
                    "public_class_bias_recalibrated": comparison(
                        row["ags"]["public_class_bias_recalibrated_accuracies"],
                        row["dpsgd"]["public_class_bias_recalibrated_accuracies"],
                    ),
                }
            )

    payload = {
        "schema": 1,
        "blobs": {
            "status": blobs["status"],
            "ags_fixed_reference": {
                "beta_fraction": 4.0,
                "bias_epsilon_ratio": 2.0,
                "selection_status": "chosen during exploratory method development",
            },
            "dpsgd_accountant": "native replace-one Gaussian PLD",
            "poisson_stress_test_caveat": dp_poisson["comparison_caveat"],
            "rows": blob_rows,
        },
        "pca_mnist": {
            "status": pca["status"],
            "rows": pca_rows,
            "sensitivity_caveat": pca["sensitivity_caveat"],
        },
        "pca_mnist_early_stop": {
            "status": pca_early["status"],
            "rows": pca_early_rows,
            "sensitivity_caveat": pca_early["sensitivity_caveat"],
        },
        "statistical_note": (
            "Intervals independently bootstrap mechanism trials. The fixed Blobs "
            "cell was chosen during exploratory development and the PCA tail-window "
            "grid is exploratory; intervals are descriptive, not confirmatory."
        ),
    }
    write_immutable_json(artifact("overnight_strong_summary.json"), payload)

    lines = [
        "# Strong-accounting AGS versus DP-SGD summary",
        "",
        payload["statistical_note"],
        "",
        "## Blobs: fixed AGS cell versus native Gaussian-PLD DP-SGD",
        "",
        "The AGS reference fixes beta fraction 4 and bias/weight epsilon ratio 2.",
        "",
        "| epsilon | AGS raw | DP raw | difference | 95% CI | "
        "latest-box AGS | projected AGS | projected DP | projected difference |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob_rows:
        raw, projected = row["raw"], row["public_cube_projected_raw"]
        latest = row["latest_completed_boxes_unprojected_raw"]
        lo, hi = raw["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {raw['ags_median']:.4f} | "
            f"{raw['dpsgd_median']:.4f} | {raw['mean_difference']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {latest['ags_median']:.4f} | "
            f"{projected['ags_median']:.4f} | "
            f"{projected['dpsgd_median']:.4f} | "
            f"{projected['mean_difference']:+.4f} |"
        )
    lines += [
        "",
        "## Blobs: standard Poisson-DP-SGD algorithmic stress test",
        "",
        dp_poisson["comparison_caveat"],
        "",
        "| epsilon | fixed-cell AGS | Poisson DP-SGD | difference | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in blob_rows:
        stress = row["poisson_dpsgd_stress_test_raw"]
        lo, hi = stress["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {stress['ags_median']:.4f} | "
            f"{stress['dpsgd_median']:.4f} | {stress['mean_difference']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += [
        "",
        "## PCA-MNIST: early-stop stress test",
        "",
        "The one-step rows have exact unconditional sensitivity; longer rows are "
        "retained in JSON with the non-expansiveness caveat.",
        "",
        "| epsilon | one-step AGS | one-step DP-SGD | difference | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pca_early_rows:
        if row["steps"] != 1:
            continue
        raw = row["raw"]
        lo, hi = raw["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {raw['ags_median']:.4f} | "
            f"{raw['dpsgd_median']:.4f} | {raw['mean_difference']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += [
        "",
        "## PCA-MNIST: native Laplace PLD versus exact Gaussian composition",
        "",
        "All public tail windows are retained below.",
        "",
        "| epsilon | tail window | AGS raw | DP raw | difference | 95% CI | "
        "bias-calibrated AGS | bias-calibrated DP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pca_rows:
        raw, recal = row["raw"], row["public_class_bias_recalibrated"]
        lo, hi = raw["unpaired_bootstrap_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {row['window']} | "
            f"{raw['ags_median']:.4f} | {raw['dpsgd_median']:.4f} | "
            f"{raw['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{recal['ags_median']:.4f} | {recal['dpsgd_median']:.4f} |"
        )
    lines += ["", "> " + pca["sensitivity_caveat"], ""]
    report = artifact("overnight_strong_summary.md")
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
