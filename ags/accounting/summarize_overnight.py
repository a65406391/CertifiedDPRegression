"""Create the immutable machine-readable and Markdown overnight summaries."""

from __future__ import annotations

import json

import numpy as np

from ags.accounting.paths import artifact
from common.accounting.artifacts import write_immutable_json


def paired_summary(ags, dp):
    a, d = np.asarray(ags, float), np.asarray(dp, float)
    n = min(len(a), len(d))
    diff = a[:n] - d[:n]
    rng = np.random.RandomState(20260826)
    boots = np.asarray([diff[rng.randint(0, n, n)].mean() for _ in range(20000)])
    unpaired = np.asarray(
        [a[rng.randint(0, n, n)].mean() - d[rng.randint(0, n, n)].mean() for _ in range(20000)]
    )
    return {
        "n": n,
        "ags_median": float(np.median(a[:n])),
        "dpsgd_median": float(np.median(d[:n])),
        "paired_mean_difference": float(diff.mean()),
        "paired_mean_difference_ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
        "unpaired_mean_difference_ci95": np.quantile(unpaired, [0.025, 0.975]).tolist(),
        "ags_win_fraction": float(np.mean(diff > 0)),
    }


def main():
    blobs = json.loads((artifact("blobs_matched_rdp_lp_complete_overnight.json")).read_text())
    blobs_generic = json.loads((artifact("blobs_cross_radius_lp_complete_overnight.json")).read_text())
    pca = json.loads((artifact("pca_matched_rdp_overnight.json")).read_text())
    blob_rows = []
    blob_fixed_rows = []
    for row in blobs["rows"]:
        fixed_cell = row["ags_public_weight_bias_allocations"]["4.0"]["2.0"]
        fixed_panel = fixed_cell["panels"]["certified_tail"]
        fixed_recalibrated = paired_summary(
            fixed_panel["public_bias_recalibrated_accuracies"],
            row["dpsgd"]["public_bias_recalibrated_accuracies"],
        )
        blob_fixed_rows.append(
            {
                "epsilon": row["target_epsilon"],
                "beta_fraction": "4.0",
                "bias_ratio": "2.0",
                "privacy_interval": fixed_cell["interval_certification"],
                "public_bias_recalibrated": fixed_recalibrated,
                **paired_summary(fixed_panel["accuracies"], row["dpsgd"]["accuracies"]),
            }
        )
        candidates = []
        for fraction, allocations in row["ags_public_weight_bias_allocations"].items():
            selected = allocations["selected_by_public_validation_certified_tail"]
            ratio = str(selected["bias_ratio"])
            panel = allocations[ratio]["panels"]["certified_tail"]
            candidates.append(
                (selected["validation_median_accuracy"], fraction, f"bias_ratio_{ratio}", panel)
            )
        for fraction, allocations in row.get("ags_public_coordinate_allocations", {}).items():
            selected = allocations["selected_by_public_validation"]
            name = selected["allocation"]
            candidates.append((selected["validation_median_accuracy"], fraction, name, allocations[name]))
        # beta fraction is a predeclared mechanism grid. Retain every cell and mark
        # this cross-beta selection exploratory because it uses public validation.
        _, fraction, allocation, panel = max(candidates, key=lambda item: item[0])
        stats = paired_summary(panel["accuracies"], row["dpsgd"]["accuracies"])
        recalibrated_stats = paired_summary(
            panel["public_bias_recalibrated_accuracies"],
            row["dpsgd"]["public_bias_recalibrated_accuracies"],
        )
        blob_rows.append(
            {
                "epsilon": row["target_epsilon"],
                "beta_fraction": fraction,
                "public_allocation": allocation,
                "public_bias_recalibrated": recalibrated_stats,
                **stats,
            }
        )
    pca_rows = [
        {
            "epsilon": row["target_epsilon"],
            **paired_summary(row["ags"]["accuracies"], row["dpsgd"]["accuracies"]),
        }
        for row in pca["rows"]
    ]
    blob_generic_rows = [
        {
            "epsilon": row["target_epsilon"],
            **paired_summary(
                row["ags"]["10"]["certified"]["accuracies"],
                row["dpsgd_matched_generic_accuracies"],
            ),
        }
        for row in blobs_generic["rows"]
    ]
    payload = {
        "schema": 1,
        "blobs_matched_generic": {
            "status": "certified_conservative_generic_pld",
            "rows": blob_generic_rows,
        },
        "blobs_matched_rdp": {"status": blobs["status"], "rows": blob_rows},
        "blobs_matched_rdp_fixed_reference_cell": {
            "status": blobs["status"],
            "rows": blob_fixed_rows,
            "beta_fraction": 4.0,
            "bias_ratio": 2.0,
            "selection_status": "chosen during exploratory method development",
        },
        "pca_mnist": {
            "status": pca["status"],
            "rows": pca_rows,
            "sensitivity_caveat": pca["sensitivity_caveat"],
        },
        "selection_note": (
            "Every Blobs allocation/beta cell is individually private and retained. "
            "The displayed cell is an exploratory research selection using a disjoint "
            "public synthetic validation set, but selection evaluates private-trained "
            "candidate releases and is not itself privacy-accounted; it is therefore "
            "not a deployable selected mechanism. The fixed reference cell was chosen "
            "during exploratory development, so its bootstrap interval is descriptive, "
            "not confirmatory. PCA has no test-driven selection."
        ),
        "audit_corrections": [
            "cross-radius T_k/T_(k+1) certificate",
            "retain adjacent pairs beyond sparse gaps",
            "Blobs final-release diameter N=200, not a function of F",
            "20 record-touching cyclic DP-SGD releases by adaptive parallel composition",
            "continuous coordinate Laplace RDP interval enclosure",
        ],
    }
    write_immutable_json(artifact("overnight_matched_summary.json"), payload)

    lines = [
        "# Overnight AGS versus DP-SGD summary",
        "",
        payload["selection_note"],
        "",
        "Table confidence intervals independently resample the AGS and "
        "DP-SGD trials; paired-seed intervals are also retained in JSON.",
        "",
        "## Audit corrections applied",
        "",
        "- Corrected cross-radius Abar uses T_k and T_(k+1); widths remain diagnostics.",
        "- Sparse adjacent pairs after a missing radius are retained.",
        "- The final Blobs dataset diameter is N=200; F is only a concretization frequency.",
        "- Fixed cyclic DP-SGD composes 20 record-touching releases via adaptive parallel composition, not all 200 updates.",
        "- Coordinate smooth-Laplace RDP maxima are enclosed continuously by interval branch-and-bound.",
        "",
        "## Blobs — matched conservative generic PLD (certified)",
        "",
        "| epsilon | AGS median | DP-SGD median | paired mean difference | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in blob_generic_rows:
        lo, hi = row["paired_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {row['ags_median']:.4f} | "
            f"{row['dpsgd_median']:.4f} | {row['paired_mean_difference']:+.4f} | "
            f"[{row['unpaired_mean_difference_ci95'][0]:+.4f}, "
            f"{row['unpaired_mean_difference_ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "## Blobs — matched mechanism-specific RDP (fixed reference cell)",
        "",
        "Exploratory-development reference: beta fraction 4, public bias-to-weight epsilon ratio 2.",
        "",
        "| epsilon | raw AGS | raw DP-SGD | public-bias AGS | public-bias DP-SGD | raw paired difference | 95% CI | certified epsilon upper |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob_fixed_rows:
        lo, hi = row["paired_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {row['ags_median']:.4f} | "
            f"{row['dpsgd_median']:.4f} | "
            f"{row['public_bias_recalibrated']['ags_median']:.4f} | "
            f"{row['public_bias_recalibrated']['dpsgd_median']:.4f} | "
            f"{row['paired_mean_difference']:+.4f} | "
            f"[{row['unpaired_mean_difference_ci95'][0]:+.4f}, "
            f"{row['unpaired_mean_difference_ci95'][1]:+.4f}] | "
            f"{row['privacy_interval']['epsilon_upper']:.6f} |"
        )
    lines += [
        "",
        "## Blobs — exploratory public-validation selection",
        "",
        "| epsilon | beta fraction | public allocation | AGS median | DP-SGD median | paired mean difference | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob_rows:
        lo, hi = row["paired_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {row['beta_fraction']} | {row['public_allocation']} | "
            f"{row['ags_median']:.4f} | {row['dpsgd_median']:.4f} | "
            f"{row['paired_mean_difference']:+.4f} | "
            f"[{row['unpaired_mean_difference_ci95'][0]:+.4f}, "
            f"{row['unpaired_mean_difference_ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "## PCA-MNIST — common analytic RDP",
        "",
        "| epsilon | AGS median | DP-SGD median | paired mean difference | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pca_rows:
        lo, hi = row["paired_mean_difference_ci95"]
        lines.append(
            f"| {row['epsilon']:g} | {row['ags_median']:.4f} | "
            f"{row['dpsgd_median']:.4f} | {row['paired_mean_difference']:+.4f} | "
            f"[{row['unpaired_mean_difference_ci95'][0]:+.4f}, "
            f"{row['unpaired_mean_difference_ci95'][1]:+.4f}] |"
        )
    lines += ["", "> PCA sensitivity status: " + pca["status"], "", pca["sensitivity_caveat"], ""]
    report = artifact("overnight_matched_summary.md")
    if report.exists():
        raise FileExistsError(report)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
