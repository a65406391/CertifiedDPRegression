"""Named figure builders for the paper, one entry point.

    python -m common.figures <family> [args...]

AGS ablation families (read ``ags/<dataset>/ablations/runs/<experiment>/results.json``,
write ``ags/<dataset>/ablations/figures/...``; ``--results`` / ``--out-dir`` override):
    ags_comparison <dataset>      pure-DP AGS (Cauchy) vs Laplace DP-SGD
    approx_dp <dataset>           approx-DP AGS (Laplace) vs Gaussian DP-SGD
    sampling_ablation <dataset>   accuracy vs AGS sampling frequency
    hybrid_ablation <dataset>     AGS<->DP-SGD hybrid schedules

Paper figures produced by their own scripts (dispatched via runpy, arguments passed through):
    blobs_headline, blobs_approx, blobs_combined, mnist, mnist_repro,
    linear_recompute, california_agt (--variant ... --plot-only), california_repro, condition_table,
    pate_linear (--tag ... --plot-only), pate_california (--plot-only)

The four ablation families were merged verbatim from ``common/plot_*.py``; their functions are
prefixed with the family name (``plot_ags_comparison_run``, ``ags_comparison_main``, ...).
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import (
    DATASET_DISPLAY,
    LOSS_LABELS,
    aggregate_by_key,
    apply_style,
    load_runs,
    nominal_metrics,
    pad_y_limits,
    paper_legend,
)
from common.paths import get_output_dir, get_repo_root


REPO_ROOT = get_repo_root()


# =============================================================================
# ags_comparison  (from plot_ags_comparison.py)
# Plot AGS vs. Laplace pure-DP comparison for any dataset.
#
# Usage:
#     python -m common.plot_ags_comparison mnist
#     python -m common.plot_ags_comparison gpt2
# =============================================================================


# ── Per-run plotting ────────────────────────────────────────────────────────


def plot_ags_comparison_run(
    run_idx: int,
    run: dict,
    out_dir: Path,
    dataset: str,
) -> list[Path]:
    """Generate all panels for a single run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DATASET_DISPLAY.get(dataset, dataset)
    loss_lbl = LOSS_LABELS.get(dataset, "Loss")

    # Empty-run guard
    if not isinstance(run, dict) or len(run) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Run {run_idx}: no data", ha="center", va="center", fontsize=14)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"ags_comparison_run_{run_idx:03d}_empty.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        return [p]

    ags = aggregate_by_key(run.get("ags", {}), "eps")
    lap = aggregate_by_key(run.get("laplace", {}), "eps")
    nom_acc, nom_loss = nominal_metrics(run.get("nominal", []))

    outputs: list[Path] = []

    # ── Accuracy ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if ags["eps"].size:
        ax.plot(ags["eps"], ags["acc"], marker="o", label="AGS")
    if lap["eps"].size:
        ax.plot(lap["eps"], lap["acc"], marker="s", label="Laplace")
    if not np.isnan(nom_acc):
        ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"AGS vs. Laplace: Test Accuracy ({ds})")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"ags_comparison_run_{run_idx:03d}_accuracy.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    outputs.append(p)

    # ── Loss ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if ags["eps"].size:
        ax.plot(ags["eps"], ags["loss"], marker="o", label="AGS")
    if lap["eps"].size:
        ax.plot(lap["eps"], lap["loss"], marker="s", label="Laplace")
    if not np.isnan(nom_loss):
        ax.axhline(nom_loss, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"AGS vs. Laplace: Test Loss ({ds})")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(loss_lbl)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"ags_comparison_run_{run_idx:03d}_loss.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    outputs.append(p)

    # ── Runtime (skip when data is absent, e.g. GPT-2) ───────────────────
    ags_has_time = ags["eps"].size and not np.all(np.isnan(ags["time"]))
    lap_has_time = lap["eps"].size and not np.all(np.isnan(lap["time"]))
    if ags_has_time or lap_has_time:
        fig, ax = plt.subplots(figsize=(8, 8))
        if ags_has_time:
            ax.plot(ags["eps"], ags["time"], marker="o", label="AGS")
        if lap_has_time:
            ax.plot(lap["eps"], lap["time"], marker="s", label="Laplace")
        ax.set_title(f"AGS vs. Laplace: Runtime ({ds})")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"ags_comparison_run_{run_idx:03d}_runtime.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        outputs.append(p)

    # ── Diagnostics (skip when data is absent) ───────────────────────────
    has_ss = ags["eps"].size and not np.all(np.isnan(ags["ss"]))
    has_nw = ags["eps"].size and not np.all(np.isnan(ags["n_win"]))
    if has_ss or has_nw:
        fig, ax = plt.subplots(figsize=(8, 8))
        if has_ss:
            ax.plot(ags["eps"], ags["ss"], marker="o", label="Avg. Smooth Sens.")
        if has_nw:
            ax.plot(ags["eps"], ags["n_win"], marker="^", label="Num. Windows")
        ax.set_title(f"AGS Diagnostics ({ds})")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Value")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"ags_comparison_run_{run_idx:03d}_diagnostics.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        outputs.append(p)

    return outputs


# ── Public API ──────────────────────────────────────────────────────────────


def plot_ags_comparison_all(
    dataset: str,
    results_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> None:
    """Plot all AGS-vs-Laplace comparison runs for *dataset*."""
    apply_style("paper_figure")

    if results_path is None:
        results_path = Path(get_output_dir(dataset, "ags_comparison")) / "results.json"
    if out_dir is None:
        out_dir = REPO_ROOT / "ags" / dataset / "ablations" / "figures" / "ags_comparison"

    results_path = Path(results_path)
    out_dir = Path(out_dir)

    runs = load_runs(results_path)

    written: list[Path] = []
    for idx, run in enumerate(runs):
        written.extend(plot_ags_comparison_run(idx, run, out_dir, dataset))

    print(f"[{dataset}] AGS comparison: {len(runs)} run(s), {len(written)} plot(s) -> {out_dir}")
    for p in written:
        print(f"  - {p.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def ags_comparison_main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AGS vs. Laplace (pure-DP) comparison.",
    )
    parser.add_argument("dataset", help="Dataset name (mnist, gpt2)")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json (default: auto)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ags/<dataset>/ablations/figures/...)",
    )
    args = parser.parse_args()
    plot_ags_comparison_all(args.dataset, args.results, args.out_dir)


# =============================================================================
# approx_dp  (from plot_approx_dp.py)
# Plot AGS (Laplace) vs. Gaussian approximate-DP comparison for any dataset.
#
# Usage:
#     python -m common.plot_approx_dp mnist
#     python -m common.plot_approx_dp gpt2
# =============================================================================


# ── Per-run plotting ────────────────────────────────────────────────────────


def plot_approx_dp_run(
    run_idx: int,
    run: dict,
    out_dir: Path,
    dataset: str,
) -> list[Path]:
    """Generate all panels for a single run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DATASET_DISPLAY.get(dataset, dataset)
    loss_lbl = LOSS_LABELS.get(dataset, "Loss")

    # Empty-run guard
    if not isinstance(run, dict) or len(run) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Run {run_idx}: no data", ha="center", va="center", fontsize=14)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"approx_dp_run_{run_idx:03d}_empty.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        return [p]

    ags = aggregate_by_key(run.get("ags", {}), "eps")
    gauss = aggregate_by_key(run.get("gaussian", {}), "eps")
    nom_acc, nom_loss = nominal_metrics(run.get("nominal", []))

    outputs: list[Path] = []

    # ── Accuracy ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if ags["eps"].size:
        ax.plot(ags["eps"], ags["acc"], marker="o", label="AGS (Laplace)")
    if gauss["eps"].size:
        ax.plot(gauss["eps"], gauss["acc"], marker="s", label="Gaussian")
    if not np.isnan(nom_acc):
        ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Approx-DP: Test Accuracy ({ds})")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"approx_dp_run_{run_idx:03d}_accuracy.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    outputs.append(p)

    # ── Loss ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if ags["eps"].size:
        ax.plot(ags["eps"], ags["loss"], marker="o", label="AGS (Laplace)")
    if gauss["eps"].size:
        ax.plot(gauss["eps"], gauss["loss"], marker="s", label="Gaussian")
    if not np.isnan(nom_loss):
        ax.axhline(nom_loss, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Approx-DP: Test Loss ({ds})")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(loss_lbl)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"approx_dp_run_{run_idx:03d}_loss.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    outputs.append(p)

    # ── Runtime (skip when data is absent, e.g. GPT-2) ───────────────────
    ags_has_time = ags["eps"].size and not np.all(np.isnan(ags["time"]))
    gauss_has_time = gauss["eps"].size and not np.all(np.isnan(gauss["time"]))
    if ags_has_time or gauss_has_time:
        fig, ax = plt.subplots(figsize=(8, 8))
        if ags_has_time:
            ax.plot(ags["eps"], ags["time"], marker="o", label="AGS (Laplace)")
        if gauss_has_time:
            ax.plot(gauss["eps"], gauss["time"], marker="s", label="Gaussian")
        ax.set_title(f"Approx-DP: Runtime ({ds})")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"approx_dp_run_{run_idx:03d}_runtime.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        outputs.append(p)

    # ── Diagnostics (skip when data is absent) ───────────────────────────
    has_ss = ags["eps"].size and not np.all(np.isnan(ags["ss"]))
    has_nw = ags["eps"].size and not np.all(np.isnan(ags["n_win"]))
    if has_ss or has_nw:
        fig, ax = plt.subplots(figsize=(8, 8))
        if has_ss:
            ax.plot(ags["eps"], ags["ss"], marker="o", label="Avg. Smooth Sens.")
        if has_nw:
            ax.plot(ags["eps"], ags["n_win"], marker="^", label="Num. Windows")
        ax.set_title(f"AGS Diagnostics, Approx-DP ({ds})")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Value")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"approx_dp_run_{run_idx:03d}_diagnostics.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        outputs.append(p)

    return outputs


# ── Public API ──────────────────────────────────────────────────────────────


def plot_approx_dp_all(
    dataset: str,
    results_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> None:
    """Plot all approx-DP comparison runs for *dataset*."""
    apply_style("paper_figure")

    if results_path is None:
        results_path = Path(get_output_dir(dataset, "approx_dp_comparison")) / "results.json"
    if out_dir is None:
        out_dir = REPO_ROOT / "ags" / dataset / "ablations" / "figures" / "approx_dp_comparison"

    results_path = Path(results_path)
    out_dir = Path(out_dir)

    runs = load_runs(results_path)

    written: list[Path] = []
    for idx, run in enumerate(runs):
        written.extend(plot_approx_dp_run(idx, run, out_dir, dataset))

    print(f"[{dataset}] Approx-DP comparison: {len(runs)} run(s), {len(written)} plot(s) -> {out_dir}")
    for p in written:
        print(f"  - {p.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def approx_dp_main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AGS (Laplace) vs. Gaussian (approx-DP) comparison.",
    )
    parser.add_argument("dataset", help="Dataset name (mnist, gpt2)")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json (default: auto)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ags/<dataset>/ablations/figures/...)",
    )
    args = parser.parse_args()
    plot_approx_dp_all(args.dataset, args.results, args.out_dir)


# =============================================================================
# sampling_ablation  (from plot_sampling_ablation.py)
# Plot AGS sampling-frequency ablation for any dataset.
#
# Sweeps the number of AGS steps while keeping ε fixed.
# Uses log-scale x-axis.
#
# Usage:
#     python -m common.plot_sampling_ablation mnist
# =============================================================================


# ── Per-run plotting ────────────────────────────────────────────────────────


def plot_sampling_ablation_run(
    run_idx: int,
    run: dict,
    out_dir: Path,
    dataset: str,
) -> list[Path]:
    """Generate all panels for a single run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DATASET_DISPLAY.get(dataset, dataset)
    loss_lbl = LOSS_LABELS.get(dataset, "Loss")

    # Empty-run guard
    if not isinstance(run, dict) or len(run) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Run {run_idx}: no data", ha="center", va="center", fontsize=14)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"sampling_ablation_run_{run_idx:03d}_empty.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        return [p]

    ags = aggregate_by_key(run.get("ags", {}), "steps")
    nom_acc, nom_loss = nominal_metrics(run.get("nominal", []))

    # Privacy mode for subtitle context
    privacy_mode = run.get("config", {}).get("privacy_mode", "pure")
    dp_tag = "Approx-DP" if privacy_mode == "approx" else "Pure-DP"

    outputs: list[Path] = []

    # ── Accuracy ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    if ags["steps"].size:
        ax.plot(ags["steps"], ags["acc"], marker="o", label="AGS")
    if not np.isnan(nom_acc):
        ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Sampling ({ds}, {dp_tag})")
    ax.set_xlabel("AGS Steps")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    _sampling_ablation_tighten_ylim_to_baseline(ax, nom_acc)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"sampling_ablation_run_{run_idx:03d}_accuracy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # ── Loss ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    if ags["steps"].size:
        ax.plot(ags["steps"], ags["loss"], marker="o", label="AGS")
    if not np.isnan(nom_loss):
        ax.axhline(nom_loss, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Sampling ({ds}, {dp_tag})")
    ax.set_xlabel("AGS Steps")
    ax.set_ylabel(loss_lbl)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    pad_y_limits(ax)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"sampling_ablation_run_{run_idx:03d}_loss.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # ── Runtime ────────────────────────────────────────────────────────────
    if ags["steps"].size and not np.all(np.isnan(ags["time"])):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(ags["steps"], ags["time"], marker="o", label="AGS")
        ax.set_title(f"Sampling ({ds}, {dp_tag})")
        ax.set_xlabel("AGS Steps")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        pad_y_limits(ax)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"sampling_ablation_run_{run_idx:03d}_runtime.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(p)

    # ── Smooth Sensitivity ───────────────────────────────────────────────
    if ags["steps"].size and not np.all(np.isnan(ags["ss"])):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(ags["steps"], ags["ss"], marker="o", color="green", label="Avg. Smooth Sens.")
        ax.set_title(f"Sampling ({ds}, {dp_tag})")
        ax.set_xlabel("AGS Steps")
        ax.set_ylabel("Avg. $\\ell_1$ Smooth Sensitivity")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        pad_y_limits(ax)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"sampling_ablation_run_{run_idx:03d}_sensitivity.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(p)

    return outputs


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sampling_ablation_tighten_ylim_to_baseline(ax, baseline, top_pad=0.05, bottom_pad=0.15):
    """Set y-limits so *baseline* sits near the top of the plot.

    Gives *top_pad* fraction of the data range above the baseline and
    *bottom_pad* fraction below the data minimum.
    """
    if np.isnan(baseline):
        pad_y_limits(ax)
        return
    ymin, ymax = ax.get_ylim()
    data_min = ymin  # auto-computed by matplotlib from plotted data
    data_range = baseline - data_min
    if data_range <= 0:
        pad_y_limits(ax)
        return
    ax.set_ylim(data_min - bottom_pad * data_range, baseline + top_pad * data_range)


# ── Public API ──────────────────────────────────────────────────────────────


def plot_sampling_ablation_all(
    dataset: str,
    results_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> None:
    """Plot all sampling-ablation runs for *dataset*."""
    apply_style("paper_figure")

    if results_path is None:
        results_path = Path(get_output_dir(dataset, "ags_sampling_ablation")) / "results.json"
    if out_dir is None:
        out_dir = REPO_ROOT / "ags" / dataset / "ablations" / "figures" / "ags_sampling_ablation"

    results_path = Path(results_path)
    out_dir = Path(out_dir)

    runs = load_runs(results_path)

    written: list[Path] = []
    for idx, run in enumerate(runs):
        written.extend(plot_sampling_ablation_run(idx, run, out_dir, dataset))

    # ── Combined accuracy figure (one pure + one approx side by side) ────
    _sampling_ablation_plot_combined_accuracy(runs, out_dir, dataset)

    print(f"[{dataset}] Sampling ablation: {len(runs)} run(s), {len(written)} plot(s) -> {out_dir}")
    for p in written:
        print(f"  - {p.name}")


def _sampling_ablation_plot_combined_accuracy(
    runs: list[dict],
    out_dir: Path,
    dataset: str,
) -> None:
    """Create a 1×2 combined accuracy figure with shared legend.

    Picks the first pure-DP and first approx-DP run.
    """
    ds = DATASET_DISPLAY.get(dataset, dataset)

    # Find the best pure and approx run.
    # When several runs share a privacy mode, prefer the one with the
    # lowest epsilon_target (tightest privacy — the paper-relevant case).
    pure_run = approx_run = None
    pure_eps = approx_eps = float("inf")
    for run in runs:
        cfg = run.get("config", {})
        pm = cfg.get("privacy_mode", "pure")
        eps = cfg.get("epsilon_target", float("inf"))
        if pm == "pure" and eps < pure_eps:
            pure_run, pure_eps = run, eps
        elif pm == "approx" and eps < approx_eps:
            approx_run, approx_eps = run, eps

    panels = []
    if pure_run is not None:
        panels.append((pure_run, "Pure-DP"))
    if approx_run is not None:
        panels.append((approx_run, "Approx-DP"))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (run, dp_label) in zip(axes, panels):
        ags = aggregate_by_key(run.get("ags", {}), "steps")
        nom_acc, _ = nominal_metrics(run.get("nominal", []))

        if ags["steps"].size:
            ax.plot(ags["steps"], ags["acc"], marker="o", label="AGS")
        if not np.isnan(nom_acc):
            ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")

        ax.set_title(f"Sampling ({ds}, {dp_label})")
        ax.set_xlabel("AGS Steps")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        _sampling_ablation_tighten_ylim_to_baseline(ax, nom_acc)

    axes[0].set_ylabel("Accuracy")
    if len(panels) > 1:
        axes[1].set_ylabel("")

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    p = out_dir / "sampling_ablation_combined_accuracy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined: {p.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def sampling_ablation_main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AGS sampling-frequency ablation.",
    )
    parser.add_argument("dataset", help="Dataset name (mnist)")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json (default: auto)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ags/<dataset>/ablations/figures/...)",
    )
    args = parser.parse_args()
    plot_sampling_ablation_all(args.dataset, args.results, args.out_dir)


# =============================================================================
# hybrid_ablation  (from plot_hybrid_ablation.py)
# Plot hybrid Laplace+AGS ablation for any dataset.
#
# Sweeps the number of AGS steps in a hybrid (Laplace then AGS) schedule.
# Uses linear x-axis.
#
# Usage:
#     python -m common.plot_hybrid_ablation mnist
# =============================================================================


# ── Per-run plotting ────────────────────────────────────────────────────────


def plot_hybrid_ablation_run(
    run_idx: int,
    run: dict,
    out_dir: Path,
    dataset: str,
) -> list[Path]:
    """Generate all panels for a single run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DATASET_DISPLAY.get(dataset, dataset)
    loss_lbl = LOSS_LABELS.get(dataset, "Loss")

    # Empty-run guard
    if not isinstance(run, dict) or len(run) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Run {run_idx}: no data", ha="center", va="center", fontsize=14)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"hybrid_ablation_run_{run_idx:03d}_empty.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        return [p]

    has_bidirectional = "ags_first" in run or "laplace_first" in run
    if has_bidirectional:
        ags_first = aggregate_by_key(run.get("ags_first", {}), "steps")
        laplace_first = aggregate_by_key(run.get("laplace_first", {}), "steps")
    else:
        # Legacy fallback (where "ags" is what was run: Laplace -> AGS)
        laplace_first = aggregate_by_key(run.get("ags", {}), "steps")
        ags_first = {
            "steps": np.array([]),
            "acc": np.array([]),
            "loss": np.array([]),
            "time": np.array([]),
            "ss": np.array([]),
            "n_win": np.array([]),
        }

    nom_acc, nom_loss = nominal_metrics(run.get("nominal", []))

    # Privacy mode for subtitle context
    privacy_mode = run.get("config", {}).get("privacy_mode", "pure")
    dp_tag = "Approx-DP" if privacy_mode == "approx" else "Pure-DP"

    outputs: list[Path] = []

    # ── Accuracy ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    if ags_first["steps"].size:
        ax.plot(ags_first["steps"], ags_first["acc"], marker="o", color="#1f77b4", label="AGS -> DP-SGD")
    if laplace_first["steps"].size:
        ax.plot(
            laplace_first["steps"], laplace_first["acc"], marker="s", color="#ff7f0e", label="DP-SGD -> AGS"
        )
    if not np.isnan(nom_acc):
        ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Hybrid ({ds}, {dp_tag})")
    ax.set_xlabel("First Phase Steps (s)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    _hybrid_ablation_tighten_ylim_to_baseline(ax, nom_acc)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"hybrid_ablation_run_{run_idx:03d}_accuracy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # ── Loss ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    if ags_first["steps"].size:
        ax.plot(ags_first["steps"], ags_first["loss"], marker="o", color="#1f77b4", label="AGS -> DP-SGD")
    if laplace_first["steps"].size:
        ax.plot(
            laplace_first["steps"], laplace_first["loss"], marker="s", color="#ff7f0e", label="DP-SGD -> AGS"
        )
    if not np.isnan(nom_loss):
        ax.axhline(nom_loss, ls="--", lw=3.0, color="k", label="Non-Private")
    ax.set_title(f"Hybrid ({ds}, {dp_tag})")
    ax.set_xlabel("First Phase Steps (s)")
    ax.set_ylabel(loss_lbl)
    ax.grid(True, alpha=0.25)
    pad_y_limits(ax)
    paper_legend(ax)
    fig.tight_layout(pad=1.0)
    p = out_dir / f"hybrid_ablation_run_{run_idx:03d}_loss.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # ── Runtime ──────────────────────────────────────────────────────────
    has_time = (ags_first["steps"].size and not np.all(np.isnan(ags_first["time"]))) or (
        laplace_first["steps"].size and not np.all(np.isnan(laplace_first["time"]))
    )
    if has_time:
        fig, ax = plt.subplots(figsize=(6, 6))
        if ags_first["steps"].size and not np.all(np.isnan(ags_first["time"])):
            ax.plot(ags_first["steps"], ags_first["time"], marker="o", color="#1f77b4", label="AGS -> DP-SGD")
        if laplace_first["steps"].size and not np.all(np.isnan(laplace_first["time"])):
            ax.plot(
                laplace_first["steps"],
                laplace_first["time"],
                marker="s",
                color="#ff7f0e",
                label="DP-SGD -> AGS",
            )
        ax.set_title(f"Hybrid ({ds}, {dp_tag})")
        ax.set_xlabel("First Phase Steps (s)")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.25)
        pad_y_limits(ax)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"hybrid_ablation_run_{run_idx:03d}_runtime.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(p)

    # ── Smooth Sensitivity ───────────────────────────────────────────────
    has_ss = (ags_first["steps"].size and not np.all(np.isnan(ags_first["ss"]))) or (
        laplace_first["steps"].size and not np.all(np.isnan(laplace_first["ss"]))
    )
    if has_ss:
        fig, ax = plt.subplots(figsize=(6, 6))
        if ags_first["steps"].size and not np.all(np.isnan(ags_first["ss"])):
            ax.plot(ags_first["steps"], ags_first["ss"], marker="o", color="#1f77b4", label="AGS -> DP-SGD")
        if laplace_first["steps"].size and not np.all(np.isnan(laplace_first["ss"])):
            ax.plot(
                laplace_first["steps"],
                laplace_first["ss"],
                marker="s",
                color="#ff7f0e",
                label="DP-SGD -> AGS",
            )
        ax.set_title(f"Hybrid ({ds}, {dp_tag})")
        ax.set_xlabel("First Phase Steps (s)")
        ax.set_ylabel("Avg. $\\ell_1$ Smooth Sensitivity")
        ax.grid(True, alpha=0.25)
        pad_y_limits(ax)
        paper_legend(ax)
        fig.tight_layout(pad=1.0)
        p = out_dir / f"hybrid_ablation_run_{run_idx:03d}_sensitivity.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(p)

    return outputs


# ── Helpers ─────────────────────────────────────────────────────────────────


def _hybrid_ablation_tighten_ylim_to_baseline(ax, baseline, top_pad=0.05, bottom_pad=0.15):
    """Set y-limits so *baseline* sits near the top of the plot."""
    if np.isnan(baseline):
        pad_y_limits(ax)
        return
    ymin, ymax = ax.get_ylim()
    data_min = ymin
    data_range = baseline - data_min
    if data_range <= 0:
        pad_y_limits(ax)
        return
    ax.set_ylim(data_min - bottom_pad * data_range, baseline + top_pad * data_range)


# ── Public API ──────────────────────────────────────────────────────────────


def plot_hybrid_ablation_all(
    dataset: str,
    results_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> None:
    """Plot all hybrid-ablation runs for *dataset*."""
    apply_style("paper_figure")

    if results_path is None:
        results_path = Path(get_output_dir(dataset, "hybrid_ablation")) / "results.json"
    if out_dir is None:
        out_dir = REPO_ROOT / "ags" / dataset / "ablations" / "figures" / "hybrid_ablation"

    results_path = Path(results_path)
    out_dir = Path(out_dir)

    runs = load_runs(results_path)

    written: list[Path] = []
    for idx, run in enumerate(runs):
        written.extend(plot_hybrid_ablation_run(idx, run, out_dir, dataset))

    # ── Combined accuracy figure (one pure + one approx side by side) ────
    _hybrid_ablation_plot_combined_accuracy(runs, out_dir, dataset)

    print(f"[{dataset}] Hybrid ablation: {len(runs)} run(s), {len(written)} plot(s) -> {out_dir}")
    for p in written:
        print(f"  - {p.name}")


def _hybrid_ablation_plot_combined_accuracy(
    runs: list[dict],
    out_dir: Path,
    dataset: str,
) -> None:
    """Create a 1×2 combined accuracy figure with shared legend.

    Picks the first pure-DP and first approx-DP run that have
    bidirectional data (ags_first + laplace_first).
    """
    ds = DATASET_DISPLAY.get(dataset, dataset)

    # Prefer runs with bidirectional data
    pure_run = approx_run = None
    for run in runs:
        pm = run.get("config", {}).get("privacy_mode", "pure")
        has_bidir = "ags_first" in run or "laplace_first" in run
        if pm == "pure" and pure_run is None and has_bidir:
            pure_run = run
        elif pm == "approx" and approx_run is None and has_bidir:
            approx_run = run
    # Fallback: accept any run if no bidirectional ones found
    if pure_run is None or approx_run is None:
        for run in runs:
            pm = run.get("config", {}).get("privacy_mode", "pure")
            if pm == "pure" and pure_run is None:
                pure_run = run
            elif pm == "approx" and approx_run is None:
                approx_run = run

    panels = []
    if pure_run is not None:
        panels.append((pure_run, "Pure-DP"))
    if approx_run is not None:
        panels.append((approx_run, "Approx-DP"))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (run, dp_label) in zip(axes, panels):
        has_bidirectional = "ags_first" in run or "laplace_first" in run
        if has_bidirectional:
            ags_first = aggregate_by_key(run.get("ags_first", {}), "steps")
            laplace_first = aggregate_by_key(run.get("laplace_first", {}), "steps")
        else:
            laplace_first = aggregate_by_key(run.get("ags", {}), "steps")
            ags_first = {k: np.array([]) for k in ["steps", "acc", "loss", "time", "ss", "n_win"]}

        nom_acc, _ = nominal_metrics(run.get("nominal", []))

        if ags_first["steps"].size:
            ax.plot(ags_first["steps"], ags_first["acc"], marker="o", color="#1f77b4", label="AGS -> DP-SGD")
        if laplace_first["steps"].size:
            ax.plot(
                laplace_first["steps"],
                laplace_first["acc"],
                marker="s",
                color="#ff7f0e",
                label="DP-SGD -> AGS",
            )
        if not np.isnan(nom_acc):
            ax.axhline(nom_acc, ls="--", lw=3.0, color="k", label="Non-Private")

        ax.set_title(f"Hybrid ({ds}, {dp_label})")
        ax.set_xlabel("First Phase Steps (s)")
        ax.grid(True, alpha=0.25)
        _hybrid_ablation_tighten_ylim_to_baseline(ax, nom_acc)

    axes[0].set_ylabel("Accuracy")
    if len(panels) > 1:
        axes[1].set_ylabel("")

    # Merge legend handles from all axes (deduplicate by label)
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    handles = list(seen.values())
    labels = list(seen.keys())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    p = out_dir / "hybrid_ablation_combined_accuracy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined: {p.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def hybrid_ablation_main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot hybrid Laplace+AGS ablation.",
    )
    parser.add_argument("dataset", help="Dataset name (mnist)")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json (default: auto)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ags/<dataset>/ablations/figures/...)",
    )
    args = parser.parse_args()
    plot_hybrid_ablation_all(args.dataset, args.results, args.out_dir)


# =============================================================================
# dispatcher
# =============================================================================

FAMILIES = {
    "ags_comparison": ags_comparison_main,
    "approx_dp": approx_dp_main,
    "sampling_ablation": sampling_ablation_main,
    "hybrid_ablation": hybrid_ablation_main,
}

SCRIPTS = {
    "blobs_headline": "ags.blobs.plot_headline",
    "blobs_approx": "ags.blobs.plot_approx",
    "blobs_combined": "ags.blobs.plot_combined",
    "mnist": "ags.mnist.plot_headline",
    "mnist_repro": "ags.mnist.plot_reproduce_paper",
    "linear_recompute": "agtr.linear.plot",
    "california_agt": "agtr.california.release",
    "california_repro": "agtr.california.repro.repro_figure2",
    "condition_table": "agtr.condition_table",
    "pate_linear": "agtr.linear.pate.release",
    "pate_california": "agtr.california.pate.release",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("families:", ", ".join(FAMILIES))
        print("scripts: ", ", ".join(SCRIPTS))
        return
    name, rest = argv[0], argv[1:]
    if name in FAMILIES:
        sys.argv = [f"figures {name}"] + rest
        FAMILIES[name]()
    elif name in SCRIPTS:
        sys.argv = [SCRIPTS[name]] + rest
        runpy.run_module(SCRIPTS[name], run_name="__main__", alter_sys=True)
    else:
        raise SystemExit(f"unknown figure family {name!r}; known: {sorted(FAMILIES) + sorted(SCRIPTS)}")


if __name__ == "__main__":
    main()
