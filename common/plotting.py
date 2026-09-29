"""Shared plotting: the paper's single matplotlib style plus figure primitives.

Style (formerly ``common/plotting.py``): ``apply()``, ``apply_combined()``, ``shared_legend()``,
``FIGSIZE*``, ``SERIES``, ``LEGEND_BELOW*``, ``COLUMN_WIDTH_IN`` -- the rcParams the AGS blobs
figures were drawn with, now used by every figure. Colour is deliberately not part of the
style: each experiment keeps its own palette (``METHOD_COLORS``, ``BATCH_COLORS``, ...).

Primitives (formerly ``common/plotting.py``): ``apply_style()`` (thin wrapper kept for old call
sites), ``save_figure()``, ``plot_staircase_and_sensitivity()``, ``plot_tradeoff()``, result
loading/aggregation helpers. Named paper-figure builders live in ``common/figures.py``.
"""

from __future__ import annotations

# --- style (from plotting.py) ---------------------------------------------
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" style)

STYLE = ["science", "no-latex", "grid"]

RC = {
    "font.size": 19,
    "axes.labelsize": 23,
    "axes.titlesize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 13,
    "lines.linewidth": 3.0,
    "lines.markersize": 9.0,
    "axes.linewidth": 1.8,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.6",
    "grid.alpha": 0.3,
    # scienceplots cycles linestyles; meaning is encoded in colour and dash at the
    # call site instead, so the cycle is pinned to a solid line
    "axes.prop_cycle": matplotlib.cycler("linestyle", ["-"]),
}

#: single-panel figure; a two-panel pair uses (14.0, 6.2)
FIGSIZE = (9.6, 6.2)
FIGSIZE_PAIR = (14.0, 6.2)

#: Combined-row boxes, by panel count. Three panels get a wider box than two so the
#: panels themselves stay a usable width; both are 6 inches tall, so the rows sit on
#: the page as a set.
COMBINED_H = 6.0
COMBINED_W = {2: 16.0, 3: 20.0}


def combined_figsize(n_panels=3):
    """The row box for `n_panels`."""
    return (COMBINED_W.get(int(n_panels), 8.0 * int(n_panels)), COMBINED_H)


#: for anything that reads the box directly
FIGSIZE_COMBINED = combined_figsize(3)

#: keys scaled with the panel width when a row is narrower than a single figure
_SCALED = (
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "lines.linewidth",
    "lines.markersize",
    "axes.linewidth",
    "xtick.major.size",
    "ytick.major.size",
    "xtick.major.width",
    "ytick.major.width",
)

#: A4 minus 25 mm margins: the width a full-width figure occupies on the page
A4_TEXT_WIDTH_IN = 6.3

#: one column of a two-column A4 page, for figures that sit beside body text
COLUMN_WIDTH_IN = 3.35

#: what the base font should measure ON THE PAGE, in points, once a combined row has
#: been scaled down to the text width. Axis labels, titles and the legend follow from
#: it in the same proportions the single figures use.
PRINT_TARGET_PT = 9.0

#: Axis labels, tick labels and the legend get their own printed targets rather than
#: being scaled in proportion to the body text. Left proportional, the legend lands near
#: 6 pt (too small to carry) while the axis labels land near 11 pt (larger than they
#: need to be, and they are what crowds a multi-panel row).
LABEL_PRINT_PT = 8.0  #: "Privacy Budget eps", "Test MSE", ...
TICK_PRINT_PT = 7.5  #: the numbers on the axes
LEGEND_PRINT_PT = 8.0

#: The series treatment the AGS figures use: a white keyline around each marker, which
#: keeps overlapping points readable where curves converge. Spread into every plotted
#: series so the two experiment families draw their lines the same way.
SERIES = dict(markeredgecolor="white", markeredgewidth=1.2)

#: legend below the axes, which is where every figure in the paper puts it
LEGEND_BELOW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.155),
    columnspacing=1.1,
    handlelength=1.9,
    handletextpad=0.5,
    labelspacing=0.4,
)


def apply(**overrides):
    """Apply the paper style. Keyword overrides go straight into rcParams.

    PAPER_DPI in the environment overrides the raster resolution, so a whole set of
    figures can be re-rendered at press resolution without editing anything:

        PAPER_DPI=600 python -m ags.blobs.plot_combined

    It affects PNG only -- the PDFs are vector and carry no resolution.
    """
    plt.style.use(STYLE)
    plt.rcParams.update(RC)
    plt.rcParams["savefig.dpi"] = float(os.environ.get("PAPER_DPI", RC["savefig.dpi"]))
    if overrides:
        plt.rcParams.update(overrides)


def apply_combined(
    n_panels,
    figsize=None,
    print_target_pt=PRINT_TARGET_PT,
    page_width_in=A4_TEXT_WIDTH_IN,
    label_pt=LABEL_PRINT_PT,
    tick_pt=TICK_PRINT_PT,
    legend_pt=LEGEND_PRINT_PT,
    **overrides,
):
    """The paper style, sized so a combined row is legible once it is on the page.

    Font size is in points against the figure's own inches, so a wide figure loses type
    when the page scales it down to the text width. A 24 x 6 row placed across A4's
    6.3-inch text block shrinks by 3.8x, and 19 pt type would land at 5 pt.

    So the type is scaled UP by exactly that factor:

        scale = (print_target_pt / RC["font.size"]) * (fig_width / page_width_in)

    which puts the base font at `print_target_pt` on the page whatever the row's width.
    Line and marker sizes follow that scale; axis labels, tick labels and the legend are
    then set from their own printed targets, because in a row they are what crowds the
    panels rather than what carries the content.
    """
    apply()
    fig_w = (figsize or combined_figsize(n_panels))[0]
    scale = (print_target_pt / RC["font.size"]) * (fig_w / page_width_in)
    plt.rcParams.update({k: RC[k] * scale for k in _SCALED})
    page = fig_w / page_width_in  # points in the file per point on the page
    plt.rcParams["axes.labelsize"] = label_pt * page
    plt.rcParams["xtick.labelsize"] = tick_pt * page
    plt.rcParams["ytick.labelsize"] = tick_pt * page
    plt.rcParams["legend.fontsize"] = legend_pt * page
    plt.rcParams["savefig.dpi"] = float(os.environ.get("PAPER_DPI", RC["savefig.dpi"]))
    if overrides:
        plt.rcParams.update(overrides)
    return scale


def shared_legend(fig, axes, ncol=None, outside=True, **kw):
    """One legend for the whole row, below the panels, with no duplicate entries.

    Panels in a combined row plot the same series, so a per-panel legend would repeat
    itself and eat the plot area. Handles are collected in first-seen order across every
    panel, so a series that appears in only one panel still shows up.

    With `outside=True` on a constrained-layout figure the legend is placed OUTSIDE the
    axes grid and the layout engine reserves exactly the space it needs -- which is the
    difference between panels filling ~58% of the figure and ~75%. Hand-set margins
    cannot do this, because the space labels need is only known after they are drawn.
    """
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    loc = kw.pop("loc", None) or ("outside lower center" if outside else "lower center")
    return fig.legend(handles, labels, loc=loc, ncol=ncol or len(labels), **{**LEGEND_BELOW_KW, **kw})


LEGEND_BELOW_KW = dict(
    columnspacing=1.1, handlelength=1.9, handletextpad=0.5, labelspacing=0.4, frameon=False
)


# --- primitives ------------------------------------------------------------

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Style presets
# =============================================================================

PAPER_STYLE = {
    "font.size": 24,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
    "figure.titlesize": 28,
    "lines.linewidth": 4,
    "axes.linewidth": 2,
}

DRAFT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
}

PAPER_FIGURE_STYLE = {
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "figure.titlesize": 24,
    "lines.linewidth": 3.5,
    "axes.linewidth": 1.8,
    "lines.markersize": 9,
}


# =============================================================================
# Dataset metadata
# =============================================================================

DATASET_DISPLAY: dict[str, str] = {
    "mnist": "MNIST",
    "gpt2": "GPT-2",
}

LOSS_LABELS: dict[str, str] = {
    "mnist": "Cross-Entropy Loss",
    "gpt2": "Loss",
}


def apply_style(style="paper"):
    """Apply the paper's single matplotlib style.

    Every figure in the paper now uses the same rcParams, defined once in this module (the values the AGS blobs figures were drawn with). The old
    'paper' / 'paper_figure' / 'draft' variants differed in font size, line width and
    tick weight, which is exactly the inconsistency this removes; the argument is kept
    so existing call sites still work, and 'draft' still means "smaller type".

    Colour is not touched here -- each experiment keeps its own palette.
    """
    if style == "draft":
        apply(**{k: v for k, v in DRAFT_STYLE.items()})
    else:
        apply()


# =============================================================================
# Color schemes
# =============================================================================

# Standard method colors used across linear ablation
# --- the AGS figure palette --------------------------------------------------------------------
# One definition, previously copied into six figure scripts (twice under different names). Blue is
# always AGS, orange the DP-SGD baseline, green PATE, grey the non-private reference. BLUES shades a
# family of AGS curves, darkest last, so an F-ladder or a batch-size sweep reads in order.
C_AGS = "#2171b5"
C_DPSGD = "#d55e00"
C_PATE = "#009e73"
C_NOMINAL = "0.45"
C_CEILING = "0.65"
BLUES = ["#c6dbef", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
# The three DP-SGD arms of the contractive figures, told apart by colour as well as line style: the matched
# arm keeps the DP-SGD orange, its exactly composed twin is a dark red, the optimized arm (Poisson
# subsampling, tight accounting) a purple.
C_DPSGD_MATCHED = C_DPSGD
C_DPSGD_COMPOSED = "#8c2d04"
C_DPSGD_OPT = "#7b3294"


def log_eps_axis(ax, ticks=(1, 2, 5, 10, 20, 50, 100), label=r"Privacy Budget  $\varepsilon$"):
    """Log epsilon axis with readable tick labels, the setup every AGS figure repeats."""
    import matplotlib

    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if label:
        ax.set_xlabel(label)


METHOD_COLORS = {
    "ibp": "tab:brown",
    "f1": "tab:blue",
    "f2": "tab:orange",
    "f4": "tab:green",
    "f8": "tab:red",
    "f16": "tab:purple",
}

METHOD_LABELS = {
    "ibp": "IBP",
    "f1": "AGT (F=1)",
    "f2": "AGT (F=2)",
    "f4": "AGT (F=4)",
    "f8": "AGT (F=8)",
    "f16": "AGT (F=16)",
}

# Standard batch-size colors
BATCH_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

# Standard depth colors
DEPTH_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}


# =============================================================================
# Save helpers
# =============================================================================


def save_figure(fig_or_path, *additional_paths, dpi=None):
    """Save figure to one or more paths.

    Args:
        fig_or_path: Either a filename (saves current figure) or (fig, path) pair.
        additional_paths: Extra paths to save copies to.
        dpi: Resolution; None means the paper style's savefig.dpi (200).
    """
    if isinstance(fig_or_path, str):
        path = fig_or_path
    else:
        path = fig_or_path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=dpi or plt.rcParams["savefig.dpi"])
    # PNG only (25 Sept 2026): no vector companion, the repository keeps one raster file per figure

    for p in additional_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=dpi or plt.rcParams["savefig.dpi"])

    print(f"Saved: {path}")


def ensure_fig_dir(*dirs):
    """Create figure output directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# Common plot components
# =============================================================================


def plot_staircase_and_sensitivity(
    ax_staircase,
    ax_sensitivity,
    methods_data,
    k_values_by_method,
    colors,
    labels,
    beta=1.0 / 6.0,
    staircase_ylabel="Instability $V_k$",
    sensitivity_ylabel="Smooth Sensitivity",
):
    """Plot instability staircase (left) and smooth sensitivity (right) on twin axes.

    Common across lin/ca/syn ablation plots.

    Args:
        ax_staircase: Matplotlib axes for staircase (dashed lines).
        ax_sensitivity: Twin axes for sensitivity curves (solid lines).
        methods_data: Dict mapping method_name -> {'avg_V_k': [...], 'k_values': [...]}.
        k_values_by_method: Dict mapping method_name -> k_values list.
        colors: Dict mapping method_name -> color.
        labels: Dict mapping method_name -> display label.
        beta: Decay parameter.
    """
    max_k = 0
    for m, data in methods_data.items():
        ks = data.get("k_values", k_values_by_method.get(m, []))
        if ks:
            max_k = max(max_k, max(ks))

    t_grid = np.linspace(0, max_k + 1, 500)

    ax_staircase.set_xlabel("Privacy Parameter k")
    ax_staircase.set_ylabel(staircase_ylabel)

    for m, data in methods_data.items():
        ks = np.array(data.get("k_values", k_values_by_method.get(m, [])))
        vs = np.array(data["avg_V_k"])

        sort_idx = np.argsort(ks)
        ks = ks[sort_idx]
        vs = vs[sort_idx]

        c = colors.get(m, "gray")
        lbl = labels.get(m, m)

        # Staircase
        ax_staircase.step(ks, vs, where="post", label=lbl, color=c, linestyle="--", linewidth=3, alpha=0.7)

        # Smooth sensitivity curve
        indices = np.searchsorted(ks, t_grid, side="right") - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        s_curve = v_interp * np.exp(-beta * t_grid)

        ax_sensitivity.plot(t_grid, s_curve, "-", color=c, linewidth=4, alpha=0.9)

        # Star at maximum
        idx_max = np.argmax(s_curve)
        ax_sensitivity.plot(
            t_grid[idx_max], s_curve[idx_max], "*", color=c, markersize=24, markeredgecolor="black", zorder=10
        )

    ax_staircase.legend(loc="upper left")
    ax_staircase.grid(True, alpha=0.3)


def plot_tradeoff(
    ax,
    epsilons,
    mse_by_method,
    colors,
    labels,
    log_x=True,
    log_y=True,
    xlabel="Privacy Budget $\\epsilon$",
    ylabel="Test MSE",
    fill_between=None,
):
    """Plot privacy-utility trade-off curves.

    Args:
        ax: Matplotlib axes.
        epsilons: List of epsilon values.
        mse_by_method: Dict mapping method_name -> mse_values.
        colors: Dict mapping method_name -> color.
        labels: Dict mapping method_name -> display label.
        fill_between: Optional dict method_name -> (lower, upper) for bands.
    """
    for m, mses in mse_by_method.items():
        c = colors.get(m, "gray")
        lbl = labels.get(m, m)
        ax.plot(epsilons, mses, "o-", label=lbl, color=c, linewidth=4)

        if fill_between and m in fill_between:
            lower, upper = fill_between[m]
            ax.fill_between(epsilons, lower, upper, color=c, alpha=0.1)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


# =============================================================================
# Shared helpers for experiment plot modules
# =============================================================================


def _to_float(value, fallback=np.nan):
    """Convert a value to float, returning *fallback* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def aggregate_by_key(
    method_results: dict,
    key_name: str = "x",
) -> dict[str, np.ndarray]:
    """Aggregate experiment results keyed by a sweep parameter.

    Works for epsilon sweeps, frequency sweeps, AGS-steps sweeps, etc.
    Returns arrays sorted by the sweep parameter.
    """
    x_vals: list[float] = []
    acc_vals: list[float] = []
    loss_vals: list[float] = []
    time_vals: list[float] = []
    ss_vals: list[float] = []
    nw_vals: list[float] = []

    for x_key, entries in method_results.items():
        if not isinstance(entries, list) or not entries:
            continue
        x = _to_float(x_key)
        if np.isnan(x):
            continue

        x_vals.append(x)
        acc_vals.append(np.mean([_to_float(e.get("test_accuracy")) for e in entries]))
        loss_vals.append(np.mean([_to_float(e.get("test_loss")) for e in entries]))
        time_vals.append(np.mean([_to_float(e.get("time_s")) for e in entries]))

        ss_col = [_to_float(e.get("avg_ss_per_window")) for e in entries]
        nw_col = [_to_float(e.get("n_windows")) for e in entries]
        ss_vals.append(np.nanmean(ss_col) if not np.all(np.isnan(ss_col)) else np.nan)
        nw_vals.append(np.nanmean(nw_col) if not np.all(np.isnan(nw_col)) else np.nan)

    if not x_vals:
        empty = np.array([])
        return {key_name: empty, "acc": empty, "loss": empty, "time": empty, "ss": empty, "n_win": empty}

    order = np.argsort(x_vals)
    return {
        key_name: np.array(x_vals)[order],
        "acc": np.array(acc_vals)[order],
        "loss": np.array(loss_vals)[order],
        "time": np.array(time_vals)[order],
        "ss": np.array(ss_vals)[order],
        "n_win": np.array(nw_vals)[order],
    }


def nominal_metrics(nominal_entries: list | None) -> tuple[float, float]:
    """Return (mean_accuracy, mean_loss) from non-private baseline entries."""
    if not nominal_entries:
        return np.nan, np.nan
    acc = float(np.mean([_to_float(e.get("test_accuracy")) for e in nominal_entries]))
    loss = float(np.mean([_to_float(e.get("test_loss")) for e in nominal_entries]))
    return acc, loss


def paper_legend(ax, **kwargs):
    """Add a paper-quality legend with sensible defaults."""
    defaults = dict(frameon=True, framealpha=0.9, edgecolor="gray", fancybox=False)
    defaults.update(kwargs)
    return ax.legend(**defaults)


def load_runs(results_path: Path) -> list[dict]:
    """Load runs from a results.json, with backward-compat for single-run files."""
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        runs = payload.get("runs")
        if isinstance(runs, list):
            return runs
        return [payload]  # single-run legacy format
    return []


def pad_y_limits(ax, pad_ratio=0.15):
    """Increase y-limits by pad_ratio (e.g. 15%) on both top and bottom to center plot lines."""
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - pad_ratio * y_range, ymax + pad_ratio * y_range)


# --- the paper's contractive comparison: two panels in a fixed rectangle, a two-column legend in a
# --- reserved band, so figures with different arm counts have identical graph geometry -----------------

PAPER_HALF_WIDTH_IN = 3.5  # one subfigure of a figure* spanning the two IEEE columns (7.16 in text width)
PAPER_ARM_ORDER = (
    "AGS",
    "DP-SGD (matched, basic)",
    "DP-SGD (matched, composed)",
    "DP-SGD (optimized)",
    "PATE",
    "non-private",
    "AGS (hidden)",
    "DP-SGD (matched, hidden)",
)


def contractive_paper_figure(
    arms,
    eps,
    stem,
    *,
    variant="tight/cross_radius",
    extra_arms=None,
    non_private=None,
    ylim=(0.4, 1.0),
    eps_max=100,
    figsize=(PAPER_HALF_WIDTH_IN, 2.7),
    legend_rows=4,
    xticks=(1, 3, 10, 30, 100),
):
    """Pure and approximate DP side by side from a headline's arms[mode][eps] rows (the contractive
    drivers' schema), budgets up to eps_max. The axes occupy a fixed rectangle and the legend a fixed
    band of `legend_rows` rows in two columns, whatever the number of arms, so two such figures placed as
    subfigures have equal graphs. `extra_arms` adds the hidden-layer group's AGS and matched arms.
    Writes stem.png; returns the figure's handles."""
    apply_combined(
        2, figsize=figsize, page_width_in=figsize[0], print_target_pt=8, label_pt=8, tick_pt=7, legend_pt=7
    )
    xs = [e for e in eps if e <= eps_max]

    def series(A, mode, getter):
        out = []
        for e in xs:
            try:
                v = getter(A[mode][str(e)])
            except (KeyError, TypeError):
                v = None
            out.append(v["test"] if v else float("nan"))
        return out

    fig = plt.figure(figsize=figsize)
    # fixed geometry in inches: a legend band of `legend_rows` rows at the legend font, the tick and
    # x-label strip, then two panels filling the rest; identical whatever the number of arms drawn
    row_in = plt.rcParams["legend.fontsize"] / 72.0 * 1.45
    band_in = legend_rows * row_in + 0.02
    strip_in = 0.30  # tick labels and the x label
    bottom = (band_in + strip_in) / figsize[1]
    top, left, right, gap = 0.905, 0.14, 0.995, 0.06
    w = (right - left - gap) / 2
    axes = [
        fig.add_axes([left, bottom, w, top - bottom]),
        fig.add_axes([left + w + gap, bottom, w, top - bottom]),
    ]
    handles = {}
    for ax, mode, title in zip(
        axes, ("pure", "approx"), (r"Pure $(\varepsilon, 0)$-DP", r"Approx. $(\varepsilon, 10^{-5})$-DP")
    ):
        spec = [
            (
                "AGS",
                series(arms, mode, lambda r: r["ags"][variant]),
                dict(color=BLUES[-1], marker="o", ls="-", zorder=6),
            ),
            (
                "DP-SGD (matched, basic)",
                series(arms, mode, lambda r: r["dpsgd_matched"]["basic"]),
                dict(color=C_DPSGD_MATCHED, marker="s", ls="--"),
            ),
        ]
        if mode == "approx":
            spec.append(
                (
                    "DP-SGD (matched, composed)",
                    series(arms, mode, lambda r: r["dpsgd_matched"]["composed"]),
                    dict(color=C_DPSGD_COMPOSED, marker="D", ls=(0, (1, 1.2)), lw=1.6),
                )
            )
        spec += [
            (
                "DP-SGD (optimized)",
                series(arms, mode, lambda r: r["dpsgd_optimized"]),
                dict(color=C_DPSGD_OPT, marker="P", ls="-"),
            ),
            ("PATE", series(arms, mode, lambda r: r["pate"]), dict(color=C_PATE, marker="^", ls="-")),
        ]
        if extra_arms is not None:
            spec += [
                (
                    "AGS (hidden)",
                    series(extra_arms, mode, lambda r: r["ags"][variant]),
                    dict(color=BLUES[-3], marker="v", ls="-."),
                ),
                (
                    "DP-SGD (matched, hidden)",
                    series(extra_arms, mode, lambda r: r["dpsgd_matched"]["basic"]),
                    dict(color=C_DPSGD_MATCHED, marker="v", ls="-.", alpha=0.75),
                ),
            ]
        for lab, y, sty in spec:
            sty = dict(dict(ms=3.2, lw=1.2, markeredgecolor="white", markeredgewidth=0.6, zorder=5), **sty)
            (h,) = ax.plot(xs, y, **sty)
            handles.setdefault(lab, h)
        if non_private is not None:
            band_h = ax.axhline(non_private, color=C_NOMINAL, ls="-", lw=5, alpha=0.28, zorder=1)
            handles.setdefault("non-private", band_h)
        ax.set_xscale("log")
        ax.set_xlim(0.9, eps_max * 1.1)
        ax.set_ylim(*ylim)
        ax.set_xticks(list(xticks))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"Privacy Budget $\varepsilon$")
        ax.set_title(title, pad=4)
    axes[0].set_ylabel("Test Accuracy")
    axes[1].tick_params(labelleft=False)
    order = [k for k in PAPER_ARM_ORDER if k in handles]
    # two one-column legends, the block centred in the band both ways: the columns keep a fixed gap and are
    # measured after a draw, so the block is centred whatever the labels' widths and row count
    rows = (len(order) + 1) // 2
    band_mid = 0.5 * band_in / figsize[1]
    legs = [
        fig.legend(
            [handles[k] for k in keys],
            keys,
            loc="center left",
            bbox_to_anchor=(0.0, band_mid),
            ncol=1,
            handlelength=1.6,
            handletextpad=0.4,
            labelspacing=0.3,
            frameon=False,
            borderaxespad=0.0,
        )
        for keys in (order[:rows], order[rows:])
        if keys
    ]
    fig.canvas.draw()
    widths = [lg.get_window_extent().transformed(fig.transFigure.inverted()).width for lg in legs]
    col_gap = 0.06
    x = 0.5 - (sum(widths) + col_gap * (len(legs) - 1)) / 2
    for lg, wd in zip(legs, widths):
        lg.set_bbox_to_anchor((x, band_mid), transform=fig.transFigure)
        x += wd + col_gap
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    return handles
