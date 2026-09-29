"""The feasibility figure for Theorems 4.3 and 4.4: when does smooth sensitivity win?

Both theorems bound the sensitivity ratio rho = SS^beta / GS below which our mechanism beats the
global-sensitivity baseline at confidence 1 - alpha and utility multiplier c. This draws those
ceilings, and the validity floor that the original version of the figure left out.

The floor matters. SS^beta decreases in beta while the mechanism is only valid for beta below a
limit set by the budget, so the largest admissible beta gives the smallest achievable smooth
sensitivity: a floor no configuration can go under. The honest feasible region is a band between
that floor and the theorem's ceiling, and where the floor sits above the ceiling there is no
feasible configuration at all. Sweeping SS^beta as a free variable hid exactly that.

    PYTHONPATH=. python -m agtr.condition_table --combined     # the figure the paper uses

Writes into agtr/figures/. Analytic throughout: it reads no stored results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D
from scipy.stats import norm

from common.plotting import apply_style, paper_legend

curr_folder = __file__.rsplit("/", 1)[0]
root_output_dir = f"{curr_folder}/figures"
os.makedirs(root_output_dir, exist_ok=True)

# =============================================================================
# WHY THIS SCRIPT WAS REWRITTEN
# -----------------------------------------------------------------------------
# The original approx-DP condition table plotted the RHS of Thm 4.5 directly:
#
#       SS^beta / Df  <  sqrt(2 ln(1.25/delta)) * Phi^{-1}(1 - alpha/2)
#                        ----------------------------------------------
#                                   2 c ln(1/alpha)
#
# and shaded everything below that line as "feasible" (AGS beats Gaussian
# DP-SGD). The bug: this treats SS^beta as a FREE variable you can slide down
# the y-axis. But SS^beta is NOT free. Two facts couple through beta:
#
#   (1) DEFINITION:  SS^beta = max_k  e^{-beta k} A_k      [decreasing in beta]
#   (2) VALIDITY:    the Laplace-on-SS^beta mechanism is (eps,delta)-DP only if
#                        beta <= beta*  with  beta* = eps / (2 ln(2/delta))
#                    (Def 3.8 in the draft).
#
# Because SS^beta DECREASES in beta, the LARGEST admissible beta = beta* gives
# the SMALLEST achievable smooth sensitivity. That smallest achievable value is
# a FLOOR on SS^beta that the original plot ignored entirely:
#
#       rho_floor = SS^{beta*} / Df = max_k e^{-beta* k} (A_k / Df).
#
# The honest feasibility region is therefore a BAND:
#
#       rho_floor  <  SS^beta/Df  <  rho_ceiling(alpha)
#        (validity)                    (Thm 4.5)
#
# If the floor sits above the ceiling, AGS CANNOT win for that (alpha, delta):
# the region is empty. That emptiness is exactly the information the original
# figure hid. Note the asymmetry that drives the whole story:
#
#   * rho_ceiling RISES with confidence (1 - alpha)  -> depends on alpha
#   * rho_floor   is FLAT in alpha (beta* depends only on eps, delta) -> no alpha
#
# so feasibility is won or lost at LOW confidence, where the ceiling is lowest.
#
# Two figures are produced below:
#   OPTION 3  (plot_cond_table_approx_dp_swept): SS^beta swept as a free variable.
#             Draws the Thm-4.5 ceiling and, where a profile is supplied, the
#             validity floor. Honest about what it cannot verify.
#   OPTION 1  (plot_cond_table_approx_dp_valid): commit to a linear instability
#             profile A_k, compute the real floor at beta*, and shade the true
#             feasibility band. This is the version that lets the beta-validity
#             constraint actually BIND.
# =============================================================================


# -------------------------------------------------------------------------
# Representative privacy level. The validity constraint beta* = eps/(2 ln(2/delta))
# reintroduces an eps dependence that had cancelled in the pure-DP plot, so we
# must FIX eps. We sweep the existing deltas, matching the original figure.
# -------------------------------------------------------------------------
EPS = 1.0
DELTAS = [1e-5, 1e-4, 1e-3]
DELTA_STYLES = ["-", "--", ":"]
DELTA_LABELS = [r"$\delta = 10^{-5}$", r"$\delta = 10^{-4}$", r"$\delta = 10^{-3}$"]
DELTA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # match tab:blue/orange/green


def beta_star_approx(eps, delta):
    """Largest admissible smoothing parameter for the approx-DP Laplace
    mechanism on smooth sensitivity (Def 3.8): beta <= eps / (2 ln(2/delta)).
    Largest valid beta  ==>  smallest achievable SS^beta  ==>  tightest floor."""
    return eps / (2.0 * np.log(2.0 / delta))


def beta_star_pure(eps):
    """Largest admissible smoothing parameter for the pure-DP Cauchy
    mechanism (Def 3.7): beta <= eps / 6.
    Largest valid beta  ==>  smallest achievable SS^beta  ==>  tightest floor."""
    return eps / 6.0


def rho_ceiling_approx(alpha, delta, c):
    """Thm 4.5 ceiling (approx DP): the SS^beta/Df value below which AGS
    (Laplace on smooth sensitivity) beats the Gaussian mechanism, at
    confidence 1 - alpha and utility multiplier c."""
    num = np.sqrt(2.0 * np.log(1.25 / delta)) * norm.ppf(1.0 - alpha / 2.0)
    den = np.log(1.0 / alpha)
    return num / den / (2.0 * c)


def rho_ceiling_pure(alpha, c):
    """Thm 4.4 ceiling (pure DP): the SS^beta/Df value below which AGS
    (Cauchy on smooth sensitivity) beats the Laplace mechanism, at
    confidence 1 - alpha and utility multiplier c.

    Pure-DP ratio: log(1/alpha) / tan(pi*(1-alpha)/2) / (6c)."""
    conf = 1.0 - alpha
    g = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)
    return g / (6.0 * c)


def rho_floor_linear(beta_val, slope, k_max=200):
    """Validity floor under a LINEAR instability profile.

    We model the AGT instability certificate as growing linearly in k until it
    saturates at the global sensitivity (a clipped-linear profile, consistent
    with the draft's observation that A_k rises roughly linearly in practice):

        A_k / Df = min(1, slope * k),    slope > 0  ("instability sharpness")

    Then the smallest SS^beta any admissible beta can deliver is realized at
    beta* (largest valid beta):

        rho_floor = SS^{beta*} / Df = max_k  e^{-beta* k} * min(1, slope * k).

    The max over k is what makes the validity constraint BIND: e^{-beta* k}
    pushes the optimum toward small k, while min(1, slope*k) pushes it up.
    Returned value is a scalar (flat in alpha)."""
    k = np.arange(1, k_max + 1)
    profile = np.minimum(1.0, slope * k)  # A_k / Df, clipped at global sens.
    return np.max(np.exp(-beta_val * k) * profile)


# =============================================================================
# Shared style setup
# =============================================================================


def _setup_style():
    """Apply the paper-figure style from the shared plotting module, then
    override a few sizes appropriate for these single/multi-panel condition
    plots."""
    apply_style("paper_figure")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
        }
    )


# =============================================================================
# OPTION 3 -- sweep SS^beta as a free variable (ceiling, with floor gestured at)
# =============================================================================
def plot_cond_table_approx_dp_swept(slope=None):
    """OPTION 3.

    The y-axis here is the smooth-sensitivity ratio rho = SS^beta/Df treated as
    a FREE swept variable -- this is the original framing, made honest.

    What we draw:
      * rho_ceiling(alpha): the Thm-4.5 threshold (solid, per delta). Below it,
        AGS wins *assuming the SS^beta is achievable by some valid beta*.
      * IF a `slope` is given, we also draw rho_floor (dashed horizontal, per
        delta): the smallest SS^beta any admissible beta = beta* can produce.
        The genuinely feasible band is ceiling-above-floor.

    The honest caveat (printed and annotated): WITHOUT a profile, the ceiling
    alone cannot tell you whether any valid beta achieves an SS^beta beneath it.
    That is the structural limitation of the swept view -- it shows the target
    but not whether the target is reachable.
    """
    _setup_style()
    file_save_name = "ags_tradeoff_approx_dp_swept.png"
    fpath = f"{root_output_dir}/{file_save_name}"

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha
    c = 1  # utility multiplier fixed to the most permissive case for clarity

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for delta, ls, dlabel, color in zip(DELTAS, DELTA_STYLES, DELTA_LABELS, DELTA_COLORS):
        ceil = rho_ceiling_approx(alpha, delta, c)
        ax.plot(conf, ceil, color=color, linestyle="-", linewidth=3, label=f"Ceiling (Thm 4.5), {dlabel}")

        if slope is not None:
            bstar = beta_star_approx(EPS, delta)
            floor = rho_floor_linear(bstar, slope)
            ax.axhline(floor, color=color, linestyle="--", linewidth=2, alpha=0.8)
            # Shade only where ceiling > floor (the genuinely feasible band).
            feasible = ceil > floor
            ax.fill_between(conf, floor, ceil, where=feasible, color=color, alpha=0.10)

    ax.set_xlabel(r"Confidence Level $1 - \alpha$")
    ax.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")

    title = (
        r"Approx-DP Condition (swept $\mathrm{SS}^\beta$)" + "\n"
        f"Ceiling = Thm 4.5;  "
        r"$\varepsilon = $" + f"{EPS},  $c = {c}$"
    )
    if slope is not None:
        title += r";  floor at $\beta^*$, slope " + f"$s={slope}$"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if slope is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                lw=2,
                label=r"Validity floor $\mathrm{SS}^{\beta^*}/\Delta f$",
            )
        )
        labels.append(r"Validity floor $\mathrm{SS}^{\beta^*}/\Delta f$")
    paper_legend(ax, handles=handles, labels=labels, loc="upper left", fontsize=14)

    if slope is None:
        ax.text(
            0.5,
            0.02,
            r"Ceiling only: cannot verify a valid $\beta$ achieves any "
            r"$\mathrm{SS}^\beta$ below it." + "\n"
            "Supply an instability slope to draw the validity floor.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.5),
        )

    fig.tight_layout()
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved to {fpath}")
    plt.show()


# =============================================================================
# OPTION 1 -- linear instability profile, real feasibility band
# =============================================================================
def plot_cond_table_approx_dp_valid(slopes=(0.02, 0.05, 0.1)):
    """OPTION 1.

    Commit to a linear (clipped) instability profile A_k/Df = min(1, slope*k)
    and fold the beta-validity constraint in directly. For each delta we draw:

      * rho_ceiling(alpha): Thm-4.5 threshold (solid).
      * rho_floor: SS^{beta*}/Df under the profile (flat dashed line, since
        beta* and hence the floor do not depend on alpha).
      * shaded band where ceiling > floor: the TRUE feasibility region in which
        AGS provably beats Gaussian DP-SGD with an ADMISSIBLE beta.

    Sweeping the instability slope s shows the mechanism's whole story: small s
    (model stays locally stable, A_k small for small k) keeps the floor low and
    the band wide; large s lifts the floor until it crosses the ceiling and the
    region collapses to empty. One subplot per slope; deltas overlaid within.
    """
    _setup_style()
    file_save_name = "ags_tradeoff_approx_dp_valid.png"
    fpath = f"{root_output_dir}/{file_save_name}"

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha
    c = 1

    n = len(slopes)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, slope in zip(axes, slopes):
        for delta, dlabel, color in zip(DELTAS, DELTA_LABELS, DELTA_COLORS):
            ceil = rho_ceiling_approx(alpha, delta, c)
            bstar = beta_star_approx(EPS, delta)
            floor = rho_floor_linear(bstar, slope)

            ax.plot(conf, ceil, color=color, linestyle="-", linewidth=3, label=f"{dlabel}")
            ax.axhline(floor, color=color, linestyle="--", linewidth=2, alpha=0.8)

            feasible = ceil > floor
            ax.fill_between(conf, floor, ceil, where=feasible, color=color, alpha=0.12)

            # Mark the confidence at which the band opens (ceiling crosses floor).
            if np.any(feasible) and not feasible[0]:
                idx = np.argmax(feasible)
                ax.scatter(conf[idx], floor, color=color, zorder=5, s=40, edgecolors="black", linewidths=0.8)

        ax.set_xlabel(r"Confidence Level $1 - \alpha$")
        ax.set_title(f"Instability slope $s = {slope}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")

    # Combined legend: deltas (color, solid = ceiling) + floor convention.
    handles = [
        Line2D([0], [0], color=col, linestyle="-", lw=3, label=lab)
        for col, lab in zip(DELTA_COLORS, DELTA_LABELS)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            lw=2,
            label=r"Validity floor $\mathrm{SS}^{\beta^*}/\Delta f$",
        )
    )
    paper_legend(axes[-1], handles=handles, loc="upper left", fontsize=14)

    fig.suptitle(
        r"Approx-DP Condition ($\beta$-valid, linear profile)" + "\n"
        r"Shaded = AGS provably beats Gaussian DP-SGD with admissible "
        r"$\beta = \beta^*$;  " + f"$\\varepsilon = {EPS}$,  $c = {c}$",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved to {fpath}")
    plt.show()


# =============================================================================
# PURE DP -- NOW WITH VALID BETA CONSTRAINT
# -----------------------------------------------------------------------------
# The Cauchy mechanism (Def 3.7) requires beta <= eps/6 for (eps,0)-DP.
# Previously the pure-DP plot did NOT account for valid beta. The ceiling
# (the swept ratio) is the same as before:
#
#   rho_ceiling = log(1/alpha) / tan(pi*(1-alpha)/2) / (6c)
#
# but SS^beta can only be as small as SS^{beta*} with beta* = eps/6. With a
# linear instability profile, this gives a floor exactly as in the approx case.
# =============================================================================
def plot_cond_table_pure_dp(slopes=None):
    """Pure-DP condition table with optional beta-validity floor.

    If slopes is None: just the ceiling curves (the original plot, restyled).
    If slopes is given: one subplot per slope, with the validity floor and
    feasibility band, exactly paralleling the approx-DP version.
    """
    _setup_style()
    file_save_name = "ags_tradeoff_pure_dp.png"
    fpath = f"{root_output_dir}/{file_save_name}"

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha

    if slopes is None:
        # ── Original ceiling-only plot (restyled) ──
        fig, ax = plt.subplots(figsize=(8, 5.5))

        g = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)
        i = np.argmax(g)

        c_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for c in range(1, 6):
            ax.plot(conf, g / (6 * c), label=f"$c = {c}$", color=c_colors[c - 1], linewidth=3)
            ax.scatter(conf[i], g[i] / (6 * c), color="k", zorder=5, s=40)
            ax.annotate(
                f"{g[i] / (6 * c):.3f}",
                (conf[i], g[i] / (6 * c)),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=14,
            )
        ax.scatter([], [], color="k", label="Peak ratio", s=40)
        ax.fill_between(conf, 0, g / 6, color="#1f77b4", alpha=0.08)

        ax.set_xlabel(r"Confidence Level $1 - \alpha$")
        ax.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")
        ax.set_title(
            "Pure-DP: Feasibility Region for Smooth-Sensitivity\nUtility Improvement over Laplace DP"
        )
        ax.set_ylim(0, 0.13)
        ax.grid(True, alpha=0.25)
        paper_legend(ax, title="Utility Multiplier", loc="upper right", fontsize=14)

        fig.tight_layout()
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved to {fpath}")
        plt.show()

    else:
        # ── Beta-valid version with floor ──
        file_save_name = "ags_tradeoff_pure_dp_valid.png"
        fpath = f"{root_output_dir}/{file_save_name}"

        c = 1  # most permissive
        bstar = beta_star_pure(EPS)

        n = len(slopes)
        fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), sharey=True)
        if n == 1:
            axes = [axes]

        ceil = rho_ceiling_pure(alpha, c)

        for ax, slope in zip(axes, slopes):
            floor = rho_floor_linear(bstar, slope)

            ax.plot(conf, ceil, color="#1f77b4", linestyle="-", linewidth=3, label="Ceiling (Thm 4.4)")
            ax.axhline(
                floor,
                color="#1f77b4",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=r"Floor at $\beta^* = \varepsilon/6$",
            )

            feasible = ceil > floor
            ax.fill_between(conf, floor, ceil, where=feasible, color="#1f77b4", alpha=0.12)

            # Mark crossing point
            if np.any(feasible) and not feasible[0]:
                idx = np.argmax(feasible)
                ax.scatter(
                    conf[idx], floor, color="#1f77b4", zorder=5, s=40, edgecolors="black", linewidths=0.8
                )

            ax.set_xlabel(r"Confidence Level $1 - \alpha$")
            ax.set_title(f"Instability slope $s = {slope}$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.25)
            paper_legend(ax, loc="upper left", fontsize=14)

        axes[0].set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")

        fig.suptitle(
            r"Pure-DP Condition ($\beta$-valid, linear profile)" + "\n"
            r"$\beta^* = \varepsilon/6$;  " + f"$\\varepsilon = {EPS}$,  $c = {c}$",
            fontsize=18,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved to {fpath}")
        plt.show()


# =============================================================================
# APPROX-DP FULL SWEEP -- reproduces the old condition_table.py layout
# (c × delta grid) but with beta-validity floor from Thm 4.5
# =============================================================================
def plot_cond_table_approx_dp_sweep(slope=None):
    """Reproduce the old condition_table.py approx-DP figure:
    c in {1,2,3} (colours) × delta in {1e-5,1e-4,1e-3} (linestyles),
    with a pure-DP reference line per c.

    When `slope` is given, the beta-validity floor is drawn per delta
    (dashed horizontal) and the feasibility band is shaded.
    """
    _setup_style()

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha

    deltas = [1e-5, 1e-4, 1e-3]
    cs = [1, 2, 3]
    c_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    delta_styles = ["-", "--", ":"]
    delta_labels = [r"$\delta = 10^{-5}$", r"$\delta = 10^{-4}$", r"$\delta = 10^{-3}$"]

    g_pure = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)

    fig, ax = plt.subplots(figsize=(9, 6))

    for ci, c in enumerate(cs):
        color = c_colors[ci]

        # Approx-DP ceiling curves — one per delta
        for delta, ls, dlabel in zip(deltas, delta_styles, delta_labels):
            ceil = rho_ceiling_approx(alpha, delta, c)
            ax.plot(conf, ceil, color=color, linestyle=ls, linewidth=2.5)

        # Pure-DP reference (short-dotted, same colour, thinner)
        ax.plot(conf, g_pure / (6 * c), color=color, linestyle=(0, (2, 4)), linewidth=1.5, alpha=0.6)

    # Validity floors (one per delta, independent of c)
    if slope is not None:
        for delta, ls, dlabel in zip(deltas, delta_styles, delta_labels):
            bstar = beta_star_approx(EPS, delta)
            floor = rho_floor_linear(bstar, slope)
            ax.axhline(floor, color="dimgray", linestyle=ls, linewidth=1.8, alpha=0.7)

        # Shade the feasibility band for c=1 (most permissive)
        for delta in deltas:
            ceil_c1 = rho_ceiling_approx(alpha, delta, cs[0])
            bstar = beta_star_approx(EPS, delta)
            floor = rho_floor_linear(bstar, slope)
            feasible = ceil_c1 > floor
            ax.fill_between(conf, floor, ceil_c1, where=feasible, color="#1f77b4", alpha=0.06)
    else:
        # No floor — shade everything below c=1, delta=1e-5
        rho_fill = rho_ceiling_approx(alpha, deltas[0], cs[0])
        ax.fill_between(conf, 0, rho_fill, color=c_colors[0], alpha=0.07)

    ax.set_xlabel(r"Confidence Level $1 - \alpha$")
    ax.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")
    title = "Feasibility Region for Smooth-Sensitivity\nUtility Improvement over Gaussian Approx-DP"
    if slope is not None:
        title += (
            f"\n(validity floor at slope $s={slope}$, "
            r"$\varepsilon=$" + f"{EPS})"
        )
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25)

    # Two-part legend: delta/mechanism (linestyle) + c (colour)
    delta_handles = [
        Line2D([0], [0], color="gray", linestyle=ls, linewidth=2.5, label=dl)
        for ls, dl in zip(delta_styles, delta_labels)
    ]
    delta_handles.append(
        Line2D([0], [0], color="gray", linestyle=(0, (2, 4)), linewidth=1.5, alpha=0.6, label="Pure DP (ref)")
    )
    if slope is not None:
        delta_handles.append(
            Line2D(
                [0],
                [0],
                color="dimgray",
                linestyle="-",
                linewidth=1.8,
                alpha=0.7,
                label=r"Validity floor $\mathrm{SS}^{\beta^*}\!/\Delta f$",
            )
        )

    c_handles = [
        Line2D([0], [0], color=c_colors[ci], linestyle="-", linewidth=3, label=f"$c = {c}$")
        for ci, c in enumerate(cs)
    ]

    leg1 = ax.legend(
        handles=delta_handles,
        title=r"Mechanism / $\delta$",
        loc="upper left",
        fontsize=14,
        title_fontsize=15,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )
    ax.add_artist(leg1)
    paper_legend(
        ax, handles=c_handles, title="Utility Multiplier", loc="upper right", fontsize=14, title_fontsize=15
    )

    fig.tight_layout()
    fname = "ags_tradeoff_approx_dp_sweep.png"
    fpath = f"{root_output_dir}/{fname}"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved to {fpath}")
    plt.show()


# =============================================================================
# PURE-DP FULL SWEEP -- c values with validity floor
# =============================================================================
def plot_cond_table_pure_dp_sweep(slope=None):
    """Pure-DP ceiling curves for c in {1,...,5}, reproducing the old layout.

    When `slope` is given, the validity floor at beta* = eps/6 is drawn and
    the feasibility band is shaded per-c.
    """
    _setup_style()

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha
    g = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)
    i = np.argmax(g)

    c_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cs = list(range(1, 6))

    fig, ax = plt.subplots(figsize=(9, 6))

    bstar = beta_star_pure(EPS)
    floor = rho_floor_linear(bstar, slope) if slope is not None else None

    for c in cs:
        ceil = g / (6 * c)
        ax.plot(conf, ceil, label=f"$c = {c}$", color=c_colors[c - 1], linewidth=3)
        ax.scatter(conf[i], g[i] / (6 * c), color="k", zorder=5, s=40)
        ax.annotate(
            f"{g[i] / (6 * c):.3f}",
            (conf[i], g[i] / (6 * c)),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=14,
        )

        if floor is not None:
            feasible = ceil > floor
            ax.fill_between(conf, floor, ceil, where=feasible, color=c_colors[c - 1], alpha=0.06)

    ax.scatter([], [], color="k", label="Peak ratio", s=40)

    if floor is not None:
        ax.axhline(
            floor,
            color="dimgray",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=r"Validity floor ($\beta^*\!=\!\varepsilon/6$)",
        )
    else:
        ax.fill_between(conf, 0, g / 6, color="#1f77b4", alpha=0.08)

    ax.set_xlabel(r"Confidence Level $1 - \alpha$")
    ax.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")
    title = "Pure-DP: Feasibility Region for Smooth-Sensitivity\nUtility Improvement over Laplace DP"
    if slope is not None:
        title += (
            f"\n(validity floor at slope $s={slope}$, "
            r"$\varepsilon=$" + f"{EPS})"
        )
    ax.set_title(title)
    ax.set_ylim(0, 0.13)
    ax.grid(True, alpha=0.25)
    paper_legend(ax, title="Utility Multiplier", loc="upper right", fontsize=14, title_fontsize=15)

    fig.tight_layout()
    fname = "ags_tradeoff_pure_dp_sweep.png"
    fpath = f"{root_output_dir}/{fname}"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    print(f"Saved to {fpath}")
    plt.show()


# =============================================================================
# COMBINED CONDITION FIGURE -- two panels side-by-side
# Left:  Pure-DP    c sweep       + validity floor    (Thm 4.4)
# Right: Approx-DP  c×delta sweep + validity floors   (Thm 4.5)
#
# Linestyle convention (unified across panels):
#   SOLID       = pure DP  (Thm 4.4 ceiling, and the pure-DP reference in (b))
#   --  / : / -.  = approx DP  (Thm 4.5 ceiling, keyed to delta)
#
# Shared legend below the figure.
# =============================================================================
def plot_combined_condition(eps=1.0, slope=0.01):
    """Combined two-panel condition figure for the paper.

    Left panel (a):  Pure-DP Thm 4.4 ceilings for c in {1,...,5} (SOLID)
                     with validity floor at beta* = eps/6.
    Right panel (b): Approx-DP Thm 4.5 ceilings for c in {1,2,3} ×
                     delta in {1e-5, 1e-4, 1e-3} (DASHED / DOTTED / DASHDOT)
                     with pure-DP reference (solid, thin) and validity
                     floors per delta.

    A single unified legend sits below both panels.  Title is clean;
    eps and slope are left for the LaTeX caption.
    """
    _setup_style()
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "lines.linewidth": 3.0,
            "axes.linewidth": 1.5,
        }
    )

    alpha = np.linspace(1e-3, 1 - 1e-3, 5000)
    conf = 1 - alpha

    # Shared colour palette for c values (c=1..5 → 5 colours)
    c_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Delta settings (right panel only) — NON-SOLID linestyles
    deltas = [1e-5, 1e-4, 1e-3]
    delta_styles = ["--", ":", "-."]
    delta_labels = [r"$\delta\!=\!10^{-5}$", r"$\delta\!=\!10^{-4}$", r"$\delta\!=\!10^{-3}$"]

    # Pure-DP helpers
    g_pure = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)
    i_peak = np.argmax(g_pure)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7))

    # ══════════════════════════════════════════════════════════════════════
    # LEFT PANEL (a): Pure-DP  c sweep  (Thm 4.4)
    #                 All curves SOLID.  Floor described in caption only.
    # ══════════════════════════════════════════════════════════════════════
    cs_pure = list(range(1, 6))
    bstar_pure = beta_star_pure(eps)
    floor_pure = rho_floor_linear(bstar_pure, slope)

    for c in cs_pure:
        ceil = g_pure / (6 * c)
        ax_left.plot(conf, ceil, color=c_colors[c - 1], linestyle="-", linewidth=3.0)
        ax_left.scatter(conf[i_peak], g_pure[i_peak] / (6 * c), color="k", zorder=5, s=50)
        ax_left.annotate(
            f"{g_pure[i_peak] / (6 * c):.3f}",
            (conf[i_peak], g_pure[i_peak] / (6 * c)),
            textcoords="offset points",
            xytext=(12, 5),
            fontsize=15,
        )

        feasible = ceil > floor_pure
        ax_left.fill_between(conf, floor_pure, ceil, where=feasible, color=c_colors[c - 1], alpha=0.08)

    # No floor line plotted — value goes in caption
    ax_left.set_xlabel(r"Confidence Level $1 - \alpha$")
    ax_left.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")
    ax_left.set_title("(a)  Pure-DP (Thm 4.4)")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 0.13)
    ax_left.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT PANEL (b): Approx-DP  c × delta  (Thm 4.5)
    #                  Curves use NON-SOLID linestyles keyed to delta.
    #                  Shaded under each c value with that c's colour.
    # ══════════════════════════════════════════════════════════════════════
    cs_approx = [1, 2, 3]

    for ci, c in enumerate(cs_approx):
        color = c_colors[ci]
        for delta, ls in zip(deltas, delta_styles):
            ceil = rho_ceiling_approx(alpha, delta, c)
            ax_right.plot(conf, ceil, color=color, linestyle=ls, linewidth=2.8)

            # Shade under this curve with the c colour
            ax_right.fill_between(conf, 0, ceil, color=color, alpha=0.04)

    ax_right.set_xlabel(r"Confidence Level $1 - \alpha$")
    ax_right.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^\beta / \Delta f$")
    ax_right.set_title("(b)  Approx-DP (Thm 4.5)")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, None)
    ax_right.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════
    # UNIFIED LEGEND (below both panels)
    # ══════════════════════════════════════════════════════════════════════
    # Row 1: c values (colours) — shown as solid (pure-DP convention)
    c_handles = [
        Line2D([0], [0], color=c_colors[ci], linestyle="-", linewidth=3.5, label=f"$c = {c}$")
        for ci, c in enumerate(range(1, 6))
    ]

    # Row 2: linestyle key + symbols
    style_handles = []
    # Solid = pure DP (panel a)
    style_handles.append(Line2D([0], [0], color="gray", linestyle="-", linewidth=2.5, label="Pure DP (a)"))
    # Non-solid = approx DP per delta (panel b)
    for ls, dl in zip(delta_styles, delta_labels):
        style_handles.append(Line2D([0], [0], color="gray", linestyle=ls, linewidth=2.5, label=dl))
    # Peak ratio
    style_handles.append(
        Line2D([0], [0], color="k", marker="o", linestyle="None", markersize=8, label="Peak ratio")
    )

    all_handles = c_handles + style_handles

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)

    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=False,
        fontsize=18,
        handlelength=2.8,
        handleheight=1.5,
        columnspacing=1.8,
    )

    fpath = f"{root_output_dir}/combined_condition.png"
    fig.savefig(fpath, dpi=600, bbox_inches="tight")
    print(f"Saved to {fpath}")
    plt.show()


def _gradient_fill(ax, x, y, color, alpha_top=0.16, alpha_bottom=0.02, y0=0.0, zorder=0):
    """Shade between y0 and the curve y(x) with a vertical alpha gradient (strongest just under the curve)."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon

    y_max = float(np.max(y))
    grad = np.linspace(alpha_bottom, alpha_top, 256).reshape(-1, 1)
    rgba = np.zeros((256, 1, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = grad
    im = ax.imshow(
        rgba,
        aspect="auto",
        origin="lower",
        extent=(x.min(), x.max(), y0, y_max),
        zorder=zorder,
        interpolation="bicubic",
    )
    clip = Polygon(
        np.column_stack([np.r_[x, x[::-1]], np.r_[y, np.full_like(y, y0)]]),
        closed=True,
        transform=ax.transData,
    )
    im.set_clip_path(clip)


def plot_combined_condition_paper(height_scale=1.0, out=None):
    """The paper's version of the combined condition figure (drawn on another machine; rebuilt here from the
    rendered PNG): matplotlib's default style, (a) the pure-DP ceilings of Thm 4.3 for c = 1..5 over the whole
    confidence range with their peaks, (b) the approximate-DP ceilings of Thm 4.4 for c = 1..3 and three deltas
    on the high-confidence range 1 - alpha >= 0.8, against Delta_2 f; one-row legend. height_scale shrinks the
    panels' height (0.7: the supervisor's 30 % lower version) at the same width and fonts."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
        }
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    deltas, styles = [1e-5, 1e-4, 1e-3], ["--", ":", "-."]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.8, 4.1 * height_scale + 0.65), layout="constrained")

    alpha = np.linspace(1e-3, 1 - 1e-3, 4000)
    conf = 1 - alpha
    g = np.log(1.0 / alpha) / np.tan(np.pi * conf / 2.0)
    i_peak = int(np.argmax(g))
    for c in range(5, 0, -1):
        y = g / (6 * c)
        _gradient_fill(ax_a, conf, y, colors[c - 1])
        ax_a.plot(conf, y, color=colors[c - 1], lw=2.2, zorder=3)
        ax_a.scatter(conf[i_peak], y[i_peak], color="k", s=40, zorder=5)
        # c = 3, 4, 5 peak 0.01 and 0.006 apart: 4 labelled on the right, raised into the gap between its
        # line and the c = 3 line; 5 below; the rest above
        off, va = {4: ((9, 5), "center"), 5: ((8, -17), "baseline")}.get(c, ((8, 6), "baseline"))
        ax_a.annotate(
            f"{y[i_peak]:.3f}",
            (conf[i_peak], y[i_peak]),
            textcoords="offset points",
            xytext=off,
            va=va,
            fontsize=12,
            zorder=6,
        )
    ax_a.set(xlim=(conf.min(), conf.max()), ylim=(0, 0.13), title="(a)  Pure-DP (Thm IV.3)")
    ax_a.set_xlabel(r"Confidence Level $1-\alpha$")
    ax_a.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^{\beta}/\Delta f$")
    ax_a.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax_a.set_yticks(np.arange(0, 0.13, 0.02))
    ax_a.grid(True, alpha=0.3)

    alpha_b = np.linspace(1e-4, 0.2, 4000)
    conf_b = 1 - alpha_b
    for c in (3, 2, 1):
        top = rho_ceiling_approx(alpha_b, deltas[0], c)
        _gradient_fill(ax_b, conf_b, top, colors[c - 1], alpha_top=0.12, alpha_bottom=0.1)
        for d, ls in zip(deltas, styles):
            ax_b.plot(conf_b, rho_ceiling_approx(alpha_b, d, c), color=colors[c - 1], ls=ls, lw=2.2, zorder=3)
    ax_b.set(xlim=(0.8, 1.0), ylim=(0, 2.05), title="(b)  Approx-DP (Thm IV.4)")
    ax_b.set_yticks(np.arange(0, 2.01, 0.25))
    ax_b.set_xlabel(r"Confidence Level $1-\alpha$")
    ax_b.set_ylabel(r"Sensitivity Ratio $\rho = \mathrm{SS}^{\beta}/\Delta_2 f$")
    ax_b.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=colors[c - 1], lw=2.2, label=f"$c = {c}$") for c in range(1, 6)]
    handles += [
        Line2D([0], [0], color="gray", ls=ls, lw=2.2, label=rf"$\delta = 10^{{{int(np.log10(d))}}}$")
        for d, ls in zip(deltas, styles)
    ]
    handles.append(Line2D([0], [0], color="k", marker="o", ls="None", ms=7, label="Peak ratio"))
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=True,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    stem = "combined_condition_paper" + ("" if height_scale == 1.0 else f"_h{int(round(height_scale * 100))}")
    fpath = out or f"{root_output_dir}/{stem}.png"
    fig.savefig(fpath, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to {fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AGS condition tables (pure DP and corrected approx DP)."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pure", action="store_true", default=False, help="Pure-DP multi-panel (slope sweep). Add --slopes."
    )
    mode.add_argument(
        "--swept", action="store_true", default=False, help="Approx-DP with SS^beta swept as free variable."
    )
    mode.add_argument(
        "--approx-sweep",
        action="store_true",
        default=False,
        help="Approx-DP full c×delta sweep (old layout + floor).",
    )
    mode.add_argument(
        "--pure-sweep", action="store_true", default=False, help="Pure-DP full c sweep (old layout + floor)."
    )
    mode.add_argument(
        "--combined", action="store_true", default=False, help="Combined two-panel pure-DP condition figure."
    )

    parser.add_argument(
        "--slope", type=float, default=None, help="Instability slope s for the linear profile."
    )
    parser.add_argument(
        "--slopes",
        type=float,
        nargs="+",
        default=None,
        help="Multiple instability slopes for multi-panel plots. Default: 0.02 0.05 0.1",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="the paper's version of the combined figure (default style, (b) on 1 - alpha >= 0.8)",
    )
    parser.add_argument("--height-scale", type=float, default=1.0, help="--paper-style: panel height factor")
    parser.add_argument(
        "--eps", type=float, default=None, help="Privacy budget epsilon (overrides the default)."
    )
    args = parser.parse_args()

    if args.eps is not None:
        EPS = args.eps

    if args.paper_style:
        plot_combined_condition_paper(height_scale=args.height_scale)
    elif args.combined:
        plot_combined_condition(
            eps=args.eps if args.eps is not None else 2.5,
            slope=args.slope if args.slope is not None else 0.01,
        )
    elif args.approx_sweep:
        plot_cond_table_approx_dp_sweep(slope=args.slope)
    elif args.pure_sweep:
        plot_cond_table_pure_dp_sweep(slope=args.slope)
    elif args.pure:
        plot_cond_table_pure_dp(slopes=tuple(args.slopes) if args.slopes else None)
    elif args.swept:
        plot_cond_table_approx_dp_swept(slope=args.slope)
    else:
        # Default: Option 1, the rigorous beta-valid feasibility band.
        plot_cond_table_approx_dp_valid(slopes=tuple(args.slopes) if args.slopes else (0.02, 0.05, 0.1))
