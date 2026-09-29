"""
One-step Cauchy vs Laplace noise decision for AGS privatisation.

For s_p = 1 the decision reduces to a scalar threshold r* on
r = SS_beta / Delta_f.  This module computes r* analytically,
stores numerical quantile tables for debugging, and verifies
the threshold via Monte Carlo simulation.

Privacy caveat
--------------
Consulting data-dependent SS values at runtime breaks DP.
This is an oracle / performance-ceiling analysis only.
"""

import dataclasses
import math
import os
from typing import Optional

import numpy as np


# ── Analytical threshold ────────────────────────────────────────────


def compute_r_star(alpha: float) -> float:
    """Compute the critical ratio r* for the one-step dominance condition.

    Cauchy beats Laplace iff r = SS_beta / Delta_f < r*(alpha).

    Args:
        alpha: Failure probability (e.g. 0.05). Must be in (0, 1).

    Returns:
        r_star: The threshold ratio.
    """
    assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
    return math.log(1.0 / alpha) / (6.0 * math.tan(math.pi * (1 - alpha) / 2))


def compute_r_star_sweep(alphas: np.ndarray) -> np.ndarray:
    """Vectorised r* over an array of alpha values."""
    alphas = np.asarray(alphas, dtype=np.float64)
    return np.log(1.0 / alphas) / (6.0 * np.tan(np.pi * (1.0 - alphas) / 2.0))


# ── Quantile formulas (normalised: Delta_f=1, epsilon=1) ───────────


def cauchy_quantile_normalized(r: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha)-quantile of |Cauchy(0, 6r)|, normalised with Delta_f=1, eps=1.

    To recover actual bounds: multiply by (Delta_f / epsilon).
    """
    return 6.0 * np.asarray(r, dtype=np.float64) * math.tan(math.pi * (1 - alpha) / 2)


def laplace_quantile_normalized(alpha: float) -> float:
    """(1-alpha)-quantile of |Lap(0, 1)|, normalised with Delta_f=1, eps=1.

    To recover actual bounds: multiply by (Delta_f / epsilon).
    """
    return math.log(1.0 / alpha)


# ── Scalar decision function ───────────────────────────────────────


def decide(r: float, alpha: float) -> tuple:
    """Return (use_cauchy, margin) for a single (r, alpha) pair.

    Returns:
        use_cauchy: True if Cauchy dominates Laplace.
        margin: Positive means Cauchy wins. Equals r* - r.
    """
    r_star = compute_r_star(alpha)
    margin = r_star - r
    return margin > 0, margin


# ── Decision table with quantile storage ────────────────────────────


@dataclasses.dataclass
class NoiseDecisionTable:
    """Pre-computed decision table with quantile values and interpolation.

    For s_p=1 the table stores, for each r on a 1-D log grid:
      - cauchy_bounds_norm: normalised (1-alpha)-quantile of |Cauchy| noise
      - laplace_bound_norm: normalised (1-alpha)-quantile of |Laplace| noise
      - decision: bool array (True = Cauchy wins)
      - margin: signed distance r* - r

    Query with actual (SS, Delta_f, epsilon) to get interpolated decisions
    and actual quantile bounds in original scale.

    """

    # ── Core fields ──
    s_p: int
    alpha: float
    r_star: float

    # ── 1-D grid (s_p=1) ──
    r_grid: np.ndarray  # shape (N,)
    cauchy_bounds_norm: np.ndarray  # shape (N,) — normalised quantiles
    laplace_bound_norm: float  # scalar
    decision: np.ndarray  # shape (N,), bool
    margin: np.ndarray  # shape (N,), float

    # ── MC verification results (optional) ──
    mc_cauchy_quantiles: Optional[np.ndarray] = None  # shape (N,)
    mc_laplace_quantile: Optional[float] = None
    mc_n_samples: Optional[int] = None

    # ── Query (scalar) ──

    def query(
        self,
        ss_beta: float,
        delta_f: float,
        epsilon_window: Optional[float] = None,
    ) -> dict:
        """Look up the noise decision with linear interpolation.

        Args:
            ss_beta: Smooth sensitivity value for this parameter.
            delta_f: Global sensitivity.
            epsilon_window: Per-window privacy budget (for scaling bounds).

        Returns:
            Dict with use_cauchy, margin, r, cauchy_bound_norm,
            laplace_bound_norm, and optionally cauchy_bound / laplace_bound
            (actual scale if epsilon_window is given).
        """
        r = ss_beta / delta_f
        margin_val = float(np.interp(r, self.r_grid, self.margin))
        cauchy_q = float(np.interp(r, self.r_grid, self.cauchy_bounds_norm))
        result = {
            "use_cauchy": margin_val > 0,
            "margin": margin_val,
            "r": r,
            "cauchy_bound_norm": cauchy_q,
            "laplace_bound_norm": self.laplace_bound_norm,
        }
        if epsilon_window is not None:
            scale = delta_f / epsilon_window
            result["cauchy_bound"] = cauchy_q * scale
            result["laplace_bound"] = self.laplace_bound_norm * scale
        return result

    # ── Query (batch of SS values) ──

    def query_batch(
        self,
        ss_beta_values: np.ndarray,
        delta_f: float,
        epsilon_window: Optional[float] = None,
    ) -> dict:
        """Vectorised query for an array of smooth sensitivity values.

        Args:
            ss_beta_values: Array of SS values (one per parameter).
            delta_f: Global sensitivity (scalar).
            epsilon_window: Per-window privacy budget.

        Returns:
            Dict with arrays: use_cauchy, margin, r, bounds.
        """
        r_vals = np.asarray(ss_beta_values, dtype=np.float64) / delta_f
        margins = np.interp(r_vals, self.r_grid, self.margin)
        cauchy_qs = np.interp(r_vals, self.r_grid, self.cauchy_bounds_norm)

        result = {
            "use_cauchy": margins > 0,
            "margin": margins,
            "r": r_vals,
            "cauchy_bound_norm": cauchy_qs,
            "laplace_bound_norm": self.laplace_bound_norm,
        }
        if epsilon_window is not None:
            scale = delta_f / epsilon_window
            result["cauchy_bound"] = cauchy_qs * scale
            result["laplace_bound"] = self.laplace_bound_norm * scale
        return result

    # ── Persistence ──

    def save(self, path: str) -> None:
        """Save to .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = dict(
            s_p=self.s_p,
            alpha=self.alpha,
            r_star=self.r_star,
            r_grid=self.r_grid,
            cauchy_bounds_norm=self.cauchy_bounds_norm,
            laplace_bound_norm=self.laplace_bound_norm,
            decision=self.decision,
            margin=self.margin,
        )
        if self.mc_cauchy_quantiles is not None:
            data["mc_cauchy_quantiles"] = self.mc_cauchy_quantiles
            data["mc_laplace_quantile"] = self.mc_laplace_quantile
            data["mc_n_samples"] = self.mc_n_samples
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "NoiseDecisionTable":
        """Load from .npz file."""
        d = np.load(path, allow_pickle=True)
        kwargs = dict(
            s_p=int(d["s_p"]),
            alpha=float(d["alpha"]),
            r_star=float(d["r_star"]),
            r_grid=d["r_grid"],
            cauchy_bounds_norm=d["cauchy_bounds_norm"],
            laplace_bound_norm=float(d["laplace_bound_norm"]),
            decision=d["decision"].astype(bool),
            margin=d["margin"],
        )
        if "mc_cauchy_quantiles" in d:
            kwargs["mc_cauchy_quantiles"] = d["mc_cauchy_quantiles"]
            kwargs["mc_laplace_quantile"] = float(d["mc_laplace_quantile"])
            kwargs["mc_n_samples"] = int(d["mc_n_samples"])
        return cls(**kwargs)

    def save_csv(self, path: str) -> None:
        """Save the table as a human-readable CSV file.

        Columns: r, cauchy_bound_norm, laplace_bound_norm, margin, decision,
        and optionally mc_cauchy_quantile, mc_laplace_quantile.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = len(self.r_grid)

        # Build columns
        cols = {
            "r": self.r_grid,
            "cauchy_bound_norm": self.cauchy_bounds_norm,
            "laplace_bound_norm": np.full(n, self.laplace_bound_norm),
            "margin": self.margin,
            "decision": np.where(self.decision, "cauchy", "laplace"),
        }
        if self.mc_cauchy_quantiles is not None:
            cols["mc_cauchy_quantile"] = self.mc_cauchy_quantiles
            cols["mc_laplace_quantile"] = np.full(n, self.mc_laplace_quantile)

        header_parts = [f"# s_p={self.s_p}  alpha={self.alpha}  r_star={self.r_star:.8f}"]
        if self.mc_n_samples is not None:
            header_parts.append(f"  mc_n_samples={self.mc_n_samples}")
        header = "".join(header_parts)

        col_names = list(cols.keys())
        with open(path, "w") as f:
            f.write(header + "\n")
            f.write(",".join(col_names) + "\n")
            for i in range(n):
                row = []
                for name in col_names:
                    val = cols[name][i]
                    if isinstance(val, str):
                        row.append(val)
                    else:
                        row.append(f"{val:.8e}")
                f.write(",".join(row) + "\n")

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"NoiseDecisionTable(s_p={self.s_p}, alpha={self.alpha})",
            f"  r* (analytic)        = {self.r_star:.6f}",
            f"  Laplace bound (norm) = {self.laplace_bound_norm:.6f}",
            f"  r grid: [{self.r_grid[0]:.2e}, {self.r_grid[-1]:.2e}], {len(self.r_grid)} points",
            f"  Cauchy wins: {self.decision.sum()}/{len(self.decision)} grid points",
        ]
        if self.mc_cauchy_quantiles is not None:
            lines.append(f"  MC verification: {self.mc_n_samples:,} samples")
            crossings = np.where(np.diff(self.mc_cauchy_quantiles < self.mc_laplace_quantile))[0]
            if len(crossings) > 0:
                emp_r = 0.5 * (self.r_grid[crossings[0]] + self.r_grid[crossings[0] + 1])
                lines.append(f"  MC empirical r* ~ {emp_r:.6f} (analytic: {self.r_star:.6f})")
        return "\n".join(lines)


# ── Table builder ───────────────────────────────────────────────────


def build_one_step_table(
    alpha: float,
    r_min: float = 1e-4,
    r_max: float = 1.0,
    n_points: int = 1000,
    mc_verify: bool = True,
    mc_n_samples: int = 2_000_000,
    mc_seed: int = 42,
) -> NoiseDecisionTable:
    """Build a one-step (s_p=1) decision table.

    Args:
        alpha: Failure probability.
        r_min: Minimum r value on log grid.
        r_max: Maximum r value on log grid.
        n_points: Number of grid points (log-spaced).
        mc_verify: Whether to run Monte Carlo verification.
        mc_n_samples: Number of MC samples for verification.
        mc_seed: Random seed for MC.

    Returns:
        NoiseDecisionTable with analytical quantiles + optional MC results.
    """
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    r_star = compute_r_star(alpha)

    # Analytical quantiles (normalised: Delta_f=1, epsilon=1)
    cauchy_bounds = cauchy_quantile_normalized(r_grid, alpha)
    lap_bound = laplace_quantile_normalized(alpha)

    decision = cauchy_bounds < lap_bound
    margin = r_star - r_grid  # positive ↔ Cauchy wins

    table = NoiseDecisionTable(
        s_p=1,
        alpha=alpha,
        r_star=r_star,
        r_grid=r_grid,
        cauchy_bounds_norm=cauchy_bounds,
        laplace_bound_norm=lap_bound,
        decision=decision,
        margin=margin,
    )

    if mc_verify:
        mc = _mc_verify(r_grid, alpha, mc_n_samples, mc_seed)
        table.mc_cauchy_quantiles = mc["cauchy_quantiles"]
        table.mc_laplace_quantile = mc["laplace_quantile"]
        table.mc_n_samples = mc_n_samples

    return table


def _mc_verify(
    r_grid: np.ndarray,
    alpha: float,
    n_samples: int,
    seed: int,
) -> dict:
    """Monte Carlo verification of the analytical quantiles.

    Draws from |standard_cauchy| and |Lap(0,1)| and computes
    empirical (1-alpha)-quantiles.  Since quantile commutes with
    positive scaling, we compute the standard quantile once and
    scale by 6*r for each grid point.
    """
    rng = np.random.default_rng(seed)
    q_level = 1 - alpha

    # Laplace: |Lap(0, 1)| quantile
    lap_samples = np.abs(rng.laplace(0, 1.0, size=n_samples))
    lap_q = float(np.quantile(lap_samples, q_level))

    # Cauchy: |standard_cauchy| quantile, then scale
    std_cauchy_abs = np.abs(rng.standard_cauchy(n_samples))
    std_cauchy_q = float(np.quantile(std_cauchy_abs, q_level))
    cauchy_qs = 6.0 * r_grid * std_cauchy_q

    return {
        "cauchy_quantiles": cauchy_qs,
        "laplace_quantile": lap_q,
        "std_cauchy_quantile": std_cauchy_q,
    }
