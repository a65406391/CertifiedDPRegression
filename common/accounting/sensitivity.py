"""One-step coordinate sensitivity certificates.

The routines in this module deliberately distinguish reachable envelopes from
the Nissim ``A^(k)`` quantity.  A reachable envelope alone is never returned as
a local-sensitivity certificate; adjacent radii are crossed as specified in the
corrected mathematical contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReachableEnvelopes:
    """Coordinate boxes for every substitution radius from zero to ``m``."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if self.lower.ndim != 2 or self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have shape [m + 1, p]")
        if self.lower.shape[0] < 1:
            raise ValueError("at least the radius-zero envelope is required")
        if torch.any(self.lower > self.upper):
            raise ValueError("invalid envelope: lower exceeds upper")

    @property
    def diameter(self) -> int:
        return self.lower.shape[0] - 1


def one_step_envelopes(
    clipped_gradients: torch.Tensor,
    coordinate_caps: torch.Tensor,
    *,
    learning_rate: float = 1.0,
    theta: torch.Tensor | None = None,
) -> ReachableEnvelopes:
    """Return exact coordinate boxes induced by the order-statistic formula.

    ``clipped_gradients`` has shape ``[m, p]``.  Each replacement gradient may
    take any value in the public coordinate box ``[-C_j, C_j]``.  The returned
    boxes describe ``theta - learning_rate * mean_gradient``.
    """
    if clipped_gradients.ndim != 2:
        raise ValueError("clipped_gradients must have shape [m, p]")
    m, p = clipped_gradients.shape
    caps = torch.as_tensor(
        coordinate_caps,
        dtype=clipped_gradients.dtype,
        device=clipped_gradients.device,
    )
    if caps.shape != (p,) or torch.any(caps < 0):
        raise ValueError("coordinate_caps must be a non-negative length-p vector")
    if torch.any(torch.abs(clipped_gradients) > caps + 1e-12):
        raise ValueError("a supplied gradient exceeds its public coordinate cap")
    if not learning_rate >= 0:
        raise ValueError("learning_rate must be non-negative")

    base = torch.zeros(p, dtype=clipped_gradients.dtype, device=clipped_gradients.device)
    if theta is not None:
        base = torch.as_tensor(theta, dtype=base.dtype, device=base.device)
        if base.shape != (p,):
            raise ValueError("theta must be a length-p vector")

    sorted_grads = torch.sort(clipped_gradients, dim=0).values
    prefix = torch.cat([torch.zeros_like(sorted_grads[:1]), sorted_grads.cumsum(0)])
    total = prefix[m]
    lowers = []
    uppers = []
    for k in range(m + 1):
        retained = m - k
        min_sum = prefix[retained] - k * caps
        max_sum = total - prefix[k] + k * caps
        grad_lower = min_sum / m
        grad_upper = max_sum / m
        # Subtraction reverses the gradient interval.
        lowers.append(base - learning_rate * grad_upper)
        uppers.append(base - learning_rate * grad_lower)
    return ReachableEnvelopes(torch.stack(lowers), torch.stack(uppers))


def cross_radius_a_bar(envelopes: ReachableEnvelopes) -> torch.Tensor:
    """Compute coordinate ``Abar_j^(k)`` for all ``k=0,...,m``.

    The final row crosses ``T_m`` with itself because a fixed-size batch has
    substitution diameter ``m``.
    """
    lower = envelopes.lower
    upper = envelopes.upper
    next_lower = torch.cat([lower[1:], lower[-1:]], dim=0)
    next_upper = torch.cat([upper[1:], upper[-1:]], dim=0)
    return torch.maximum((lower - next_upper).abs(), (upper - next_lower).abs())


def smooth_sensitivity(a_bar: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    """Return per-coordinate ``max_k exp(-beta_j*k) Abar_j^(k)``."""
    if a_bar.ndim != 2:
        raise ValueError("a_bar must have shape [m + 1, p]")
    beta_t = torch.as_tensor(beta, dtype=a_bar.dtype, device=a_bar.device)
    if beta_t.ndim == 0:
        beta_t = beta_t.expand(a_bar.shape[1])
    if beta_t.shape != (a_bar.shape[1],) or torch.any(beta_t < 0):
        raise ValueError("beta must be non-negative and scalar or length p")
    radii = torch.arange(a_bar.shape[0], dtype=a_bar.dtype, device=a_bar.device)
    decay = torch.exp(-radii[:, None] * beta_t[None, :])
    return torch.max(decay * a_bar, dim=0).values


def coordinate_global_sensitivity(
    coordinate_caps: torch.Tensor, m: int, learning_rate: float = 1.0
) -> torch.Tensor:
    """Clipping-derived sensitivity of one mean-gradient SGD update."""
    if m <= 0:
        raise ValueError("m must be positive")
    return 2.0 * learning_rate * torch.as_tensor(coordinate_caps) / m


def clipped_mean_smooth_sensitivity(
    clipped_gradients: torch.Tensor,
    coordinate_caps: torch.Tensor,
    beta: torch.Tensor | float,
    *,
    learning_rate: float = 1.0,
) -> torch.Tensor:
    """Specialized smooth sensitivity for a coordinate-clipped mean update."""
    if clipped_gradients.ndim != 2 or clipped_gradients.shape[0] == 0:
        raise ValueError("clipped_gradients must have nonempty shape [m, p]")
    m, p = clipped_gradients.shape
    caps = torch.as_tensor(
        coordinate_caps,
        dtype=clipped_gradients.dtype,
        device=clipped_gradients.device,
    )
    if caps.shape != (p,) or torch.any(caps < 0):
        raise ValueError("coordinate_caps must be a non-negative length-p vector")
    if torch.any(torch.abs(clipped_gradients) > caps + 1e-12):
        raise ValueError("a supplied gradient exceeds its public coordinate cap")
    beta_t = torch.as_tensor(beta, dtype=clipped_gradients.dtype, device=clipped_gradients.device)
    if beta_t.ndim == 0:
        beta_t = beta_t.expand(p)
    if beta_t.shape != (p,) or torch.any(beta_t < 0):
        raise ValueError("beta must be non-negative and scalar or length p")
    local = learning_rate * (caps + clipped_gradients.abs().amax(dim=0)) / m
    global_sensitivity = 2.0 * learning_rate * caps / m
    result = torch.maximum(local, torch.exp(-beta_t) * global_sensitivity)
    return torch.minimum(result, global_sensitivity)
