"""The cross-radius certificate and the smooth-sensitivity staircase.

This is the paper's Theorem 4.2 object, in one place. Before this module it existed in nine
independent copies whose conventions differed in ways that mattered, which is how the
width-versus-radius defect and the sparse-grid smoothness defect both went unnoticed.

    A(x, k) = max( y_u^k - y_l^(k+1),  y_u^(k+1) - y_l^k ),      k = 0 .. k_max
    SS^beta(x) = max( max_k e^(-beta k) A(x, k),  e^(-beta (k_max+1)) * tail_width )

`k = 0` is the nominal point, so `T_0` is the degenerate box `[y_n, y_n]`. The box that pairs with
the largest computed radius is the `tail` argument: the data-independent clipping ceiling where one
is available, otherwise the public output range. There is no radius beyond it, so no analytical tail
term is invented (see docs on the k-battery rule in Notes/REPO_NOTES.md section 5).

Two properties the callers rely on, both checked in common/tests/test_certificate.py:
  * boxes must be nested, `T_k subset T_(k+1)`; `nest` enforces it with a running hull.
  * a gap in the radius grid is filled upward by `backfill`, with the smallest computed radius that
    is at least the missing one. That keeps the bound sound, but it is only (beta*gap)-smooth, not
    beta-smooth, so `cross_radius` warns unless the grid is dense.
"""

from __future__ import annotations

import warnings

import numpy as np

Box = tuple[np.ndarray, np.ndarray]


def nest(boxes: dict[int, Box]) -> dict[int, Box]:
    """Return the boxes made nested by a running hull over increasing k.

    Certified boxes should already satisfy `T_k subset T_(k+1)`; solver tolerances can break it by
    a few ulp. Taking the running hull restores the property without ever tightening a bound.
    """
    out, prev = {}, None
    for k in sorted(boxes):
        lo, hi = (np.asarray(v, dtype=float) for v in boxes[k])
        if prev is not None:
            lo, hi = np.minimum(lo, out[prev][0]), np.maximum(hi, out[prev][1])
        out[k] = (lo, hi)
        prev = k
    return out


def backfill(boxes: dict[int, Box], k_max: int) -> dict[int, Box]:
    """Fill gaps in the radius grid with the smallest computed radius that is at least the missing one.

    Sound because the reachable sets are nested, but see the module docstring on smoothness.
    """
    have = sorted(boxes)
    if not have:
        raise ValueError("no boxes")
    out = {}
    for k in range(min(have), k_max + 1):
        nxt = next((c for c in have if c >= k), have[-1])
        out[k] = boxes[nxt]
    return out


def cross_radius(
    nominal: np.ndarray,
    boxes: dict[int, Box],
    tail: Box,
    *,
    cap: float | None = None,
    warn_on_gaps: bool = True,
) -> np.ndarray:
    """A(x, k) for k = 0 .. k_max, shape (k_max + 1, n).

    Args:
        nominal: (n,) the nominal prediction; the k = 0 box is [nominal, nominal].
        boxes: radius -> (lower, upper), each (n,). Radii need not be consecutive; gaps are
            back-filled. Must not contain k = 0.
        tail: the box pairing with k_max, i.e. `T_(k_max+1)`. The clipping ceiling where one exists,
            otherwise the public output range.
        cap: optional upper bound on every entry, normally the global sensitivity.
        warn_on_gaps: warn when the grid is not dense, since back-filling costs smoothness.

    Returns:
        A with A[k] the certificate at radius k.
    """
    nominal = np.asarray(nominal, dtype=float)
    if 0 in boxes:
        raise ValueError("k = 0 is the nominal point and is supplied by `nominal`, not in `boxes`")
    radii = sorted(boxes)
    if not radii:
        raise ValueError("no boxes")
    k_max = radii[-1]
    if warn_on_gaps and radii != list(range(radii[0], k_max + 1)):
        missing = sorted(set(range(radii[0], k_max + 1)) - set(radii))
        warnings.warn(
            f"radius grid has gaps at {missing[:8]}{'...' if len(missing) > 8 else ''}; the "
            "back-filled bound is only (beta*gap)-smooth, not beta-smooth. Run the missing radii, "
            "or discount each gap by its first index.",
            stacklevel=2,
        )
    full = nest(backfill(boxes, k_max))
    lo = {0: nominal, **{k: full[k][0] for k in full}}
    hi = {0: nominal, **{k: full[k][1] for k in full}}
    lo[k_max + 1] = np.asarray(tail[0], dtype=float)
    hi[k_max + 1] = np.asarray(tail[1], dtype=float)

    A = np.stack([np.maximum(hi[k] - lo[k + 1], hi[k + 1] - lo[k]) for k in range(k_max + 1)])
    return A if cap is None else np.minimum(A, cap)


def smooth_sensitivity(A: np.ndarray, beta: float, tail_width: np.ndarray | float) -> np.ndarray:
    """max_k e^(-beta k) A[k], floored by the discounted tail e^(-beta (k_max+1)) * tail_width.

    `beta` must already respect the mechanism's limit; use common.mechanisms.beta_max.
    """
    A = np.asarray(A, dtype=float)
    if beta < 0:
        raise ValueError("beta must be non-negative")
    k_max = A.shape[0] - 1
    discounted = (np.exp(-beta * np.arange(k_max + 1))[:, None] * A).max(axis=0)
    return np.maximum(discounted, np.exp(-beta * (k_max + 1)) * np.asarray(tail_width, dtype=float))


def affine_output_interval(box: Box, x: np.ndarray, clip: float | None = None) -> Box:
    """Interval image of a 1-D affine model y = w x + b over a parameter box, optionally clipped.

    `box` is ((w_lo, b_lo), (w_hi, b_hi)). Exact, because the model is monotone in each parameter.
    """
    (w_lo, b_lo), (w_hi, b_hi) = box
    x = np.asarray(x, dtype=float)
    products = np.stack([w_lo * x, w_hi * x])
    lo, hi = products.min(axis=0) + b_lo, products.max(axis=0) + b_hi
    if clip is not None:
        lo, hi = np.clip(lo, -clip, clip), np.clip(hi, -clip, clip)
    return lo, hi
