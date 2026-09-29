"""A sound replacement for the package's l2 ("norm") gradient clipping.

`abstract_gradient_training.training_utils.propagate_clipping` reshapes the per-sample clip factor
with `view(-1, 1, 1)`, which assumes every per-sample gradient has exactly two trailing dimensions.
A Linear layer's bias gradient has one, so the branch is broken for any model with biases and the
factor is silently broadcast into the wrong shape. The patch below reshapes over however many
trailing dimensions the tensor actually has, and additionally replaces a non-finite pre-clipping
interval by the sound box [-gamma, gamma], since every coordinate of an l2-clipped vector lies in it
and `inf * 0` would otherwise produce NaN.

This lived inside a driver and was monkeypatched by hand, two lines at a time, in four places. Use:

    from common import agt_patches
    agt_patches.install()                 # for a script that patches once and runs
    with agt_patches.patched():           # for anything that should leave the package as it found it
        ...

`selftest()` is a randomized containment check: every concrete gradient inside the input interval
must clip to a point inside the output interval. Drivers run it before training as a guard.

The defect is recorded in Notes/REPO_NOTES.md section 3 and Notes/PACKAGE_CHANGES.md; the package
itself is deliberately left unmodified.
"""

from __future__ import annotations

import contextlib

import torch

from abstract_gradient_training import interval_arithmetic, training_utils


def propagate_clipping_patched(x_l, x, x_u, gamma, method):
    """training_utils.propagate_clipping with the per-sample factor broadcast over
    however many trailing dims the tensor has (the package assumes exactly two)."""
    if method != "norm":
        return _ORIG_PROPAGATE_CLIPPING(x_l, x, x_u, gamma, method)
    interval_arithmetic.validate_interval(x_l, x_u, msg="input")
    for i in range(len(x_l)):
        shape = [-1] + [1] * (x[i].dim() - 1)
        # Once the parameter box is large, pre-clipping gradient intervals can overflow to
        # +-inf; inf * 0 (clip factor lower bound) would give NaN.  Every coordinate of an
        # l2-clipped vector lies in [-gamma, gamma], so for any sample with a non-finite
        # interval that box is a sound replacement, and the product below is clamped to it.
        bad = ~(torch.isfinite(x_l[i]) & torch.isfinite(x_u[i])).flatten(1).all(dim=1)
        xl_i = torch.where(bad.view(*shape), torch.full_like(x_l[i], -gamma), x_l[i])
        xu_i = torch.where(bad.view(*shape), torch.full_like(x_u[i], gamma), x_u[i])
        norms = x[i].flatten(1).norm(2, dim=1)
        norms_l, norms_u = interval_arithmetic.propagate_norm(xl_i, xu_i, p=2)
        interval_arithmetic.validate_interval(norms_l, norms_u, msg="norm")
        clip_factor = (gamma / (norms + 1e-6)).clamp(max=1.0)
        clip_factor_l = (gamma / (norms_u + 1e-6)).clamp(max=1.0)
        clip_factor_u = (gamma / (norms_l + 1e-6)).clamp(max=1.0)
        interval_arithmetic.validate_interval(clip_factor_l, clip_factor_u, msg="clip factor")
        lo, hi = interval_arithmetic.propagate_elementwise(
            xl_i, xu_i, clip_factor_l.view(*shape), clip_factor_u.view(*shape)
        )
        lo = torch.nan_to_num(lo, nan=-gamma).clamp(min=-gamma, max=gamma)
        hi = torch.nan_to_num(hi, nan=gamma).clamp(min=-gamma, max=gamma)
        x_l[i], x_u[i] = lo, hi
        x[i] = x[i] * clip_factor.view(*shape)
    interval_arithmetic.validate_interval(x_l, x_u, msg="clipped input")
    return x_l, x, x_u


_ORIG_PROPAGATE_CLIPPING = training_utils.propagate_clipping


def selftest(gamma=1.0, trials=200, seed=0):
    """Randomized containment check of the patched norm branch: every concrete gradient
    inside the input interval must clip to a point inside the output interval."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(trials):
        b = 7
        shapes = [(b, 5, 3), (b, 5), (b, 2, 5), (b, 2)]
        centre = [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]
        rad = [torch.rand(s, generator=g, dtype=torch.float64) * 0.5 for s in shapes]
        x_l = [c - r for c, r in zip(centre, rad)]
        x_u = [c + r for c, r in zip(centre, rad)]
        sample = [
            l + (u - l) * torch.rand(l.shape, generator=g, dtype=torch.float64) for l, u in zip(x_l, x_u)
        ]
        cl, cn, cu = propagate_clipping_patched(
            [t.clone() for t in x_l], [t.clone() for t in centre], [t.clone() for t in x_u], gamma, "norm"
        )
        # exact clipping of the sampled point
        for i, s in enumerate(sample):
            f = (
                (gamma / (s.flatten(1).norm(2, dim=1) + 1e-6))
                .clamp(max=1.0)
                .view(*([-1] + [1] * (s.dim() - 1)))
            )
            sc = s * f
            if not (torch.all(sc >= cl[i] - 1e-12) and torch.all(sc <= cu[i] + 1e-12)):
                raise RuntimeError("patched norm clipping is NOT sound on a random instance")
            if not (torch.all(cn[i] >= cl[i] - 1e-12) and torch.all(cn[i] <= cu[i] + 1e-12)):
                raise RuntimeError("patched norm clipping: nominal outside its own interval")
    return True


def install() -> None:
    """Install the patch on both module references the package reaches it through."""
    import abstract_gradient_training.certified_training.privacy as _privacy

    training_utils.propagate_clipping = propagate_clipping_patched
    _privacy.training_utils.propagate_clipping = propagate_clipping_patched


def uninstall() -> None:
    import abstract_gradient_training.certified_training.privacy as _privacy

    training_utils.propagate_clipping = _ORIG_PROPAGATE_CLIPPING
    _privacy.training_utils.propagate_clipping = _ORIG_PROPAGATE_CLIPPING


@contextlib.contextmanager
def patched():
    """Install the patch for the duration of the block, then restore the package."""
    install()
    try:
        yield
    finally:
        uninstall()


# The name the drivers used before this module existed.
selftest_clipping = selftest
