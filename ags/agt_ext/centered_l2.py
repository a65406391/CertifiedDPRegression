"""L2 regularization toward a public warm start, for the package's bounded SGD optimizer.

`abstract_gradient_training.bounded_optimizers.BoundedSGD` adds l2_reg * theta to every gradient
bound, so the trajectory contracts toward zero. A head fine-tuned from a public warm start theta_pub
should contract toward theta_pub instead:

    theta <- theta - lr * (g + lambda * (theta - theta_pub))

The shift is a constant, so nothing about the interval propagation changes: g + lambda (theta - c)
= (g - lambda c) + lambda theta, and the package's own sound pairing (the lower gradient bound takes
the upper parameter bound and vice versa) applies to the shifted gradient interval verbatim. The
certified envelope is identical to the uncentered run from the same start; only the fixed point moves.

WRITTEN AS AN ADDITION. `CenteredBoundedSGD` subclasses the package's optimizer and overrides `step`;
`centered()` makes `AGTConfig.get_bounded_optimizer` construct it for the duration of a block and
restores the package afterwards. The center travels through `AGTConfig.optimizer_kwargs["l2_center"]`
as a list of tensors in the model's parameter order; without it the subclass is the package's
optimizer bit for bit. Recorded in Notes/PACKAGE_CHANGES.md section 4. Test: test_centered_l2.py.
"""

from __future__ import annotations

import contextlib

import torch

from abstract_gradient_training import bounded_optimizers
from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD


class CenteredBoundedSGD(BoundedSGD):
    """BoundedSGD whose L2 term pulls toward `l2_center` (a list of tensors) instead of zero."""

    def __init__(self, bounded_model, **optimizer_kwargs):
        center = optimizer_kwargs.pop("l2_center", None)
        super().__init__(bounded_model, **optimizer_kwargs)
        if center is None:
            self.l2_center = None
            return
        if len(center) != len(self.param_n):
            raise ValueError(f"l2_center has {len(center)} tensors, the model {len(self.param_n)}")
        self.l2_center = []
        for c, p in zip(center, self.param_n):
            c = torch.as_tensor(c, dtype=p.dtype, device=p.device)
            if c.shape != p.shape:
                raise ValueError(f"l2_center shape {tuple(c.shape)} != parameter shape {tuple(p.shape)}")
            self.l2_center.append(c)

    def step(self, mean_grads_l, mean_grads_n, mean_grads_u):
        if self.l2_center is None or self.l2_reg == 0:
            return super().step(mean_grads_l, mean_grads_n, mean_grads_u)
        # g + lambda (p - c) = (g - lambda c) + lambda p: shift all three gradient bounds by the same
        # constant and let the package add lambda p with its own sound pairing.
        shift = [self.l2_reg * c for c in self.l2_center]
        return super().step(
            [g - s for g, s in zip(mean_grads_l, shift)],
            [g - s for g, s in zip(mean_grads_n, shift)],
            [g - s for g, s in zip(mean_grads_u, shift)],
        )


@contextlib.contextmanager
def centered():
    """Within the block, AGTConfig.get_bounded_optimizer builds CenteredBoundedSGD for optimizer="SGD"."""
    original = bounded_optimizers.BoundedSGD
    bounded_optimizers.BoundedSGD = CenteredBoundedSGD
    try:
        yield
    finally:
        bounded_optimizers.BoundedSGD = original
