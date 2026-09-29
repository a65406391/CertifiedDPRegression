"""Bounded version of the Hinge loss function."""

import logging
from typing import Literal

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_losses import BoundedLoss

LOGGER = logging.getLogger(__name__)


class BoundedHingeLoss(BoundedLoss):
    """Bounded version of the Hinge loss function.

    The loss is given by

        l(x, y) = max(0, margin - y * x)

    This is not the same as the HingeEmbeddingLoss in PyTorch, which has a
    slightly different form.
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialize the BoundedHingeLoss loss function.

        Args:
            margin: The margin for the hinge loss. Defaults to 1.0.
            reduction: Specifies the reduction to apply to the output.
        """
        self.margin = margin
        self.reduction = reduction
        if reduction not in ["sum", "mean", "none"]:
            raise ValueError(f"Reduction {reduction} not recognised.")

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Nominal forward pass through the loss function.

        Args:
            inputs: Inputs to the loss function.
            targets: Target values for the loss function.

        Returns:
            torch.Tensor: Loss values.
        """
        # the hinge loss expects targets to be -1 or 1, but in AGT we'll allow
        # 0 or 1 and transform them appropriately
        if ((targets == 0) | (targets == 1)).all():
            targets = 2 * targets - 1
        if not ((targets == 1) | (targets == -1)).all():
            raise ValueError("Labels must be binary for hinge loss.")

        # handle the shapes and types
        input_shape = inputs.shape  # save the input shape
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs.dtype)
        try:
            inputs = inputs.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"{inputs.shape=} does not match {targets.shape=} for BCE."
            ) from e

        # compute the loss
        loss = (self.margin - targets * inputs).relu()

        # perform the reduction over any remaining dimensions
        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        return loss.reshape(input_shape)

    def backward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the gradients of the loss wrt inputs.

        Args:
            inputs: Inputs to the loss function.
            targets: Target values for the loss function.

        Returns:
            torch.Tensor: Loss gradient values.
        """
        # the hinge loss expects targets to be -1 or 1, but in AGT we'll allow
        # 0 or 1 and transform them appropriately
        if ((targets == 0) | (targets == 1)).all():
            targets = 2 * targets - 1
        if not ((targets == 1) | (targets == -1)).all():
            raise ValueError("Labels must be binary for hinge loss.")

        # handle the shapes and types
        input_shape = inputs.shape  # save the input shape
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs.dtype)
        try:
            inputs = inputs.reshape(targets.shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"{inputs.shape=} does not match {targets.shape=} for BCE."
            ) from e

        # compute the loss gradient (there are three cases to consider)
        grad = (
            (-targets * (inputs * targets < self.margin))
            .type(inputs.dtype)
            .reshape(input_shape)
        )

        if self.reduction == "mean":
            LOGGER.warning(
                "AGT processes grads in fragments, so using 'reduction=mean' "
                "is not recommended."
            )
            LOGGER.warning(
                "Specify 'reduction=none' and normalise manually to ensure "
                "correct scaling."
            )
            return grad / grad.nelement()
        return grad

    def bound_forward(
        self,
        inputs_l: torch.Tensor,
        inputs_u: torch.Tensor,
        targets: torch.Tensor,
        *,
        label_k_poison: int = 0,
        label_epsilon: float = 0.0,
        poison_target_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded forward pass through the loss function.

        The forward pass of this loss does not support label poisoning.

        Args:
            inputs_l: Lower bounds on the inputs to the loss function.
            inputs_u: Upper bounds on the inputs to the loss function.
            targets: Target values for the loss function.
            label_k_poison: Not used for this loss.
            label_epsilon: Not used for this loss.
            poison_target_idx: Not used for this loss.

        Returns:
            tuple: Lower and upper bounds on the loss values.
        """
        # validate input
        if label_k_poison > 0 or label_epsilon > 0 or poison_target_idx >= 0:
            raise ValueError(
                "Hinge loss does not support label poisoning in forward pass."
            )
        # the hinge loss expects targets to be -1 or 1, but in AGT we'll allow
        # 0 or 1 and transform them appropriately
        if ((targets == 0) | (targets == 1)).all():
            targets = 2 * targets - 1
        if not ((targets == 1) | (targets == -1)).all():
            raise ValueError("Labels must be binary for hinge loss.")

        interval_arithmetic.validate_interval(
            inputs_l, inputs_u, msg="loss input bounds"
        )

        # handle shapes and types
        input_shape = inputs_l.shape  # save the input shape
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs_l.dtype)
        try:
            inputs_l, inputs_u = (
                inputs_l.reshape(targets.shape),
                inputs_u.reshape(targets.shape),
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{inputs_l.shape=} does not match {targets.shape=} for BCE."
            ) from e

        # compute the min and max of the hinge loss value
        inter_l = torch.where(targets == 1, inputs_l, -inputs_u)
        inter_u = torch.where(targets == 1, inputs_u, -inputs_l)
        loss_l = (self.margin - inter_u).relu()
        loss_u = (self.margin - inter_l).relu()
        interval_arithmetic.validate_interval(
            loss_l, loss_u, msg="loss interval"
        )

        # perform the reduction over any remaining dimensions
        if self.reduction == "sum":
            return torch.sum(loss_l), torch.sum(loss_u)
        if self.reduction == "mean":
            return torch.mean(loss_l), torch.mean(loss_u)
        return loss_l.reshape(input_shape), loss_u.reshape(input_shape)

    def bound_backward(
        self,
        inputs_l: torch.Tensor,
        inputs_u: torch.Tensor,
        targets: torch.Tensor,
        *,
        label_k_poison: int = 0,
        label_epsilon: float = 0.0,
        poison_target_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bounds on the gradients of the loss wrt the inputs.

        If a poisoning attack is specified, we also compute bounds wrt the
        poisoning attack with the following parameters:

            - label_k_poison: Max number of data-points with poisoned targets.

        Args:
            inputs_l: Lower bounds on the inputs to the loss function.
            inputs_u: Upper bounds on the inputs to the loss function.
            targets: Target values for the loss function.
            label_k_poison: Maximum number of data-points with poisoned targets.
            label_epsilon: Not supported for this loss.
            poison_target_idx: Not supported for this loss.

        Returns:
            tuple: Lower and upper bounds on the gradients of the loss.
        """
        # validate input
        if label_epsilon > 0 or poison_target_idx >= 0:
            raise ValueError(
                "Unsupported or irrelevant options for the hinge loss."
            )
        # the hinge loss expects targets to be -1 or 1, but in AGT we'll allow
        # 0 or 1 and transform them appropriately
        if ((targets == 0) | (targets == 1)).all():
            targets = 2 * targets - 1
        if not ((targets == 1) | (targets == -1)).all():
            raise ValueError("Labels must be binary for hinge loss.")

        # handle shapes and types
        input_shape = inputs_l.shape  # save the input shape
        targets = targets.squeeze() if targets.dim() > 1 else targets
        targets = targets.type(inputs_l.dtype)
        try:
            inputs_l, inputs_u = (
                inputs_l.reshape(targets.shape),
                inputs_u.reshape(targets.shape),
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{inputs_l.shape=} does not match {targets.shape=} for BCE."
            ) from e

        # compute the min and max of the hinge loss gradient
        if label_k_poison == 0:
            grad_l = 1 * (targets == -1) * (-inputs_l < self.margin) - 1 * (
                targets == 1
            ) * (inputs_l < self.margin)
            grad_u = 1 * (targets == -1) * (-inputs_u < self.margin) - 1 * (
                targets == 1
            ) * (inputs_u < self.margin)
        else:
            grad_l = -1 * (inputs_l < self.margin)  # assume all targets are 1
            grad_u = 1 * (-inputs_u < self.margin)  # assume all targets are -1

        grad_l = grad_l.reshape(input_shape).type(inputs_l.dtype)
        grad_u = grad_u.reshape(input_shape).type(inputs_l.dtype)

        interval_arithmetic.validate_interval(
            grad_l, grad_u, msg="loss gradient interval"
        )

        if self.reduction == "mean":
            LOGGER.warning(
                "AGT processes grads in fragments, so using 'reduction=mean' "
                "is not recommended."
            )
            LOGGER.warning(
                "Specify 'reduction=none' and normalise manually to ensure "
                "correct scaling."
            )
            return grad_l / grad_l.nelement(), grad_u / grad_u.nelement()
        return grad_l, grad_u


if __name__ == "__main__":
    # test that the loss function matches the standard Hinge
    m = 2.0
    batchsize = 100

    def nominal_loss(x, y):
        """Hinge loss function."""
        return (m - y * x).relu().sum()

    bounded_loss = BoundedHingeLoss(margin=m, reduction="sum")

    # validate forward passes
    inp = (
        torch.linspace(-2 * m, 2 * m, batchsize, requires_grad=True)
        .reshape(-1, 1)
        .repeat(2, 1)
    )
    target = torch.cat(
        [torch.zeros(batchsize, 1), torch.ones(batchsize, 1)], dim=0
    )
    nominal_loss_val = nominal_loss(inp, 2 * target.float() - 1)
    loss_val = bounded_loss.forward(inp, target)
    loss_val_l, loss_val_u = bounded_loss.bound_forward(inp, inp, target)
    assert torch.allclose(nominal_loss_val, loss_val), (
        f"Expected {nominal_loss_val}, got {loss_val}"
    )
    assert torch.allclose(nominal_loss_val, loss_val_l)
    assert torch.allclose(nominal_loss_val, loss_val_u)
    loss_val_l, loss_val_u = bounded_loss.bound_forward(
        inp - 0.01, inp + 0.05, target
    )
    assert torch.all(nominal_loss_val >= loss_val_l)
    assert torch.all(nominal_loss_val <= loss_val_u)

    # validate backward passes
    nominal_loss_grad = torch.autograd.grad(
        loss_val.sum(), inp, create_graph=True
    )[0]
    loss_grad = bounded_loss.backward(inp, target)
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(inp, inp, target)
    assert torch.allclose(nominal_loss_grad, loss_grad), (
        f"{nominal_loss_grad.flatten()} vs {loss_grad.flatten()}"
    )
    assert torch.allclose(nominal_loss_grad, loss_grad_l), (
        f"{nominal_loss_grad.flatten()} vs {loss_grad_l.flatten()}"
    )
    assert torch.allclose(nominal_loss_grad, loss_grad_u)
    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(
        inp - 0.01, inp + 0.05, target
    )
    assert torch.all(nominal_loss_grad >= loss_grad_l)
    assert torch.all(nominal_loss_grad <= loss_grad_u)

    loss_grad_l, loss_grad_u = bounded_loss.bound_backward(
        inp - 0.01, inp + 0.05, target, label_k_poison=1
    )
    assert torch.all(nominal_loss_grad >= loss_grad_l), (
        f"{nominal_loss_grad.flatten()} vs {loss_grad_l.flatten()}"
    )
    assert torch.all(nominal_loss_grad <= loss_grad_u), (
        f"{nominal_loss_grad.flatten()} vs {loss_grad_u.flatten()}"
    )
