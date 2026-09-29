"""Node representing an affine transform in the LBP computation graph."""

from __future__ import annotations

from typing import Literal

import torch

from abstract_gradient_training.bounded_models._crown_bounds import (
    IntervalBounds,
)
from abstract_gradient_training.bounded_models._crown_bounds import LinearBounds
from abstract_gradient_training.bounded_models._crown_bounds import Node


class AffineNode(Node):
    """A node in the CROWN computation graph consisting of an affine transform.

    The node represents the operation:
        out_var = W @ in_var + b
    where W and b are intervals. Unlike the more general CROWN nodes, the lower
    and upper linear bounds are the same here, allowing for a more efficient
    backpropagation:
            out_var = W @ in_var + b <= Lambda @ in_var + delta <= upper
            out_var = W @ in_var + b >= Omega @ in_var + theta >= lower
    """

    def __init__(
        self,
        in_var: Node,
        w_l: torch.Tensor,
        w_u: torch.Tensor,
        b_l: torch.Tensor | None = None,
        b_u: torch.Tensor | None = None,
        interval_matmul: str = "rump",
    ):
        """Initialize the affine node.

        Args:
            in_var: [batchsize x n] or [n] Input node to the affine transform.
            w_l: [m x n] Lower bound on the weight matrix.
            w_u: [m x n] Upper bound on the weight matrix.
            b_l: [m] Lower bound on the bias vector. Defaults to None.
            b_u: [m] Upper bound on the bias vector. Defaults to None.
            interval_matmul: The method to use for interval matrix mul.
        """
        super().__init__()
        assert w_l.dim() == 2, "Expected W to be a 2D tensor"
        assert w_u.shape == w_l.shape, (
            "Expected w_l and w_u to have the same shape"
        )

        self.W = IntervalBounds(w_l, w_u, interval_matmul=interval_matmul)

        if b_l is not None:
            assert b_u is not None, "Got b_l, but not b_u"
            assert b_l.dim() == 1, "Expected b to be a 1D tensor"
            assert b_u.shape == b_l.shape, (
                "Expected b_l and b_u to have the same shape"
            )
        else:
            device = w_l.get_device()
            device = (
                torch.device(device) if device != -1 else torch.device("cpu")
            )
            b_l = b_u = torch.zeros(
                w_l.shape[-2], dtype=w_l.dtype, device=device
            )
        self.b = IntervalBounds(b_l, b_u)
        # add batch dimension to the node
        self.in_var = in_var
        self.interval_matmul = interval_matmul

    def backpropagate(self, backward_bounds: LinearBounds) -> None:
        """Extend the linear bounds passed in to include this node.

        Say the computational graph represented has the following structure:
            x0 -> x1 -> ... -> xn-1 -> xn -> xn+1 -> ... -> xN
        where
            - the current node (self) represents bounds
                L @ xn + d <= xn+1 <= O @ xn + t
            - the argument bounds (backward bounds) represents bounds
                L @ xn-1 + d <= xN <= O @ xn-1 + t
        then return linear bounds from xn-1 to xN of the form
            L @ xn-1 + d <= xN <= O @ xn-1 + t
        """
        Lambda, Omega, delta, theta = backward_bounds
        conc = self.concretize()
        assert conc.dim() == 2, "Expected shape of [batchsize x n]"

        # backpropagate the upper bound
        mixed_mask = (Lambda.lb <= 0) & (Lambda.ub >= 0)
        backward_bounds.delta = (
            delta
            + (Lambda * self.b * ~mixed_mask).sum(-1)
            + ((Lambda * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Lambda = (Lambda * ~mixed_mask) @ self.W.unsqueeze(0)

        # backpropagate the lower bound
        mixed_mask = (Omega.lb < 0) & (Omega.ub > 0)
        backward_bounds.theta = (
            theta
            + (Omega * self.b * ~mixed_mask).sum(-1)
            + ((Omega * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Omega = (Omega * ~mixed_mask) @ self.W.unsqueeze(0)

    def _init_backpropagation(self) -> LinearBounds:
        """Initialise the coefficients for backpropagating from this node."""
        # shape should be [batchsize x m x n]
        Lambda0 = self.W.clone().unsqueeze(0)
        # shape should be [batchsize x m x n]
        Omega0 = self.W.clone().unsqueeze(0)
        delta0 = self.b.clone().unsqueeze(0)  # shape should be [batchsize x m]
        theta0 = self.b.clone().unsqueeze(0)  # shape should be [batchsize x m]
        return LinearBounds(Lambda0, Omega0, delta0, theta0)


if __name__ == "__main__":
    from abstract_gradient_training.bounded_models._crown_bounds import (
        InputNode,
    )

    x = torch.randn(5, 10)
    W1 = torch.randn(4, 10)
    b1 = torch.randn(4)
    W2 = torch.randn(3, 4)
    b2 = torch.randn(3)
    print((x @ W1.T + b1) @ W2.T + b2)

    x = AffineNode(InputNode(x, x), W1, W1, b1, b1)
    x = AffineNode(x, W2, W2, b2, b2)
    l, u = x.concretize()
    assert torch.allclose(l, u)
    print(l.squeeze(-1))
