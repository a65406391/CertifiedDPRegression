"""Tests for interval arithmetic helper functions."""

import torch
import pytest
from abstract_gradient_training.interval_arithmetic import propagate_softmax


@pytest.mark.parametrize(
    "ndim, dim",
    [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (5, 1),
        (3, 2),
        (6, 4),
    ],
)
def test_propagate_softmax(ndim: int, dim: int) -> None:
    """Test the propagate_softmax function."""
    torch.manual_seed(0)
    shape = torch.Size(torch.randint(1, 10, (ndim,)))  # type: ignore
    A = torch.randn(*shape, dtype=torch.float64)
    A_l = A - torch.rand(*shape, dtype=torch.float64)
    A_u = A + torch.rand(*shape, dtype=torch.float64)
    y = torch.softmax(A, dim=dim)
    y_l, y_u = propagate_softmax(A_l, A_u, dim=dim)
    assert y_l.shape == shape and y_u.shape == shape and y_u.shape == y.shape
    assert torch.all(y_l >= 0) and torch.all(y_u >= 0)
    assert torch.all(y_l <= 1) and torch.all(y_u <= 1)
    assert torch.all(y_l <= y_u) and torch.all(y_l <= y) and torch.all(y_u >= y)

    # Test that if the lower and upper bounds are the same, the output is the same
    A = torch.randn(*shape)
    y_l, y_u = propagate_softmax(A, A, dim=dim)
    y = torch.softmax(A, dim=dim)
    assert torch.allclose(y_l, y)
    assert torch.allclose(y_u, y)

    # Test worst case logits for some target class
    target_class = 0
    A = A_l.clone()
    index = [slice(None)] * A.dim()  # Create a full slice for all dimensions
    index[dim] = target_class  # type: ignore
    A[tuple(index)] = A_u[tuple(index)]
    assert torch.all(A_l <= A) and torch.all(A <= A_u)

    y_l, y_u = propagate_softmax(A_l, A_u, dim=dim)
    y = torch.softmax(A, dim=dim)
    assert y_l.shape == shape and y_u.shape == shape
    assert torch.all(y_l <= y + 1e-12), f"{y_l - y}"
    assert torch.all(y_u + 1e-12 >= y), f"{y_u - y}"
    assert torch.allclose(y_u[tuple(index)], y[tuple(index)])
