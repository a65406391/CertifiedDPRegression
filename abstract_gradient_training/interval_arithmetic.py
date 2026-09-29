"""Interval arithmetic helper functions."""

import inspect
import logging
from collections.abc import Callable

import torch

LOGGER = logging.getLogger(__name__)


def propagate_affine(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    w_l: torch.Tensor,
    w_u: torch.Tensor,
    b_l: torch.Tensor | None = None,
    b_u: torch.Tensor | None = None,
    interval_matmul: str = "rump",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the affine transformation A @ x + b.

    Args:
        x_l: Lower bound of the input tensor x.
        x_u: Upper bound of the input tensor x.
        w_l: Lower bound of the weight matrix A.
        w_u: Upper bound of the weight matrix A.
        b_l: Lower bound of the bias vector b.
        b_u: Upper bound of the bias vector b.
        interval_matmul: Method for interval matmul, "rump", "exact", "nguyen".

    Returns:
        e_l: Lower bound of the output tensor.
        e_u: Upper bound of the output tensor.
    """
    h_l, h_u = propagate_matmul(w_l, w_u, x_l, x_u, interval_matmul)
    if b_l is not None or b_u is not None:
        assert b_u is not None, (
            "Upper bound of bias tensor must be provided "
            "if lower bound is provided"
        )
        assert b_l is not None, (
            "Lower bound of bias tensor must be provided "
            "if upper bound is provided"
        )
        validate_interval(b_l, b_u)
        h_l, h_u = h_l + b_l, h_u + b_u
    return h_l, h_u


def propagate_matmul(
    a_l: torch.Tensor,
    a_u: torch.Tensor,
    b_l: torch.Tensor,
    b_u: torch.Tensor,
    interval_matmul: str = "rump",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the matrix multiplication A @ B.

    Args:
        a_l: Lower bound of the matrix A.
        a_u: Upper bound of the matrix A.
        b_l: Lower bound of the matrix B.
        b_u: Upper bound of the matrix B.
        interval_matmul: Method to use for interval matmul, one of
            ["rump", "exact", "nguyen"].

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    validate_interval(a_l, a_u, msg="input interval A")
    validate_interval(b_l, b_u, msg="input interval B")
    if interval_matmul == "rump":
        h_l, h_u = propagate_matmul_rump(a_l, a_u, b_l, b_u)
    elif interval_matmul == "exact":
        h_l, h_u = propagate_matmul_exact(a_l, a_u, b_l, b_u)
    elif interval_matmul == "nguyen":
        h_l, h_u = propagate_matmul_nguyen(a_l, a_u, b_l, b_u)
    else:
        raise ValueError(f"Unknown interval matmul method: {interval_matmul}")
    validate_interval(h_l, h_u, msg="output interval")
    return h_l, h_u


def propagate_matmul_rump(
    a_l: torch.Tensor, a_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval on the matrix multiplication A @ B using Rump.

    Args:
        a_l: Lower bound of the matrix A.
        a_u: Upper bound of the matrix A.
        b_l: Lower bound of the matrix B.
        b_u: Upper bound of the matrix B.

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    a_mu = (a_u + a_l) / 2
    a_r = (a_u - a_l) / 2
    b_mu = (b_u + b_l) / 2
    b_r = (b_u - b_l) / 2

    h_mu = a_mu @ b_mu
    h_r = torch.abs(a_mu) @ b_r + a_r @ torch.abs(b_mu) + a_r @ b_r
    h_l = h_mu - h_r
    h_u = h_mu + h_r
    return h_l, h_u


def propagate_matmul_nguyen(
    a_l: torch.Tensor, a_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval on matrix multiplication using Nguyen's algorithm.

    The algorithm performs
    a decomposition of A into two intervals A^0 and A^* such that
    - A^0 is a zero-centered interval
    - A^* is an interval that does not contain zero

    Args:
        a_l: Lower bound of the matrix A.
        a_u: Upper bound of the matrix A.
        b_l: Lower bound of the matrix B.
        b_u: Upper bound of the matrix B.

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    # A0 is an interval centered at zero
    a0_l, a0_u = torch.zeros_like(a_l), torch.zeros_like(a_l)
    # astar is an interval not containing zero.
    # we further split astar into its negative and positive parts
    astar_neg_l, astar_neg_u = torch.zeros_like(a_l), torch.zeros_like(a_l)
    astar_pos_l, astar_pos_u = torch.zeros_like(a_l), torch.zeros_like(a_l)

    # case where a_l * a_u >= 0
    condition = (a_l >= 0) & (a_u >= 0)
    astar_pos_l = torch.where(condition, a_l, astar_pos_l)
    astar_pos_u = torch.where(condition, a_u, astar_pos_u)
    condition = (a_l <= 0) & (a_u <= 0)
    astar_neg_l = torch.where(condition, a_l, astar_neg_l)
    astar_neg_u = torch.where(condition, a_u, astar_neg_u)

    # case where a_l < 0 < |a_l| <= a_u
    condition = (a_l < 0) & (torch.abs(a_l) <= a_u)
    a0_l = torch.where(condition, a_l, a0_l)
    a0_u = torch.where(condition, -a_l, a0_u)
    astar_pos_u = torch.where(condition, a_l + a_u, astar_pos_u)

    # case where a_l < 0 < a_u < |a_l|
    condition = (a_l < 0) & (0 < a_u) & (a_u < torch.abs(a_l))
    a0_l = torch.where(condition, -a_u, a0_l)
    a0_u = torch.where(condition, a_u, a0_u)
    astar_neg_l = torch.where(condition, a_l + a_u, astar_neg_l)

    # these asserts are only valid when the input bounds aren't inf or nan
    if a_l.isfinite().all() and a_u.isfinite().all():
        assert torch.allclose(a_l, a0_l + astar_pos_l + astar_neg_l)
        assert torch.allclose(a_u, a0_u + astar_pos_u + astar_neg_u)
        assert torch.allclose(a0_l + a0_u, torch.zeros_like(a0_l))
        assert torch.all((astar_neg_l <= 0) & (astar_neg_u <= 0))
        assert torch.all((astar_pos_l >= 0) & (astar_pos_u >= 0))

    # compute the interval A0 @ B
    h_u = a0_u @ torch.maximum(torch.abs(b_l), torch.abs(b_u))
    h_l = -h_u

    # split the matrix B into its negative and positive parts
    b_neg_l, b_neg_u = torch.clamp(b_l, max=0), torch.clamp(b_u, max=0)
    b_pos_l, b_pos_u = torch.clamp(b_l, min=0), torch.clamp(b_u, min=0)

    # compute the interval astar_pos @ B
    h_l += astar_pos_l @ b_pos_l + astar_pos_u @ b_neg_l
    h_u += astar_pos_u @ b_pos_u + astar_pos_l @ b_neg_u

    # compute the interval astar_neg @ B
    h_l += astar_neg_l @ b_pos_u + astar_neg_u @ b_neg_u
    h_u += astar_neg_u @ b_pos_l + astar_neg_l @ b_neg_l

    return h_l, h_u


def propagate_matmul_exact(
    a_l: torch.Tensor, a_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval on a matrix multiplication using exact arithmetic.

    Args:
        a_l: Lower bound of the matrix A.
        a_u: Upper bound of the matrix A.
        b_l: Lower bound of the matrix B.
        b_u: Upper bound of the matrix B.

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    if a_l.dim() == b_l.dim() + 1:  # the case of [... x m x n] @ [... x n]
        assert a_l.size(-1) == b_l.size(-1), (
            f"Expected [...x m x n] @ [...x n], got {a_l.shape} @ {b_l.shape}"
        )
        assert a_l.shape[:-2] == b_l.shape[:-1], (
            f"Expected [...x m x n] @ [...x n], got {a_l.shape} @ {b_l.shape}"
        )
        b_l, b_u = b_l.unsqueeze(-2), b_u.unsqueeze(-2)  # shape [... x 1 x n]
        #  shape [... x m x n]
        e_l, e_u = propagate_elementwise(a_l, a_u, b_l, b_u)
        e_l, e_u = e_l.sum(-1), e_u.sum(-1)  # shape [... x m]
    elif a_l.dim() == b_l.dim():  # the case of [... x m x n] @ [... x n x p]
        assert a_l.size(-1) == b_l.size(-2), (
            "Expected [...x m x n] @ [...x n x 1], "
            f"got {a_l.shape} @ {b_l.shape}"
        )
        assert a_l.shape[:-2] == b_l.shape[:-2], (
            "Expected [...x m x n] @ [...x n x 1], "
            f"got {a_l.shape} @ {b_l.shape}"
        )
        # shape [... x m x n x 1]
        a_l, a_u = a_l.unsqueeze(-1), a_u.unsqueeze(-1)
        # shape [... x 1 x n x p]
        b_l, b_u = b_l.unsqueeze(-3), b_u.unsqueeze(-3)
        # shape [ ... x m x n x p]
        e_l, e_u = propagate_elementwise(a_l, a_u, b_l, b_u)
        e_l, e_u = e_l.sum(-2), e_u.sum(-2)  # shape [... x m x p]
    elif a_l.dim() + 1 == b_l.dim():  # the case of [... x m] * [... x m x n]
        assert a_l.size(-1) == b_l.size(-2), (
            f"Expected [...x m] @ [...x m x n], got {a_l.shape} @ {b_l.shape}"
        )
        a_l, a_u = a_l.unsqueeze(-1), a_u.unsqueeze(-1)  # shape [... x m x 1]
        # shape [... x m x n]
        e_l, e_u = propagate_elementwise(a_l, a_u, b_l, b_u)
        e_l, e_u = e_l.sum(-2), e_u.sum(-2)  # shape [... x n]
    else:
        raise ValueError(
            f"Invalid shapes for matmul: {a_l.shape} * {b_l.shape}"
        )
    return e_l, e_u


def propagate_elementwise(
    a_l: torch.Tensor, a_u: torch.Tensor, b_l: torch.Tensor, b_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the matrix multiplication A * B.

    Args:
        a_l: Lower bound of the matrix A.
        a_u: Upper bound of the matrix A.
        b_l: Lower bound of the matrix B.
        b_u: Upper bound of the matrix B.

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    validate_interval(a_l, a_u)
    validate_interval(b_l, b_u)
    cases = torch.stack([a_l * b_l, a_u * b_l, a_l * b_u, a_u * b_u])
    lb = torch.min(cases, dim=0)[0]
    ub = torch.max(cases, dim=0)[0]
    validate_interval(lb, ub)
    return lb, ub


def propagate_norm(
    x_l: torch.Tensor, x_u: torch.Tensor, p: float = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval on the Lp norm of the input interval [x_l, x_u].

    If dim x > 1, then the first dimension is treated as a batch dimension and
    the norm is computed over the remaining dimensions flattened into a single
    dimension.

    Args:
        x_l: Lower bound of the input tensor x.
        x_u: Upper bound of the input tensor x.
        p (float, optional): Order of the norm. Defaults to 2.

    Returns:
        e_l: Lower bound of the output tensor.
        e_u: Upper bound of the output tensor.
    """
    validate_interval(x_l, x_u)
    if x_l.dim() > 2:
        x_l, x_u = x_l.flatten(start_dim=1), x_u.flatten(start_dim=1)

    if p == 1:
        # compute interval over abs(x)
        x_abs_l = torch.minimum(x_l.abs(), x_u.abs()) * ((x_l > 0) | (x_u < 0))
        x_abs_u = torch.maximum(x_l.abs(), x_u.abs())
        validate_interval(x_abs_l, x_abs_u)
        norm_l = x_abs_l.sum(-1)
        norm_u = x_abs_u.sum(-1)
    elif p == 2:
        # compute interval over x^2, then sum and take the sqrt
        x_square_l = torch.minimum(x_l.abs(), x_u.abs()).square() * (
            (x_l > 0) | (x_u < 0)
        )
        x_square_u = torch.maximum(x_l.abs(), x_u.abs()).square()
        validate_interval(x_square_l, x_square_u)
        norm_l = x_square_l.sum(-1).sqrt()
        norm_u = x_square_u.sum(-1).sqrt()
    else:
        raise ValueError(f"Norm of order {p} not suppported.")
    validate_interval(norm_l, norm_u)
    return norm_l, norm_u


def propagate_square(
    x_l: torch.Tensor, x_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the square of the interval [x_l, x_u].

    Args:
        x_l: Lower bound of the input tensor x.
        x_u: Upper bound of the input tensor x.

    Returns:
        h_l: Lower bound of the output tensor.
        h_u: Upper bound of the output tensor.
    """
    validate_interval(x_l, x_u)
    # calculate the bounds on the squared result with the following cases:
    # 1. if the input interval spans zero:
    #   - the lower bound is zero, the upper bound is max(x_l**2, x_u**2)
    # 2. if the input interval is positive:
    #   - the lower bound is x_l ** 2, the upper bound is x_u ** 2
    # 3. if the input interval is negative:
    #   the lower bound is the x_u ** 2, the upper bound is x_l ** 2
    x_square_l = torch.minimum(x_l.abs(), x_u.abs()).square() * (
        (x_l > 0) | (x_u < 0)
    )
    x_square_u = torch.maximum(x_l.abs(), x_u.abs()).square()
    validate_interval(x_square_l, x_square_u)
    return x_square_l, x_square_u


def propagate_conv2d(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    w_l: torch.Tensor,
    w_u: torch.Tensor,
    b_l: torch.Tensor | None = None,
    b_u: torch.Tensor | None = None,
    transpose: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate the interval over x_l and x_u through the convolutional layer.

    Args:
        x_l: Lower bound of the input tensor x.
        x_u: Upper bound of the input tensor x.
        w_l: Lower bound of the weight tensor of the convolutional layer.
        w_u: Upper bound of the weight tensor of the convolutional layer.
        b_l: Lower bound of the bias tensor of the convolutional layer.
        b_u: Upper bound of the bias tensor of the convolutional layer.
        transpose (bool, optional): Whether the convolution is a transposed
            convolution. Defaults to False.
        **kwargs: Additional arguments to pass to the convolutional layer.

    Returns:
        e_l: Lower bound of the output tensor.
        e_u: Upper bound of the output tensor.
    """
    # validate shapes and bounds
    validate_interval(x_l, x_u, msg="input interval")
    validate_interval(w_l, w_u, msg="weight interval")
    assert x_l.dim() == 4
    assert w_l.dim() == 4
    if b_l is not None:
        assert b_u is not None, (
            "Upper bound of bias tensor must be provided "
            "if lower bound is provided"
        )
        # require the bias to be a vector, even though for affine layers we
        # require it to be 2d
        assert b_l.dim() == 1

    # get the appropriate conv function to use
    def transform(x_, w_):
        if transpose:
            return torch.nn.functional.conv_transpose2d(
                x_, w_, bias=None, **kwargs
            )
        return torch.nn.functional.conv2d(x_, w_, bias=None, **kwargs)

    # apply the linear transform
    h_l, h_u = propagate_linear_transform(x_l, x_u, w_l, w_u, transform)

    # add the bias
    if b_l is not None and b_u is not None:
        validate_interval(b_l, b_u, msg="bias interval")
        h_l, h_u = h_l + b_l.view(1, -1, 1, 1), h_u + b_u.view(1, -1, 1, 1)
        validate_interval(h_l, h_u, msg="output interval with bias")

    return h_l, h_u


def propagate_linear_transform(
    a_l: torch.Tensor,
    a_u: torch.Tensor,
    b_l: torch.Tensor,
    b_u: torch.Tensor,
    transform: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the output of a linear transformation.

    Given any linear transformation f (i.e. f(A + B) = f(A) + f(B)), compute the
    interval bound on the output of the transformation given an interval over
    the input using Rump's algorithm.

    Args:
        a_l: Lower bound on the first input tensor A.
        a_u: Upper bound on the first input tensor A.
        b_l: Lower bound on the second input tensor B.
        b_u: Upper bound on the second input tensor B.
        transform: The linear transformation to apply to the input interval.

    Returns:
        tuple: Interval over the output of the transformation.
    """
    # validate input
    validate_interval(a_l, a_u, msg="input interval A")
    validate_interval(b_l, b_u, msg="input interval B")

    # compute the "mean" and "radius" of the input intervals
    a_mu = (a_u + a_l) / 2
    a_r = (a_u - a_l) / 2
    b_mu = (b_u + b_l) / 2
    b_r = (b_u - b_l) / 2

    # compute the "mean" and "radius" of the output
    h_mu = transform(a_mu, b_mu)
    h_r = (
        transform(torch.abs(a_mu), b_r)
        + transform(a_r, torch.abs(b_mu))
        + transform(a_r, b_r)
    )

    # convert to lower and upper bounds
    h_l = h_mu - h_r
    h_u = h_mu + h_r
    validate_interval(h_l, h_u, msg="output interval")
    return h_l, h_u


def propagate_softmax(
    a_l: torch.Tensor, a_u: torch.Tensor, dim=-1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the softmax function.

    Args:
        a_l: Lower bound of the input logits.
        a_u: Upper bound of the input logits.
        dim: Dimension along which to compute the softmax.

    Returns:
        y_l: Lower bound of the output probabilities.
        y_u: Upper bound of the output probabilities.
    """
    validate_interval(a_l, a_u, msg="input interval")
    # handle dimension
    if dim < 0:
        dim += a_l.ndim
    # get the tensor of logit differences
    a_diff_l = a_l.unsqueeze(dim) - a_u.unsqueeze(dim + 1)
    a_diff_u = a_u.unsqueeze(dim) - a_l.unsqueeze(dim + 1)
    # set diagonals to 0
    a_diff_l.diagonal(dim1=dim, dim2=dim + 1).zero_()
    a_diff_u.diagonal(dim1=dim, dim2=dim + 1).zero_()
    # compute softmax output
    y_l = 1 / torch.exp(a_diff_u).sum(dim=dim + 1)
    y_u = 1 / torch.exp(a_diff_l).sum(dim=dim + 1)
    validate_interval(y_l, y_u, msg="final interval")
    return y_l, y_u


def propagate_binary_cross_entropy_with_logits(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an interval bound on the binary cross entropy loss.

    Compute an interval bound on the binary cross entropy loss given by
    BCE(x, y) = -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))

    Args:
        x_l: Lower bound of the input logits.
        x_u: Upper bound of the input logits.
        y: Target labels, expected to be binary (0 or 1).

    Returns:
        loss_l: Lower bound of the loss.
        loss_u: Upper bound of the loss.
    """
    validate_interval(x_l, x_u, msg="input interval")
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("Labels must be binary for binary cross entropy.")
    if y.shape != x_l.shape:
        raise ValueError(
            f"Labels shape {y.shape} must match logits shape {x_l.shape}."
        )

    # compute the best and worst case sigmoid outputs
    p_worst = torch.sigmoid((1 - y) * x_u + y * x_l)
    p_best = torch.sigmoid(y * x_u + (1 - y) * x_l)
    # lowest loss computed with best-case logits
    t = 1e-10  # tolerance to avoid log(0)
    loss_l = -(1 - y) * torch.log(1 - p_best + t) - y * torch.log(p_best + t)
    # highest loss computed with worst-case logits
    loss_u = -(1 - y) * torch.log(1 - p_worst + t) - y * torch.log(p_worst + t)
    validate_interval(loss_l, loss_u, msg="loss interval")
    return loss_l, loss_u


def validate_interval(
    l: torch.Tensor | list[torch.Tensor],
    u: torch.Tensor | list[torch.Tensor],
    n: torch.Tensor | list[torch.Tensor] | None = None,
    msg: str = "",
) -> None:
    """Validate an arbitrary interval n in [l, u] and log any violations.

    The function either takes in tensors or lists of tensors, in which case it
    will validate each pair of bounds in the list.

    Args:
        l: Lower bound of the interval(s).
        u: Upper bound of the interval(s).
        n: Nominal value of the interval(s). Defaults to None.
        msg: Optional message to log with the bound violation for debugging.
    """
    if isinstance(l, list) and n is not None:
        for i, (li, ui, ni) in enumerate(zip(l, u, n)):
            validate_interval(li, ui, ni, msg=f"{msg} (element {i})")
        return

    if isinstance(l, list):
        for i, (li, ui) in enumerate(zip(l, u)):
            validate_interval(li, ui, msg=f"{msg} (element {i})")
        return

    assert isinstance(l, torch.Tensor) and isinstance(u, torch.Tensor)
    if n is None:
        diff = torch.max(l - u).item()
    else:
        diff = max(
            torch.max(l - u).item(),
            torch.max(l - n).item(),
            torch.max(n - u).item(),
        )
    # this should be negative if all bounds are satisfied
    if diff <= 0:
        return
    # get caller function name, if available
    current_frame = inspect.currentframe()
    f_back = current_frame.f_back if current_frame is not None else None
    func_name = f_back.f_code.co_name if f_back is not None else ""
    if diff > 1e-3:  # a major infraction of the bound
        LOGGER.error("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-4:  # a minor infraction of the bound
        LOGGER.warning("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-5:  # a minor infraction of the bound
        LOGGER.info("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
    elif diff > 1e-6:  # a tiny infraction of the bound
        LOGGER.debug("Violated bound in %s: %.2e (%s)", func_name, diff, msg)
