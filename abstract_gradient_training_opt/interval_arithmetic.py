"""Numpy wrappers of abstract_gradient_training.interval_arithmetic."""

from collections.abc import Callable

import torch

from abstract_gradient_training import interval_arithmetic


def torch_function_wrapper(func) -> Callable:
    """Wrapper that converts function arguments from numpy to torch.

    Wrapper that takes a function that accepts torch tensors as arguments
    and returns a function that accepts numpy arrays as arguments, while also
    converting the output back to numpy arrays.
    """

    def wrapped_fn(*args, **kwargs):
        args = [torch.tensor(a) for a in args]
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(r.detach().numpy() for r in result)
        if isinstance(result, torch.Tensor):
            return result.detach().numpy()
        return result

    return wrapped_fn


propagate_matmul = torch_function_wrapper(interval_arithmetic.propagate_matmul)
propagate_affine = torch_function_wrapper(interval_arithmetic.propagate_affine)
propagate_elementwise = torch_function_wrapper(
    interval_arithmetic.propagate_elementwise
)
propagate_softmax = torch_function_wrapper(
    interval_arithmetic.propagate_softmax
)
propagate_square = torch_function_wrapper(interval_arithmetic.propagate_square)
