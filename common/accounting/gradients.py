"""Per-example gradients and the two clipping rules.

`per_example_gradients` uses `torch.func` vmap, chunked to bound memory. `coordinate_clip` bounds
every coordinate independently, `l2_clip` the vector norm; they give different certificates, and the
choice must match whatever the certificate assumes about a substituted record's contribution.

Both return the fraction of examples that were actually clipped. An inactive clip is what makes the
closed-form sensitivity sound, so that number decides whether a run's guarantee is conditional.
"""

from __future__ import annotations

from collections import OrderedDict
import time

import torch
from torch import nn
from torch.func import functional_call
from torch.func import grad
from torch.func import vmap


def per_example_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 16,
    output_device: torch.device | str | None = None,
    progress_label: str | None = None,
) -> torch.Tensor:
    """Return an ``[batch,p]`` matrix in named-parameter order."""
    params = OrderedDict((name, value.detach()) for name, value in model.named_parameters())
    buffers = OrderedDict((name, value.detach()) for name, value in model.named_buffers())

    def one_loss(current, image, target):
        logits = functional_call(model, (current, buffers), (image.unsqueeze(0),))
        return nn.functional.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

    one_grad = grad(one_loss)
    chunks = []
    n_chunks = (len(inputs) + chunk_size - 1) // chunk_size
    report_every = max(1, n_chunks // 10)
    started = time.perf_counter()
    for start in range(0, len(inputs), chunk_size):
        values = vmap(one_grad, in_dims=(None, 0, 0))(
            params, inputs[start : start + chunk_size], targets[start : start + chunk_size]
        )
        chunk = torch.cat([values[name].reshape(values[name].shape[0], -1) for name in params], dim=1)
        if output_device is not None:
            chunk = chunk.to(output_device)
        chunks.append(chunk)
        completed = len(chunks)
        if progress_label and n_chunks >= 100 and (completed % report_every == 0 or completed == n_chunks):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (n_chunks - completed)
            print(
                f"[{progress_label}:gradients] chunks={completed}/{n_chunks} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return torch.cat(chunks)


def coordinate_clip(grads: torch.Tensor, cap: float | torch.Tensor) -> tuple[torch.Tensor, float]:
    """Clip each column to a public scalar or broadcastable vector cap."""
    caps = torch.as_tensor(cap, dtype=grads.dtype, device=grads.device)
    clipped = torch.maximum(torch.minimum(grads, caps), -caps)
    rate = float((grads.abs() > caps).to(torch.float64).mean())
    return clipped, rate


def l2_clip(grads: torch.Tensor, cap: float) -> tuple[torch.Tensor, float]:
    norms = torch.linalg.vector_norm(grads, dim=1)
    factors = torch.clamp(cap / norms.clamp_min(torch.finfo(grads.dtype).tiny), max=1.0)
    return grads * factors[:, None], float((norms > cap).to(torch.float64).mean())
