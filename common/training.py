"""
AGT robust training loops for all benchmarks.

Consolidates the training loop from:
- run_ca_ablation.py:train_ablation_models()
- run_syn_ablation.py:train_model_k()
- run_syn_context_ablation.py:train_ablation_models()
- run_lin_dim_ablation.py:train_agt_model()

These all implement the same pattern:
  forward -> backward -> per-sample clip -> accumulate -> bounded step
with minor parameter variations.
"""

import warnings
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np

from abstract_gradient_training.bounded_losses import BoundedMSELoss
from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD
from abstract_gradient_training.bounded_models import IntervalBoundedModel


def train_robust_ibp(
    train_dataset,
    model_or_fn,
    k_val,
    batch_size,
    epochs,
    lr,
    seed=42,
    clip_C=1.0,
    huber_delta=1.0,
    accumulation_freq=1,
    reduction="sum",
    fixed_init_state=None,
    shuffle=True,
    drop_last=True,
    lr_nominal=None,
):
    """Train a model using AGT interval bound propagation with per-sample gradient clipping.

    This is the core training loop shared by California, Synthea, and dimension
    ablation experiments. It handles both k=0 (nominal) and k>0 (robust) cases.

    Args:
        train_dataset: TensorDataset of (X, y).
        model_or_fn: Either an nn.Module or a callable returning one.
        k_val: Privacy parameter. k=0 means standard nominal training.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate for robust training (k>0).
        seed: Random seed for deterministic shuffling.
        clip_C: Per-sample gradient clipping norm.
        huber_delta: Gradient clamp value (Huber-like loss stabilization).
        accumulation_freq: Number of batches to accumulate before stepping.
        reduction: Loss reduction mode ('sum' or 'mean').
        fixed_init_state: Pre-computed state dict for deterministic init.
        shuffle: Whether to shuffle the dataloader.
        drop_last: Whether to drop the last incomplete batch.
        lr_nominal: Learning rate for nominal training (k=0). If None, uses lr.

    Returns:
        IntervalBoundedModel with trained parameters and bounds.
    """
    warnings.warn(
        "train_robust_ibp bounds the update by the closed form g -/+ k*2C/b, which is a valid\n"
        "parameter-space bound only for non-expansive updates (convex loss, lr <= 2/L, inactive\n"
        "clipping). It is NOT an AGT envelope for ReLU networks; use train_certified instead.",
        stacklevel=2,
    )
    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create deterministic dataloader
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        drop_last=drop_last,
    )

    # Create model
    if callable(model_or_fn) and not isinstance(model_or_fn, nn.Module):
        model = model_or_fn()
    else:
        model = model_or_fn

    # Apply fixed initialization if provided
    if fixed_init_state is not None:
        model.load_state_dict(fixed_init_state)

    # Wrap in bounded model
    bounded_model = IntervalBoundedModel(model)

    # === k=0: Standard Nominal Training ===
    if k_val == 0:
        effective_lr = lr_nominal if lr_nominal is not None else lr
        optimizer = torch.optim.SGD(model.parameters(), lr=effective_lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for x_b, y_b in train_loader:
                optimizer.zero_grad()
                out = model(x_b).squeeze()
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

        return IntervalBoundedModel(model)

    # === k>0: Robust Training with Clipping ===
    optimizer = BoundedSGD(bounded_model, learning_rate=lr)
    criterion = BoundedMSELoss(reduction=reduction)

    for epoch in range(epochs):
        accum_grads_n = None
        total_accum_bs = 0

        for i, (x_batch, y_batch) in enumerate(train_loader):
            curr_bs = x_batch.size(0)

            # Forward
            logits_n = bounded_model.forward(x_batch, retain_intermediate=True)
            logits_input = logits_n.squeeze()
            loss_grad_n = criterion.backward(logits_input, y_batch)

            # Huber-like gradient clamping
            loss_grad_n = torch.clamp(loss_grad_n, -huber_delta, huber_delta)
            loss_grad_n = loss_grad_n.unsqueeze(1)

            grads_n = bounded_model.backward(loss_grad_n)

            # Per-sample gradient clipping
            total_norm = torch.zeros(curr_bs, device=x_batch.device)
            for g in grads_n:
                g_flat = g.view(curr_bs, -1)
                total_norm += g_flat.norm(2, dim=1) ** 2
            total_norm = torch.sqrt(total_norm)
            clip_coef = torch.clamp(clip_C / (total_norm + 1e-6), max=1.0)

            grads_clipped_sum = []
            for g in grads_n:
                view_shape = [curr_bs] + [1] * (g.dim() - 1)
                g_clipped = g * clip_coef.view(*view_shape)
                grads_clipped_sum.append(g_clipped.sum(dim=0))

            if accum_grads_n is None:
                accum_grads_n = grads_clipped_sum
            else:
                accum_grads_n = [a + g for a, g in zip(accum_grads_n, grads_clipped_sum)]

            total_accum_bs += curr_bs

            # Step condition: every `accumulation_freq` batches or at end
            if (i + 1) % accumulation_freq == 0 or (i + 1) == len(train_loader):
                grads_n_mean = [g / total_accum_bs for g in accum_grads_n]

                delta = k_val * 2 * clip_C / total_accum_bs
                grads_l = [g - delta for g in grads_n_mean]
                grads_u = [g + delta for g in grads_n_mean]
                optimizer.step(grads_l, grads_n_mean, grads_u)

                accum_grads_n = None
                total_accum_bs = 0

    return bounded_model


def train_nominal_clipped(
    train_dataset,
    model_or_fn,
    batch_size,
    epochs,
    lr,
    clip_gamma,
    seed=42,
    fixed_init_state=None,
    shuffle=False,
    dtype=torch.float64,
):
    """Train a nominal model with per-element gradient clamping.

    Used by run_lin_ablation.py for the k=0 case with clipping.
    This is distinct from standard SGD because it clamps individual
    gradient elements to [-gamma, gamma].

    Returns:
        IntervalBoundedModel wrapping the trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if callable(model_or_fn) and not isinstance(model_or_fn, nn.Module):
        model = model_or_fn()
    else:
        model = model_or_fn

    if fixed_init_state is not None:
        model.load_state_dict(fixed_init_state)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param.grad.clamp_(-clip_gamma, clip_gamma)
            optimizer.step()

    return IntervalBoundedModel(model)


# =============================================================================
# Genuine AGT (interval bound propagation) training -- the sound replacement for
# train_robust_ibp. Thin wrapper over abstract_gradient_training.privacy_certified_training.
# =============================================================================


def train_certified(
    train_dataset,
    model_or_fn,
    k_val,
    batch_size,
    epochs,
    lr,
    *,
    clip_gamma,
    clip_method="clamp",
    loss="mse",
    seed=42,
    fixed_init_state=None,
    shuffle=True,
    dtype=torch.float64,
    device="cpu",
    val_dataset=None,
    **agt_config_kwargs,
):
    """Train with genuine AGT and return the IntervalBoundedModel (param_l, param_n, param_u).

    Every gradient is bounded by interval propagation through the model and the loss, and the k
    substituted records enter only through +/- k*clip_gamma (per element for ``clamp``). The
    nominal trajectory equals plain SGD with per-sample clipping under the same seed and order.

    Args:
        train_dataset: TensorDataset of (X, y).
        model_or_fn: nn.Sequential or a callable returning one.
        k_val: substitution radius k (0 gives nominal-only bounds).
        batch_size, epochs, lr: fixed batch size (incomplete batches are dropped by AGT),
            number of epochs, learning rate.
        clip_gamma: per-sample clipping threshold (required; the only thing bounding a
            substituted record).
        clip_method: "clamp" (per element, default) or "norm" (per tensor l2; see Notes on the
            broadcast defect for models with biases).
        loss: "mse" | "binary_cross_entropy" | "cross_entropy" | "hinge".
        seed: seeds torch (initialisation and shuffling).
        fixed_init_state: optional state_dict loaded before training.
        dtype: float64 recommended for sound intervals.
        agt_config_kwargs: forwarded to AGTConfig (e.g. lr_decay, fragsize, l2_reg).

    Returns:
        abstract_gradient_training.bounded_models.IntervalBoundedModel after training.
    """
    import abstract_gradient_training as agt

    torch.manual_seed(seed)
    model = model_or_fn() if callable(model_or_fn) and not isinstance(model_or_fn, nn.Module) else model_or_fn
    if fixed_init_state is not None:
        model.load_state_dict(fixed_init_state)
    model = model.to(dtype=dtype, device=device)
    X, y = train_dataset.tensors
    ds = TensorDataset(X.to(dtype=dtype, device=device), y.to(dtype=dtype, device=device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=g)
    dl_val = None
    if val_dataset is not None:
        Xv, yv = val_dataset.tensors
        dl_val = DataLoader(
            TensorDataset(Xv.to(dtype=dtype, device=device), yv.to(dtype=dtype, device=device)),
            batch_size=len(Xv),
        )
    config = agt.AGTConfig(
        n_epochs=epochs,
        learning_rate=lr,
        loss=loss,
        k_private=int(k_val),
        clip_gamma=float(clip_gamma),
        clip_method=clip_method,
        device=device,
        **agt_config_kwargs,
    )
    bounded = agt.bounded_models.IntervalBoundedModel(model)
    agt.privacy_certified_training(bounded, config, dl, dl_val)
    return bounded
