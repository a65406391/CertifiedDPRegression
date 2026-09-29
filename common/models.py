"""
Model factory functions for all benchmarks.

Consolidates model creation from:
- run_lin_ablation.py (nn.Sequential(nn.Linear(d, 1)))
- run_ca_ablation.py / run_ca_baselines.py (create_model, create_mlp)
- run_syn_ablation.py (create_model with variable depth)
- run_syn_context_ablation.py (create_model for embeddings)
"""

import os

import torch
import torch.nn as nn


def create_linear_model(input_dim=1, output_dim=1, dtype=torch.float64):
    """Create a simple linear model (no hidden layers).

    Used by: linear ablation, linear baselines.
    """
    return nn.Sequential(nn.Linear(input_dim, output_dim)).to(dtype)


def create_mlp(input_dim, hidden_dims, output_dim=1):
    """Create a multi-layer perceptron with ReLU activations.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer sizes, e.g. [64, 64] or [256, 128, 64].
        output_dim: Output dimension (default 1 for regression).

    Returns:
        nn.Sequential model.

    Used by: California ablation [64,64], California baselines [64,64],
             Synthea baselines [256,128,64], Synthea context ablation [128,64].
    """
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h, bias=True))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim, bias=True))
    return nn.Sequential(*layers)


def create_depth_model(input_dim, depth, output_dim=1):
    """Create a depth-parameterized MLP for Synthea experiments.

    Extracted from run_syn_ablation.py:create_model().

    Depth 1: input -> 256 -> ReLU -> 1
    Depth 2: input -> 256 -> ReLU -> 128 -> ReLU -> 1
    Depth 3: input -> 256 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 1

    Args:
        input_dim: Input feature dimension.
        depth: Number of hidden layers (1, 2, or 3).
        output_dim: Output dimension.
    """
    depth_configs = {
        1: [256],
        2: [256, 128],
        3: [256, 128, 64],
    }
    if depth not in depth_configs:
        raise ValueError(f"Unknown depth: {depth}. Must be 1, 2, or 3.")
    return create_mlp(input_dim, depth_configs[depth], output_dim)


def get_or_create_init_state(model_fn, path="agtr/linear/data/init_state.pth", seed=42):
    """Ensure identical model initialization across all runs.

    Extracted from run_lin_ablation.py:get_or_create_init_state().

    Args:
        model_fn: Callable that returns an nn.Module.
        path: Path to save/load the initialization state.
        seed: Seed for initial model creation.

    Returns:
        State dict for the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        torch.manual_seed(seed)
        model = model_fn()
        torch.save(model.state_dict(), path)
        print(f"Created shared initialization at {path}")
    return torch.load(path)
