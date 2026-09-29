"""
Shared data generation and loading utilities for all benchmarks.

Consolidates data routines from:
- run_lin_ablation.py (simulated_linear_data, get_or_create_w_true)
- run_lin_baselines.py (simulated_linear_data with slope/intercept)
- run_ca_ablation.py / run_ca_baselines.py (load_california_data_dataset)
- run_syn_ablation.py / run_syn_baselines.py / run_syn_context_ablation.py (load_synthea_data)
"""

import os

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.reproducibility import set_reproducibility

# =============================================================================
# Linear Data
# =============================================================================


def get_or_create_w_true(dim=1, sparsity=1, path="agtr/linear/data/w_true.npy", seed=42):
    """Get or create consistent true weights for linear experiments.

    Extracted from run_lin_ablation.py:get_or_create_w_true().
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        w_true = np.load(path)
        if w_true.shape[0] == dim:
            return w_true

    rng = np.random.RandomState(seed)
    w_true = np.zeros(dim)
    w_true[:sparsity] = rng.randn(sparsity)

    np.save(path, w_true)
    print(f"Saved w_true to {path}: {w_true[:sparsity]}")
    return w_true


def simulated_linear_data(
    n_samples=1000,
    dim=1,
    sparsity=1,
    x_range=(-2, 2),
    seed=42,
    w_true=None,
    slope=None,
    intercept=1.0,
    noise_std=0.1,
    w_true_path="agtr/linear/data/w_true.npy",
):
    """Generate linear regression data.

    Supports two modes:
    - slope/intercept mode: y = slope * X + intercept + noise  (used by run_lin_baselines, run_lin_manual)
    - w_true mode: y = X @ w_true + intercept + noise  (used by run_lin_ablation)

    If `slope` is provided, it takes precedence over `w_true`.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(x_range[0], x_range[1], (n_samples, dim))

    if slope is not None:
        # Simple slope/intercept mode (1D)
        noise = rng.normal(0, noise_std, (n_samples, 1))
        y = slope * X + intercept + noise
    else:
        # w_true mode
        if w_true is None:
            w_true = get_or_create_w_true(dim, sparsity, path=w_true_path, seed=42)
        noise = rng.normal(0, noise_std, (n_samples, 1))
        y = X @ w_true.reshape(-1, 1) + intercept + noise

    return X, y


def simulated_high_dim_data(n_samples, d, seed=42):
    """Generate high-dimensional linear data with scaled weights.

    Extracted from run_lin_dim_ablation.py:simulated_high_dim_data().
    y = X @ w + 1.0 + noise, where w_i = 2/sqrt(d).
    """
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n_samples, d))
    w_val = 2.0 / np.sqrt(d)
    w = np.full((d, 1), w_val)
    noise = np.random.normal(0, 0.1, (n_samples, 1))
    y = X @ w + 1.0 + noise
    return X, y


# =============================================================================
# California Housing Data
# =============================================================================


def load_california_data(seed=42, clip_x=None, clip_y=None):
    """Load and preprocess California Housing dataset.

    Extracted from run_ca_ablation.py:load_california_data_dataset()
    and run_ca_baselines.py:load_california_data_dataset().

    Args:
        seed: Random state for train/test split.
        clip_x: If not None, clip features to [-clip_x, clip_x] (for baselines).
        clip_y: If not None, clip targets to [-clip_y, clip_y] (for baselines).

    Returns:
        (train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np)
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize targets
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std

    # Optional clipping (for baselines that need bounded data)
    if clip_x is not None:
        X_scaled = np.clip(X_scaled, -clip_x, clip_x)
    if clip_y is not None:
        y_scaled = np.clip(y_scaled, -clip_y, clip_y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)

    return train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test


# =============================================================================
# Synthea Data
# =============================================================================


def load_synthea_data(
    context_size_str="10k",
    base_dir="syn/embeddings_consolidated",
    max_train_samples=10000,
    test_size=0.2,
    seed=42,
    clip_x=None,
    clip_y=None,
):
    """Load Synthea embeddings and targets for a specific context size.

    Extracted from run_syn_ablation.py:load_synthea_data()
    and run_syn_baselines.py:load_synthea_data_dataset().

    Args:
        context_size_str: Context size identifier (e.g. '1k', '5k', '10k').
        base_dir: Directory containing embedding/target .npy files.
        max_train_samples: Max training samples (subsampled for speed).
        test_size: Fraction of data reserved for testing.
        seed: Random state for train/test split.
        clip_x: If not None, clip features to [-clip_x, clip_x].
        clip_y: If not None, clip targets to [-clip_y, clip_y].

    Returns:
        (train_dataset, X_test_t, y_test_t, y_mean, y_std)
    """
    emb_file = f"{base_dir}/synthea_embeddings_test_random_{context_size_str}.npy"
    target_file = f"{base_dir}/synthea_targets_test_random_{context_size_str}.npy"

    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"Embedding file not found: {emb_file}")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target file not found: {target_file}")

    print(f"Loading data for Context={context_size_str}...", flush=True)
    X = np.load(emb_file)
    y = np.load(target_file)

    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.flatten()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / (y_std + 1e-8)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )

    # Subsample training data for speed
    if max_train_samples and X_train.shape[0] > max_train_samples:
        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    # Optional clipping (for baselines)
    if clip_x is not None:
        X_train = np.clip(X_train, -clip_x, clip_x)
    if clip_y is not None:
        y_train = np.clip(y_train, -clip_y, clip_y)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)

    return train_dataset, X_test_t, y_test_t, y_mean, y_std


def make_dataloader(dataset, batch_size, shuffle=True, seed=None, drop_last=False):
    """Create a DataLoader with optional deterministic shuffling.

    Common pattern across all experiment scripts.
    """
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    if seed is not None and shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        kwargs["generator"] = g
    return DataLoader(dataset, **kwargs)


def partition_dataset(dataset: TensorDataset, ensemble_size: int, seed: int = 42) -> list[TensorDataset]:
    X, y = dataset.tensors
    n = len(y)
    chunk = (n + ensemble_size - 1) // ensemble_size
    set_reproducibility(seed)

    # Randomly assign samples to disjoint subsets
    # (random permutations then chunking)
    perm = torch.randperm(n)
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    return [TensorDataset(X_shuffled[i : i + chunk], y_shuffled[i : i + chunk]) for i in range(0, n, chunk)]
