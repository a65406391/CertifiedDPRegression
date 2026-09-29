"""
DP baseline implementations for comparison against AGT.

Consolidates the three copies (lin/ca/syn) of each baseline:
- Functional Mechanism (suffix-statistics perturbation)
- Laplace DP-SGD (pure DP, per-sample clipping + Laplace noise)
- Gaussian DP-SGD (approx DP via Opacus)
- Sharding (data partitioning + averaging)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine, GradSampleModule

# imports for the DP-SGD arms and PATE moved here from common/ags.py and common/pate.py
import copy
import math
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.ags import _infinite_loader
import os
from typing import Callable, Optional
import tqdm
from scipy.stats import cauchy
from torch.utils.data import DataLoader, TensorDataset
from common.data import make_dataloader, partition_dataset
from common.sensitivity import compute_sensitivity_full_pipeline
from common.training import train_robust_ibp


# =============================================================================
# Functional Mechanism (Linear Regression)
# =============================================================================


def run_functional_mechanism(X_train, y_train, X_test, y_test, epsilon, clip_x, clip_y, seed=42):
    """Functional Mechanism baseline for linear regression.

    Perturbs sufficient statistics (X^T X, X^T y) with Laplace noise.
    Based on "Functional Mechanism: Regression Analysis under DP" (Zhang et al. 2012).

    Args:
        X_train, y_train: Training data (numpy arrays).
        X_test, y_test: Test data (numpy arrays).
        epsilon: Privacy budget.
        clip_x: Feature bound (||x||_inf <= clip_x).
        clip_y: Target bound (|y| <= clip_y).
        seed: Random seed.

    Returns:
        Test MSE.
    """
    np.random.seed(seed)

    # Ensure y is 1D for matrix ops
    y_train_flat = y_train.ravel() if y_train.ndim > 1 else y_train
    y_test_flat = y_test.ravel() if y_test.ndim > 1 else y_test

    # Add bias term
    X_b = np.c_[X_train, np.ones(len(X_train))]
    d = X_b.shape[1]

    XTX = X_b.T @ X_b
    XTY = X_b.T @ y_train_flat

    # Sensitivity computation
    input_dim = X_train.shape[1]
    R2 = input_dim * clip_x**2 + 1.0  # ||x_augmented||^2 upper bound
    eps_1 = epsilon / 2.0
    eps_2 = epsilon / 2.0

    # Noise for XTX
    s_xtx = R2
    noise_xtx = np.random.laplace(0, s_xtx / eps_1, (d, d))
    noise_xtx = np.triu(noise_xtx) + np.triu(noise_xtx, 1).T  # Symmetrize
    XTX_p = XTX + noise_xtx

    # Noise for XTY
    s_xty = np.sqrt(R2) * clip_y
    noise_xty = np.random.laplace(0, s_xty / eps_2, (d, 1))
    XTY_p = XTY.reshape(-1, 1) + noise_xty

    try:
        theta = np.linalg.solve(XTX_p, XTY_p)
    except np.linalg.LinAlgError:
        theta = np.zeros((d, 1))

    # Evaluate
    X_test_b = np.c_[X_test, np.ones(len(X_test))]
    y_pred = X_test_b @ theta
    mse = np.mean((y_pred.ravel() - y_test_flat) ** 2)
    return mse


# =============================================================================
# Laplace DP-SGD (Pure DP)
# =============================================================================


def run_laplace_sgd(model_fn, train_loader, X_test_t, y_test_t, epsilon, epochs, lr, clip_C=1.0, seed=42):
    """DP-SGD with Laplace noise (pure DP) using Opacus GradSampleModule.

    Per-sample gradient clipping + Laplace noise addition.

    Args:
        model_fn: Callable returning nn.Module.
        train_loader: DataLoader for training.
        X_test_t: Test features (tensor).
        y_test_t: Test targets (tensor).
        epsilon: Privacy budget.
        epochs: Number of training epochs.
        lr: Learning rate.
        clip_C: Gradient clipping norm.
        seed: Random seed.

    Returns:
        Test MSE.
    """
    torch.manual_seed(seed)

    model = model_fn()
    model = GradSampleModule(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    total_steps = len(train_loader) * epochs

    if epsilon > 100:
        noise_scale = 0
    else:
        eps_step = epsilon / total_steps
        noise_scale = 2 * clip_C / eps_step

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            with torch.no_grad():
                params = list(model.parameters())
                batch_size = X_batch.shape[0]
                per_sample_norms_sq = torch.zeros(batch_size)

                for p in params:
                    if hasattr(p, "grad_sample") and p.grad_sample is not None:
                        if isinstance(p.grad_sample, list):
                            p_grad_sample = torch.cat(p.grad_sample, dim=0)
                        else:
                            p_grad_sample = p.grad_sample
                        flat_g = p_grad_sample.reshape(batch_size, -1)
                        per_sample_norms_sq += flat_g.norm(2, dim=1) ** 2

                per_sample_norms = torch.sqrt(per_sample_norms_sq)
                clip_factors = torch.clamp(clip_C / (per_sample_norms + 1e-6), max=1.0)

                for p in params:
                    if hasattr(p, "grad_sample") and p.grad_sample is not None:
                        if isinstance(p.grad_sample, list):
                            p_grad_sample = torch.cat(p.grad_sample, dim=0)
                        else:
                            p_grad_sample = p.grad_sample

                        view_shape = [batch_size] + [1] * (p_grad_sample.ndim - 1)
                        clipped_grads = p_grad_sample * clip_factors.view(*view_shape)
                        grad_sum = clipped_grads.sum(dim=0)

                        if noise_scale > 0:
                            noise = np.random.laplace(0, noise_scale, p.shape)
                            grad_sum += torch.tensor(noise, dtype=torch.float32)

                        p.grad = grad_sum / batch_size
                        p.grad_sample = None

            optimizer.step()

    # Eval
    with torch.no_grad():
        preds = model(X_test_t)
        mse = nn.MSELoss()(preds, y_test_t).item()
    return mse


# =============================================================================
# Gaussian DP-SGD (Approx DP via Opacus)
# =============================================================================


def run_gaussian_sgd(
    model_fn, train_loader, X_test_t, y_test_t, epsilon, epochs, lr, clip_C=1.0, delta=1e-5, seed=42
):
    """DP-SGD with Gaussian noise (approximate DP) using Opacus PrivacyEngine.

    Args:
        model_fn: Callable returning nn.Module.
        train_loader: DataLoader for training.
        X_test_t: Test features (tensor).
        y_test_t: Test targets (tensor).
        epsilon: Target privacy budget.
        epochs: Number of training epochs.
        lr: Learning rate.
        clip_C: Max gradient norm.
        delta: Privacy delta parameter.
        seed: Random seed.

    Returns:
        Test MSE.
    """
    torch.manual_seed(seed)

    model = model_fn()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()

    try:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=clip_C,
        )
    except AttributeError:
        # Fallback for older Opacus
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=clip_C,
        )

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test_t)
        mse = nn.MSELoss()(preds, y_test_t).item()
    return mse


# =============================================================================
# Sharding Baseline
# =============================================================================


def run_sharding(
    model_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    epsilon,
    n_shards=10,
    epochs=50,
    lr=0.05,
    batch_size=5,
    clip_y=6.0,
    seed=42,
):
    """Sharding baseline for DP regression.

    1. Split dataset into S shards
    2. Train S separate models (non-private)
    3. Average predictions + Laplace noise

    Extracted from run_lin_baselines.py:run_sharding().

    Args:
        model_fn: Callable returning nn.Module.
        X_train, y_train: Training data (numpy).
        X_test, y_test: Test data (numpy).
        epsilon: Privacy budget.
        n_shards: Number of shards.
        epochs: Training epochs per shard.
        lr: Learning rate.
        batch_size: Batch size for shard training.
        clip_y: Target bound for sensitivity calculation.
        seed: Random seed.

    Returns:
        Test MSE.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_train)
    shard_size = n // n_shards

    indices = np.random.permutation(n)
    shard_models = []

    for s in range(n_shards):
        start_idx = s * shard_size
        end_idx = start_idx + shard_size if s < n_shards - 1 else n
        shard_indices = indices[start_idx:end_idx]

        X_shard = X_train[shard_indices]
        y_shard = y_train[shard_indices]

        model = model_fn()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_shard, dtype=torch.float32)
        y_t = torch.tensor(y_shard, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)

        shard_batch_size = max(1, batch_size // n_shards)
        loader = DataLoader(dataset, batch_size=shard_batch_size, shuffle=True)

        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        shard_models.append(model)

    # Evaluation: average predictions + Laplace noise
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    GS = 2 * clip_y
    noise_scale = (GS / n_shards) / epsilon

    with torch.no_grad():
        all_preds = [model(X_test_t).numpy() for model in shard_models]
        avg_preds = np.mean(all_preds, axis=0)
        noise = np.random.laplace(0, noise_scale, avg_preds.shape)
        noisy_preds = avg_preds + noise

    y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()
    mse = np.mean((noisy_preds - y_test_np) ** 2)
    return mse


# =============================================================================
# DP-SGD baselines -- moved verbatim from common/ags.py (train_laplace_dpsgd,
# train_gaussian_dpsgd). common/ags.py now holds AGS only.
# =============================================================================


def train_laplace_dpsgd(
    model: nn.Sequential,
    train_loader: DataLoader,
    *,
    n_total_steps: int,
    epsilon_per_step: float,
    lr: float,
    clip_gamma: float,
    loss_fn: nn.Module | None = None,
    device: str = "cpu",
    seed: int = 42,
    eval_callback: Optional[callable] = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(model).to(device)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_iter = _infinite_loader(train_loader)

    first_batch = next(iter(train_loader))[0]
    batchsize = first_batch.size(0)

    # Count total parameters for L1 sensitivity correction
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # With per-element clipping to C, each coordinate's sensitivity = 2C/B.
    # The Laplace mechanism requires L1 sensitivity of the full vector:
    #   Delta_1 = d * 2C / B
    # where d = total number of parameters.
    # noise_scale = Delta_1 / epsilon  (Laplace mechanism: Lap(Delta_1 / eps))
    per_element_sensitivity = 2.0 * clip_gamma / batchsize
    l1_sensitivity = per_element_sensitivity * n_params
    noise_scale = l1_sensitivity / epsilon_per_step if epsilon_per_step > 0 else 0.0

    model.train()

    from torch.func import functional_call, grad, vmap

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss_stateless(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), x.unsqueeze(0))
        loss = loss_fn(logits, y.unsqueeze(0))
        return loss

    ft_compute_grad = grad(compute_loss_stateless)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    for step in tqdm(range(n_total_steps), desc="Laplace DP-SGD Steps"):
        batch, labels = next(data_iter)
        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()

        # Calculate per-example gradients: dict of (B, ...) tensors
        per_sample_grads = ft_compute_sample_grad(params, buffers, batch, labels)

        # Per-element gradient clipping + Laplace noise
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Per-example gradients for this parameter: shape (B, ...)
                per_sample = per_sample_grads[name]

                # Per-element clip each example's gradient
                per_sample.clamp_(-clip_gamma, clip_gamma)

                # Average per-element over batch
                avg_grad = per_sample.mean(dim=0)

                # Add Laplace noise
                if noise_scale > 0:
                    noise = torch.distributions.Laplace(torch.zeros_like(avg_grad), noise_scale).sample()
                    avg_grad += noise

                param.grad = avg_grad

        optimizer.step()

        if eval_callback is not None:
            eval_callback(model, step + 1)

    model.eval()
    info = {
        "n_total_steps": n_total_steps,
        "epsilon_per_step": epsilon_per_step,
        "epsilon_total": epsilon_per_step * n_total_steps,
        "noise_scale": noise_scale,
        "l1_sensitivity": l1_sensitivity,
        "n_params": n_params,
    }
    return model, info


def train_gaussian_dpsgd(
    model: nn.Sequential,
    train_loader: DataLoader,
    *,
    n_total_steps: int,
    epsilon_per_step: float,
    delta: float,
    lr: float,
    clip_gamma: float,
    loss_fn: nn.Module | None = None,
    device: str = "cpu",
    seed: int = 42,
    eval_callback: Optional[callable] = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(model).to(device)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_iter = _infinite_loader(train_loader)

    first_batch = next(iter(train_loader))[0]
    batchsize = first_batch.size(0)

    # Count total parameters for L2 sensitivity correction
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # With per-element clipping to C, each coordinate's sensitivity = 2C/B.
    # But the Gaussian mechanism requires L2 sensitivity of the full vector:
    #   Delta_2 = sqrt(d) * 2C / B
    # where d = total number of parameters.
    # sigma >= (Delta_2 / epsilon) * sqrt(2 * ln(1.25 / delta))
    per_element_sensitivity = 2.0 * clip_gamma / batchsize
    l2_sensitivity = per_element_sensitivity * math.sqrt(n_params)
    if epsilon_per_step > 0:
        factor = math.sqrt(2.0 * math.log(1.25 / delta))
        noise_scale = (l2_sensitivity / epsilon_per_step) * factor
    else:
        noise_scale = 0.0

    model.train()

    from torch.func import functional_call, grad, vmap

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss_stateless(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), x.unsqueeze(0))
        loss = loss_fn(logits, y.unsqueeze(0))
        return loss

    ft_compute_grad = grad(compute_loss_stateless)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    for step in tqdm(range(n_total_steps), desc="Gaussian DP-SGD Steps"):
        batch, labels = next(data_iter)
        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()

        # Calculate per-example gradients: dict of (B, ...) tensors
        per_sample_grads = ft_compute_sample_grad(params, buffers, batch, labels)

        # Per-element gradient clipping + Gaussian noise
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Per-example gradients for this parameter: shape (B, ...)
                per_sample = per_sample_grads[name]

                # Per-element clip each example's gradient to [-C, C].
                # With element-wise clipping, the L2 sensitivity of the full
                # gradient vector is sqrt(d) * 2C/B, which is accounted for
                # in the noise_scale computation above.
                per_sample.clamp_(-clip_gamma, clip_gamma)

                # Average per-element over batch
                avg_grad = per_sample.mean(dim=0)

                # Add Gaussian noise
                if noise_scale > 0:
                    noise = torch.distributions.Normal(torch.zeros_like(avg_grad), noise_scale).sample()
                    avg_grad += noise

                param.grad = avg_grad

        optimizer.step()

        if eval_callback is not None:
            eval_callback(model, step + 1)

    model.eval()
    info = {
        "n_total_steps": n_total_steps,
        "epsilon_per_step": epsilon_per_step,
        "epsilon_total": epsilon_per_step * n_total_steps,
        "noise_scale": noise_scale,
        "l2_sensitivity": l2_sensitivity,
        "n_params": n_params,
        "delta": delta,
    }
    return model, info


# =============================================================================
# PATE for regression -- moved verbatim from common/pate.py
# =============================================================================


class PateRegressor:
    """PATE regressor with mean aggregation over T teachers."""

    def __init__(
        self,
        ensemble_size: int,
        global_sensitivity: float,
    ):
        self.ensemble_size = ensemble_size
        self.global_sensitivity = global_sensitivity

    # =========================================================================
    # Training
    # =========================================================================

    def _train_teacher_model(
        self,
        model: nn.Module,
        shard_dataloader: DataLoader,
        model_idx: int,
        epochs: int,
        lr: float,
        save_path: str,
        loss_fn: nn.Module = None,
        optimizer_cls: type = torch.optim.Adam,
    ) -> nn.Module:
        """Train a single (non-private) teacher on one data shard."""
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        optimizer = optimizer_cls(model.parameters(), lr=lr)
        progress_bar = tqdm.tqdm(range(epochs), desc=f"Training teacher {model_idx}", leave=True)

        for epoch in progress_bar:
            train_loss = 0.0
            num_batches = 0
            model.train()

            for x_batch, y_batch in shard_dataloader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            progress_bar.set_postfix({"avg_train_loss": avg_train_loss})

        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/teacher_{model_idx}.pt")
        return model

    def train_ensemble(
        self,
        train_dataset: TensorDataset,
        model_fn: Callable[[], nn.Module],
        epochs: int,
        lr: float,
        save_path: str,
        with_smooth_sensitivity: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        loss_fn: Optional[nn.Module] = None,
        optimizer_cls: type = torch.optim.Adam,
        k_values: list[int] = None,
        clip_C: float = 1.0,
        huber_delta: float = 1.0,
        accumulation_freq: int = 1,
    ) -> None:
        """Partition data into T shards and train all teachers (standard or AGT)."""
        os.makedirs(save_path, exist_ok=True)

        if with_smooth_sensitivity:
            assert k_values is not None, "k_values must be provided for smooth sensitivity"

        shards = partition_dataset(train_dataset, self.ensemble_size, seed=seed)

        for i, shard in enumerate(shards):
            if with_smooth_sensitivity:
                # AGT branch: train_robust_ibp expects a TensorDataset
                print(f"Training teacher {i} with AGT (k_values={k_values}).")
                for k_val in k_values:
                    bounded_model = train_robust_ibp(
                        train_dataset=shard,
                        model_or_fn=model_fn,
                        k_val=k_val,
                        batch_size=batch_size,
                        epochs=epochs,
                        lr=lr,
                        seed=seed + i,
                        clip_C=clip_C,
                        huber_delta=huber_delta,
                        accumulation_freq=accumulation_freq,
                    )
                    # Save the full IntervalBoundedModel (carries weight intervals + architecture)
                    torch.save(bounded_model, f"{save_path}/teacher_{i}_k{k_val}.pt")
            else:
                # Standard branch: plain nn.Module saved as state_dict
                print(f"Training teacher {i} (standard).")
                shard_loader = make_dataloader(
                    shard,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=seed + i,
                )
                model = model_fn()
                self._train_teacher_model(
                    model,
                    shard_loader,
                    model_idx=i,
                    epochs=epochs,
                    lr=lr,
                    save_path=save_path,
                    loss_fn=loss_fn,
                    optimizer_cls=optimizer_cls,
                )

    # =========================================================================
    # Loading
    # =========================================================================

    @staticmethod
    def load_teacher(
        models_dir: str,
        teacher_idx: int,
        model_fn: Optional[Callable[[], nn.Module]] = None,
        k: Optional[int] = None,
    ):
        """Load teacher_idx from disk: state_dict if k=None, full IntervalBoundedModel if k is set."""
        if k is not None:
            path = f"{models_dir}/teacher_{teacher_idx}_k{k}.pt"
            ibm = torch.load(path, weights_only=False)
            return ibm
        else:
            assert model_fn is not None, "model_fn is required to load a global-sensitivity teacher"
            path = f"{models_dir}/teacher_{teacher_idx}.pt"
            model = model_fn()
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            return model

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _collect_teacher_predictions(
        self,
        test_loader: DataLoader,
        models_dir: str,
        model_fn: Optional[Callable[[], nn.Module]] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """Load all T teachers and return their predictions as an (T, N_test) array."""
        all_preds = []
        for shard_idx in range(self.ensemble_size):
            teacher = self.load_teacher(models_dir, shard_idx, model_fn=model_fn, k=k)

            preds = []
            with torch.no_grad():
                for x_batch, _ in test_loader:
                    y_pred = teacher.forward(x_batch)
                    preds.append(y_pred.cpu().numpy().squeeze())

            all_preds.append(np.concatenate(preds, axis=0))

        return np.stack(all_preds, axis=0)  # (T, N_test)

    def _mean_aggregation(
        self,
        test_loader: DataLoader,
        models_dir: str,
        model_fn: Optional[Callable[[], nn.Module]] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """Non-private mean aggregation: g_r(x) = (1/T) sum_i f^(i)(x)."""
        teacher_preds = self._collect_teacher_predictions(
            test_loader,
            models_dir,
            model_fn=model_fn,
            k=k,
        )
        return np.mean(teacher_preds, axis=0)  # (N_test,)

    # =========================================================================
    # Private prediction — Global Sensitivity + Laplace
    # =========================================================================

    def private_predict_global(
        self,
        test_loader: DataLoader,
        models_dir: str,
        epsilon: float,
        model_fn: Callable[[], nn.Module],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """(eps,0)-DP prediction: g_r(x) + Lap(Delta_g_r / (T * epsilon))."""
        ensemble_mean = self._mean_aggregation(
            test_loader,
            models_dir,
            model_fn=model_fn,
        )

        # Laplace scale = Delta_g_r / (T * epsilon)
        gamma = self.global_sensitivity / (self.ensemble_size * epsilon)

        if seed is not None:
            rng = np.random.default_rng(seed)
            noise = rng.laplace(0, gamma, size=ensemble_mean.shape)
        else:
            noise = np.random.laplace(0, gamma, size=ensemble_mean.shape)

        return ensemble_mean + noise

    # =========================================================================
    # Private prediction — Smooth Sensitivity + Cauchy
    # =========================================================================

    def _compute_teacher_ss(
        self,
        test_loader: DataLoader,
        models_dir: str,
        model_idx: int,
        k_values: list[int],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute smooth sensitivity for a single teacher across all test points."""
        k_to_ibm = {}
        for k in k_values:
            k_to_ibm[k] = PateRegressor.load_teacher(models_dir, model_idx, k=k)

        X_eval_t = test_loader.dataset.tensors[0]
        vk_mat, ss_vals, avg_ss, avg_vk = compute_sensitivity_full_pipeline(
            k_to_ibm,
            X_eval_t,
            k_values=k_values,
            gs_val=self.global_sensitivity,
        )

        return ss_vals, avg_ss, avg_vk

    def private_predict_smooth(
        self,
        test_loader: DataLoader,
        models_dir: str,
        epsilon: float,
        k_values: list[int],
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """(eps,0)-DP prediction: g_r(x) + Cauchy(6 * SS^beta_PATE / epsilon)."""
        per_teacher_ss = []
        for i in range(self.ensemble_size):
            ss_vals, _, _ = self._compute_teacher_ss(test_loader, models_dir, i, k_values)
            per_teacher_ss.append(ss_vals)

        # SS^beta_PATE = (1/T) * max_i SS^beta(f_i, x)
        smooth_sensitivities_per_teacher = np.stack(per_teacher_ss, axis=0)
        per_point_max_ss = np.max(smooth_sensitivities_per_teacher, axis=0)
        ss_pate = per_point_max_ss / self.ensemble_size

        # Mean aggregation using the k=0 (nominal) IntervalBoundedModel
        ensemble_mean = self._mean_aggregation(test_loader, models_dir, k=0)

        # Cauchy noise: scale = 6 * SS^beta_PATE / epsilon
        scale = 6 * ss_pate / epsilon
        noise = cauchy.rvs(loc=0, scale=scale, random_state=seed)

        return ensemble_mean + noise

    # =========================================================================
    # End-to-end evaluation
    # =========================================================================

    def evaluate(
        self,
        test_loader: DataLoader,
        models_dir: str,
        mode: str = "nonprivate",
        model_fn: Optional[Callable[[], nn.Module]] = None,
        k: Optional[int] = None,
        seed: Optional[int] = 42,
        n_trials: int = 1,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """Run private prediction and return {mse_mean, mse_std, predictions}."""
        assert isinstance(test_loader.dataset, torch.utils.data.TensorDataset), (
            "Dataset must be a TensorDataset"
        )
        if mode != "nonprivate":
            assert epsilon is not None, "Epsilon must be provided for non-nonprivate modes."

        y_test = test_loader.dataset.tensors[1].numpy().ravel()

        if mode == "nonprivate":
            preds = self._mean_aggregation(
                test_loader,
                models_dir,
                model_fn=model_fn,
                k=k,
            )
            mse = np.mean((preds - y_test) ** 2)
            return {"mse_mean": mse, "mse_std": 0.0, "predictions": preds}

        mse_trials = []
        last_preds = None
        for t in range(n_trials):
            trial_seed = (seed + t) if seed is not None else None
            if mode == "global":
                preds = self.private_predict_global(
                    test_loader,
                    models_dir,
                    epsilon=epsilon,
                    model_fn=model_fn,
                    seed=trial_seed,
                )
            elif mode == "smooth":
                preds = self.private_predict_smooth(
                    test_loader,
                    models_dir,
                    epsilon=epsilon,
                    seed=trial_seed,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
            last_preds = preds
            mse_trials.append(np.mean((preds - y_test) ** 2))

        return {
            "mse_mean": np.mean(mse_trials),
            "mse_std": np.std(mse_trials),
            "predictions": last_preds,
        }
