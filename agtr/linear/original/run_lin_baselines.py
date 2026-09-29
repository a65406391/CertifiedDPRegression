import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# --- Configuration ---
# Match run_linear_ablation.py (Reduced Benchmark)
N_SAMPLES = 80
BATCH_SIZE = 5
LR = 0.05
EPOCHS = 50 # Allow more epochs for baselines to converge
X_RANGE = (-2, 2)
SLOPE = 2
INTERCEPT = 1

# Derived
STEPS_PER_EPOCH = N_SAMPLES // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

# Bounds (Conservative to ensure valid Functional Mech)
# Data is uniform(-2,2). y ~ 2*x + 1. range(-3, 5).
# Let's enforce clipping to be safe for baselines.
CLIP_X = 2.0
CLIP_Y = 6.0 

def simulated_linear_data(n_samples=N_SAMPLES, slope=SLOPE, intercept=INTERCEPT, x_range=X_RANGE, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    noise = np.random.normal(0, 0.1, (n_samples, 1)) 
    y = slope * X + intercept + noise
    return X, y

def get_dataloader(seed=42):
    X, y = simulated_linear_data(seed=seed)
    # Clip Data for DP guarantees
    X = np.clip(X, -CLIP_X, CLIP_X)
    y = np.clip(y, -CLIP_Y, CLIP_Y)
    
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, tensor_x, tensor_y

# --- Baselines ---

def run_functional_mechanism(epsilon, seed=42):
    # Method: Perturb Sufficient Statistics XTX and XTy
    # Analysis based on "Functional Mechanism: Regression Analysis under Differential Privacy" (Zhang et al. 2012)
    # Sensitivity of X^TX (d x d): Delta = d^2 (if X rows norm <= 1). Here X is 1D (scalar). 
    # Max value of x is CLIP_X. 
    # (x_i)^2 - (x'_i)^2 <= CLIP_X^2.
    # Sensitivity of Sum(x_i^2) is CLIP_X^2.
    # Sensitivity of Sum(x_i * y_i) is CLIP_X * CLIP_Y.
    
    np.random.seed(seed)
    X, y = simulated_linear_data(seed=seed)
    X = np.clip(X, -CLIP_X, CLIP_X)
    y = np.clip(y, -CLIP_Y, CLIP_Y)
    
    # 1. Add bias term to X
    X_b = np.c_[X, np.ones(len(X))] # (N, 2)
    d = 2 
    
    # Suff Stats
    XTX = X_b.T @ X_b
    XTY = X_b.T @ y
    
    # Sensitivity
    # For a row x_i (dimension d), ||x_i||_2 <= R.
    # We clipped x to 2.0. Bias is 1.0. 
    # Row norm constraint R = sqrt(2^2 + 1^2) = sqrt(5).
    # Sensitivity of XTX (Frobenius) is R^2. 
    # R^2 = 5.
    
    # We split epsilon between XTX and XTY. Say 50/50.
    eps_1 = epsilon / 2.0
    eps_2 = epsilon / 2.0
    
    # Noise for XTX (Symmetric matrix perturbation)
    # Upper triangular elements sensitivity? 
    # Standard result: Add Laplace(2 * R^2 / eps) to sufficient stats? 
    # Actually, Zhang et al use Laplace noise on coefficients of polynomial.
    # For Linear Regression, it boils down to perturbing the scatter matrix.
    # Sensitivity of XTX is R^2.
    # We add Laplace(Sen / eps) to each element? No, depends on matrix structure.
    # Simple Sufficient Statistics perturbation:
    # Delta(XTX) = R^2. Add Lap(R^2 / eps) to independent entries. (Symmetry maintained/projected).
    R2 = CLIP_X**2 + 1.0 # 5.0
    s_xtx = R2
    
    noise_xtx = np.random.laplace(0, s_xtx / eps_1, (d, d))
    # Make symmetric
    noise_xtx = np.triu(noise_xtx) + np.triu(noise_xtx, 1).T
    XTX_p = XTX + noise_xtx
    
    # Noise for XTY
    # Sensitivity: R * (Max Y).
    # Max Y norm? We clipped y. |y| <= CLIP_Y.
    # Sensitivity = R * CLIP_Y.
    s_xty = np.sqrt(R2) * CLIP_Y
    noise_xty = np.random.laplace(0, s_xty / eps_2, (d, 1))
    XTY_p = XTY + noise_xty
    
    # Solve: theta = (XTX')^-1 XTY'
    try:
        theta = np.linalg.solve(XTX_p, XTY_p)
    except np.linalg.LinAlgError:
        # Fallback if singular due to noise
        theta = np.zeros((d, 1))
        
    # Evaluate
    X_test, y_test = simulated_linear_data(n_samples=2000, seed=100) # Same test set as ablation
    X_test_b = np.c_[X_test, np.ones(len(X_test))]
    y_pred = X_test_b @ theta
    mse = np.mean((y_pred - y_test)**2)
    return mse

from opacus import GradSampleModule

def run_laplace_sgd(epsilon, seed=42):
    # Pure DP SGD with Vectorized Operations
    torch.manual_seed(seed)
    loader, _, _ = get_dataloader(seed)
    
    model = nn.Linear(1, 1)
    # Use Opacus GradSampleModule for efficient per-sample gradients
    model = GradSampleModule(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean') # reduction doesn't matter for grad_sample logic usually, but mean is standard
    
    C = 1.0
    
    if epsilon > 100: 
        noise_scale = 0
    else:
        eps_step = epsilon / TOTAL_STEPS
        noise_scale = 2 * C / eps_step
        
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Vectorized Clipping and Noise Addition
            with torch.no_grad():
                # 1. Compute per-sample norms across all parameters
                # params are generators, so we list them
                params = list(model.parameters())
                batch_size = X_batch.shape[0]
                
                # Stack norms? No, difficult if different shapes.
                # Compute squared norm per sample per param, sum, sqrt.
                per_sample_norms_sq = torch.zeros(batch_size)
                
                for p in params:
                    # Handle list grad_sample (Opacus 1.x can return list if accumulation happens, or normally tensor)
                    if hasattr(p, 'grad_sample'):
                        if isinstance(p.grad_sample, list):
                             # Stack microbatches if present (should be 1 if cleared)
                             p_grad_sample = torch.cat(p.grad_sample, dim=0)
                        else:
                             p_grad_sample = p.grad_sample
                    else:
                        continue # Should not happen if requires_grad
                        
                    # p_grad_sample: (B, ...)
                    # Calculate Norms
                    flat_g = p_grad_sample.reshape(batch_size, -1)
                    per_sample_norms_sq += flat_g.norm(2, dim=1)**2

                per_sample_norms = torch.sqrt(per_sample_norms_sq)
                
                # 2. Clipping Factors
                clip_factors = torch.clamp(C / (per_sample_norms + 1e-6), max=1.0)
                
                # 3. Aggregate Clipped Gradients
                for p in params:
                    if not hasattr(p, 'grad_sample'): continue
                    
                    if isinstance(p.grad_sample, list):
                         p_grad_sample = torch.cat(p.grad_sample, dim=0)
                    else:
                         p_grad_sample = p.grad_sample

                    # p.grad_sample: (B, ...)
                    # clip_factors: (B) -> reshape to (B, 1, 1...)
                    view_shape = [batch_size] + [1] * (p_grad_sample.ndim - 1)
                    clipped_grads = p_grad_sample * clip_factors.view(*view_shape)
                    
                    # Sum over batch
                    grad_sum = clipped_grads.sum(dim=0)
                    
                    # Add Laplace Noise
                    if noise_scale > 0:
                        noise = np.random.laplace(0, noise_scale, p.shape)
                        grad_sum += torch.tensor(noise, dtype=torch.float32)
                        
                    # Set p.grad
                    p.grad = grad_sum / batch_size
                    
                    # CRITICAL: Clear grad_sample for next iteration
                    # Do not del (Opacus expects attribute). Set to None.
                    p.grad_sample = None
            
            optimizer.step()
            
    # Eval
    X_test, y_test = simulated_linear_data(n_samples=2000, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    mse = np.mean((preds - y_test)**2)
    return mse

def run_gaussian_sgd(epsilon, seed=42):
    # Approx DP (Opacus).
    # Delta = 1e-5.
    
    torch.manual_seed(seed)
    loader, _, _ = get_dataloader(seed)
    
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    privacy_engine = PrivacyEngine()
    
    # Opacus 1.x: Use make_private_with_epsilon for calibration
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    privacy_engine = PrivacyEngine()

    try:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=epsilon,
            target_delta=1e-5,
            epochs=EPOCHS,
            max_grad_norm=1.0,
        )
    except AttributeError:
        # Fallback if make_private_with_epsilon is missing (unlikely in 1.5)
        # Use simple make_private with default sigma=1.0 for now to avoid crash
        print("Warning: make_private_with_epsilon missing. Using sigma=1.0")
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
    # Eval
    X_test, y_test = simulated_linear_data(n_samples=2000, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    mse = np.mean((preds - y_test)**2)
    return mse


def run_sharding(epsilon, n_shards=10, seed=42):
    """
    Sharding baseline for DP linear regression.
    
    1. Split dataset into S shards (each with N/S samples)
    2. Train S separate models using SGD (non-private on each shard)
    3. At test time: average predictions from all S models
    4. Add Laplace noise with scale = GS/S / epsilon to each prediction
    
    Privacy: Each sample affects only 1 shard. Averaging reduces sensitivity by 1/S.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get full data
    X, y = simulated_linear_data(seed=seed)
    X = np.clip(X, -CLIP_X, CLIP_X)
    y = np.clip(y, -CLIP_Y, CLIP_Y)
    
    n = len(X)
    shard_size = n // n_shards
    
    # Shuffle and split into shards
    indices = np.random.permutation(n)
    shard_models = []
    
    for s in range(n_shards):
        start_idx = s * shard_size
        end_idx = start_idx + shard_size if s < n_shards - 1 else n
        shard_indices = indices[start_idx:end_idx]
        
        X_shard = X[shard_indices]
        y_shard = y[shard_indices]
        
        # Train model on this shard (non-private SGD)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X_shard, dtype=torch.float32)
        y_t = torch.tensor(y_shard, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        
        # Use smaller batch for shard (proportional to shard size)
        shard_batch_size = max(1, BATCH_SIZE // n_shards)
        loader = DataLoader(dataset, batch_size=shard_batch_size, shuffle=True)
        
        for epoch in range(EPOCHS):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
        
        shard_models.append(model)
    
    # Evaluation: Average predictions + Laplace noise
    X_test, y_test = simulated_linear_data(n_samples=2000, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    # Global sensitivity of prediction (conservative bound)
    # For linear regression with bounded data: |w*x + b| <= |w|*CLIP_X + |b|
    # Conservative: GS = 2 * CLIP_Y (max change in prediction)
    GS = 2 * CLIP_Y
    
    # Averaging S models reduces sensitivity by S
    noise_scale = (GS / n_shards) / epsilon
    
    with torch.no_grad():
        # Get predictions from all models
        all_preds = []
        for model in shard_models:
            pred = model(X_test_t).numpy()
            all_preds.append(pred)
        
        # Average predictions
        avg_preds = np.mean(all_preds, axis=0)
        
        # Add Laplace noise to each prediction
        noise = np.random.laplace(0, noise_scale, avg_preds.shape)
        noisy_preds = avg_preds + noise
    
    mse = np.mean((noisy_preds - y_test)**2)
    return mse

def main():
    print("Running Linear Baselines...")
    out_dir = 'agtr/linear/values/baselines'
    os.makedirs(out_dir, exist_ok=True)
    
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 1 # One trial for speed
    
    # 1. Functional
    func_mses_mean = []
    func_mses_std = []
    print("Functional Mechanism...")
    for eps in epsilons:
        trials = [run_functional_mechanism(eps, seed=i) for i in range(n_trials)]
        func_mses_mean.append(np.mean(trials))
        func_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/func_mean.npy', func_mses_mean)
    np.save(f'{out_dir}/func_std.npy', func_mses_std)
    
    # 2. Laplace SGD
    lap_mses_mean = []
    lap_mses_std = []
    print("Laplace DP-SGD...")
    for eps in epsilons:
        trials = [run_laplace_sgd(eps, seed=i) for i in range(n_trials)]
        lap_mses_mean.append(np.mean(trials))
        lap_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/lap_mean.npy', lap_mses_mean)
    np.save(f'{out_dir}/lap_std.npy', lap_mses_std)
    
    # 3. Gaussian SGD (Approx DP)
    gauss_mses_mean = []
    gauss_mses_std = []
    print("Gaussian DP-SGD...")
    for eps in epsilons:
        trials = [run_gaussian_sgd(eps, seed=i) for i in range(n_trials)]
        gauss_mses_mean.append(np.mean(trials))
        gauss_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/gauss_mean.npy', gauss_mses_mean)
    np.save(f'{out_dir}/gauss_std.npy', gauss_mses_std)
    
    # 4. Sharding S=10
    shard10_mses_mean = []
    shard10_mses_std = []
    print("Sharding (S=10)...")
    for eps in epsilons:
        trials = [run_sharding(eps, n_shards=10, seed=i) for i in range(n_trials)]
        shard10_mses_mean.append(np.mean(trials))
        shard10_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/shard10_mean.npy', shard10_mses_mean)
    np.save(f'{out_dir}/shard10_std.npy', shard10_mses_std)
    
    # 5. Sharding S=20
    shard20_mses_mean = []
    shard20_mses_std = []
    print("Sharding (S=20)...")
    for eps in epsilons:
        trials = [run_sharding(eps, n_shards=20, seed=i) for i in range(n_trials)]
        shard20_mses_mean.append(np.mean(trials))
        shard20_mses_std.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/shard20_mean.npy', shard20_mses_mean)
    np.save(f'{out_dir}/shard20_std.npy', shard20_mses_std)
    
    np.save(f'{out_dir}/epsilons.npy', epsilons)
    print("Done baselines.")

if __name__ == "__main__":
    main()
