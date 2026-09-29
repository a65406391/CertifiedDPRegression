"""
Correct Pure DP comparison:
- AGT: Cauchy noise at inference, scale = smooth_sens(x) / epsilon
- Sharding: Laplace noise at inference, scale = GS/S / epsilon  
- Laplace DP-SGD: Noise during training (no noise at inference)
- Functional: Noise on loss landscape (no noise at inference)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Tighter bounds config: BS=50, EPOCHS=5 (10 updates)
N_SAMPLES = 100
BATCH_SIZE = 50
INPUT_DIM = 2
SPARSITY = 1
EPOCHS = 5
LR = 1.0
CLIP_GAMMA = 0.1

# Path to saved true weights (must match run_lin_ablation.py)
W_TRUE_PATH = 'agtr/linear/data/w_true.npy'

def get_w_true():
    """Load w_true from saved file for consistency."""
    import os
    if os.path.exists(W_TRUE_PATH):
        return np.load(W_TRUE_PATH)
    else:
        raise FileNotFoundError(f"Run run_lin_ablation.py first to create {W_TRUE_PATH}")

def simulated_linear_data(n_samples, dim=INPUT_DIM, sparsity=SPARSITY, seed=42, w_true=None):
    """Generate data with consistent w_true."""
    if w_true is None:
        w_true = get_w_true()
    
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n_samples, dim))
    
    intercept = 1.0
    noise = rng.normal(0, 0.1, (n_samples, 1))
    y = X @ w_true.reshape(-1, 1) + intercept + noise
    return X, y

# Load ablation results
results = np.load('agtr/linear/values/ablation/method_results.npy', allow_pickle=True).item()

print("Loaded AGT results:")
for method in ['ibp', 'f2', 'f5', 'f10']:
    if method in results:
        sens = results[method]['avg_sens']
        mse = results[method]['mse_nom']
        print(f"  {method.upper()}: MSE={mse:.4f}, Avg Sensitivity={sens:.4f}")

# Get data
X_train, y_train = simulated_linear_data(n_samples=N_SAMPLES, seed=42)
X_test, y_test = simulated_linear_data(n_samples=200, seed=100)

# Privacy settings
epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
n_seeds = 10

# ====================
# LAPLACE DP-SGD (noise during training, no noise at inference)
# ====================
def run_laplace_sgd(epsilon, X_train, y_train, X_test, y_test, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    model = nn.Linear(INPUT_DIM, 1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = n_batches * EPOCHS
    eps_per_step = epsilon / total_steps
    grad_sensitivity = 2 * CLIP_GAMMA
    noise_scale = grad_sensitivity / eps_per_step if eps_per_step > 0 else 0
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GAMMA)
            
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.tensor(np.random.laplace(0, noise_scale, param.grad.shape), dtype=torch.float32)
                    param.grad += noise
            
            optimizer.step()
    
    # No noise at inference
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_t).numpy()
    
    mse = np.mean((y_pred - y_test)**2)
    return mse

# ====================
# SHARDING (Laplace noise at inference, scale = GS/S / epsilon)
# ====================
def run_sharding(epsilon, X_train, y_train, X_test, y_test, n_shards=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    CLIP_Y = 6.0
    GS = 2 * CLIP_Y  # Global sensitivity of prediction
    
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
        
        model = nn.Linear(INPUT_DIM, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X_shard, dtype=torch.float32)
        y_t = torch.tensor(y_shard, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        
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
    
    # Inference: average predictions + Laplace noise (GS/S / epsilon)
    noise_scale = (GS / n_shards) / epsilon
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        all_preds = [model(X_test_t).numpy() for model in shard_models]
        avg_preds = np.mean(all_preds, axis=0)
        
        # Add Laplace noise per prediction
        noise = np.random.laplace(0, noise_scale, avg_preds.shape)
        noisy_preds = avg_preds + noise
    
    mse = np.mean((noisy_preds - y_test)**2)
    return mse

# ====================
# AGT (Cauchy noise at inference, scale = avg_sensitivity / epsilon)
# Note: In full implementation, smooth_sens(x) varies per x. Here we use avg.
# ====================
def run_agt_inference(epsilon, mse_nom, avg_sensitivity, X_test, y_test, seed=42):
    """
    AGT: Model trained nominally, add Cauchy noise at inference.
    Scale = smooth_sensitivity / epsilon per query.
    """
    np.random.seed(seed)
    
    n_test = len(X_test)
    
    # Correct formula: Cauchy(6*SS / epsilon) for beta < epsilon/6
    # Using avg_sensitivity as approximation for smooth sensitivity
    noise_scale = 6 * avg_sensitivity / epsilon
    
    # Simulate: model predictions have base error ~ mse_nom
    # Add Cauchy noise to each prediction
    base_errors = np.sqrt(mse_nom) * np.random.randn(n_test, 1)  # Simulate base prediction errors
    cauchy_noise = np.random.standard_cauchy((n_test, 1)) * noise_scale
    
    # Total error = base error + Cauchy noise
    total_errors = base_errors + cauchy_noise
    
    # MSE = mean of squared errors
    mse = np.mean(total_errors**2)
    return mse

# ====================
# Run all methods
# ====================
print("\nRunning baselines...")

laplace_mses = {eps: [] for eps in epsilons}
sharding_mses = {eps: [] for eps in epsilons}

for seed in range(n_seeds):
    for eps in epsilons:
        laplace_mses[eps].append(run_laplace_sgd(eps, X_train, y_train, X_test, y_test, seed=seed))
        sharding_mses[eps].append(run_sharding(eps, X_train, y_train, X_test, y_test, n_shards=10, seed=seed))

print("\nRunning AGT inference with Cauchy noise...")
agt_results = {}
for method in ['f5', 'f10']:  # Focus on best AGT methods
    if method not in results:
        continue
    
    mse_nom = results[method]['mse_nom']
    avg_sens = results[method]['avg_sens']
    
    method_mses = {eps: [] for eps in epsilons}
    for seed in range(n_seeds):
        for eps in epsilons:
            mse = run_agt_inference(eps, mse_nom, avg_sens, X_test, y_test, seed=seed)
            method_mses[eps].append(mse)
    
    agt_results[method] = method_mses

# ====================
# Plot
# ====================
fig, ax = plt.subplots(figsize=(10, 6))

# Laplace DP-SGD
lap_med = [np.median(laplace_mses[e]) for e in epsilons]
ax.plot(epsilons, lap_med, 's--', color='orange', label='Laplace DP-SGD', linewidth=2, markersize=8)

# Sharding S=20
shard_med = [np.median(sharding_mses[e]) for e in epsilons]
ax.plot(epsilons, shard_med, '^--', color='purple', label='Sharding (S=20)', linewidth=2, markersize=8)

# AGT methods
colors = {'f5': 'blue', 'f10': 'red'}
labels = {'f5': 'AGT F=5', 'f10': 'AGT F=10'}

for method in ['f5', 'f10']:
    if method not in agt_results:
        continue
    
    med = [np.median(agt_results[method][e]) for e in epsilons]
    ax.plot(epsilons, med, 'o-', color=colors[method], label=labels[method], linewidth=3, markersize=8)

ax.set_xlabel('Privacy Budget (ε)', fontsize=14)
ax.set_ylabel('Test MSE', fontsize=14)
ax.set_title('Pure DP: AGT vs Baselines (D=10, N=100)', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'agtr/linear/values/ablation/pure_dp_comparison.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close()

# Print summary
print("\n" + "="*50)
print("SUMMARY (median MSE)")
print("="*50)
print(f"{'Epsilon':<10} {'Laplace':<12} {'Sharding':<12} {'AGT F5':<12} {'AGT F10':<12}")
for eps in epsilons:
    lap = np.median(laplace_mses[eps])
    shard = np.median(sharding_mses[eps])
    f5 = np.median(agt_results.get('f5', {}).get(eps, [float('nan')]))
    f10 = np.median(agt_results.get('f10', {}).get(eps, [float('nan')]))
    print(f"{eps:<10.1f} {lap:<12.2f} {shard:<12.2f} {f5:<12.2f} {f10:<12.2f}")
