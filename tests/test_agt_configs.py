"""
Specific AGT configurations test.
All use CLIP_GAMMA=1.0, LR=0.05, D=2.
Uses OPTConfig with optimal frequency (concretize once at end).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel

# Fixed params
INPUT_DIM = 2
SPARSITY = 1
CLIP_GAMMA = 1.0
LR = 0.05
K_SUB = 1  # Privacy: 1 sample can be substituted

# Configs: (N, BS, EPOCHS, total_updates)
CONFIGS = [
    (100, 10, 1, 10),     # 10 batches × 1 epoch
    (100, 50, 5, 10),     # 2 batches × 5 epochs
    (100, 100, 10, 10),   # 1 batch × 10 epochs
    (1000, 100, 1, 10),   # 10 batches × 1 epoch
    (1000, 500, 5, 10),   # 2 batches × 5 epochs
    (1000, 1000, 10, 10), # 1 batch × 10 epochs
]

def simulated_linear_data(n_samples, seed=42):
    """Generate linear data with fixed w_true."""
    rng = np.random.RandomState(42)  # Fixed seed for w_true
    w_true = np.zeros(INPUT_DIM)
    w_true[:SPARSITY] = rng.randn(SPARSITY)
    
    rng_data = np.random.RandomState(seed)
    X = rng_data.uniform(-2, 2, (n_samples, INPUT_DIM))
    intercept = 1.0
    noise = rng_data.normal(0, 0.1, (n_samples, 1))
    y = X @ w_true.reshape(-1, 1) + intercept + noise
    return X, y, w_true

def run_config(n_samples, batch_size, epochs, total_updates):
    """Train with optimal frequency and return MSE + sensitivity."""
    torch.manual_seed(42)
    
    # Data
    X_train, y_train, w_true = simulated_linear_data(n_samples, seed=42)
    X_test, y_test, _ = simulated_linear_data(200, seed=100)
    
    # Optimal frequency = total_updates (concretize once at end)
    optimal_freq = total_updates
    
    print(f"  N={n_samples}, BS={batch_size}, EPOCHS={epochs} -> F={optimal_freq}")
    
    # Model
    model = nn.Sequential(nn.Linear(INPUT_DIM, 1)).double()
    
    # Dataset
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64)
    )
    
    # OPTConfig for optimal frequency MIP
    config = OPTConfig(
        log_level="WARNING",
        n_epochs=epochs,
        batchsize=batch_size,
        learning_rate=LR,
        loss="mse",
        concretization_freq=optimal_freq,
        
        # Privacy path
        k_substitution=K_SUB,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP_GAMMA,
        
        # Exact solve
        relax_binaries=False,
        relax_bilinear=False,
        
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 0, "Threads": 4},
        metadata={"n": n_samples}
    )
    
    try:
        final_model, _ = certified_training(model, config, dataset, solve=True)
        
        # Evaluate
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        
        with torch.no_grad():
            y_pred = final_model.forward(X_test_t).squeeze().numpy()
            mse = np.mean((y_pred - y_test.flatten())**2)
            
            # Get bounds width (sensitivity)
            yl, yu = final_model.bound_forward(X_test_t, X_test_t)
            sensitivity = (yu - yl).mean().item()
        
        return mse, sensitivity
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return float('nan'), float('nan')

def main():
    print("="*70)
    print("AGT Configuration Test - Optimal Frequency (concretize once)")
    print(f"CLIP_GAMMA={CLIP_GAMMA}, LR={LR}, D={INPUT_DIM}, K={K_SUB}")
    print("="*70)
    
    results = []
    
    for n, bs, epochs, updates in CONFIGS:
        mse, sens = run_config(n, bs, epochs, updates)
        results.append({
            'n': n, 'bs': bs, 'epochs': epochs,
            'updates': updates, 'mse': mse, 'sens': sens
        })
        
        if not np.isnan(mse):
            print(f"  -> MSE={mse:.4f}, Sensitivity={sens:.4f}")
        print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'N':>6} {'BS':>6} {'EPOCHS':>6} {'Updates':>8} {'MSE':>10} {'Sens':>10}")
    print("-"*70)
    for r in results:
        mse_str = f"{r['mse']:.4f}" if not np.isnan(r['mse']) else "nan"
        sens_str = f"{r['sens']:.4f}" if not np.isnan(r['sens']) else "nan"
        print(f"{r['n']:>6} {r['bs']:>6} {r['epochs']:>6} {r['updates']:>8} {mse_str:>10} {sens_str:>10}")
    
    # Save
    np.save('plotted_values/linear/ablation/config_test_results.npy', results)
    print("\nSaved to plotted_values/linear/ablation/config_test_results.npy")

if __name__ == "__main__":
    main()
