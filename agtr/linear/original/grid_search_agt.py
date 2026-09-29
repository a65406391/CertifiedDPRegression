"""
Grid search for optimal AGT parameters.
Uses optimal frequency (F = total_updates, concretize once at end).
Searches over: LR, BATCH_SIZE, EPOCHS, CLIP_GAMMA
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstract_gradient_training import AGTConfig
from abstract_gradient_training.privacy_certified_training import privacy_certified_training

# Fixed params
N_SAMPLES = 100
INPUT_DIM = 2
SPARSITY = 1

# Grid search params
LR_VALUES = [0.1, 0.5, 1.0]
BATCH_SIZE_VALUES = [10, 25, 50]
EPOCHS_VALUES = [1, 3, 5]
CLIP_GAMMA_VALUES = [0.1, 0.5, 1.0]

# Load w_true
W_TRUE_PATH = 'agtr/linear/data/w_true.npy'
w_true = np.load(W_TRUE_PATH)

def simulated_linear_data(n_samples, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n_samples, INPUT_DIM))
    intercept = 1.0
    noise = rng.normal(0, 0.1, (n_samples, 1))
    y = X @ w_true.reshape(-1, 1) + intercept + noise
    return X, y

def run_single_config(lr, batch_size, epochs, clip_gamma):
    """Train with optimal frequency and return MSE + sensitivity."""
    torch.manual_seed(42)
    
    # Data
    X_train, y_train = simulated_linear_data(N_SAMPLES, seed=42)
    X_test, y_test = simulated_linear_data(200, seed=100)
    
    # Total updates = optimal frequency (concretize once at end)
    n_batches = N_SAMPLES // batch_size
    total_updates = n_batches * epochs
    optimal_freq = total_updates  # F = total updates
    
    # Model
    model = nn.Sequential(nn.Linear(INPUT_DIM, 1)).double()
    
    # Config
    config = AGTConfig(
        fragsize=batch_size,
        n=N_SAMPLES,
        eps_l_inf=0.0,  # No input perturbation
        seed=42,
        concretization_freq=optimal_freq,
    )
    
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64)
    )
    
    try:
        bounded_model = privacy_certified_training(
            model=model,
            config=config,
            dataset=dataset,
            epochs=epochs,
            lr=lr,
            epsilon=1.0,  # Doesn't matter for sensitivity
            clip_gamma=clip_gamma,
            reuse_input_bounds=True,
            verbose=False,
        )
        
        # Evaluate
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        y_test_t = torch.tensor(y_test, dtype=torch.float64)
        
        with torch.no_grad():
            y_pred = bounded_model.forward(X_test_t).squeeze().numpy()
            mse = np.mean((y_pred - y_test.flatten())**2)
            
            # Get bounds width (sensitivity proxy)
            yl, yu = bounded_model.bound_forward(X_test_t, X_test_t)
            sensitivity = (yu - yl).mean().item()
        
        return mse, sensitivity
    except Exception as e:
        return float('nan'), float('nan')

def main():
    print("="*60)
    print("GRID SEARCH: Optimal AGT Parameters")
    print(f"N={N_SAMPLES}, D={INPUT_DIM}")
    print("="*60)
    
    results = []
    
    for lr in LR_VALUES:
        for bs in BATCH_SIZE_VALUES:
            for epochs in EPOCHS_VALUES:
                for gamma in CLIP_GAMMA_VALUES:
                    mse, sens = run_single_config(lr, bs, epochs, gamma)
                    total_updates = (N_SAMPLES // bs) * epochs
                    results.append({
                        'lr': lr, 'bs': bs, 'epochs': epochs, 'gamma': gamma,
                        'updates': total_updates, 'mse': mse, 'sens': sens
                    })
                    
                    if not np.isnan(mse):
                        print(f"LR={lr}, BS={bs}, EPOCHS={epochs}, GAMMA={gamma} -> MSE={mse:.4f}, Sens={sens:.4f}")
    
    # Find best
    valid_results = [r for r in results if not np.isnan(r['mse'])]
    if valid_results:
        best_sens = min(valid_results, key=lambda x: x['sens'])
        best_mse = min(valid_results, key=lambda x: x['mse'])
        
        print("\n" + "="*60)
        print("BEST RESULTS")
        print("="*60)
        print(f"Best Sensitivity: Sens={best_sens['sens']:.4f}, MSE={best_sens['mse']:.4f}")
        print(f"  -> LR={best_sens['lr']}, BS={best_sens['bs']}, EPOCHS={best_sens['epochs']}, GAMMA={best_sens['gamma']}")
        print(f"Best MSE: MSE={best_mse['mse']:.4f}, Sens={best_mse['sens']:.4f}")
        print(f"  -> LR={best_mse['lr']}, BS={best_mse['bs']}, EPOCHS={best_mse['epochs']}, GAMMA={best_mse['gamma']}")
        
        # Save results
        np.save('agtr/linear/values/ablation/grid_search_results.npy', results)
        print("\nSaved results to agtr/linear/values/ablation/grid_search_results.npy")

if __name__ == "__main__":
    main()
