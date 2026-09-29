"""
Linear Benchmark Paper Plots - Self-contained script.
Generates publication-quality plots for the linear regression benchmark.

Plots generated:
1. Staircase & Sensitivity (dual-axis, for F=1,5,10,20,25)
2. Test MSE Bar Chart (nominal accuracy for each frequency)
3. Prediction Bands (colored by frequency)
4. Approximate DP Trade-off (AGT frequencies vs Sharding vs Nominal+GS)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig

# =============================================================================
# Configuration
# =============================================================================
N_SAMPLES = 25
INPUT_DIM = 2
SPARSITY = 1
BATCH_SIZE = 25  # Full batch
EPOCHS = 25
LR = 0.05
CLIP_GAMMA = 1.0
K_SUB = 1

# Frequencies to test
FREQUENCIES = [1, 5, 10, 20, 25]

# DP Parameters
DELTA = 1e-5
GS = 20.0  # Global output range [-10, 10]
BETA = 1.0 / 6.0  # For smooth sensitivity

# Sharding
N_SHARDS = 5

# Plotting
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
N_SEEDS = 20

# Style (matching CA/Synthea)
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.titlesize': 28,
    'lines.linewidth': 4,
    'axes.linewidth': 2
})

COLORS = {1: 'tab:blue', 5: 'tab:orange', 10: 'tab:green', 20: 'tab:red', 25: 'tab:purple'}

# =============================================================================
# Data Generation
# =============================================================================
def generate_data():
    """Generate train/test data with fixed w_true."""
    torch.manual_seed(42)
    rng = np.random.RandomState(42)
    w_true = np.zeros(INPUT_DIM)
    w_true[:SPARSITY] = rng.randn(SPARSITY)
    
    rng_train = np.random.RandomState(42)
    X_train = rng_train.uniform(-2, 2, (N_SAMPLES, INPUT_DIM))
    y_train = X_train @ w_true.reshape(-1, 1) + 1.0 + rng_train.normal(0, 0.1, (N_SAMPLES, 1))
    
    rng_test = np.random.RandomState(100)
    X_test = rng_test.uniform(-2, 2, (200, INPUT_DIM))
    y_test = X_test @ w_true.reshape(-1, 1) + 1.0 + rng_test.normal(0, 0.1, (200, 1))
    
    return X_train, y_train, X_test, y_test, w_true

# =============================================================================
# AGT Training (for each frequency)
# =============================================================================
def train_agt_freq(freq):
    """Train AGT with specified frequency and return MSE, sensitivity, staircase."""
    X_train, y_train, X_test, y_test, _ = generate_data()
    
    model = nn.Sequential(nn.Linear(INPUT_DIM, 1)).double()
    
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64)
    )
    
    config = OPTConfig(
        log_level="WARNING",
        n_epochs=EPOCHS,
        batchsize=BATCH_SIZE,
        learning_rate=LR,
        loss="mse",
        concretization_freq=freq,
        k_substitution=K_SUB,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP_GAMMA,
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 0, "Threads": 4},
    )
    
    final_model, _ = certified_training(model, config, dataset, solve=True)
    
    # Evaluate on test set
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    
    with torch.no_grad():
        y_pred = final_model.forward(X_test_t).squeeze().numpy()
        mse = np.mean((y_pred - y_test.flatten())**2)
        
        yl, yu = final_model.bound_forward(X_test_t, X_test_t)
        sensitivities = (yu - yl).squeeze().numpy()  # Per-point sensitivity
        avg_sens = np.mean(sensitivities)
    
    # Compute staircase (V_k for k=0..20)
    X_eval = torch.tensor(X_test[:20], dtype=torch.float64)  # Use 20 points for staircase
    ks = list(range(21))
    vs = []
    for k in ks:
        with torch.no_grad():
            yl, yu = final_model.bound_forward(X_eval, X_eval)
            max_radius = (yu - yl).max().item()
        vs.append(max_radius)
    
    # Predictions for bands
    with torch.no_grad():
        predictions = final_model.forward(X_test_t).squeeze().numpy()
        yl_all, yu_all = final_model.bound_forward(X_test_t, X_test_t)
        bounds = (yl_all.squeeze().numpy(), yu_all.squeeze().numpy())
    
    return {
        'freq': freq,
        'mse': mse,
        'avg_sens': avg_sens,
        'sensitivities': sensitivities,
        'ks': ks,
        'vs': vs,
        'predictions': predictions,
        'bounds': bounds,
        'y_test': y_test.flatten(),
        'X_test': X_test
    }

# =============================================================================
# Sharding Baseline
# =============================================================================
def train_sharding():
    """Train sharding ensemble and return nominal MSE."""
    X_train, y_train, X_test, y_test, _ = generate_data()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    shard_size = N_SAMPLES // N_SHARDS
    indices = np.random.permutation(N_SAMPLES)
    shard_models = []
    
    for s in range(N_SHARDS):
        start = s * shard_size
        end = start + shard_size
        X_shard = X_train[indices[start:end]]
        y_shard = y_train[indices[start:end]]
        
        model = nn.Linear(INPUT_DIM, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X_shard, dtype=torch.float32)
        y_t = torch.tensor(y_shard, dtype=torch.float32)
        
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
        
        shard_models.append(model)
    
    # Evaluate
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        all_preds = [m(X_test_t).numpy() for m in shard_models]
        avg_preds = np.mean(all_preds, axis=0)
    
    mse = np.mean((avg_preds - y_test)**2)
    return mse, shard_models

# =============================================================================
# Nominal Model
# =============================================================================
def train_nominal():
    """Train nominal model and return MSE."""
    X_train, y_train, X_test, y_test, _ = generate_data()
    
    torch.manual_seed(42)
    model = nn.Linear(INPUT_DIM, 1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
    
    return np.mean((y_pred - y_test)**2)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_staircase_sensitivity(results, out_dir):
    """Plot 1: Dual-axis staircase and smooth sensitivity."""
    fig, ax1 = plt.subplots(figsize=(10, 11))
    ax2 = ax1.twinx()
    
    ax1.set_xlabel('Privacy Parameter k')
    ax1.set_ylabel('Max. Local Sens. Certificate $A_k$ (dashed)')
    ax2.set_ylabel('Smooth Sensitivity $SS(f, x)$ (solid)')
    ax1.set_title('Linear')
    
    for res in results:
        freq = res['freq']
        ks = np.array(res['ks'])
        vs = np.array(res['vs'])
        c = COLORS[freq]
        
        # Staircase (dashed)
        ax1.step(ks, vs, where='post', label=f'F={freq}', color=c, linestyle='--', linewidth=4, alpha=0.75)
        
        # Smooth sensitivity curve (solid)
        t_grid = np.linspace(0, max(ks) + 5, 500)
        indices = np.searchsorted(ks, t_grid, side='right') - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        s_curve = v_interp * np.exp(-BETA * t_grid)
        
        ax2.plot(t_grid, s_curve, '-', color=c, linewidth=5, alpha=1.0)
        
        # Mark max smooth sensitivity
        idx_max = np.argmax(s_curve)
        t_max = t_grid[idx_max]
        s_max = s_curve[idx_max]
        ax2.plot(t_max, s_max, '*', color=c, markersize=28, markeredgecolor='black')
    
    ax1.legend(loc='upper right', title='Frequency')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lin_ablation_freq_staircase.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/lin_ablation_freq_staircase.png")

def plot_mse_bar(results, out_dir):
    """Plot 2: Bar chart of nominal MSE for each frequency."""
    plt.figure(figsize=(10, 11))
    
    freqs = [str(r['freq']) for r in results]
    mses = [r['mse'] for r in results]
    colors = [COLORS[r['freq']] for r in results]
    
    plt.bar(freqs, mses, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Concretization Frequency')
    plt.ylabel('Test MSE')
    plt.title('Linear')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lin_ablation_freq_mse_bar.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/lin_ablation_freq_mse_bar.png")

def plot_prediction_bands(results, out_dir):
    """Plot 3: Prediction bands colored by frequency."""
    X_train, y_train, _, _, _ = generate_data()
    
    plt.figure(figsize=(10, 11))
    
    # Plot training data
    plt.scatter(X_train[:, 0], y_train.flatten(), c='black', alpha=0.3, label='Data', s=100)
    
    # Use first dimension for plotting
    for res in results:
        freq = res['freq']
        X_test = res['X_test']
        y_pred = res['predictions']
        y_low, y_high = res['bounds']
        c = COLORS[freq]
        
        # Sort by x for clean bands
        sort_idx = np.argsort(X_test[:, 0])
        x_sorted = X_test[:, 0][sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_low_sorted = y_low[sort_idx]
        y_high_sorted = y_high[sort_idx]
        
        # Plot prediction and bands
        plt.plot(x_sorted, y_pred_sorted, '-', color=c, linewidth=2, label=f'F={freq}')
        plt.fill_between(x_sorted, y_low_sorted, y_high_sorted, color=c, alpha=0.15)
    
    plt.xlabel('Input $x_1$')
    plt.ylabel('Output $y$')
    plt.title('Linear Prediction Bounds')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lin_ablation_freq_prediction_bands.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/lin_ablation_freq_prediction_bands.png")

def plot_approx_dp_tradeoff(results, nom_mse, shard_mse, shard_models, out_dir):
    """Plot 4: Approximate DP trade-off for all frequencies."""
    plt.figure(figsize=(10, 11))
    
    n_test = 200
    n_trials = 500
    
    # Nominal + GS baseline
    nom_gs_mse = []
    for eps in EPSILONS:
        sigma = (GS / eps) * np.sqrt(2 * np.log(1.25 / DELTA))
        mse_trials = [nom_mse + sigma**2 for _ in range(n_trials)]
        nom_gs_mse.append(np.mean(mse_trials))
    plt.plot(EPSILONS, nom_gs_mse, 'o--', label='Nominal + GS', color='gray', alpha=0.5, linewidth=4)
    
    # Sharding baseline
    shard_mse_list = []
    for eps in EPSILONS:
        sigma = ((GS / N_SHARDS) / eps) * np.sqrt(2 * np.log(1.25 / DELTA))
        mse_trials = [shard_mse + sigma**2 for _ in range(n_trials)]
        shard_mse_list.append(np.mean(mse_trials))
    plt.plot(EPSILONS, shard_mse_list, 's-', label=f'Sharding (S={N_SHARDS})', color='black', linewidth=4)
    
    # AGT for each frequency
    for res in results:
        freq = res['freq']
        avg_sens = res['avg_sens']
        mse_nom = res['mse']
        c = COLORS[freq]
        
        mse_list = []
        std_list = []
        for eps in EPSILONS:
            scale = 2 * avg_sens / eps  # Laplace scale for Approx DP
            laplace_var = 2 * scale**2
            mse_exp = mse_nom + laplace_var
            mse_list.append(mse_exp)
        
        plt.plot(EPSILONS, mse_list, 'o-', label=f'AGT (F={freq})', color=c, linewidth=4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\\epsilon$')
    plt.ylabel('Test MSE')
    plt.title('Linear (Approx DP)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lin_ablation_freq_tradeoff_approx.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/lin_ablation_freq_tradeoff_approx.png")

# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("LINEAR BENCHMARK - Paper Plots")
    print("="*60)
    print(f"Config: N={N_SAMPLES}, D={INPUT_DIM}, BS={BATCH_SIZE}, EPOCHS={EPOCHS}")
    print(f"        Frequencies: {FREQUENCIES}")
    print(f"        GS={GS}, S={N_SHARDS}, delta={DELTA}")
    print()
    
    out_dir = 'agtr/linear/figures/linear'
    os.makedirs(out_dir, exist_ok=True)
    
    # Train AGT for each frequency
    print("Training AGT for each frequency...")
    results = []
    for freq in FREQUENCIES:
        print(f"  F={freq}...")
        res = train_agt_freq(freq)
        results.append(res)
        print(f"    MSE={res['mse']:.4f}, Avg Sens={res['avg_sens']:.4f}")
    
    # Train baselines
    print("\nTraining baselines...")
    nom_mse = train_nominal()
    print(f"  Nominal MSE: {nom_mse:.4f}")
    
    shard_mse, shard_models = train_sharding()
    print(f"  Sharding MSE: {shard_mse:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_staircase_sensitivity(results, out_dir)
    plot_mse_bar(results, out_dir)
    plot_prediction_bands(results, out_dir)
    plot_approx_dp_tradeoff(results, nom_mse, shard_mse, shard_models, out_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
