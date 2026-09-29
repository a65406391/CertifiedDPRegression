"""
Final Linear Benchmark Figures - Consolidated Script
Generates publication-quality plots for the linear regression benchmark.

Figures generated:
1. Staircase & Smooth Sensitivity (dual-axis)
2a. Pure DP Trade-off (AGT vs Sharding)
2b. Approximate DP Trade-off (AGT vs Sharding)
3. Function Bands (data scatter + DP bounds)
4. Baseline Comparison (Pure & Approx DP)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys
import pickle
import time
from tqdm import tqdm

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
EPOCHS = 3  # Reduced for faster iteration
LR = 0.1
CLIP_GAMMA = 1.0
K_SUB = 1
HIDDEN_DIM = 4  # Reduced from 8 for smaller MIP

# Frequencies to test (with full batch, F=EPOCHS is global visibility)
FREQUENCIES = [1, 5, 10]

# Checkpoint directory
CHECKPOINT_DIR = "agtr/linear/checkpoints"

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
    'legend.fontsize': 20,
    'figure.titlesize': 28,
    'lines.linewidth': 4,
    'axes.linewidth': 2
})

COLORS = {1: 'tab:blue', 5: 'tab:orange', 10: 'tab:green'}

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
# Checkpoint Utilities
# =============================================================================
def get_checkpoint_path(freq, k):
    """Get checkpoint file path for given (freq, k) combination."""
    return os.path.join(CHECKPOINT_DIR, f"freq{freq}_k{k}.pkl")

def save_checkpoint(freq, k, model_data):
    """Save model bounds to checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = get_checkpoint_path(freq, k)
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

def load_checkpoint(freq, k):
    """Load model bounds from checkpoint if exists."""
    path = get_checkpoint_path(freq, k)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def extract_model_data(model, X_test_t, y_test):
    """Extract serializable data from trained model."""
    with torch.no_grad():
        y_pred = model.forward(X_test_t).squeeze().numpy()
        mse = float(np.mean((y_pred - y_test.flatten())**2))
        yl, yu = model.bound_forward(X_test_t, X_test_t)
        param_l = [p.detach().cpu().numpy() for p in model.param_l]
        param_u = [p.detach().cpu().numpy() for p in model.param_u]
        param_n = [p.detach().cpu().numpy() for p in model.param_n]
    
    return {
        'param_l': param_l,
        'param_u': param_u,
        'param_n': param_n,
        'y_pred': y_pred,
        'mse': mse,
        'y_l': yl.squeeze().numpy(),
        'y_u': yu.squeeze().numpy(),
    }

# =============================================================================
# AGT Training - Train model for specific (freq, k) config
# =============================================================================
def train_agt_model(freq, k_sub):
    """Train AGT model with specified frequency and k_substitution value."""
    X_train, y_train, X_test, y_test, _ = generate_data()
    
    # 1-hidden-layer ReLU network to introduce non-linearity
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, 1)
    ).double()
    
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
        k_substitution=k_sub,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP_GAMMA,
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 0, "Threads": 4},
    )
    
    final_model, _ = certified_training(model, config, dataset, solve=True)
    return final_model


def train_agt_freq(freq):
    """Train AGT with specified frequency for all k values and compute staircase.
    
    Uses checkpointing to save each model after training and resume from cached.
    Includes TQDM progress bars with time estimates.
    """
    X_train, y_train, X_test, y_test, _ = generate_data()
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    
    # k values for staircase - use even numbers only for speed (0,2,4,...,20)
    K_MAX = 20
    k_values = [0] + list(range(2, K_MAX + 1, 2))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Train models for each k value with TQDM progress bar
    models_data = {}
    start_time = time.time()
    completed = 0
    
    pbar = tqdm(k_values, desc=f"  Freq={freq}", leave=True)
    for k in pbar:
        # Check for cached result
        actual_k = 1 if k == 0 else k  # k=0 uses k=1 model
        cached = load_checkpoint(freq, actual_k)
        
        if cached is not None:
            models_data[k] = cached
            completed += 1
            elapsed = time.time() - start_time
            pbar.set_postfix({'status': 'cached', 'elapsed': f'{elapsed:.0f}s'})
        else:
            # Train model
            model = train_agt_model(freq, actual_k)
            
            # Extract and save data
            model_data = extract_model_data(model, X_test_t, y_test)
            save_checkpoint(freq, actual_k, model_data)
            
            # Also store the actual model for staircase computation
            model_data['_model'] = model
            models_data[k] = model_data
            
            completed += 1
            elapsed = time.time() - start_time
            per_model = elapsed / completed
            remaining = (len(k_values) - completed) * per_model
            pbar.set_postfix({
                'elapsed': f'{elapsed:.0f}s',
                'remaining': f'{remaining:.0f}s'
            })
    
    # For staircase computation, we need the actual model objects
    # Load models from checkpoints if needed
    models = {}
    for k in k_values:
        if '_model' in models_data[k]:
            models[k] = models_data[k]['_model']
        else:
            # Reload model from checkpoint data (need to reconstruct)
            # For now, use cached bound data directly
            models[k] = None
    
    # Use cached data for evaluation
    nom_data = models_data[0]
    y_pred = nom_data['y_pred']
    mse = nom_data['mse']
    yl, yu = nom_data['y_l'], nom_data['y_u']
    sensitivities = yu - yl
    avg_sens = float(np.mean(sensitivities))
    
    # Compute staircase: V_k for each k using cached bounds
    X_eval_idx = slice(0, 50)  # Use first 50 points for staircase
    y_nom_eval = nom_data['y_pred'][:50] if len(nom_data['y_pred']) >= 50 else nom_data['y_pred']
    
    vs = []
    for k in k_values:
        if k == 0:
            vs.append(0.0)  # V_0 = 0 by definition
        else:
            data_k = models_data[k]
            yl_np = data_k['y_l'][:50] if len(data_k['y_l']) >= 50 else data_k['y_l']
            yu_np = data_k['y_u'][:50] if len(data_k['y_u']) >= 50 else data_k['y_u']
            
            # V_k = max deviation from nominal across all eval points
            v_k_per_point = np.maximum(np.abs(yl_np - y_nom_eval), np.abs(yu_np - y_nom_eval))
            v_k = float(np.max(v_k_per_point))
            vs.append(v_k)
    
    # Predictions and bounds for function bands (use k=1 model data)
    predictions = nom_data['y_pred']
    bounds = (nom_data['y_l'], nom_data['y_u'])
    
    total_time = time.time() - start_time
    print(f"  Freq={freq} completed in {total_time:.1f}s")
    
    return {
        'freq': freq,
        'mse': mse,
        'avg_sens': avg_sens,
        'sensitivities': sensitivities,
        'ks': k_values,
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
        
        # Use same ReLU architecture as AGT for fair comparison
        model = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
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
    # Use same ReLU architecture as AGT for fair comparison
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, 1)
    )
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
# Figure 1: Staircase & Smooth Sensitivity
# =============================================================================
def plot_staircase_sensitivity(results, out_dir):
    """Plot 1: Dual-axis staircase (dashed) and smooth sensitivity (solid)."""
    fig, ax1 = plt.subplots(figsize=(10, 11))
    ax2 = ax1.twinx()
    
    ax1.set_xlabel('Privacy Parameter $k$')
    ax1.set_ylabel('Local Sens. Certificate $A_k$ (dashed)')
    ax2.set_ylabel('Smooth Sensitivity $SS(f, x)$ (solid)')
    ax1.set_title('Linear Benchmark')
    
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
    plt.savefig(f'{out_dir}/fig1_staircase_sensitivity.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/fig1_staircase_sensitivity.png")

# =============================================================================
# Figure 2a: Pure DP Trade-off
# =============================================================================
def plot_pure_dp_tradeoff(results, nom_mse, shard_mse, out_dir):
    """Plot 2a: Pure DP trade-off (AGT Cauchy vs Sharding Laplace)."""
    plt.figure(figsize=(10, 11))
    
    n_trials = 500
    
    # Nominal + GS baseline (Laplace for Pure DP)
    nom_gs_mse = []
    for eps in EPSILONS:
        scale = GS / eps  # Laplace scale
        laplace_var = 2 * scale**2
        nom_gs_mse.append(nom_mse + laplace_var)
    plt.plot(EPSILONS, nom_gs_mse, 'o--', label='Nominal + GS', color='gray', alpha=0.5, linewidth=4)
    
    # Sharding baseline (Laplace)
    shard_mse_list = []
    for eps in EPSILONS:
        scale = (GS / N_SHARDS) / eps  # Laplace scale after sharding
        laplace_var = 2 * scale**2
        shard_mse_list.append(shard_mse + laplace_var)
    plt.plot(EPSILONS, shard_mse_list, 's-', label=f'Sharding (S={N_SHARDS}, Laplace)', color='black', linewidth=4)
    
    # AGT for each frequency (Cauchy - use median over trials)
    for res in results:
        freq = res['freq']
        avg_sens = res['avg_sens']
        mse_nom = res['mse']
        c = COLORS[freq]
        
        # Cauchy noise: scale = 6*S/epsilon for pure DP smooth sensitivity
        mse_medians = []
        for eps in EPSILONS:
            scale = 6 * avg_sens / eps  # Cauchy scale
            # Simulate Cauchy noise addition (median of trials)
            np.random.seed(42)
            trials = []
            for _ in range(n_trials):
                noise = np.random.standard_cauchy(1)
                noisy_pred_mse = mse_nom + (scale * noise[0])**2
                trials.append(noisy_pred_mse)
            mse_medians.append(np.median(trials))
        
        plt.plot(EPSILONS, mse_medians, 'o-', label=f'AGT (F={freq}, Cauchy)', color=c, linewidth=4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\\epsilon$')
    plt.ylabel('Test MSE')
    plt.title('Linear (Pure DP)')
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fig2a_pure_dp_tradeoff.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/fig2a_pure_dp_tradeoff.png")

# =============================================================================
# Figure 2b: Approximate DP Trade-off
# =============================================================================
def plot_approx_dp_tradeoff(results, nom_mse, shard_mse, out_dir):
    """Plot 2b: Approximate DP trade-off (AGT Laplace vs Sharding Gaussian)."""
    plt.figure(figsize=(10, 11))
    
    # Nominal + GS baseline (Gaussian for Approx DP)
    nom_gs_mse = []
    for eps in EPSILONS:
        sigma = (GS / eps) * np.sqrt(2 * np.log(1.25 / DELTA))
        nom_gs_mse.append(nom_mse + sigma**2)
    plt.plot(EPSILONS, nom_gs_mse, 'o--', label='Nominal + GS', color='gray', alpha=0.5, linewidth=4)
    
    # Sharding baseline (Gaussian)
    shard_mse_list = []
    for eps in EPSILONS:
        sigma = ((GS / N_SHARDS) / eps) * np.sqrt(2 * np.log(1.25 / DELTA))
        shard_mse_list.append(shard_mse + sigma**2)
    plt.plot(EPSILONS, shard_mse_list, 's-', label=f'Sharding (S={N_SHARDS}, Gaussian)', color='black', linewidth=4)
    
    # AGT for each frequency (Laplace for Approx DP)
    for res in results:
        freq = res['freq']
        avg_sens = res['avg_sens']
        mse_nom = res['mse']
        c = COLORS[freq]
        
        mse_list = []
        for eps in EPSILONS:
            scale = 2 * avg_sens / eps  # Laplace scale for Approx DP
            laplace_var = 2 * scale**2
            mse_list.append(mse_nom + laplace_var)
        
        plt.plot(EPSILONS, mse_list, 'o-', label=f'AGT (F={freq}, Laplace)', color=c, linewidth=4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\\epsilon$')
    plt.ylabel('Test MSE')
    plt.title('Linear (Approx DP)')
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fig2b_approx_dp_tradeoff.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/fig2b_approx_dp_tradeoff.png")

# =============================================================================
# Figure 3: Function Bands Plot
# =============================================================================
def plot_function_bands(results, out_dir):
    """Plot 3: Data scatter with prediction bounds as shaded regions."""
    X_train, y_train, _, _, _ = generate_data()
    
    plt.figure(figsize=(10, 11))
    
    # Plot training data as scatter
    plt.scatter(X_train[:, 0], y_train.flatten(), c='black', alpha=0.6, label='Data', s=120, zorder=10, edgecolors='white')
    
    # Plot prediction bands for each frequency (use only a subset for clarity)
    freqs_to_plot = [1, 10, 25]  # Show 3 representative frequencies
    
    for res in results:
        freq = res['freq']
        if freq not in freqs_to_plot:
            continue
            
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
        
        # Plot prediction line and bands
        plt.plot(x_sorted, y_pred_sorted, '-', color=c, linewidth=3, label=f'F={freq}')
        plt.fill_between(x_sorted, y_low_sorted, y_high_sorted, color=c, alpha=0.2)
    
    plt.xlabel('Input $x_1$')
    plt.ylabel('Output $y$')
    plt.title('Linear Prediction Bounds')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fig3_function_bands.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/fig3_function_bands.png")

# =============================================================================
# Figure 4: Baseline Comparison (Pure & Approx DP)
# =============================================================================
def plot_baseline_comparison(results, nom_mse, shard_mse, out_dir):
    """Plot 4: Side-by-side comparison of AGT vs Sharding for Pure and Approx DP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    n_trials = 500
    
    # Use best AGT frequency (F=25) for comparison
    best_res = [r for r in results if r['freq'] == 25][0]
    avg_sens = best_res['avg_sens']
    mse_nom = best_res['mse']
    
    # --- Pure DP (left panel) ---
    # Sharding + Laplace
    shard_pure = []
    for eps in EPSILONS:
        scale = (GS / N_SHARDS) / eps
        shard_pure.append(shard_mse + 2 * scale**2)
    ax1.plot(EPSILONS, shard_pure, 's-', label='Sharding + Laplace', color='black', linewidth=4)
    
    # AGT + Cauchy
    agt_pure = []
    for eps in EPSILONS:
        scale = 6 * avg_sens / eps
        np.random.seed(42)
        trials = [mse_nom + (scale * np.random.standard_cauchy(1)[0])**2 for _ in range(n_trials)]
        agt_pure.append(np.median(trials))
    ax1.plot(EPSILONS, agt_pure, 'o-', label='AGT (F=25) + Cauchy', color='tab:purple', linewidth=4)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Privacy Budget $\\epsilon$')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Pure DP')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Approx DP (right panel) ---
    # Sharding + Gaussian
    shard_approx = []
    for eps in EPSILONS:
        sigma = ((GS / N_SHARDS) / eps) * np.sqrt(2 * np.log(1.25 / DELTA))
        shard_approx.append(shard_mse + sigma**2)
    ax2.plot(EPSILONS, shard_approx, 's-', label='Sharding + Gaussian', color='black', linewidth=4)
    
    # AGT + Laplace
    agt_approx = []
    for eps in EPSILONS:
        scale = 2 * avg_sens / eps
        agt_approx.append(mse_nom + 2 * scale**2)
    ax2.plot(EPSILONS, agt_approx, 'o-', label='AGT (F=25) + Laplace', color='tab:purple', linewidth=4)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Privacy Budget $\\epsilon$')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Approximate DP')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fig4_baseline_comparison.png', dpi=300)
    plt.close()
    print(f"  Saved: {out_dir}/fig4_baseline_comparison.png")

# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("LINEAR BENCHMARK - Final Figures")
    print("="*60)
    print(f"Config: N={N_SAMPLES}, D={INPUT_DIM}, BS={BATCH_SIZE}, EPOCHS={EPOCHS}")
    print(f"        Frequencies: {FREQUENCIES}")
    print(f"        GS={GS}, S={N_SHARDS}, delta={DELTA}")
    print()
    
    out_dir = 'agtr/linear/figures/final_linear'
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
    
    # Generate all figures
    print("\nGenerating figures...")
    plot_staircase_sensitivity(results, out_dir)
    plot_pure_dp_tradeoff(results, nom_mse, shard_mse, out_dir)
    plot_approx_dp_tradeoff(results, nom_mse, shard_mse, out_dir)
    plot_function_bands(results, out_dir)
    plot_baseline_comparison(results, nom_mse, shard_mse, out_dir)
    
    print("\n" + "="*60)
    print("All figures saved to:", out_dir)
    print("="*60)

if __name__ == "__main__":
    main()
