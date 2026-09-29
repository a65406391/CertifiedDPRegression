"""
Linear Regression Ablation Study for Privacy Certified Training.

This script studies how concretization frequency affects certified bounds
when using gradient clipping for privacy certification.

Privacy Model: Per-sample gradient clipping (clip_method="clamp")
- Each gradient element is clamped to [-gamma, gamma]
- Privacy bounds use SEMin/SEMax accumulation with k*gamma sensitivity
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset

# Import Certified Training
from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training import privacy_certified_training, AGTConfig

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# REDUCED DATA BENCHMARK (User Requested)
# ============================================================
QUICK_TEST = False 

# Constants
# N=80, BS=5, 1 Epoch => 16 Updates.
# F=16 covers full dataset.
BATCH_SIZE = 5
N_SAMPLES = 80
EPOCHS = 1
INPUT_DIM = 1
SPARSITY = 1 

LR = 0.1
CLIP_GAMMA = 0.1 
W_TRUE_PATH = 'agtr/linear/data/w_true.npy'


def get_or_create_w_true(dim=INPUT_DIM, sparsity=SPARSITY, seed=42):
    """Get or create consistent true weights for the experiment."""
    import os
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(W_TRUE_PATH):
        w_true = np.load(W_TRUE_PATH)
        # Verify dimensions match
        if w_true.shape[0] == dim:
            return w_true
    
    # Create new w_true with fixed seed
    rng = np.random.RandomState(seed)
    w_true = np.zeros(dim)
    w_true[:sparsity] = rng.randn(sparsity)
    
    # Save for consistency
    np.save(W_TRUE_PATH, w_true)
    print(f"Saved w_true to {W_TRUE_PATH}: {w_true[:sparsity]}")
    return w_true


def simulated_linear_data(n_samples=1000, dim=INPUT_DIM, sparsity=SPARSITY, x_range=(-2, 2), seed=42, w_true=None):
    """Generate high-dimensional linear regression data with sparse true weights.
    
    y = X @ w_true + intercept + noise
    where w_true has only `sparsity` non-zero entries.
    
    IMPORTANT: w_true is shared across train/test to ensure consistency.
    """
    # Get consistent w_true
    if w_true is None:
        w_true = get_or_create_w_true(dim, sparsity)
    
    # Use seed only for X and noise (not w_true)
    rng = np.random.RandomState(seed)
    X = rng.uniform(x_range[0], x_range[1], (n_samples, dim))
    
    intercept = 1.0
    noise = rng.normal(0, 0.1, (n_samples, 1))
    y = X @ w_true.reshape(-1, 1) + intercept + noise
    return X, y



# Fixed Initialization State
INIT_STATE_PATH = 'agtr/linear/data/init_state.pth'

def get_or_create_init_state(model_fn):
    """Ensure identical model initialization across all runs."""
    if not os.path.exists(INIT_STATE_PATH):
        torch.manual_seed(42)
        model = model_fn()
        torch.save(model.state_dict(), INIT_STATE_PATH)
        print(f"Created shared initialization at {INIT_STATE_PATH}")
    return torch.load(INIT_STATE_PATH)

def train_certified_model(X_train, y_train, epochs, lr, seed, method, k):
    """Train a certified model with privacy bounds and STRICT DETERMINISM."""
    # 1. Global Seeding (just in case)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 2. Strict Model Initialization
    def create_model():
        return nn.Sequential(nn.Linear(INPUT_DIM, 1)).double()
    
    init_state = get_or_create_init_state(create_model)
    model = create_model()
    model.load_state_dict(init_state)
    
    # 3. Deterministic Data Shuffling
    # We shuffle ONCE based on the seed, then pass shuffle=False to loader
    # This ensures IBP (Python loop) and MIP (Internal flow) see exact same batch ordering
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(X_train), generator=g)
    
    X_train_shuffled = torch.tensor(X_train[perm], dtype=torch.float64)
    y_train_shuffled = torch.tensor(y_train[perm], dtype=torch.float64)
    
    dataset = TensorDataset(X_train_shuffled, y_train_shuffled)
    
    # ========================================
    # Case k=0: Standard Nominal Training
    # ========================================
    if k == 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='mean')
        # SHUFFLE=FALSE because we handled it above
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param.grad.clamp_(-CLIP_GAMMA, CLIP_GAMMA)
                optimizer.step()
        return IntervalBoundedModel(model)

    # ========================================
    # Case 'ibp': Pure IBP using AGTConfig
    # ========================================
    if method == 'ibp':
        agt_config = AGTConfig(
            loss="mse",
            n_epochs=epochs,
            learning_rate=lr,
            device="cpu",
            k_private=k,
            clip_gamma=CLIP_GAMMA,
            clip_method="clamp",
            log_level="WARNING",
            metadata={"seed": seed, "strict_mode": True}
        )
        # SHUFFLE=FALSE
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        final_model = privacy_certified_training(IntervalBoundedModel(model), agt_config, loader)
        return final_model

    # ========================================
    # Case 'fN': OPTConfig with MIP, freq=N
    # ========================================
    freq_map = {'f1': 1, 'f2': 2, 'f4': 4, 'f8': 8, 'f16': 16}
    freq = freq_map.get(method, 1)
    
    effective_batch_size = freq * BATCH_SIZE
    if k > effective_batch_size:
        agt_config = AGTConfig(
            loss="mse",
            n_epochs=epochs,
            learning_rate=lr,
            device="cpu",
            k_private=k,
            clip_gamma=CLIP_GAMMA,
            clip_method="clamp",
            log_level="WARNING",
            metadata={"seed": seed, "strict_mode": True}
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        final_model = privacy_certified_training(IntervalBoundedModel(model), agt_config, loader)
        return final_model
    
    config = OPTConfig(
        log_level="WARNING",
        n_epochs=epochs,
        batchsize=BATCH_SIZE,
        learning_rate=lr,
        loss="mse",
        concretization_freq=freq,
        k_substitution=k,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP_GAMMA,
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 1, "Threads": 1, "TimeLimit": 600},
        metadata={"seed": seed, "strict_mode": True}
    )
    
    # Pass PRE-SHUFFLED dataset directly.
    # certified_training will create a loader. We trust it uses the seed in config 
    # or that OPT defaults to shuffle=False for certification (usually true for MIP).
    # Even if it shuffles, it uses the seed from config, so it remains deterministic.
    final_model, _ = certified_training(model, config, dataset, solve=True)
    return final_model


def run_lin_ablation():
    """Run the linear ablation study.
    
    Compares three training methods:
    - 'ibp': Pure IBP using AGTConfig (baseline)
    - 'f1': OPTConfig with concretization_freq=1 (MIP every step)
    - 'f2': OPTConfig with concretization_freq=2 (MIP every 2 steps)
    
    Higher frequency (lower freq number) means more MIP solving, which should
    give tighter bounds.
    """
    print("=" * 60)
    print("Linear Ablation Study - Privacy Certified Training")
    print(f"Mode: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    print(f"Samples: {N_SAMPLES}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Clipping: gamma={CLIP_GAMMA}")
    print("=" * 60)
    
    out_dir = 'agtr/linear/values/ablation'
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate Data
    X_train, y_train = simulated_linear_data(n_samples=N_SAMPLES, seed=42)
    X_test, y_test = simulated_linear_data(n_samples=200 if QUICK_TEST else 2000, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Methods to compare
    # - 'ibp': Pure IBP (baseline)
    # - 'f1': MIP every step (should be tighter than ibp)
    # - 'f2': MIP every 2 steps (should be tighter than f1)
    all_methods = ['ibp', 'f1', 'f2', 'f4', 'f8', 'f16']
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=all_methods, help='Methods to run')
    args, unknown = parser.parse_known_args()
    
    methods = [m for m in args.methods if m in all_methods]
    
    k_values = list(range(0, 6))  # 0..5
    print(f"Methods to run: {methods}")
    print(f"k_values: {k_values}")
    
    # Sensitivity params
    GS_VAL = 10.0  # Global sensitivity cap
    beta = 1.0 / 6.0
    max_k_eval = max(k_values)
    tail_S = GS_VAL * np.exp(-beta * (max_k_eval + 1))
    
    parser.add_argument('--merge', action='store_true', help='Merge results and plot')
    args, unknown = parser.parse_known_args()
    
    methods = [m for m in args.methods if m in all_methods]
    k_values = list(range(0, 6))
    
    results = {}
    if args.merge:
        print("Merging existing results...")
        for m in all_methods:
            fname = f'{out_dir}/method_results_{m}.npy'
            if os.path.exists(fname):
                try:
                    res = np.load(fname, allow_pickle=True).item()
                    results[m] = res
                    print(f"Loaded {m} from {fname}")
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        # Skip training if merging
        methods = []

    # Eval points for sensitivity
    np.random.seed(999)
    n_eval = 10 if QUICK_TEST else 50
    X_sens_eval = np.random.uniform(-2, 2, (n_eval, INPUT_DIM))
    X_sens_eval_t = torch.tensor(X_sens_eval, dtype=torch.float64)

    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print("-" * 40)
        
        if method in results:
             print(f"Skipping {method} (Already computed)")
             continue
             
        method_models = {}
        
        # Use TQDM for progress tracking
        pbar = tqdm(k_values, desc=f"Training {method.upper()}")
        for k in pbar:
            pbar.set_description(f"Method {method.upper()} (k={k})")
            # print(f"  Training k={k}...", end=" ") # Removed print in favor of tqdm
            try:
                m = train_certified_model(X_train, y_train, epochs=EPOCHS, lr=LR, seed=42, method=method, k=k)
                method_models[k] = m
                
                # Print bounds for verification
                w_l = float(m.param_l[0].flatten()[0])
                w_u = float(m.param_u[0].flatten()[0])
                print(f"w: [{w_l:.4f}, {w_u:.4f}] (width={w_u-w_l:.4f})")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                method_models[k] = None
        
        # Analysis
        nom_model = method_models.get(0)
        if nom_model is None:
            print("  Nominal model failed, skipping method.")
            continue
            
        with torch.no_grad():
            # MSE on test data
            test_pred = nom_model.forward(X_test_t).squeeze().numpy()
            mse_nom = np.mean((test_pred - y_test_t.numpy().flatten())**2)
            
            y_pred_sens = nom_model.forward(X_sens_eval_t).squeeze().numpy()
            
            V_k_mat = np.zeros((len(X_sens_eval), len(k_values)))
            for j, k in enumerate(k_values):
                if k == 0 or method_models.get(k) is None:
                    continue
                mk = method_models[k]
                
                # Get output interval at each evaluation point
                yl, yu = mk.bound_forward(X_sens_eval_t, X_sens_eval_t, retain_intermediate=False)
                yl = yl.squeeze().numpy()
                yu = yu.squeeze().numpy()
                
                # Sensitivity = max deviation from nominal
                vk = np.maximum(np.abs(yl - y_pred_sens), np.abs(yu - y_pred_sens))
                
                # Cap at global sensitivity
                vk = np.minimum(vk, GS_VAL)
                V_k_mat[:, j] = vk

            # Compute S(x)
            S_vals = []
            for idx in range(len(X_sens_eval)):
                ss = []
                for j, k in enumerate(k_values):
                    if k == 0:
                        continue
                    ss.append(V_k_mat[idx, j] * np.exp(-beta * k))
                ss.append(tail_S)
                S_vals.append(max(ss))
            avg_S = np.mean(S_vals)
            
        # Compute average V_k across eval points for staircase plot
        avg_V_k = np.mean(V_k_mat, axis=0)
        
        results[method] = {
            'mse_nom': mse_nom,
            'avg_sens': avg_S,
            'V_k_mat': V_k_mat,
            'avg_V_k': avg_V_k,  # For staircase plot
            'k_values': k_values
        }
        
        print(f"\n  Results: MSE={mse_nom:.4f}, Avg Sensitivity={avg_S:.4f}")
        print(f"  V_k (avg): {[f'{v:.4f}' for k, v in zip(k_values, avg_V_k)]}")
        
        # Save per-method to avoid race conditions in parallel run
        np.save(f'{out_dir}/method_results_{method}.npy', results[method])
        print(f"Saved: {out_dir}/method_results_{method}.npy")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Expected: F2 <= F1 <= IBP (tighter bounds with more MIP)")
    print("-" * 60)
    for method, res in results.items():
        print(f"  {method.upper():4s}: MSE={res['mse_nom']:.4f}, Sensitivity={res['avg_sens']:.4f}")
    
    # ======== STAIRCASE PLOT ========
    print("\nGenerating Staircase Plot...")
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'ibp': 'blue', 
        'f1': 'cyan',
        'f2': 'green',
        'f4': 'lime', 
        'f8': 'yellow', 
        'f16': 'orange'
    }
    labels = {
        'ibp': 'IBP', 
        'f1': 'MIP F=1',
        'f2': 'MIP F=2',
        'f4': 'MIP F=4', 
        'f8': 'MIP F=8',
        'f16': 'MIP F=16'
    }
    
    # Left plot: V_k (instability) vs k
    ax1.set_xlabel('Privacy Parameter k')
    ax1.set_ylabel('Instability V_k (average)')
    ax1.set_title('Instability V_k by Method')
    
    for method, res in results.items():
        ks = res['k_values']
        vs = res['avg_V_k']
        ax1.step(ks, vs, where='post', label=labels[method], color=colors[method], linewidth=2)
        ax1.scatter(ks[1:], vs[1:], color=colors[method], s=50)  # Skip k=0
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Sensitivity curve V_k * exp(-beta*k)
    ax2.set_xlabel('Privacy Parameter t')
    ax2.set_ylabel('Sensitivity V_k × exp(-β×k)')
    ax2.set_title('Smooth Sensitivity (should differ if bounds differ)')
    
    t_grid = np.linspace(0, max(k_values) + 1, 200)
    
    for method, res in results.items():
        ks = res['k_values']
        vs = res['avg_V_k']
        
        # Interpolate V_k (step function)
        indices = np.searchsorted(ks, t_grid, side='right') - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        
        # Sensitivity curve
        s_curve = v_interp * np.exp(-beta * t_grid)
        ax2.plot(t_grid, s_curve, label=labels[method], color=colors[method], linewidth=2)
        
        # Mark maximum
        idx_max = np.argmax(s_curve)
        ax2.scatter(t_grid[idx_max], s_curve[idx_max], color=colors[method], s=100, marker='*', edgecolors='black')
        
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/staircase_comparison.png', dpi=150)
    print(f"Saved: {out_dir}/staircase_comparison.png")
    plt.close()
    
    print("\nAblation Complete.")


if __name__ == "__main__":
    run_lin_ablation()
