import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Style
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

def simulated_linear_data(n_samples=1000, slope=2, intercept=1, x_range=(-2, 2), seed=42):
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    noise = np.random.normal(0, 0.1, (n_samples, 1)) 
    y = slope * X + intercept + noise
    return X, y

def plot_certified_ablation():
    print("Plotting Certified Ablation (Paper Quality)...")
    in_dir = 'agtr/linear/values/ablation'
    out_dir = 'agtr/linear/figures/linear'
    artifact_dir = 'agtr/linear/figures/linear'
    os.makedirs(out_dir, exist_ok=True)
    
    # Updated input file
    res_file = f'{in_dir}/manual_results.npy'
    if not os.path.exists(res_file):
        print(f"Results file not found at {res_file}")
        return
        
    results = np.load(res_file, allow_pickle=True).item()
    
    # Standard Ablation Colors (Tab10)
    colors = {
        'ibp': 'tab:brown', 
        'f1': 'tab:blue',
        'f2': 'tab:orange',
        'f4': 'tab:green', 
        'f8': 'tab:red', 
        'f16': 'tab:purple'
    }
    
    ordered_methods = ['ibp', 'f1', 'f2', 'f4', 'f8', 'f16']
    
    labels = {
        'ibp': 'IBP', 
        'f1': 'AGT (F=1)',
        'f2': 'AGT (F=2)',
        'f4': 'AGT (F=4)', 
        'f8': 'AGT (F=8)',
        'f16': 'AGT (F=16)'
    }

    # Helper to save to both locations
    def save_plot(filename):
        plt.savefig(f'{out_dir}/{filename}', bbox_inches='tight')
        plt.savefig(f'{artifact_dir}/{filename}', bbox_inches='tight')
        print(f"Saved {filename}")

    # Recover Eval Data AND Compute Nominal MSE for Manual Model
    np.random.seed(999)
    # Range -2, 2
    X_sens_eval = np.random.uniform(-2, 2, (50, 1))
    
    # Re-init manual model to get w0, b0 (Seed 42)
    # Same as "Manual Run" setup
    torch.manual_seed(42)
    mod_dummy = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    w0_nom = mod_dummy[0].weight.item()
    b0_nom = mod_dummy[0].bias.item()
    
    def get_nominal_pred(x_arr):
        return w0_nom * x_arr + b0_nom

    y_nom_eval = get_nominal_pred(X_sens_eval)
    
    # Compute Nominal Test MSE (Manual Model)
    # Test set: 1000 points
    np.random.seed(12345)
    X_test_mse = np.random.uniform(-2, 2, (1000, 1))
    noise_test = np.random.normal(0, 0.1, (1000, 1))
    # True Data Gen: y = 0.5*x + 1 + noise
    y_test_true = 0.5 * X_test_mse + 1.0 + noise_test
    
    y_test_pred = get_nominal_pred(X_test_mse)
    manual_mse_nom = np.mean((y_test_pred - y_test_true)**2)
    print(f"Manual Model Nominal MSE: {manual_mse_nom:.6f}")
    
    # --- Plot 1: Instability Staircase & Smooth Sensitivity ---
    # Dual Axis: Left=Staircase(V_k), Right=Curve(V_k * e^-beta*k)
    fig, ax1 = plt.subplots(figsize=(10, 11))
    ax2 = ax1.twinx()
    
    beta = 1.0 / 6.0 
    
    # Determine max_k for grid
    max_k_val = 6.0 
    
    t_grid = np.linspace(0, max_k_val, 500)
    
    ax1.set_xlabel('Privacy Parameter k')
    ax1.set_ylabel('Max. Local Sens. Certificate $A_k$ (dashed)')
    ax2.set_ylabel('Smooth Sensitivity Term (solid)')
    ax1.set_title('Instability & Smooth Sensitivity')
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        
        # Determine V_k
        if 'avg_V_k' in res:
             avg_V_k = res['avg_V_k']
        else:
             continue
             
        k_values = res['k_values']
        max_k_val = max(max_k_val, max(k_values))
        
        sort_idx = np.argsort(k_values)
        ks = np.array(k_values)[sort_idx]
        vs = np.array(avg_V_k)[sort_idx]
        
        c = colors[m]
        lbl = labels[m]
        
        # 1. Staircase
        ax1.step(ks, vs, where='post', label=lbl, color=c, linestyle='--', linewidth=3, alpha=0.7)
        
        # 2. Smooth Sensitivity Curve
        indices = np.searchsorted(ks, t_grid, side='right') - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        
        s_curve = v_interp * np.exp(-beta * t_grid)
        
        ax2.plot(t_grid, s_curve, '-', color=c, linewidth=4, alpha=0.9)
        
        # 3. Star at Max
        idx_max = np.argmax(s_curve)
        t_max = t_grid[idx_max]
        s_max = s_curve[idx_max]
        ax2.plot(t_max, s_max, '*', color=c, markersize=24, markeredgecolor='black', zorder=10)

    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False) 
    
    plt.tight_layout()
    save_plot('lin_ablation_final_manual_staircase.png')
    plt.close()
    
    # --- Plot 2: Sensitivity Bands ---
    plt.figure(figsize=(10, 6))
    
    sort_idx = np.argsort(X_sens_eval.flatten())
    X_sorted = X_sens_eval.flatten()[sort_idx]
    y_nom_sorted = y_nom_eval.flatten()[sort_idx]
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        
        ks = res['k_values']
        avg_V_k = res['avg_V_k']
        
        # S scalar
        S_vals = [avg_V_k[i] * np.exp(-beta * ks[i]) for i in range(len(ks))]
        S_final_W = max(S_vals)
        
        # Band = S_W * |x|
        band_width = S_final_W * np.abs(X_sorted)
        
        y_low = y_nom_sorted - band_width
        y_high = y_nom_sorted + band_width
        
        c = colors[m]
        plt.fill_between(X_sorted, y_low, y_high, color=c, alpha=0.1, label=labels[m])
        plt.plot(X_sorted, y_low, color=c, alpha=0.4, linewidth=1)
        plt.plot(X_sorted, y_high, color=c, alpha=0.4, linewidth=1)
        
    plt.xlabel('Input x')
    plt.ylabel('Prediction $\pm$ Smooth Sensitivity')
    plt.title('Smooth Sensitivity Bands')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('lin_ablation_final_manual_bands.png')
    plt.close()
    
    print("Saved bands. Computing Tradeoffs...")

    # Load Baselines
    base_dir = 'agtr/linear/values/baselines'
    baselines = {}
    try:
        if os.path.exists(f'{base_dir}/shard20_mean.npy'):
            baselines['shard20'] = np.load(f'{base_dir}/shard20_mean.npy')
            print("Loaded shard20")
        if os.path.exists(f'{base_dir}/gauss_mean.npy'):
            baselines['gauss'] = np.load(f'{base_dir}/gauss_mean.npy')
            print("Loaded gauss")
        if os.path.exists(f'{base_dir}/func_mean.npy'):
            baselines['func'] = np.load(f'{base_dir}/func_mean.npy')
            print("Loaded func")
    except Exception as e:
        print(f"Error loading baselines: {e}")
    
    epsilons_base = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    epsilons = epsilons_base
    n_trials = 100

    # --- Plot 3: Trade-off (Pure DP) ---
    print("Plotting Pure DP...")
    plt.figure(figsize=(10, 11))

    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        avg_V_k = res['avg_V_k']
        ks = res['k_values']
        
        S_vals = [avg_V_k[i] * np.exp(-beta * ks[i]) for i in range(len(ks))]
        S_W = max(S_vals)
        
        mses = []
        for eps in epsilons:
            scale = S_W / (beta * eps)
            noise_W = np.random.standard_cauchy(n_trials * 10) * scale 
            noise_mses = []
            current_noises = noise_W[:n_trials]
            for eta in current_noises:
                 resid = y_test_pred.flatten() - y_test_true.flatten()
                 impact = eta * X_test_mse.flatten()
                 full_err = resid + impact
                 noise_mses.append(np.mean(full_err**2))
            mses.append(np.median(noise_mses))
            
        plt.plot(epsilons, mses, 'o-', label=labels[m], color=colors[m], linewidth=4)
        
    # Plot Baselines (Pure DP)
    if 'shard20' in baselines:
         plt.plot(epsilons_base, baselines['shard20'], 's--', label='Shard (S=20)', color='black', linewidth=3)
    if 'func' in baselines:
         plt.plot(epsilons_base, baselines['func'], '^--', label='Functional', color='gray', linewidth=3)
        
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE (Pure DP)')
    plt.title('Freq Trade-off (Pure DP)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('lin_ablation_final_manual_tradeoff_pure.png')
    plt.close()

    # --- Plot 4: Trade-off (Approx DP) ---
    print("Plotting Approx DP...")
    plt.figure(figsize=(10, 11))
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        avg_V_k = res['avg_V_k']
        ks = res['k_values']
        
        S_vals = [avg_V_k[i] * np.exp(-beta * ks[i]) for i in range(len(ks))]
        S_W = max(S_vals)
        
        mses = []
        mean_scale_sq = 1.3333 # E[x^2]
        
        for eps in epsilons:
            scale = S_W / eps
            noise_term = (2 * (scale**2)) * mean_scale_sq
            mses.append(manual_mse_nom + noise_term)
            
        plt.plot(epsilons, mses, 'o-', label=labels[m], color=colors[m], linewidth=4)
    
    # Plot Baselines (Approx DP)
    if 'shard20' in baselines:
         plt.plot(epsilons_base, baselines['shard20'], 's--', label='Shard (S=20)', color='black', linewidth=3)
    if 'gauss' in baselines:
         plt.plot(epsilons_base, baselines['gauss'], 'D--', label='Gaussian SGD', color='gray', linewidth=3)

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE (Approx DP)')
    plt.title('Freq Trade-off (Approx DP)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('lin_ablation_final_manual_tradeoff_approx.png')
    plt.close()
    
    print("All Manual Plots Saved Successfully.")

if __name__ == "__main__":
    plot_certified_ablation()
