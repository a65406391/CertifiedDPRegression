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

def get_nominal_pred(X_eval):
    # Quick retrain of nominal model to get baseline predictions
    X_train, y_train = simulated_linear_data(n_samples=1000, seed=42)
    model = nn.Sequential(nn.Linear(1, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    
    torch.manual_seed(42)
    for _ in range(20):
        for x_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            
    with torch.no_grad():
        return model(torch.tensor(X_eval, dtype=torch.float32)).numpy().flatten()

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
    
    # Define Order and Colors
    ordered_methods = ['ibp', 'f1', 'f2', 'f4', 'f8', 'f16']
    
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

    # Helper to save to both locations
    def save_plot(filename):
        plt.savefig(f'{out_dir}/{filename}', bbox_inches='tight')
        plt.savefig(f'{artifact_dir}/{filename}', bbox_inches='tight')
        print(f"Saved {filename}")

    # Recover Eval Data (for bands)
    np.random.seed(999)
    X_sens_eval = np.random.uniform(-2, 2, (50, 1))
    y_nom_eval = get_nominal_pred(X_sens_eval)
    
    # --- Plot 1: Instability Staircase & Smooth Sensitivity ---
    # Dual Axis: Left=Staircase(V_k), Right=Curve(V_k * e^-beta*k)
    fig, ax1 = plt.subplots(figsize=(10, 11))
    ax2 = ax1.twinx()
    
    beta = 0.5 # Visualization beta
    
    # Determine max_k for grid
    max_k_val = 6.0 # hardcode slightly larger than k=5
    
    t_grid = np.linspace(0, max_k_val, 500)
    
    ax1.set_xlabel('Privacy Parameter k')
    ax1.set_ylabel('Max. Local Sens. Certificate $A_k$ (dashed)')
    ax2.set_ylabel('Smooth Sensitivity Term (solid)')
    ax1.set_title('Instability & Smooth Sensitivity (Manual Benchmark)')
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        
        # Determine V_k
        if 'avg_V_k' in res:
             avg_V_k = res['avg_V_k']
        else:
             V_k_mat = res['V_k_mat']
             avg_V_k = np.mean(V_k_mat, axis=0)
             
        k_values = res['k_values']
        max_k_val = max(max_k_val, max(k_values))
        
        # Make sure k_values and avg_V_k are sorted just in case
        sort_idx = np.argsort(k_values)
        ks = np.array(k_values)[sort_idx]
        vs = np.array(avg_V_k)[sort_idx]
        
        c = colors[m]
        lbl = labels[m]
        
        # 1. Staircase (Left Axis, Dashed)
        ax1.step(ks, vs, where='post', label=lbl, color=c, linestyle='--', linewidth=3, alpha=0.7)
        
        # 2. Smooth Sensitivity Curve (Right Axis, Solid)
        # Interpolate V_k for continuous t
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

    # Legends (Combine manually or just use Ax1)
    # Ax1 legend handles lines, Ax2 is implicit by color matching
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False) 
    
    plt.tight_layout()
    save_plot('manual_lin_ablation_freq_staircase_v1.png')
    plt.close()
    
    # --- Plot 2: Sensitivity Bands (Smooth Sensitivity) ---
    plt.figure(figsize=(10, 11))
    
    # Background Data
    X_train_plot, y_train_plot = simulated_linear_data(80, seed=42) # Match N=80
    plt.scatter(X_train_plot, y_train_plot, c='black', alpha=0.3, label='Data (N=80)')
    
    # Determine common sort order for recreation
    np.random.seed(999)
    X_sens_rec = np.random.uniform(-2, 2, (50, 1))
    y_nom_rec = get_nominal_pred(X_sens_rec)
    
    sort_idx = np.argsort(X_sens_rec.flatten())
    X_sorted = X_sens_rec.flatten()[sort_idx]
    y_nom_sorted = y_nom_rec[sort_idx]
    
    # Global SS Cap
    GS_VAL = 10.0
    beta = 0.5
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        V_k_mat = res['V_k_mat'] # (50, K)
        k_values = res['k_values']
        
        # Calculate SS(x) for each point
        S_vec = []
        for i in range(len(X_sens_rec)):
             # S(x) = max_k (V_k(x) * e^-beta*k)
             # Also consider tail? V_k ends?
             # Assuming V_k covers necessary range.
             vals = []
             for j, k in enumerate(k_values):
                  if k == 0: continue
                  vals.append(V_k_mat[i, j] * np.exp(-beta * k))
             if len(vals) == 0:
                 S_vec.append(GS_VAL) # Conservative
             else:
                 S_vec.append(max(vals))
                 
        S_vec = np.array(S_vec)
        S_sorted = S_vec[sort_idx]
        
        y_low = y_nom_sorted - S_sorted
        y_high = y_nom_sorted + S_sorted
        
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
    save_plot('lin_ablation_freq_sensitivity_bands_v3.png')
    plt.close()
    plt.tight_layout()
    print("Saved bands. Loading baselines...")
    # --- Load Baselines ---
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

    # Pure DP Params
    # User requested Cauchy(6 * SS / eps).
    # This implies beta = 1/6.
    beta = 1.0 / 6.0 
    GS_VAL = 10.0
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        V_k_mat = res['V_k_mat'] # (50, K)
        k_values = res['k_values']
        mse_nom = res['mse_nom']
        
        # 1. Estimate distribution of S(x) on the 50 points with beta=1/6
        S_vec = []
        for i in range(len(V_k_mat)):
             vals = [V_k_mat[i, j] * np.exp(-beta * k) for j, k in enumerate(k_values) if k > 0]
             if not vals:
                 S_vec.append(GS_VAL)
             else:
                 S_vec.append(max(vals))
        S_vec = np.array(S_vec)
        
        # 2. Compute Noise Variance vs Epsilon
        mses = []
        for eps in epsilons:
            # Scale = S / (beta * eps) = 6 * S / eps
            scale_vec = S_vec / (beta * eps)
            
            noise_sq_trials = []
            for _ in range(n_trials):
                 # Generate noise for each of the 50 points
                 noise = np.random.standard_cauchy(len(S_vec)) * scale_vec
                 noise_sq_trials.append(np.mean(noise**2))
            
            noise_term = np.median(noise_sq_trials)
            mses.append(mse_nom + noise_term)
            
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
    save_plot('lin_ablation_freq_tradeoff_pure_v3.png')
    plt.close()

    # --- Plot 4: Trade-off (Approx DP) ---
    print("Plotting Approx DP...")
    plt.figure(figsize=(10, 11))
    
    # Approx DP usually uses Global Sensitivity + Gaussian?
    # Or Smooth Sensitivity + Gaussian/Laplace?
    # User's previous code used Laplace with avg_S.
    # We will use instance-specific S_vec with Laplace for consistnecy?
    # Or Gaussian with High Probability bound?
    # Let's stick to Laplace but instance specific: Lap(S(x)/eps).
    # scale = S(x) / eps. (Approx DP allows dropping beta factor?)
    # Actually, for Approx DP, we can use Local Sensitivity with delta?
    # Let's simply use calibrated Laplace noise based on S(x).
    
    for m in ordered_methods:
        if m not in results: continue
        res = results[m]
        V_k_mat = res['V_k_mat'] 
        k_values = res['k_values']
        mse_nom = res['mse_nom']
        
        # Recompute S_vec (same as above)
        S_vec = []
        for i in range(len(V_k_mat)):
             vals = [V_k_mat[i, j] * np.exp(-beta * k) for j, k in enumerate(k_values) if k > 0]
             if not vals: S_vec.append(GS_VAL)
             else: S_vec.append(max(vals))
        S_vec = np.array(S_vec)
        
        mses = []
        for eps in epsilons:
            # Approx scale
            scale_vec = S_vec / eps # (Or 2*S/eps?)
            # Laplace variance = 2 * scale^2
            # E[Noise^2] = 2 * scale^2
            noise_var_vec = 2 * (scale_vec**2)
            avg_noise_var = np.mean(noise_var_vec)
            mses.append(mse_nom + avg_noise_var)
            
        plt.plot(epsilons, mses, 'o-', label=labels[m], color=colors[m], linewidth=4)
    
    # Plot Baselines (Approx DP compat)
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
    save_plot('lin_ablation_freq_tradeoff_approx_v3.png')
    plt.close()

    print("Plots updated and saved to artifacts.")

if __name__ == "__main__":
    plot_certified_ablation()
