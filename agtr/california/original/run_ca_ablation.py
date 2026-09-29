import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import copy
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import Abstract Gradient Training Modules
from abstract_gradient_training.bounded_losses import BoundedMSELoss
from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD
from abstract_gradient_training.bounded_models import IntervalBoundedModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Plotting Style ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

# --- Data Loading ---
def load_california_data_dataset():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize targets
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    
    return train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test

# --- Model Definition ---
def create_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# --- Training Function ---
def train_ablation_models(k_values, train_dataset, epochs, lr, seed, freq, batch_size):
    # Strict Seed for Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Ensure shuffling is deterministic with respect to this seed
    g = torch.Generator()
    g.manual_seed(seed)
    
    # DataLoader with specific batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    
    model = create_model(8) 
    
    bounded_model = IntervalBoundedModel(model)
    
    optimizer = BoundedSGD(bounded_model, learning_rate=lr)
    criterion = BoundedMSELoss(reduction='sum')
    
    clip_C = 1.0
    huber_delta = 1.0
    
    accumulation_steps = freq 
    
    for epoch in range(epochs):
        accum_grads_l = None
        accum_grads_n = None
        accum_grads_u = None
        total_accum_bs = 0
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            curr_bs = x_batch.size(0)
            k = k_values[0] 
            
            # Forward
            logits_n = bounded_model.forward(x_batch, retain_intermediate=True)
            logits_input = logits_n.squeeze()
            loss_grad_n = criterion.backward(logits_input, y_batch)
            
            # Huber Loss
            loss_grad_n = torch.clamp(loss_grad_n, -huber_delta, huber_delta)
            loss_grad_n = loss_grad_n.unsqueeze(1)
            
            grads_n = bounded_model.backward(loss_grad_n)
            
            if k == 0:
                g_mean = [g.sum(dim=0) for g in grads_n] 
                
                if accum_grads_n is None:
                    accum_grads_n = g_mean
                    accum_grads_l = g_mean 
                    accum_grads_u = g_mean
                else:
                    accum_grads_n = [a + g for a, g in zip(accum_grads_n, g_mean)]
                    accum_grads_l = [a + g for a, g in zip(accum_grads_l, g_mean)]
                    accum_grads_u = [a + g for a, g in zip(accum_grads_u, g_mean)]
                    
            else:
                 # Robust Training with Clipping
                 total_norm = torch.zeros(curr_bs, device=x_batch.device)
                 for g in grads_n:
                     g_flat = g.view(curr_bs, -1)
                     total_norm += g_flat.norm(2, dim=1)**2
                 total_norm = torch.sqrt(total_norm)
                 clip_coef = torch.clamp(clip_C / (total_norm + 1e-6), max=1.0)
                 
                 grads_clipped_sum = []
                 for g in grads_n:
                      view_shape = [curr_bs] + [1] * (g.dim() - 1)
                      g_clipped = g * clip_coef.view(*view_shape)
                      grads_clipped_sum.append(g_clipped.sum(dim=0))
                      
                 if accum_grads_n is None:
                     accum_grads_n = grads_clipped_sum
                 else:
                     accum_grads_n = [a + g for a, g in zip(accum_grads_n, grads_clipped_sum)]
            
            total_accum_bs += curr_bs
            
            # Step Condition
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                grads_n_mean = [g / total_accum_bs for g in accum_grads_n]
                
                if k == 0:
                     optimizer.step(grads_n_mean, grads_n_mean, grads_n_mean)
                else:
                     delta = k * 2 * clip_C / total_accum_bs
                     grads_l = [g - delta for g in grads_n_mean]
                     grads_u = [g + delta for g in grads_n_mean]
                     optimizer.step(grads_l, grads_n_mean, grads_u)
                
                accum_grads_n = None
                accum_grads_l = None
                accum_grads_u = None
                total_accum_bs = 0

    return bounded_model

def evaluate_staircase(models_dict, x_eval_t, k_values_sorted, y_min, y_max):
    nom_model = models_dict[0]
    y_nom = nom_model.forward(x_eval_t).item()
    y_nom = np.clip(y_nom, y_min, y_max)
    
    results = []
    results.append((0, 0.0))
    
    for k in k_values_sorted:
        if k == 0: continue
        model = models_dict[k]
        y_l, y_u = model.bound_forward(x_eval_t, x_eval_t)
        y_l = y_l.item()
        y_u = y_u.item()
        
        y_l = np.clip(y_l, y_min, y_max)
        y_u = np.clip(y_u, y_min, y_max)
        
        v_k = max(abs(y_l - y_nom), abs(y_u - y_nom))
        results.append((k, v_k))
        
    return results

def run_batch_ablation():
    print("Script Started...", flush=True)
    print("Running California Housing Batch Ablation (BS=[250, 500, 1000, 2000, 4000])...", flush=True)
    
    # Output Directory & Plot Prefix
    out_dir = 'agtr/california/values/ablation'
    plot_prefix = 'ca_batch_scaled'
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Params
    # Params
    batch_sizes = [250, 500, 1000, 2000, 4000]
    base_batch_size = 1000
    
    lr = 0.01
    base_epochs = 20
    freq = 1 # Fixed frequency
    
    # Experimental Mode Toggle
    # If True: Scale epochs to keep total optimization steps constant (previous behavior).
    # If False: Constant epochs (fixed data passes). Steps will decrease as BS increases.
    FIXED_STEPS_MODE = True
    
    Y_MIN = -5.0
    Y_MAX = 5.0
    GS_VAL = Y_MAX - Y_MIN
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
    
    k_dense = [0] + list(range(1, 11)) + list(range(13, 31, 3))
    k_values = sorted(list(set(k_dense)))
    
    # Load Data
    train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np = load_california_data_dataset()
    
    # Save Raw Data
    np.save(f'{out_dir}/X_test.npy', X_test_np)
    np.save(f'{out_dir}/y_test.npy', y_test_t.numpy())
    np.save(f'{out_dir}/y_mean.npy', y_mean)
    np.save(f'{out_dir}/y_std.npy', y_std)
    np.save(f'{out_dir}/k_values.npy', k_values)
    np.save(f'{out_dir}/batch_sizes.npy', batch_sizes)
    
    x_sample_t = X_test_t[0].unsqueeze(0)
    
    plot_data_staircase = {}   
    plot_data_avg_S = {}       
    plot_data_nominal_mse = {} 
    
    # We need to re-train the nominal model for different batch sizes, because batch size 
    # affects optimization trajectory even with freq=1.
    
    for i, bs in enumerate(batch_sizes):
        print(f"Training with Batch Size: {bs}...", flush=True)
        
        if FIXED_STEPS_MODE:
            # Scale Epochs
            scaled_epochs = int(base_epochs * (bs / base_batch_size))
            if scaled_epochs < 1: scaled_epochs = 1
            mode_str = "Fixed Steps (Scaled Epochs)"
        else:
            # Fixed Epochs
            scaled_epochs = base_epochs
            mode_str = "Fixed Epochs (Constant Data Passes)"
        
        print(f"  > BS={bs}, BaseBS={base_batch_size}, Epochs={scaled_epochs} [{mode_str}]", flush=True)
        
        t0 = time.time()
        
        bs_models = {}
        
        # 1. Train Nominal for this BS
        print("    > Training Nominal...", flush=True)
        nom_model = train_ablation_models([0], train_dataset, epochs=scaled_epochs, lr=lr, seed=42, freq=freq, batch_size=bs)
        bs_models[0] = nom_model
        
        torch.save(nom_model, f'{out_dir}/model_bs{bs}.pt')
        
        # 2. Train Robust Models
        for k in k_values:
            if k == 0: continue 
            m = train_ablation_models([k], train_dataset, epochs=scaled_epochs, lr=lr, seed=42, freq=freq, batch_size=bs)
            bs_models[k] = m
            
        t1 = time.time()
        total_time = t1 - t0
        print(f"BatchSize={bs} | Total Training Time: {total_time:.2f}s", flush=True)
        np.save(f'{out_dir}/time_bs{bs}.npy', total_time)
        
        # --- Evaluation & Saving ---
        with torch.no_grad():
             logits = nom_model.forward(X_test_t)
             y_pred_cont = logits.squeeze().numpy()
             y_pred_cont = np.clip(y_pred_cont, Y_MIN, Y_MAX)
             
             # SAVE NOMINAL PREDS
             np.save(f'{out_dir}/nominal_preds_bs{bs}.npy', y_pred_cont)
             
             y_pred_real = y_pred_cont * y_std + y_mean
             y_test_real = y_test_t.numpy() * y_std + y_mean
             mse_val = np.mean((y_pred_real - y_test_real)**2)
             plot_data_nominal_mse[bs] = mse_val
             print(f"BatchSize={bs} | Nominal MSE: {mse_val:.4f}", flush=True)
             
             beta = 1.0 / 6.0
             max_k_eval = max(k_values)
             tail_S = GS_VAL * np.exp(-beta * (max_k_eval + 1))
             
             n_test = X_test_t.size(0)
             n_k = len(k_values)
             
             Y_L_ALL = np.zeros((n_test, n_k))
             Y_U_ALL = np.zeros((n_test, n_k))
             V_all = np.zeros((n_test, n_k))
             
             for j, k in enumerate(k_values):
                 if k == 0: 
                     Y_L_ALL[:, j] = y_pred_cont
                     Y_U_ALL[:, j] = y_pred_cont
                     continue
                     
                 model_k = bs_models[k]
                 y_l, y_u = model_k.bound_forward(X_test_t, X_test_t)
                 y_l_np = np.clip(y_l.flatten().numpy(), Y_MIN, Y_MAX)
                 y_u_np = np.clip(y_u.flatten().numpy(), Y_MIN, Y_MAX)
                 
                 Y_L_ALL[:, j] = y_l_np
                 Y_U_ALL[:, j] = y_u_np
                 
                 v_k = np.maximum(np.abs(y_l_np - y_pred_cont), np.abs(y_u_np - y_pred_cont))
                 V_all[:, j] = v_k
                 
             # SAVE BOUNDS
             np.save(f'{out_dir}/bounds_bs{bs}_yl.npy', Y_L_ALL)
             np.save(f'{out_dir}/bounds_bs{bs}_yu.npy', Y_U_ALL)
                 
             S_list = []
             for idx in range(n_test):
                 s_vals = []
                 for j, k in enumerate(k_values):
                     if k == 0: continue
                     s = V_all[idx, j] * np.exp(-beta * k)
                     s_vals.append(s)
                 s_vals.append(tail_S)
                 S_list.append(max(s_vals))
             
             S_vec_norm = np.array(S_list)
             S_vec_real = S_vec_norm * y_std
             np.save(f'{out_dir}/sensitivity_bs{bs}.npy', S_vec_real)
             
             plot_data_avg_S[bs] = np.mean(S_vec_real)
             print(f"BatchSize={bs} | Avg Sensitivity (Real): {np.mean(S_vec_real):.4f}", flush=True)
             
        # Staircase Plot Data (Single Sample)
        res_f = evaluate_staircase(bs_models, x_sample_t, k_values, Y_MIN, Y_MAX)
        plot_data_staircase[bs] = zip(*res_f)


    # --- PLOTTING (Matched to Lin Ablation Style) ---
    print("Generating Plots...", flush=True)
    
    # Plot 1: Instability & Smooth Sensitivity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    beta = 1.0 / 6.0
    max_k_val = max(k_values)
    t_grid = np.linspace(0, max_k_val + 5, 500)
    
    ax1.set_xlabel('Privacy Parameter k')
    ax1.set_ylabel('Instability V_k')
    ax2.set_ylabel('Sensitivity S(t)')
    ax1.set_title('Instability & Smooth Sensitivity') 
    
    for i, bs in enumerate(batch_sizes):
        ks_b, vs_b = plot_data_staircase[bs]
        ks_b = list(ks_b)
        vs_b = list(vs_b)
        c = colors[i % len(colors)]
        
        np.save(f'{out_dir}/staircase_bs{bs}_ks.npy', ks_b)
        np.save(f'{out_dir}/staircase_bs{bs}_vs.npy', vs_b)
        
        ax1.step(ks_b, vs_b, where='post', label=f'BS={bs}', color=c, linewidth=2, alpha=0.8)
        
        indices = np.searchsorted(ks_b, t_grid, side='right') - 1
        indices = np.clip(indices, 0, len(ks_b) - 1)
        v_interp = np.array([vs_b[idx] for idx in indices])
        s_curve = v_interp * np.exp(-beta * t_grid)
        
        ax2.plot(t_grid, s_curve, '--', color=c, linewidth=1.5, alpha=0.8)
        
        idx_max = np.argmax(s_curve)
        t_max = t_grid[idx_max]
        s_max = s_curve[idx_max]
        ax2.plot(t_max, s_max, '*', color=c, markersize=15, markeredgecolor='black')

    ax1.legend(loc='upper right', title="Batch Size")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/california/figures/{plot_prefix}_staircase.png', dpi=300)
    plt.close()
    
    # Plot 2b: Average Sensitivity Bar Chart
    plt.figure(figsize=(10, 6))
    
    bar_bs = []
    bar_sens = []
    
    for bs in batch_sizes:
         if bs in plot_data_avg_S:
             bar_bs.append(str(bs))
             bar_sens.append(plot_data_avg_S[bs])
             
    plt.bar(bar_bs, bar_sens, color=colors[:len(bar_bs)], alpha=0.8, edgecolor='black')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Smooth Sensitivity')
    plt.title('Average Smooth Sensitivity by Batch Size')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/california/figures/{plot_prefix}_avg_sensitivity_bar.png', dpi=300)
    plt.close()

    # Trade-offs Comparison
    epsilon_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 500 
    
    # Plot 3: Approx DP
    plt.figure(figsize=(10, 8))
    delta_dp = 1e-5
    factor = np.sqrt(2 * np.log(1/delta_dp))
    GS_real = GS_VAL * y_std
    
    gs_sigma_vals = []
    for eps in epsilon_grid:
        gs_sigma_vals.append((GS_real * factor / eps)**2)
    plt.plot(epsilon_grid, gs_sigma_vals, 'o--', label='Global Sensitivity', color='gray', alpha=0.5)
    
    for i, bs in enumerate(batch_sizes):
        nom_mse = plot_data_nominal_mse[bs]
        avg_S = plot_data_avg_S[bs]
        c = colors[i % len(colors)]
        mse_mean = []
        mse_std = []
        for eps in epsilon_grid:
            sigma = avg_S * factor / eps
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.normal(0, sigma, len(y_test_t))
                mse_trials.append(nom_mse + np.mean(noise**2))
            mse_mean.append(np.mean(mse_trials))
            mse_std.append(np.std(mse_trials))
        mse_mean = np.array(mse_mean)
        mse_std = np.array(mse_std)
        plt.plot(epsilon_grid, mse_mean, 'o-', label=f'BS={bs}', color=c)
        plt.fill_between(epsilon_grid, mse_mean - 3*mse_std, mse_mean + 3*mse_std, color=c, alpha=0.1)
    
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE (Approx DP)')
    plt.ylim(0, 1.0) # MSE is usually < 1.0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/california/figures/{plot_prefix}_tradeoff_approx.png', dpi=300)
    plt.close()

    # Plot 4: Pure DP
    plt.figure(figsize=(10, 8))
    gs_mse_median = []
    gs_mse_p25 = []
    gs_mse_p75 = []
    for eps in epsilon_grid:
        scale_gs = GS_real / eps
        mse_trials = []
        for _ in range(n_trials):
            noise = np.random.standard_cauchy(len(y_test_t)) * scale_gs
            mse_trials.append(np.mean(noise**2))
        gs_mse_median.append(np.median(mse_trials))
        gs_mse_p25.append(np.percentile(mse_trials, 25))
        gs_mse_p75.append(np.percentile(mse_trials, 75))
    gs_mse_median = np.array(gs_mse_median)
    gs_mse_p25 = np.array(gs_mse_p25)
    gs_mse_p75 = np.array(gs_mse_p75)
    plt.plot(epsilon_grid, gs_mse_median, 'o--', label='Global Sensitivity', color='gray', alpha=0.5)
    plt.fill_between(epsilon_grid, gs_mse_p25, gs_mse_p75, color='gray', alpha=0.1)
    
    for i, bs in enumerate(batch_sizes):
        nom_mse = plot_data_nominal_mse[bs]
        avg_S = plot_data_avg_S[bs]
        c = colors[i % len(colors)]
        mse_median = []
        mse_p25 = []
        mse_p75 = []
        for eps in epsilon_grid:
            scale_ss = avg_S / eps
            mse_trials = []
            for _ in range(n_trials):
                noise = np.random.standard_cauchy(len(y_test_t)) * scale_ss
                mse_trials.append(nom_mse + np.mean(noise**2))
            mse_median.append(np.median(mse_trials))
            mse_p25.append(np.percentile(mse_trials, 25))
            mse_p75.append(np.percentile(mse_trials, 75))
        mse_median = np.array(mse_median)
        mse_p25 = np.array(mse_p25)
        mse_p75 = np.array(mse_p75)
        plt.plot(epsilon_grid, mse_median, 'o-', label=f'BS={bs}', color=c)
        plt.fill_between(epsilon_grid, mse_p25, mse_p75, color=c, alpha=0.1)
        
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE (Pure DP)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/california/figures/{plot_prefix}_tradeoff_pure.png', dpi=300)
    plt.close()    
    print("Batch Ablation Complete. All data saved.", flush=True)

if __name__ == "__main__":
    run_batch_ablation()
