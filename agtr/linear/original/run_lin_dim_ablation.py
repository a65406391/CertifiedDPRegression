import torch
import torch.nn as nn
import numpy as np
import os
import logging
from torch.utils.data import TensorDataset, DataLoader
from opacus import GradSampleModule

# Import AGT (MIPBoundedModel)
from abstract_gradient_training.bounded_losses import BoundedMSELoss
from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD
from abstract_gradient_training.bounded_models.mip_bounded_model import MIPBoundedModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 50
LR = 0.05
CLIP_X = 2.0
CLIP_Y = 6.0 

def simulated_high_dim_data(n_samples, d, seed=42):
    np.random.seed(seed)
    # X ~ U(-2, 2)
    X = np.random.uniform(-2, 2, (n_samples, d))
    
    # Weights scaled by 2/sqrt(d)
    w_val = 2.0 / np.sqrt(d)
    w = np.full((d, 1), w_val)
    
    noise = np.random.normal(0, 0.1, (n_samples, 1))
    y = X @ w + 1.0 + noise
    return X, y

def get_dataloader(n_samples, d, seed=42):
    X, y = simulated_high_dim_data(n_samples, d, seed=seed)
    X = np.clip(X, -CLIP_X, CLIP_X)
    y = np.clip(y, -CLIP_Y, CLIP_Y)
    
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

# --- AGT Training ---

def train_agt_model(train_loader, d, epochs, lr, seed, freq, k):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = nn.Sequential(nn.Linear(d, 1))
    
    bounded_model = MIPBoundedModel(
        model, 
        forward_bound_mode='miqp', 
        optimize_inter=True,
        # CRITICAL: Concretization Frequency = F (freq)
        concretization_frequency=freq 
    )
    
    optimizer = BoundedSGD(bounded_model, learning_rate=lr)
    criterion = BoundedMSELoss(reduction='sum')
    
    for epoch in range(epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            curr_bs = x_batch.size(0)
            
            # --- Standard SGD Step within AGT ---
            # We calculate gradients and apply bounds (interval training) every step.
            # MIPBoundedModel handles the deferred concretization internally via 'concretization_frequency'.
            
            if k == 0:
                logits = bounded_model.forward(x_batch, retain_intermediate=True)
                loss_g = criterion.backward(logits, y_batch)
                grads = bounded_model.backward(loss_g)
                # Sum over batch
                grads_n_sum = [g.sum(dim=0) for g in grads]
                # Mean
                grads_n_mean = [g / curr_bs for g in grads_n_sum]
                
                optimizer.step(grads_n_mean, grads_n_mean, grads_n_mean)
                    
            else:
                 logits = bounded_model.forward(x_batch, retain_intermediate=True)
                 loss_g = criterion.backward(logits, y_batch)
                 grads = bounded_model.backward(loss_g)
                 
                 # Norm clipping (Per sample)
                 total_norm = torch.zeros(curr_bs, device=x_batch.device)
                 for g in grads:
                     g_flat = g.view(curr_bs, -1)
                     total_norm += g_flat.norm(2, dim=1)**2
                 total_norm = torch.sqrt(total_norm)
                 clip_coef = torch.clamp(1.0 / (total_norm + 1e-6), max=1.0)
                 
                 grads_clipped_sum = []
                 for g in grads:
                      view_shape = [curr_bs] + [1] * (g.dim() - 1)
                      g_clipped = g * clip_coef.view(*view_shape)
                      grads_clipped_sum.append(g_clipped.sum(dim=0))
                 
                 # Mean Gradients (Nominal)
                 grads_n_mean = [g / curr_bs for g in grads_clipped_sum]
                 
                 # Bounds on Gradients
                 # Delta = k * 2 / B (Interval Training Noise)
                 delta = k * 2.0 * 1.0 / curr_bs
                 grads_l = [g - delta for g in grads_n_mean]
                 grads_u = [g + delta for g in grads_n_mean]
                 
                 optimizer.step(grads_l, grads_n_mean, grads_u)

    return bounded_model

# --- Baselines ---

def run_functional_mechanism(n_samples, d, epsilon, seed=42):
    np.random.seed(seed)
    X, y = simulated_high_dim_data(n_samples, d, seed=seed)
    X = np.clip(X, -CLIP_X, CLIP_X)
    y = np.clip(y, -CLIP_Y, CLIP_Y)
    X_b = np.c_[X, np.ones(len(X))]
    d_full = d + 1
    XTX = X_b.T @ X_b
    XTY = X_b.T @ y
    R2 = d * (CLIP_X**2) + 1.0
    s_xtx = R2
    eps_1 = epsilon / 2.0
    eps_2 = epsilon / 2.0
    noise_xtx = np.random.laplace(0, s_xtx / eps_1, (d_full, d_full))
    noise_xtx = np.triu(noise_xtx) + np.triu(noise_xtx, 1).T
    XTX_p = XTX + noise_xtx
    s_xty = np.sqrt(R2) * CLIP_Y
    noise_xty = np.random.laplace(0, s_xty / eps_2, (d_full, 1))
    XTY_p = XTY + noise_xty
    try:
        theta = np.linalg.solve(XTX_p, XTY_p)
    except np.linalg.LinAlgError:
        theta = np.zeros((d_full, 1))
    X_test, y_test = simulated_high_dim_data(2000, d, seed=100) 
    X_test_b = np.c_[X_test, np.ones(len(X_test))]
    y_pred = X_test_b @ theta
    mse = np.mean((y_pred - y_test)**2)
    return mse

def run_laplace_sgd(n_samples, d, epsilon, seed=42):
    torch.manual_seed(seed)
    loader = get_dataloader(n_samples, d, seed=seed)
    model = nn.Linear(d, 1)
    model = GradSampleModule(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean')
    EPOCHS = 20
    TOTAL_STEPS = len(loader) * EPOCHS
    C = 1.0 
    if epsilon > 100: noise_scale = 0
    else:
        eps_step = epsilon / TOTAL_STEPS
        noise_scale = 2 * C / eps_step
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            with torch.no_grad():
                params = list(model.parameters())
                batch_size = X_batch.shape[0]
                per_sample_norms_sq = torch.zeros(batch_size)
                for p in params:
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        if isinstance(p.grad_sample, list): p_grad_sample = torch.cat(p.grad_sample, dim=0)
                        else: p_grad_sample = p.grad_sample
                        flat_g = p_grad_sample.reshape(batch_size, -1)
                        per_sample_norms_sq += flat_g.norm(2, dim=1)**2
                per_sample_norms = torch.sqrt(per_sample_norms_sq)
                clip_factors = torch.clamp(C / (per_sample_norms + 1e-6), max=1.0)
                for p in params:
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        if isinstance(p.grad_sample, list): p_grad_sample = torch.cat(p.grad_sample, dim=0)
                        else: p_grad_sample = p.grad_sample
                        view_shape = [batch_size] + [1] * (p_grad_sample.ndim - 1)
                        clipped_grads = p_grad_sample * clip_factors.view(*view_shape)
                        grad_sum = clipped_grads.sum(dim=0)
                        if noise_scale > 0:
                            noise = np.random.laplace(0, noise_scale, p.shape)
                            grad_sum += torch.tensor(noise, dtype=torch.float32)
                        p.grad = grad_sum / batch_size
                        p.grad_sample = None
            optimizer.step()
    X_test, y_test = simulated_high_dim_data(2000, d, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad(): preds = model(X_test_t).numpy()
    mse = np.mean((preds - y_test)**2)
    return mse

# --- Main ---

def run_dim_ablation():
    print("Running Dimension Ablation (d=10, 100, 1000)...")
    out_dir = 'agtr/linear/values/dim_ablation'
    os.makedirs(out_dir, exist_ok=True)
    
    dims = [10, 100, 1000] # Added 1000
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    Y_MIN = -5.0
    Y_MAX = 7.0
    GS_VAL = 12.0
    beta = 1.0 / 6.0
    
    k_dense = [0] + list(range(1, 11)) + list(range(12, 36, 2))
    k_values = sorted(list(set(k_dense)))
    max_k_eval = max(k_values)
    tail_S = GS_VAL * np.exp(-beta * (max_k_eval + 1))
    
    results = {} 
    
    for d in dims:
        results[d] = {}
        print(f"\n=== Dimension {d} ===")
        
        # d=1000 has special N=2500
        if d == 1000:
             n_samples_list = [250, 500, 1000, 2500]
        else:
             n_samples_list = [250, 500, 1000]

        X_test, y_test = simulated_high_dim_data(2000, d, seed=100)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        np.random.seed(999)
        X_sens_eval = np.random.uniform(-2, 2, (50, d))
        X_sens_eval_t = torch.tensor(X_sens_eval, dtype=torch.float32)
        
        for n_samples in n_samples_list:
            results[d][n_samples] = {}
            freq = int(n_samples / BATCH_SIZE)
            print(f"  N={n_samples} (F={freq})")
            
            # 1. AGT
            print("    Running AGT...")
            # Use standard 20 Epochs. Freq passed as concretization freq.
            X_train, y_train = simulated_high_dim_data(n_samples, d, seed=42)
            dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            freq_models = {}
            for k in k_values:
                # Epochs = 20. No scaling! Process A fixed.
                # Freq passed to MIPBoundedModel
                m = train_agt_model(loader, d, epochs=20, lr=LR, seed=42, freq=freq, k=k)
                freq_models[k] = m
                
            nom_model = freq_models[0]
            with torch.no_grad():
                test_pred = nom_model.forward(X_test_t).squeeze().numpy()
                mse_nom = np.mean((test_pred - y_test.flatten())**2)
                
                y_pred_sens = nom_model.forward(X_sens_eval_t).squeeze().numpy()
                V_k_mat = np.zeros((len(X_sens_eval), len(k_values)))
                for j, k in enumerate(k_values):
                     if k == 0: continue
                     mk = freq_models[k]
                     yl, yu = mk.bound_forward(X_sens_eval_t, X_sens_eval_t)
                     yl = yl.squeeze().numpy()
                     yu = yu.squeeze().numpy()
                     vk = np.maximum(np.abs(yl - y_pred_sens), np.abs(yu - y_pred_sens))
                     V_k_mat[:, j] = vk
                S_vals = []
                for idx in range(len(X_sens_eval)):
                    ss = []
                    for j, k in enumerate(k_values):
                        if k == 0: continue
                        ss.append(V_k_mat[idx, j] * np.exp(-beta * k))
                    ss.append(tail_S)
                    S_vals.append(max(ss))
                avg_S = np.mean(S_vals)
                
            agt_mses = []
            n_trials = 100 
            for eps in epsilons:
                scale = avg_S / eps
                noise = np.random.standard_cauchy((n_trials, 2000)) * scale
                noise_sq_med = np.median(np.mean(noise**2, axis=1))
                agt_mses.append(mse_nom + noise_sq_med)
            results[d][n_samples]['agt'] = agt_mses
            print(f"    AGT Avg S: {avg_S:.4f}")
            
            # 2. Functional
            print("    Running Functional...")
            func_mses = []
            for eps in epsilons:
                trials = [run_functional_mechanism(n_samples, d, eps, seed=i) for i in range(5)]
                func_mses.append(np.mean(trials))
            results[d][n_samples]['func'] = func_mses
            
            # 3. Laplace
            print("    Running Laplace...")
            lap_mses = []
            for eps in epsilons:
                trials = [run_laplace_sgd(n_samples, d, eps, seed=i) for i in range(5)]
                lap_mses.append(np.mean(trials))
            results[d][n_samples]['lap'] = lap_mses
            
    np.save(f'{out_dir}/dim_results.npy', results)
    np.save(f'{out_dir}/epsilons.npy', epsilons)
    print("Dimension Ablation Complete.")

if __name__ == "__main__":
    run_dim_ablation()
