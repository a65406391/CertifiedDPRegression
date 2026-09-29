import torch
import torch.nn as nn
import numpy as np
import os
import logging
from torch.utils.data import TensorDataset, DataLoader

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
EPOCHS = 20 # Standard Process A

# Data Gen (Same as core)
def simulated_linear_data(n_samples=1000, slope=2, intercept=1, x_range=(-2, 2), seed=42):
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    noise = np.random.normal(0, 0.1, (n_samples, 1)) 
    y = slope * X + intercept + noise
    return X, y

def train_agt_model(train_loader, epochs, lr, seed, freq, k):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = nn.Sequential(nn.Linear(1, 1))
    
    bounded_model = MIPBoundedModel(
        model, 
        forward_bound_mode='miqp', 
        optimize_inter=True,
        concretization_frequency=freq 
    )
    
    optimizer = BoundedSGD(bounded_model, learning_rate=lr)
    criterion = BoundedMSELoss(reduction='sum')
    
    for epoch in range(epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            curr_bs = x_batch.size(0)
            
            if k == 0:
                logits = bounded_model.forward(x_batch, retain_intermediate=True) # (B, 1)
                loss_g = criterion.backward(logits, y_batch) # (B, 1)
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
                 delta = k * 2.0 * 1.0 / curr_bs
                 grads_l = [g - delta for g in grads_n_mean]
                 grads_u = [g + delta for g in grads_n_mean]
                 
                 optimizer.step(grads_l, grads_n_mean, grads_u)

    return bounded_model

def run_data_ablation():
    print("Running Data Ablation (N vs Performance)...")
    out_dir = 'agtr/linear/values/data_ablation'
    os.makedirs(out_dir, exist_ok=True)
    
    n_samples_list = [250, 500, 1000]
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # K values
    k_dense = [0] + list(range(1, 11)) + list(range(12, 36, 2))
    k_values = sorted(list(set(k_dense)))
    
    # Test Data
    X_test, y_test = simulated_linear_data(n_samples=2000, seed=100)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Sensitivity Params
    Y_MIN = -4.0; Y_MAX = 6.0; GS_VAL = 10.0
    beta = 1.0 / 6.0
    max_k_eval = max(k_values)
    tail_S = GS_VAL * np.exp(-beta * (max_k_eval + 1))
    
    results = {}
    
    np.random.seed(999)
    # Sens Evaluation Points (Since d=1, we can grid or random sample. Random sample consistent with dim ablation)
    X_sens_eval = np.random.uniform(-2, 2, (50, 1))
    X_sens_eval_t = torch.tensor(X_sens_eval, dtype=torch.float32)
    
    for n in n_samples_list:
        BATCH_SIZE = n // 5 # Scale Batch Size with N to keep steps constant
        freq = 5 # Constant steps per epoch.
        print(f"N={n} (BS={BATCH_SIZE}, F={freq})...")
        
        X_train, y_train = simulated_linear_data(n_samples=n, seed=42)
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        freq_models = {}
        for k in k_values:
            m = train_agt_model(loader, epochs=EPOCHS, lr=LR, seed=42, freq=freq, k=k)
            freq_models[k] = m
            
        # Analysis
        nom_model = freq_models[0]
        with torch.no_grad():
            test_pred = nom_model.forward(X_test_t).squeeze().numpy()
            mse_nom = np.mean((test_pred - y_test_t.numpy().flatten())**2)
            
            y_pred_sens = nom_model.forward(X_sens_eval_t).squeeze().numpy()
            V_k_mat = np.zeros((len(X_sens_eval), len(k_values)))
            
            # Calculate Raw Radii (for k of interest, e.g. k=1)
            k_interest = 1
            radius_k1 = 0.0
            if k_interest in freq_models:
                 mk1 = freq_models[k_interest]
                 yl, yu = mk1.bound_forward(X_sens_eval_t, X_sens_eval_t)
                 yl = yl.squeeze().numpy()
                 yu = yu.squeeze().numpy()
                 radii = np.maximum(np.abs(yl - y_pred_sens), np.abs(yu - y_pred_sens))
                 radius_k1 = np.mean(radii)

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
            
        # Pure DP MSEs
        agt_mses = []
        n_trials = 100 
        for eps in epsilons:
            scale = avg_S / eps
            noise = np.random.standard_cauchy((n_trials, 2000)) * scale
            noise_sq_med = np.median(np.mean(noise**2, axis=1))
            agt_mses.append(mse_nom + noise_sq_med)
        
        results[n] = agt_mses
        print(f"  Avg S: {avg_S:.4f}")
        print(f"  [DEBUG] N={n} | Nominal MSE: {mse_nom:.4f} | Radius (k={k_interest}): {radius_k1:.4f}")
        
    np.save(f'{out_dir}/data_results.npy', results)
    np.save(f'{out_dir}/epsilons.npy', epsilons)
    print("Data Ablation Complete.")

if __name__ == "__main__":
    run_data_ablation()
