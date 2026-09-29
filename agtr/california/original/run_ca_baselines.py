import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine, GradSampleModule

# --- Configuration ---
# Match run_ca_batch_ablation.py
# Match Optimized AGT
BATCH_SIZE = 8000
EPOCHS = 40
LR = 0.08

# Clip features/targets globally for baselines to ensure finite sensitivity
# AGT bounds sensitivity per-sample, but baselines need a global bounds assumption or clipping.
CLIP_X = 5.0
CLIP_Y = 5.0

def load_california_data_dataset(seed=42):
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
    
    # CLIP DATA GLOBALLY (Baseline Requirement)
    X_scaled = np.clip(X_scaled, -CLIP_X, CLIP_X)
    y_scaled = np.clip(y_scaled, -CLIP_Y, CLIP_Y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32) # Target shape?
    
    if y_train_t.ndim == 1:
        y_train_t = y_train_t.unsqueeze(1)
    if y_test_t.ndim == 1:
        y_test_t = y_test_t.unsqueeze(1)
        
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, X_test_t, y_test_t

# --- Model Definition (MLP) ---
def create_mlp(input_dim=8):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# --- Baselines ---

def run_functional_mechanism(epsilon, seed=42):
    # Functional Mechanism is designed for LINEAR Regression.
    # We will use it as a Linear Baseline.
    # Sensitivity depends on bounds.
    
    np.random.seed(seed)
    # Load raw numpy for algebra
    loader, X_test_t, y_test_t = load_california_data_dataset(seed)
    X_train = loader.dataset.tensors[0].numpy()
    y_train = loader.dataset.tensors[1].numpy().ravel()
    
    # Add bias
    X_b = np.c_[X_train, np.ones(len(X_train))]
    d = X_b.shape[1] # 8 + 1 = 9
    
    XTX = X_b.T @ X_b
    XTY = X_b.T @ y_train
    
    # Sensitivity
    # ||x||_2 <= sqrt(8 * CLIP_X^2 + 1).
    R2 = 8 * CLIP_X**2 + 1.0 # 201
    eps_1 = epsilon / 2.0
    eps_2 = epsilon / 2.0
    
    # Noise XTX
    s_xtx = R2
    noise_xtx = np.random.laplace(0, s_xtx / eps_1, (d, d))
    noise_xtx = np.triu(noise_xtx) + np.triu(noise_xtx, 1).T
    XTX_p = XTX + noise_xtx
    
    # Noise XTY
    # Sens = R * CLIP_Y
    s_xty = np.sqrt(R2) * CLIP_Y # sqrt(201) * 5 ~ 70
    noise_xty = np.random.laplace(0, s_xty / eps_2, (d, 1))
    XTY_p = XTY.reshape(-1, 1) + noise_xty
    
    try:
        theta = np.linalg.solve(XTX_p, XTY_p)
    except np.linalg.LinAlgError:
        theta = np.zeros((d, 1))
        
    # Eval
    X_test_b = np.c_[X_test_t.numpy(), np.ones(len(X_test_t))]
    y_pred = X_test_b @ theta
    mse = np.mean((y_pred - y_test_t.numpy())**2)
    return mse

def run_laplace_sgd_mlp(epsilon, seed=42):
    # Pure DP SGD on MLP.
    torch.manual_seed(seed)
    loader, X_test_t, y_test_t = load_california_data_dataset(seed)
    
    model = create_mlp()
    model = GradSampleModule(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean')
    
    C = 1.0
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    
    if epsilon > 100:
        noise_scale = 0
    else:
        eps_step = epsilon / total_steps
        noise_scale = 2 * C / eps_step
        
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            with torch.no_grad():
                # 1. Per Sample Norms
                params = list(model.parameters())
                batch_size = X_batch.shape[0]
                per_sample_norms_sq = torch.zeros(batch_size)
                
                for p in params:
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                        if isinstance(p.grad_sample, list):
                            p_gs = torch.cat(p.grad_sample, dim=0)
                        else:
                            p_gs = p.grad_sample
                        
                        flat_g = p_gs.reshape(batch_size, -1)
                        per_sample_norms_sq += flat_g.norm(2, dim=1)**2
                        
                per_sample_norms = torch.sqrt(per_sample_norms_sq)
                clip_factors = torch.clamp(C / (per_sample_norms + 1e-6), max=1.0)
                
                # 2. Add Noise & Average
                for p in params:
                    if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                         if isinstance(p.grad_sample, list):
                             p_gs = torch.cat(p.grad_sample, dim=0)
                         else:
                             p_gs = p.grad_sample
                             
                         view_shape = [batch_size] + [1] * (p_gs.ndim - 1)
                         clipped = p_gs * clip_factors.view(*view_shape)
                         grad_sum = clipped.sum(dim=0)
                         
                         if noise_scale > 0:
                             noise = np.random.laplace(0, noise_scale, p.shape)
                             grad_sum += torch.tensor(noise, dtype=torch.float32)
                             
                         p.grad = grad_sum / batch_size
                         p.grad_sample = None
                         
            optimizer.step()
            
    # Eval
    with torch.no_grad():
        preds = model(X_test_t)
        mse = nn.MSELoss()(preds, y_test_t).item()
    return mse

def run_gaussian_sgd_mlp(epsilon, seed=42):
    # Approx DP SGD on MLP using Opacus PrivacyEngine.
    torch.manual_seed(seed)
    loader, X_test_t, y_test_t = load_california_data_dataset(seed)
    
    model = create_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    privacy_engine = PrivacyEngine()
    
    try:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=epsilon,
            target_delta=1e-5,
            epochs=EPOCHS,
            max_grad_norm=1.0,
        )
    except AttributeError:
        # Fallback
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0, 
            max_grad_norm=1.0,
        )
        
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
    with torch.no_grad():
        preds = model(X_test_t)
        mse = nn.MSELoss()(preds, y_test_t).item()
    return mse


def main():
    print("Running California Baselines (Fast Recovery Mode)...")
    out_dir = 'agtr/california/values/baselines'
    os.makedirs(out_dir, exist_ok=True)
    
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 1 # One trial for fair comparison speed check
    
    # 1. Functional (Linear Baseline)
    func_means = []
    func_stds = []
    print("Functional Mechanism (Linear)...")
    for eps in epsilons:
        trials = [run_functional_mechanism(eps, seed=i) for i in range(n_trials)]
        func_means.append(np.mean(trials))
        func_stds.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
    
    np.save(f'{out_dir}/func_mean.npy', func_means)
    np.save(f'{out_dir}/func_std.npy', func_stds)
    
    # 2. Laplace MLP
    lap_means = []
    lap_stds = []
    print("Laplace DP-SGD (MLP)...")
    for eps in epsilons:
        # Skip low eps for Laplace MLP? It's usually catastrophically bad.
        # But we need data.
        trials = [run_laplace_sgd_mlp(eps, seed=i) for i in range(n_trials)]
        lap_means.append(np.mean(trials))
        lap_stds.append(np.std(trials))
        print(f"  Eps={eps}: {np.mean(trials):.4f}")
        
    np.save(f'{out_dir}/lap_mean.npy', lap_means)
    np.save(f'{out_dir}/lap_std.npy', lap_stds)
    
    # 3. Gaussian MLP
    gauss_means = []
    gauss_stds = []
    print("Gaussian DP-SGD (MLP)...")
    for eps in epsilons:
         trials = [run_gaussian_sgd_mlp(eps, seed=i) for i in range(n_trials)]
         gauss_means.append(np.mean(trials))
         gauss_stds.append(np.std(trials))
         print(f"  Eps={eps}: {np.mean(trials):.4f}")
         
    np.save(f'{out_dir}/gauss_mean.npy', gauss_means)
    np.save(f'{out_dir}/gauss_std.npy', gauss_stds)
    
    np.save(f'{out_dir}/epsilons.npy', epsilons)
    print("Done CA Baselines.")

if __name__ == "__main__":
    main()
