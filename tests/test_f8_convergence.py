
import numpy as np
import torch
import random
import os
import sys
import time
from tqdm import tqdm

# Constants (Match Reduced Benchmark)
BATCH_SIZE = 5
N_SAMPLES = 80
EPOCHS = 1
INPUT_DIM = 1
CLIP_GAMMA = 0.1 
LR = 1.0

# Imports
from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel

def simulated_linear_data(n_samples=100, input_dim=1, noise=0.05, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(n_samples, input_dim))
    w_true = np.ones((input_dim, 1))
    e = np.random.normal(0, noise, size=(n_samples, 1))
    y = X @ w_true + e
    return X, y

def train_f8_convergence_test():
    print("DEBUG: Testing F8 Convergence with TimeLimit=600s")
    
    # Data
    X_train, y_train = simulated_linear_data(n_samples=N_SAMPLES, seed=42)
    
    # Model: Match run_lin_ablation.py exactly
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1) # Defaults to bias=True
    ).double()
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].bias.fill_(0.0)
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float64), 
        torch.tensor(y_train, dtype=torch.float64)
    )
    
    # Config for F8
    freq = 8
    k = 1 # Test at k=1
    time_limit = 600 # 10 minutes
    
    print(f"Configuring F8: freq={freq}, k={k}, TimeLimit={time_limit}")
    
    config = OPTConfig(
        n_epochs=EPOCHS,
        batchsize=BATCH_SIZE,
        learning_rate=LR,
        loss="mse",
        concretization_freq=freq,
        k_substitution=k,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CLIP_GAMMA,
        relax_binaries=False,
        relax_bilinear=False,
        # OutputFlag=1 to see solver progress
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 1, "Threads": 1, "TimeLimit": time_limit}, 
        metadata={"seed": 42, "strict_mode": True}
    )
    
    start_time = time.time()
    final_model, stats = certified_training(model, config, dataset, solve=True)
    end_time = time.time()
    
    print(f"\nTotal Execution Time: {end_time - start_time:.2f}s")
    print(f"Stats: {stats}")
    
    # Eval width
    np.random.seed(999)
    X_eval = np.random.uniform(-2, 2, (10, 1))
    X_eval_t = torch.tensor(X_eval, dtype=torch.float64)
    yl, yu = final_model.bound_forward(X_eval_t, X_eval_t, retain_intermediate=False)
    yl = yl.detach().numpy().flatten()
    yu = yu.detach().numpy().flatten()
    width = np.mean(yu - yl)
    print(f"\nResult Width: {width:.4f}")

if __name__ == "__main__":
    train_f8_convergence_test()
