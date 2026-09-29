"""Test that a 1-hidden-layer ReLU network learns the linear regression task well.

We need to verify that adding non-linearity doesn't break convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def generate_linear_data(n_samples, n_features, seed=42):
    """Generate simple linear regression data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    w_true = np.random.randn(n_features).astype(np.float64)
    y = X @ w_true + 0.1 * np.random.randn(n_samples).astype(np.float64)
    
    return X, y, w_true


def create_linear_model(in_features):
    """Original linear model."""
    model = nn.Sequential(
        nn.Linear(in_features, 1, bias=True)
    )
    return model.to(torch.float64)


def create_relu_model(in_features, hidden_dim=4):
    """1-hidden-layer ReLU network."""
    model = nn.Sequential(
        nn.Linear(in_features, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1, bias=True)
    )
    return model.to(torch.float64)


def train_model(model, X, y, epochs=100, lr=0.1):
    """Train model with SGD and MSE loss."""
    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def evaluate_model(model, X, y):
    """Evaluate model MSE."""
    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    
    with torch.no_grad():
        y_pred = model(X_tensor)
        mse = ((y_pred - y_tensor) ** 2).mean().item()
    
    return mse


def main():
    print("="*60)
    print("Testing ReLU Network for Linear Regression")
    print("="*60)
    
    # Config matching the linear benchmark
    N_TRAIN = 25
    N_TEST = 100
    D = 2
    EPOCHS = 100  # More epochs for ReLU to converge
    LR = 0.1
    HIDDEN_DIMS = [4, 8]
    
    # Generate data
    X_train, y_train, w_true = generate_linear_data(N_TRAIN, D, seed=42)
    X_test, y_test, _ = generate_linear_data(N_TEST, D, seed=123)
    # Use same w_true for test data
    y_test = X_test @ w_true + 0.1 * np.random.randn(N_TEST)
    
    print(f"\nData: N_train={N_TRAIN}, N_test={N_TEST}, D={D}")
    print(f"Training: EPOCHS={EPOCHS}, LR={LR}")
    
    # Test linear model first
    print("\n" + "-"*60)
    print("Linear Model (baseline):")
    print("-"*60)
    
    linear_model = create_linear_model(D)
    linear_losses = train_model(linear_model, X_train, y_train, EPOCHS, LR)
    linear_train_mse = evaluate_model(linear_model, X_train, y_train)
    linear_test_mse = evaluate_model(linear_model, X_test, y_test)
    
    print(f"  Train MSE: {linear_train_mse:.6f}")
    print(f"  Test MSE:  {linear_test_mse:.6f}")
    
    # Test ReLU models with different hidden dims
    print("\n" + "-"*60)
    print("ReLU Models:")
    print("-"*60)
    
    results = {"Linear": (linear_train_mse, linear_test_mse, linear_losses)}
    
    for H in HIDDEN_DIMS:
        print(f"\n  Hidden dim H={H}:")
        relu_model = create_relu_model(D, hidden_dim=H)
        relu_losses = train_model(relu_model, X_train, y_train, EPOCHS, LR)
        relu_train_mse = evaluate_model(relu_model, X_train, y_train)
        relu_test_mse = evaluate_model(relu_model, X_test, y_test)
        
        print(f"    Train MSE: {relu_train_mse:.6f}")
        print(f"    Test MSE:  {relu_test_mse:.6f}")
        
        results[f"ReLU H={H}"] = (relu_train_mse, relu_test_mse, relu_losses)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"{'Model':<15} {'Train MSE':>12} {'Test MSE':>12}")
    print("-"*40)
    for name, (train_mse, test_mse, _) in results.items():
        print(f"{name:<15} {train_mse:>12.6f} {test_mse:>12.6f}")
    
    # Check if ReLU models match linear performance
    best_relu = min(
        [(name, test_mse) for name, (_, test_mse, _) in results.items() if "ReLU" in name],
        key=lambda x: x[1]
    )
    
    print(f"\n✓ Best ReLU model: {best_relu[0]} (Test MSE: {best_relu[1]:.6f})")
    
    if best_relu[1] <= linear_test_mse * 1.5:
        print("✓ ReLU model achieves comparable performance to linear model")
        print("  Safe to use for the benchmark!")
    else:
        print("✗ ReLU model significantly worse than linear model")
        print("  May need hyperparameter tuning")
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for name, (_, _, losses) in results.items():
        plt.plot(losses, label=name, linewidth=2 if "Linear" in name else 1)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Convergence Comparison: Linear vs ReLU Models")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(".figures/relu_convergence_test.png", dpi=150)
    print(f"\nSaved convergence plot to .figures/relu_convergence_test.png")


if __name__ == "__main__":
    main()
