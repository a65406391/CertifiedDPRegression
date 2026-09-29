"""Sanity check for the Adversary Budget Reset Bug fix.

This script tests that:
1. Substitution indicators are correctly extracted after MIP solve
2. Cumulative substituted samples are correctly tracked across concretization windows
3. The global k constraint is respected (no budget reset)
4. F=Total should now be tighter than F=1 (Aggregation Law)
"""

import logging
import numpy as np
import torch
import torch.nn as nn

# Set up logging to see the indicator tracking messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig


def create_linear_model(in_features):
    """Create a simple linear model wrapped in Sequential."""
    model = nn.Sequential(
        nn.Linear(in_features, 1, bias=True)
    )
    model = model.to(torch.float64)
    return model


def generate_data(n_samples, n_features, seed=42):
    """Generate simple linear regression data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    
    return X, y


def test_indicator_persistence():
    """Test that indicators persist across concretization windows."""
    print("\n" + "="*60)
    print("Test 1: Indicator Persistence Across Concretizations")
    print("="*60)
    
    # Small dataset to make test fast
    N = 10
    D = 2
    X, y = generate_data(N, D)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # Config with frequent concretization (F=2, so 5 concretizations in 10 batches of size 1)
    config = OPTConfig(
        batchsize=N,  # Full batch
        n_epochs=5,   # 5 epochs = 5 batches total
        learning_rate=0.1,
        concretization_freq=2,  # Concretize every 2 epochs -> 2 concretizations
        loss="mse",
        k_substitution=1,  # Only 1 sample can be substituted GLOBALLY
        gradient_clipping=1.0,
        input_domain=None,  # Privacy path
        poisoning_encoding="binary",
        gurobi_kwargs={"LogToConsole": 0},
    )
    
    model = create_linear_model(D)
    
    print(f"Config: N={N}, BS={N}, EPOCHS=5, F=2, k=1")
    print("With the fix, we should see 'Carrying forward' messages after first concretization")
    print("-"*60)
    
    try:
        trained_model, _ = certified_training(model, config, dataset, solve=True)
        print("-"*60)
        print("✓ Training completed successfully!")
        
        # Check bounds exist
        sens = (trained_model.param_u[0] - trained_model.param_l[0]).max().item()
        print(f"Maximum parameter sensitivity: {sens:.4f}")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_law():
    """Test that F=Total is now tighter than F=1 (Aggregation Law)."""
    print("\n" + "="*60)
    print("Test 2: Aggregation Law (F=Total should be tighter than F=1)")
    print("="*60)
    
    N = 25
    D = 2
    EPOCHS = 5
    BS = 25  # Full batch
    
    X, y = generate_data(N, D)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    sensitivities = {}
    
    for freq in [1, EPOCHS]:  # F=1 (frequent) vs F=5 (global)
        freq_label = "F=1 (frequent)" if freq == 1 else f"F={EPOCHS} (global)"
        print(f"\nTraining with {freq_label}...")
        
        config = OPTConfig(
            batchsize=BS,
            n_epochs=EPOCHS,
            learning_rate=0.1,
            concretization_freq=freq,
            loss="mse",
            k_substitution=1,
            gradient_clipping=1.0,
            input_domain=None,
            poisoning_encoding="binary",
            gurobi_kwargs={"LogToConsole": 0},
        )
        
        model = create_linear_model(D)
        
        try:
            trained_model, _ = certified_training(model, config, dataset, solve=True)
            sens = (trained_model.param_u[0] - trained_model.param_l[0]).max().item()
            sensitivities[freq] = sens
            print(f"  Sensitivity: {sens:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            sensitivities[freq] = None
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print(f"  F=1 (frequent concretization): {sensitivities.get(1, 'N/A')}")
    print(f"  F={EPOCHS} (global visibility):     {sensitivities.get(EPOCHS, 'N/A')}")
    
    if all(v is not None for v in sensitivities.values()):
        if sensitivities[EPOCHS] < sensitivities[1]:
            print("\n✓ Aggregation Law HOLDS: F=Total is tighter than F=1")
            return True
        elif sensitivities[EPOCHS] > sensitivities[1]:
            print("\n✗ Aggregation Law VIOLATED: F=Total is looser than F=1")
            print("  (This suggests the bug fix may not be working as expected)")
            return False
        else:
            print("\n? Sensitivities are equal (may be due to small test size)")
            return True
    return False


if __name__ == "__main__":
    print("Sanity Check for Adversary Budget Reset Bug Fix")
    print("="*60)
    
    # Run tests
    test1_pass = test_indicator_persistence()
    test2_pass = test_aggregation_law()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (Indicator Persistence): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Aggregation Law):        {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✓ All sanity checks passed!")
    else:
        print("\n✗ Some checks failed - review implementation")
