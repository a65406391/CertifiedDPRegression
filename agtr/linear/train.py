"""
Linear Regression Ablation Study for Privacy Certified Training.

Refactored from run_lin_ablation.py.

Studies how concretization frequency affects certified bounds
when using gradient clipping for privacy certification.

Privacy Model: Per-sample gradient clipping (clip_method="clamp")
- Each gradient element is clamped to [-gamma, gamma]
- Privacy bounds use SEMin/SEMax accumulation with k*gamma sensitivity
"""

import torch
import numpy as np
import os
import logging
import argparse
from tqdm import tqdm


from agtr.linear.certify import train_certified_model
from agtr.linear.config import ABLATION as CFG
from common.data import simulated_linear_data, get_or_create_w_true
from common.sensitivity import compute_vk_matrix, compute_smooth_sensitivity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_lin_ablation():
    """Run the linear ablation study."""
    print("=" * 60)
    print("Linear Ablation Study - Privacy Certified Training")
    print(f"Samples: {CFG['N_SAMPLES']}, Batch: {CFG['BATCH_SIZE']}, Epochs: {CFG['EPOCHS']}")
    print(f"Clipping: gamma={CFG['CLIP_GAMMA']}")
    print("=" * 60)

    # CLI first: a rerun with a different k grid must not overwrite the published arrays
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument(
        "--out",
        default=CFG["OUT_DIR"],
        help="where to write method_results_*.npy (default: the config OUT_DIR)",
    )
    _p.add_argument(
        "--k-max", type=int, default=None, help="override the top of the k grid; the grid stays 0..k-max"
    )
    _a, _ = _p.parse_known_args()

    out_dir = _a.out
    os.makedirs(out_dir, exist_ok=True)

    # Generate Data
    w_true = get_or_create_w_true(CFG["INPUT_DIM"], CFG["SPARSITY"], path=CFG["W_TRUE_PATH"])
    X_train, y_train = simulated_linear_data(
        n_samples=CFG["N_SAMPLES"],
        dim=CFG["INPUT_DIM"],
        sparsity=CFG["SPARSITY"],
        seed=42,
        w_true=w_true,
        w_true_path=CFG["W_TRUE_PATH"],
    )
    X_test, y_test = simulated_linear_data(
        n_samples=2000,
        dim=CFG["INPUT_DIM"],
        sparsity=CFG["SPARSITY"],
        seed=100,
        w_true=w_true,
        w_true_path=CFG["W_TRUE_PATH"],
    )
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    # Methods
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=CFG["METHODS"], help="Methods to run")
    parser.add_argument("--merge", action="store_true", help="Merge results and plot")
    args, unknown = parser.parse_known_args()

    methods = [m for m in args.methods if m in CFG["METHODS"]]
    k_values = list(range(0, _a.k_max + 1)) if _a.k_max is not None else CFG["K_VALUES"]
    print(f"k grid: {k_values[0]}..{k_values[-1]} ({len(k_values)} values) -> {out_dir}")

    # Sensitivity params
    beta = CFG["BETA"]
    max_k_eval = max(k_values)
    tail_S = CFG["GS_VAL"] * np.exp(-beta * (max_k_eval + 1))

    results = {}
    if args.merge:
        print("Merging existing results...")
        for m in CFG["METHODS"]:
            fname = f"{out_dir}/method_results_{m}.npy"
            if os.path.exists(fname):
                try:
                    res = np.load(fname, allow_pickle=True).item()
                    results[m] = res
                    print(f"Loaded {m} from {fname}")
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        methods = []

    # Eval points for sensitivity
    np.random.seed(999)
    n_eval = CFG["N_EVAL"]
    X_sens_eval = np.random.uniform(-2, 2, (n_eval, CFG["INPUT_DIM"]))
    X_sens_eval_t = torch.tensor(X_sens_eval, dtype=torch.float64)

    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print("-" * 40)

        if method in results:
            print(f"Skipping {method} (Already computed)")
            continue

        method_models = {}

        pbar = tqdm(k_values, desc=f"Training {method.upper()}")
        for k in pbar:
            pbar.set_description(f"Method {method.upper()} (k={k})")
            try:
                m = train_certified_model(
                    X_train, y_train, epochs=CFG["EPOCHS"], lr=CFG["LR"], seed=42, method=method, k=k
                )
                method_models[k] = m

                w_l = float(m.param_l[0].flatten()[0])
                w_u = float(m.param_u[0].flatten()[0])
                print(f"w: [{w_l:.4f}, {w_u:.4f}] (width={w_u - w_l:.4f})")
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
            test_pred = nom_model.forward(X_test_t).squeeze().numpy()
            mse_nom = np.mean((test_pred - y_test_t.numpy().flatten()) ** 2)

            y_pred_sens = nom_model.forward(X_sens_eval_t).squeeze().numpy()

            V_k_mat = compute_vk_matrix(
                method_models, X_sens_eval_t, k_values, y_pred_nom=y_pred_sens, gs_cap=CFG["GS_VAL"]
            )
            S_vals = compute_smooth_sensitivity(V_k_mat, k_values, beta=beta, gs_val=CFG["GS_VAL"])
            avg_S = np.mean(S_vals)

        avg_V_k = np.mean(V_k_mat, axis=0)

        results[method] = {
            "mse_nom": mse_nom,
            "avg_sens": avg_S,
            "V_k_mat": V_k_mat,
            "avg_V_k": avg_V_k,
            "k_values": k_values,
        }

        print(f"\n  Results: MSE={mse_nom:.4f}, Avg Sensitivity={avg_S:.4f}")
        print(f"  V_k (avg): {[f'{v:.4f}' for k, v in zip(k_values, avg_V_k)]}")

        np.save(f"{out_dir}/method_results_{method}.npy", results[method])
        print(f"Saved: {out_dir}/method_results_{method}.npy")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Expected: F2 <= F1 <= IBP (tighter bounds with more MIP)")
    print("-" * 60)
    for method, res in results.items():
        print(f"  {method.upper():4s}: MSE={res['mse_nom']:.4f}, Sensitivity={res['avg_sens']:.4f}")

    # Inline staircase plot
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"ibp": "blue", "f1": "cyan", "f2": "green", "f4": "lime", "f8": "yellow", "f16": "orange"}
    labels = {
        "ibp": "IBP",
        "f1": "MIP F=1",
        "f2": "MIP F=2",
        "f4": "MIP F=4",
        "f8": "MIP F=8",
        "f16": "MIP F=16",
    }

    ax1.set_xlabel("Privacy Parameter k")
    ax1.set_ylabel("Instability V_k (average)")
    ax1.set_title("Instability V_k by Method")

    for method, res in results.items():
        ks = res["k_values"]
        vs = res["avg_V_k"]
        ax1.step(ks, vs, where="post", label=labels[method], color=colors[method], linewidth=2)
        ax1.scatter(ks[1:], vs[1:], color=colors[method], s=50)

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Privacy Parameter t")
    ax2.set_ylabel("Sensitivity V_k × exp(-β×k)")
    ax2.set_title("Smooth Sensitivity (should differ if bounds differ)")

    t_grid = np.linspace(0, max(k_values) + 1, 200)

    for method, res in results.items():
        ks = res["k_values"]
        vs = res["avg_V_k"]
        indices = np.searchsorted(ks, t_grid, side="right") - 1
        indices = np.clip(indices, 0, len(ks) - 1)
        v_interp = np.array([vs[idx] for idx in indices])
        s_curve = v_interp * np.exp(-beta * t_grid)
        ax2.plot(t_grid, s_curve, label=labels[method], color=colors[method], linewidth=2)
        idx_max = np.argmax(s_curve)
        ax2.scatter(
            t_grid[idx_max], s_curve[idx_max], color=colors[method], s=100, marker="*", edgecolors="black"
        )

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/staircase_comparison.png", dpi=150)
    print(f"Saved: {out_dir}/staircase_comparison.png")
    plt.close()

    print("\nAblation Complete.")


if __name__ == "__main__":
    run_lin_ablation()
