import numpy as np
import matplotlib.pyplot as plt
import os

# --- Plotting Style ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_data_ablation():
    print("Plotting Data Ablation...")
    in_dir = 'agtr/linear/values/data_ablation' # Correct Dir
    out_dir = 'agtr/linear/figures/linear/data_ablation'
    os.makedirs(out_dir, exist_ok=True)
    
    results = np.load(f'{in_dir}/data_results.npy', allow_pickle=True).item()
    epsilons = np.load(f'{in_dir}/epsilons.npy')
    
    # Plot: N vs MSE (Trade-off curves for each N)
    plt.figure(figsize=(10, 6))
    
    sorted_ns = sorted(results.keys())
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, n in enumerate(sorted_ns):
        mse_vals = results[n]
        label = f'N={n}'
        plt.plot(epsilons, mse_vals, 'o-', label=label, color=colors[i % len(colors)], linewidth=2)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test Median Squared Error (Pure DP)')
    plt.title('Data Ablation: Dataset Size vs Utility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/data_focused_tradeoff.png', dpi=300)
    print("Data Ablation Plot Complete.")

if __name__ == "__main__":
    plot_data_ablation()
