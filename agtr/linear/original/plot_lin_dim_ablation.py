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

def plot_dim_ablation():
    print("Plotting Dimension Ablation...")
    in_dir = 'agtr/linear/values/dim_ablation'
    out_dir = 'agtr/linear/figures/linear/dim_ablation'
    os.makedirs(out_dir, exist_ok=True)
    
    results = np.load(f'{in_dir}/dim_results.npy', allow_pickle=True).item()
    epsilons = np.load(f'{in_dir}/epsilons.npy')
    
    dims = sorted(results.keys())
    
    for d in dims:
        print(f"Generating plot for d={d}...")
        n_samples_list = sorted(results[d].keys())
        
        n_panels = len(n_samples_list)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        
        if n_panels == 1: axes = [axes]
        
        for i, n in enumerate(n_samples_list):
            ax = axes[i]
            res = results[d][n]
            
            # AGT
            ax.plot(epsilons, res['agt'], 'o-', label='AGT (Pure DP)', color='tab:blue', linewidth=2)
            
            # Functional
            ax.plot(epsilons, res['func'], 's--', label='Functional Mech', color='tab:orange', linewidth=2)
            
            # Laplace
            ax.plot(epsilons, res['lap'], '^:', label='Laplace SGD', color='tab:green', linewidth=2)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Privacy Budget $\epsilon$')
            ax.set_title(f'N = {n}')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_ylabel('Test MSE')
                ax.legend()
                
        fig.suptitle(f'Dimension d={d} (Weights scaled)', y=1.05)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/dim_{d}_ablation.png', dpi=300, bbox_inches='tight')
        
    print("Dimension Ablation Plotting Complete.")

if __name__ == "__main__":
    plot_dim_ablation()
