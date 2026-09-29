import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
data_dir = 'agtr/linear/values/baselines'
out_dir = 'agtr/linear/values'
# Define variables used in main()
out_dir_base = data_dir
out_dir_agt = 'agtr/linear/values/ablation' 

plot_prefix = 'lin_baselines'
os.makedirs('agtr/linear/figures/linear', exist_ok=True) # Ensure agtr/linear/figures/linear exists

# Plotting Style
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.titlesize': 28,
    'lines.linewidth': 4,
    'axes.linewidth': 2
})

def main():
    print("Generating Linear Baseline Plots...")
    
    # Load Epsilon Grid (Use Baselines one, should match)
    epsilons = np.load(f'{out_dir_base}/epsilons.npy')
    
    # Load Baseline Data
    func_mean = np.load(f'{out_dir_base}/func_mean.npy')
    func_std = np.load(f'{out_dir_base}/func_std.npy')
    lap_mean = np.load(f'{out_dir_base}/lap_mean.npy')
    lap_std = np.load(f'{out_dir_base}/lap_std.npy')
    gauss_mean = np.load(f'{out_dir_base}/gauss_mean.npy')
    gauss_std = np.load(f'{out_dir_base}/gauss_std.npy')
    
    # AGT Frequencies in run_linear_ablation.py
    freqs = [1, 3, 5, 10, 20]
    # Matched to plot_lin_ablation.py colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
    
    # 1. Pure DP Comparison
    plt.figure(figsize=(10, 11))
    
    # Baselines (Mean)
    plt.plot(epsilons, func_mean, 'o-', label='Functional Mechanism', color='gold', linewidth=5) # Bright Yellow (Gold for visibility)
    plt.fill_between(epsilons, func_mean - func_std, func_mean + func_std, color='gold', alpha=0.1)
    
    plt.plot(epsilons, lap_mean, 's-', label='Laplace DP-SGD (C=1)', color='hotpink', linewidth=5) # Hot Pink
    plt.fill_between(epsilons, lap_mean - lap_std, lap_mean + lap_std, color='hotpink', alpha=0.1)
    
    # AGT (Median - Cauchy)
    for idx, f in enumerate(freqs):
        try:
            # Load pre-calculated median from run_linear_ablation.py
            agt_pure_median = np.load(f'{out_dir_agt}/pure_f{f}_median.npy')
            
            label_agt = f'AGT (F={f})'
            # Check length match
            if len(agt_pure_median) == len(epsilons):
                 plt.plot(epsilons, agt_pure_median, marker='.', linestyle='--', label=label_agt, color=colors[idx], linewidth=5)
            else:
                 print(f"Warning: AGT F={f} shape mismatch {len(agt_pure_median)} vs {len(epsilons)}")
            
        except FileNotFoundError:
            print(f"AGT Freq={f} Pure DP data not found in {out_dir_agt}.")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE')
    plt.title('Linear')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/linear/figures/{plot_prefix}_pure.png', dpi=300)
    plt.close()
    
    # 2. Approx DP Comparison
    plt.figure(figsize=(10, 11))
    
    plt.plot(epsilons, gauss_mean, 'o-', label='Gaussian DP-SGD', color='black', alpha=0.5) # Distinct
    plt.fill_between(epsilons, gauss_mean - gauss_std, gauss_mean + gauss_std, color='tab:red', alpha=0.1)
    
    for idx, f in enumerate(freqs):
        try:
            # Load pre-calculated mean from run_linear_ablation.py (Approx uses Gaussian, so Mean is fine)
            agt_approx_mean = np.load(f'{out_dir_agt}/approx_f{f}_mean.npy')
            
            label_agt = f'AGT (F={f})'
            if len(agt_approx_mean) == len(epsilons):
                plt.plot(epsilons, agt_approx_mean, marker='.', linestyle='--', label=label_agt, color=colors[idx], linewidth=4)
                
        except FileNotFoundError:
             print(f"AGT Freq={f} Approx DP data not found.")
             
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Privacy Budget $\epsilon$')
    plt.ylabel('Test MSE')
    plt.title('Linear')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'agtr/linear/figures/{plot_prefix}_approx.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to agtr/linear/figures/{plot_prefix}_*.png")

if __name__ == "__main__":
    main()
