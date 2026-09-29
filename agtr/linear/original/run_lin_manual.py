
import torch
import torch.nn as nn
import gurobipy as gp
import numpy as np
import os

# Helper to capture results
results_store = {}

def simulated_linear_data(n_samples=1000, slope=0.5, intercept=1.0, x_range=(-2, 2), seed=42):
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    noise = np.random.normal(0, 0.1, (n_samples, 1)) 
    y = slope * X + intercept + noise
    return X, y

def run_manual_mip(freq, k_private, N=80, BS=5, lr=1.0, clip_gamma=0.1, seed=42):
    # Data Setup (Benchmark Match)
    X_np, y_np = simulated_linear_data(n_samples=N, seed=seed)
    
    # Shuffle (Benchmark Match)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X = torch.tensor(X_np, dtype=torch.float64)[perm]
    y = torch.tensor(y_np, dtype=torch.float64)[perm]
    
    batches_X = list(X.split(BS))
    batches_y = list(y.split(BS))
    num_batches = len(batches_X)
    
    # Model Init (Benchmark Match)
    torch.manual_seed(seed)
    init_w = torch.randn(1, 1, dtype=torch.float64) # Typical nn.Linear init (random)
    # Actually nn.Linear uses Uniform(-1/sqrt(in), 1/sqrt(in)).
    # Let's match line 31 of run_lin_ablation logic?
    # run_lin_ablation uses:
    # model = nn.Sequential(nn.Linear(1, 1)).double()
    # torch.manual_seed(seed)
    # The linear layer init depends on seed.
    
    mod = nn.Sequential(nn.Linear(1, 1)).double()
    w0_val = mod[0].weight.item()
    b0_val = mod[0].bias.item()
    
    # Bounds: Start at point
    w0_l, w0_u = w0_val, w0_val
    b0_l, b0_u = b0_val, b0_val

    # Iterate in blocks of FREQ
    # Accumulate bounds over windows.
    # Note: Manual IBP propagation is needed between MIPs?
    # Or do we use the MIP output of window t as input to window t+1?
    # The Manual impl code in sanity_check.py (lines 64) read from `bm.param_l`.
    # Here we track `w0_l, w0_u` manually.
    
    current_w_l, current_w_u = w0_l, w0_u
    current_b_l, current_b_u = b0_l, b0_u
    
    for step_idx in range(0, num_batches, freq):
        current_steps = range(step_idx, min(step_idx + freq, num_batches))
        
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', 1)
        env.setParam('NonConvex', 2)
        env.setParam('TimeLimit', 600)
        env.start()
        m = gp.Model("unrolled_steps", env=env)

        # Init variables from current bounds
        W_curr = m.addVar(lb=current_w_l, ub=current_w_u, name="W_0")
        b_curr = m.addVar(lb=current_b_l, ub=current_b_u, name="b_0")
        
        # --- Constraints ---
        # Budget variables for this window
        all_indicators = []
        attack_indicators = {}
        
        for i in current_steps:
             batch_size_i = batches_X[i].size(0)
             for j in range(batch_size_i):
                 z = m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{j}")
                 attack_indicators[(i, j)] = z
                 all_indicators.append(z)
        
        # Local Budget Constraint (Reset every window)
        m.addConstr(gp.quicksum(all_indicators) <= k_private, name="Budget")
        
        # Unroll steps
        W_next = W_curr
        b_next = b_curr
        
        for i in current_steps:
            inputs = batches_X[i]
            targets = batches_y[i]
            batch_size_i = inputs.size(0)
            
            step_grads_W = []
            step_grads_b = []
            
            for j in range(batch_size_i):
                x_val = inputs[j].item()
                t_val = targets[j].item()
                
                # Gradient: 2(y_hat - y)x
                # y_hat = W*x + b
                # g_W = 2*(W*x + b - t)*x
                # g_b = 2*(W*x + b - t)
                
                # Raw Gradient Vars
                raw_g_W = m.addVar(lb=-float('inf'), name=f"raw_gW_{i}_{j}")
                m.addConstr(raw_g_W == 2 * (W_next * x_val + b_next - t_val) * x_val)
                
                raw_g_b = m.addVar(lb=-float('inf'), name=f"raw_gb_{i}_{j}")
                m.addConstr(raw_g_b == 2 * (W_next * x_val + b_next - t_val))
                
                # Clipped Gradient
                # clamp(g, -gamma, gamma)
                # Max(-gamma, Min(gamma, g))
                
                # W Clip
                cw_1 = m.addVar(lb=-float('inf'))
                m.addGenConstrMin(cw_1, [raw_g_W], constant=clip_gamma)
                cw_2 = m.addVar(lb=-float('inf'))
                m.addGenConstrMax(cw_2, [cw_1], constant=-clip_gamma)
                clipped_g_W = cw_2
                
                # b Clip
                cb_1 = m.addVar(lb=-float('inf'))
                m.addGenConstrMin(cb_1, [raw_g_b], constant=clip_gamma)
                cb_2 = m.addVar(lb=-float('inf'))
                m.addGenConstrMax(cb_2, [cb_1], constant=-clip_gamma)
                clipped_g_b = cb_2
                
                # Substitution
                z = attack_indicators[(i, j)]
                
                # eff = (1-z)*clipped + z*noise
                # eff = clipped - z*clipped + v
                # u = z*clipped. v = z*noise.
                
                # u_W linearization
                u_W = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                # Big-M = 2*clip_gamma? No, clipped is in [-gamma, gamma]. Dist to 0 is gamma.
                # u <= clipped + M(1-z). If z=1, u <= 0 + 0? No u <= gamma.
                # u = z * clipped.
                # z=0 -> u=0. z=1 -> u=clipped.
                # Linearization:
                # u <= clipped + M(1-z)
                # u >= clipped - M(1-z)
                # u <= M*z
                # u >= -M*z
                M = 2.0 * clip_gamma # Safe M
                m.addConstr(u_W <= clipped_g_W + M*(1-z))
                m.addConstr(u_W >= clipped_g_W - M*(1-z))
                m.addConstr(u_W <= M*z)
                m.addConstr(u_W >= -M*z)
                
                # v_W linearization (v = z*noise, noise free in [-gamma, gamma])
                v_W = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(v_W <= clip_gamma * z)
                m.addConstr(v_W >= -clip_gamma * z)
                
                eff_g_W = m.addVar(lb=-float('inf'))
                m.addConstr(eff_g_W == clipped_g_W - u_W + v_W)
                step_grads_W.append(eff_g_W)
                
                # Bias Substitution (Same logic)
                u_b = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(u_b <= clipped_g_b + M*(1-z))
                m.addConstr(u_b >= clipped_g_b - M*(1-z))
                m.addConstr(u_b <= M*z)
                m.addConstr(u_b >= -M*z)
                
                v_b = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(v_b <= clip_gamma * z)
                m.addConstr(v_b >= -clip_gamma * z)
                
                eff_g_b = m.addVar(lb=-float('inf'))
                m.addConstr(eff_g_b == clipped_g_b - u_b + v_b)
                step_grads_b.append(eff_g_b)
            
            # Batch Avg Gradient
            avg_g_W = m.addVar(lb=-float('inf'))
            m.addConstr(avg_g_W == gp.quicksum(step_grads_W) / batch_size_i)
            
            avg_g_b = m.addVar(lb=-float('inf'))
            m.addConstr(avg_g_b == gp.quicksum(step_grads_b) / batch_size_i)
            
            # Update
            entries_w_next = m.addVar(lb=-float('inf'))
            entries_b_next = m.addVar(lb=-float('inf'))
            m.addConstr(entries_w_next == W_next - lr * avg_g_W)
            m.addConstr(entries_b_next == b_next - lr * avg_g_b)
            
            W_next = entries_w_next
            b_next = entries_b_next
            
        # Optimize Output Min/Max
        m.update()
        
        # Max W
        m.setObjective(W_next, gp.GRB.MAXIMIZE)
        m.optimize()
        if m.status != gp.GRB.OPTIMAL: return 999.0 # Fail
        w_max = m.objVal
        
        # Min W
        m.setObjective(W_next, gp.GRB.MINIMIZE)
        m.optimize()
        if m.status != gp.GRB.OPTIMAL: return 999.0
        w_min = m.objVal
        
        # Max b
        m.setObjective(b_next, gp.GRB.MAXIMIZE)
        m.optimize()
        b_max = m.objVal
        
        # Min b
        m.setObjective(b_next, gp.GRB.MINIMIZE)
        m.optimize()
        b_min = m.objVal
        
        # Update bounds for next window
        current_w_l, current_w_u = w_min, w_max
        current_b_l, current_b_u = b_min, b_max
        
    return current_w_u - current_w_l

def main():
    freqs = [1, 2, 4, 8, 16]
    ks = [0, 1, 2, 3, 4, 5]
    
    # Store compatible with method_results format somewhat
    # method_results['fN']['avg_V_k'] = list of widths
    final_output = {}
    
    for f in freqs:
        print(f"Running Manual F={f}")
        widths = []
        for k in ks:
            width = run_manual_mip(freq=f, k_private=k)
            widths.append(width)
            print(f"  k={k}: {width:.4f}")
        
        # Store
        key = f"f{f}"
        final_output[key] = {
            'avg_V_k': widths,
            'k_values': ks
        }
    
    # Save
    import numpy as np
    save_path = 'agtr/linear/values/ablation/manual_results.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_output)
    print(f"Saved manual results to {save_path}")

if __name__ == "__main__":
    main()
