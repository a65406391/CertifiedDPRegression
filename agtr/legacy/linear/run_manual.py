# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""
Manual MIP verification for Linear benchmark.

Refactored from run_lin_manual.py.
Exact Gurobi MIP formulation for ground truth bounds.
"""

import torch
import torch.nn as nn
import gurobipy as gp
import numpy as np
import os

from agtr.linear.config import MANUAL as CFG
from common.data import simulated_linear_data


def run_manual_mip(freq, k_private, N=80, BS=5, lr=1.0, clip_gamma=0.1, seed=42):
    """Run manual MIP verification for a given frequency and k."""
    X_np, y_np = simulated_linear_data(
        n_samples=N,
        slope=CFG["SLOPE"],
        intercept=CFG["INTERCEPT"],
        seed=seed,
        dim=1,
    )

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X = torch.tensor(X_np, dtype=torch.float64)[perm]
    y = torch.tensor(y_np, dtype=torch.float64)[perm]

    batches_X = list(X.split(BS))
    batches_y = list(y.split(BS))
    num_batches = len(batches_X)

    # Model Init (match benchmark)
    torch.manual_seed(seed)
    mod = nn.Sequential(nn.Linear(1, 1)).double()
    w0_val = mod[0].weight.item()
    b0_val = mod[0].bias.item()

    current_w_l, current_w_u = w0_val, w0_val
    current_b_l, current_b_u = b0_val, b0_val

    for step_idx in range(0, num_batches, freq):
        current_steps = range(step_idx, min(step_idx + freq, num_batches))

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.setParam("NonConvex", 2)
        env.setParam("TimeLimit", 600)
        env.start()
        m = gp.Model("unrolled_steps", env=env)

        W_curr = m.addVar(lb=current_w_l, ub=current_w_u, name="W_0")
        b_curr = m.addVar(lb=current_b_l, ub=current_b_u, name="b_0")

        all_indicators = []
        attack_indicators = {}

        for i in current_steps:
            batch_size_i = batches_X[i].size(0)
            for j in range(batch_size_i):
                z = m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{j}")
                attack_indicators[(i, j)] = z
                all_indicators.append(z)

        m.addConstr(gp.quicksum(all_indicators) <= k_private, name="Budget")

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

                raw_g_W = m.addVar(lb=-float("inf"), name=f"raw_gW_{i}_{j}")
                m.addConstr(raw_g_W == 2 * (W_next * x_val + b_next - t_val) * x_val)

                raw_g_b = m.addVar(lb=-float("inf"), name=f"raw_gb_{i}_{j}")
                m.addConstr(raw_g_b == 2 * (W_next * x_val + b_next - t_val))

                # W Clip
                cw_1 = m.addVar(lb=-float("inf"))
                m.addGenConstrMin(cw_1, [raw_g_W], constant=clip_gamma)
                cw_2 = m.addVar(lb=-float("inf"))
                m.addGenConstrMax(cw_2, [cw_1], constant=-clip_gamma)
                clipped_g_W = cw_2

                # b Clip
                cb_1 = m.addVar(lb=-float("inf"))
                m.addGenConstrMin(cb_1, [raw_g_b], constant=clip_gamma)
                cb_2 = m.addVar(lb=-float("inf"))
                m.addGenConstrMax(cb_2, [cb_1], constant=-clip_gamma)
                clipped_g_b = cb_2

                z = attack_indicators[(i, j)]
                M = 2.0 * clip_gamma

                # u_W linearization (u = z * clipped)
                u_W = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(u_W <= clipped_g_W + M * (1 - z))
                m.addConstr(u_W >= clipped_g_W - M * (1 - z))
                m.addConstr(u_W <= M * z)
                m.addConstr(u_W >= -M * z)

                # v_W linearization (v = z * noise)
                v_W = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(v_W <= clip_gamma * z)
                m.addConstr(v_W >= -clip_gamma * z)

                eff_g_W = m.addVar(lb=-float("inf"))
                m.addConstr(eff_g_W == clipped_g_W - u_W + v_W)
                step_grads_W.append(eff_g_W)

                # Bias substitution
                u_b = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(u_b <= clipped_g_b + M * (1 - z))
                m.addConstr(u_b >= clipped_g_b - M * (1 - z))
                m.addConstr(u_b <= M * z)
                m.addConstr(u_b >= -M * z)

                v_b = m.addVar(lb=-clip_gamma, ub=clip_gamma)
                m.addConstr(v_b <= clip_gamma * z)
                m.addConstr(v_b >= -clip_gamma * z)

                eff_g_b = m.addVar(lb=-float("inf"))
                m.addConstr(eff_g_b == clipped_g_b - u_b + v_b)
                step_grads_b.append(eff_g_b)

            avg_g_W = m.addVar(lb=-float("inf"))
            m.addConstr(avg_g_W == gp.quicksum(step_grads_W) / batch_size_i)

            avg_g_b = m.addVar(lb=-float("inf"))
            m.addConstr(avg_g_b == gp.quicksum(step_grads_b) / batch_size_i)

            entries_w_next = m.addVar(lb=-float("inf"))
            entries_b_next = m.addVar(lb=-float("inf"))
            m.addConstr(entries_w_next == W_next - lr * avg_g_W)
            m.addConstr(entries_b_next == b_next - lr * avg_g_b)

            W_next = entries_w_next
            b_next = entries_b_next

        m.update()

        m.setObjective(W_next, gp.GRB.MAXIMIZE)
        m.optimize()
        if m.status != gp.GRB.OPTIMAL:
            return 999.0
        w_max = m.objVal

        m.setObjective(W_next, gp.GRB.MINIMIZE)
        m.optimize()
        if m.status != gp.GRB.OPTIMAL:
            return 999.0
        w_min = m.objVal

        m.setObjective(b_next, gp.GRB.MAXIMIZE)
        m.optimize()
        b_max = m.objVal

        m.setObjective(b_next, gp.GRB.MINIMIZE)
        m.optimize()
        b_min = m.objVal

        current_w_l, current_w_u = w_min, w_max
        current_b_l, current_b_u = b_min, b_max

    return current_w_u - current_w_l


def main():
    freqs = CFG["FREQS"]
    ks = CFG["K_VALUES"]

    final_output = {}

    for f in freqs:
        print(f"Running Manual F={f}")
        widths = []
        for k in ks:
            width = run_manual_mip(freq=f, k_private=k)
            widths.append(width)
            print(f"  k={k}: {width:.4f}")

        key = f"f{f}"
        final_output[key] = {"avg_V_k": widths, "k_values": ks}

    save_path = f"{CFG['OUT_DIR']}/manual_results.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_output)
    print(f"Saved manual results to {save_path}")


if __name__ == "__main__":
    main()
