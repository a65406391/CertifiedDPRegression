# LEGACY -- superseded, kept so the figures and stored values it produced stay reproducible.
# It uses the retired conventions: the envelope width or radius rather than the cross-radius
# certificate, a fixed beta rather than one derived from the budget, and in places Gaussian noise
# on smooth sensitivity, which no theorem in the paper covers. Do not build new work on it.
# The current AGT-R path is agtr/linear and agtr/california; see Notes/REPO_NOTES.md sections 5-7.
"""Diagnostic script to investigate V_k non-monotonicity for F=8 and F=16.

Runs certified_training with verbose Gurobi output and captures solver status,
optimality gaps, runtime, and model statistics for each parameter bound solve.
"""

import torch
import numpy as np
import logging

from abstract_gradient_training_opt.certified_training import (
    formulate_model_and_data,
    bound_parameters,
    add_batch_to_formulation,
    compute_batch_ibp,
)
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training_opt import training_utils

from agtr.linear.config import ABLATION as CFG
from common.data import simulated_linear_data, get_or_create_w_true
from common.models import create_linear_model, get_or_create_init_state

import gurobipy as gp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_mip_solve(method, k):
    """Run certified training and capture detailed Gurobi solve info."""
    freq = CFG["FREQ_MAP"].get(method, 1)
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    def create_model():
        return create_linear_model(CFG["INPUT_DIM"], 1, dtype=torch.float64)

    init_state = get_or_create_init_state(create_model, path=CFG["INIT_STATE_PATH"], seed=seed)
    model = create_model()
    model.load_state_dict(init_state)

    # Deterministic data shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    w_true = get_or_create_w_true(CFG["INPUT_DIM"], CFG["SPARSITY"], path=CFG["W_TRUE_PATH"])
    X_train, y_train = simulated_linear_data(
        n_samples=CFG["N_SAMPLES"],
        dim=CFG["INPUT_DIM"],
        sparsity=CFG["SPARSITY"],
        seed=42,
        w_true=w_true,
        w_true_path=CFG["W_TRUE_PATH"],
    )

    perm = torch.randperm(len(X_train), generator=g)
    X_train_shuffled = torch.tensor(X_train[perm], dtype=torch.float64)
    y_train_shuffled = torch.tensor(y_train[perm], dtype=torch.float64)
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(X_train_shuffled, y_train_shuffled)

    config = OPTConfig(
        log_level="DEBUG",  # Verbose logging
        n_epochs=CFG["EPOCHS"],
        batchsize=CFG["BATCH_SIZE"],
        learning_rate=CFG["LR"],
        loss="mse",
        concretization_freq=freq,
        k_substitution=k,
        input_domain=None,
        label_domain=None,
        gradient_clipping=CFG["CLIP_GAMMA"],
        relax_binaries=False,
        relax_bilinear=False,
        gurobi_kwargs={"NonConvex": 2, "OutputFlag": 1, "Threads": 1, "TimeLimit": 600},
        metadata={"seed": seed, "strict_mode": True},
    )

    print(f"\n{'=' * 80}")
    print(f"DIAGNOSING: method={method}, k={k}, freq={freq}")
    print(f"{'=' * 80}")

    # Run certified training but with instrumented bound_parameters
    model_double = model.to(torch.float64)
    data, labels = dataset.tensors
    data_np = data.detach().cpu().clone().double().numpy()
    labels_np = labels.detach().cpu().clone().double().numpy()

    train_dataset = TensorDataset(torch.tensor(data_np), torch.tensor(labels_np), torch.arange(len(data_np)))
    generator = torch.Generator().manual_seed(config.metadata.get("seed", 0))
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batchsize, shuffle=False, generator=generator
    )

    ibp_model = IntervalBoundedModel(model_double, interval_matmul="exact")
    formulation, mip_model, perturbed_data = formulate_model_and_data(config, ibp_model, data_np, labels_np)

    n_batches = len(dataset) // config.batchsize
    iter_ = 0
    learning_rate = config.learning_rate

    for e in range(config.n_epochs):
        for b, (x, y, idxs) in enumerate(dataloader):
            data_slice = idxs.numpy()
            mip_model = add_batch_to_formulation(
                formulation, mip_model, perturbed_data, data_slice, b, config, learning_rate
            )
            ibp_model = compute_batch_ibp(
                ibp_model, data_np, labels_np, data_slice, config, learning_rate, dataset_val=None
            )
            mip_model.add_parameter_bounds(
                [p.cpu().numpy() for p in ibp_model.param_l],
                [p.cpu().numpy() for p in ibp_model.param_n],
                [p.cpu().numpy() for p in ibp_model.param_u],
            )
            iter_ += 1

            # Concretize if needed
            if (
                config.concretization_freq != -1
                and iter_ % config.concretization_freq == 0
                and iter_ != config.n_epochs * n_batches
            ):
                print(f"\n--- CONCRETIZING at iter={iter_} (epoch={e}, batch={b}) ---")

                # Get model stats before solving
                formulation.update()
                stats = training_utils.get_gurobi_model_stats(formulation)
                print(stats)

                params_l, params_n, params_u = bound_parameters(mip_model)

                # Capture solve details
                print(f"  Solve Status: {formulation.Status}")
                print(f"  Solve Time: {formulation.Runtime:.2f}s")
                try:
                    print(f"  MIP Gap: {formulation.MIPGap:.6f}")
                except:
                    print(f"  MIP Gap: N/A")
                try:
                    print(f"  ObjBound: {formulation.ObjBound:.6f}")
                except:
                    pass

                for i, (pl, pu) in enumerate(zip(params_l, params_n)):
                    w = params_u[i] - params_l[i]
                    print(
                        f"  Param {i}: bounds=[{params_l[i].flatten()}, {params_u[i].flatten()}], width={w.flatten()}"
                    )

                ibp_model.apply_parameter_update(params_l, params_n, params_u)
                formulation, mip_model, perturbed_data = formulate_model_and_data(
                    config, ibp_model, data_np, labels_np
                )

    # Final concretization
    print(f"\n--- FINAL CONCRETIZATION (iter={iter_}) ---")
    formulation.update()
    stats = training_utils.get_gurobi_model_stats(formulation)
    print(stats)

    # Instrument the final solve to capture per-objective status
    params = mip_model.get_parameters()
    results = {}
    for i, param in enumerate(params):
        formulation.update()
        obj = param.v
        for idx in range(obj.size):
            element = obj[np.unravel_index(idx, obj.shape)]

            # MINIMIZE
            formulation.setObjective(element, gp.GRB.MINIMIZE)
            formulation.reset()
            formulation.optimize()
            min_status = formulation.Status
            min_time = formulation.Runtime
            min_val = (
                formulation.objVal
                if min_status == gp.GRB.OPTIMAL
                else getattr(formulation, "objBound", -np.inf)
            )
            try:
                min_gap = formulation.MIPGap
            except:
                min_gap = None

            # MAXIMIZE
            formulation.setObjective(element, gp.GRB.MAXIMIZE)
            formulation.reset()
            formulation.optimize()
            max_status = formulation.Status
            max_time = formulation.Runtime
            max_val = (
                formulation.objVal
                if max_status == gp.GRB.OPTIMAL
                else getattr(formulation, "objBound", np.inf)
            )
            try:
                max_gap = formulation.MIPGap
            except:
                max_gap = None

            status_map = {
                2: "OPTIMAL",
                3: "INFEASIBLE",
                9: "TIME_LIMIT",
                11: "INTERRUPTED",
                13: "SUBOPTIMAL",
            }
            min_status_str = status_map.get(min_status, f"STATUS_{min_status}")
            max_status_str = status_map.get(max_status, f"STATUS_{max_status}")

            print(f"  Param {i}, elem {idx}:")
            print(f"    MIN: val={min_val:.8f}, status={min_status_str}, time={min_time:.2f}s, gap={min_gap}")
            print(f"    MAX: val={max_val:.8f}, status={max_status_str}, time={max_time:.2f}s, gap={max_gap}")
            print(f"    Width: {max_val - min_val:.8f}")

    # Also get IBP bounds for comparison
    print(f"\n  IBP bounds for comparison:")
    for i in range(len(ibp_model.param_l)):
        print(f"    Param {i}: [{ibp_model.param_l[i].flatten()}, {ibp_model.param_u[i].flatten()}]")
        print(f"    Width: {(ibp_model.param_u[i] - ibp_model.param_l[i]).flatten()}")


if __name__ == "__main__":
    # Run diagnosis for the problematic cases
    cases = [
        ("f8", 1),
        ("f8", 2),
        ("f16", 1),
        ("f16", 2),
    ]
    for method, k in cases:
        diagnose_mip_solve(method, k)
