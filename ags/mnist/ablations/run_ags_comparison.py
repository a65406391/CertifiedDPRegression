"""
MNIST experiment: AGS vs Laplace DP-SGD.

Sweeps concretization frequency (s_p) and records:
  - Test cross-entropy loss
  - Test accuracy
  - Total privacy cost (ε_total)
  - Per-window smooth sensitivity diagnostics

Results are saved as a .json file for later plotting.

Usage:
    python -m ags.mnist.ablations.run_ags_comparison
    python -m ags.mnist.ablations.run_ags_comparison --total-steps 100 --n-seeds 1
"""

import argparse
import datetime
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common.ags import evaluate_model, setup_file_logger, train_ags
from common.baselines import train_laplace_dpsgd
from common.models import create_mlp
from ags.mnist.ablations.config import AGS_COMPARISON as CFG

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-8s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mnist(n_train=None, n_test=None, seed=42):
    """Load MNIST, flatten to 784-d, and return DataLoaders.

    Returns:
        train_loader, test_loader, (X_test_t, y_test_t)
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to 784
        ]
    )

    data_root = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "mnist",
    )

    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    # Subsample for speed
    rng = np.random.RandomState(seed)
    if n_train is not None and n_train < len(train_ds):
        idx = rng.choice(len(train_ds), n_train, replace=False)
        train_ds = Subset(train_ds, idx)
    if n_test is not None and n_test < len(test_ds):
        idx = rng.choice(len(test_ds), n_test, replace=False)
        test_ds = Subset(test_ds, idx)

    # Materialize to TensorDatasets for speed during many-step training
    from torch.utils.data import TensorDataset

    tx, ty = zip(*[train_ds[i] for i in range(len(train_ds))])
    train_ds = TensorDataset(torch.stack(tx), torch.tensor(ty))

    tx, ty = zip(*[test_ds[i] for i in range(len(test_ds))])
    test_ds = TensorDataset(torch.stack(tx), torch.tensor(ty))

    pca_dims = CFG.get("PCA_DIMS", 0)
    if pca_dims > 0:
        import logging

        from sklearn.decomposition import PCA

        logger = logging.getLogger(__name__)
        logger.info(f"Applying PCA to reduce dimensions from 784 to {pca_dims}")

        X_train, Y_train = zip(*[(x, y) for x, y in train_ds])
        X_train = torch.stack(X_train).numpy()
        Y_train = torch.tensor(Y_train)

        pca = PCA(n_components=pca_dims, whiten=True)
        X_train_pca = pca.fit_transform(X_train)

        X_test, Y_test = zip(*[(x, y) for x, y in test_ds])
        X_test = torch.stack(X_test).numpy()
        Y_test = torch.tensor(Y_test)

        X_test_pca = pca.transform(X_test)

        from torch.utils.data import TensorDataset

        train_ds = TensorDataset(torch.from_numpy(X_train_pca).float(), Y_train)
        test_ds = TensorDataset(torch.from_numpy(X_test_pca).float(), Y_test)

    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment(args):
    """Main experimental loop."""
    # Resolved with `is None` rather than `or`: the previous `args.x or CFG[...]` silently
    # discarded an explicit 0 or 0.0 on the command line in favour of the config value.
    total_steps = args.total_steps
    n_seeds = args.n_seeds
    batch_size = args.batch_size
    lr_ags = args.lr_ags
    lr_laplace = args.lr_laplace
    eta = CFG.get("ETA", 0.0)
    clip_gamma = args.clip_gamma
    bound_mode = args.bound_mode
    device = args.device
    loss_name = CFG["LOSS"]
    hidden_dims = CFG["HIDDEN_DIMS"]
    output_dim = CFG["OUTPUT_DIM"]

    epsilon_totals = args.epsilon_totals
    sampling_freq = args.sampling_freq
    concretization_freq = args.concretization_freq

    out_dir = CFG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    setup_file_logger(dataset="mnist", privacy_mode="pure", exp_type="performance")

    # per-method batch sizes (allow AGS/Laplace to differ; default to shared `batch_size`)
    laplace_batch_size = getattr(args, "laplace_batch_size", None) or batch_size
    ags_batch_size = getattr(args, "ags_batch_size", None) or batch_size

    logger.info("=" * 60)
    logger.info("AGS vs Laplace DP-SGD on MNIST")
    logger.info(
        "total_steps=%d  batch_size=%d  lr_ags=%.4f  lr_laplace=%.4f  eta=%.4f  clip=%.4f",
        total_steps,
        batch_size,
        lr_ags,
        lr_laplace,
        eta,
        clip_gamma,
    )
    logger.info("epsilon_totals=%s", epsilon_totals)
    logger.info("seeds=%d  device=%s", n_seeds, device)
    logger.info("=" * 60)

    # Load data once
    train_ds, test_ds = load_mnist(
        n_train=CFG["N_TRAIN"],
        n_test=CFG["N_TEST"],
        seed=42,
    )

    dataset_size = len(train_ds)
    laplace_loader = DataLoader(
        train_ds,
        batch_size=laplace_batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=None,  # ensure deterministic with single worker
        num_workers=0,
    )
    ags_loader = DataLoader(
        train_ds,
        batch_size=ags_batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=None,
        num_workers=0,
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()

    # ---- Results storage ----
    results = {
        "config": {
            "total_steps": total_steps,
            "batch_size": batch_size,
            "laplace_batch_size": laplace_batch_size,
            "ags_batch_size": ags_batch_size,
            "dataset_size": dataset_size,
            "lr_ags": lr_ags,
            "lr_laplace": lr_laplace,
            "eta": eta,
            "clip_gamma": clip_gamma,
            "bound_mode": bound_mode,
            "hidden_dims": hidden_dims,
            "epsilon_totals": epsilon_totals,
            "n_seeds": n_seeds,
            "k_subset": args.k_subset,
        },
        "ags": {},  # keyed by eps_total -> list of per-seed results
        "laplace": {},  # keyed by eps_total -> list of per-seed results
        "nominal": {},  # non-private baseline
    }

    # ---- Nominal (non-private) baseline ----
    pca_dims = CFG.get("PCA_DIMS", 0)
    input_dim = pca_dims if pca_dims > 0 else 784

    logger.info("--- Nominal (no noise) baseline ---")
    nom_results = []
    for seed_idx in range(n_seeds):
        seed = CFG["BASE_SEED"] + seed_idx
        set_seed(seed)
        model = create_mlp(input_dim, hidden_dims, output_dim)
        # Use standard SGD (no noise, no clipping) for nominal
        nom_model, nom_info = train_laplace_dpsgd(
            model,
            laplace_loader,
            n_total_steps=total_steps,
            epsilon_per_step=float("inf"),  # effectively no noise
            lr=lr_laplace,
            clip_gamma=clip_gamma,
            loss_fn=loss_fn,
            device=device,
            seed=seed,
        )
        loss_val, acc_val = evaluate_model(
            nom_model,
            test_loader,
            loss_fn,
            device,
        )
        nom_results.append(
            {
                "seed": seed,
                "test_loss": loss_val,
                "test_accuracy": acc_val,
                "epsilon_total": 0.0,  # non-private
            }
        )
        logger.info(
            "  Nominal seed=%d  loss=%.4f  acc=%.4f",
            seed,
            loss_val,
            acc_val,
        )
    results["nominal"] = nom_results

    # ---- Sweep over Epsilon Totals ----
    for eps in epsilon_totals:
        eps_key = str(eps)
        eps_step = eps / total_steps

        logger.info("--- Sweep: eps_total=%.2f ---", eps)

        # --- Laplace ---
        lap_results = []
        for seed_idx in range(n_seeds):
            seed = CFG["BASE_SEED"] + seed_idx
            set_seed(seed)
            model = create_mlp(input_dim, hidden_dims, output_dim)

            t0 = time.time()
            lap_model, lap_info = train_laplace_dpsgd(
                model,
                laplace_loader,
                n_total_steps=total_steps,
                epsilon_per_step=eps_step,
                lr=lr_laplace,
                clip_gamma=clip_gamma,
                loss_fn=loss_fn,
                device=device,
                seed=seed,
            )
            elapsed = time.time() - t0
            loss_val, acc_val = evaluate_model(lap_model, test_loader, loss_fn, device)
            lap_results.append(
                {
                    "seed": seed,
                    "test_loss": loss_val,
                    "test_accuracy": acc_val,
                    "epsilon_total": eps,
                    "noise_scale": lap_info["noise_scale"],
                    "time_s": elapsed,
                }
            )
            logger.info("  Laplace seed=%d  loss=%.4f  acc=%.4f  (%.1fs)", seed, loss_val, acc_val, elapsed)
        results["laplace"][eps_key] = lap_results

        # --- AGS ---
        ags_results = []
        for seed_idx in range(n_seeds):
            seed = CFG["BASE_SEED"] + seed_idx
            set_seed(seed)
            model = create_mlp(input_dim, hidden_dims, output_dim)

            t0 = time.time()
            # For minimal composition penalty, we use 1 window
            s_p = total_steps
            ags_model, ags_info = train_ags(
                model,
                ags_loader,
                n_total_steps=total_steps,
                sampling_freq=sampling_freq,
                concretization_freq=concretization_freq,
                lr=lr_ags,
                clip_gamma=clip_gamma,
                loss=loss_name,
                bound_mode=bound_mode,
                device=device,
                seed=seed,
                epsilon_per_step=eps_step,
                eta=eta,
                k_subset=args.k_subset,
            )
            elapsed = time.time() - t0
            loss_val, acc_val = evaluate_model(ags_model, test_loader, loss_fn, device)
            ags_results.append(
                {
                    "seed": seed,
                    "test_loss": loss_val,
                    "test_accuracy": acc_val,
                    "epsilon_total": eps,
                    "n_windows": ags_info["n_windows"],
                    "avg_ss_per_window": np.mean([w["ss_mean"] for w in ags_info["windows"]])
                    if ags_info["windows"]
                    else 0.0,
                    "time_s": elapsed,
                }
            )
            logger.info("  AGS     seed=%d  loss=%.4f  acc=%.4f  (%.1fs)", seed, loss_val, acc_val, elapsed)
        results["ags"][eps_key] = ags_results

    # ---- Save ----
    out_path = os.path.join(out_dir, "results.json")

    run_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": results["config"],
        "ags": results["ags"],
        "laplace": results["laplace"],
        "nominal": results["nominal"],
    }

    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                master_data = json.load(f)
            if "runs" not in master_data:
                # Migrate old schema
                master_data = {"runs": [master_data]}
        except Exception:
            master_data = {"runs": []}
    else:
        master_data = {"runs": []}

    master_data["runs"].append(run_record)

    with open(out_path, "w") as f:
        json.dump(master_data, f, indent=2)
    logger.info("Results appended to %s", out_path)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<20} {'Loss':>8} {'Acc':>8} {'ε_total':>10} {'Winner':>10}")
    print("-" * 82)

    nom_loss = np.mean([r["test_loss"] for r in results["nominal"]])
    nom_acc = np.mean([r["test_accuracy"] for r in results["nominal"]])
    print(f"{'Nominal':<20} {nom_loss:8.4f} {nom_acc:8.4f} {'N/A':>10} {'-':>10}")

    for eps in epsilon_totals:
        key = str(eps)

        lap_loss = np.mean([r["test_loss"] for r in results["laplace"][key]])
        lap_acc = np.mean([r["test_accuracy"] for r in results["laplace"][key]])

        ags_loss = np.mean([r["test_loss"] for r in results["ags"][key]])
        ags_acc = np.mean([r["test_accuracy"] for r in results["ags"][key]])

        winner = "AGS" if ags_acc > lap_acc else "Laplace"
        if abs(ags_acc - lap_acc) < 1e-4:
            winner = "Tie"

        print(f"{'Laplace eps=' + key:<20} {lap_loss:8.4f} {lap_acc:8.4f} {eps:10.2f} {'':>10}")
        print(f"{'AGS     eps=' + key:<20} {ags_loss:8.4f} {ags_acc:8.4f} {eps:10.2f} {winner:>10}")
        print("-" * 82)

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AGS vs Laplace DP-SGD on MNIST",
    )
    parser.add_argument("--total-steps", type=int, default=CFG["TOTAL_STEPS"])
    parser.add_argument("--n-seeds", type=int, default=CFG["N_SEEDS"])
    parser.add_argument("--batch-size", type=int, default=CFG["BATCH_SIZE"])
    parser.add_argument("--laplace-batch-size", type=int, default=None)
    parser.add_argument("--ags-batch-size", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=CFG["N_TRAIN"])
    parser.add_argument("--pca-dims", type=int, default=CFG.get("PCA_DIMS", 0))
    parser.add_argument("--sampling-freq", type=int, default=CFG["SAMPLING_FREQ"])
    parser.add_argument("--concretization-freq", type=int, default=CFG["CONCRETIZATION_FREQ"])
    parser.add_argument("--lr-ags", type=float, default=CFG["LR_AGS"])
    parser.add_argument("--lr-laplace", type=float, default=CFG["LR_LAPLACE"])
    parser.add_argument("--clip-gamma", type=float, default=CFG["CLIP_GAMMA"])
    parser.add_argument("--epsilon-totals", type=float, nargs="+", default=CFG["EPSILON_TOTALS"])
    parser.add_argument("--k-subset", type=int, nargs="+", default=CFG.get("K_SUBSET"))
    parser.add_argument(
        "--bound-mode",
        type=str,
        default=CFG.get("BOUND_MODE", "crown"),
        choices=["ibp", "crown", "miqp", "milp"],
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Update CFG with overrides for functions like load_mnist
    if args.pca_dims is not None:
        CFG["PCA_DIMS"] = args.pca_dims
    if args.n_train is not None:
        CFG["N_TRAIN"] = args.n_train

    run_experiment(args)


if __name__ == "__main__":
    main()
