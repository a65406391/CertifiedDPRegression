"""
MNIST experiment: Hybrid Laplace+AGS Ablation.

Sweeps number of AGS steps vs Laplace steps and records:
  - Test cross-entropy loss
  - Test accuracy
  - Time elapsed
  - Per-window smooth sensitivity diagnostics

Results are saved as a .json file for later plotting.
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
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from common.ags import evaluate_model, setup_file_logger, train_ags
from common.baselines import train_laplace_dpsgd, train_gaussian_dpsgd
from common.models import create_mlp
from ags.mnist.ablations.config import HYBRID_ABLATION as PURE_CFG
from ags.mnist.ablations.config import APPROX_DP_HYBRID_ABLATION as APPROX_CFG

CFG = None

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
    """Load MNIST, flatten to 784-d, and return datasets."""
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
    full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # optionally subset
    if n_train is not None and n_train < len(full_train):
        set_seed(seed)
        indices = torch.randperm(len(full_train))[:n_train].tolist()
        train_ds = Subset(full_train, indices)
    else:
        train_ds = full_train

    if n_test is not None and n_test < len(full_test):
        set_seed(seed + 1)
        indices = torch.randperm(len(full_test))[:n_test].tolist()
        test_ds = Subset(full_test, indices)
    else:
        test_ds = full_test

    # Materialize to TensorDatasets for speed during many-step training
    # This prevents torchvision from applying PIL transforms dynamically on every batch
    tx, ty = zip(*[train_ds[i] for i in range(len(train_ds))])
    train_ds = TensorDataset(torch.stack(tx), torch.tensor(ty))

    tx, ty = zip(*[test_ds[i] for i in range(len(test_ds))])
    test_ds = TensorDataset(torch.stack(tx), torch.tensor(ty))

    pca_dims = CFG.get("PCA_DIMS", 0)
    if pca_dims > 0:
        import logging
        from sklearn.decomposition import PCA

        local_logger = logging.getLogger(__name__)
        local_logger.info(f"Applying PCA to reduce dimensions from 784 to {pca_dims}")

        X_train, Y_train = zip(*[(x, y) for x, y in train_ds])
        X_train = torch.stack(X_train).numpy()
        Y_train = torch.tensor(Y_train)

        pca = PCA(n_components=pca_dims, whiten=True)
        X_train_pca = pca.fit_transform(X_train)

        X_test, Y_test = zip(*[(x, y) for x, y in test_ds])
        X_test = torch.stack(X_test).numpy()
        Y_test = torch.tensor(Y_test)

        X_test_pca = pca.transform(X_test)

        train_ds = TensorDataset(torch.from_numpy(X_train_pca).float(), Y_train)
        test_ds = TensorDataset(torch.from_numpy(X_test_pca).float(), Y_test)

    return train_ds, test_ds


def get_dataloaders(train_ds, test_ds, batch_size):
    """Wrap datasets in DataLoaders."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(5000, len(test_ds)),  # Evaluate in chunks if needed
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(args):
    """Main experimental loop."""
    # Resolved with `is None` rather than `or`: the previous `args.x or CFG[...]` silently
    # discarded an explicit 0 or 0.0 on the command line in favour of the config value.
    total_steps = CFG["TOTAL_STEPS"] if args.total_steps is None else args.total_steps
    n_seeds = args.n_seeds
    batch_size = CFG["BATCH_SIZE"] if args.batch_size is None else args.batch_size
    lr_ags = CFG["LR_AGS"] if args.lr_ags is None else args.lr_ags
    lr_laplace = CFG["LR_LAPLACE"] if args.lr_laplace is None else args.lr_laplace
    eta = CFG.get("ETA", 0.0)
    clip_gamma = CFG["CLIP_GAMMA"] if args.clip_gamma is None else args.clip_gamma
    bound_mode = args.bound_mode
    device = args.device
    loss_name = CFG["LOSS"]
    hidden_dims = CFG["HIDDEN_DIMS"]
    output_dim = CFG["OUTPUT_DIM"]

    epsilon_target = args.epsilon_target
    eps_step = epsilon_target / total_steps
    ags_steps_list = args.ags_steps_list
    concretization_freq = (
        CFG["CONCRETIZATION_FREQ"] if args.concretization_freq is None else args.concretization_freq
    )
    privacy_mode = args.privacy_mode
    delta = args.delta

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    setup_file_logger(dataset="mnist", privacy_mode=privacy_mode, exp_type="hybrid")

    logger.info("=" * 60)
    logger.info("Hybrid Laplace+AGS Ablation on MNIST")
    logger.info(
        "total_steps=%d  batch_size=%d  lr_ags=%.4f  eta=%.4f  clip=%.4f",
        total_steps,
        batch_size,
        lr_ags,
        eta,
        clip_gamma,
    )
    logger.info(
        "epsilon_target=%.3f  ags_steps_list=%s  privacy_mode=%s",
        epsilon_target,
        ags_steps_list,
        privacy_mode,
    )
    logger.info("seeds=%d  device=%s", n_seeds, device)
    logger.info("=" * 60)

    # Load data once
    train_ds, test_ds = load_mnist(
        n_train=CFG["N_TRAIN"],
        n_test=CFG["N_TEST"],
        seed=CFG["BASE_SEED"],
    )
    # Support per-method batch sizes (allow AGS/Laplace to differ; default to `batch_size`)
    laplace_batch_size = getattr(args, "laplace_batch_size", None) or batch_size
    ags_batch_size = getattr(args, "ags_batch_size", None) or batch_size
    dataset_size = len(train_ds)

    laplace_loader = DataLoader(train_ds, batch_size=laplace_batch_size, shuffle=True, drop_last=False)
    ags_loader = DataLoader(train_ds, batch_size=ags_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=min(5000, len(test_ds)), shuffle=False)

    pca_dims = CFG.get("PCA_DIMS", 0)
    input_dim = pca_dims if pca_dims > 0 else 784

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "batch_size": batch_size,
            "laplace_batch_size": laplace_batch_size,
            "ags_batch_size": ags_batch_size,
            "dataset_size": dataset_size,
            "total_steps": total_steps,
            "lr_ags": lr_ags,
            "clip_gamma": clip_gamma,
            "epsilon_target": epsilon_target,
            "bound_mode": bound_mode,
            "n_seeds": n_seeds,
            "eta": eta,
            "loss": loss_name,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "concretization_freq": concretization_freq,
            "ags_steps_list": ags_steps_list,
            "lr_laplace": lr_laplace,
            "k_subset": args.k_subset,
            "privacy_mode": privacy_mode,
            "delta": delta,
        },
        "nominal": [],
        "ags_first": {},
        "laplace_first": {},
    }

    # 1. Nominal training
    loss_fn = nn.CrossEntropyLoss()
    for seed_idx in range(n_seeds):
        seed = CFG["BASE_SEED"] + seed_idx
        set_seed(seed)
        model = create_mlp(input_dim, hidden_dims, output_dim).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG["LR_LAPLACE"])

        # Train
        model.train()
        train_iter = iter(laplace_loader)
        for _ in tqdm(range(total_steps), desc=f"Nominal Seed {seed}"):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(laplace_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            l = loss_fn(model(x), y)
            l.backward()
            optimizer.step()

        # Evaluate
        loss_val, acc_val = evaluate_model(model, test_loader, loss_fn, device)
        results["nominal"].append(
            {
                "seed": seed,
                "test_loss": loss_val,
                "test_accuracy": acc_val,
            }
        )
        logger.info("  Nominal seed=%d  loss=%.4f  acc=%.4f", seed, loss_val, acc_val)

    # 2. Hybrid Ablation (AGS first vs Laplace first)
    for first_phase_steps in ags_steps_list:
        key = str(first_phase_steps)
        results["ags_first"][key] = []
        results["laplace_first"][key] = []
        logger.info("--- Sweep: first_phase_steps=%d ---", first_phase_steps)

        for seed_idx in range(n_seeds):
            seed = CFG["BASE_SEED"] + seed_idx

            # (A) Schedule: AGS first -> Laplace second
            set_seed(seed)
            model = create_mlp(input_dim, hidden_dims, output_dim).to(device)
            t0 = time.time()

            # Phase 1: AGS
            if first_phase_steps > 0:
                model, ags_info = train_ags(
                    model=model,
                    train_loader=ags_loader,
                    n_total_steps=first_phase_steps,
                    sampling_freq=first_phase_steps,
                    concretization_freq=concretization_freq,
                    epsilon_per_step=eps_step,
                    lr=lr_ags,
                    clip_gamma=clip_gamma,
                    loss=loss_name,
                    bound_mode=bound_mode,
                    device=device,
                    seed=seed,
                    eta=eta,
                    k_subset=args.k_subset,
                    privacy_mode=privacy_mode,
                    delta=delta if privacy_mode == "approx" else None,
                )
            else:
                ags_info = {"n_windows": 0, "windows": []}

            # Phase 2: DP-SGD
            phase_2_steps = total_steps - first_phase_steps
            if phase_2_steps > 0:
                if privacy_mode == "approx":
                    model, _ = train_gaussian_dpsgd(
                        model=model,
                        train_loader=laplace_loader,
                        n_total_steps=phase_2_steps,
                        epsilon_per_step=eps_step,
                        delta=delta,
                        lr=lr_laplace,
                        clip_gamma=clip_gamma,
                        loss_fn=loss_fn,
                        device=device,
                        seed=seed,
                    )
                else:
                    model, _ = train_laplace_dpsgd(
                        model=model,
                        train_loader=laplace_loader,
                        n_total_steps=phase_2_steps,
                        epsilon_per_step=eps_step,
                        lr=lr_laplace,
                        clip_gamma=clip_gamma,
                        loss_fn=loss_fn,
                        device=device,
                        seed=seed,
                    )
            elapsed_ags_first = time.time() - t0
            loss_val_ags, acc_val_ags = evaluate_model(model, test_loader, loss_fn, device)

            # Record stats
            l1_ss_vals_ags = [w["ss_mean"] for w in ags_info["windows"]]
            avg_ss_ags = float(np.mean(l1_ss_vals_ags)) if l1_ss_vals_ags else 0.0

            results["ags_first"][key].append(
                {
                    "seed": seed,
                    "test_loss": loss_val_ags,
                    "test_accuracy": acc_val_ags,
                    "avg_ss_per_window": avg_ss_ags,
                    "n_windows": ags_info["n_windows"],
                    "time_s": elapsed_ags_first,
                    "first_phase_steps": first_phase_steps,
                }
            )

            # (B) Schedule: Laplace first -> AGS second
            set_seed(seed)
            model = create_mlp(input_dim, hidden_dims, output_dim).to(device)
            t0 = time.time()

            # Phase 1: DP-SGD
            if first_phase_steps > 0:
                if privacy_mode == "approx":
                    model, _ = train_gaussian_dpsgd(
                        model=model,
                        train_loader=laplace_loader,
                        n_total_steps=first_phase_steps,
                        epsilon_per_step=eps_step,
                        delta=delta,
                        lr=lr_laplace,
                        clip_gamma=clip_gamma,
                        loss_fn=loss_fn,
                        device=device,
                        seed=seed,
                    )
                else:
                    model, _ = train_laplace_dpsgd(
                        model=model,
                        train_loader=laplace_loader,
                        n_total_steps=first_phase_steps,
                        epsilon_per_step=eps_step,
                        lr=lr_laplace,
                        clip_gamma=clip_gamma,
                        loss_fn=loss_fn,
                        device=device,
                        seed=seed,
                    )

            # Phase 2: AGS
            ags_steps_2 = total_steps - first_phase_steps
            if ags_steps_2 > 0:
                model, ags_info = train_ags(
                    model=model,
                    train_loader=ags_loader,
                    n_total_steps=ags_steps_2,
                    sampling_freq=ags_steps_2,
                    concretization_freq=concretization_freq,
                    epsilon_per_step=eps_step,
                    lr=lr_ags,
                    clip_gamma=clip_gamma,
                    loss=loss_name,
                    bound_mode=bound_mode,
                    device=device,
                    seed=seed,
                    eta=eta,
                    k_subset=args.k_subset,
                    privacy_mode=privacy_mode,
                    delta=delta if privacy_mode == "approx" else None,
                )
            else:
                ags_info = {"n_windows": 0, "windows": []}
            elapsed_laplace_first = time.time() - t0
            loss_val_lap, acc_val_lap = evaluate_model(model, test_loader, loss_fn, device)

            # Record stats
            l1_ss_vals_lap = [w["ss_mean"] for w in ags_info["windows"]]
            avg_ss_lap = float(np.mean(l1_ss_vals_lap)) if l1_ss_vals_lap else 0.0

            results["laplace_first"][key].append(
                {
                    "seed": seed,
                    "test_loss": loss_val_lap,
                    "test_accuracy": acc_val_lap,
                    "avg_ss_per_window": avg_ss_lap,
                    "n_windows": ags_info["n_windows"],
                    "time_s": elapsed_laplace_first,
                    "first_phase_steps": first_phase_steps,
                }
            )

            logger.info(
                "  Steps=%4d | AGS-1st: loss=%.4f acc=%.4f ss=%.4f | Lap-1st: loss=%.4f acc=%.4f ss=%.4f",
                first_phase_steps,
                loss_val_ags,
                acc_val_ags,
                avg_ss_ags,
                loss_val_lap,
                acc_val_lap,
                avg_ss_lap,
            )

    out_path = os.path.join(out_dir, "results.json")

    # Read existing
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                master_data = json.load(f)
            if "runs" not in master_data:
                master_data = {"runs": [master_data]}
        except Exception:
            master_data = {"runs": []}
    else:
        master_data = {"runs": []}

    master_data["runs"].append(results)

    # Write out
    with open(out_path, "w") as f:
        json.dump(master_data, f, indent=2)

    logger.info("Results appended to %s", out_path)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY (Hybrid Ablation: AGS vs Laplace)")
    print("=" * 90)
    print(
        f"{'Method':<20} {'Steps':>8} {'AGS-1st Loss':>13} {'AGS-1st Acc':>13} {'Lap-1st Loss':>13} {'Lap-1st Acc':>13}"
    )
    print("-" * 90)

    nom_loss = np.mean([r["test_loss"] for r in results["nominal"]])
    nom_acc = np.mean([r["test_accuracy"] for r in results["nominal"]])
    print(f"{'Nominal':<20} {'N/A':>8} {nom_loss:13.4f} {nom_acc:13.4f} {nom_loss:13.4f} {nom_acc:13.4f}")

    for s in ags_steps_list:
        key = str(s)
        loss_ags = np.mean([r["test_loss"] for r in results["ags_first"][key]])
        acc_ags = np.mean([r["test_accuracy"] for r in results["ags_first"][key]])
        loss_lap = np.mean([r["test_loss"] for r in results["laplace_first"][key]])
        acc_lap = np.mean([r["test_accuracy"] for r in results["laplace_first"][key]])
        print(f"{'Hybrid':<20} {s:8d} {loss_ags:13.4f} {acc_ags:13.4f} {loss_lap:13.4f} {acc_lap:13.4f}")

    print("-" * 90)


def main():
    global CFG
    parser = argparse.ArgumentParser(description="Hybrid Laplace+AGS Ablation on MNIST.")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-ags", type=float, default=None)
    parser.add_argument("--clip-gamma", type=float, default=None)
    parser.add_argument(
        "--bound-mode",
        type=str,
        default="crown",
        choices=["ibp", "crown", "crown-ibp"],
    )
    parser.add_argument("--laplace-batch-size", type=int, default=None)
    parser.add_argument("--ags-batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epsilon-target", type=float, default=None)
    parser.add_argument("--k-subset", type=int, nargs="+", default=None)
    parser.add_argument("--ags-steps-list", type=int, nargs="+", default=None)
    parser.add_argument("--lr-laplace", type=float, default=None)
    parser.add_argument("--concretization-freq", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--privacy-mode", type=str, default="pure", choices=["pure", "approx"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    CFG = PURE_CFG if args.privacy_mode == "pure" else APPROX_CFG

    if args.epsilon_target is None:
        args.epsilon_target = CFG["EPSILON_TOTALS_SAMPLING_FREQ"][-1]
    if args.k_subset is None:
        args.k_subset = CFG.get("K_SUBSET")
    if args.ags_steps_list is None:
        args.ags_steps_list = CFG["HYBRID_STEPS_LIST"]
    if args.lr_laplace is None:
        args.lr_laplace = CFG["LR_LAPLACE"]
    if args.delta is None:
        args.delta = CFG["DELTA"]
    if args.out_dir is None:
        args.out_dir = CFG["OUT_DIR"]

    run_experiment(args)


if __name__ == "__main__":
    main()
