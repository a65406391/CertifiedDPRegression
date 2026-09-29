"""
MNIST experiment: AGS Sampling Frequency Ablation.

Sweeps sampling frequency (SAMPLING_FREQ) keeping CONCRETIZATION_FREQ constant and records:
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
from common.models import create_mlp
from ags.mnist.ablations.config import SAMPLING_ABLATION as PURE_CFG
from ags.mnist.ablations.config import APPROX_DP_SAMPLING_ABLATION as APPROX_CFG

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


CFG = None


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
    eta = CFG.get("ETA", 0.0)
    clip_gamma = CFG["CLIP_GAMMA"] if args.clip_gamma is None else args.clip_gamma
    bound_mode = args.bound_mode
    device = args.device
    loss_name = CFG["LOSS"]
    hidden_dims = CFG["HIDDEN_DIMS"]
    output_dim = CFG["OUTPUT_DIM"]

    epsilon_target = args.epsilon_target
    eps_step = epsilon_target / total_steps
    sampling_freqs = args.sampling_freqs
    concretization_freq = (
        CFG["CONCRETIZATION_FREQ"] if args.concretization_freq is None else args.concretization_freq
    )
    privacy_mode = args.privacy_mode
    delta = args.delta

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    setup_file_logger(dataset="mnist", privacy_mode=privacy_mode, exp_type="sampling")

    logger.info("=" * 60)
    logger.info("AGS Sampling Frequency Ablation on MNIST")
    logger.info(
        "total_steps=%d  batch_size=%d  lr_ags=%.4f  eta=%.4f  clip=%.4f",
        total_steps,
        batch_size,
        lr_ags,
        eta,
        clip_gamma,
    )
    logger.info(
        "epsilon_target=%.3f  sampling_freqs=%s  privacy_mode=%s",
        epsilon_target,
        sampling_freqs,
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
    # Allow separate AGS / Laplace batch sizes (defaults to shared `batch_size`)
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
            "sampling_freqs": sampling_freqs,
            "k_subset": args.k_subset,
            "privacy_mode": privacy_mode,
            "delta": delta,
        },
        "nominal": [],
        "ags": {},
    }

    # 1. Nominal training
    logger.info("--- Nominal (no noise) baseline ---")
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

    # 2. AGS Ablation
    for freq in sampling_freqs:
        freq_key = str(freq)
        results["ags"][freq_key] = []
        logger.info("--- Sweep: sampling_freq=%d ---", freq)

        for seed_idx in range(n_seeds):
            seed = CFG["BASE_SEED"] + seed_idx
            set_seed(seed)
            model = create_mlp(input_dim, hidden_dims, output_dim)

            t0 = time.time()
            ags_model, ags_info = train_ags(
                model=model,
                train_loader=ags_loader,
                n_total_steps=total_steps,
                sampling_freq=freq,
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
            elapsed = time.time() - t0

            loss_val, acc_val = evaluate_model(ags_model, test_loader, loss_fn, device)

            # Record stats
            ss_vals = [w["ss_mean"] for w in ags_info["windows"]]
            avg_ss = float(np.mean(ss_vals)) if ss_vals else 0.0

            results["ags"][freq_key].append(
                {
                    "seed": seed,
                    "test_loss": loss_val,
                    "test_accuracy": acc_val,
                    "avg_ss_per_window": avg_ss,
                    "n_windows": ags_info["n_windows"],
                    "time_s": elapsed,
                    "sampling_freq": freq,
                }
            )

            logger.info(
                "  AGS freq=%4d  seed=%d  loss=%.4f  acc=%.4f  time=%.1fs  avg_ss=%.4e",
                freq,
                seed,
                loss_val,
                acc_val,
                elapsed,
                avg_ss,
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
    print("\n" + "=" * 80)
    print("SUMMARY (AGS Sampling Freq Ablation)")
    print("=" * 80)
    print(f"{'Method':<20} {'Loss':>8} {'Acc':>8} {'Time(s)':>8} {'S_Freq':>10}")
    print("-" * 80)

    nom_loss = np.mean([r["test_loss"] for r in results["nominal"]])
    nom_acc = np.mean([r["test_accuracy"] for r in results["nominal"]])
    print(f"{'Nominal':<20} {nom_loss:8.4f} {nom_acc:8.4f} {'N/A':>8} {'N/A':>10}")

    for freq in sampling_freqs:
        key = str(freq)
        ags_loss = np.mean([r["test_loss"] for r in results["ags"][key]])
        ags_acc = np.mean([r["test_accuracy"] for r in results["ags"][key]])
        ags_time = np.mean([r["time_s"] for r in results["ags"][key]])
        print(f"{'AGS':<20} {ags_loss:8.4f} {ags_acc:8.4f} {ags_time:8.1f} {freq:10d}")

    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGS Sampling Frequency Ablation on MNIST.")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--laplace-batch-size", type=int, default=None)
    parser.add_argument("--ags-batch-size", type=int, default=None)
    parser.add_argument("--lr-ags", type=float, default=None)
    parser.add_argument("--clip-gamma", type=float, default=None)
    parser.add_argument(
        "--bound-mode",
        type=str,
        default="crown",
        choices=["ibp", "crown", "crown-ibp"],
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--privacy-mode", type=str, default="pure", choices=["pure", "approx"])
    parser.add_argument("--epsilon-target", type=float, default=None)
    parser.add_argument("--k-subset", type=int, nargs="+", default=None)
    parser.add_argument("--sampling-freqs", type=int, nargs="+", default=[2500, 5000, 10000, 20000, 40000])
    parser.add_argument("--concretization-freq", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    # Slightly clunky but works
    CFG = PURE_CFG if args.privacy_mode == "pure" else APPROX_CFG

    if args.epsilon_target is None:
        args.epsilon_target = CFG["EPSILON_TOTALS_SAMPLING_FREQ"][-1]

    if args.k_subset is None:
        args.k_subset = CFG.get("K_SUBSET")

    if args.delta is None:
        args.delta = CFG["DELTA"]

    if args.out_dir is None:
        args.out_dir = CFG["OUT_DIR"]

    run_experiment(args)
