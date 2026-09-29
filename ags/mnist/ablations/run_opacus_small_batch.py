"""Opacus DP-SGD on MNIST with small batch size for privacy amplification by subsampling."""

import math
import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from opacus import PrivacyEngine
from ags.mnist.ablations.run_approx_dp_comparison import load_mnist
from ags.mnist.ablations.config import APPROX_DP_COMPARISON as CFG

logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] %(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SMALL_BATCH_SIZE = 256  # Much smaller than dataset for privacy amplification
STEP_MULTIPLIER = 2  # Train 2x longer to compensate for smaller batches


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    train_ds, test_ds = load_mnist(CFG["N_TRAIN"], CFG["N_TEST"], CFG["BASE_SEED"])
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    dataset_size = len(train_ds)
    q = SMALL_BATCH_SIZE / dataset_size
    logger.info(f"Small batch size: {SMALL_BATCH_SIZE}, dataset: {dataset_size}, sampling rate q={q:.4f}")

    # Give Opacus more training time with small batches.
    # With batch_size=256 vs original 2000, each step sees ~8x less data.
    # We train 10x longer to give the optimizer more opportunity to converge.
    batches_per_epoch = math.ceil(CFG["N_TRAIN"] / SMALL_BATCH_SIZE)
    base_epochs = CFG["TOTAL_STEPS"] // batches_per_epoch
    epochs = base_epochs * STEP_MULTIPLIER
    actual_steps = epochs * batches_per_epoch
    logger.info(
        f"Training: {epochs} epochs x {batches_per_epoch} batches = {actual_steps} steps "
        f"(base: {base_epochs} epochs x {STEP_MULTIPLIER}x multiplier)"
    )

    # max_grad_norm scaled to match element-wise clipping
    dummy_model = nn.Linear(64, 10)
    n_params = sum(p.numel() for p in dummy_model.parameters())
    max_grad_norm = CFG["CLIP_GAMMA"] * math.sqrt(n_params)
    logger.info(f"max_grad_norm: {max_grad_norm:.4f}")

    results = {}
    pbar = tqdm(CFG["EPSILON_TOTALS"], desc="MNIST Opacus (small batch)")
    for eps in pbar:
        set_seed(CFG["BASE_SEED"])
        model = nn.Linear(64, 10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG["LR_LAPLACE"])
        train_loader = DataLoader(train_ds, batch_size=SMALL_BATCH_SIZE, shuffle=True)

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=eps,
            target_delta=CFG["DELTA"],
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

        model.train()
        for epoch in trange(epochs, desc=f"eps={eps}", leave=False):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(x), y)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=-1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        results[str(eps)] = acc
        logger.info(f"Epsilon: {eps}, Accuracy: {acc:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output", "mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "opacus_small_batch.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_dir}/opacus_small_batch.json")


if __name__ == "__main__":
    main()
