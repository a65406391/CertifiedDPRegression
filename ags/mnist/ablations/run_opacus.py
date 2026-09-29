"""Gaussian DP-SGD on PCA-MNIST, accounted by Opacus.

The baseline arm for the MNIST ablations: the same logistic head and schedule as
`run_approx_dp_comparison.py`, trained with Opacus's PrivacyEngine so DP-SGD gets its own
subsampled-Gaussian RDP accountant. It reuses that module's loader, so both arms see the same
PCA projection and split.

    PYTHONPATH=. python -m ags.mnist.ablations.run_opacus

Writes accuracy per epsilon to `ags/mnist/ablations/runs/opacus.json`.
"""

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

    # Match the hand DP-SGD step count exactly.
    # batch_size == N_TRAIN so batches_per_epoch == 1, hence epochs == TOTAL_STEPS.
    batches_per_epoch = math.ceil(CFG["N_TRAIN"] / CFG["BATCH_SIZE"])
    epochs = CFG["TOTAL_STEPS"] // batches_per_epoch
    actual_steps = epochs * batches_per_epoch
    logger.info(
        f"Training: {epochs} epochs x {batches_per_epoch} batches = {actual_steps} steps "
        f"(target: {CFG['TOTAL_STEPS']})"
    )

    # Compute max_grad_norm comparable to element-wise clipping.
    # The hand DP-SGD clips each gradient element to [-C, C] (L∞ clipping).
    # Opacus clips per-sample gradient L2-norm. For a d-dimensional gradient,
    # element-wise clipping to C gives max L2 norm = C * sqrt(d).
    # We set max_grad_norm = C * sqrt(d) so the effective per-param bound matches.
    dummy_model = nn.Linear(64, 10)
    n_params = sum(p.numel() for p in dummy_model.parameters())
    max_grad_norm = CFG["CLIP_GAMMA"] * math.sqrt(n_params)
    logger.info(
        f"max_grad_norm: {max_grad_norm:.4f} "
        f"(CLIP_GAMMA={CFG['CLIP_GAMMA']} * sqrt({n_params})={math.sqrt(n_params):.2f})"
    )

    results = {}
    pbar = tqdm(CFG["EPSILON_TOTALS"], desc="MNIST Opacus Sweep")
    for eps in pbar:
        set_seed(CFG["BASE_SEED"])
        model = nn.Linear(64, 10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG["LR_LAPLACE"])
        train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True)

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
    with open(os.path.join(out_dir, "opacus.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
