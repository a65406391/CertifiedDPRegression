"""Certified training for the linear AGT-R experiments: the shared machinery.

`train_certified_model` trains one cell and returns its parameter envelope, by interval propagation
for the `ibp` method or by the unrolled Gurobi problem at concretization frequency F for `f1`, `f2`,
`f4`, `f8`, `f16`. The nominal trajectory is plain clipped SGD from a fixed seed, so it is identical
across methods and every method certifies the same run.

It lived inside `run_ablation.py`, which `release.py` and the attack probe both imported it from.
The mirror of `agtr/california/certify.py`; the drivers beside it only parse arguments and loop.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import TensorDataset

from abstract_gradient_training_opt.certified_training import certified_training
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training import privacy_certified_training, AGTConfig

from agtr.linear.config import ABLATION as CFG
from common.models import create_linear_model, get_or_create_init_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_certified_model(X_train, y_train, epochs, lr, seed, method, k):
    """Train a certified model with privacy bounds and STRICT DETERMINISM."""
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
    perm = torch.randperm(len(X_train), generator=g)

    X_train_shuffled = torch.tensor(X_train[perm], dtype=torch.float64)
    y_train_shuffled = torch.tensor(y_train[perm], dtype=torch.float64)
    dataset = TensorDataset(X_train_shuffled, y_train_shuffled)

    # Case k=0: Standard Nominal Training
    if k == 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="mean")
        loader = torch.utils.data.DataLoader(dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)

        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param.grad.clamp_(-CFG["CLIP_GAMMA"], CFG["CLIP_GAMMA"])
                optimizer.step()
        return IntervalBoundedModel(model)

    # Case 'ibp': Pure IBP using AGTConfig
    if method == "ibp":
        agt_config = AGTConfig(
            loss="mse",
            n_epochs=epochs,
            learning_rate=lr,
            device="cpu",
            k_private=k,
            clip_gamma=CFG["CLIP_GAMMA"],
            clip_method="clamp",
            log_level="WARNING",
            metadata={"seed": seed, "strict_mode": True},
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)
        final_model = privacy_certified_training(IntervalBoundedModel(model), agt_config, loader)
        return final_model

    # Case 'fN': OPTConfig with MIP, freq=N
    freq = CFG["FREQ_MAP"].get(method, 1)

    effective_batch_size = freq * CFG["BATCH_SIZE"]
    if k > effective_batch_size:
        agt_config = AGTConfig(
            loss="mse",
            n_epochs=epochs,
            learning_rate=lr,
            device="cpu",
            k_private=k,
            clip_gamma=CFG["CLIP_GAMMA"],
            clip_method="clamp",
            log_level="WARNING",
            metadata={"seed": seed, "strict_mode": True},
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)
        final_model = privacy_certified_training(IntervalBoundedModel(model), agt_config, loader)
        return final_model

    config = OPTConfig(
        log_level="WARNING",
        n_epochs=epochs,
        batchsize=CFG["BATCH_SIZE"],
        learning_rate=lr,
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

    final_model, _ = certified_training(model, config, dataset, solve=True)
    return final_model
