"""Certified training for the California Housing experiments: the shared machinery.

Everything here was inside `train.py`, which three other modules imported it from. A
driver that other drivers import is a driver you have to read three times, so the machinery lives
here and the drivers below it only parse arguments and loop:

    train.py                 the batch-size ablation
    pate/train.py             the PATE shard teachers
    probes/probe_tighten.py             the steps-by-learning-rate probe

`train_cell` is one certified cell: a model trained from a fixed seed with genuine AGT, returning
the bounded model whose `param_l`, `param_n`, `param_u` are the certified envelope. The l2 clipping
branch of the package is broken for models with biases, so callers install `common.agt_patches`
first; `run_selftest` is the guard they run before training.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from agtr.california.config import ABLATION as CFG
from common import agt_patches

OUT_ROOT = "agtr/california/values/agt_2026-09-14"
STORED = "agtr/california/values/ablation"
Y_MIN, Y_MAX = CFG["Y_MIN"], CFG["Y_MAX"]


def make_loader(train_dataset, bs, seed=42):
    X = train_dataset.tensors[0].double()
    y = train_dataset.tensors[1].double().unsqueeze(1)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True, generator=g, drop_last=True)


def build_model(hidden, activation):
    """8 -> hidden[0] -> ... -> 1 with the given activation; (64, 64, relu) is the paper's model."""
    act = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[activation]
    layers, d = [], 8
    for h in hidden:
        layers += [torch.nn.Linear(d, h), act()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


def train_cell(
    train_dataset, bs, k, epochs, gamma, clip_method, device, hidden=(64, 64), activation="relu", lr=None
):
    """One certified cell. `lr` defaults to the ablation config; pass it to override without
    mutating the shared CFG dict, which a probe used to do as a side effect."""
    torch.manual_seed(42)
    np.random.seed(42)
    dl = make_loader(train_dataset, bs)
    model = build_model(hidden, activation).double()
    bm = IntervalBoundedModel(model)
    cfg = agt.AGTConfig(
        loss="mse",
        n_epochs=epochs,
        learning_rate=CFG["LR"] if lr is None else lr,
        device=device,
        k_private=k,
        clip_gamma=gamma,
        clip_method=clip_method,
        log_level="ERROR",
    )
    t0 = time.time()
    bm = agt.privacy_certified_training(bm, cfg, dl)
    return bm, time.time() - t0


def run_selftest(gamma: float = 1.0) -> bool:
    """Guard the patched l2 clipping before any certified run. See common.agt_patches."""
    return agt_patches.selftest(gamma)
