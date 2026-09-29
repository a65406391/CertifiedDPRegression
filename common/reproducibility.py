"""Seeding and device selection, so a run can be repeated exactly.

`set_reproducibility` fixes every generator the experiments draw from and turns off the cuDNN
autotuner, whose kernel choice varies with load and changes results in the last digits. Call it
before building data or a model, not after.

This lived in `agtr/linear/misc_utils.py`, a grab-bag that `common/data.py` imported, so the
shared library depended on one experiment's leftovers. Nothing else used it.
"""

import numpy as np
import torch


def set_reproducibility(seed: int = 42, use_gpu: bool = True):
    """Set all the random seeds and determinism necessary for experiment reproducibility."""
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(use_gpu: bool = True, device: str = "cuda:0") -> torch.device:
    """Set the GPU for the experiment."""
    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(device)
        return torch.device(device)
    else:
        return torch.device("cpu")
