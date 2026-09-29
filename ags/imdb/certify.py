"""One certified cell for a head on frozen features: AGT's parameter-space bound [theta_L, theta_n, theta_U].

The IMDB counterpart of ags.mnist.ags.agt_bound with the model generalised to any `heads.Head`
(logistic or hidden-layer), a start `theta0` in flat form, and an optional L2 `center` (the public
warm start) applied through ags.agt_ext.centered_l2. Softmax cross-entropy, element-wise clipping,
full batch, lr / (1 + decay t) per step; "ibp" or "crown" bound propagation.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from abstract_gradient_training import AGTConfig, privacy_certified_training

from ags.agt_ext.centered_l2 import centered
from ags.mnist.ags import BOUNDED_MODELS, select_loss


def agt_bound(
    head,
    X,
    y,
    *,
    k,
    b,
    T,
    lr,
    gamma,
    lam,
    decay=0.0,
    theta0=None,
    center=None,
    device="cpu",
    bound_model="crown",
):
    """(param_l, param_n, param_u) flat, after T full-batch steps under k substitutions."""
    model = head.torch_model(theta0 if theta0 is not None else head.init(0))
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(np.asarray(X), dtype=torch.float64), torch.as_tensor(np.asarray(y)).long()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False)
    kw = {} if center is None else {"l2_center": head.split_torch(center, device)}
    cfg = AGTConfig(
        loss=select_loss("cross_entropy"),
        n_epochs=int(T),
        learning_rate=lr,
        l2_reg=lam,
        lr_decay=decay,
        lr_decay_type="batch",
        device=device,
        k_private=int(k),
        clip_gamma=gamma,
        clip_method="clamp",
        log_level="ERROR",
        metadata={"seed": 0, "strict_mode": True},
        optimizer_kwargs=kw,
    )
    with warnings.catch_warnings(), centered():
        warnings.simplefilter("ignore")
        bm = privacy_certified_training(BOUNDED_MODELS[bound_model](model), cfg, loader)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(bm.param_l), f(bm.param_n), f(bm.param_u)
