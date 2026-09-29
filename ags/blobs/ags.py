"""AGS: valid parameter-space bound via AGT.

SUPERSEDED INFERENCE. The release path in this module (smooth_sensitivity, release,
beta_for, global_sensitivity) was a re-derivation of experiments/common/ags.py and
deviated from it in six ways -- wrong quantity (l1 radius rather than per-coordinate
width), wrong cap, an undivided beta, a different allocation, a different constant, and
a missing k=0 check-point. Those functions now RAISE. Use ss_reference, which is a
line-by-line port. Only agt_bound() below is still current.

A^k is read off the bound exactly as the paper does (§Valid Parameter Space Bounds):
    A^k <= max_{theta' in T_k} ||theta_nominal - theta'||_1
i.e. the l1 radius of the envelope about the nominal.

The release follows Def. 7: eta ~ Cauchy(6 SS/eps) with beta <= eps/6, which is the
gamma_c = 2 member of the generalised family (scale 2(gamma_c+1)/eps,
beta = eps/(2(gamma_c+1))). gamma_c is exposed so knob 3 can sweep it.
"""

import warnings

import numpy as np
import torch

from abstract_gradient_training import AGTConfig, privacy_certified_training
from abstract_gradient_training.bounded_models import IntervalBoundedModel


def agt_bound(model, X, ys, *, k, b, n_epochs, lr, gamma, l2_reg, clip_method="clamp"):
    """Valid parameter-space bound T_k = [theta_L, theta_n, theta_U] from AGT."""
    ds = torch.utils.data.TensorDataset(torch.tensor(X).double(), torch.tensor(ys[:, None]).double())
    loader = torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False)
    cfg = AGTConfig(
        loss="hinge",
        n_epochs=n_epochs,
        learning_rate=lr,
        l2_reg=l2_reg,
        device="cpu",
        k_private=k,
        clip_gamma=gamma,
        clip_method=clip_method,
        log_level="ERROR",
        metadata={"seed": 0, "strict_mode": True},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = privacy_certified_training(IntervalBoundedModel(model), cfg, loader)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(bm.param_l), f(bm.param_n), f(bm.param_u)


def A_of_k(pl, pn, pu):
    """l1 radius of the envelope about the nominal -- the paper's A^k."""
    return float(np.maximum(pu - pn, pn - pl).sum())


_SUPERSEDED = (
    "{name}() is superseded and no longer computes the AGS release.\n"
    "  It deviated from experiments/common/ags.py in six ways; see the module\n"
    "  docstring and AGS_Ablation/blobs/ss_reference.py, which ports the reference\n"
    "  line by line. Migrate:\n"
    "    gs   = ss_reference.gs_window(lr*n_s, gamma, b, n_coords)\n"
    "    ls   = {{k: ss_reference.local_sens(param_l, param_u, gs, k) for k in ks}}\n"
    "    ss   = ss_reference.smooth_sensitivity(ls, ks, gs, ss_reference.beta_for(eps), P)\n"
    "    sig  = ss_reference.sigma(ss, eps, P)\n"
    "    out  = ss_reference.release(theta_n, sig, rng)"
)


def _superseded(name):
    def _raise(*_a, **_k):
        raise RuntimeError(_SUPERSEDED.format(name=name))

    _raise.__name__ = name
    return _raise


# Deliberately raise rather than delegate: the signatures take different QUANTITIES
# (l1 radius vs per-coordinate width), so a silent shim would keep producing wrong
# numbers. Failing loudly is the point.
global_sensitivity = _superseded("global_sensitivity")
smooth_sensitivity = _superseded("smooth_sensitivity")
release = _superseded("release")
beta_for = _superseded("beta_for")
