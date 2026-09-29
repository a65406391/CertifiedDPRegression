"""AGS: valid parameter-space bound via AGT, for the MNIST PCA-64 10-class head.

Port of AGS_Ablation/blobs/ags.py. `agt_bound` takes a `loss` argument rather than
hardcoding hinge, and maps labels into the form each loss requires.

--------------------------------------------------------------------------------
GLOBAL SENSITIVITY IS A COMPUTED QUANTITY, NOT A FORMULA
--------------------------------------------------------------------------------
GS is the envelope AGT returns when every record the bound ranges over is
substitutable: k = n for the full-dataset bound, or k = b for the per-batch IBP
recursion. `global_sensitivity()` below computes it that way and nothing in this
directory is permitted to call anything else "GS".

In particular `ss_reference.gs_window()` computes

    sum_lr * 2 * clip_gamma / batchsize

which is the reference implementation's `true_gs_window` and is reproduced there
verbatim because ss_reference.py is a line-by-line port under a bit-equality test.
That expression is a CLOSED-FORM SUBSTITUTION quantity that assumes the two
trajectories' non-substituted gradients stay identical -- it is not global
sensitivity, it is not what this function returns, and a certified width exceeding
it says nothing about whether the certificate is doing work. Runs in this directory
record both, under names that distinguish them.

Measured once on the binary pilot of this port (N=200, d=64, b=20, n_s=200, lr=0.5,
gamma=0.2), which is the check every future setting should repeat:

    k          1       2       3       4       8      16      20 (= b)
    width  8.516  10.792  12.912  14.979  23.051  36.651  40.000
                                                            ^ GS

    GS = 40.000 = 2*n_s*lr*gamma exactly, i.e. GS/ceiling = 1.0000

So GS lands on the clipping ceiling, and `clipping_ceiling()` is the cheap
consistency check on any computed GS: they must agree.

--------------------------------------------------------------------------------
Which losses AGT can bound, verified by running them
--------------------------------------------------------------------------------
    cross_entropy   10-class softmax, integer targets                  OK
    hinge           one-vs-all, (n, 10) target matrix of +-1           OK
    hinge (WW)      Weston-Watkins multiclass                          NOT AVAILABLE

BoundedHingeLoss accepts a (n, C) target matrix because it only checks that every
entry is +-1 and reshapes inputs to match -- so the one-vs-all arm certifies as a
stack of C independent binary hinges. There is no coupled-logit multiclass hinge in
abstract_gradient_training, nor in abstract_gradient_training_opt; losses.hinge_ww
is therefore measurable but not certifiable, and select_loss() below refuses it.

A^k is read off the bound exactly as the paper does (Valid Parameter Space Bounds):
    A^k <= max_{theta' in T_k} ||theta_nominal - theta'||_1
i.e. the l1 radius of the envelope about the nominal.
"""

import warnings

import numpy as np
import torch

from abstract_gradient_training import AGTConfig, privacy_certified_training
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training.bounded_models.crown_bounded_model import CROWNBoundedModel

BOUNDED_MODELS = {"ibp": IntervalBoundedModel, "crown": CROWNBoundedModel}

from ags.mnist.shared import unpack, logistic_model, N_CLASSES
from ags.mnist.losses import CERTIFIABLE


def select_loss(loss):
    """Map a losses.GRADS name to the AGTConfig loss that certifies it."""
    if loss not in CERTIFIABLE:
        raise NotImplementedError(
            f"'{loss}' has no bounded-loss implementation in abstract_gradient_training. "
            f"Certifiable arms: {sorted(CERTIFIABLE)}. See this module's docstring."
        )
    return CERTIFIABLE[loss]


def _targets(y, loss, n_classes=N_CLASSES):
    """Labels in the form the AGT loss expects: integers for CE, +-1 matrix for OvA."""
    y = np.asarray(y).astype(np.int64)
    if loss == "hinge_ova":
        return torch.tensor(np.where(np.eye(n_classes)[y] > 0, 1.0, -1.0)).double()
    return torch.tensor(y).long()


def agt_bound(
    model,
    X,
    y,
    *,
    k,
    b,
    n_epochs,
    lr,
    gamma,
    l2_reg,
    clip_method="clamp",
    loss="cross_entropy",
    n_classes=N_CLASSES,
    lr_decay=0.0,
    lr_min=0.0,
    device="cpu",
    bound_model="ibp",
    theta0=None,
):
    """Valid parameter-space bound T_k = [theta_L, theta_n, theta_U] from AGT.

    bound_model selects the package's bounded model: "ibp" (interval) or "crown". On a single
    linear layer the two are identical (measured: zero difference), since CROWN's relaxations act on
    activations; the choice starts to matter with a hidden layer.

    lr_decay is the reference's `eta`: bounded_sgd applies lr / (1 + lr_decay * batch)
    per BATCH, which is exactly experiments/common/ags.py's lr_n = lr / (1 + eta n).
    It is the knob that makes a long trajectory affordable -- the envelope grows by
    lr_n * 2 * gamma per step, so a decaying lr makes the total grow logarithmically
    in the step count rather than linearly.
    """
    if theta0 is not None:  # Algorithm 2: a window starts from the previously released parameters
        W0, b0 = unpack(theta0, next(model.parameters()).shape[1], n_classes)
        with torch.no_grad():
            model[0].weight.copy_(torch.as_tensor(W0, dtype=model[0].weight.dtype))
            model[0].bias.copy_(torch.as_tensor(b0, dtype=model[0].bias.dtype))
    agt_loss = select_loss(loss)
    ds = torch.utils.data.TensorDataset(torch.tensor(X).double(), _targets(y, loss, n_classes))
    loader = torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False)
    cfg = AGTConfig(
        loss=agt_loss,
        n_epochs=n_epochs,
        learning_rate=lr,
        l2_reg=l2_reg,
        lr_decay=lr_decay,
        lr_decay_type="batch",
        lr_min=lr_min,
        device=device,
        k_private=k,
        clip_gamma=gamma,
        clip_method=clip_method,
        log_level="ERROR",
        metadata={"seed": 0, "strict_mode": True},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = privacy_certified_training(BOUNDED_MODELS[bound_model](model), cfg, loader)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(bm.param_l), f(bm.param_n), f(bm.param_u)


def sum_lr(lr, n_steps, lr_decay=0.0, lr_min=0.0):
    """Sum of the per-step learning rates -- the reference's sum_lr_window.

    With lr_decay = 0 this is just lr * n_steps. With decay it is
    sum_n max(lr/(1+lr_decay n), lr_min), which grows like (lr/eta) ln(1+eta T).
    """
    return float(sum(max(lr / (1.0 + lr_decay * n), lr_min) for n in range(int(n_steps))))


def global_sensitivity(
    X,
    y,
    *,
    b,
    n_epochs,
    lr,
    gamma,
    l2_reg=0.0,
    loss="cross_entropy",
    d=None,
    n_classes=N_CLASSES,
    model_fn=None,
    lr_decay=0.0,
    lr_min=0.0,
):
    """GS: the AGT envelope with every record in the bound's scope substitutable.

    For the IBP recursion the bound is formed per batch, so the maximal setting is
    k = b; the returned width is then a per-coordinate global sensitivity, valid at
    every k because no dataset admits a larger local sensitivity than the one where
    the whole batch is free.

    Returns (gs_width, gs_radius, param_l, param_n, param_u) with gs_width the
    per-coordinate |theta_U - theta_L| -- the convention ss_reference.local_sens
    compares against -- and gs_radius the one-sided max(u - n, n - l).
    """
    d = d if d is not None else X.shape[1]
    mk = model_fn if model_fn is not None else (lambda dd: logistic_model(dd, n_classes))
    pl, pn, pu = agt_bound(
        mk(d),
        X,
        y,
        k=b,
        b=b,
        n_epochs=n_epochs,
        lr=lr,
        gamma=gamma,
        l2_reg=l2_reg,
        loss=loss,
        n_classes=n_classes,
        lr_decay=lr_decay,
        lr_min=lr_min,
    )
    return np.abs(pu - pl), np.maximum(pu - pn, pn - pl), pl, pn, pu


def A_of_k(pl, pn, pu):
    """l1 radius of the envelope about the nominal -- the paper's A^k."""
    return float(np.maximum(pu - pn, pn - pl).sum())


def clipping_ceiling(n_s, lr, gamma, lr_decay=0.0, lr_min=0.0):
    """Unconditional upper bound on the envelope WIDTH after n_s steps.

    Follows from clipping alone (explanation.md 4.1): every delta lies in
    [-gamma, gamma], so the interval widens by at most lr*2*gamma per step. No sound
    bound can exceed this, and GS computed at k = b must land at or below it.
    """
    return 2.0 * sum_lr(lr, n_s, lr_decay, lr_min) * float(gamma)
