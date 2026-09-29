"""CenteredBoundedSGD: without a center it is the package's optimizer; with one, the nominal
trajectory contracts toward the center and the envelope width is unchanged."""

import numpy as np
import torch

from abstract_gradient_training import AGTConfig, privacy_certified_training
from abstract_gradient_training.bounded_models import IntervalBoundedModel

from ags.agt_ext.centered_l2 import centered


def _run(center, seed=0, lam=0.5, T=20):
    torch.manual_seed(seed)
    X = torch.randn(64, 4, dtype=torch.float64)
    y = (X[:, 0] > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(4, 2)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=False)
    kw = {} if center is None else {"l2_center": [torch.full_like(p, center) for p in model.parameters()]}
    cfg = AGTConfig(
        loss="cross_entropy",
        n_epochs=T,
        learning_rate=0.5,
        l2_reg=lam,
        k_private=2,
        clip_gamma=0.1,
        clip_method="clamp",
        log_level="ERROR",
        optimizer_kwargs=kw,
    )
    with centered():
        bm = privacy_certified_training(IntervalBoundedModel(model), cfg, loader)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(bm.param_l), f(bm.param_n), f(bm.param_u)


def _run_package(seed=0, lam=0.5, T=20):
    torch.manual_seed(seed)
    X = torch.randn(64, 4, dtype=torch.float64)
    y = (X[:, 0] > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(4, 2)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=False)
    cfg = AGTConfig(
        loss="cross_entropy",
        n_epochs=T,
        learning_rate=0.5,
        l2_reg=lam,
        k_private=2,
        clip_gamma=0.1,
        clip_method="clamp",
        log_level="ERROR",
    )
    bm = privacy_certified_training(IntervalBoundedModel(model), cfg, loader)
    f = lambda ps: np.concatenate([p.detach().cpu().numpy().ravel() for p in ps])
    return f(bm.param_l), f(bm.param_n), f(bm.param_u)


def test_no_center_is_the_package():
    a, b = _run(None), _run_package()
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_center_moves_the_fixed_point_not_the_width():
    l0, n0, u0 = _run(None)
    l1, n1, u1 = _run(3.0)
    assert np.all(n1 > n0)  # pulled toward +3 instead of 0
    np.testing.assert_allclose(u1 - l1, u0 - l0, rtol=1e-9, atol=1e-12)
    assert np.all(l1 <= n1 + 1e-12) and np.all(n1 <= u1 + 1e-12)


def test_center_is_restored():
    from abstract_gradient_training import bounded_optimizers
    from abstract_gradient_training.bounded_optimizers.bounded_sgd import BoundedSGD

    with centered():
        assert bounded_optimizers.BoundedSGD is not BoundedSGD
    assert bounded_optimizers.BoundedSGD is BoundedSGD
