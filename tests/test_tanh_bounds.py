"""Soundness of torch.nn.Tanh in IntervalBoundedModel (added 2026-09-15).

Randomised containment checks: for parameter boxes [param_l, param_u] around a nominal
8-16-1 tanh network, every concrete network with parameters sampled inside the box must
produce (i) outputs inside bound_forward's interval and (ii) per-sample parameter
gradients inside bound_backward's interval. Also checks the nominal tanh backward pass
against autograd.
"""
import torch
import pytest

from abstract_gradient_training.bounded_models import IntervalBoundedModel


def _net(seed, dtype=torch.float64):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(dtype)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("radius", [0.0, 0.05, 0.5, 3.0])
def test_tanh_interval_containment(seed, radius):
    g = torch.Generator().manual_seed(seed)
    net = _net(seed)
    bm = IntervalBoundedModel(net)
    for l, n, u in zip(bm.param_l, bm.param_n, bm.param_u):
        r = radius * torch.rand(n.shape, generator=g, dtype=n.dtype)
        l.copy_(n - r); u.copy_(n + r)
    x = torch.randn(64, 8, generator=g, dtype=torch.float64) * 2
    with torch.no_grad():
        y_l, y_u = bm.bound_forward(x, x, retain_intermediate=True)
        dl = torch.randn(64, 1, generator=g, dtype=torch.float64)
        g_l, g_u = bm.bound_backward(dl, dl)
    for _ in range(20):
        sampled = _net(seed)
        with torch.no_grad():
            for p, l, u in zip(sampled.parameters(), bm.param_l, bm.param_u):
                p.copy_(l + (u - l) * torch.rand(p.shape, generator=g, dtype=p.dtype))
        # concrete forward
        y = sampled(x)
        assert torch.all(y >= y_l - 1e-9) and torch.all(y <= y_u + 1e-9)
        # concrete per-sample parameter gradients of <dl, y>
        params = list(sampled.parameters())
        for i in range(x.shape[0]):
            grads = torch.autograd.grad((dl[i] * sampled(x[i:i + 1])).sum(), params)
            for gi, gl, gu in zip(grads, g_l, g_u):
                assert torch.all(gi >= gl[i] - 1e-9) and torch.all(gi <= gu[i] + 1e-9)


def test_tanh_nominal_backward_matches_autograd():
    net = _net(0)
    bm = IntervalBoundedModel(net)
    x = torch.randn(32, 8, dtype=torch.float64)
    dl = torch.randn(32, 1, dtype=torch.float64)
    with torch.no_grad():
        bm.forward(x, retain_intermediate=True)
        grads = bm.backward(dl)
    params = list(net.parameters())
    for i in range(x.shape[0]):
        ref = torch.autograd.grad((dl[i] * net(x[i:i + 1])).sum(), params)
        for gi, ri in zip(grads, ref):
            assert torch.allclose(gi[i], ri, atol=1e-10)


def test_tanh_derivative_interval_is_tight_and_sound():
    # direct check of the derivative interval logic on intervals straddling / not straddling 0
    for lo, hi in [(-2.0, 1.0), (0.5, 2.0), (-3.0, -0.2), (-1e-6, 1e-6)]:
        xs = torch.linspace(lo, hi, 10001, dtype=torch.float64)
        d = 1 - torch.tanh(xs) ** 2
        d_lo, d_hi = 1 - torch.tanh(torch.tensor(lo, dtype=torch.float64)) ** 2, 1 - torch.tanh(torch.tensor(hi, dtype=torch.float64)) ** 2
        d_l = min(d_lo, d_hi); d_u = 1.0 if lo <= 0 <= hi else max(d_lo, d_hi)
        assert d.min() >= d_l - 1e-12 and d.max() <= d_u + 1e-12
        assert abs(d.min() - d_l) < 1e-6 and abs(d.max() - d_u) < 1e-6
