"""heads.py: the torch trainer is ags.mnist.baselines.train on the logistic head; the hidden-layer
gradients are autograd's; AGT's nominal trajectory (with and without a center) is the trainer's."""

import numpy as np
import torch

from ags.imdb import heads as H
from ags.imdb.certify import agt_bound
from ags.mnist import baselines as B


def _data(n=200, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True).max()
    y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(np.int64)
    return X, y


def test_logistic_matches_numpy_trainer():
    X, y = _data()
    head = H.Logistic(5, 2)
    for noise in (0.0, 0.05):
        kw = dict(n_s=30, lr=1.5, gamma=0.1, lam=0.4, lr_decay=0.01)
        a = B.train(X, y, n_classes=2, noise_scale=noise, rng=np.random.RandomState(3), **kw)
        b = H.train(
            head, X, y, theta0=np.zeros(head.P), noise_scale=noise, rng=np.random.RandomState(3), **kw
        )
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_mlp_grads_are_autograd():
    X, y = _data()
    head = H.MLP(5, 4, 2)
    theta = head.init(1)
    G = head.grads(torch.tensor(theta), torch.tensor(X), torch.tensor(y)).numpy()
    model = head.torch_model(theta)
    for i in (0, 7, 199):
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(torch.tensor(X[i : i + 1])), torch.tensor(y[i : i + 1])
        )
        loss.backward()
        g = np.concatenate([p.grad.numpy().ravel() for p in model.parameters()])
        np.testing.assert_allclose(G[i], g, rtol=1e-10, atol=1e-12)


def test_agt_nominal_is_the_trainer():
    X, y = _data()
    for head in (H.Logistic(5, 2), H.MLP(5, 3, 2)):
        theta0 = head.init(2)
        center = 0.5 * np.ones(head.P)
        kw = dict(T=15, lr=1.0, gamma=0.1, lam=0.3, decay=0.01)
        for c in (None, center):
            pl, pn, pu = agt_bound(
                head, X, y, k=1, b=len(y), theta0=theta0, center=c, bound_model="ibp", **kw
            )
            th = H.train(
                head,
                X,
                y,
                theta0=theta0,
                center=c,
                n_s=kw["T"],
                lr=kw["lr"],
                gamma=kw["gamma"],
                lam=kw["lam"],
                lr_decay=kw["decay"],
            )
            np.testing.assert_allclose(pn, th, rtol=0, atol=1e-10)
            assert np.all(pl <= pn + 1e-12) and np.all(pn <= pu + 1e-12)


def test_split_roundtrip_and_init():
    head = H.MLP(6, 4, 2)
    assert head.P == 4 * 7 + 2 * 5
    theta = np.arange(head.P, dtype=float)
    back = np.concatenate([p.ravel() for p in head.split(theta)])
    np.testing.assert_array_equal(theta, back)
    np.testing.assert_array_equal(head.init(5), head.init(5))
    assert np.all(H.Logistic(6, 2).init(0) == 0)
