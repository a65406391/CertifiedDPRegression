"""Flat-vector heads on frozen features, and the one trainer every arm runs through.

Two heads: the logistic head Linear(d, C), P = C (d + 1), and the hidden-layer head
Linear(d, h) -> ReLU -> Linear(h, C), P = h (d + 1) + C (h + 1), the certifiable form of "more than
the last layer". A head knows its torch.nn.Sequential (what the AGT bounded models certify), the
flat theta <-> parameter mapping in the package's order (each module's weight, then bias), logits
and accuracy from a flat theta, and the per-sample parameter gradients of the softmax cross-entropy.

`train` is the element-wise clipped SGD the certificate describes, AGT's clip_method="clamp": every
coordinate of every per-sample gradient clamped to [-gamma, gamma], the mean taken, L2 toward a
center added, one step at lr / (1 + decay t). Additive noise on the mean gradient makes it the
matched DP-SGD arm; a center makes it the warm-started arm (ags/agt_ext/centered_l2.py is the same
change on the certified side). On the logistic head it agrees with ags.mnist.baselines.train to
round-off (test_heads.py); the point of the module is that the same code runs the hidden-layer head,
for which no numpy trainer exists. `poisson_steps` is the optimized DP-SGD loop (Poisson sample,
per-sample clip, noise on the sum) and `teachers_and_votes` the PATE teachers.

Torch, float64, on whichever device is passed; DP noise is drawn from a numpy RandomState so a seed
means the same draw on CPU and GPU.
"""

from __future__ import annotations

import numpy as np
import torch


def _t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)


class Head:
    """Base: shapes in torch parameter order; subclasses give the model and the gradients."""

    name = ""

    def __init__(self, d, n_classes, shapes):
        self.d, self.C, self.shapes = int(d), int(n_classes), [tuple(s) for s in shapes]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.P = int(sum(self.sizes))

    # ---- flat theta <-> parameters
    def split(self, theta):
        """Flat theta -> list of arrays with the parameter shapes (torch order)."""
        theta = np.asarray(theta, float).ravel()
        if theta.size != self.P:
            raise ValueError(f"theta has {theta.size} entries, the head {self.P}")
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[i : i + n].reshape(s))
            i += n
        return out

    def split_torch(self, theta, device="cpu"):
        return [_t(p, device) for p in self.split(theta)]

    def _split_t(self, theta_t):
        return [p.reshape(s) for p, s in zip(torch.split(theta_t, self.sizes), self.shapes)]

    def torch_model(self, theta=None):
        m = self._sequential().double()
        if theta is not None:
            with torch.no_grad():
                for p, v in zip(m.parameters(), self.split(theta)):
                    p.copy_(torch.as_tensor(v, dtype=p.dtype))
        return m

    def init(self, seed=0):
        """The flat initial parameters: torch's default init under `seed` (zero for the logistic head)."""
        torch.manual_seed(int(seed))
        m = self._sequential().double()
        return np.concatenate([p.detach().numpy().ravel() for p in m.parameters()])

    # ---- evaluation
    def logits(self, theta, X, device="cpu"):
        with torch.no_grad():
            return self._logits_t(_t(theta, device), _t(X, device)).cpu().numpy()

    def accuracy(self, theta, X, y, device="cpu"):
        return float((self.logits(theta, X, device).argmax(1) == np.asarray(y)).mean())

    # ---- subclass interface
    def _sequential(self):
        raise NotImplementedError

    def _logits_t(self, theta_t, X_t):
        raise NotImplementedError

    def grads(self, theta_t, X_t, y_t):
        """Per-sample dL/dtheta of the softmax cross-entropy: (n, P) in flat order."""
        raise NotImplementedError


def _dz(z, y):
    """softmax(z) - onehot(y): the logit gradient of the cross-entropy."""
    p = torch.softmax(z, dim=1)
    p[torch.arange(len(y), device=z.device), y] -= 1.0
    return p


class Logistic(Head):
    name = "logistic"

    def __init__(self, d, n_classes):
        super().__init__(d, n_classes, [(n_classes, d), (n_classes,)])

    def _sequential(self):
        m = torch.nn.Sequential(torch.nn.Linear(self.d, self.C))
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        return m

    def _logits_t(self, theta_t, X_t):
        W, b = self._split_t(theta_t)
        return X_t @ W.T + b

    def grads(self, theta_t, X_t, y_t):
        dz = _dz(self._logits_t(theta_t, X_t), y_t)
        dW = dz[:, :, None] * X_t[:, None, :]
        return torch.cat([dW.reshape(len(X_t), -1), dz], dim=1)


class MLP(Head):
    name = "mlp"

    def __init__(self, d, h, n_classes):
        self.h = int(h)
        super().__init__(d, n_classes, [(h, d), (h,), (n_classes, h), (n_classes,)])

    def _sequential(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.d, self.h), torch.nn.ReLU(), torch.nn.Linear(self.h, self.C)
        )

    def _forward(self, theta_t, X_t):
        W1, b1, W2, b2 = self._split_t(theta_t)
        z1 = X_t @ W1.T + b1
        a = torch.relu(z1)
        return z1, a, a @ W2.T + b2, W2

    def _logits_t(self, theta_t, X_t):
        return self._forward(theta_t, X_t)[2]

    def grads(self, theta_t, X_t, y_t):
        z1, a, z2, W2 = self._forward(theta_t, X_t)
        n = len(X_t)
        dz2 = _dz(z2, y_t)  # (n, C)
        dW2 = dz2[:, :, None] * a[:, None, :]  # (n, C, h)
        dz1 = (dz2 @ W2) * (z1 > 0).to(z1.dtype)  # (n, h); ReLU' = 1[z > 0], as the bounded model
        dW1 = dz1[:, :, None] * X_t[:, None, :]  # (n, h, d)
        return torch.cat([dW1.reshape(n, -1), dz1, dW2.reshape(n, -1), dz2], dim=1)


def make_head(kind, d, n_classes, h=None):
    if kind == "logistic":
        return Logistic(d, n_classes)
    if kind == "mlp":
        return MLP(d, h, n_classes)
    raise ValueError(f"unknown head {kind!r}")


# ------------------------------------------------------------------------------------------------
# the trainer: clipped SGD, L2 toward a center, optional noise
# ------------------------------------------------------------------------------------------------


def train(
    head,
    X,
    y,
    *,
    theta0,
    n_s,
    lr,
    gamma,
    lam,
    center=None,
    b=None,
    noise_scale=0.0,
    noise_kind="laplace",
    rng=None,
    lr_decay=0.0,
    lr_min=0.0,
    device="cpu",
):
    """Element-wise clipped SGD from theta0; gamma=None trains unclipped. Fixed cyclic batches.

    Each step: theta <- theta - lr_t (mean_i clamp(g_i, -gamma, gamma) + noise + lam (theta - center)),
    lr_t = max(lr / (1 + lr_decay t), lr_min). This is bounded_sgd with clip_method="clamp" and
    ags.mnist.baselines.train with a center, so the certified, matched and warm arms run one algorithm.
    """
    Xt, yt = _t(X, device), torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    theta = _t(theta0, device).clone()
    c = torch.zeros_like(theta) if center is None else _t(center, device)
    n = len(yt)
    b = b or n
    spe = max(1, n // b)
    for t in range(int(n_s)):
        sl = slice((t % spe) * b, ((t % spe) + 1) * b)
        G = head.grads(theta, Xt[sl], yt[sl])
        if gamma is not None:
            G = G.clamp(-gamma, gamma)
        upd = G.mean(0)
        if noise_scale > 0:
            z = (
                rng.normal(0.0, noise_scale, size=head.P)
                if noise_kind == "gauss"
                else rng.laplace(0.0, noise_scale, size=head.P)
            )
            upd = upd + _t(z, device)
        lr_t = max(lr / (1.0 + lr_decay * t), lr_min) if lr_decay else lr
        theta = theta - lr_t * (upd + lam * (theta - c))
    return theta.cpu().numpy()


def poisson_steps(
    head, X, y, *, theta0, steps, lr, q, rng, per_step_noise, clip_rows, lr_decay=0.0, device="cpu"
):
    """Poisson-sampled SGD: per-sample gradients on the sample, a clip, noise on the sum, divided by the
    expected sample size. The optimized DP-SGD loop of ags.mnist.baselines, for any head."""
    Xt, yt = _t(X, device), torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    theta = _t(theta0, device).clone()
    n = len(yt)
    expected = q * n
    for t in range(int(steps)):
        mask = torch.as_tensor(rng.random_sample(n) < q, device=device)
        if bool(mask.any()):
            total = clip_rows(head.grads(theta, Xt[mask], yt[mask])).sum(0)
        else:
            total = torch.zeros(head.P, dtype=torch.float64, device=device)
        total = total + _t(per_step_noise(rng, head.P), device)
        lr_t = lr / (1.0 + lr_decay * t) if lr_decay else lr
        theta = theta - lr_t * total / expected
    return theta.cpu().numpy()


def clip_l2(clip):
    def f(G):
        norms = G.norm(dim=1, keepdim=True)
        return G * torch.clamp(clip / torch.clamp(norms, min=1e-12), max=1.0)

    return f


def clip_elementwise(gamma):
    return lambda G: G.clamp(-gamma, gamma)


def teachers_and_votes(head, X, y, Xpool, *, n_teachers, n_s, lr, lam, rng, device="cpu", seed=0):
    """Disjoint shards -> unclipped teachers (each from head.init(seed + i)) -> per-class vote counts
    (C, len(Xpool)) on the public pool. Only the votes are privatised, so the teachers are not clipped."""
    n = len(y)
    perm = rng.permutation(n)
    counts = np.zeros((head.C, len(Xpool)), dtype=np.int64)
    for i, idx in enumerate(np.array_split(perm, n_teachers)):
        th = train(
            head,
            X[idx],
            y[idx],
            theta0=head.init(seed + i),
            n_s=n_s,
            lr=lr,
            gamma=None,
            lam=lam,
            device=device,
        )
        pred = head.logits(th, Xpool, device).argmax(1)
        counts[pred, np.arange(len(Xpool))] += 1
    return counts
