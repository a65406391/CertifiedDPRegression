"""Baselines for the blobs ablation. Hinge loss throughout, pure (eps,0)-DP only.

Approximate DP is deliberately excluded: AGS uses a different mechanism there
(Def. 8 rather than Def. 7), so the arms would not be comparable.

  non_private   the ceiling
  dpsgd_pure    Laplace noise every step, basic composition at eps/T
  pate          disjoint teachers label a public pool by noisy argmax; a student is
                trained on those labels and released
  objpert       Chaudhuri objective perturbation (requires lambda > 0)
"""

import numpy as np


def hinge_grads(theta, P, ys, margin=1.0):
    """Per-sample dL/dtheta for hinge, shape (n, d+1)."""
    z = P @ theta
    active = (margin - ys * z) > 0
    return np.where(active, -ys, 0.0)[:, None] * P


def train_hinge(
    P, ys, *, n_s, lr, gamma, lam, b=None, noise_scale=0.0, rng=None, theta0=None, noise_kind="laplace"
):
    """Element-wise clipped hinge SGD, matching AGT's clip_method='clamp'."""
    n, dd = P.shape
    b = b or n
    theta = np.zeros(dd) if theta0 is None else theta0.copy()
    spe = max(1, n // b)
    for t in range(n_s):
        sl = slice((t % spe) * b, ((t % spe) + 1) * b)
        g = np.clip(hinge_grads(theta, P[sl], ys[sl]), -gamma, gamma)
        upd = g.sum(0) / b
        if noise_scale > 0:
            upd = upd + (
                rng.normal(0.0, noise_scale, size=dd)
                if noise_kind == "gauss"
                else rng.laplace(0.0, noise_scale, size=dd)
            )
        theta = (1.0 - lr * lam) * theta - lr * upd
    return theta


def acc(theta, Pte, yts):
    return float((np.sign(Pte @ theta) == np.sign(yts)).mean())


def non_private(P, ys, *, n_s, lr, gamma, lam):
    return train_hinge(P, ys, n_s=n_s, lr=lr, gamma=gamma, lam=lam)


def dpsgd_pure(P, ys, eps, *, n_s, lr, gamma, lam, seed=0, b=None):
    """Pure eps-DP. Element-wise clip => l1 sensitivity of the mean grad is 2.d.gamma/b.

    Basic composition only: pure DP admits neither advanced composition nor
    subsampling amplification, so each step gets eps/n_s.
    """
    n, dd = P.shape
    b = b or n
    delta1 = 2.0 * dd * gamma / b
    scale = delta1 * n_s / eps
    return train_hinge(
        P, ys, n_s=n_s, lr=lr, gamma=gamma, lam=lam, b=b, noise_scale=scale, rng=np.random.RandomState(seed)
    ), scale


def dpsgd_gaussian(P, ys, eps, delta, *, n_s, lr, gamma, lam, seed=0, b=None):
    """(eps, delta)-DP. Port of experiments/common/ags.py:train_gaussian_dpsgd (L757-770).

    Element-wise clipping to gamma gives per-coordinate sensitivity 2*gamma/b, so the
    Gaussian mechanism sees l2 sensitivity sqrt(d) * 2*gamma/b. Basic composition over
    n_s steps, exactly as the reference does: eps_per_step = eps / n_s.
    """
    n, dd = P.shape
    b = b or n
    eps_step = eps / n_s
    l2 = np.sqrt(dd) * 2.0 * gamma / b
    scale = (l2 / eps_step) * np.sqrt(2.0 * np.log(1.25 / delta))
    return train_hinge(
        P,
        ys,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        noise_kind="gauss",
    ), scale


def pate_gaussian(P, ys, Ppub, eps, delta, *, n_teachers, n_queries, n_s, lr, gamma, lam, seed=0):
    """(eps, delta)-DP PATE. One record moves one vote between two counts, so the count
    vector has l2 sensitivity sqrt(2). Gaussian per query, basic composition over q."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(P))
    shards = np.array_split(idx, n_teachers)
    teachers = [train_hinge(P[s], ys[s], n_s=n_s, lr=lr, gamma=gamma, lam=lam) for s in shards]
    q = min(n_queries, len(Ppub))
    Xq = Ppub[:q]
    votes = np.stack([np.sign(Xq @ t) for t in teachers])
    counts = np.stack([(votes == -1).sum(0), (votes == 1).sum(0)]).astype(float)
    eps_q, delta_q = eps / q, delta / q
    sig = (np.sqrt(2.0) / eps_q) * np.sqrt(2.0 * np.log(1.25 / delta_q))
    counts = counts + rng.normal(0.0, sig, size=counts.shape)
    labels = np.where(counts[1] >= counts[0], 1.0, -1.0)
    return train_hinge(Xq, labels, n_s=n_s, lr=lr, gamma=gamma, lam=lam), eps_q


def pate(P, ys, Ppub, eps, *, n_teachers, n_queries, n_s, lr, gamma, lam, seed=0):
    """Teachers vote on a public pool with Laplace noise; a student is trained on the
    resulting labels and released.

    One record sits in one teacher, so it moves at most one vote: the count vector has
    l1 sensitivity 2. Each query is therefore eps_q-DP with Laplace(2/eps_q), and basic
    composition over n_queries gives eps = n_queries * eps_q.
    """
    rng = np.random.RandomState(seed)
    n = len(P)
    idx = rng.permutation(n)
    shards = np.array_split(idx, n_teachers)
    teachers = [train_hinge(P[s], ys[s], n_s=n_s, lr=lr, gamma=gamma, lam=lam) for s in shards]
    q = min(n_queries, len(Ppub))
    Xq = Ppub[:q]
    votes = np.stack([np.sign(Xq @ t) for t in teachers])  # (m, q) in {-1,+1}
    counts = np.stack([(votes == -1).sum(0), (votes == 1).sum(0)]).astype(float)  # (2,q)
    eps_q = eps / q
    counts = counts + rng.laplace(0.0, 2.0 / eps_q, size=counts.shape)
    labels = np.where(counts[1] >= counts[0], 1.0, -1.0)
    student = train_hinge(Xq, labels, n_s=n_s, lr=lr, gamma=gamma, lam=lam)
    return student, eps_q


def objective_perturbation(P, ys, eps, lam, seed=0, c=0.25):
    """Chaudhuri Alg. 2 for a convex loss with l2 regulariser; needs lambda > 0."""
    from scipy.optimize import minimize

    if lam <= 0:
        raise ValueError("objective perturbation requires lambda > 0")
    rng = np.random.RandomState(seed)
    n, dd = P.shape
    eps_p = eps - 2.0 * np.log(1.0 + c / (n * lam))
    delta = 0.0
    if eps_p <= 0:
        delta = max(c / (n * (np.exp(eps / 4.0) - 1.0)) - lam, 0.0)
        eps_p = eps / 2.0
    nrm = rng.gamma(dd, 2.0 / eps_p)
    bvec = rng.normal(size=dd)
    bvec = nrm * bvec / np.linalg.norm(bvec)

    def obj(th):
        m = np.maximum(1.0 - ys * (P @ th), 0.0).mean()
        return m + 0.5 * (lam + delta) * th @ th + (bvec @ th) / n

    return minimize(obj, np.zeros(dd), method="L-BFGS-B").x


def dpsgd_opacus(
    P, ys, eps, delta, *, n_s, lr, gamma, lam, b, seed=0, adjacency="substitution", accountant="rdp"
):
    """DP-SGD with subsampled-Gaussian accounting via Opacus (RDP or PRV).

    Two things this must get right to be sound, both of which differ from
    `dpsgd_gaussian`:

    Poisson sampling. Privacy amplification by subsampling is only valid if each
    record is included independently with probability q. Our other runs use fixed
    cyclic batches, for which amplification does NOT apply, so this function samples
    the batch Poisson-style at rate q = b/N and divides by the EXPECTED batch size.

    Adjacency. Opacus accounts add/remove; AGS is substitution. A substitution is a
    removal followed by an addition, so a mechanism that is (eps/2, delta/2)-DP under
    add/remove is (eps, delta)-DP under substitution. `adjacency="substitution"`
    therefore requests half the budget from the accountant. `adjacency="add_remove"`
    reports the looser, more common convention.

    Clipping stays element-wise at gamma to match AGS, which bounds the per-sample
    l2 norm by gamma*sqrt(d) -- the same bound the reference's Gaussian baseline uses.
    """
    from opacus.accountants.utils import get_noise_multiplier

    n, dd = P.shape
    q = b / n
    tgt_e, tgt_d = (eps / 2.0, delta / 2.0) if adjacency == "substitution" else (eps, delta)
    sigma_mult = get_noise_multiplier(
        target_epsilon=tgt_e, target_delta=tgt_d, sample_rate=q, steps=n_s, accountant=accountant
    )
    C = gamma * np.sqrt(dd)  # l2 bound implied by element-wise clip
    noise_std = sigma_mult * C  # std on the SUM of clipped gradients
    rng = np.random.RandomState(seed)
    theta = np.zeros(dd)
    for _ in range(n_s):
        mask = rng.random_sample(n) < q  # Poisson sample
        idx = np.flatnonzero(mask)
        g = np.clip(hinge_grads(theta, P[idx], ys[idx]), -gamma, gamma) if idx.size else np.zeros((0, dd))
        summed = g.sum(0) if idx.size else np.zeros(dd)
        upd = (summed + rng.normal(0.0, noise_std, size=dd)) / (q * n)
        theta = (1.0 - lr * lam) * theta - lr * upd
    return theta, sigma_mult


def dpsgd_rdp_fixed(
    P, ys, eps, delta, *, n_s, lr, gamma, lam, b, seed=0, adjacency="substitution", accountant="rdp"
):
    """DP-SGD under RDP composition with NO subsampling, on identical fixed batching.

    This is the honest "matched" baseline under approximate DP. It keeps the exact
    algorithm AGS certifies -- same fixed cyclic partition, same element-wise clip --
    and changes only the accountant. `sample_rate=1.0` makes no amplification claim,
    which would require Poisson sampling we do not do; the gain over basic composition
    comes purely from composing Renyi divergences (noise ~ sqrt(T)) rather than
    splitting epsilon linearly (noise ~ T).

    Adjacency: Opacus accounts add/remove; AGS is substitution, so we request half the
    budget (a substitution is a removal plus an addition).
    """
    from opacus.accountants.utils import get_noise_multiplier

    n, dd = P.shape
    tgt_e, tgt_d = (eps / 2.0, delta / 2.0) if adjacency == "substitution" else (eps, delta)
    sigma_mult = get_noise_multiplier(
        target_epsilon=tgt_e, target_delta=tgt_d, sample_rate=1.0, steps=n_s, accountant=accountant
    )
    C = gamma * np.sqrt(dd)  # l2 bound implied by element-wise clip
    scale = sigma_mult * C / b  # std on the MEAN gradient
    return train_hinge(
        P,
        ys,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        noise_kind="gauss",
    ), sigma_mult
