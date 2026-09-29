"""Baselines for the MNIST PCA-64 ablation, 10 classes.

Port of AGS_Ablation/blobs/baselines.py. The mechanisms, accountants and composition
arguments are copied verbatim; three things follow from the task being 10-class
rather than binary, and nothing else moves:

  gradients   the per-sample dL/dtheta comes from losses.GRADS[loss] instead of
              blobs' hardcoded hinge. Verified against AGT: the numpy trajectory
              matches privacy_certified_training's nominal to 2e-14 for both
              cross_entropy and one-vs-all hinge.
  labels      integers 0..9 throughout (blobs uses {-1,+1} because it is binary)
  PATE votes  the count vector has 10 entries rather than 2. One record still sits
              in one teacher and still moves at most one vote, so the l1 sensitivity
              is still 2 and the l2 sensitivity still sqrt(2) -- the counts being
              longer does not change either.

Sensitivities used by DP-SGD follow from element-wise clipping alone and are
therefore loss-independent, exactly as in blobs; only P changes, from d+1 = 9 to
n_classes*(d+1) = 650.

  non_private   the ceiling
  dpsgd_pure    Laplace noise every step, basic composition at eps/T
  pate          disjoint teachers label a public pool by noisy argmax; a student is
                trained on those labels and released
  objpert       Chaudhuri objective perturbation (requires lambda > 0)
"""

import numpy as np

from ags.mnist.shared import N_CLASSES, accuracy, logits
from ags.mnist.losses import GRADS

DEFAULT_LOSS = "cross_entropy"


def n_params(d, n_classes=N_CLASSES):
    return n_classes * d + n_classes


def train(
    X,
    y,
    *,
    n_s,
    lr,
    gamma,
    lam,
    b=None,
    loss=DEFAULT_LOSS,
    margin=1.0,
    noise_scale=0.0,
    rng=None,
    theta0=None,
    noise_kind="laplace",
    n_classes=N_CLASSES,
    lr_decay=0.0,
    lr_min=0.0,
):
    """Element-wise clipped SGD, matching AGT's clip_method='clamp'.

    Fixed cyclic batching, as blobs/baselines.py:train_hinge does, so the algorithm
    AGS certifies and the one the baselines run are the same object.

    lr_decay is the reference's `eta`: lr_n = max(lr/(1 + lr_decay*n), lr_min), applied
    per step, matching abstract_gradient_training's bounded_sgd with
    lr_decay_type="batch" so the certified and nominal trajectories stay identical.
    """
    n, d = X.shape
    b = b or n
    P = n_params(d, n_classes)
    theta = np.zeros(P) if theta0 is None else theta0.copy()
    grad_fn = GRADS[loss]
    spe = max(1, n // b)
    for t in range(n_s):
        sl = slice((t % spe) * b, ((t % spe) + 1) * b)
        Xb, yb = X[sl], y[sl]
        g = np.clip(grad_fn(logits(theta, Xb, n_classes), yb, Xb, n_classes, margin), -gamma, gamma)
        upd = g.sum(0) / len(Xb)
        if noise_scale > 0:
            upd = upd + (
                rng.normal(0.0, noise_scale, size=P)
                if noise_kind == "gauss"
                else rng.laplace(0.0, noise_scale, size=P)
            )
        lr_t = max(lr / (1.0 + lr_decay * t), lr_min) if lr_decay else lr
        theta = (1.0 - lr_t * lam) * theta - lr_t * upd
    return theta


def acc(theta, X, y, n_classes=N_CLASSES):
    return accuracy(theta, X, y, n_classes)


def non_private(
    X, y, *, n_s, lr, gamma, lam, b=None, loss=DEFAULT_LOSS, lr_decay=0.0, lr_min=0.0, n_classes=N_CLASSES
):
    return train(
        X,
        y,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        loss=loss,
        lr_decay=lr_decay,
        lr_min=lr_min,
        n_classes=n_classes,
    )


def dpsgd_pure(
    X,
    y,
    eps,
    *,
    n_s,
    lr,
    gamma,
    lam,
    seed=0,
    b=None,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
    lr_decay=0.0,
    lr_min=0.0,
):
    """Pure eps-DP. Element-wise clip => l1 sensitivity of the mean grad is 2.P.gamma/b.

    Basic composition only: pure DP admits neither advanced composition nor
    subsampling amplification, so each step gets eps/n_s.
    """
    n, d = X.shape
    b = b or n
    P = n_params(d, n_classes)
    delta1 = 2.0 * P * gamma / b
    scale = delta1 * n_s / eps
    return train(
        X,
        y,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        loss=loss,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        lr_decay=lr_decay,
        lr_min=lr_min,
        n_classes=n_classes,
    ), scale


def dpsgd_gaussian(
    X,
    y,
    eps,
    delta,
    *,
    n_s,
    lr,
    gamma,
    lam,
    seed=0,
    b=None,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
    lr_decay=0.0,
    lr_min=0.0,
):
    """(eps, delta)-DP. Port of experiments/common/ags.py:train_gaussian_dpsgd (L757-770).

    Element-wise clipping to gamma gives per-coordinate sensitivity 2*gamma/b, so the
    Gaussian mechanism sees l2 sensitivity sqrt(P) * 2*gamma/b. Basic composition over
    n_s steps, exactly as the reference does: eps_per_step = eps / n_s.
    """
    n, d = X.shape
    b = b or n
    P = n_params(d, n_classes)
    eps_step = eps / n_s
    l2 = np.sqrt(P) * 2.0 * gamma / b
    scale = (l2 / eps_step) * np.sqrt(2.0 * np.log(1.25 / delta))
    return train(
        X,
        y,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        loss=loss,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        noise_kind="gauss",
        lr_decay=lr_decay,
        lr_min=lr_min,
        n_classes=n_classes,
    ), scale


def _teachers_and_votes(
    X, y, Xpub, *, n_teachers, n_queries, n_s, lr, gamma, lam, loss, rng, n_classes=N_CLASSES
):
    """Disjoint shards -> teachers -> per-class vote counts on the public pool."""
    idx = rng.permutation(len(X))
    shards = np.array_split(idx, n_teachers)
    teachers = [
        train(X[s], y[s], n_s=n_s, lr=lr, gamma=gamma, lam=lam, loss=loss, n_classes=n_classes)
        for s in shards
    ]
    q = min(n_queries, len(Xpub))
    Xq = Xpub[:q]
    preds = np.stack([logits(t, Xq, n_classes).argmax(1) for t in teachers])  # (m, q)
    counts = np.stack([(preds == c).sum(0) for c in range(n_classes)]).astype(float)
    return Xq, counts, q


def pate(
    X,
    y,
    Xpub,
    eps,
    *,
    n_teachers,
    n_queries,
    n_s,
    lr,
    gamma,
    lam,
    seed=0,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
):
    """Teachers vote on a public pool with Laplace noise; a student is trained on the
    resulting labels and released.

    One record sits in one teacher, so it moves at most one vote: the count vector has
    l1 sensitivity 2. Each query is therefore eps_q-DP with Laplace(2/eps_q), and basic
    composition over n_queries gives eps = n_queries * eps_q.
    """
    rng = np.random.RandomState(seed)
    Xq, counts, q = _teachers_and_votes(
        X,
        y,
        Xpub,
        n_teachers=n_teachers,
        n_queries=n_queries,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        loss=loss,
        rng=rng,
        n_classes=n_classes,
    )
    eps_q = eps / q
    counts = counts + rng.laplace(0.0, 2.0 / eps_q, size=counts.shape)
    labels = counts.argmax(0)
    student = train(Xq, labels, n_s=n_s, lr=lr, gamma=gamma, lam=lam, loss=loss, n_classes=n_classes)
    return student, eps_q


def pate_gaussian(
    X,
    y,
    Xpub,
    eps,
    delta,
    *,
    n_teachers,
    n_queries,
    n_s,
    lr,
    gamma,
    lam,
    seed=0,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
):
    """(eps, delta)-DP PATE. One record moves one vote between two counts, so the count
    vector has l2 sensitivity sqrt(2). Gaussian per query, basic composition over q."""
    rng = np.random.RandomState(seed)
    Xq, counts, q = _teachers_and_votes(
        X,
        y,
        Xpub,
        n_teachers=n_teachers,
        n_queries=n_queries,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        loss=loss,
        rng=rng,
        n_classes=n_classes,
    )
    eps_q, delta_q = eps / q, delta / q
    sig = (np.sqrt(2.0) / eps_q) * np.sqrt(2.0 * np.log(1.25 / delta_q))
    counts = counts + rng.normal(0.0, sig, size=counts.shape)
    labels = counts.argmax(0)
    return train(Xq, labels, n_s=n_s, lr=lr, gamma=gamma, lam=lam, loss=loss, n_classes=n_classes), eps_q


def objective_perturbation(X, y, eps, lam, seed=0, c=0.25, n_classes=N_CLASSES):
    """Chaudhuri Alg. 2 for a convex loss with l2 regulariser; needs lambda > 0.

    The objective is multinomial logistic here rather than blobs' binary hinge.
    Chaudhuri's Alg. 2 wants a differentiable convex loss, which multinomial logistic
    satisfies and hinge satisfies only at the subgradient -- so this arm is on firmer
    ground in this study than it was in the blobs one.
    """
    from scipy.optimize import minimize

    if lam <= 0:
        raise ValueError("objective perturbation requires lambda > 0")
    rng = np.random.RandomState(seed)
    n, d = X.shape
    P = n_params(d, n_classes)
    eps_p = eps - 2.0 * np.log(1.0 + c / (n * lam))
    delta = 0.0
    if eps_p <= 0:
        delta = max(c / (n * (np.exp(eps / 4.0) - 1.0)) - lam, 0.0)
        eps_p = eps / 2.0
    nrm = rng.gamma(P, 2.0 / eps_p)
    bvec = rng.normal(size=P)
    bvec = nrm * bvec / np.linalg.norm(bvec)

    onehot = np.eye(n_classes)[y]

    def obj(th):
        z = logits(th, X, n_classes)
        z = z - z.max(1, keepdims=True)
        ll = -(onehot * (z - np.log(np.exp(z).sum(1, keepdims=True)))).sum(1).mean()
        return ll + 0.5 * (lam + delta) * th @ th + (bvec @ th) / n

    return minimize(obj, np.zeros(P), method="L-BFGS-B").x


def dpsgd_rdp_fixed(
    X,
    y,
    eps,
    delta,
    *,
    n_s,
    lr,
    gamma,
    lam,
    b,
    seed=0,
    adjacency="substitution",
    accountant="rdp",
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
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

    n, d = X.shape
    P = n_params(d, n_classes)
    tgt_e, tgt_d = (eps / 2.0, delta / 2.0) if adjacency == "substitution" else (eps, delta)
    sigma_mult = get_noise_multiplier(
        target_epsilon=tgt_e, target_delta=tgt_d, sample_rate=1.0, steps=n_s, accountant=accountant
    )
    C = gamma * np.sqrt(P)  # l2 bound implied by element-wise clip
    scale = sigma_mult * C / b  # std on the MEAN gradient
    return train(
        X,
        y,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        loss=loss,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        noise_kind="gauss",
        n_classes=n_classes,
    ), sigma_mult


def dpsgd_opacus(
    X,
    y,
    eps,
    delta,
    *,
    n_s,
    lr,
    gamma,
    lam,
    b,
    seed=0,
    adjacency="substitution",
    accountant="rdp",
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
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
    add/remove is (eps, delta)-DP under substitution.

    Clipping stays element-wise at gamma to match AGS, which bounds the per-sample
    l2 norm by gamma*sqrt(P) -- the same bound the reference's Gaussian baseline uses.
    """
    from opacus.accountants.utils import get_noise_multiplier

    n, d = X.shape
    P = n_params(d, n_classes)
    q = b / n
    tgt_e, tgt_d = (eps / 2.0, delta / 2.0) if adjacency == "substitution" else (eps, delta)
    sigma_mult = get_noise_multiplier(
        target_epsilon=tgt_e, target_delta=tgt_d, sample_rate=q, steps=n_s, accountant=accountant
    )
    C = gamma * np.sqrt(P)
    noise_std = sigma_mult * C  # std on the SUM of clipped gradients
    rng = np.random.RandomState(seed)
    theta = np.zeros(P)
    grad_fn = GRADS[loss]
    for _ in range(n_s):
        idx = np.flatnonzero(rng.random_sample(n) < q)  # Poisson sample
        if idx.size:
            g = np.clip(
                grad_fn(logits(theta, X[idx], n_classes), y[idx], X[idx], n_classes, 1.0), -gamma, gamma
            )
            summed = g.sum(0)
        else:
            summed = np.zeros(P)
        upd = (summed + rng.normal(0.0, noise_std, size=P)) / (q * n)
        theta = (1.0 - lr * lam) * theta - lr * upd
    return theta, sigma_mult


# ------------------------------------------------------------------------------------------------
# The arms of ags/mnist/run_optimized.py
# ------------------------------------------------------------------------------------------------


def dpsgd_gaussian_composed(
    X,
    y,
    eps,
    delta,
    *,
    n_s,
    lr,
    gamma,
    lam,
    b=None,
    seed=0,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
    lr_decay=0.0,
    lr_min=0.0,
):
    """Matched Gaussian DP-SGD under exact Gaussian composition (the accounting lineage's baseline).

    Same fixed-cyclic trajectory and element-wise clip as AGS certifies; each step is one Gaussian
    release of the mean gradient with per-example l2 bound gamma * sqrt(P), and the n_s releases are
    composed exactly (common.accounting.exact_accounting.calibrate_composed_gaussian, replace-one).
    Returns (theta, noise_multiplier).
    """
    from common.accounting.exact_accounting import calibrate_composed_gaussian

    n, d = X.shape
    b = b or n
    P = n_params(d, n_classes)
    multiplier, _ = calibrate_composed_gaussian(eps, delta, n_s)
    scale = multiplier * gamma * np.sqrt(P) / b
    theta = train(
        X,
        y,
        n_s=n_s,
        lr=lr,
        gamma=gamma,
        lam=lam,
        b=b,
        loss=loss,
        noise_scale=scale,
        rng=np.random.RandomState(seed),
        noise_kind="gauss",
        lr_decay=lr_decay,
        lr_min=lr_min,
        n_classes=n_classes,
    )
    return theta, multiplier


def _poisson_steps(
    X,
    y,
    *,
    steps,
    lr,
    q,
    rng,
    per_step_noise,
    clip_rows,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
    lr_decay=0.0,
):
    """Poisson-sampled SGD on the logistic head: per-example gradients, a clip, noise on the sum."""
    n, d = X.shape
    P = n_params(d, n_classes)
    theta = np.zeros(P)
    grad_fn = GRADS[loss]
    expected = q * n
    for t in range(steps):
        mask = rng.random_sample(n) < q
        Xb, yb = X[mask], y[mask]
        if len(yb):
            g = clip_rows(grad_fn(logits(theta, Xb, n_classes), yb, Xb, n_classes))
            total = g.sum(0)
        else:
            total = np.zeros(P)
        total = total + per_step_noise(rng, P)
        lr_t = lr / (1.0 + lr_decay * t) if lr_decay else lr
        theta = theta - lr_t * total / expected
    return theta


_MULTIPLIERS: dict = {}


def poisson_gaussian_multiplier(eps, delta, *, q, steps, interval=1e-3):
    """The PLD-calibrated noise multiplier for Poisson-sampled Gaussian DP-SGD, cached: it depends on
    (eps, delta, q, steps, interval) only, and one calibration composes `steps` PLDs a few dozen times."""
    import contextlib
    import io

    from common.accounting.accounting import calibrate_dpsgd

    key = (float(eps), float(delta), float(q), int(steps), float(interval))
    if key not in _MULTIPLIERS:
        with contextlib.redirect_stdout(io.StringIO()):
            _MULTIPLIERS[key] = calibrate_dpsgd(
                eps, delta, q=q, steps=steps, interval=interval
            ).mechanism_parameter
    return _MULTIPLIERS[key]


def dpsgd_poisson_gaussian(
    X,
    y,
    eps,
    delta,
    *,
    steps,
    lr,
    clip,
    q,
    seed=0,
    loss=DEFAULT_LOSS,
    n_classes=N_CLASSES,
    lr_decay=0.0,
    interval=1e-3,
):
    """Optimized DP-SGD: Poisson sampling, per-example l2 clipping, Gaussian noise, PLD accounting.

    The noise multiplier comes from common.accounting.accounting.calibrate_dpsgd (dp_accounting PLD,
    replace-one relation, Poisson-subsampled Gaussian composed over `steps`), so the amplification
    claim is exactly the one the accountant certifies. Returns (theta, noise_multiplier).
    """
    m = poisson_gaussian_multiplier(eps, delta, q=q, steps=steps, interval=interval)
    rng = np.random.RandomState(seed)

    def clip_rows(g):
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        return g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))

    theta = _poisson_steps(
        X,
        y,
        steps=steps,
        lr=lr,
        q=q,
        rng=rng,
        loss=loss,
        n_classes=n_classes,
        lr_decay=lr_decay,
        clip_rows=clip_rows,
        per_step_noise=lambda r, P: r.normal(0.0, m * clip, size=P),
    )
    return theta, m


def dpsgd_poisson_laplace(
    X, y, eps, *, steps, lr, gamma, q, seed=0, loss=DEFAULT_LOSS, n_classes=N_CLASSES, lr_decay=0.0
):
    """Optimized pure-DP DP-SGD: Poisson sampling with amplification, element-wise clip, Laplace.

    Basic composition over `steps`, each step eps/steps on the full data, achieved by a base budget
    eps_s = log(1 + (e^(eps/steps) - 1) / q) on the Poisson sample (Balle et al., replace-one). The
    element-wise clip gives the summed gradient l1 sensitivity 2 P gamma. Returns (theta, eps_s).
    """
    n, d = X.shape
    P = n_params(d, n_classes)
    eps_step = eps / steps
    eps_s = np.log1p(np.expm1(eps_step) / q) if q < 1 else eps_step
    scale = 2.0 * P * gamma / eps_s
    rng = np.random.RandomState(seed)
    theta = _poisson_steps(
        X,
        y,
        steps=steps,
        lr=lr,
        q=q,
        rng=rng,
        loss=loss,
        n_classes=n_classes,
        lr_decay=lr_decay,
        clip_rows=lambda g: np.clip(g, -gamma, gamma),
        per_step_noise=lambda r, P_: r.laplace(0.0, scale, size=P_),
    )
    return theta, eps_s
