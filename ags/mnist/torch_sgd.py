"""The same clipped-SGD trajectory as baselines.train, on the GPU.

This is a port, not a re-derivation: every step performs the identical arithmetic to
`baselines.train` with loss="cross_entropy", in float64, and the DP noise is drawn from
`numpy.random.RandomState(seed)` in the same order the numpy path draws it. On identical
inputs the two agree to float64 round-off, which `validate()` checks by running both and
reporting max|theta_gpu - theta_cpu|.

Why it exists: at T = 40000 a single training is ~200 s of numpy on a loaded CPU box and
a few seconds here, because the whole trajectory is 2000x64 @ 64x10 products that a GPU
does in one kernel each.

    PYTHONPATH=$PWD ${PYTHON:-python} -m ags.mnist.torch_sgd   # validate
"""

import numpy as np
import torch

from ags.mnist.shared import N_CLASSES


def pick_device():
    """The CUDA device with the most free memory, or CPU if there is none."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def train(
    X,
    y,
    *,
    n_s,
    lr,
    gamma,
    lam,
    b=None,
    noise=None,
    lr_decay=0.0,
    lr_min=0.0,
    n_classes=N_CLASSES,
    device=None,
    theta0=None,
    tail_average_windows=None,
):
    """Element-wise clipped SGD on cross-entropy, fixed cyclic batching.

    noise: (n_s, P) array added to the mean gradient at each step, or None.
    Returns the flat parameter vector as float64 numpy, in Linear(d, C) order.
    """
    device = device or pick_device()
    n, d = X.shape
    b = b or n
    P = n_classes * d + n_classes
    spe = max(1, n // b)

    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    th = (
        torch.zeros(P, dtype=torch.float64, device=device)
        if theta0 is None
        else torch.as_tensor(theta0, dtype=torch.float64, device=device).clone()
    )
    ns = None if noise is None else torch.as_tensor(noise, dtype=torch.float64, device=device)
    rows = torch.arange(b, device=device)

    windows = sorted(set(int(w) for w in (tail_average_windows or ())))
    if any(w < 1 or w > n_s for w in windows):
        raise ValueError("tail-average windows must lie between one and n_s")
    trajectory_sum = torch.zeros_like(th) if windows else None
    tail_prefix = {w: None for w in windows}

    for t in range(n_s):
        sl = slice((t % spe) * b, ((t % spe) + 1) * b)
        Xb, yb = Xt[sl], yt[sl]
        W = th[: n_classes * d].view(n_classes, d)
        bias = th[n_classes * d :]
        dz = torch.softmax(Xb @ W.T + bias, dim=1)  # (nb, C)
        dz[rows[: len(Xb)], yb] -= 1.0
        g = torch.cat([(dz.unsqueeze(2) * Xb.unsqueeze(1)).reshape(len(Xb), -1), dz], 1)
        upd = g.clamp_(-gamma, gamma).sum(0) / len(Xb)
        if ns is not None:
            upd = upd + ns[t]
        lr_t = max(lr / (1.0 + lr_decay * t), lr_min) if lr_decay else lr
        th = (1.0 - lr_t * lam) * th - lr_t * upd
        if windows:
            for window in windows:
                if t == n_s - window:
                    tail_prefix[window] = trajectory_sum.clone()
            trajectory_sum.add_(th)
    final = th.cpu().numpy()
    if windows:
        return final, {
            window: ((trajectory_sum - tail_prefix[window]) / window).cpu().numpy() for window in windows
        }
    return final


def gaussian_noise(eps, delta, *, n_s, gamma, b, P, seed):
    """The reference's Gaussian mechanism, and its noise drawn exactly as numpy draws it.

    baselines.dpsgd_gaussian calls rng.normal(0, scale, size=P) once per step; a single
    (n_s, P) draw from the same RandomState consumes the stream in the same order, so the
    two are bit-identical.
    """
    l2 = np.sqrt(P) * 2.0 * gamma / b
    scale = (l2 / (eps / n_s)) * np.sqrt(2.0 * np.log(1.25 / delta))
    return np.random.RandomState(seed).normal(0.0, scale, size=(n_s, P)), float(scale)


def validate(n_s=2000, N=2000, d=64, b=2000, lr=0.05, eta=0.01, gamma=0.1, eps=100, delta=1e-5, seed=0):
    """Run numpy and GPU on the same config and report the largest disagreement."""
    from ags.mnist.shared import load_mnist_pca, accuracy
    from ags.mnist import baselines as B
    import time

    X, y, Xte, yte = load_mnist_pca(N, d, seed=0, n_test=2000, scale_unit_norm=False)
    P = N_CLASSES * d + N_CLASSES
    dev = pick_device()
    print(f"device: {dev}  ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'})")

    for tag, kw_np, noise in (
        ("nominal", dict(), None),
        (
            "dp-sgd gaussian",
            dict(eps=eps, delta=delta),
            gaussian_noise(eps, delta, n_s=n_s, gamma=gamma, b=b, P=P, seed=seed),
        ),
    ):
        t0 = time.time()
        if noise is None:
            th_np = B.non_private(X, y, n_s=n_s, lr=lr, gamma=gamma, lam=0.0, b=b, lr_decay=eta)
            nz = None
        else:
            th_np, sc = B.dpsgd_gaussian(
                X, y, eps, delta, n_s=n_s, lr=lr, gamma=gamma, lam=0.0, b=b, seed=seed, lr_decay=eta
            )
            nz = noise[0]
            assert abs(sc - noise[1]) < 1e-15, "noise scale disagrees with baselines"
        t_np = time.time() - t0

        t0 = time.time()
        th_gpu = train(X, y, n_s=n_s, lr=lr, gamma=gamma, lam=0.0, b=b, noise=nz, lr_decay=eta, device=dev)
        t_gpu = time.time() - t0

        dmax = float(np.abs(th_gpu - th_np).max())
        rel = dmax / max(float(np.abs(th_np).max()), 1e-300)
        print(
            f"{tag:<16} max|d theta| = {dmax:.3e}  (rel {rel:.2e})   "
            f"acc np {accuracy(th_np, Xte, yte):.4f} / gpu "
            f"{accuracy(th_gpu, Xte, yte):.4f}   "
            f"[numpy {t_np:.1f}s, gpu {t_gpu:.1f}s, {t_np / max(t_gpu, 1e-9):.0f}x]",
            flush=True,
        )


if __name__ == "__main__":
    validate()
