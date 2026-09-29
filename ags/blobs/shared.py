"""Data, model and run-logging shared across the blobs ablation.

Every run appends a record to runs/history.json and writes its parameter bound to
runs/parameter_bounds/<run_id>.npz, so the full trace of computed bounds is
recoverable without re-running anything.
"""

import hashlib
import fcntl
import json
import os
import time

import math

import numpy as np
import torch

RUNS = "ags/blobs/runs"
BOUNDS = f"{RUNS}/parameter_bounds"
HISTORY = f"{RUNS}/history.json"


def load_blobs(N, d, sep=4.0, std=1.0, seed=0, n_test=4000, aniso=1.0):
    """Two Gaussian blobs separated along coordinate 0.

    `sep` sets the Bayes rate via sep/std. `aniso` inflates the noise in the d-1
    NON-discriminative coordinates: the Bayes rate is unchanged (the signal direction
    and its noise are untouched), but a small sample is dominated by variance in
    directions carrying no signal, so it estimates the boundary badly. That raises
    SAMPLE COMPLEXITY while holding the achievable ceiling fixed -- which is what
    separates a data-hungry baseline (PATE's per-teacher shards) from AGS, whose
    envelope has no term for either quantity.
    """
    # generated directly rather than via sklearn (broken scipy in this env)
    rng = np.random.RandomState(seed)
    m = N + n_test
    y = rng.randint(0, 2, m)
    centres = np.zeros((2, d))
    centres[0, 0], centres[1, 0] = -sep / 2, sep / 2
    noise = rng.randn(m, d)
    noise[:, 1:] *= aniso  # nuisance directions only; coordinate 0 untouched
    X = centres[y] + std * noise
    # scale so ||x||_2 <= 1, which the objective-perturbation baseline requires
    r = np.linalg.norm(X, axis=1).max()
    X = X / r
    ys = (2 * y - 1).astype(np.float64)  # hinge convention
    return X[:N], ys[:N], X[N:], ys[N:]


def bayes_accuracy(sep, std, d):
    """Analytic Bayes rate for two isotropic Gaussians separated along one axis."""
    z = sep / (2.0 * std)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def linear_model(d, seed=0, zero_init=True):
    """Zero-initialised by default so AGT's nominal trajectory matches the numpy
    baselines exactly. With a random torch init the arms start from different points,
    which is invisible at large n_s and dominates at small n_s."""
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(d, 1)).double()
    if zero_init:
        with torch.no_grad():
            for prm in m.parameters():
                prm.zero_()
    return m


def relu_model(d, width, depth=1, seed=0):
    torch.manual_seed(seed)
    layers, prev = [], d
    for _ in range(depth):
        layers += [torch.nn.Linear(prev, width), torch.nn.ReLU()]
        prev = width
    layers += [torch.nn.Linear(prev, 1)]
    return torch.nn.Sequential(*layers).double()


def accuracy(theta_flat, model_shapes, X, ys):
    """Accuracy of a flat parameter vector for a linear head."""
    d = X.shape[1]
    w, b = theta_flat[:d], theta_flat[d]
    return float((np.sign(X @ w + b) == np.sign(ys)).mean())


def run_id(tag, cfg):
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{tag}_{h}"


def save_bound(rid, param_l, param_n, param_u, cfg, extra=None):
    """Serialize a parameter bound and append it to the run history."""
    os.makedirs(BOUNDS, exist_ok=True)
    flat = lambda ps: np.concatenate([np.asarray(p).ravel() for p in ps])
    path = f"{BOUNDS}/{rid}.npz"
    np.savez_compressed(
        path,
        param_l=flat(param_l),
        param_n=flat(param_n),
        param_u=flat(param_u),
        shapes=np.array([np.asarray(p).shape for p in param_l], dtype=object),
        cfg=np.array(json.dumps(cfg, default=str)),
    )
    rec = dict(run_id=rid, path=path, t=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=cfg, **(extra or {}))
    os.makedirs(RUNS, exist_ok=True)
    # Multiple serialized radius jobs run concurrently. Protect the secondary
    # history index from lost read-modify-write updates; each primary run
    # directory remains independently serialized as before.
    with open(HISTORY + ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hist = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
        hist = [r for r in hist if r["run_id"] != rid] + [rec]
        temporary = f"{HISTORY}.{os.getpid()}.tmp"
        with open(temporary, "w") as output:
            json.dump(hist, output, indent=1, default=str)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, HISTORY)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return path


def random_relu_features(X, n_features, seed=0, scale=1.0):
    """Frozen random ReLU features: phi(x) = relu(W x + b), W,b fixed and NOT trained.

    The MILP formulation supports exactly one TRAINABLE layer -- per-sample parameter
    gradients through a deeper stack need per-sample backprop of dl_dx, which the module
    chain does not carry (formulations/modules.py:349). Folding a frozen first layer into
    the data sidesteps that entirely: the certified model becomes Sequential(Linear(h,1))
    on phi(x), which is a single trainable layer.

    This is exactly the frozen-backbone setting the paper already uses (PLAN.md risk 6,
    the frozen-encoder + logistic head arm). Note the honest limitation: the trainable problem is
    then CONVEX again, so this tests a non-linear model class, not non-convex training.
    """
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.randn(d, n_features) * (scale / np.sqrt(d))
    b = rng.randn(n_features) * 0.1
    Phi = np.maximum(0.0, X @ W + b)
    nrm = np.linalg.norm(Phi, axis=1).max()
    if nrm > 0:
        Phi = Phi / nrm  # keep ||phi(x)||_2 <= 1, as load_blobs does for x
    return Phi, (W, b)
