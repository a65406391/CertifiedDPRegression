"""Data, model and run-logging shared across the MNIST PCA-64 ablation.

Port of AGS_Ablation/blobs/common.py. Two things change and nothing else:

  load_blobs   -> load_mnist_pca    64-dimensional PCA of MNIST, all 10 classes
  linear_model -> logistic_model    Sequential(Linear(64, 10)), P = 650

`run_id` and `save_bound` are copied verbatim; only the runs/ root moves under
mnist_pca/ so the two studies never write to each other's directories.

LABEL CONVENTION. blobs carries labels in {-1, +1} because it is binary. This study
is 10-class, so labels are INTEGERS 0..9 everywhere -- which is what AGT's
BoundedCrossEntropyLoss expects. The one-vs-all hinge arm derives its own
(n, 10) sign matrix internally; no parallel label array is ever carried around.

PARAMETER ORDER. torch flattens Linear(d, C) as weight (C, d) row-major, then bias
(C,), so a flat theta is [W.ravel(), b] with W of shape (C, d). `unpack` below is
the only place that ordering is written down, and every module goes through it.

Every run appends a record to runs/history.json and writes its parameter bound to
runs/parameter_bounds/<run_id>.npz, so the full trace of computed bounds is
recoverable without re-running anything.
"""

import gzip
import hashlib
import json
import os
import struct
import time

import numpy as np
import torch

RUNS = "ags/mnist/runs"
BOUNDS = f"{RUNS}/parameter_bounds"
HISTORY = f"{RUNS}/history.json"

# The raw IDX files already on disk from the AGS_Extensions work -- no download, and
# no torchvision dependency in the certified path.
MNIST_DIR = "ags/mnist/data/mnist"

N_CLASSES = 10


def _idx(path, labels):
    """Read a gzipped IDX file. Labels are 1-D uint8; images flatten to 784."""
    with gzip.open(path, "rb") as f:
        if labels:
            struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).copy()
        _, count, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows * cols).copy()


def load_mnist_pca(N, d=64, seed=0, n_test=2000, root=MNIST_DIR, scale_unit_norm=True):
    """MNIST reduced to `d` PCA components, all 10 classes.

    Returns (X, y, X_test, y_test) with y integer in 0..9, in the same positional
    convention load_blobs uses, so every downstream module keeps its signature.

    Three preprocessing choices, each matching an existing arm of this project:

    - Standardisation (0.1307, 0.3081) then PCA with whiten=True, following
      experiments/mnist/run_ags_comparison.py:load_mnist and PCA_DIMS=64.
    - PCA is fitted on the PRIVATE training split. This is not accounted for in
      epsilon. It is the same public-preprocessing assumption the paper's MNIST arm
      and AGS_Extensions/experiments/mnist_pca_benchmark.py:load_data both make, and
      it is inherited deliberately rather than silently.
    - Features are scaled so max ||x||_2 <= 1, which is what load_blobs does and what
      the objective-perturbation baseline requires.
    """
    xtr = _idx(f"{root}/train-images-idx3-ubyte.gz", False)
    ytr = _idx(f"{root}/train-labels-idx1-ubyte.gz", True)
    xte = _idx(f"{root}/t10k-images-idx3-ubyte.gz", False)
    yte = _idx(f"{root}/t10k-labels-idx1-ubyte.gz", True)

    rng = np.random.RandomState(seed)
    itr = rng.choice(len(xtr), N, replace=False)
    ite = rng.choice(len(xte), min(n_test, len(xte)), replace=False)

    X = (xtr[itr].astype(np.float64) / 255.0 - 0.1307) / 0.3081
    Xte = (xte[ite].astype(np.float64) / 255.0 - 0.1307) / 0.3081

    from sklearn.decomposition import PCA

    pca = PCA(n_components=d, whiten=True, random_state=seed)
    X, Xte = pca.fit_transform(X), pca.transform(Xte)

    if scale_unit_norm:
        r = np.linalg.norm(X, axis=1).max()
        X, Xte = X / r, Xte / r

    return X, ytr[itr].astype(np.int64), Xte, yte[ite].astype(np.int64)


def reference_accuracy(X, y, Xte, yte):
    """Unregularised multinomial logistic fit -- a REFERENCE POINT, not a ceiling.

    Stands in for blobs' analytic bayes_accuracy(), which has no MNIST counterpart.
    Note what it is and is not: with C=1e6 and N=2000 this overfits, and clipped SGD
    measurably BEATS it (0.8615 vs 0.8080 at N=2000), because element-wise clipping
    is itself a regulariser. So it is a like-for-like "what does the model class do
    unconstrained" number, and must not be quoted as an upper bound on any arm.
    """
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    return float(m.score(Xte, yte))


def logistic_model(d, n_classes=N_CLASSES, seed=0, zero_init=True):
    """Multinomial logistic regression head: Sequential(Linear(d, n_classes)).

    d=64, n_classes=10 gives P = 650, which is the paper's MNIST arm
    (experiments/mnist/config.py: PCA_DIMS=64, HIDDEN_DIMS=[], OUTPUT_DIM=10).

    Zero-initialised by default so AGT's nominal trajectory matches the numpy
    baselines exactly. With a random torch init the arms start from different points,
    which is invisible at large n_s and dominates at small n_s.
    """
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(d, n_classes)).double()
    if zero_init:
        with torch.no_grad():
            for prm in m.parameters():
                prm.zero_()
    return m


def unpack(theta, d, n_classes=N_CLASSES):
    """Flat theta -> (W, b) in torch's Linear(d, C) order: W is (C, d), b is (C,)."""
    theta = np.asarray(theta, float).ravel()
    return theta[: n_classes * d].reshape(n_classes, d), theta[n_classes * d :]


def pack(W, b):
    """(W, b) -> flat theta, inverse of unpack."""
    return np.concatenate([np.asarray(W, float).ravel(), np.asarray(b, float).ravel()])


def logits(theta, X, n_classes=N_CLASSES):
    """Forward pass of the linear head: (n, d) -> (n, C)."""
    W, b = unpack(theta, X.shape[1], n_classes)
    return X @ W.T + b


def accuracy(theta, X, y, n_classes=N_CLASSES):
    """Top-1 accuracy of a flat parameter vector."""
    return float((logits(theta, X, n_classes).argmax(1) == np.asarray(y)).mean())


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
    hist = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    hist = [r for r in hist if r["run_id"] != rid] + [rec]
    os.makedirs(RUNS, exist_ok=True)
    json.dump(hist, open(HISTORY, "w"), indent=1, default=str)
    return path
