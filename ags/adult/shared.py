"""Adult (UCI census income) as a low-dimensional tabular task for the contractive comparison and the
sampling-frequency ablation.

The dataset of Chaudhuri, Monteleoni and Sarwate (2011), Iyengar et al. (2019) and Redberg, Koskela and
Wang (2023): predict whether income exceeds 50k from 14 demographic attributes; binary, 48,842 rows
(OpenML "adult", version 2), 45,222 after dropping rows with missing values, 24.8% positives. Fetched
once through scikit-learn and cached under ags/adult/data/adult.npz (raw columns, no preprocessing).

`load_task` follows ags/imdb/shared.py exactly, so the IMDB driver runs unchanged through the
`loader` hook of its config: a fixed seeded split into test (`n_test`), public (`n_public`: the first
`n_pool` the PATE pool and warm-start set, the rest the selection set) and private; one-hot encoding of
the categoricals and standardisation of the numerics fitted on the public rows; PCA to d with whitening
fitted on the public rows only, so the projection costs no privacy; every vector scaled by the public
radius and projected onto the unit ball (per example, so no neighbouring relation changes).

    PYTHONPATH=. python -m ags.adult.shared     # fetch and cache, print the split sizes
"""

from __future__ import annotations

import os
import time

import numpy as np

DATA_DIR = "ags/adult/data"
RAW = f"{DATA_DIR}/adult.npz"
N_CLASSES = 2
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def raw():
    """(numeric (n, 6) float, categorical (n, 8) str, y (n,) int64) with the rows with missing values dropped."""
    if os.path.exists(RAW):
        z = np.load(RAW, allow_pickle=True)
        return z["num"], z["cat"], z["y"]
    from sklearn.datasets import fetch_openml

    ds = fetch_openml("adult", version=2, as_frame=True)
    X, y = ds.data, ds.target
    keep = ~X.isna().any(axis=1)
    X, y = X[keep], y[keep]
    num = X[NUMERIC].to_numpy(dtype=np.float64)
    cat = X[CATEGORICAL].astype(str).to_numpy()
    yy = (y.astype(str) == ">50K").to_numpy().astype(np.int64)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{RAW}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, num=num, cat=cat, y=yy, columns=np.array(NUMERIC + CATEGORICAL))
    os.replace(tmp, RAW)
    log(
        f"wrote {RAW}: {len(yy)} rows (dropped {int((~keep).sum())} with missing values), positives {yy.mean():.3f}"
    )
    return num, cat, yy


_DATA = {}


def load_task(C, d, device=None):
    """The split, the public-fitted preprocessing, PCA-d on the public rows, the unit ball. Cached per (task, d)."""
    key = (C["task"], int(d))
    if key in _DATA:
        return _DATA[key]
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num, cat, y = raw()
    n = len(y)
    perm = np.random.RandomState(C["seed"] + 11).permutation(n)
    te, rest = perm[: C["n_test"]], perm[C["n_test"] :]
    pub, priv = rest[: C["n_public"]], rest[C["n_public"] :]
    if C.get("n_private"):
        priv = priv[: C["n_private"]]
    pool, sel = pub[: C["n_pool"]], pub[C["n_pool"] :]

    scaler = StandardScaler().fit(num[pub])
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat[pub])
    F = np.hstack([scaler.transform(num), onehot.transform(cat)])  # (n, 6 + one-hot)
    pca = PCA(n_components=int(d), whiten=True, random_state=C["seed"]).fit(F[pub])
    Z = pca.transform(F)
    r = np.linalg.norm(Z[pub], axis=1).max()  # the public radius
    Z = Z / r
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(nrm, 1.0)  # onto the unit ball, per example
    D = dict(
        C=N_CLASSES,
        d=int(d),
        Xfull=Z[priv],
        yfull=y[priv],
        Xpool=Z[pool],
        ypool=y[pool],
        Xsel=Z[sel],
        ysel=y[sel],
        Xte=Z[te],
        yte=y[te],
        n_raw_features=F.shape[1],
        clipped=int((nrm[priv] > 1).sum()),
        tune_threshold=True,  # 75/25 classes: the decision threshold is chosen on the public selection set
    )
    _DATA[key] = D
    log(
        f"task {C['task']} d={d}: private {len(priv)} ({D['clipped']} projected onto the ball), pool {len(pool)}, "
        f"select {len(sel)}, test {len(te)}; {F.shape[1]} raw features, PCA explained {pca.explained_variance_ratio_.sum():.3f}"
    )
    return D


def main():
    num, cat, y = raw()
    log(
        f"adult: {len(y)} rows, {num.shape[1]} numeric + {cat.shape[1]} categorical attributes, positives {y.mean():.3f}"
    )
    from ags.adult.config import ADULT

    for d in (16, 24, 32):
        load_task(ADULT, d)


if __name__ == "__main__":
    main()
