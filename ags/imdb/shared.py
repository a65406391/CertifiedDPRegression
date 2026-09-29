"""IMDB as frozen sentence-encoder features: the embeddings, cached once, and the task split.

The encoder (all-mpnet-base-v2 by default; RoBERTa-large mean-pooled as the alternative) is run once
over the 25k training and 25k test reviews and its mean-pooled, unit-normalised sentence embeddings
are stored under ags/imdb/data/<encoder>_<split>.npz. Nothing private touches the encoder: it is a
public model applied to each review on its own.

`load_task` then declares `n_public` of the training reviews public (the first `n_pool` of them the
PATE pool and the warm-start training set, the rest the selection set) and the remaining 20k private.
PCA to d components is fitted on the public reviews only, so the projection costs no privacy and
rests on no assumption; features are scaled so that every public feature vector has ||x|| <= 1 and
any private or test vector beyond that radius is projected onto the unit ball (a per-example
operation, so it changes no neighbouring relation). Test accuracy is on all 25k test reviews.

    PYTHONPATH=. python -m ags.imdb.shared --encoder mpnet --device cuda:0     # compute the cache
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

DATA_DIR = "ags/imdb/data"
ENCODERS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta-large": "roberta-large",
}
N_CLASSES = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def embedding_path(encoder, split):
    return f"{DATA_DIR}/{encoder}_{split}.npz"


def embeddings(encoder, split, device="cuda:0", batch_size=64, max_length=512):
    """(X float32 (n, H), y int64) for an IMDB split, computed once and cached."""
    path = embedding_path(encoder, split)
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    name = ENCODERS[encoder]
    ds = load_dataset("imdb")[split]
    texts, labels = list(ds["text"]), np.asarray(ds["label"], dtype=np.int64)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    if device != "cpu":
        model = model.half()
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1)  # mean pooling over tokens, as the model card
            out.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy())
            if i % (batch_size * 50) == 0:
                log(f"  {encoder} {split}: {i}/{len(texts)} reviews  {time.time() - t0:.0f}s")
    X = np.concatenate(out).astype(np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, X=X, y=labels, encoder=np.array(name), max_length=max_length)
    os.replace(tmp, path)
    log(f"wrote {path}: {X.shape} in {time.time() - t0:.0f}s")
    return X, labels


_DATA = {}


def load_task(C, d, device="cuda:0"):
    """Public/private split, PCA-d on the public reviews, the unit-ball scaling. Cached per (task, d)."""
    key = (C["task"], int(d))
    if key in _DATA:
        return _DATA[key]
    from sklearn.decomposition import PCA

    X, y = embeddings(C["encoder"], "train", device)
    Xte, yte = embeddings(C["encoder"], "test", device)
    X, Xte = X.astype(np.float64), Xte.astype(np.float64)
    perm = np.random.RandomState(C["seed"] + 11).permutation(len(X))
    pub, priv = perm[: C["n_public"]], perm[C["n_public"] :]
    if C.get("n_private"):
        priv = priv[: C["n_private"]]
    pool, sel = pub[: C["n_pool"]], pub[C["n_pool"] :]
    pca = PCA(n_components=int(d), whiten=True, random_state=C["seed"]).fit(X[pub])
    Z, Zte = pca.transform(X), pca.transform(Xte)
    r = np.linalg.norm(Z[pub], axis=1).max()  # the public radius
    Z, Zte = Z / r, Zte / r

    def ball(A):  # project onto the unit ball: per example, so no neighbouring relation changes
        nrm = np.linalg.norm(A, axis=1, keepdims=True)
        return A / np.maximum(nrm, 1.0)

    Z, Zte = ball(Z), ball(Zte)
    D = dict(
        C=N_CLASSES,
        d=int(d),
        Xfull=Z[priv],
        yfull=y[priv],
        Xpool=Z[pool],
        ypool=y[pool],
        Xsel=Z[sel],
        ysel=y[sel],
        Xte=Zte,
        yte=yte,
        clipped=int((np.linalg.norm(pca.transform(X[priv]) / r, axis=1) > 1).sum()),
    )
    _DATA[key] = D
    log(
        f"task {C['task']} d={d}: private {len(priv)} ({D['clipped']} projected onto the ball), pool {len(pool)}, "
        f"select {len(sel)}, test {len(yte)}; PCA explained {pca.explained_variance_ratio_.sum():.3f}"
    )
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", choices=tuple(ENCODERS), default="mpnet")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for split in ("train", "test"):
        X, y = embeddings(a.encoder, split, a.device)
        log(f"{a.encoder} {split}: {X.shape}, positives {y.mean():.3f}")


if __name__ == "__main__":
    main()
