"""SST-2 for the hybrid experiment: the data, the splits, the sentence embeddings, the driver's loader.

GLUE SST-2 (Socher et al. 2013; HF `glue`/`sst2`): 67,349 training sentences and the 872-sentence dev set,
the test labels being hidden. The encoder is all-mpnet-base-v2 (mean pooling, L2-normalised, as the model
card and ags/imdb/shared.py), frozen or with a LoRA adapter merged in (ags/sst2/finetune.py). Splits, seeded:
the first N_TEST of a permutation of the training set are the test set, the rest is private (capped by the
task's `n_private` for the data-size panel); the 872 dev sentences are the public set (PCA fit, threshold,
selection, and the PATE pool when a baseline needs one). Embeddings are cached under ags/sst2/data/ per
encoder tag ("frozen", or a fine-tune tag).
"""

import os
import time

import numpy as np

DATA_DIR = "ags/sst2/data"
BASE = "sentence-transformers/all-mpnet-base-v2"
N_TEST = 5000
MAX_LEN = 64  # SST-2 sentences average about ten tokens


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def raw():
    """(train sentences, train labels, dev sentences, dev labels)."""
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2")
    tr, dv = ds["train"], ds["validation"]
    return (
        list(tr["sentence"]),
        np.asarray(tr["label"], dtype=np.int64),
        list(dv["sentence"]),
        np.asarray(dv["label"], dtype=np.int64),
    )


def split(n_train, seed=0):
    """Indices into the training set: (test, private) from one seeded permutation."""
    perm = np.random.RandomState(seed + 11).permutation(n_train)
    return perm[:N_TEST], perm[N_TEST:]


def embedding_path(tag):
    return f"{DATA_DIR}/emb_{tag}.npz"


def encoder(adapter_dir=None, device="cuda:0"):
    """The tokenizer and the base encoder, with the LoRA adapter under `adapter_dir` merged in if given."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    return tok, model.to(device).eval()


def pooled(model, tok, texts, device, batch_size=256):
    """Mean-pooled, L2-normalised embeddings (n, 768) of `texts`."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embeddings(tag, adapter_dir=None, device="cuda:0"):
    """Cached (Xtr, ytr, Xdv, ydv) under encoder `tag`; computed once from `adapter_dir` (None: frozen)."""
    path = embedding_path(tag)
    if os.path.exists(path):
        z = np.load(path)
        return z["Xtr"], z["ytr"], z["Xdv"], z["ydv"]
    tr_s, tr_y, dv_s, dv_y = raw()
    tok, model = encoder(adapter_dir, device)
    t0 = time.time()
    Xtr, Xdv = pooled(model, tok, tr_s, device), pooled(model, tok, dv_s, device)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(
        tmp, Xtr=Xtr, ytr=tr_y, Xdv=Xdv, ydv=dv_y, encoder=np.array(BASE), adapter=np.array(adapter_dir or "")
    )
    os.replace(tmp, path)
    log(f"wrote {path}: {Xtr.shape} + {Xdv.shape} in {time.time() - t0:.0f}s")
    return Xtr, tr_y, Xdv, dv_y


_DATA = {}


def load_task(C, d, device=None):
    """The contractive driver's loader (same keys as ags.imdb.shared.load_task): PCA-d, whitened, fitted on the
    public dev sentences; features scaled by the public maximum radius and projected onto the unit ball."""
    key = (C["task"], int(d))
    if key in _DATA:
        return _DATA[key]
    from sklearn.decomposition import PCA

    Xtr, ytr, Xdv, ydv = embeddings(C["encoder"], C.get("adapter_dir"), device or "cuda:0")
    te, priv = split(len(Xtr), C["seed"])
    if C.get("n_private"):
        priv = priv[: C["n_private"]]
    # whitening is a per-task choice made on the public set: it helps the frozen features (0.78 against 0.71
    # tuned-threshold accuracy of the contracted head) and hurts the adapted ones, whose first component
    # carries 95 % of the variance and whose seven others whitening inflates (0.84 against 0.91)
    pca = PCA(n_components=int(d), whiten=C.get("whiten", True), random_state=C["seed"]).fit(Xdv)
    Ztr, Zdv = pca.transform(Xtr), pca.transform(Xdv)
    r = np.linalg.norm(Zdv, axis=1).max()  # the public radius
    Ztr, Zdv = Ztr / r, Zdv / r
    nrm = np.linalg.norm(Ztr, axis=1, keepdims=True)
    clipped = int((nrm[priv] > 1).sum())
    Ztr = Ztr / np.maximum(nrm, 1.0)
    Zdv = Zdv / np.maximum(np.linalg.norm(Zdv, axis=1, keepdims=True), 1.0)
    D = dict(
        C=2,
        d=int(d),
        Xfull=Ztr[priv],
        yfull=ytr[priv],
        Xpool=Zdv,
        ypool=ydv,
        Xsel=Zdv,
        ysel=ydv,
        Xte=Ztr[te],
        yte=ytr[te],
        n_raw_features=Xtr.shape[1],
        clipped=clipped,
        tune_threshold=True,  # 56/44 classes: under contraction the bias dominates the argmax (as on Adult)
    )
    _DATA[key] = D
    log(
        f"task {C['task']} d={d}: private {len(priv)} ({clipped} projected onto the ball), public {len(Zdv)} "
        f"(pool = selection), test {len(te)}; PCA explained {pca.explained_variance_ratio_.sum():.3f}"
    )
    return D
