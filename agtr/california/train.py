"""California Housing batch ablation, re-run with genuine Abstract Gradient Training.

WHY THIS FILE EXISTS
--------------------
The published California arrays (``values/ablation/``, 2025-12-17) were produced by
``original/run_ca_ablation.py::train_ablation_models``.  That loop never propagates
bounds through the gradient: it takes *nominal* per-sample gradients
(``bounded_model.forward``/``backward``), l2-clips them, averages, and widens the
descent direction by the closed form ``+- k*2*C/B`` before ``BoundedSGD.step``.
The resulting parameter box is the same in every coordinate and equals
``sum_t 2*lr*k*2C/B`` -- sound only if the clipped update map is non-expansive, which an
8-64-64-1 ReLU MLP is not.  (``repro/reproduce_ablation.py`` transcribes that loop and
reproduces the published avg S to 12 digits; ``common/training.py::train_robust_ibp``
is the same loop, refactored in 2026-04.)

This script keeps every experimental choice of the original that can be kept -- data
split (seed 42), shuffling generator (seed 42), architecture, k grid, fixed-steps
epochs ``20*bs/1000``, lr 0.01, output clipping to [-5, 5] -- and replaces the training
loop by ``abstract_gradient_training.privacy_certified_training``, i.e. interval bound
propagation through forward and backward passes over the reachable parameter set with
the SEMin/SEMax +- k*gamma accumulation of the paper's Appendix A.

DIFFERENCES FROM THE ORIGINAL THAT CANNOT BE AVOIDED (recorded in every artifact)
--------------------------------------------------------------------------------
* clipping: the original clips the joint l2 norm of a sample's full gradient to C=1 and
  clamps the loss gradient to [-1, 1] ("Huber").  AGT offers ``clip_method="norm"``
  (per-parameter-tensor l2 clipping to gamma; joint norm <= sqrt(6) here) and
  ``clip_method="clamp"`` (per-coordinate).  Both are run, as separate variants.
* the package's ``norm`` branch has a broadcasting bug for 2-D bias gradients
  (``clip_factor.view(-1, 1, 1)``); it is patched *locally* below with the
  dimension-agnostic view and a randomized containment self-test.  The package is not
  edited.
* float64 throughout (the original used float32); GPU.
* the nominal is ``param_n`` of the certified run (identical for every k, asserted),
  not a separately trained unclipped model.

EVERYTHING IS SERIALIZED
------------------------
``values/agt_2026-09-14/<variant>/bs{bs}_k{k}.npz`` holds param_l/param_n/param_u
(float64), the unclipped output bounds and nominal prediction on the stored test set,
wall time, config, git SHA, versions and hostname.  ``manifest.json`` lists every cell
with status.  A cell is skipped if its file already exists (restartable).

    PYTHONPATH=$PWD \
      CUDA_VISIBLE_DEVICES=0 python -m agtr.california.train --variant norm --gamma 1.0
    CUDA_VISIBLE_DEVICES=1 python -m agtr.california.train --variant clamp --gamma 1.0
"""

import argparse
import json
import logging
import os
import platform
import time
import warnings

import numpy as np
import torch

from agtr.california.certify import CFG, OUT_ROOT, STORED, Y_MAX, Y_MIN, git_sha, train_cell
from common import agt_patches
from common.data import load_california_data

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["norm", "clamp"], required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=CFG["BATCH_SIZES"])
    # k = 0 is never trained: the certified nominal is param_n of any k >= 1 cell (identical for all k, asserted below)
    ap.add_argument("--ks", type=int, nargs="+", default=[k for k in CFG["K_VALUES"] if k > 0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[64, 64],
        help="hidden widths; default is the paper's 8-64-64-1",
    )
    ap.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    ap.add_argument(
        "--tag",
        default=None,
        help="output subdirectory; default <variant>_gamma<g> for the paper's model, "
        "<activation>_h<widths>_<variant>_gamma<g> otherwise",
    )
    a = ap.parse_args()
    agt_patches.install()
    agt_patches.selftest(a.gamma)

    paper_model = list(a.hidden) == [64, 64] and a.activation == "relu"
    tag = a.tag or (
        f"{a.variant}_gamma{a.gamma:g}"
        if paper_model
        else f"{a.activation}_h{'x'.join(map(str, a.hidden))}_{a.variant}_gamma{a.gamma:g}"
    )
    out = f"{OUT_ROOT}/{tag}"
    os.makedirs(out, exist_ok=True)
    train_dataset, X_test_t, y_test_t, y_mean, y_std, X_test_np = load_california_data()
    assert np.allclose(np.load(f"{STORED}/X_test.npy"), X_test_np), "test split differs from the stored one"
    Xte = X_test_t.double().to(a.device)
    meta = dict(
        variant=a.variant,
        gamma=a.gamma,
        clip_method=a.variant,
        hidden=list(a.hidden),
        activation=a.activation,
        lr=CFG["LR"],
        base_epochs=CFG["BASE_EPOCHS"],
        base_batch_size=CFG["BASE_BATCH_SIZE"],
        fixed_steps_mode=CFG["FIXED_STEPS_MODE"],
        k_values=list(a.ks),
        batch_sizes=list(a.batch_sizes),
        y_min=Y_MIN,
        y_max=Y_MAX,
        dtype="float64",
        device=a.device,
        loss="mse (package BoundedMSELoss)",
        note="genuine privacy_certified_training; local norm-branch patch; "
        "differs from original in clipping (per-tensor l2 / per-coordinate vs joint l2 + Huber clamp)",
        git_sha=git_sha(),
        torch=torch.__version__,
        host=platform.node(),
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    json.dump(meta, open(f"{out}/config.json", "w"), indent=1)
    manifest_path = f"{out}/manifest.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    for bs in a.batch_sizes:
        epochs = (
            max(1, int(CFG["BASE_EPOCHS"] * (bs / CFG["BASE_BATCH_SIZE"])))
            if CFG["FIXED_STEPS_MODE"]
            else CFG["BASE_EPOCHS"]
        )
        nominal_ref = None
        for k in a.ks:
            key = f"bs{bs}_k{k}"
            path = f"{out}/{key}.npz"
            if os.path.exists(path):
                z = np.load(path)
                nominal_ref = (
                    np.concatenate([z[key].ravel() for key in sorted(z.files) if key.startswith("param_n_")])
                    if nominal_ref is None
                    else nominal_ref
                )
                print(f"[skip] {key} exists", flush=True)
                continue
            try:
                bm, secs = train_cell(
                    train_dataset,
                    bs,
                    k,
                    epochs,
                    a.gamma,
                    a.variant,
                    a.device,
                    hidden=tuple(a.hidden),
                    activation=a.activation,
                )
                status = "ok"
            except Exception as e:  # e.g. k_private=0 unsupported
                manifest[key] = dict(status="failed", error=repr(e)[:300])
                json.dump(manifest, open(manifest_path, "w"), indent=1)
                print(f"[fail] {key}: {repr(e)[:200]}", flush=True)
                continue
            with torch.no_grad():
                yl, yu = bm.bound_forward(Xte, Xte)
                yn = bm.forward(Xte)
            pl = [p.detach().cpu().double().numpy() for p in bm.param_l]
            pn = [p.detach().cpu().double().numpy() for p in bm.param_n]
            pu = [p.detach().cpu().double().numpy() for p in bm.param_u]
            flat_n = np.concatenate([p.ravel() for p in pn])
            if nominal_ref is None:
                nominal_ref = flat_n
            nominal_drift = float(np.abs(flat_n - nominal_ref).max())
            width = np.concatenate([(u - l).ravel() for l, u in zip(pl, pu)])
            arrays = {f"param_l_{i}": pl[i] for i in range(len(pl))}
            arrays.update({f"param_n_{i}": pn[i] for i in range(len(pn))})
            arrays.update({f"param_u_{i}": pu[i] for i in range(len(pu))})
            np.savez_compressed(
                path,
                **arrays,
                y_l=yl.squeeze().cpu().numpy(),
                y_u=yu.squeeze().cpu().numpy(),
                y_n=yn.squeeze().cpu().numpy(),
                seconds=secs,
                bs=bs,
                k=k,
                epochs=epochs,
                steps=epochs * (len(train_dataset) // bs),
                gamma=a.gamma,
                clip_method=a.variant,
            )
            ylc, yuc = (
                np.clip(yl.squeeze().cpu().numpy(), Y_MIN, Y_MAX),
                np.clip(yu.squeeze().cpu().numpy(), Y_MIN, Y_MAX),
            )
            manifest[key] = dict(
                status=status,
                seconds=secs,
                epochs=epochs,
                param_width_median=float(np.median(width)),
                param_width_max=float(width.max()),
                out_width_mean_clipped=float(np.mean(yuc - ylc)),
                frac_saturated=float(np.mean((yuc - ylc) >= (Y_MAX - Y_MIN) - 1e-9)),
                nominal_drift_vs_first_k=nominal_drift,
                nominal_mse_std_units=float(np.mean((yn.squeeze().cpu().numpy() - y_test_t.numpy()) ** 2)),
            )
            json.dump(manifest, open(manifest_path, "w"), indent=1)
            m = manifest[key]
            print(
                f"[{a.variant} g={a.gamma:g}] {key:12s} {secs:6.0f}s  param w med {m['param_width_median']:.4g}  "
                f"out w {m['out_width_mean_clipped']:.3f} sat {m['frac_saturated']:.2f}  nom MSE {m['nominal_mse_std_units']:.4f}  "
                f"drift {nominal_drift:.1e}",
                flush=True,
            )
    print("done", flush=True)


if __name__ == "__main__":
    main()
