# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Setting scan for a one-hidden-layer California model under genuine AGT (k = 1 only).

Purpose: pick (activation, width H, clip gamma) for the single-hidden-layer California
re-run before spending a full k-grid on it.  Every cell is k = 1, lr 0.01, fixed-steps
epochs 20*bs/1000, clamp clipping, float64, `privacy_certified_training`.  Records the
median/max parameter width against the ceiling 2*T*lr*gamma, the mean clipped output
width on the stored test set, the saturated fraction, and the nominal test MSE
(standardised units; predict-the-mean = 1.0).  Nothing here is a privacy result; the
output feeds the choice of a configuration for `train.py`.

Probe results that motivated it (2026-09-15, k=1, bs=4000, gamma=0.1):
    ReLU H=64: param width 1% of ceiling, output width 0.33/10, MSE 0.74
    tanh H=64: param width 42% of ceiling, output SATURATED, MSE 0.54
    tanh H=16: param width  2% of ceiling, output width 0.21/10, MSE 0.67

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
      python -m agtr.california.scan_tanh_settings --bs 4000 --out scan_bs4000.json
"""

import argparse
import json
import logging
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

import abstract_gradient_training as agt
from abstract_gradient_training.bounded_models import IntervalBoundedModel

from agtr.california import certify as RA
from common import agt_patches
from common.data import load_california_data

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)
OUT_DIR = "agtr/california/values/agt_2026-09-14/scan_one_layer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--acts", nargs="+", default=["tanh", "relu"])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    a = ap.parse_args()
    agt_patches.install()
    os.makedirs(OUT_DIR, exist_ok=True)
    train_dataset, X_test_t, y_test_t, *_ = load_california_data()
    Xte = X_test_t.double().cuda()
    yt = y_test_t.double()
    bs = a.bs
    epochs = max(1, int(20 * bs / 1000))
    T = epochs * (len(train_dataset) // bs)
    rows = []
    for act in a.acts:
        for H in a.widths:
            for g in a.gammas:
                torch.manual_seed(42)
                model = nn.Sequential(
                    nn.Linear(8, H), nn.Tanh() if act == "tanh" else nn.ReLU(), nn.Linear(H, 1)
                ).double()
                cfg = agt.AGTConfig(
                    loss="mse",
                    n_epochs=epochs,
                    learning_rate=0.01,
                    device="cuda",
                    k_private=1,
                    clip_gamma=g,
                    clip_method="clamp",
                    log_level="ERROR",
                )
                t0 = time.time()
                bm = agt.privacy_certified_training(
                    IntervalBoundedModel(model), cfg, RA.make_loader(train_dataset, bs)
                )
                secs = time.time() - t0
                pw = torch.cat([(u - l).flatten() for l, u in zip(bm.param_l, bm.param_u)]).cpu()
                with torch.no_grad():
                    l, u = bm.bound_forward(Xte, Xte)
                    yn = bm.forward(Xte).squeeze().cpu()
                l = np.clip(l.squeeze().cpu().numpy(), -5, 5)
                u = np.clip(u.squeeze().cpu().numpy(), -5, 5)
                row = dict(
                    act=act,
                    H=H,
                    gamma=g,
                    bs=bs,
                    T=T,
                    secs=secs,
                    param_width_median=float(pw.median()),
                    param_width_max=float(pw.max()),
                    ceiling=2 * T * 0.01 * g,
                    out_width_mean=float(np.mean(u - l)),
                    frac_saturated=float(np.mean((u - l) >= 10 - 1e-9)),
                    nominal_mse_std=float(((yn - yt) ** 2).mean()),
                )
                rows.append(row)
                print(
                    f"{act:4s} H={H:<3d} g={g:<5g} bs={bs:<5d} {secs:4.0f}s | pw med {row['param_width_median']:.4g} ({row['param_width_median'] / row['ceiling']:.0%} of ceiling) | out w {row['out_width_mean']:.3f} sat {row['frac_saturated']:.2f} | MSE {row['nominal_mse_std']:.4f}",
                    flush=True,
                )
                json.dump(rows, open(f"{OUT_DIR}/{a.out}", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
