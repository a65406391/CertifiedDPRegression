"""Rebuild ``manifest.json`` (and the ``batch_sizes`` list in ``config.json``) of an AGT variant
directory from its ``bs{bs}_k{k}.npz`` cells.

Needed because the per-batch-size workers of ``train.py`` run in parallel and each
rewrites the shared manifest, so the file on disk only reflects the last worker.  Every field is
recomputed from the npz arrays (no cell is modified); ``status`` is ``ok`` for every saved cell.

    PYTHONPATH=$PWD python -m agtr.california.rebuild_manifest --variant relu_h64_clamp_gamma0.1
"""

import argparse
import glob
import json
import re

import numpy as np

from agtr.california.config import ABLATION as CFG

ROOT = "agtr/california/values/agt_2026-09-14"
STORED = "agtr/california/values/ablation"
Y_MIN, Y_MAX = CFG["Y_MIN"], CFG["Y_MAX"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    vdir = f"{ROOT}/{a.variant}"
    y_test = np.load(f"{STORED}/y_test.npy").squeeze()
    files = sorted(glob.glob(f"{vdir}/bs*_k*.npz"))
    cells = {}
    for f in files:
        m = re.search(r"bs(\d+)_k(\d+)\.npz", f)
        cells[(int(m.group(1)), int(m.group(2)))] = np.load(f)
    manifest = {}
    for bs, k in sorted(cells):
        d = cells[(bs, k)]
        n_t = len([n for n in d.files if n.startswith("param_l_")])
        width = np.concatenate([(d[f"param_u_{i}"] - d[f"param_l_{i}"]).ravel() for i in range(n_t)])
        yl, yu, yn = d["y_l"], d["y_u"], d["y_n"]
        ylc, yuc = np.clip(yl, Y_MIN, Y_MAX), np.clip(yu, Y_MIN, Y_MAX)
        ref = [kk for (b, kk) in cells if b == bs and kk >= 1]
        drift = 0.0
        if ref:
            r = cells[(bs, min(ref))]
            drift = float(max(np.abs(d[f"param_n_{i}"] - r[f"param_n_{i}"]).max() for i in range(n_t)))
        manifest[f"bs{bs}_k{k}"] = dict(
            status="ok",
            seconds=float(d["seconds"]),
            epochs=int(d["epochs"]),
            steps=int(d["steps"]),
            param_width_median=float(np.median(width)),
            param_width_max=float(width.max()),
            out_width_mean_clipped=float(np.mean(yuc - ylc)),
            frac_saturated=float(np.mean((yuc - ylc) >= (Y_MAX - Y_MIN) - 1e-9)),
            nominal_drift_vs_first_k=drift,
            nominal_mse_std_units=float(np.mean((yn - y_test) ** 2)),
        )
    json.dump(manifest, open(f"{vdir}/manifest.json", "w"), indent=1)
    cfg_path = f"{vdir}/config.json"
    cfg = json.load(open(cfg_path))
    cfg["batch_sizes"] = sorted({b for (b, _) in cells})
    cfg["k_values"] = sorted({k for (_, k) in cells})
    cfg["manifest_note"] = (
        "manifest.json and batch_sizes/k_values rebuilt from the npz cells by rebuild_manifest.py"
    )
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    print(f"{a.variant}: {len(manifest)} cells, batch sizes {cfg['batch_sizes']}, k {cfg['k_values']}")
    for bs in cfg["batch_sizes"]:
        row = [(k, manifest[f"bs{bs}_k{k}"]) for k in cfg["k_values"] if f"bs{bs}_k{k}" in manifest]
        print(
            f"  bs={bs:5d} nomMSE {row[0][1]['nominal_mse_std_units']:.4f}  out w: "
            + " ".join(
                f"k{k}:{m['out_width_mean_clipped']:.3f}" for k, m in row if k in (1, 5, 10, 16, 22, 28)
            )
            + f"  max drift {max(m['nominal_drift_vs_first_k'] for _, m in row):.1e}"
        )


if __name__ == "__main__":
    main()
