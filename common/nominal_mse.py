"""The non-private MSE for each AGT-R experiment, from the stored artifacts.

There is no single "nominal MSE" per dataset -- there is one per trajectory, and the
trajectories differ between the certified ablation and the baseline runs. This prints
each of them with the configuration it belongs to, so a number quoted in the paper can
be traced to the run that produced it.

    python -m common.nominal_mse
"""

import glob
import json
import os

import numpy as np

LIN_VALUES = "agtr/linear/values/ablation"
CA_VALUES = "agtr/california/values/ablation"


def linear(values_dir=LIN_VALUES, label="certified ablation"):
    rows = []
    for f in sorted(glob.glob(f"{values_dir}/method_results_*.npy")):
        m = os.path.basename(f)[len("method_results_") : -len(".npy")]
        d = np.load(f, allow_pickle=True).item()
        if "mse_nom" in d:
            rows.append((m, float(d["mse_nom"])))
    if not rows:
        return
    from agtr.linear.config import ABLATION as C

    print(f"\nLINEAR -- {label}  ({values_dir})")
    print(
        f"  N={C['N_SAMPLES']}, batch={C['BATCH_SIZE']}, epochs={C['EPOCHS']}, "
        f"lr={C['LR']}, gamma={C['CLIP_GAMMA']}, dim={C['INPUT_DIM']}"
    )
    for m, v in rows:
        print(f"    {m:<5} nominal test MSE = {v:.6f}")
    vals = {v for _, v in rows}
    if len(vals) > 1:
        print(
            "  NOTE: the arms disagree on the nominal. The nominal trajectory is the "
            "same clipped\n        SGD run in every arm, so they should be identical; "
            "the odd one out was\n        produced by a different run of the script."
        )


def california(values_dir=CA_VALUES):
    y = np.load(f"{values_dir}/y_test.npy").ravel()
    y_std = float(np.load(f"{values_dir}/y_std.npy"))
    bss = sorted(
        int(os.path.basename(p)[len("nominal_preds_bs") : -len(".npy")])
        for p in glob.glob(f"{values_dir}/nominal_preds_bs*.npy")
    )
    print(f"\nCALIFORNIA -- certified ablation  ({values_dir})")
    print(
        f"  n_test={len(y)}, y_std={y_std:.4f}  (targets are standardised; "
        f"multiply MSE by y_std^2 for original units, 100k USD squared)"
    )
    print(f"  {'batch':>6} {'MSE (standardised)':>20} {'MSE (original units)':>22}")
    out = {}
    for bs in bss:
        p = np.load(f"{values_dir}/nominal_preds_bs{bs}.npy").ravel()
        mse = float(np.mean((p - y) ** 2))
        print(f"  {bs:>6} {mse:>20.6f} {mse * y_std**2:>22.6f}")
        out[bs] = mse
    best = min(out, key=out.get)
    print(
        f"  best batch size {best}: {out[best]:.6f} standardised, "
        f"{out[best] * y_std**2:.6f} in original units"
    )


def shard_noiseless():
    """The shard ensembles' own no-noise MSE, for reference -- a different trajectory."""
    for name, path in (
        ("linear", "agtr/linear/values/baselines/shard_sweep.json"),
        ("california", "agtr/california/values/baselines/shard_sweep.json"),
    ):
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        nm = d["noiseless_mse"]
        best = str(d["best_fixed_S"])
        print(f"\n{name.upper()} -- shard ensemble with the noise switched off ({path})")
        print("  " + "  ".join(f"S={k}:{float(v):.4f}" for k, v in nm.items()))
        print(
            f"  at the selected S={best}: {float(nm[best]):.6f}  "
            f"(trained non-privately, so this is an upper reference, not the "
            f"certified nominal)"
        )


if __name__ == "__main__":
    linear()
    if os.path.isdir(LIN_VALUES + "_k20"):
        linear(LIN_VALUES + "_k20", "certified ablation, k grid extended to 20")
    california()
    shard_noiseless()
