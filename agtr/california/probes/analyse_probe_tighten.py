# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Summarise probe_tighten: per (T, steps, lr) the nominal error of shard 0's teacher, the k = 1 output width and
SS_i / GS at beta = eps/2 with the dense k <= 10 grid and the GS tail beyond (conservative), against the
330-step main-run teacher truncated to the same k <= 10.  Also the mean over the T shards is NOT available
here (only shard 0 is trained), so the nominal column is a single teacher."""

import glob
import re
import numpy as np
from agtr.california.release import ROOT, STORED, GS
from agtr.california.pate.release import load_teacher_cells, certificate, smooth_sensitivity

EPS = [0.2, 0.5, 1.0, 2.0]


def row(files, kcap=10):
    files = {k: f for k, f in files.items() if k <= kcap}
    yn, ks, boxes = load_teacher_cells(files)
    A = certificate(yn, boxes)
    y_test = np.load(f"{STORED}/y_test.npy").ravel()
    mse = float(np.mean((yn - y_test) ** 2))
    w1 = float(np.mean(boxes[0][1] - boxes[0][0]))
    ss = [float(np.mean(smooth_sensitivity(A, e / 2.0))) / GS for e in EPS]
    return mse, w1, ss


def main():
    print(
        f"{'config':28s} {'nomMSE(std)':>11s} {'w@k1':>6s}  SS/GS at eps=" + " ".join(f"{e:>6g}" for e in EPS)
    )
    for T in (8, 16, 32):
        base = {
            int(re.search(r"_k(\d+)\.npz", f).group(1)): f
            for f in glob.glob(f"{ROOT}/pate_relu_h64_clamp_gamma0.1/T{T}/shard0_k*.npz")
        }
        mse, w1, ss = row(base)
        print(
            f"{'T%d s330 lr0.01 (main run)' % T:28s} {mse:11.4f} {w1:6.3f}  "
            + " ".join(f"{v:6.3f}" for v in ss)
        )
        for d in sorted(
            glob.glob(f"{ROOT}/probe_tighten/T{T}_s*_lr*"),
            key=lambda p: (
                int(re.search(r"_s(\d+)", p).group(1)),
                float(re.search(r"_lr([0-9.]+)", p).group(1)),
            ),
        ):
            files = {int(re.search(r"_k(\d+)\.npz", f).group(1)): f for f in glob.glob(f"{d}/shard0_k*.npz")}
            if len(files) < 10:
                print(f"{d.split('/')[-1]:28s} incomplete ({len(files)}/10)")
                continue
            mse, w1, ss = row(files)
            print(f"{d.split('/')[-1]:28s} {mse:11.4f} {w1:6.3f}  " + " ".join(f"{v:6.3f}" for v in ss))
        print()


if __name__ == "__main__":
    main()
