# FINISHED PROBE -- kept for provenance; its conclusion is written up in the run log next to
# the values it produced. Not part of any current figure.
"""Single-substitution attack against the published linear ablation boxes at k = 1.

Retrains the nominal model on every dataset one substitution away from the real one, using the
package's own clipped trainer, and checks whether any result escapes the certified box. The
published boxes withstand it; the same attack against the full-batch PATE MILP configuration does
not, which is what condemned that path. See agtr/linear/values/pate_matched/RUNLOG.md.

    PYTHONPATH=. python -m agtr.linear.probes.attack_ablation

Roughly 2000 trainings, a few minutes. It used to run on import, with no __main__ guard, so merely
importing the module launched the whole attack.
"""

import json
import logging

import numpy as np

from agtr.linear.certify import train_certified_model
from agtr.linear.config import ABLATION as CFG
from common.data import get_or_create_w_true, simulated_linear_data

logging.disable(logging.WARNING)

BOXES = "agtr/linear/values/recompute_2026-09-14/boxes_hull.json"
METHODS = ["ibp", "f1", "f2", "f4", "f8", "f16"]
REPLACEMENTS = [(xp, yp) for xp in (-2.0, -1.0, 0.0, 1.0, 2.0) for yp in (-1e6, 1e6, -3.0, 3.0, 0.0)]


def nominal(Xa, ya):
    """The certified nominal parameters for one dataset: (w, b)."""
    m = train_certified_model(Xa, ya, epochs=CFG["EPOCHS"], lr=CFG["LR"], seed=42, method="ibp", k=1)
    return np.array([float(m.param_n[0]), float(m.param_n[1])])


def main():
    n_samples = CFG["N_SAMPLES"]
    # This was guarded by `if "get_or_create_w_true" in dir()`, which is True at module scope because
    # the name is imported above, so the call always ran. Written out directly: inside a function
    # `dir()` returns locals, and the guard would have silently switched w_true to None.
    w_true = get_or_create_w_true(CFG["INPUT_DIM"], CFG["SPARSITY"], path=CFG["W_TRUE_PATH"])
    X, y = simulated_linear_data(
        n_samples=n_samples, dim=1, sparsity=1, seed=42, w_true=w_true, w_true_path=CFG["W_TRUE_PATH"]
    )

    nom = nominal(X, y)
    print("nominal", nom, "(expected 0.654802642671 0.891434521991)")
    raw = json.load(open(BOXES))
    boxes = {m: np.array(raw[m]["1"]) for m in METHODS}  # (w_lo, w_hi, b_lo, b_hi)
    print({m: b.round(5).tolist() for m, b in boxes.items()})

    ext = {
        k: (lim, None)
        for k, lim in (("w_max", -np.inf), ("w_min", np.inf), ("b_max", -np.inf), ("b_min", np.inf))
    }
    n = 0
    for j in range(n_samples):
        for xp, yp in REPLACEMENTS:
            Xa, ya = X.copy(), y.copy()
            Xa[j], ya[j] = xp, yp
            th = nominal(Xa, ya)
            n += 1
            if th[0] > ext["w_max"][0]:
                ext["w_max"] = (th[0], (j, xp, yp))
            if th[0] < ext["w_min"][0]:
                ext["w_min"] = (th[0], (j, xp, yp))
            if th[1] > ext["b_max"][0]:
                ext["b_max"] = (th[1], (j, xp, yp))
            if th[1] < ext["b_min"][0]:
                ext["b_min"] = (th[1], (j, xp, yp))
    print(f"{n} attacked trainings; extremes:", {k: (round(v[0], 6), v[1]) for k, v in ext.items()})

    for meth, b in boxes.items():
        viol = dict(
            w_max=ext["w_max"][0] > b[1] + 1e-9,
            w_min=ext["w_min"][0] < b[0] - 1e-9,
            b_max=ext["b_max"][0] > b[3] + 1e-9,
            b_min=ext["b_min"][0] < b[2] - 1e-9,
        )
        print(f"{meth:4s} box k=1 {b.round(5)}  violated: {[k for k, v in viol.items() if v] or 'none'}")


if __name__ == "__main__":
    main()
