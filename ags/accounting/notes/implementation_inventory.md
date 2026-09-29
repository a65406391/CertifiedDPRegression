# Implementation inventory

The repository contains several older AGS implementations with incompatible
sensitivity conventions. `AGS_Ablation/mnist_pca/ss_reference.py` uses envelope
widths in one path, while `AGS_Ablation/mnist_pca/ags.py::A_of_k` uses an l1
one-sided radius about the nominal parameters. The manuscript first defines the
standard Nissim `A^(k)` as the maximum local sensitivity within a radius-k ball,
but later describes a nominal-to-reachable-set radius.

This package uses none of those quantities for privacy calibration. It uses the
corrected coordinate cross-radius certificate in `sensitivity.py`; width and
nominal-radius values are diagnostics only.

`AGS_Extensions/task.md` specifies a separate one-shot mechanism with one
persistent Poisson mask. That mechanism contradicts the current per-step brief
(fresh WOR sampling and a private release at every AGS step), so it is not used
here.

