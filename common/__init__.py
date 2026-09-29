"""Shared experiment code for the AGT-R (agtr/) and AGS (ags/) experiments.

ags.py          AGS core: train_ags (reference release, pinned by ags/*/test_ss_reference.py),
                ags_inference_from_bounds, ags_release_cross_radius (paper's certificate convention)
baselines.py    functional mechanism, Laplace/Gaussian DP-SGD, sharding, PATE regression
accounting/     cross-radius sensitivity, DP accountants, calibration, amplification (own tests)
noise_decision.py  one-step Cauchy-vs-Laplace threshold (oracle analysis, not a DP mechanism)
plotting.py     paper style + figure primitives;  figures.py  named figure builders / CLI
sensitivity.py  V_k / staircase (legacy width convention) + cross_radius_certificate
privacy.py      mechanism scales and beta limits per the paper (+ legacy MSE simulations)
training.py     train_certified (genuine AGT), train_nominal_clipped, train_robust_ibp (closed form,
                warns: not sound for ReLU nets)
data.py models.py paths.py shard_sweep.py nominal_mse.py
"""
