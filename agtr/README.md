# AGT-R — private prediction for regression

Noise calibrated to a certified upper bound on the smooth sensitivity of a *prediction*, so a
non-privately trained regressor can answer queries under differential privacy.

This is the only README in this tree. The root `README.md` gives the repository layout,
`Notes/PIPELINE.md` draws the pipeline, and `Notes/REPO_NOTES.md` holds the conventions and the list
of results that must not be reused.

## Layout

Each dataset reads the same way: a library, then drivers that only parse arguments and loop.

    linear/           certify.py    train_certified_model: one cell, by interval propagation or
                                    the unrolled Gurobi problem at concretization frequency F
                      train.py      the frequency ablation driver
                      release.py    certificate, staircase, noise, metrics -> results.json
                      plot.py       the figure, from results.json only
                      pate/         the matched PATE-R comparison
                      probes/ bounds/ checkpoints/ values/ data/ figures/ original/
    california/       certify.py train.py release.py  rebuild_manifest.py
                      pate/ repro/ probes/ bounds/ checkpoints/ values/ figures/ original/
    condition_table.py  the Theorem 4.3 and 4.4 feasibility figure, analytic
    plot_pate_paper.py  the paper's AGT-R figure: both matched PATE comparisons in one row (figures/)
    legacy/           superseded drivers, kept runnable so their figures stay reproducible

`original/` is frozen: the scripts that produced the published December 2025 figures. Do not import
from it and do not edit it. `legacy/` is the same idea for drivers retired later; each file carries a
banner naming what replaced it.

## Running things

    PYTHONPATH=. python -m agtr.california.train --variant clamp --gamma 0.1    # certified cells
    PYTHONPATH=. python -m agtr.california.release --variant relu_h64_clamp_gamma0.1
    PYTHONPATH=. python -m agtr.linear.release                                  # the cross-radius release
    PYTHONPATH=. python -m agtr.linear.plot
    PYTHONPATH=. python -m agtr.condition_table --combined
    PYTHONPATH=. python -m agtr.linear.pate.release --tag exact_n40000_e20_lr0.3_g1 --plot-only
    PYTHONPATH=. python -m agtr.plot_pate_paper [--no-gs | --no-gs-linear | --excess-linear | --group Linear]

Adding `--plot-only` to a release redraws from the stored `results.json` without recomputing.

## The matched PATE comparison

`<dataset>/pate/` holds the four-arm benchmark on one secret shard partition: PATE-R (the mean of T
teachers with Laplace noise on GS/T), PATE-AGT-R (Cauchy on the largest per-shard smooth
sensitivity, divided by T), AGT-R on the full data, and AGT-R on one shard with the amplified budget
from sampling without replacement.

On the linear task the teachers are certified by `exact_linear_certifier.py`, a closed-form exact
per-step reachable box, **not** by the MILP. `train_milp_unsound.py` is the MILP producer, kept as
evidence and named for what it is: in that full-batch configuration a single-substitution attack
escapes its boxes in three directions of four, while the published linear-ablation boxes withstand
the same attack. Do not use it for new releases. The attack itself is
`linear/probes/attack_ablation.py`.

## The Figure 2 reproduction

`california/repro/` carries the strongest claim in this tree. `repro_figure2.py` redraws the
published panels from the stored arrays and `reproduce_ablation.py` retrains all 85 cells from
scratch; both land on the published numbers exactly.

| batch size | re-run average S | stored average S |
|---|---|---|
| 250 | 4.589628832264 | 4.589629 |
| 500 | 3.742602343582 | 3.742602 |
| 1000 | 2.381459269329 | 2.381459 |
| 2000 | 1.062799045496 | 1.062799 |
| 4000 | 0.483113914738 | 0.483114 |

It reproduces them under the **radius** convention, which is what the published run used. The live
release path uses the paper's **cross-radius** certificate from `common/certificate.py` instead, and
the two are not the same number: the envelope width inflates sensitivity by 1.4 to 1.8 times over
the radius. `common/sensitivity.py` still carries the width routines for the legacy plots, which is
why the retired `legacy/california/run_ablation.py` cannot reproduce Figure 2 and `repro/` can.

## What the stored values mean

`values/<tag>/` holds one `.npz` per cell with the parameter box, the output box on the test split
and the nominal, plus `config.json` and `manifest.json`. A cell absent from the manifest was never
completed. `release.py` refuses a radius grid with gaps unless `--allow-gaps` is passed, because
back-filling a gap makes the released bound only `(beta * gap)`-smooth rather than `beta`-smooth;
the flag exists for saturated batteries, where the certificate is constant anyway.

The California variants `norm_gamma1` and `clamp_gamma1` are the paper's two-hidden-layer model, and
under a sound certificate every cell saturates the output range: the published batch-size ordering is
a property of the closed-form bound, not of a certificate. The live configuration is
`relu_h64_clamp_gamma0.1`, one hidden layer at a smaller clipping threshold, where the ordering is
real.
