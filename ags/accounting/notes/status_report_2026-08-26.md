# AGS accounting overnight status report — 26 August 2026

## Executive status

Last night's accounting work produced a defensible positive result on the fully
certified BLOBS experiment. For the fixed exploratory AGS configuration, AGS
outperforms the exactly matched fixed-cyclic DP-SGD baseline from epsilon 20 onward.
It also outperforms the stronger, but algorithmically unmatched, Poisson-sampled
DP-SGD stress test from epsilon 30 onward.

The PCA-MNIST long-trajectory experiment also favours AGS, but it relies on a
conditional sensitivity argument that is not yet fully proved. More importantly,
an exact, unconditional one-step early-stopping comparison favours DP-SGD at every
tested epsilon. PCA-MNIST should therefore remain a conditional sensitivity study,
not the headline empirical result.

All 71 final calibrated privacy quantities passed the automated audit. The final
artifacts were written between 05:54 and 07:21 on 26 August. No AGS accounting
process was still running when this report was prepared.

## Headline certified BLOBS result

The strongest completed comparison uses the certified continuous native-Laplace
PLD envelope for AGS, native Gaussian PLD for matched fixed-cyclic DP-SGD, 101
release seeds, and independently resampled bootstrap intervals. The fixed AGS cell
uses beta fraction 4 and bias-to-weight epsilon allocation ratio 2.

| epsilon | AGS median | matched DP-SGD | mean difference (95% CI) | Poisson DP-SGD | mean difference (95% CI) |
|---:|---:|---:|:---|---:|:---|
| 15 | 0.8753 | 0.8763 | -0.0013 [-0.0114, +0.0087] | 0.8983 | -0.0250 [-0.0333, -0.0170] |
| 20 | 0.8967 | 0.8853 | +0.0133 [+0.0067, +0.0200] | 0.8990 | -0.0006 [-0.0058, +0.0045] |
| 25 | 0.9000 | 0.8890 | +0.0121 [+0.0068, +0.0173] | 0.8993 | +0.0033 [-0.0009, +0.0075] |
| 30 | 0.9017 | 0.8913 | +0.0101 [+0.0059, +0.0145] | 0.8993 | +0.0044 [+0.0007, +0.0082] |
| 40 | 0.9020 | 0.8950 | +0.0072 [+0.0039, +0.0106] | 0.8997 | +0.0054 [+0.0023, +0.0088] |
| 50 | 0.9020 | 0.8973 | +0.0054 [+0.0027, +0.0082] | 0.9000 | +0.0057 [+0.0027, +0.0089] |
| 100 | 0.9023 | 0.8993 | +0.0023 [+0.0006, +0.0040] | 0.9003 | +0.0055 [+0.0029, +0.0084] |

Interpretation:

- Against the exactly matched baseline, the completed descriptive intervals are
  positive from epsilon 20 through 100.
- Against Poisson DP-SGD, which benefits from sampling amplification but does not
  match AGS's fixed-cyclic optimizer, intervals are positive from epsilon 30.
- At low epsilon (1–10), AGS is substantially worse. At epsilon 15, the fixed cell
  is statistically inconclusive against the matched baseline.
- These intervals are descriptive, not confirmatory: the fixed cell was chosen
  during exploratory development rather than preregistered.

## Robustness across allocation rules

All nine retained beta-fraction-4 allocation rules were reported, rather than
selecting the best test result. The number with a positive unadjusted interval was:

| epsilon | positive vs matched DP-SGD | positive vs Poisson DP-SGD |
|---:|---:|---:|
| 15 | 1 of 9 | 0 of 9 |
| 20 | 2 of 9 | 0 of 9 |
| 25 | 5 of 9 | 0 of 9 |
| 30 | 5 of 9 | 2 of 9 |

The public-RMS-power-0.25 rule is a promising future preregistration target at
epsilon 15 (median 0.8857 versus 0.8763; mean difference +0.0100, 95% interval
[+0.0005,+0.0197]). This is post-hoc evidence across nine inspected rules and is
not multiplicity-adjusted, so it is not presently a confirmatory claim.

## PCA-MNIST status

The native-accountant 40,000-step PCA experiment completed all nine privacy targets,
six public tail windows, and 41 seeds. For the final release, AGS medians were 0.755
versus 0.685 at epsilon 10 and 0.799 versus 0.777 at epsilon 20. By epsilon 100 the
gap narrowed to 0.819 versus 0.817.

There are two decisive limitations:

1. The long-trajectory closed-form sensitivity assumes that the elementwise-clipped
   update is non-expansive. The repository checks the unclipped softmax condition but
   does not yet contain a complete proof for the clipped update.
2. The exact unconditional one-step experiment favours DP-SGD throughout epsilon
   1–20. At epsilon 10 the medians are AGS 0.781 versus DP-SGD 0.800; at epsilon 20
   they are 0.809 versus 0.814, with the interval still favouring DP-SGD.

Consequently, the apparent PCA advantage depends on fixing the long training
trajectory and is not robust to public early stopping. The fully MILP-certified BLOBS
comparison is the defensible positive result.

## Accounting and implementation corrections completed

- Fixed-cyclic DP-SGD now charges one record-touching release per epoch (20 total),
  rather than all 200 optimizer updates. This uses adaptive parallel composition on
  the public fixed partition; it is not sampling amplification.
- BLOBS cross-radius certificates now retain sparse adjacent pairs after missing
  radii, use the correct final dataset diameter N=200, and intersect soundly with the
  public output range and larger-radius boxes.
- Exact targeted T9 and T10 MILPs completed all 360 solves each; exact T7 also
  completed and was incorporated in the follow-up comparison.
- AGS uses analytic/interval-certified RDP at low epsilon and a certified continuous
  native-PLD envelope where it is stronger. The envelope includes a finer audit.
- Native Gaussian accounting is used for the matched DP-SGD reference, and a separate
  Poisson-sampled reference is clearly labelled as an unmatched stress test.
- Privacy calibration keeps a safety margin. All 71 audited calibrated quantities
  met their targets; the largest Gaussian delta overshoot was approximately 2.94e-19,
  consistent with floating-point evaluation.

## Results that must not be used

- The original repeated-release F=1,…,10 online screen used an incorrect omitted-
  radius tail for F>1 and is superseded.
- `logistic_ags_tensor_l1_seed0.json` and `logistic_ags_joint_l1_seed0.json` used an
  invalid multidimensional smooth-scale condition.
- `compact_cnn_ags_clip_screen_b2000.json` was corrupted by a float32 inverse-CDF
  endpoint issue; the `_v2` artifact supersedes it.
- Public-bias diagnostics in `pca_early_stop_strong_overnight.json` are invalid;
  `pca_early_stop_strong_corrected_overnight.json` supersedes them. Raw releases in
  the earlier artifact remain valid.
- The unequal-scale endpoint native-PLD diagnostic is not certified for continuous
  interior scale ratios. Its improved numbers must not be presented as privacy
  claims. The certified continuous envelope is the usable replacement.

## Recommended next actions

1. Treat certified BLOBS epsilon 20+ versus matched fixed-cyclic DP-SGD as the primary
   completed result, with the exploratory/descriptive qualification stated visibly.
2. Use epsilon 30+ comparisons against Poisson DP-SGD only as a robustness stress
   test, since the optimizer and sampling scheme are not exactly matched.
3. Preregister a small follow-up comparison, ideally including the fixed ratio-2 cell
   and the public-RMS-power-0.25 rule, with fresh seeds and multiplicity handling.
4. Keep PCA-MNIST as supporting conditional analysis unless the clipped-update
   non-expansiveness proof is completed; report the one-step reversal alongside it.
5. Commit or otherwise preserve the current uncommitted accounting code and artifacts
   after review. The `AGS_Accounting` tree currently contains substantial modified and
   untracked overnight work.

## Primary source artifacts

- `artifacts/blobs_native_pld_envelope_comparison_overnight.md`
- `artifacts/blobs_native_pld_envelope_allocation_summary_overnight.md`
- `artifacts/overnight_strong_summary.md`
- `artifacts/overnight_matched_summary.md`
- `notes/overnight_2026-08-26.md`

