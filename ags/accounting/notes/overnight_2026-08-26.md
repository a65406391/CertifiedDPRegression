# Overnight matched AGS analysis — 2026-08-26

## Scientific objective

Find and quantify matched settings in which AGS improves on DP-SGD, while keeping
release schedule, accountant strength, and any conditional sensitivity assumptions
visible. All configured cells and all random seeds are retained; no test-accuracy
selection is permitted.

## Blobs

- Preserve the completed MILP envelopes from the historical one-final-release run.
- Complete consecutive F=10 radii through k=20.
- Add paired geometric tail checkpoints (21,22), (32,33), (64,65), (128,129), and
  saturation radius 200. Every solve is serialized and time-limit dual bounds remain
  conservative.
- Recompute corrected cross-radius certificates and fill every omitted radius with a
  sound enclosing-box or unconditional clipping-diameter fallback.
- Intersect each radius with the public output range `[-steps*lr*gamma,
  steps*lr*gamma]` and with all larger-radius boxes. Dataset-ball nesting makes every
  such suffix intersection sound. A code audit found and fixed two bookkeeping bugs:
  the sparse certificate loop previously stopped at its first missing radius, and one
  driver incorrectly treated `b*F` as the dataset diameter even though `F` is only a
  concretization frequency. The correct final-release diameter is `N=200`.
- Use one-second LP time limits as a fast, sound relaxation panel. A time-limited
  objective uses Gurobi's dual bound and the implementation intersects it with the
  existing IBP bound; the serialized LP boxes are therefore conservative even when
  they are looser than the slower MILP boxes.
- Run the existing conservative generic-PLD AGS panel and Gaussian-specific PLD
  diagnostic with 101 release seeds.
- Correct the cyclic DP-SGD composition count. A fixed record occurs in one of ten
  disjoint minibatches per epoch, so adaptive parallel composition charges one release
  per epoch and 20 releases total—not all 200 optimizer updates. This is not sampling
  amplification; it is the standard neighboring-pair/parallel-composition argument for
  a public fixed partition.
- Run a matched-RDP panel: analytic bidirectional Laplace Renyi integrals for AGS and
  full-participation replace-one Gaussian RDP for DP-SGD, followed by one common RDP
  conversion. For coordinate releases, analytic monotonicity in absolute shift reduces
  the worst case to one scale-ratio dimension; interval branch-and-bound certifies that
  continuous maximum at the RDP-conversion order. The experimental joint-l1 panel
  remains numerical and is labelled separately. Calibration leaves a two-parts-per-
  million parameter safety margin before the interval check.
- Explore public beta ratios above the black-box theorem's sufficient limit. These
  cells are valid only under the direct neighboring-law RDP calculation (which charges
  the larger data-dependent scale ratio); they are not claimed under Definition 8's
  theorem-level shortcut.
- Report the clipping-derived trajectory cap separately from the unconditional tail;
  it is not silently promoted to a certified result.
- The targeted `T_9` and `T_10` MILPs completed all 360 solves each. Their mean widths
  are 1.4446 and 1.6383 after sound intersections; the previous fast `T_9` LP width was
  about 4.60. Individually complete radii from longer multi-radius jobs are reused only
  after verifying all 20 step files and 360 solve records for that radius.
- The longer serialized MILP sweep subsequently completed exact `T_7` as well. The
  comprehensive RDP process had already loaded its boxes, so its artifact remains a
  frozen result using the earlier LP `T_7`. A separate follow-up recomputes both
  unprojected and public-cube-projected releases with the newer exact radius, keeping
  certificate tightening distinct from postprocessing.
- Report deterministic public-validation intercept recalibration for every released
  linear model, identically for AGS and DP-SGD. This is privacy-free postprocessing;
  raw accuracies remain primary and are retained alongside it.

## PCA-MNIST

- Dataset/model/training match the existing promising PCA-64 logistic experiment:
  N=2000, full batch, 40,000 steps, lr=0.05 with decay 0.01, coordinate clip 0.1.
- Run epsilon in {1,2,4,8,10,20,50,100}, delta=1e-5, with 41 fixed seeds.
- First panel preserves the current conservative generic-PLD AGS versus tight Gaussian
  PLD calculation.
- Second panel gives both mechanisms analytic RDP composition and a common conversion.
- An audit found that the first RDP attempt silently enabled max-norm feature scaling
  while the historical/generic PCA experiment did not. That unmatched attempt was
  stopped without an artifact. The corrected runner records `scale_unit_norm=false`;
  its first results reproduce the generic panel (epsilon 1: 0.230 versus 0.138,
  epsilon 2: 0.374 versus 0.198, epsilon 4: 0.572 versus 0.381).
- A separate exploratory panel evaluates fixed public tail-averaging windows. AGS
  calibrates to the corresponding weighted sensitivity certificate; DP-SGD averages
  its noisy internal iterates as postprocessing. Every window is retained.
- The PCA fit retains the documented public-preprocessing assumption.
- The closed-form final sensitivity is explicitly conditional on proving
  non-expansiveness for the elementwise per-example clipped update; the current
  unclipped-softmax smoothness check alone does not discharge that proof.
- In this specialized PCA panel the accepted closed-form certificate is constant
  across coordinates/datasets. Consequently the mechanism is mathematically a
  one-release coordinate output-perturbation bound (and can be presented as the
  specialized AGS construction), but it is not evidence that data-adaptive smooth
  sensitivity itself improved utility. The fully MILP-certified Blobs panel is the
  primary evidence for that stronger claim.
- A public early-stop audit found that the identical unscaled-PCA configuration
  reaches about 0.821 heldout accuracy after only one full-batch clipped update.
  This one-step sensitivity is exact and unconditional. Under native accountants,
  DP-SGD beats coordinate-Laplace AGS throughout epsilon 1--20: at epsilon 10 the
  raw medians are 0.800 versus 0.781, and at epsilon 20 they are 0.814 versus 0.809.
  Public affine-logit calibration (positive temperature plus class intercepts) raises
  the epsilon-20 medians to 0.828 and 0.822, preserving DP-SGD's lead. Thus the long-
  trajectory PCA AGS win is not robust to public early stopping, even though it is
  reproducible for the matched 40,000-step trajectory.
- An initial additive-bias-only postprocessing diagnostic was invalidated because it
  overwhelmed small but correctly ordered early logits. The corrected affine-logit
  routine bounds log-temperature to [-10,10]; raw release results were never affected.

## Repeated releases

The original F=1,...,10 online screen is superseded and must not be published. It
conservatively composed every release, but its purported omitted-radius GS tail for
F>1 incorrectly divided the whole block-output diameter by the batch size. The audited
runner now uses adaptive parallel composition within each fixed cyclic epoch (20
record-touching releases), direct interval-certified RDP, and either the exact
one-step clipped-mean GS (`F=1`) or the full clipping-only block diameter (`F>1`). It
also keeps clean nominals solely in the explicitly non-public certificate audit tree.

At epsilon 10, the corrected F=1 beta-fraction-1 run achieved 0.7613 on its final
release and 0.7895 with the best reported fixed tail average, below matched DP-SGD
(about 0.848). Thus adaptive parallel accounting materially repairs the run but does
not create a repeated-release win in this setting.

## Preliminary checkpoint (before the 101-seed finalizer)

With the corrected 20-release cyclic DP-SGD accounting and the complete sparse LP
tail, a fixed, individually certified coordinate cell (beta ratio 4, public
weight/bias ratio 2) gave median Blobs accuracy 0.89825 versus 0.88725 for DP-SGD at
epsilon 30 over 41 seeds; at epsilon 50 the corresponding medians were 0.90025 and
0.89325. At epsilon 20 AGS was close but slightly behind (0.8685 versus 0.8805 before
the targeted T9/T10 MILPs). The epsilon-30 coordinate RDP interval upper bound was
29.9999988 at order 1.65. These values are checkpoints, not the final held-out table.

After adding exact completed `T_5/T_6` radii and the targeted `T_9/T_10` bounds, focused
101-seed held-out checks improved materially: epsilon 20 gave AGS 0.8913 versus DP-SGD
0.8837; epsilon 30 gave 0.9017 versus 0.8910; epsilon 50 gave 0.9020 versus 0.8963.
Public intercept recalibration gave (AGS, DP-SGD) medians (0.9030, 0.8973), (0.9043,
0.9013), and (0.9047, 0.9037), respectively. These are focused exploratory cells;
the comprehensive retained-grid artifact and paired intervals remain the final gate.

## Strong-accountant extension

- A separate Blobs DP-SGD reference applies the native replace-one Gaussian PLD to
  the same 20 record-touching releases and the same public-validation/test split as
  the direct AGS panel. Grid intervals 0.001, 0.0005, 0.0002, and 0.0001 gave the
  same epsilon-20 multiplier to the displayed precision. At epsilon 20 its raw and
  public-intercept-recalibrated medians are 0.8853 and 0.8997; at epsilon 30 they
  are 0.8913 and 0.9017; at epsilon 50 they are 0.8973 and 0.9043.
- For PCA-MNIST, native coordinate-Laplace PLD composition improves the permitted
  epsilon-20 coordinate parameter from 0.13215 under the common RDP conversion to
  about 0.13888 on a preliminary 0.001 PLD grid (roughly 5% less release noise).
  The queued strong-accounting panel uses a pessimistic 0.0001 grid and audits it
  again at half that interval.
- DP-SGD's full-participation Gaussian releases admit a closed-form exact composed
  privacy-loss tradeoff. This is also queued for PCA, so the final strong panel gives
  both arms their native mechanism-specific accountant rather than deliberately
  weakening either side to a common generic accounting class.
- Statistical summaries retain paired common-seed diagnostics but use independently
  resampled bootstrap intervals for primary cross-mechanism uncertainty, avoiding a
  claim that Laplace and Gaussian pseudorandom streams form a natural paired sample.
- A separate standard Poisson-sampled DP-SGD stress test uses q=0.1, 200 steps, and
  the native replace-one subsampled-Gaussian PLD. It is not called an exactly matched
  optimizer because the AGS/MILP trajectory uses fixed cyclic batches. As expected,
  amplification makes this substantially stronger than fixed-cyclic DP-SGD: raw
  medians are 0.8950, 0.8990, 0.8993, and 0.9000 at epsilon 10, 20, 30, and 50.
  The focused AGS checkpoints therefore lose at epsilon 10/20 but remain narrowly
  above this stress test at epsilon 30/50 before uncertainty intervals are applied.
- The completed 101-seed exact-`T_7` follow-up confirms the fully certified fixed
  AGS cell against both baselines. Against native-PLD fixed-cyclic DP-SGD, unpaired
  bootstrap mean-difference intervals are positive from epsilon 20 onward (epsilon
  20: +0.0124, 95% CI [+0.0056,+0.0190]). Against the stronger but algorithmically
  unmatched Poisson stress test, the intervals become positive at epsilon 30 and
  remain positive through 100: epsilon 30 +0.0042 [+0.0004,+0.0081], epsilon 40
  +0.0053 [+0.0021,+0.0087], epsilon 50 +0.0055 [+0.0026,+0.0087], and epsilon 100
  +0.0055 [+0.0029,+0.0084]. These intervals are descriptive because the fixed beta-
  4/bias-ratio-2 cell was chosen during exploratory development, not preregistered.
- The corrected native-accountant PCA sweep completed all 9 privacy targets, 6 public
  tail windows, and 41 release seeds. Native coordinate-Laplace PLD gives AGS a large
  apparent advantage on the fixed 40,000-step trajectory (for example, epsilon 10,
  window 1: median 0.755 versus 0.685; epsilon 20: 0.799 versus 0.777). At epsilon 100
  the gap narrows to 0.819 versus 0.817. However, the one-step exact unconditional
  early-stop experiment reverses the ordering at every tested epsilon through 20.
  The long-trajectory PCA result is therefore retained as a conditional sensitivity
  study, not recommended as the headline empirical win. The fully certified Blobs
  comparison is the defensible positive result.
- A final automated privacy audit checked 71 calibrated quantities. Every Blobs AGS
  interval upper bound, fixed-cyclic Gaussian value, Poisson-amplified Gaussian value,
  PCA pessimistic PLD value, half-grid PLD audit, and Gaussian delta value met its
  target. The largest apparent Gaussian-delta excess over `1e-5` was approximately
  `2.94e-19`, attributable to floating-point evaluation.
- An all-rule robustness audit reports all nine retained beta-fraction-4 allocation
  rules rather than selecting the best test cell. At epsilon 30, five rules have a
  positive descriptive interval versus matched fixed-cyclic DP-SGD and two do so
  versus the Poisson stress test; at epsilon 40 and 50, five rules have positive
  intervals versus both. This is post-hoc robustness evidence, not a confirmatory
  multiplicity-adjusted analysis.
- A native unequal-scale Laplace PLD diagnostic found genuine accountant headroom.
  It pessimistically discretizes every fixed neighboring pair and searches all mixed
  low/high endpoint assignments over the eight weight coordinates and bias, but does
  not yet certify the continuous interior of the scale-ratio envelope. It permits an
  approximately 1.107--1.113 multiplier on epsilon and beta parameters. Diagnostic
  median accuracies at epsilon 15, 20, 25, and 30 are 0.8880, 0.8983, 0.9013, and
  0.9013. Against matched native-PLD DP-SGD, their mean-difference intervals are all
  positive; against Poisson DP-SGD the intervals become positive at epsilon 25.
  These numbers are explicitly non-certified and must not be used as privacy claims
  until an interior-ratio maximization proof or interval certificate is implemented.
  A 17-by-17 common-ratio grid at epsilon 15, 25, and 50 placed the maximum at
  an endpoint to numerical precision, and halving the PLD interval from 0.002 to
  0.001 reduced the evaluated epsilon by about 0.0045 in every checked cell. However,
  neither first-order stochastic dominance nor all-epsilon hockey-stick dominance by
  the endpoint law holds; both proposed proof shortcuts have explicit numerical
  counterexamples. For the epsilon-15 bias law, endpoint-CDF minus equal-scale-CDF is
  about 0.289 near privacy loss 1.028. For the epsilon-50 bias law at hockey-stick
  epsilon 1.8, the equal-scale divergence is about 0.8095 versus 0.6002 at the low-
  ratio endpoint. Continuous composed-PLD optimization therefore remains essential.
- A separate certified native-PLD construction avoids that unresolved optimization:
  it partitions normalized Laplace output and log-scale-ratio space, assigns each
  output-bin's exact probability to a pointwise rectangle upper bound on privacy
  loss, rounds upward, and assigns truncated tails infinite loss. The resulting
  variables dominate every continuous admissible shift/scale pair under an explicit
  coupling and therefore remain sound under coordinate composition. A finer audit
  halves both output and privacy-loss intervals and doubles ratio partitions.
  The envelope is looser than RDP at epsilon <=5, so the final selector retains RDP
  there; from epsilon 10 onward it permits factors 1.0039, 1.0242, 1.0268, 1.0312,
  1.0370, 1.0402, 1.0470, and 1.0800 through epsilon 100. The certified comparison
  remains positive versus matched native-PLD DP-SGD from epsilon 20 and versus the
  Poisson stress test from epsilon 30. At epsilon 15 it improves AGS to 0.8753 versus
  0.8763, a descriptive mean-difference interval spanning zero.
- Applying the same certified envelope to all nine retained beta-4 allocation rules
  yields one positive unadjusted interval versus matched DP-SGD at epsilon 15, two at
  epsilon 20, and five at epsilon 25/30. The epsilon-15 public-RMS-power-0.25 rule has
  median 0.8857 versus 0.8763, mean difference +0.0100, and descriptive 95% interval
  [+0.0005,+0.0197]. This is a promising preregistration target, not a confirmatory
  result: it is one of nine post-hoc inspected rules and the intervals are not adjusted
  for multiplicity. All 36 rule/epsilon privacy upper bounds pass their targets.
- Symmetric postprocessing does not create a stronger comparative claim. Public
  intercept recalibration improves both arms' absolute accuracy but benefits DP-SGD
  more: for the fixed cell at epsilon 20, medians become 0.9020 versus 0.8997 but the
  mean-difference interval spans zero. Public cube projection is essentially neutral
  above epsilon 15. For the epsilon-15 RMS-0.25 rule, the raw interval is narrowly
  positive, projection makes it span zero, and intercept recalibration reverses the
  mean ordering. Raw private-release accuracy therefore remains the primary metric.
