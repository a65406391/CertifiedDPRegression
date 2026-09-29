# California — genuine AGT re-run, 2026-09-14

Runner: `agtr/california/train.py` (self-contained; the package is not edited).
Launched 22:45 local, 10 processes: `--variant norm` (GPU 0) and `--variant clamp` (GPU 1),
one process per batch size ∈ {250, 500, 1000, 2000, 4000}, γ = 1.0, all 17 k ∈
{0..10, 13, 16, 19, 22, 25, 28}. Logs in `logs/`. Per-cell artifacts `<variant>_gamma1/bs{bs}_k{k}.npz`
(param_l/n/u float64, unclipped y_l/y_u/y_n on the stored test set, wall time, config);
`manifest.json` per variant. Restartable: existing cells are skipped.

## What is being trained

Same as `original/run_ca_ablation.py` where possible: California Housing, StandardScaler
features, standardised targets, split seed 42, DataLoader shuffle generator seed 42,
`drop_last`, 8-64-64-1 ReLU MLP from `torch.manual_seed(42)`, lr 0.01, fixed-steps epochs
`20·bs/1000`, output clipping to [−5, 5] at evaluation. Training via
`abstract_gradient_training.privacy_certified_training` (IBP forward/backward over the
reachable parameter box, SEMin/SEMax ± kγ accumulation), float64.

Differences from the original (unavoidable):
- clipping: original = joint ℓ2 clip to C=1 of the whole per-sample gradient + Huber clamp
  of the loss gradient. `norm` variant = per-parameter-tensor ℓ2 clip to γ=1 (joint norm ≤ √6);
  `clamp` variant = per-coordinate clip to γ=1 (the paper's textual AGT).
- nominal = `param_n` of the certified run (asserted identical across k), not a separately
  trained unclipped model.
- package `norm` branch patched locally: dimension-agnostic broadcast of the per-sample clip
  factor (the package assumes 3-D tensors and crashes on 2-D bias gradients), and non-finite
  pre-clipping intervals replaced by the sound ℓ∞ hull [−γ, γ] of the ℓ2 ball (they arise
  once the parameter box is huge and `inf·0` would otherwise give NaN). Randomised containment
  self-test runs at start-up.

## Smoke results before launch (bs=250, 5 epochs = 330 steps)

| variant | k | param width (median) | output width (mean, of 10) | saturated | nominal MSE (std units) | secs |
|---|---:|---:|---:|---:|---:|---:|
| norm γ=1 | 0 | 0 | 0 | — | 0.6602 | 34 |
| norm γ=1 | 1 | 5.94 | 10.00 | 100 % | 0.6602 | 35 |
| clamp γ=1 (earlier probe) | 1 | 5.6–6.2 | 10.00 | 100 % | 0.676 | 36 |

Expectation, from the probes: at γ = 1 every cell saturates the output range (the certificate
equals global sensitivity); at γ = 0.01 boxes stay small but the model barely trains
(MSE ≈ 0.98 ≈ predict-the-mean). This run establishes the γ = 1 result with full
serialization; a γ scan is a separate decision.

## Timeline

- 22:34 smoke bs250 k0/k1 (norm) — k1 NaN (inf·0 in norm branch) → patch → re-run finite.
- 22:45 full grids launched.

## Notes during the run

- `manifest.json` is written by five concurrent workers per variant and is overwritten
  (last writer wins), so it under-counts cells. The per-cell `.npz` files are the record;
  `release.py` reads them directly and a manifest is rebuilt from them after the run.
- Early cells confirm the probe: at γ = 1 every k ≥ 1 box seen so far saturates the output
  range (norm: bs250 k1 pw_med 5.94; bs2000 k1 5.42, k2 5.53; all `sat = 1.00`).
- Nominal MSE differs between variants because the clipping changes the trajectory
  (norm bs2000: 0.672; clamp bs4000: 0.458, standardised units).

## Result — `norm_gamma1` (85/85 cells, 91 GPU-min, all workers exited with `done`)

`release.py --variant norm_gamma1` → `norm_gamma1/results_cross_radius.json`,
`figures/agt_2026-09-14/ca_cross_radius_norm_gamma1.{png,pdf}`.

- Every k ≥ 1 cell, every batch size: output boxes saturate the full [−5, 5] range for 100 %
  of test points (A(x, k=1) = 10 in standardised units; parameter widths 5.4–6.2 against the
  ceiling 2·T·lr·γ ≈ 6.4–6.6).  The certificate equals global sensitivity, so the batch-size
  ordering of the published figure disappears entirely (all five curves coincide).
- Smooth sensitivity in real units: 11.35 → 6.1 (pure, ε 0.1 → 10) and 11.49 → 7.7 (approx)
  against GS = 11.54; the only reduction is the e^{−βk} discount on a flat staircase.
- Pure DP (90 % |error| quantile): AGT-R 371 vs GS-Laplace 26.6 at ε = 1 — worse, as Thm 4.3
  requires when SS/GS ≈ 1.  Approx DP (MSE): 983 vs 3120 at ε = 1 — the 3× is the
  Laplace-vs-Gaussian mechanism difference at equal sensitivity, not the certificate.
- Nominal MSE (real units) 0.874–0.890 vs the published nominal 0.689 (which was unclipped SGD).
- Published avg S (closed form): 4.59 / 3.74 / 2.38 / 1.06 / 0.48 for BS 250…4000.  None of
  these is reproduced by a sound bound at these hyperparameters.

Conclusion: at the paper's California setting (γ = 1, lr 0.01, 320–330 steps, 8-64-64-1 ReLU),
genuine AGT certificates are vacuous; the published California panel is a property of the
closed-form bound, not of the trained model.  `clamp_gamma1` was still running when this was written (norm analysed 23:03).

## Result — `clamp_gamma1` (85/85 cells, 176 GPU-min, all workers exited with `done`)

`release.py --variant clamp_gamma1` → `clamp_gamma1/results_cross_radius.json`,
`figures/agt_2026-09-14/ca_cross_radius_clamp_gamma1.{png,pdf}`.

Identical to `norm`: 80/80 k ≥ 1 cells saturate the output range for 100 % of test points
(min median parameter width 5.38 vs ceiling ≈ 6.5); all five batch-size curves coincide;
SS ≈ GS (11.35 → 6.4 pure, 11.49 → 7.7 approx, vs GS 11.54). Nominal MSE (real units) 0.608–0.610,
closer to the published unclipped nominal (0.689) than the `norm` variant (0.87–0.89).

## Bottom line (both variants)

At the paper's California hyperparameters the genuine AGT certificate is vacuous for every
k ≥ 1, batch size and clipping variant. The published batch-size ordering (avg S 4.59 → 0.48)
exists only under the closed-form ±k·2C/b bound, which is not sound for the ReLU MLP.
Redesign options: (γ, lr, T, width) scan for a regime where the IBP box stays well below the
output range while the nominal still fits (probe: γ = 0.01 keeps the box at ~10 % of its ceiling
but the model does not train in 330 steps); or a convex head, where the closed form is sound
(Wu et al. 2017) and the MILP is cheap. Run ended 23:22 (clamp analysed).

## Single-hidden-layer follow-up (2026-09-15)

Request: reduce the California model to one hidden layer with tanh. Tanh was not supported by
`IntervalBoundedModel` (Linear/Conv2d/ReLU/AvgPool2d/Flatten/Dropout only). Added it to the
package — additive only, 21 lines: `torch.nn.Tanh` in `nominal_modules.ALLOWED_MODULES`; nominal
backward `dl_dy · (1 − tanh²x)`; interval backward multiplies the incoming gradient interval by
`[min(d(l),d(u)), 1 if l ≤ 0 ≤ u else max(d(l),d(u))]` with `d = 1 − tanh²` (even, decreasing in
|x|); the interval forward is the generic monotone branch. Tests: `tests/test_tanh_bounds.py`
(20 randomised containment checks of forward outputs and per-sample parameter gradients
against concrete networks sampled inside the box; autograd check; derivative-interval check).
Existing bounded-SGD / MSE-loss tests unchanged (105 passed).

Probes (not saved), k = 1, lr 0.01, fixed-steps epochs, clamp:

| act | H | γ | bs | param width / ceiling | output width (of 10) | saturated | nominal MSE (std) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReLU | 64 | 1 | 250 | 0.81 | 10.00 | 100 % | 0.452 |
| ReLU | 64 | 1 | 4000 | 0.69 | 10.00 | 100 % | 0.454 |
| ReLU | 64 | 0.1 | 4000 | 0.01 | 0.327 | 0 % | 0.739 |
| ReLU | 16 | 0.1 | 4000 | 0.00 | 0.013 | 0 % | 0.914 |
| tanh | 64 | 1 | 250 | 0.90 | 10.00 | 100 % | 0.423 |
| tanh | 64 | 1 | 4000 | 0.85 | 10.00 | 100 % | 0.424 |
| tanh | 64 | 0.1 | 4000 | 0.42 | 10.00 | 100 % | 0.541 |
| tanh | 64 | 0.1 | 250 | 0.65 | 10.00 | 100 % | 0.538 |
| tanh | 16 | 0.1 | 4000 | 0.02 | 0.209 | 0 % | 0.670 |
| tanh (norm) | 64 | 1 | 4000 | 0.82 | 10.00 | 100 % | 0.502 |

Reading: depth reduction alone does not rescue γ = 1. At H = 64 tanh is *looser* than ReLU
(every unit keeps a positive derivative interval, so no feedback is zeroed, and tanh trains
harder → larger gradients); at small width tanh gives the best utility/width cell found
(H = 16, γ = 0.1: output width 0.21 of 10 at MSE 0.67 vs ReLU 0.33 / 0.74).
→ `scan_tanh_settings.py` launched: act ∈ {tanh, relu} × H ∈ {8,16,32} × γ ∈ {0.03,0.1,0.3},
k = 1, bs ∈ {4000 (GPU 0), 250 (GPU 1)}; results `scan_one_layer/scan_bs{4000,250}.json`.

## Single-hidden-layer ReLU k-grid (2026-09-15) — `relu_h64_clamp_gamma0.1`

Requested: full k-grid, ReLU only. Configuration chosen from the probes: 8-64-1 ReLU, clamp
γ = 0.1 (the only ReLU cell with direct evidence of a non-vacuous certificate: output width
0.33 of 10 at MSE 0.74, BS 4000, k = 1). Everything else as before (lr 0.01, fixed-steps
epochs, k ∈ {0..10,13,…,28}, five batch sizes, float64). `train.py` gained
`--hidden/--activation/--tag` (defaults reproduce the paper model, so the earlier directories
are unchanged). Five workers, BS 250/1000/4000 on GPU 0, BS 500/2000 on GPU 1, while the
one-layer scan is still running. Data: California Housing, 20,640 records → 16,512 train /
4,128 test (80/20, seed 42).
- Added BS = 16512 (full batch; 330 epochs × 1 step = T 330) as a sixth worker on GPU 1 at the
  user's request — the tightest cell, substitution term kγ/16512. `release.py` now discovers
  batch sizes from the serialized cells.
- k = 0 is redundant (the nominal is `param_n` of any k ≥ 1 cell, asserted identical across k):
  the runner's default grid now excludes it and `release.py` takes the nominal from the
  smallest k ≥ 1 cell. The k = 0 cells already produced (incl. the 337 s full-batch one) are kept
  but unnecessary.

## One-layer scan — result (k = 1, clamp, lr 0.01, fixed-steps; `scan_one_layer/scan_bs{4000,250}.json`)

Output width (of 10) / nominal MSE (std units; unclipped ≈ 0.45), bs = 4000:

| | γ=0.03 | γ=0.1 | γ=0.3 |
|---|---|---|---|
| ReLU H=8 | 0.001 / 0.96 | 0.013 / 0.82 | 3.56 / 0.63 |
| ReLU H=16 | 0.001 / 1.08 | 0.013 / 0.91 | 2.87 / 0.70 |
| ReLU H=32 | 0.003 / 1.07 | 0.043 / 0.88 | sat / 0.62 |
| ReLU H=64 (probe) | — | **0.33 / 0.74** | — |
| tanh H=8 | 0.003 / 0.88 | 0.085 / 0.72 | sat / 0.57 |
| tanh H=16 | 0.003 / 0.94 | **0.21 / 0.67** | sat / 0.51 |
| tanh H=32 | 0.009 / 0.95 | 4.00 / 0.67 | sat / 0.48 |

At bs = 250 every cell is 5–20× wider (e.g. ReLU H=64 γ=0.1 saturates; tanh H=16 γ=0.1: 5.06).
Reading: γ sets the trade-off almost entirely — γ ≤ 0.03 certifies but does not learn,
γ = 0.3 learns but saturates; at γ = 0.1 the frontier is tanh H=16 (0.21 / 0.67) then ReLU
H=64 (0.33 / 0.74). The running ReLU grid (H=64, γ=0.1) is therefore the best ReLU
configuration found; tanh H=16 would be the natural second variant if wanted.

## ReLU 8-64-1, clamp γ = 0.1 — k-grid result (2026-09-15 00:35; `relu_h64_clamp_gamma0.1/`)

102 cells (six batch sizes × k ∈ {0..10,13,16,19,22,25,28}; k = 0 cells are from before the
default was changed and are ignored by the analysis). Manifest rebuilt from the npz cells with
`rebuild_manifest.py` (parallel workers had overwritten each other's entries; `config.json`
now lists all six batch sizes). Nominal identical across k in every battery (max drift 0.0).
Nominal MSE 0.73–0.74 (std units) = 0.938 real units; noise-free 90 % |error| quantile
1.564 (real units). GS = output range 10·y_std = 11.54.

Mean clipped output width (of 10) at k = 1 / 10 / 28: BS 250 8.80/10/10, 500 5.51/10/10,
1000 2.17/9.97/10, 2000 0.81/8.96/10.0, 4000 0.33/5.73/9.73, **16512 0.085/1.40/4.74** — width
at k = 1 ∝ 1/b and roughly linear in k; full batch is the only battery not saturated at k = 28.

First ε at which the arm beats the sound GS baseline (Laplace-on-GS for pure DP, Gaussian-on-GS
for approx; 90 % |error| quantile for pure, MSE for approx):
- Def 3.7 Cauchy(6SS/ε), β = ε/6: BS 250: never, BS 500: never, BS 1000: never, BS 2000: ε=10, BS 4000: ε=5, BS 16512: ε=1
- cor:TighterCauchy, β = ε/6: BS 250: never, BS 500: ε=10, BS 1000: ε=2, BS 2000: ε=1, BS 4000: ε=0.5, BS 16512: ε=0.5
- cor:TighterCauchy, β = ε/2: BS 250: never, BS 500: never, BS 1000: ε=2, BS 2000: ε=1, BS 4000: ε=0.5, BS 16512: ε=0.2
- Approx Laplace(2SS/ε), β = ε/(2 ln(2/δ)): BS 250: ε=0.1, BS 500: ε=0.1, BS 1000: ε=0.1, BS 2000: ε=0.1, BS 4000: ε=0.1, BS 16512: ε=0.1

Pure DP, Def 3.7 — 90 % |error| quantile (real units):

| | ε=0.1 | ε=0.2 | ε=0.5 | ε=1 | ε=2 | ε=5 | ε=10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BS=250 | 4.26e+03 | 2.08e+03 | 773 | 354 | 148 | 40.5 | 20.1 |
| BS=500 | 4.17e+03 | 2.00e+03 | 708 | 301 | 115 | 27.9 | 12.5 |
| BS=1000 | 3.98e+03 | 1.82e+03 | 564 | 202 | 64 | 13.2 | 5.11 |
| BS=2000 | 3.65e+03 | 1.55e+03 | 397 | 119 | 33 | 5.94 | 2.65 |
| BS=4000 | 3.12e+03 | 1.15e+03 | 249 | 61.4 | 14.8 | 2.89 | 1.99 |
| BS=16512 | 2.69e+03 | 832 | 83 | 14.2 | 3.22 | 1.85 | 1.68 |
| GS baseline | 266 | 133 | 53.2 | 26.6 | 13.4 | 5.52 | 3.09 |

Pure DP, tighter Cauchy β = ε/2 — 90 % |error| quantile:

| | ε=0.1 | ε=0.2 | ε=0.5 | ε=1 | ε=2 | ε=5 | ε=10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BS=250 | 1.35e+03 | 634 | 214 | 84.3 | 33.3 | 13.4 | 6.83 |
| BS=500 | 1.28e+03 | 573 | 173 | 64.2 | 20.8 | 8.38 | 4.4 |
| BS=1000 | 1.11e+03 | 437 | 103 | 31 | 9.37 | 3.64 | 2.55 |
| BS=2000 | 883 | 290 | 56.5 | 15.4 | 4.33 | 2.28 | 1.92 |
| BS=4000 | 604 | 174 | 27 | 6.32 | 2.48 | 1.85 | 1.72 |
| BS=16512 | 343 | 50.3 | 5.18 | 2.29 | 1.77 | 1.64 | 1.61 |
| GS baseline | 266 | 133 | 53.2 | 26.6 | 13.4 | 5.52 | 3.09 |

Approx (ε, 1e-5)-DP — test MSE (real units):

| | ε=0.1 | ε=0.2 | ε=0.5 | ε=1 | ε=2 | ε=5 | ε=10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BS=250 | 1.05e+05 | 2.59e+04 | 4.00e+03 | 944 | 212 | 26.6 | 5.19 |
| BS=500 | 1.04e+05 | 2.54e+04 | 3.82e+03 | 865 | 180 | 18.8 | 3.52 |
| BS=1000 | 1.02e+05 | 2.42e+04 | 3.38e+03 | 685 | 116 | 8.39 | 1.68 |
| BS=2000 | 9.74e+04 | 2.22e+04 | 2.76e+03 | 472 | 60.1 | 3.4 | 1.13 |
| BS=4000 | 8.99e+04 | 1.90e+04 | 1.91e+03 | 248 | 24.9 | 1.56 | 0.982 |
| BS=16512 | 8.41e+04 | 1.65e+04 | 1.30e+03 | 101 | 3.6 | 0.961 | 0.939 |
| GS baseline | 3.12e+05 | 7.81e+04 | 1.25e+04 | 3.12e+03 | 781 | 126 | 32.2 |

Reading: the batch-size ordering the paper claims is now real under genuine AGT. Full batch beats
sound GS from ε = 1 with the paper's Def 3.7 mechanism, from ε = 0.5 with the tighter Cauchy at
β = ε/6 and from ε = 0.2 at β = ε/2; in approx DP every batch size beats GS at every ε and full
batch is within 3 % of the nominal MSE by ε = 5. At ε = 0.1 nothing beats GS: SS is
tail-dominated (the k > 28 tail is the output-range GS), so extending the full-batch battery to
k ≈ 40–200 is the natural next run. Figure `figures/agt_2026-09-14/ca_cross_radius_relu_h64_clamp_gamma0.1.png`.

## Dense-k fill for `relu_h64_clamp_gamma0.1` (2026-09-15 13:20)

The k grid {1..10, 13, 16, …, 28} has gaps of 3 above k = 10. Back-filling a gap with the next
computed k makes the released smooth-sensitivity bound only (3β)-smooth, not β-smooth (derivation in
`agtr/linear/values/pate_matched/RUNLOG.md`, "Why the k grid is dense"), so the numbers reported
from this grid at small ε are optimistic. Fix: run the missing cells k ∈ {11, 12, 14, 15, 17, 18,
20, 21, 23, 24, 26, 27} for all six batch sizes (same runner, same settings; six workers, GPU 0:
bs 250/1000/4000, GPU 1: bs 500/2000/16512; logs `fill_bs*.log`), rebuild the manifest, re-run the
analysis with a dense 1..28 grid. `release.py` will refuse gaps from now on.

## PATE-R vs AGT-R on California (2026-09-15, 15:55 →) — `pate_relu_h64_clamp_gamma0.1/`

`pate_matched/train.py`: the training split is cut into T ∈ {2, 4, 8, 16, 32, 64} disjoint shards
(one permutation, seed 123, consecutive chunks); every shard gets an 8-64-1 ReLU teacher trained by
full-shard clipped GD (clamp γ = 0.1, lr 0.01, 330 steps = the schedule of the full-batch arm of
`relu_h64_clamp_gamma0.1`, whose bs16512 cells are the T = 1 arm) with genuine AGT bounds for the dense grid
k = 1..28; each cell stores parameter box, output box on the test split and nominal. 3 528 cells.
Workers: 6 forward (T = 2 first) from 15:55, 6 more in reverse order (T = 64 first) from 16:50; GPU 0 is
shared with another user's job. `pate_matched/release.py` builds the four arms exactly as the
linear benchmark (PATE-R Laplace GS/T, PATE-AGT-R Cauchy on max_i SS_i/T, AGT-R full, AGT-R on shard 0
with ε_b = log(1 + T(e^ε − 1))), mean absolute error of the release clipped to the output range.
Smoke test on T = 1, 2 (16:50): nominal MAE 0.718 for both (the mean of two half-data teachers loses
nothing); at ε = 1 PATE-R 3.71 (T = 2) vs AGT-R full 1.01 vs WOR 1.03; approx MSE 782 vs 31.

**Result (20:08; 3 528 cells, 18 workers, ~4 h; no failures).** Mean absolute error of the clipped release,
real units, noise-free 0.718 at every T (0.719 at T = 64): the clipped 8-64-1 teacher's accuracy does not
depend on the amount of data between 258 and 16 512 points.

| ε | PATE-R best (T) | PATE-AGT-R best (T) | AGT-R full | WOR best (T) | approx PATE-R (T) | approx PATE-AGT-R (T) |
|---:|---|---|---:|---|---|---|
| 0.1 | 1.92 (64) | 3.50 (64) | 5.58 | 4.06 (32) | 77.3 (64) | 26.8 (64) |
| 0.2 | 1.20 (64) | 2.50 (64) | 4.44 | 3.09 (8) | 20.0 (64) | 7.30 (64) |
| 0.5 | 0.829 (64) | 1.49 (64) | 1.74 | 1.58 (2) | 3.99 (64) | 1.93 (64) |
| 1 | 0.749 (64) | 1.01 (1) | 1.01 | 1.01 (1) | 1.70 (64) | 1.17 (64) |
| 2 | 0.727 (64) | 0.805 (1) | 0.805 | 0.805 (1) | 1.13 (64) | 0.991 (64) |
| 5 | 0.720 (64) | 0.749 (1) | 0.749 | 0.749 (1) | 0.968 (64) | 0.945 (64) |
| 10 | 0.719 (64) | 0.734 (1) | 0.734 | 0.734 (1) | 0.945 (64) | 0.939 (64) |

Reading: in pure DP PATE-R with the smallest shards wins at every ε — subsample-and-aggregate is unbeatable
when the learner's accuracy is insensitive to N, and the γ = 0.1 single-layer teacher is exactly that
(its error is dominated by the clipping-induced bias, not by variance). In approx DP PATE-AGT-R beats PATE-R
at every ε and T, i.e. AGT inside PATE strictly improves PATE there, as conjectured. At matched T ≤ 4 the AGT
arms beat PATE-R in pure DP from ε = 0.5. Per-shard certificates saturate from k ≈ 5 at 258 points, so
PATE-AGT-R with small shards is GS/T plus Cauchy tails, which is why it trails Laplace-on-GS/T in pure DP.
Consequence for the paper: the matched PATE comparison needs a learner that actually needs data (or a
teacher that is not clipped into a bias floor); with the present teacher the honest statement is the
approx-DP win and the matched-T pure-DP win for T ≤ 4.

## Two probes after the PATE result (2026-09-16, 00:00–00:40)

**A. Amplified aggregate** (`pate_matched/probe_hybrid.py`, analysis only, `pate_relu_h64_clamp_gamma0.1/probe_hybrid.json`):
release the mean of a uniformly random subset of T' of the T teachers with the budget amplified by q = T'/T
(fixed-size WOR sampling of shards; for fixed S the record is in a selected shard w.p. q, otherwise the output is
unchanged). Result: for every T the best T' is T (no subsampling) at every ε, apart from two marginal cases at
ε = 0.2 with informative shards (T = 8, T' = 1: 3.15 vs 4.33; T = 16, T' = 2: 3.20 vs 3.97). Global best is
still un-amplified PATE-R at T = 64. Reason: amplification pays through the ε_b² dependence of the AGT scale,
which needs a certificate that shrinks with β; with saturated shards the amplified scale GS/(T' ε_b) ≈ GS/(T ε)
equals the un-amplified one and the Cauchy penalty stays.

**B. Tighter shard certificates** (`pate_matched/probe_tighten.py`, shard 0 of T ∈ {8, 16, 32}, steps ∈ {50, 100,
200} × lr ∈ {0.01, 0.03}, dense k ≤ 10; `probe_tighten/summary.txt`; SS/GS there is tail-limited by k ≤ 10, i.e.
floored at e^{−11β}). Findings, single-teacher nominal MSE in standardised units:
- steps × lr sets the accuracy: product 0.5–1 → MSE 0.95–1.04 (predicts the mean), 1.5 → 0.87, 3 → 0.72
  (= the main run's 3.3), 6 → 0.56 (better than the main run) — and the accuracy is the same at 516, 1 032 and
  2 064 points at every setting: this teacher is N-insensitive whatever its budget.
- at equal product, fewer larger steps give a ~2.4× tighter certificate at the same accuracy (T = 16, k = 1
  output width 2.39 at 330 × 0.01 vs 0.99 at 100 × 0.03; T = 32: 5.25 vs 2.79), because the certificate grows
  with the number of steps while the accuracy grows with steps × lr.
- product 6 (200 × 0.03) saturates the certificate at k = 1 for every shard size.
Consequence: the best available lever (100 steps at lr 0.03) shifts the PATE-AGT-R vs PATE-R crossover by
about one doubling of T; it does not reach T = 64, where it would need ~8×. Since the teacher's accuracy does
not depend on N between 258 and 16 512 points at any budget, PATE-R at the largest T cannot be beaten on this
task with this model class; the AGT advantage on California is the matched-T statement (approx DP at every
T; pure DP for T ≤ 4–8), plus the linear 40 000-point case where AGT-R wins outright.
