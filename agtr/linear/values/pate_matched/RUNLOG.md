# Matched PATE-R vs AGT-R on the linear task — run log

Scripts: `agtr/linear/pate/train_milp_unsound.py` (certified teacher training, MILP) and
`agtr/linear/pate/release.py` (release, arms, figures). Cells live in
`values/pate_matched/<tag>/T<T>/shard<i>_k<k>.npz`; Gurobi logs next to them (`*.log` is git-ignored,
the parsed solve statuses are inside every npz and in `manifest.json`).

## Question (2026-09-15)

Benchmark PATE-R (mean of T teachers, Laplace on GS/T), find the best T, then run AGT-R inside the
shards (PATE-AGT-R). Note the two ways the 1/T enters: PATE's mean aggregator divides the *whole*
(smooth) sensitivity, tail included, by T; the amplification lemma (Lemma `agtr_wor_amplification`,
Balle et al. Thm 9) applies to a single without-replacement subsample and gives a larger base budget
eps_b = log(1 + T (e^eps − 1)) instead. Both are arms here.

## Design

- Data: y = 2x + 1 + N(0, 0.1²), x ~ U[−2, 2], y clipped to [−6, 6] (`agtr.linear.config.PATE`
  generative model; N = 4000 train instead of 40 000, 2000 test). Output range GS = 12 (predictions
  clipped to [−6, 6]).
- Teacher / learner (identical in every arm): y = w x + b, init (0, 0), full-batch GD, 20 steps,
  lr 0.3, per-coordinate clip γ = 1 (clipping ceiling 20·0.3·1 = 6 per coordinate → output width
  > 12, so GS is the output range).
- Partition: one secret permutation (seed 43), consecutive chunks; T ∈ {1, 2, 4, 8, 16, 32, 64, 128}
  (shards of 4000 … 31 points). Shard 0 of the T-partition is a uniform WOR subsample of size N/T,
  so it doubles as the amplified single-subsample arm.
- Certification: `abstract_gradient_training_opt.certified_training`, MILP with concretisation
  after every step (F = 1), 4 solves per step (min/max of w and b), Threads 1, TimeLimit 1800 s,
  every step's box serialised.
- k grid: dense 1..k_max(m), k_max(m) = min(64, m // 8); tail k > k_max = GS.

## Probe results that fixed the design (2026-09-15, 12:50–13:10)

| m | steps | F | k | MILP time | MILP width (w, b) | IBP width | ceiling |
|---:|---:|---:|---:|---:|---|---|---:|
| 200 | 5 | 1 | 1 | 3 s | 0.026, 0.027 | 0.034, 0.043 | 3 |
| 200 | 5 | 1 | 5 | 4 s | 0.129, 0.133 | 0.168, 0.213 | 3 |
| 1000 | 5 | 1 | 5 | 14 s | 0.026, 0.027 | 0.035, 0.043 | 3 |
| 4000 | 5 | 1 | 5 | 68 s | 0.0065, 0.0067 | 0.0086, 0.0106 | 3 |
| 50 | 20 | 1 | 5 | 19 s | 0.168, 0.180 | 8.63, 9.23 | 12 |
| 1000 | 20 | 1 | 5 | 83 s | 0.0071, 0.0079 | 5.71, 6.25 | 12 |
| 200 / 1000 | 5 | 5 | 5 | > 10 min (killed) | — | — | 3 |

IBP is vacuous after 20 steps (interval arithmetic cannot see the contraction of GD on a convex
loss, widths grow geometrically), per-step MILP tracks it (box ∝ k/m); 5-step windows are far slower
for no gain. Nominal after 20 steps: (2.003, 1.003) at m = 1000, (2.002, 0.990) at m = 50.

Smoke test (N = 400, T ∈ {1, 4}, k ∈ {1, 5, 50}): k = 50 on shards of 100–400 points produced
single solves of 110–600 s (one hit the 600 s limit) and boxes at ~8 of the 12 ceiling — hence the
cap k ≤ m/8. Smoke outputs were deleted.

## Why the k grid is dense (correction to the sparse-grid back-filling used so far)

Nissim et al. need the released scale S(D) to satisfy (i) S ≥ LS and (ii) S(D) ≤ e^β S(D') for
neighbours. (ii) for S(D) = max_k e^{−βk} Ā(D,k) follows from the monotonicity of the certificate
procedure, Ā(D,k) ≤ Ā(D',k+1) (exact reachable sets, IBP and per-step exact MILP all have it; the
paper's proof of `thm:ss_agtr_ub` uses exactly this). Back-filling a gap k ∉ K with the next grid
point k* = min{k' ∈ K : k' ≥ k} breaks it: for k in a gap the term e^{−βk} Ā(D,k*) must be matched
by a term of S(D'), and the first one available is e^{−β(k*+1)} Ā(D', k**) with k** ≥ k*+1, so
S(D) ≤ e^{β(k*−k+1)} S(D') — only (β·Δ)-smooth with Δ the gap size. Example with K = {1, 4}: if the
4-box of D is at the ceiling while the 1- and 4-boxes of D' are tight, S(D)/S(D') → e^{3β}. The
claim in §4.1 that arbitrary K "is β-smooth due to the triangle inequality" is therefore not right as
stated; consecutive K up to k_max (GS beyond) is, and so is discounting a gap by its first index
(β·(k_prev+1) instead of β·k, which costs e^{β(Δ−1)} on the SS and is the cheap fix for existing
sparse runs). Consequences: the linear recompute (K = 0..20 dense) is unaffected; the California
grids (K = 1..10, 13, 16, …, 28; Δ = 3) need the gap-discount or the missing k cells — the missing
cells are being run (2026-09-15) so the CA release becomes dense as well.

The same reasoning is why a solve that stops at its time limit is a problem: the best bound is not
a monotone function of the instance, so every solve in the release must be optimal (checked from
the Gurobi logs; counts are in every npz and in `manifest.json`).

## Runs

- `n4000_e20_lr0.3_g1` — launched 2026-09-15 13:17, 100 workers, ≈ 2 770 cells.

## 2026-09-15 14:10–15:30 — the MILP path is unsound; replaced by a closed-form exact certifier

**Observation.** In the MILP cells the steady-state box width at k = 1 is exactly 4·(lr/m)·γ, i.e. the
substitution term plus the top-1/bottom-1 term, with *no* contribution from the incoming box (which should
add ≈ 0.19 × its width through the contraction of the affine GD map). A closed-form computation of the
exact one-step image from the concretised box (below) is consistently *wider* than the MILP box while
brute force confirms the closed form is exact where no clamp straddles the box (max |diff| 2e-16, k = 1..3).

**Attack.** T = 2, shard 0 (2 000 points), k = 1: replacing one of 300 random training points by
(x', y') ∈ {−2, −1, 0, 1, 2} × {±10⁶, ±6} and retraining with the same clipped GD gives final parameters
outside the MILP's k = 1 box in three of four directions (w_max 2.003198 > 2.003167, w_min 2.002513 <
2.002567, b_min 0.999958 < 1.000058); every attack lies inside the closed-form box. The
optimisation-based certifier (`abstract_gradient_training_opt`, clipping path `AGT_OPT_CLIP_SITE=param`,
substitution-aware clipping with global indicators) therefore under-approximates the reachable set; the
cause has not been located yet (`add_substitution_aware_gradient_clipping` / per-sample gradient
formulation are the suspects). The same attack against the *published* linear ablation boxes (F = 1…16
and IBP, k = 1) is in `scratchpad/pate/attack_ablation.py`; result recorded below when finished.
The 175 MILP cells of `n4000_e20_lr0.3_g1/` are kept as evidence only; the run was stopped (it was also
projected at > 15 h with 35 % of cells hitting the 1 800 s solve limit).

**Closed-form exact certifier** (`exact_linear_certifier.py`). For coordinate q of θ = (w, b):
max q' = max_{θ∈B} [θ_q − (lr/m) Σ_j s_j(θ) + (lr/m) TopK_k(s(θ))] + (lr/m) k γ (min symmetric with
BottomK), s_j = clamp(g_j(θ)_q). Over a box each s_j is affine (clamp inactive), constant (saturated) or
straddling; straddling clamps get affine under/over-estimators (chord on the concave/convex side,
constant ±γ on the other) — the *only* over-approximation. The max-objective is then convex and the
min-objective concave in θ, so both are attained at the 4 box vertices: O(m log m) per step, exact
otherwise. Validation: brute force (grid over θ × all substitution sets × extreme u) on 40 random
heavily-straddling problems, 0 violations; the nominal trajectory matches the MILP's to 4e-16 (loss scale
c = 2, torch MSELoss); inclusion-monotonicity box(D,k) ⊆ box(D',k+1) tested on 800 random (box ⊆ larger
box, one substituted sample, k → k+1) pairs, 0 violations. Monotonicity is *provable* for the release
estimator set (`mode="mono"`): every per-sample estimator moves outward when the box grows (chords of a
concave/convex function, constants, the exact affine piece), and the (k+1)-instance on D' emulates the
k-instance on D by substituting the differing sample. The tighter tangent-mixture over-estimators
(`mode="affine"`) and box-relative splitting (`n_split`) are kept for diagnostics only: neither is
inclusion-monotone (a fixed public lattice would be). Clipping ceiling init ± t·lr·γ intersected at every
step. Limitation: with many straddling samples the relaxation feeds back on itself; for the 4 000-point
teacher the certificate is linear in k up to k ≈ 80 and saturates by k ≈ 100 (`affine`: 150; `n_split=4`:
> 400). For a shard of m points the saturation is at k ≈ m/40, which is what limits PATE-AGT-R at
moderate T and every arm at ε = 0.1 (see below).

**Run `exact_n4000_e20_lr0.3_g1`** — T ∈ {1, …, 1024}, dense k = 1..min(1024, m), 2 047 teachers, 2 min on
40 cores. Results (`results.json`, `analysis.log`; pure DP = tighter Cauchy β = ε/2, 90 % |error|
quantile, noise-free floor 0.164; approx = Laplace(2SS/ε) vs Gaussian on GS, MSE, nominal 0.0100):

| ε | PATE-R best (T) | PATE-AGT-R best (T) | AGT-R full | AGT-R WOR best (T) | approx PATE-R best (T) | approx PATE-AGT-R best (T) |
|---:|---|---|---:|---|---|---|
| 0.1 | 0.381 (1024) | 1.43 (1024) | 6.19 | 3.94 (4) | 0.352 (1024) | 0.138 (1024) |
| 0.2 | 0.305 (1024) | 0.255 (1) | 0.255 | 0.201 (4) | 0.109 (1024) | 0.056 (1024) |
| 0.5 | 0.203 (512) | 0.178 (2) | 0.178 | 0.175 (4) | 0.041 (1024) | 0.028 (512) |
| 1 | 0.180 (512) | 0.168 (2) | 0.168 | 0.168 (1) | 0.024 (512) | 0.015 (512) |
| 2 | 0.173 (512) | 0.165 (2) | 0.165 | 0.165 (1) | 0.014 (512) | 0.0100 (1) |
| 5–10 | 0.167 (256) | 0.165 (8) | 0.165 | 0.165 (1) | 0.011 (512) | 0.0100 (2) |

Reading.
- At matched T, PATE-AGT-R beats PATE-R by 1–3 orders of magnitude wherever the per-shard certificate is
  non-vacuous (T ≤ 4 at ε = 0.2, T ≤ 2 at ε = 0.1, all T ≤ 64 at ε ≥ 1) and in approx DP at every T.
- The best PATE-R uses T = 512–1024 (4–8 points per teacher): this 1-D task is so stable that the mean of
  1 024 four-point fits still has MSE 0.028, so GS/T with T = N/4 is unbeatable at ε = 0.1. That is a
  property of the toy, not of the mechanisms; a harder task (d > 1, non-convex) sets the best T where
  teachers still learn, and that is the regime where the table above says AGT-R wins.
- Amplification helps AGT-R exactly as the user expected, on the *single-subsample* arm: with β_b = ε_b/2
  the noise scale is ≈ 4c/(N e ε_b²)·T with ε_b ≈ Tε for Tε ≲ 1, i.e. a factor T smaller than on the full
  data; the optimum is T ≈ 2–4 at ε ≤ 0.5 (0.201 vs 0.255 at ε = 0.2), and it dies once the shard's
  certificate saturates (T ≥ 16 at ε = 0.1).
- Inside PATE the 1/T only shrinks the GS tail; the head term of SS_g = (1/T) max_i SS_i is T-invariant,
  and the per-shard saturation at k ≈ m/40 makes PATE-AGT-R *worse* at moderate T (hump in the figure).
  A tighter (lattice-split or exact-arrangement) certifier would flatten that hump; a "PATE with a random
  subset of T' teachers" hybrid would add real amplification to the aggregate (q = T'/T) — both are
  follow-ups, not done.

**Attack against the published linear ablation boxes (k = 1; 15:50).** All 80 training points × 25
replacements (x' ∈ {−2,…,2}, y' ∈ {±10⁶, ±3, 0}), retrained with the package's own clipped-SGD nominal
(batch 5, one epoch, lr 0.1, γ = 0.1; nominal reproduced to 1e-9): extremes w ∈ [0.650803, 0.658825],
b ∈ [0.887435, 0.895435]. No box is violated; the F = 16 box [0.65080, 0.65883] × [0.88743, 0.89552]
is tight to 3e-5 against the attack, F = 8 … IBP contain it with growing slack. So the published
single-window and windowed MILP boxes of the ablation withstand the attack; the unsoundness is
reproduced only in the full-batch PATE configuration (batch = shard, 20 steps, N = 4000) and its cause
in `abstract_gradient_training_opt` is still to be located (differences: batch size 2000–4000 vs 5, k
up to 64 vs 20, 20 windows of one step). Until then the MILP path should not be used for new releases;
the ablation's boxes are kept with this evidence attached (`scratchpad/pate/attack_ablation.py`, output
copied below).
    {'ibp': [0.61124, 0.73794, 0.8096, 0.96315], 'f1': [0.61725, 0.71985, 0.83205, 0.94349], 'f2': [0.6296, 0.68674, 0.86135, 0.921], 'f4': [0.64001, 0.67082, 0.87592, 0.90662], 'f8': [0.64721, 0.66282, 0.88376, 0.89911], 'f16': [0.6508, 0.65883, 0.88743, 0.89552]}
    2000 attacked trainings; extremes: {'w_max': (0.658825, (10, -2.0, -1000000.0)), 'w_min': (0.650803, (55, -2.0, 1000000.0)), 'b_max': (0.895435, (2, -2.0, 1000000.0)), 'b_min': (0.887435, (10, -2.0, -1000000.0))}
    ibp  box k=1 [0.61124 0.73794 0.8096  0.96315]  violated: none
    f1   box k=1 [0.61725 0.71985 0.83205 0.94349]  violated: none
    f2   box k=1 [0.6296  0.68674 0.86135 0.921  ]  violated: none
    f4   box k=1 [0.64001 0.67082 0.87592 0.90662]  violated: none
    f8   box k=1 [0.64721 0.66282 0.88376 0.89911]  violated: none
    f16  box k=1 [0.6508  0.65883 0.88743 0.89552]  violated: none
    [exited with code 0]

**Run `exact_n40000_e20_lr0.3_g1`** (the paper's PATE size; 5 000 test points; T ≤ 1024; 26 min on 60
cores, the 40 000-point teacher alone 1 536 s). Noise-free floor 0.164, nominal MSE 0.0098.

| ε | PATE-R best (T) | PATE-AGT-R best (T) | AGT-R full | AGT-R WOR best (T) | approx PATE-R (T) | approx PATE-AGT-R (T) |
|---:|---|---|---:|---|---|---|
| 0.1 | 0.311 (1024) | 0.195 (4) | 0.195 | 0.17 (16) | 0.333 (1024) | 0.118 (1024) |
| 0.2 | 0.211 (1024) | 0.171 (8) | 0.171 | 0.167 (16) | 0.0903 (1024) | 0.0125 (1) |
| 0.5 | 0.173 (1024) | 0.165 (8) | 0.165 | 0.165 (2) | 0.0227 (1024) | 0.0099 (2) |
| 1 | 0.166 (1024) | 0.164 (1) | 0.164 | 0.164 (2) | 0.013 (1024) | 0.00983 (4) |
| 2 | 0.164 (1024) | 0.164 (64) | 0.164 | 0.164 (2) | 0.0106 (1024) | 0.00983 (8) |
| 5 | 0.164 (1024) | 0.164 (128) | 0.164 | 0.164 (2) | 0.00996 (1024) | 0.00983 (8) |
| 10 | 0.164 (1024) | 0.164 (128) | 0.164 | 0.164 (2) | 0.00986 (1024) | 0.00983 (64) |

AGT-R on the full data beats the best-T PATE-R at every ε; the amplified single-shard arm is the best
pure-DP arm at ε ≤ 0.2 (T = 16, ε_b ≈ 1.0 at ε = 0.1); PATE-R needs 39-point teachers everywhere.
Figures `figures/pate_matched/pate_{vs_T,best_vs_eps}_exact_n40000_e20_lr0.3_g1.png`.

**Metric switch (16:30, user request).** Every pure-DP panel and table now reports the mean |error| over
test points averaged over the first 5 noise draws (`mae`, `MAE_DRAWS = 5` in all three analyses) instead of
the 90 % quantile; `q90_abs`, `med_se` and `mse` stay in every results file. Under Cauchy noise the mean
has infinite expectation, so the 5-draw average is a sample statistic dominated by the occasional large
draw — it is the readable number the user asked for, the quantile is the tail-robust one. First-win
ε under MAE: linear F = 16 vs GS ceiling from ε = 0.5 (β = ε/2; never with Def 3.7); California full
batch from ε = 0.2 (β = ε/2) / 0.5 (β = ε/6) / 5 (Def 3.7); PATE N = 40 000 at ε = 0.1: PATE-R 0.145,
AGT-R full 0.139, amplified single shard 0.089 (floor 0.079).

**Metric, final form (16:45).** `mae` is now the mean |error| of the *released* prediction clipped to the
public output domain (linear [−5, 5] = the published GS 10; California [−5, 5] standardised; PATE [−6, 6]),
averaged over all noise draws. Clipping the release is post-processing (no privacy cost), it is what a
deployed predictor does, and it makes the mean finite under Cauchy noise — the 5-draw unclipped mean
was dominated by single extreme draws and non-monotone in ε. `q90_abs`, `med_se`, `mse` unchanged.
