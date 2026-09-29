# AGS — Abstract Gradient Sampling

Private *learning*: a certified parameter-space envelope from AGT, turned into a release through the
smooth-sensitivity staircase, compared against DP-SGD and PATE at matched settings.

This is the only README in this tree. The root `README.md` gives the repository layout,
`Notes/PIPELINE.md` draws the pipeline, and `Notes/REPO_NOTES.md` holds the conventions and the
list of results that must not be reused.

## Layout

One directory per dataset, the study named beneath it.

    agt_ext/          extensions to the Gurobi package, additive: nothing in it is edited
                        clipped_disjoint.py   disjoint clipped-substitution encoding
                        deep_per_sample.py    more than one trainable layer, via McCormick
    blobs/            the certified lineage: envelopes, release, paper figures
                        accounting/           the matched AGS vs DP-SGD accounting, with its artifacts/
                        probes/ runs/ figures/
    mnist/            the same lineage on PCA-MNIST, plus
                        ablations/            sampling frequency and hybrid schedules
                        accounting/pca/       closed-form PCA-64 accounting
                        accounting/screens/   config-driven CNN and logistic screens, configs/ beside them
                        accounting/artifacts/
                        probes/ runs/ reference/ data/ figures/
    imdb/             the contractive comparison on frozen sentence-encoder features (fig9)
                        heads.py certify.py shared.py run_contractive.py plot_contractive.py
                        runs/ figures/ data/
    adult/            AGS alone on Adult (fig10) and the sampling-frequency ablation (fig11)
    plot_paper_combined.py  the paper's AGS figure: fig8 and fig9's four panels in one row (figures/)
    sst2/             the hybrid: DP-LoRA fine-tuning of the encoder, then an AGS head (fig17)
                        shared.py config.py finetune.py run_contractive.py plot_hybrid.py
                        runs/ figures/ data/
                        shared.py config.py run_contractive.py plot_contractive.py plot_windows.py
                        runs/ figures/ data/
    accounting/       only what belongs to no dataset: the accountant audit sweep, the
                      cross-dataset overnight summaries and status plotter, paths.py, notes/

The mathematics the accounting rests on lives in `common/accounting/`; the shared certificate,
mechanisms and metrics in `common/`.

## Running things

Figures are PNG only: every plotter writes `<stem>.png` and no PDF (see `CONVENTIONS.md`).

    PYTHONPATH=. python -m ags.blobs.run_ibp_sweep         # IBP across batch size -> runs/ibp_sweep_ref.json
    PYTHONPATH=. python -m ags.blobs.run_milp_serialized   # the certified envelopes (needs Gurobi)
    PYTHONPATH=. python -m ags.blobs.plot_headline     # figures 1 and 2 -> runs/pure_headline.json
    PYTHONPATH=. python -m ags.blobs.plot_approx       # figure 3        -> runs/approx_headline_v2.json
    PYTHONPATH=. python -m ags.blobs.plot_combined     # figure 4, reads the two JSONs, never recomputes
    PYTHONPATH=. python -m ags.blobs.plot_headline --certificate cross_radius   # then plot_approx, plot_combined: fig6

    PYTHONPATH=. python -m ags.mnist.run_headline      # envelopes and every baseline, pure and approx
    PYTHONPATH=. python -m ags.mnist.plot_headline     # the two figures, from runs/headline.json only

    PYTHONPATH=. python -m common.figures ags_comparison mnist    # the ablation families
    PYTHONPATH=. pytest -q ags/blobs/test_ss_reference.py ags/mnist/test_ss_reference.py

## blobs — the certified lineage

**`ss_reference.py` is the only place the AGS release is computed.** It is a line-by-line port of
the reference implementation, and every function names the lines it implements. If the reference
changes, change both and re-run `test_ss_reference.py`; never one alone. The release path that used
to live in `ags.py` was a re-derivation, deviated in six ways and under-noised by about 71 times; it
now raises. `ags.py` keeps only `agt_bound()` and `A_of_k()`.

**Two certificates, one flag.** The reference takes the per-coordinate *width* of the k-envelope as
$A^{(k)}$. The paper's bound is the cross-radius, $\bar A_j(k) = \max(U_j^{k+1} - L_j^k,\; U_j^k -
L_j^{k+1})$, which sits between $W(k)$ and $W(k+1)$ and equals the radius of the 1-box about the
nominal at $k = 0$. `cross_radius.py` implements it on the same stored envelopes through
`common.certificate`, the code AGT-R releases with. `plot_headline`, `plot_approx` and
`plot_combined` take `--certificate cross_radius`; the default is the reference, so every stored
figure and JSON is untouched, and the cross-radius outputs carry a `_cross_radius` suffix
(`fig6_combined_cross_radius` is fig 4 under it). On the stored envelopes the switch moves fig 1 by
at most $-0.034$ (F=10, $\varepsilon = 20$) and $+0.075$ (F=2, $\varepsilon \ge 30$), fig 3 by under
$0.01$, and F=1 not at all; `Notes/REFACTOR_LOG.md` has the table. fig 2 is not drawn under it
because the batch-size sweep stored only widths.

`run_milp_serialized.py` produces the certified envelopes and `serialize.py` writes every
`(param_l, param_n, param_u)` for every radius, window and model. Those bounds are
inference-independent and survived the correction unchanged.

Settled results:

- **The k axis is exhausted.** A check-point only matters if its envelope width is below the
  one-substitution global sensitivity; measured widths at batch 20 put every unsolved radius above it.
- **AGS is invariant to pure gain factors.** Epochs, learning rate and L2 all scale the parameters
  and the sensitivity identically. Only the MILP at F >= 5 moves the trade-off.
- **Interval propagation and MILP at F <= 2 are clamped at the global sensitivity** and give
  identical curves.

## mnist — the same lineage, and the ablations

PCA-64 on all ten classes, multinomial logistic regression, **650 parameters**. That count is not
incidental: the coordinate-wise release splits the budget across every coordinate, so it enters the
noise scale directly. Batch 2000, five steps per epoch, learning rate 15, clipping 0.12; release at
radii 1 to 2000, pure and approximate.

**No tail term is needed here, and that is sound.** The staircase normally ends with a term covering
every radius past the computed grid. Here the grid reaches the batch size, and AGT caps the
per-batch substitution count there, so the certificate is constant beyond it and the discounted tail
is already one of the grid's own check-points. `release.staircase()` raises if the grid stops short,
so it cannot be violated silently.

Which losses certify, verified by running them: cross-entropy certifies, one-vs-all hinge certifies,
and Weston-Watkins multiclass hinge **has no implementation** in either package, so `select_loss()`
refuses it rather than silently substituting something else. The numpy trajectories match AGT's
nominal to 2e-14 on both certifiable arms.

**The optimized comparison** (`run_optimized.py`, `plot_optimized.py`, settings in `config.py::OPTIMIZED`)
puts an optimized AGS release against matched and optimized DP-SGD and PATE on all of MNIST: 55k
private, 5k public split into a PATE query pool and the selection set every tuned arm picks its
hyperparameters on, per $\varepsilon$; 10k test. The AGS release is one multivariate draw calibrated for
the *whole* parameter vector (`common/vector_release.py`): the $\ell_1$ cross-radius certificate over
all $P = 650$ coordinates, one smooth sensitivity, one $\varepsilon$; never noise per parameter with its
own budget. The sound rule makes the scale $(\beta/P)$-smooth (the dilation condition of a
$P$-dimensional product mechanism); the paper's Corollary 4.1 as written (`dimension_free`), the
reference's width certificate, the three allocations and the PLD-composed per-coordinate release are
all computed from the same envelopes and stored beside it, and `plot_optimized.py` draws any of them.
Envelopes are saved per configuration and radius under `runs/optimized/bounds/`; `run_optimized.sh`
runs the sweep on the GPU and the baselines on the CPU under one wall-clock deadline, then the
release and `figures/fig5_combined_optimized`.

The result of that run (September 2026, `Notes/REFACTOR_LOG.md`): the sound whole-vector release is at
chance at every $\varepsilon \le 100$ on this head, because the $\ell_1$ certificate sums $P$
coordinates and the $\beta/P$ dilation condition puts the staircase maximum near $k \approx P/\beta$;
matched DP-SGD reaches the non-private ceiling from $\varepsilon \approx 20$. The stored envelopes and
every variant are kept so the paper's choice of mechanism can be revisited from the same data.

**The contractive comparison** (`run_contractive.py`, `plot_contractive.py`, `config.py::CONTRACTIVE`
and `CONTRACTIVE_5`) is the same comparison on a trajectory that makes the certificate usable: L2
regularization with $\lambda$ above the interval propagation's own growth constant (about 0.2 on
PCA-16) makes the certified envelope contract and saturate at $2\gamma k/(b(\lambda - c))$ whatever
the trajectory length, while matched DP-SGD keeps paying the step count under composition. Same
whole-vector release, budgets to $\varepsilon = 1000$, approx DP taking the better of Laplace and the
Cauchy release, both matched approx baselines drawn (basic and exact Gaussian composition), and
Algorithm 2's windows chained from the noisy release as a check. A sweep over $(d, \lambda, \gamma,
\eta, T)$ on a 10k subset ranks configurations by their margin over matched pure DP-SGD at large
$\varepsilon$; the winners are re-certified on the full private split for `fig7_combined_contractive`
(10 classes) and `fig8_combined_contractive_5class` (digits 0–4: there is no standard 5-class
grouping in the literature, and a subset keeps every class unimodal). `run_contractive.sh` runs both
tasks under one deadline.

**The release rules.** Every contractive figure's headline AGS arm is the tightest release proven in
`Notes/AGS_RELEASE_ADMISSIBILITY.md`: independent noise per coordinate, per-coordinate smoothness
$\beta_j$ with $\sum_j \beta_j = \beta$, a fixed half of the budget for the shift of the centre and half for
the change of scale. Pure DP is Cauchy with $\beta_j = \varepsilon/(2P)$ and scale $2 a_j S_j/\varepsilon$;
approximate DP is Laplace with $\beta_j$ from the exact Gamma-tail condition of the dilation loss
(`common.vector_release.laplace_dilation_beta`), about $\varepsilon / (2\sqrt{2P\ln(2/\delta)})$. The earlier
$\beta/P$ rule with the reference constants (`_aggregate`), the corollary at $\beta = \varepsilon/(6P)$
(`_corollary_p`) and the corollary as written (`_dimension_free`, not established for $P > 1$) are
registered variants of each figure.

`ablations/` is a different lineage from the files above it: torch, built on `common.ags.train_ags`,
producing the sampling-frequency and hybrid-schedule figures. (The same study on an IMDB head over
frozen GPT-2 embeddings, `gpt2/ablations/`, was removed on 25 September 2026: superseded by `imdb/`;
it is in the history before that date.)

## imdb — the contractive comparison on a foundation-model backbone

`ags/imdb/` moves the contractive comparison to frozen features of a public sentence encoder
(`all-mpnet-base-v2`, 768-d, computed once by `shared.py` and cached under `data/`): IMDB, 2 classes,
5k of the 25k training reviews declared public (2.5k the PATE pool and warm-start set, 2.5k the
selection set), 20k private, PCA-$d$ fitted on the public reviews only, so the projection costs no
privacy. The private split is small enough to certify whole, so there is one scale and no final stage.
The binary head has $P = 2(d+1)$, the smallest of any AGS experiment here, which is what lets $d$ reach
32. Four groups share the finalist's $(\lambda, \gamma, \eta, T)$ and differ in the head and the start:
the logistic head from zero (the paper's setting); the logistic head started at, and contracted toward,
the public head $\theta_{pub}$ trained on the pool (`agt_ext/centered_l2.py`: the L2 term pulls toward
$\theta_{pub}$, the envelope is unchanged, the fixed point moves from chance to the public head); a
Linear–ReLU–Linear head of width $h$ (`heads.py`, the certifiable form of "more than the last layer",
certified by the same bound propagation); and its warm-started twin. Every arm of every group runs
through one torch trainer (`heads.train`, equal to `mnist/baselines.train` to round-off on the
logistic head), so the matched, optimized and PATE baselines exist for the hidden-layer head too.
`run_contractive.py` has the stages (`embed`, `sweep`, `baselines --group`, `release`);
`run_contractive.sh` runs them under one deadline; `plot_contractive.py` draws
`fig9_combined_contractive_imdb` (cold groups), `_warm` (warm groups, with the public head as a
second reference line) and `_long` (the logistic finalist on a $T = 3000$ trajectory: AGS's release
is $T$-invariant under contraction while the matched arm pays $T$, the same panel as fig7's sweep).
`config.py::CONTRACTIVE_N2K` (`--task imdb_n2k`, `fig9_..._n2k`) certifies the same head on 2k of the
private reviews: matched DP-SGD's per-step noise grows as $1/b$ and AGS's certificate as $1/b$ but its
smooth sensitivity is taken at the staircase maximum, so the comparison moves in AGS's favour as $b$
shrinks; that is the IMDB panel where AGS is ahead of the matched arm (ε = 20 to 100).

## adult — AGS on its own, and how often to release

`ags/adult/` runs the IMDB driver (`ags/imdb/run_contractive.py`, through the config's `loader` hook)
on Adult (Kohavi 1996; OpenML `adult` v2, 45,222 rows after dropping missing values, cached by
`shared.py` under `data/`): 10k test, 5k public (2.5k pool, 2.5k selection), 30,222 private; one-hot
plus standardisation and PCA-$d$ fitted on the public rows, features scaled into the unit ball. The
task is imbalanced (75/25), and under contraction the head's argmax is the majority class at every
$(\lambda, \gamma)$ while it ranks well (AUC 0.84–0.88), so every arm is scored at the decision
threshold chosen on the public selection set (post-processing of the released head; `scores` in the
driver, `tune_threshold` in the loader) and the AUC is stored beside it. Two finalists are pinned in
`config.py` after the 18-configuration stage-1 sweep ($d \in \{16, 24, 32\}$, $\lambda \in \{0.3, 0.6\}$,
$\gamma \in \{0.02, 0.05, 0.1\}$): `ADULT` ($d = 16$, $P = 34$) and `ADULT_D32` ($d = 32$, $P = 66$),
both $\lambda = 0.3$, $\gamma = 0.05$, $\eta = 2$, $T = 1000$. Only the logistic group is run (no
warm, hidden-layer or baseline arms: the task investigates AGS alone), so `fig10_combined_contractive_adult`
and `_d32` show the AGS release against the non-private line. On the $T = 1000$ trajectory the AGT boxes
at the largest radius $k = b$ are numerically inverted; `load_boxes` intersects each box with the sound
clipping ceiling $2\gamma\sum_t \eta_t$ and replaces an invalid one by it (logged), so the certificate
stays sound. `run_contractive windows` is the sampling-frequency ablation (Algorithm 2): $n$ releases
along one $T = 300$ trajectory, each window certified from the previous noisy release, pure DP with
$\varepsilon / n$ per window, and, with `--modes approx`, approximate DP with Laplace noise at
$(\varepsilon / n, \delta / n)$ per window; chains per finalist ($n \in \{1, 2, 4, 8, 16\}$, $\varepsilon \in
\{5, 10\}$, one seed) into `runs/*/windows.json`, drawn by `plot_windows.py` as `fig11_windows_adult`
and `_d32` (accuracy and the released head's AUC against $n$) and, with `--variant accounting`, as
`fig12_windows_accounting_adult` and `_d32`: the accuracy panel beside the smallest $\delta$ at which the
chain of $n$ pure releases is $(\varepsilon', \delta)$-DP, by the optimal composition theorem for pure DP
(`common/accounting/exact_accounting.py::composed_pure_delta`).

## sst2 — the hybrid: DP-SGD below, AGS on top

`ags/sst2/` runs the IMDB driver on GLUE SST-2 (67,349 training sentences; `shared.py` caches the
all-mpnet-base-v2 embeddings under `data/`): 5k training sentences held out as the test set, the 872
public dev sentences as the public set (PCA fit, selection, threshold), 62,349 private. The DP phase is
`finetune.py`: LoRA (rank 16) on the attention projections plus a linear head, trained by DP-SGD with
Opacus (Poisson batches of 1,024, clipping at 1, the PRV accountant, three epochs) to
$(\varepsilon_1, 5\cdot 10^{-6})$; its own head scored on the test set is the end-to-end arm, and
every sentence is then re-embedded with the adapted encoder (the recipe of Yu et al. 2022). The AGS phase
is the contractive driver on PCA-8 of those features, one task per encoder (`config.py`: `sst2` is the
frozen encoder with the $\lambda,\gamma$ sweep and the DP-SGD head baselines, `sst2_ft_eps<e>` the
fine-tuned ones, which pin the frozen finalist), releasing the $P = 18$ head at every head budget in
both modes. `plot_hybrid.py` composes them: at a total budget $\varepsilon$ the hybrid is the encoder
at $\varepsilon_1$ with the head at $\varepsilon_2 = \varepsilon - \varepsilon_1$, the pair chosen
on the public set; $(\varepsilon_1, \delta/2) + (\varepsilon_2, \delta/2)$ is $(\varepsilon, 10^{-5})$-DP,
and a Cauchy head makes the head's own release pure-DP. `fig17_hybrid_sst2` has the arms against the
end-to-end fine-tuning and the frozen encoder with an AGS or a DP-SGD head (approximate pipeline,
pure-DP head, and a third panel against the private set size at a total budget of 3); the arms and the
chosen splits are written to `runs/hybrid_summary.json`. Tasks `sst2_n<N>` and
`sst2_ft_eps<e>_n<N>` are the data-size panel (the first $N$ private sentences; their fine-tunes use
`PANEL_SETTING`, smaller batches so that 2k and 10k sentences get the full-data number of steps). Two
per-task choices are made on the public set: the decision threshold of every head is tuned on it (as on
Adult, `tune_threshold`) and PCA whitening is on for the frozen encoder and off for the fine-tuned ones.

## accounting — experiments with their dataset, maths in one place

The privacy-critical quantity is the coordinate-wise cross-radius certificate

    Abar_j(k; D) = max( |L_j(k; D) - U_j(k+1; D)| , |U_j(k; D) - L_j(k+1; D)| )

where `T_k(D)` contains every parameter reachable from a dataset within `k` substitutions of `D`,
and at the dataset diameter `T_(m+1)` is defined to equal `T_m`. The engine that turns it into a
budget, calibration, Laplace PLD and RDP, the config schema and immutable artifact writing, is
`common/accounting/`, with its own tests. Run the validation gate with

    PYTHONPATH=. pytest -q common/accounting/tests

Experiments live with their dataset, `ags/<dataset>/accounting/`, each defining the experiment it
accounts for in a `config.py` rather than in whichever driver was written first, and writing into
an `artifacts/` folder beside it. Adding a dataset means a sibling folder, not a copied driver.
`ags/accounting/paths.py` resolves an artifact name to its dataset's folder for the few scripts that
read across datasets.

The blobs drivers, in the order they are usually run: `run_cross_radius_comparison.py` (the generic-PLD
panel and the native Gaussian reference), `run_rdp_comparison.py` (coordinate RDP with interval
certification), `run_dpsgd_pld_reference.py` (the matched baseline, strongest native accountant),
`run_poisson_dpsgd_reference.py` (an unmatched stress test, kept separate on purpose),
`run_public_projection.py` (a symmetric, privacy-free projection), `run_envelope_pld_comparison.py` (the
certified continuous envelope) and `run_endpoint_pld_diagnostic.py` (explicitly **not** certified).

MNIST has two sub-studies with different standing. `pca/`: `run_matched_rdp.py` and
`run_strong_accounting.py` are the long-trajectory comparison, `run_tail_average_rdp.py` a symmetric public
tail averaging, and `run_early_stop_strong.py` the short trajectory whose one-step sensitivity is exact.
Read the last beside the first two: the blobs MILP and LP boxes are unconditional, but the PCA-MNIST
closed-form sensitivity remains conditional on a non-expansiveness proof for the elementwise-clipped
update, and the exact short-trajectory comparison reverses its ordering. `screens/`: the
config-driven LeNet, compact-CNN and logistic screens, run through `run_experiment.py` from a YAML in
`screens/configs/`; these use the engine directly. `run_experiment.py` loads MNIST by name, so a
screen on another dataset needs that loader parameterised first.

The root `ags/accounting/` keeps only what belongs to no dataset: `run_ags_accountant_sweep.py`
(audits the PLD accountant's resolution, touches no data), the overnight summaries and
`plot_overnight_status.py` that read blobs and MNIST together, and `notes/`.

Every generated artifact is written immutably: a rerun must choose a new filename rather than
silently replacing evidence.

## What a serialized run contains

Per cell, under `blobs/runs/certified/<run_id>/`:

    meta.json          full config, data sha256, git sha, versions
    results.json       the certificate grid, the nominal, the per-epsilon rows
    MANIFEST.json      steps and solves written against expected
    steps/k*_w*.npz    (param_l, param_n, param_u) for every radius and window
    solves.jsonl.gz    one record per Gurobi solve: status, MIPGap, ObjBound, ObjVal, runtime
    progress.jsonl.gz  a 30-second heartbeat

Nothing is aggregated away. A solve that hit its time limit yields a valid but loose bound, and the
status field is what tells you which is which. A run without `MANIFEST.json` marked completed is not
a certified cell. Alongside `certified/`, the directory keeps `certified_lp_dropped/` (LP semantics,
sound but looser), `certified_partials/` (killed mid-sweep, kept for their partial steps) and
`repeated_release/`, whose `online_screen_*` runs are superseded by the corrected parallel accounting.

## Two cautions that cost real time

**Global sensitivity is computed, never assumed.** `ags.global_sensitivity()` measures the envelope
with every record substitutable, and it lands exactly on the clipping ceiling. A different quantity,
`ss_reference.gs_window()`, is a closed-form substitution value that assumes the two trajectories'
unsubstituted gradients stay identical. It is not the global sensitivity and is never used as one.

**The paper's stored MNIST run is under-noised by exactly 650**, the parameter count, because the
per-coordinate allocation factor was omitted. `mnist/reference/results.json` holds that run and
`mnist/plot_reproduce_paper.py` redraws the setting with the factor restored. Anything derived from
it, including the GPT-2/IMDB numbers from the same pipeline (that module is gone), must not be reused.

The result JSONs under `<dataset>/ablations/runs/` were copied from the pre-refactor tree so the
plotting pipeline has inputs. They are paper-era runs with the same defect; re-run the drivers to
regenerate them.
