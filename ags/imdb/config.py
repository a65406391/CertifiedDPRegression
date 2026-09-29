"""Settings of the IMDB contractive comparison (run_contractive.py, plot_contractive.py; fig9).

IMDB, 2 classes, 25k training reviews as all-mpnet-base-v2 sentence embeddings (768-d, public
model, computed once): 5k reviews declared public (2.5k the PATE pool and the warm-start training
set, 2.5k the selection set), 20k private; PCA to d fitted on the public 5k. Test = the 25k test
reviews. The private split is small enough to certify whole, so there is one scale and no final stage.

Four groups share every hyperparameter of the finalist trajectory and differ in the head and the
start: `logistic` (the paper's setting, cold start), `logistic_warm` (start at the public head and
contract toward it), `mlp` (Linear-ReLU-Linear, hidden width h, random start) and `mlp_warm`.
"""

_RADII = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # the staircase maximum sits near k = P / beta <= 6 P

CONTRACTIVE = dict(
    task="imdb",
    encoder="mpnet",
    n_public=5000,
    n_pool=2500,
    seed=0,
    eps=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    delta=1e-5,
    seeds=41,  # AGS release draws per point
    tune_seeds=11,  # baseline grids
    train_seeds=9,  # trained arms in the figure
    select_seeds=3,  # the quick matched DP-SGD that ranks sweep configurations
    select_eps=[10, 20, 50, 100],
    gamma_cauchy=2,
    k_grid=_RADII,  # the batch size (= N, full batch) is appended
    bound_model="crown",
    # stage 1: the certificate and the nominal depend on (d, lambda, gamma); a short trajectory ranks them
    # (0.6 to 0.9 s per certified step on the 20k split, so the grid is what three workers certify in an hour)
    stage1=dict(d=[8, 16, 32], lam=[0.6], gamma=[0.01, 0.02], lr=2.0, decay=0.01, T=100),
    # stage 2: the best stage-1 combinations, longer (AGS's curve is T-invariant under contraction)
    stage2=dict(top=1, variants=[dict(T=1000)]),
    # finalists chosen by hand from the sweep, per group; None = the sweep's ranking (margin over matched)
    # the hidden-layer finalist is pinned: at d = 8 only h = 8 trains (h = 4 stays at the dead fixed point,
    # nominal 0.500, and would win a ranking by margin because a dead network loses nothing to noise)
    final=dict(logistic=None, mlp=dict(h=8, d=8, lam=0.05, gamma=0.02, lr=2.0, decay=0.01, T=1000)),
    # the extra arms, at the finalist's (d, lambda, gamma, lr, decay, T) unless a value is listed here. The
    # hidden-layer head cannot take the finalist's contraction: zero is a dead fixed point of a ReLU
    # network, and lambda >= 0.1 pulls the nominal into it (test accuracy 0.5); at 0.05 it trains, but
    # only with gamma = 0.02 and T = 1000 (at the finalist's gamma = 0.01 and T = 300 it is still at 0.5).
    arms=dict(mlp=dict(h=[4, 8], d=[None], gamma=[0.02], lam=[0.05], T=[1000])),
    # the long trajectory (group logistic_long, `sweep --long`): AGS's release does not change with T under
    # contraction, matched DP-SGD pays T under composition; the figure that shows the gap, as fig7's sweep panel
    long=dict(T=3000),
    # the public head: unclipped SGD on the first n labelled reviews of the pool. With all 2.5k the
    # public head already matches the 20k-review ceiling (0.825 / 0.838 / 0.858 at d = 8 / 16 / 32);
    # 100 leave room for the private data (0.802 / 0.783 / 0.776).
    warm=dict(n=100, T=2000, lr=0.5, lam=1e-3),
    dpsgd_opt_approx=dict(q=[0.01, 0.05], steps=[300, 1000], lr=[0.5, 2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.05, 0.2, 1.0], steps=[5, 20, 100], lr=[2.0, 8.0, 15.0], gamma=[0.1]),
    pld_interval=1e-3,
    pate=dict(teachers=[5, 10, 20, 50], queries=[100, 250, 500, 1000, 2500], n_s=50, lr=2.0, student_n_s=200),
    runs_dir="ags/imdb/runs/contractive",
    figures_dir="ags/imdb/figures",
    fig_stem="fig9_combined_contractive_imdb",
)

# The same head on 2k private reviews (the rest of the private split unused): the b-scaling test. Matched
# DP-SGD's per-step noise grows as 1/b, AGS's certificate as 1/b but its smooth sensitivity is taken at the
# staircase maximum, so the ratio of the two moves in AGS's favour as b shrinks. Logistic groups only.
CONTRACTIVE_N2K = dict(
    CONTRACTIVE,
    task="imdb_n2k",
    n_private=2000,
    stage1=dict(d=[8], lam=[0.6], gamma=[0.01], lr=2.0, decay=0.01, T=100),
    stage2=dict(top=1, variants=[dict(T=1000)]),
    arms=dict(mlp=dict(h=[], d=[None])),
    long=None,
    runs_dir="ags/imdb/runs/contractive_n2k",
    fig_stem="fig9_combined_contractive_imdb_n2k",
)

# every path in minutes: 1500 private reviews, three radii, two budgets
CONTRACTIVE_SMOKE = dict(
    CONTRACTIVE,
    task="imdb_smoke",
    n_private=1500,
    eps=[10, 100],
    seeds=3,
    tune_seeds=2,
    train_seeds=2,
    select_seeds=1,
    select_eps=[100],
    k_grid=[1, 2, 4],
    stage1=dict(d=[8], lam=[0.6], gamma=[0.02], lr=2.0, decay=0.01, T=6),
    stage2=dict(top=1, variants=[dict(T=9)]),
    arms=dict(mlp=dict(h=[4], d=[None], gamma=[None], lam=[0.05], T=[6])),
    long=dict(T=12),
    warm=dict(n=100, T=200, lr=0.5, lam=1e-3),
    dpsgd_opt_approx=dict(q=[0.05], steps=[10], lr=[2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.2], steps=[5], lr=[8.0], gamma=[0.1]),
    pate=dict(teachers=[5], queries=[100], n_s=5, lr=2.0, student_n_s=5),
    runs_dir="ags/imdb/runs/contractive_smoke",
    figures_dir="ags/imdb/runs/contractive_smoke",
    fig_stem="smoke_contractive_imdb",
)


# The paper figure of the IMDB comparison: budgets capped at 100 with the intermediate values that show the
# rise; AGS re-released from the stored envelopes, missing baseline cells added; headline_dense.json.
CONTRACTIVE_DENSE = dict(
    CONTRACTIVE,
    task="imdb",
    eps=[1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100],
    groups=("logistic", "mlp"),
    headline_file="headline_dense.json",
    tight_dir="releases_tight_dense",
    fig_stem="fig9_paper_contractive_imdb",
)
