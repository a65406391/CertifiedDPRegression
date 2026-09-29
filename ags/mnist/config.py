"""The MNIST PCA-64 experiments' settings.

OPTIMIZED is the setting of `run_optimized.py`, the figure that pits an optimized AGS release against
matched and optimized DP-SGD and PATE on the logistic head (P = 650). One dictionary, read by the
driver and the plot, so the grid, the budgets and the split are stated once.
"""

# ------------------------------------------------------------------------------------------------
# run_optimized.py
# ------------------------------------------------------------------------------------------------

_RADII = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
_RADII += [1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152]

OPTIMIZED = dict(
    # data: all of MNIST, PCA-64 fitted on the private split, features scaled to max ||x|| <= 1.
    # 5000 training points are held out as public data: 2500 as the PATE query pool, 2500 as the
    # selection set every tuned arm picks its hyperparameters on. Reporting is on the 10k test set.
    N=60000,
    d=64,
    n_val=5000,
    n_pool=2500,
    n_test=10000,
    seed=0,
    # budgets and evaluation
    eps=[1, 2, 5, 10, 15, 20, 30, 50, 100],
    delta=1e-5,
    seeds=41,  # release draws / training seeds for every reported point (median)
    tune_seeds=11,  # for the grids the optimized arms are selected from
    gamma_cauchy=2,
    # AGS envelopes: radius grid (the batch size is appended), bound model, subsample fractions
    k_grid=_RADII,
    bound_model="crown",  # identical to ibp on one linear layer; kept as the record
    # stage 1: full data, q = 1. b: "full" = one batch of all training points, "fifth" = five per epoch
    ags_stage1=[
        dict(b="full", n_s=1, lr=8.0, decay=0.0, gamma=0.1),
        dict(b="full", n_s=2, lr=8.0, decay=0.0, gamma=0.1),
        dict(b="full", n_s=3, lr=4.0, decay=0.0, gamma=0.1),
        dict(b="full", n_s=5, lr=4.0, decay=0.2, gamma=0.1),
        dict(b="full", n_s=10, lr=2.0, decay=0.1, gamma=0.1),
        dict(b="full", n_s=10, lr=2.0, decay=0.1, gamma=0.05),
        dict(b="full", n_s=10, lr=8.0, decay=0.5, gamma=0.1),
        dict(b="full", n_s=20, lr=2.0, decay=0.2, gamma=0.1),
        dict(b="full", n_s=40, lr=1.0, decay=0.1, gamma=0.1),
        dict(b="full", n_s=5, lr=15.0, decay=0.0, gamma=0.12),  # the stored headline's schedule
        dict(b="fifth", n_s=10, lr=2.0, decay=0.1, gamma=0.1),
        dict(b="fifth", n_s=25, lr=2.0, decay=0.2, gamma=0.1),
    ],
    # stage 2: the two best stage-1 configurations by selection-set accuracy at eps = 20 (sound rule),
    # re-run on persistent without-replacement subsamples, and with the learning rate halved/doubled
    ags_stage2_q=[0.5, 0.25],
    ags_stage2_lr_factors=[0.5, 2.0],
    ags_stage2_top=2,
    # DP-SGD, optimized: Poisson sampling, PLD accounting (approx) / amplified basic composition (pure)
    dpsgd_opt_approx=dict(q=[0.01, 0.05], steps=[300, 1000], lr=[0.5, 2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.01, 0.05, 0.2, 1.0], steps=[5, 20], lr=[0.5, 2.0, 8.0, 15.0], gamma=[0.1]),
    pld_interval=1e-3,
    # PATE: disjoint teachers, noisy argmax on the public pool, a student on the labels
    # PATE: only the votes are privatised, so teachers and student train unclipped and to convergence;
    # a clipped few-step student turned the votes' class skew into bias terms and predicted one class
    pate=dict(
        teachers=[5, 10, 20, 50, 100],
        queries=[100, 250, 500, 1000, 2500],
        n_s=50,
        lr=2.0,
        gamma=1.0,
        student_n_s=200,
    ),
    # where things go
    runs_dir="ags/mnist/runs/optimized",
    figures_dir="ags/mnist/figures",
)

# A small setting that exercises every code path in minutes; outputs go to a separate folder.
OPTIMIZED_SMOKE = dict(
    OPTIMIZED,
    N=6000,
    n_val=1000,
    n_pool=500,
    n_test=1000,
    eps=[10, 100],
    seeds=3,
    tune_seeds=2,
    k_grid=[1, 2, 4, 8],
    ags_stage1=[
        dict(b="full", n_s=2, lr=4.0, decay=0.0, gamma=0.1),
        dict(b="fifth", n_s=5, lr=2.0, decay=0.1, gamma=0.1),
    ],
    ags_stage2_q=[0.5],
    ags_stage2_lr_factors=[2.0],
    ags_stage2_top=1,
    dpsgd_opt_approx=dict(q=[0.05], steps=[20], lr=[2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.05], steps=[10], lr=[2.0], gamma=[0.1]),
    pate=dict(teachers=[5], queries=[100], n_s=5, lr=2.0, gamma=1.0, student_n_s=10),
    runs_dir="ags/mnist/runs/optimized_smoke",
)


# ------------------------------------------------------------------------------------------------
# run_contractive.py: L2-contractive training, whole-vector release, budgets to 1000
# ------------------------------------------------------------------------------------------------

_CONTRACTIVE_RADII = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 4096]

CONTRACTIVE = dict(
    task="mnist10",
    classes=list(range(10)),
    # data: the task's digits, PCA-d fitted on the private split, features scaled to max ||x|| <= 1.
    # 2000 training points are public: 1000 the PATE pool, 1000 the selection set. Test = every test
    # image of the task's classes. The sweep certifies on a 10k subset of the private split; the final
    # figure re-certifies the winners on all of it.
    n_val=2000,
    n_pool=1000,
    n_sweep=10000,
    seed=0,
    eps=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    delta=1e-5,
    seeds=41,  # AGS release draws per point (cheap)
    tune_seeds=11,  # grids and the sweep's matched arms
    final_train_seeds=9,  # trained arms in the final figure (each is a full trajectory on the full split; 21 would take 5 h)
    select_seeds=3,  # the quick matched DP-SGD used only to rank sweep configurations
    select_eps=[100, 200, 500, 1000],
    gamma_cauchy=2,
    k_grid=_CONTRACTIVE_RADII,  # the batch size (= N, full batch) is appended
    bound_model="crown",
    # stage 1: the certificate and the nominal depend on (d, lambda, gamma); a short trajectory ranks them
    stage1=dict(d=[16, 32], lam=[0.3, 0.6], gamma=[0.02, 0.05, 0.1], lr=1.0, decay=0.01, T=100),
    # stage 2: the best stage-1 combinations, longer and with other learning rates
    stage2=dict(top=2, variants=[dict(T=300, lr=2.0), dict(T=1000, lr=2.0), dict(T=300, lr=0.5)]),
    # re-certified on the full private split: one finalist per task fits the budget (about 2.5 h at
    # T = 1000 on 58k), on the radii that matter at eps >= 100 (the staircase maximum sits below k = 40)
    # the 10-class finalist, chosen from the sweep: beats matched pure DP-SGD 0.584 vs 0.416 at eps=100 and
    # 0.646 vs 0.535 at eps=200 on the selection set, then both sit at the nominal (0.658)
    final=dict(
        top=1,
        k_grid=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        list=[dict(d=8, lam=0.6, gamma=0.01, lr=2.0, decay=0.01, T=1000)],
    ),
    # Algorithm 2: windows with envelope collapse at each released point; pure DP, two budgets, few seeds
    windows=dict(
        n=[2, 4], seeds=2, eps=[100], k_grid=[1, 2, 4, 8, 16, 32, 128, 1024]
    ),  # bounded: ~80 min on the 10k subset
    dpsgd_opt_approx=dict(q=[0.01, 0.05], steps=[300, 1000], lr=[0.5, 2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.05, 0.2, 1.0], steps=[5, 20, 100], lr=[2.0, 8.0, 15.0], gamma=[0.1]),
    pld_interval=1e-3,
    pate=dict(
        teachers=[5, 10, 20, 50, 100],
        queries=[100, 250, 500, 1000],
        n_s=50,
        lr=2.0,
        gamma=1.0,
        student_n_s=200,
    ),
    # the sweep-scale figure shows the longest trajectory; AGS's curve is T-invariant, the matched arm is not
    sweep_figure=[dict(d=8, lam=0.6, gamma=0.01, lr=2.0, decay=0.01, T=3000)],
    runs_dir="ags/mnist/runs/contractive10",
    figures_dir="ags/mnist/figures",
    fig_stem="fig7_combined_contractive",
)

# The 5-class task: digits 0-4 (there is no standard 5-class grouping in the literature; a subset keeps
# each class unimodal for a linear model).
CONTRACTIVE_5 = dict(
    CONTRACTIVE,
    task="mnist5",
    classes=[0, 1, 2, 3, 4],
    # what the 10-class sweep taught: small d, and only the long trajectory matters in stage 2
    stage1=dict(d=[8, 12, 16], lam=[0.3, 0.6], gamma=[0.02, 0.05, 0.1], lr=1.0, decay=0.01, T=100),
    stage2=dict(top=2, variants=[dict(T=1000, lr=2.0)]),
    # the 5-class finalist: the higher-nominal d = 8 trajectory (0.854); its certificate is T-invariant, so the
    # T = 100 sweep numbers (0.853 at eps=100, 0.836 at 50, 0.62 at 20) carry over to T = 1000
    final=dict(
        top=1,
        k_grid=CONTRACTIVE["final"]["k_grid"],
        list=[dict(d=8, lam=0.6, gamma=0.02, lr=2.0, decay=0.01, T=1000)],
    ),
    sweep_figure=[dict(d=8, lam=0.6, gamma=0.02, lr=2.0, decay=0.01, T=3000)],
    runs_dir="ags/mnist/runs/contractive5",
    fig_stem="fig8_combined_contractive_5class",
)

CONTRACTIVE_SMOKE = dict(
    CONTRACTIVE,
    task="mnist10",
    n_val=600,
    n_pool=300,
    n_sweep=3000,
    eps=[100, 1000],
    seeds=3,
    tune_seeds=2,
    final_train_seeds=2,
    select_seeds=1,
    select_eps=[100, 1000],
    k_grid=[1, 2, 8],
    stage1=dict(d=[16], lam=[0.3], gamma=[0.1], lr=1.0, decay=0.01, T=5),
    stage2=dict(top=1, variants=[dict(T=10, lr=2.0)]),
    final=dict(top=1),
    windows=dict(n=[2], seeds=1, eps=[100], k_grid=[1, 8]),
    dpsgd_opt_approx=dict(q=[0.05], steps=[10], lr=[2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.2], steps=[5], lr=[8.0], gamma=[0.1]),
    pate=dict(teachers=[5], queries=[100], n_s=5, lr=2.0, gamma=1.0, student_n_s=10),
    runs_dir="ags/mnist/runs/contractive_smoke",
    fig_stem="fig7_combined_contractive_smoke",
)


# Extra grids added mid-run, on the second GPU, writing their own sweep file (sweep_x.json) in the
# same runs_dir; the final stage reads the union. Stage 1 showed smaller d and smaller gamma helping
# AGS, so these push both further.
CONTRACTIVE_X = dict(
    CONTRACTIVE,
    sweep_file="sweep_x.json",
    stage1=dict(d=[8, 12, 16], lam=[0.3, 0.6], gamma=[0.02, 0.01, 0.005], lr=1.0, decay=0.01, T=100),
)
CONTRACTIVE_5X = dict(
    CONTRACTIVE_5,
    sweep_file="sweep_x.json",
    stage1=dict(d=[8, 12, 16], lam=[0.3, 0.6], gamma=[0.02, 0.01, 0.005], lr=1.0, decay=0.01, T=100),
)

# T = 3000 on the two best trajectories: AGS is invariant to T once the envelope has saturated, matched
# DP-SGD is not. Explicit configurations, own sweep file, second GPU after the extra grid.
_LONG = [
    dict(d=8, lam=0.6, gamma=0.01, lr=2.0, decay=0.01, T=3000),
    dict(d=16, lam=0.6, gamma=0.02, lr=2.0, decay=0.01, T=3000),
]
CONTRACTIVE_Y = dict(
    CONTRACTIVE,
    sweep_file="sweep_y.json",
    stage1_list=_LONG,
    stage2=dict(top=0, variants=[]),
    final=dict(
        top=2, k_grid=CONTRACTIVE["final"]["k_grid"], list=_LONG
    ),  # matched arms for both long trajectories
)
_LONG_5 = [
    dict(d=8, lam=0.6, gamma=0.02, lr=2.0, decay=0.01, T=1000),
    dict(d=8, lam=0.6, gamma=0.02, lr=2.0, decay=0.01, T=3000),
]
CONTRACTIVE_5Y = dict(
    CONTRACTIVE_5,
    sweep_file="sweep_y.json",
    stage1_list=_LONG_5,
    stage2=dict(top=0, variants=[]),
    final=dict(top=2, k_grid=CONTRACTIVE["final"]["k_grid"], list=_LONG_5),
)


# The paper figure of the 5-class comparison: the same runs and finalist, budgets capped at 100 with the
# intermediate values that show the rise, AGS re-released from the stored envelopes and the missing baseline
# cells added to baselines_full.json (additive; existing cells are kept). Headline in headline_dense.json.
CONTRACTIVE_5_DENSE = dict(
    CONTRACTIVE_5,
    task="mnist5",
    eps=[1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100],
    headline_file="headline_dense.json",
    tight_dir="releases_tight_dense",
    fig_stem="fig8_paper_contractive_5class",
)
