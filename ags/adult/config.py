"""Settings of the Adult contractive comparison (run_contractive.py, plot_contractive.py; fig10).

Adult, 2 classes, 45,222 rows after dropping missing values (ags/adult/shared.py): 10k test, 5k public
(2.5k the PATE pool and warm-start set, 2.5k the selection set), 30,222 private. One-hot plus
standardisation, then PCA to d fitted on the public rows; the head is Linear(d, 2) with P = 2(d + 1),
so d = 16 / 24 / 32 give P = 34 / 50 / 66. The IMDB driver runs this task through the `loader` hook;
everything else (heads, certificate, proven release rule, baselines, figure) is the IMDB code.
"""

from ags.adult.shared import load_task

_RADII = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

ADULT = dict(
    task="adult",
    loader=load_task,
    n_test=10000,
    n_public=5000,
    n_pool=2500,
    seed=0,
    eps=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    delta=1e-5,
    seeds=41,  # AGS release draws per point
    tune_seeds=11,  # baseline grids
    train_seeds=9,  # trained arms in the figure
    select_seeds=3,  # the quick matched DP-SGD that ranks sweep configurations
    select_eps=[2, 5, 10, 20],  # P = 34..66 on 30k rows: the release should be usable from a few units of eps
    gamma_cauchy=2,
    k_grid=_RADII,  # the batch size (= N, full batch) is appended
    bound_model="crown",
    # stage 1: d >= 16 as asked; lambda = 0.6 and lr = 2 with decay 0.01 won every earlier sweep; gamma is the
    # remaining certificate knob (the envelope saturates at 2 gamma k / (b (lambda - c)))
    # nominal probes (13:40, notes in REFACTOR_LOG): under contraction the argmax is the majority class at every
    # (lambda, gamma), but the head ranks well (AUC 0.84-0.88) and with the threshold chosen on the public selection
    # set it reaches 0.80-0.82 (ceiling 0.83); larger gamma and smaller lambda help accuracy, the certificate wants
    # the opposite, so both are swept. T = 50 suffices for the nominal and the envelope to settle (1.1 s per step).
    stage1=dict(d=[16, 24, 32], lam=[0.3, 0.6], gamma=[0.02, 0.05, 0.1], lr=2.0, decay=0.01, T=50),
    # stage 2: the two best stage-1 configurations at the figure's trajectory length
    stage2=dict(top=0, variants=[dict(T=1000)]),
    # pinned after stage 1 (14:35): the margin over a T = 50 matched arm favours gamma = 0.02 (ceiling 0.80); at
    # gamma = 0.05 the ceiling is 0.822 and AGS is 0.817 at eps = 10 against 0.800, and the matched arm pays 20x
    # more at the figure's T = 1000. lambda = 0.3 beats 0.6 at eps = 5 (0.777 against 0.750) with the same line.
    final=dict(logistic=dict(d=16, lam=0.3, gamma=0.05, lr=2.0, decay=0.01, T=1000), mlp=None),
    # the hidden-layer arm at the finalist's d: h = 8 and 16, lambda = 0.05 (a ReLU head dies under stronger
    # contraction), gamma 0.02, T = 300 (its release is expected at chance, as on IMDB; kept as the record)
    arms=dict(mlp=dict(h=[8, 16], d=[None], gamma=[0.02], lam=[0.05], T=[300])),
    long=None,
    # the public head: unclipped SGD on the first n labelled public rows
    warm=dict(n=100, T=2000, lr=0.5, lam=1e-3),
    dpsgd_opt_approx=dict(q=[0.01, 0.05], steps=[300, 1000], lr=[0.5, 2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.05, 0.2, 1.0], steps=[5, 20, 100], lr=[2.0, 8.0, 15.0], gamma=[0.1]),
    pld_interval=1e-3,
    pate=dict(teachers=[5, 10, 20, 50], queries=[100, 250, 500, 1000, 2500], n_s=50, lr=2.0, student_n_s=200),
    groups=(
        "logistic",
    ),  # 24 Sept: AGS on its own, no warm, hidden-layer or baseline arms; the ablation follows
    # the sampling-frequency ablation (run_contractive windows): n releases along the finalist's trajectory at
    # T = 300 (the release is T-invariant under contraction: identical envelopes at T = 50 and 1000), pure DP,
    # eps / n per window, each window certified from the previous noisy release; n = 1 is the single release
    # ten chains per dataset (24 Sept), then n = 32 and 64 added the same evening (T_w = 9 and 4 steps): the
    # release counts at the two budgets where the single release is
    # on its way up (eps = 5, 0.78) and just at the ceiling (eps = 10, 0.82), one seed
    # both release modes (24 Sept, late): approximate DP is Laplace at (eps / n, delta / n) per window
    windows=dict(
        n=[1, 2, 4, 8, 16, 32, 64], eps=[5, 10], seeds=1, T=300, k_grid=_RADII, modes=("pure", "approx")
    ),
    runs_dir="ags/adult/runs/contractive",
    figures_dir="ags/adult/figures",
    fig_stem="fig10_combined_contractive_adult",
)

# The d = 32 finalist asked for on 24 September (P = 66, ceiling 0.828 in the sweep): the logistic group alone,
# same split and baselines, its own runs directory and figure.
ADULT_D32 = dict(
    ADULT,
    task="adult_d32",
    stage1=dict(d=[32], lam=[0.3], gamma=[0.05], lr=2.0, decay=0.01, T=50),
    stage2=dict(top=0, variants=[dict(T=1000)]),
    final=dict(logistic=dict(d=32, lam=0.3, gamma=0.05, lr=2.0, decay=0.01, T=1000), mlp=None),
    groups=("logistic",),
    runs_dir="ags/adult/runs/contractive_d32",
    fig_stem="fig10_combined_contractive_adult_d32",
    windows_fig_stem="fig11_windows_adult_d32",
)

# every path in minutes: 1500 private rows, three radii, two budgets
ADULT_SMOKE = dict(
    ADULT,
    task="adult_smoke",
    n_private=1500,
    eps=[10, 100],
    seeds=3,
    tune_seeds=2,
    train_seeds=2,
    select_seeds=1,
    select_eps=[100],
    k_grid=[1, 2, 4],
    stage1=dict(d=[16], lam=[0.6], gamma=[0.05], lr=2.0, decay=0.01, T=6),
    stage2=dict(top=1, variants=[dict(T=9)]),
    windows=dict(n=[1, 2], eps=[10], seeds=1, T=8, k_grid=[1, 2, 4], modes=("pure", "approx")),
    arms=dict(mlp=dict(h=[8], d=[None], gamma=[0.02], lam=[0.05], T=[6])),
    warm=dict(n=100, T=200, lr=0.5, lam=1e-3),
    dpsgd_opt_approx=dict(q=[0.05], steps=[10], lr=[2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.2], steps=[5], lr=[8.0], gamma=[0.1]),
    pate=dict(teachers=[5], queries=[100], n_s=5, lr=2.0, student_n_s=5),
    runs_dir="ags/adult/runs/contractive_smoke",
    figures_dir="ags/adult/runs/contractive_smoke",
    fig_stem="smoke_contractive_adult",
)
