"""Hyperparameters for the linear AGT-R experiments.

`ABLATION` is the concretization-frequency study: the methods (interval propagation, then the
unrolled MILP at F = 1, 2, 4, 8, 16), the radius grid and the clipping threshold. The remaining
blocks configure the baselines, the manual MILP check, and the dimension and data-size ablations.

`BETA` and `GS_VAL` are the retired conventions: a fixed smoothing parameter and an assumed global
sensitivity. The current release derives beta from the budget and takes the global sensitivity as
the computed clipping ceiling. See Notes/REPO_NOTES.md section 5.
"""

# =============================================================================
# Ablation Study (from run_lin_ablation.py)
# Uses w_true from file, privacy certified training with clipping
# =============================================================================
ABLATION = dict(
    N_SAMPLES=80,
    BATCH_SIZE=5,
    EPOCHS=1,
    INPUT_DIM=1,
    SPARSITY=1,
    LR=0.1,
    CLIP_GAMMA=0.1,
    W_TRUE_PATH="agtr/linear/data/w_true.npy",
    INIT_STATE_PATH="agtr/linear/data/init_state.pth",
    METHODS=["ibp", "f1", "f2", "f4", "f8", "f16"],
    FREQ_MAP={"f1": 1, "f2": 2, "f4": 4, "f8": 8, "f16": 16},
    K_VALUES=list(range(0, 21)),
    # Sensitivity params
    GS_VAL=10.0,
    BETA=1.0 / 6.0,
    N_EVAL=50,
    # Output
    OUT_DIR="agtr/linear/values/ablation",
    FIG_DIR="agtr/linear/figures/linear",
)

# =============================================================================
# Baseline Comparison (from run_lin_baselines.py)
# Uses slope/intercept data generation, compares against DP baselines
# =============================================================================
BASELINES = dict(
    N_SAMPLES=80,
    BATCH_SIZE=5,
    LR=0.05,
    EPOCHS=50,
    SLOPE=2,
    INTERCEPT=1,
    X_RANGE=(-2, 2),
    CLIP_X=2.0,
    CLIP_Y=6.0,
    EPSILONS=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    N_TRIALS=1,
    # Output
    OUT_DIR="agtr/linear/values/baselines",
    FIG_DIR="agtr/linear/figures/linear",
)

# =============================================================================
# Manual MIP (from run_lin_manual.py)
# Exact Gurobi MIP formulation for ground truth
# =============================================================================
MANUAL = dict(
    N_SAMPLES=80,
    BATCH_SIZE=5,
    LR=1.0,
    CLIP_GAMMA=0.1,
    SLOPE=0.5,
    INTERCEPT=1.0,
    FREQS=[1, 2, 4, 8, 16],
    K_VALUES=[0, 1, 2, 3, 4, 5],
    # Output
    OUT_DIR="agtr/linear/values/ablation",
)

# =============================================================================
# Dimension Ablation (from run_lin_dim_ablation.py)
# Studies effect of input dimensionality
# =============================================================================
DIM_ABLATION = dict(
    BATCH_SIZE=50,
    LR=0.05,
    EPOCHS=20,
    CLIP_X=2.0,
    CLIP_Y=6.0,
    DIMS=[10, 100, 1000],
    K_VALUES=sorted(list(set([0] + list(range(1, 11)) + list(range(12, 36, 2))))),
    EPSILONS=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    # Sensitivity params
    Y_MIN=-5.0,
    Y_MAX=7.0,
    GS_VAL=12.0,
    BETA=1.0 / 6.0,
    N_EVAL=50,
    # Output
    OUT_DIR="agtr/linear/values/dim_ablation",
    FIG_DIR="agtr/linear/figures/linear",
)

# =============================================================================
# Data Ablation (from run_lin_data_ablation.py)
# Studies effect of dataset size
# =============================================================================
DATA_ABLATION = dict(
    LR=0.05,
    EPOCHS=20,
    N_SAMPLES_LIST=[250, 500, 1000],
    K_VALUES=sorted(list(set([0] + list(range(1, 11)) + list(range(12, 36, 2))))),
    EPSILONS=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    # Sensitivity params
    Y_MIN=-4.0,
    Y_MAX=6.0,
    GS_VAL=10.0,
    BETA=1.0 / 6.0,
    N_EVAL=50,
    SLOPE=2,
    INTERCEPT=1,
    # Output
    OUT_DIR="agtr/linear/values/data_ablation",
)

# =============================================================================
# PATE Comparison: Global Sensitivity (Laplace) vs Smooth Sensitivity (Cauchy)
# =============================================================================
PATE = dict(
    # Data generation
    N_TRAIN=40000,
    N_TEST=5000,
    SLOPE=2,
    INTERCEPT=1,
    X_RANGE=(-2, 2),
    CLIP_Y=6.0,
    # Ensemble
    ENSEMBLE_SIZES=[40, 50, 80],  # [5, 10, 20],
    # Teacher training (standard, for global-sensitivity PATE)
    EPOCHS=25,
    LR=0.05,
    BATCH_SIZE=512,
    # Teacher training (AGT, for smooth-sensitivity PATE)
    K_VALUES=sorted(list(set([0] + list(range(1, 11)) + list(range(12, 36, 2))))),
    CLIP_C=1.0,
    HUBER_DELTA=1.0,
    AGT_EPOCHS=10,
    AGT_LR=0.05,
    AGT_BATCH_SIZE=1024,
    # Sensitivity
    GS_VAL=12.0,  # 2 * CLIP_Y
    BETA=1.0 / 6.0,
    # Privacy
    EPSILONS=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    N_TRIALS=10,
    # Output
    OUT_DIR="agtr/linear/values/pate",
    FIG_DIR="agtr/linear/figures/linear",
)
