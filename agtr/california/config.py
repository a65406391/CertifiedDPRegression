"""Hyperparameters for the California Housing AGT-R experiments.

`ABLATION` is the batch-size study: the batch sizes, the radius grid, the learning rate and the
standardised output range that fixes the global sensitivity. `BASELINES` configures the DP-SGD and
functional-mechanism arms.

Two cautions. `BETA` here is the retired fixed smoothing parameter; the current mechanisms derive
beta from the budget through `common.mechanisms.beta_max`, so a fixed value is only valid above a
threshold in epsilon. And the paper's 8-64-64-1 model saturates under a sound certificate, which is
why the live configuration is one hidden layer at a smaller clipping threshold.
"""

# =============================================================================
# Ablation Study (from run_ca_ablation.py)
# Batch size sweep with fixed optimization steps
# =============================================================================
ABLATION = dict(
    BATCH_SIZES=[250, 500, 1000, 2000, 4000],
    BASE_BATCH_SIZE=1000,
    BASE_EPOCHS=20,
    LR=0.01,
    FREQ=1,
    FIXED_STEPS_MODE=True,
    # Sensitivity params
    Y_MIN=-5.0,
    Y_MAX=5.0,
    BETA=1.0 / 6.0,
    K_VALUES=sorted(list(set([0] + list(range(1, 11)) + list(range(13, 31, 3))))),
    # Clipping
    CLIP_C=1.0,
    HUBER_DELTA=1.0,
    # Output
    OUT_DIR="agtr/california/values/ablation",
    FIG_DIR="agtr/california/figures/california",
    PLOT_PREFIX="ca_batch_scaled",
)

# =============================================================================
# Baseline Comparison (from run_ca_baselines.py)
# =============================================================================
BASELINES = dict(
    BATCH_SIZE=8000,
    EPOCHS=40,
    LR=0.08,
    CLIP_X=5.0,
    CLIP_Y=5.0,
    EPSILONS=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    N_TRIALS=1,
    # Output
    OUT_DIR="agtr/california/values/baselines",
    FIG_DIR="agtr/california/figures/california",
)
