"""Hyperparameters for the PCA-MNIST AGS ablations.

Multinomial logistic regression on 64 PCA features, which is 650 parameters: that count enters the
noise scale directly, because the coordinate-wise release splits the budget across every coordinate.
`BASE_CONFIG` holds the shared settings and each experiment block overlays what it changes.
"""

from typing import TypedDict
from common.paths import get_output_dir


class Config(TypedDict, total=False):
    """Type definition for configuration."""

    N_TRAIN: int
    N_TEST: int
    BATCH_SIZE: int
    PCA_DIMS: int
    HIDDEN_DIMS: list[int]
    OUTPUT_DIM: int
    TOTAL_STEPS: int
    LR_AGS: float
    LR_LAPLACE: float
    ETA: float
    CLIP_GAMMA: float
    SAMPLING_FREQ: int
    CONCRETIZATION_FREQ: int
    LOSS: str
    BOUND_MODE: str
    EPSILON_TOTALS: list[float]
    EPSILON_TOTALS_SAMPLING_FREQ: list[float]
    N_SEEDS: int
    BASE_SEED: int
    OUT_DIR: str
    DELTA: float
    K_SUBSET: list[int]
    HYBRID_STEPS_LIST: list[int]


BASE_CONFIG: Config = dict(
    N_TRAIN=2000,
    N_TEST=1000,
    BATCH_SIZE=2000,
    PCA_DIMS=64,
    HIDDEN_DIMS=[],
    OUTPUT_DIM=10,
    TOTAL_STEPS=40000,
    LR_AGS=0.05,  # Optimal learning rate for AGS with long windows
    LR_LAPLACE=0.05,  # Matching learning rate for fair comparison
    ETA=0.01,
    CLIP_GAMMA=0.1,
    SAMPLING_FREQ=40000,  # Single abstract window -> 1 epsilon release (AGS dominates)
    CONCRETIZATION_FREQ=100,
    LOSS="cross_entropy",
    BOUND_MODE="crown",  # CROWN bounds for unrolled graph bounding
    EPSILON_TOTALS=[5.0, 10.0, 15.0, 20.0],  # Focus on tight privacy regimes where AGS outperforms
    EPSILON_TOTALS_SAMPLING_FREQ=[20.0],  # Focus on tight privacy regimes where AGS outperforms
    N_SEEDS=1,
    BASE_SEED=42,
    DELTA=2e-4,  # 1/N_TRAIN is 1/2000 = 5e-4, so this is slightly tighter than that
    K_SUBSET=[1, 4, 8],  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 500],
    HYBRID_STEPS_LIST=[0, 8000, 16000, 24000, 32000, 40000],
)  # type: ignore[typeddict-item]

AGS_COMPARISON: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "ags_comparison"),
    "K_SUBSET": [1, 4, 7, 10],
    "EPSILON_TOTALS": [10, 50, 100, 500, 1000, 5000, 10000, 15000, 20000],
    "SAMPLING_FREQ": 10000,
}

APPROX_DP_COMPARISON: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "approx_dp_comparison"),
    "EPSILON_TOTALS": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "TOTAL_STEPS": 12000,
    "K_SUBSET": [1, 5, 9, 13],
    "SAMPLING_FREQ": 6000,
}

SAMPLING_ABLATION: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "ags_sampling_ablation"),
    "K_SUBSET": [1, 5, 9],
}

APPROX_DP_SAMPLING_ABLATION: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "ags_sampling_ablation"),
    "EPSILON_TOTALS_SAMPLING_FREQ": [0.5],
    "TOTAL_STEPS": 9000,
    "K_SUBSET": [1, 10, 20],
}

HYBRID_ABLATION: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "hybrid_ablation"),
}

APPROX_DP_HYBRID_ABLATION: Config = {
    **BASE_CONFIG,
    "OUT_DIR": get_output_dir("mnist", "hybrid_ablation"),
    "EPSILON_TOTALS_SAMPLING_FREQ": [0.5],
    "TOTAL_STEPS": 9000,
    "K_SUBSET": [1, 4, 8],
    "HYBRID_STEPS_LIST": [0, 1800, 3600, 5400, 7200, 9000],
}
