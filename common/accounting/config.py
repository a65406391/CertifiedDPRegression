"""The validated experiment configuration, loaded from YAML.

A frozen dataclass rather than a dict, so a typo fails at load instead of silently taking a default
halfway through a run. It constrains the choices that change what a number means: the sensitivity
method (cross-radius, or the closed form for a clipped mean), the release mode (per coordinate,
tensor-wise or joint in l1), and the mechanism, which may only be Cauchy alongside a coordinate
release.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "lenet5"
    seed: int = 0
    batch_size: int = 128
    epochs: int = 1
    learning_rate: float = 0.01
    target_epsilon: float = 4.0
    target_delta: float = 1e-5
    beta_epsilon_ratio: float = 1 / 6
    coordinate_clip: float = 0.1
    coordinate_clip_weight: float | None = None
    coordinate_clip_bias: float | None = None
    l2_clip: float = 1.0
    ags_mechanism: str = "cauchy"
    ags_delta_fraction: float = 0.5
    pld_interval: float = 1e-4
    gradient_chunk_size: int = 16
    accountant_search_iterations: int = 24
    sensitivity_method: str = "cross_radius"
    ags_release_mode: str = "coordinate"
    arms: tuple[str, ...] = ("nonprivate", "ags_ss", "dpsgd", "ags_gs")
    experiment: str = "smoke"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.target_epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not 0 < self.target_delta < 1:
            raise ValueError("target_delta must lie in (0,1)")
        if not 0 < self.beta_epsilon_ratio < 1:
            raise ValueError("beta_epsilon_ratio must lie in (0,1)")
        if self.coordinate_clip <= 0 or self.l2_clip <= 0:
            raise ValueError("clipping thresholds must be positive")
        if self.coordinate_clip_weight is not None and self.coordinate_clip_weight <= 0:
            raise ValueError("coordinate_clip_weight must be positive when set")
        if self.coordinate_clip_bias is not None and self.coordinate_clip_bias <= 0:
            raise ValueError("coordinate_clip_bias must be positive when set")
        if self.accountant_search_iterations <= 0:
            raise ValueError("accountant_search_iterations must be positive")
        if self.sensitivity_method not in {"cross_radius", "direct_clipped_mean"}:
            raise ValueError("unsupported sensitivity_method")
        if self.ags_release_mode not in {"coordinate", "tensor_l1", "joint_l1"}:
            raise ValueError("unsupported ags_release_mode")
        if self.ags_mechanism not in {"cauchy", "laplace"}:
            raise ValueError("unsupported AGS mechanism")
        if self.ags_mechanism == "cauchy" and self.ags_release_mode != "coordinate":
            raise ValueError("block l1 release is currently implemented only for Laplace")
        if self.model not in {"lenet5", "logistic", "sample_cnn", "compact_selu_cnn"}:
            raise ValueError("unsupported model")
        if not 0 < self.ags_delta_fraction < 1:
            raise ValueError("ags_delta_fraction must lie in (0,1)")

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if "arms" in raw:
            raw["arms"] = tuple(raw["arms"])
        return cls(**raw)

    def public_dict(self) -> dict[str, object]:
        return asdict(self)
