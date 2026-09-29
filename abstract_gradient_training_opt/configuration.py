"""Configuration object for the abstract gradient training module."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import pydantic

LOGGER = logging.getLogger(__name__)

# fields that aren't used when serializing and hashing the configuration
EXCLUDE_FIELDS = [
    "log_level",
    "gurobi_callback",
    "gurobi_env",
    "hash",
    "asdict",
    "set_env",
    "gurobi_kwargs",
    "input_domain",
]


class OPTConfig(
    pydantic.BaseModel, extra="forbid", arbitrary_types_allowed=True
):
    """Configuration class for optimization-based AGT."""

    # logging and callbacks
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        pydantic.Field("INFO", description="Logging level")
    )
    gurobi_callback: Callable = pydantic.Field(
        lambda *args: None, description="Gurobi callback function."
    )
    gurobi_kwargs: dict = pydantic.Field(
        {}, description="Additional gurobi parameters to set."
    )

    # gurobi strategies
    obbt: bool = pydantic.Field(
        default=False, description="Whether to use OBBT in when formulating."
    )
    obbt_timeout: int = pydantic.Field(
        30, gt=0, description="Timeout for OBBT in seconds."
    )
    # optimizer parameters
    n_epochs: int = pydantic.Field(
        ..., gt=0, description="Number of epochs to train the model"
    )
    batchsize: int = pydantic.Field(
        ..., gt=0, description="Batch size for training"
    )
    learning_rate: float = pydantic.Field(
        ..., gt=0, description="Initial learning rate"
    )
    loss: Literal["mse", "hinge", "binary_cross_entropy"] = pydantic.Field(
        "hinge", description="Loss function"
    )
    concretization_freq: int = pydantic.Field(
        -1,
        ge=-1,
        description="Frequency of concretization during training. "
        "If set to -1, the entire training is formulated as a single problem.",
    )
    gradient_clipping: float | None = pydantic.Field(
        None,
        gt=0,
        description="Maximum norm for gradient clipping. "
        "If None, no clipping is applied.",
    )
    encode_feature_transform: bool = pydantic.Field(
        False,
        description="Whether to encode the feature"
        " transform in the optimization problem",
    )

    # perturbation model specification
    k_removed: int = pydantic.Field(
        0, ge=0, description="Number of removals per batch to be certified"
    )
    k_substitution: int = pydantic.Field(
        0,
        ge=0,
        description="Number of removals/insertions per batch to be certified",
    )
    input_domain: tuple[np.ndarray, np.ndarray] | None = pydantic.Field(
        None,
        description="Tuple of (lower_bounds, upper_bounds)"
        "for input feature substitution",
    )
    label_domain: tuple[float, float] | None = pydantic.Field(
        None,
        description="Tuple of (min_label, max_label) for label substitution in regression",
    )
    k_perturbed_features: int = pydantic.Field(
        0, ge=0, description="Number of samples whose features can be perturbed"
    )
    k_perturbed_labels: int = pydantic.Field(
        0, ge=0, description="Number of samples whose labels can be perturbed"
    )
    feature_perturbation: float = pydantic.Field(
        0.0, ge=0, description="Maximum perturbation for features"
    )
    label_perturbation: float = pydantic.Field(
        0.0, ge=0, description="Maximum perturbation for labels"
    )
    paired_perturbation: bool = pydantic.Field(
        False,
        description="Whether the label and feature poisoning are paired",
    )

    # overall formulation specification
    relax_binaries: bool = pydantic.Field(
        False,
        description="Whether to relax binary variables to continuous variables",
    )
    relax_bilinear: bool = pydantic.Field(
        False, description="Whether to relax bilinear terms to linear terms"
    )
    perturbation_constraint: Literal["input", "gradient"] = pydantic.Field(
        "input",
        description="How to encode data perturbation constraints:"
        "'input' for applying the constraint in the input space, "
        "'gradient' for applying the constraint in the gradient space.",
    )

    # fine-grained encoding specification
    poisoning_encoding: Literal["linear", "binary"] = pydantic.Field(
        default=None,  # type: ignore
        description="Encoding for poisoning constraints: 'linear' or 'binary'",
    )
    label_poisoning_encoding: Literal["type_1", "type_2", "type_3"] = (
        pydantic.Field(
            default=None,  # type: ignore
            description="Encoding for label poisoning.",
        )
    )
    relu_encoding: Literal["triangle", "bigm", "complementary"] = (
        pydantic.Field(
            None,  # type: ignore
            description="Encoding for ReLU constraints: 'triangle' or 'bigm'",
        )
    )
    heaviside_encoding: Literal[
        "bigm_binary",
        "bigm_linear",
        "indicator",
        "quadratic_binary",
        "quadratic_linear",
        "all",
    ] = pydantic.Field(
        default=None,  # type: ignore
        description="Encoding for heaviside constraints: 'quadratic' or 'bigm'",
    )
    accuracy_loss_encoding: Literal[
        "quadratic", "bigm", "sum_of_binary", "integer"
    ] = pydantic.Field(
        None,  # type: ignore
        description="Encoding for accuracy loss: 'quadratic' or 'bigm'",
    )
    relu_grad_encoding: Literal["mccormick", "bilinear", "complementary"] = (
        pydantic.Field(
            None,  # type: ignore
            description="Encoding for ReLU gradient: 'mccormick' or 'bilinear'",
        )
    )
    matmul_encoding: Literal["mccormick", "bilinear"] = pydantic.Field(
        None,  # type: ignore
        description="Encoding for matrix mul: 'mccormick' / 'bilinear'",
    )
    linear_grad_encoding: Literal["mccormick", "bilinear"] = pydantic.Field(
        None,  # type: ignore
        description="Encoding for linear gradient: 'mccormick' or 'bilinear'",
    )
    hinge_loss_prod_encoding: Literal["mccormick", "bilinear"] = pydantic.Field(
        None,  # type: ignore
        description="Encoding for hinge loss product: 'mccormick' / 'bilinear'",
    )
    sigmoid_encoding: Literal["interval", "envelope", "pwl"] = pydantic.Field(
        None,  # type: ignore
        description="Relaxation of the sigmoid in the logistic loss: "
        "'interval' (box only), 'envelope' (2 linear estimators, no binaries) "
        "or 'pwl' (sigmoid_n_pieces segments, binary-selected).",
    )
    sigmoid_n_pieces: int = pydantic.Field(
        4,
        ge=1,
        description="Number of segments for the 'pwl' sigmoid encoding.",
    )

    metadata: dict = pydantic.Field(
        default_factory=dict,
        description="Metadata for the config, e.g., dataset or model name.",
    )

    def hash(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> str:
        """Return a hash of the configuration, used for tracking experiments.

        Should not be used for dynamic storage of configurations, as this object
        is mutable.
        """
        self_dict = self.asdict(drop_fields)
        return hashlib.md5(str(self_dict).encode()).hexdigest()

    def json(
        self, drop_fields: list[str] = EXCLUDE_FIELDS, *args, **kwargs
    ) -> str:
        """Return the configuration as JSON, excluding `drop_fields`."""
        return self.model_dump_json(exclude=set(drop_fields), *args, **kwargs)

    def asdict(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> dict:
        """Return the configuration as a dictionary, excluding `drop_fields`."""
        self_dict = dict(self)
        # drop fields that don't change the results we are trying to store
        for field in drop_fields:
            self_dict.pop(field, None)
        return self_dict
        # encoding specifications

    def __init__(self, **data):
        """Set default encodings based on relaxation setting if not provided."""
        super().__init__(**data)
        if self.poisoning_encoding is None:
            self.poisoning_encoding = (
                "linear" if self.relax_binaries else "binary"
            )
        # if self.label_poisoning_encoding is None:
        #     self.label_poisoning_encoding = (
        #         "type_1" if self.relax_binaries else "type_3"
        #     )
        if self.relu_encoding is None:
            self.relu_encoding = "triangle" if self.relax_binaries else "bigm"
        # if self.heaviside_encoding is None:
        #     self.heaviside_encoding = (
        #         "triangle" if self.relax_binaries else "bigm"
        #     )
        if self.accuracy_loss_encoding is None:
            self.accuracy_loss_encoding = (
                "quadratic" if self.relax_binaries else "bigm"
            )
        if self.relu_grad_encoding is None:
            self.relu_grad_encoding = (
                "mccormick" if self.relax_bilinear else "bilinear"
            )
        if self.matmul_encoding is None:
            self.matmul_encoding = (
                "mccormick" if self.relax_bilinear else "bilinear"
            )
        if self.linear_grad_encoding is None:
            self.linear_grad_encoding = (
                "mccormick" if self.relax_bilinear else "bilinear"
            )
        if self.hinge_loss_prod_encoding is None:
            self.hinge_loss_prod_encoding = (
                "mccormick" if self.relax_bilinear else "bilinear"
            )
        if self.sigmoid_encoding is None:
            # Binaries are the scarce resource in the unrolled MILP, and measured
            # at F=4 the binary-free envelope already matches pwl to 4 d.p.
            # (ratio 0.5001 vs 0.5000) because gradient clipping at gamma
            # dominates the sigmoid slack. So "envelope" is the default and
            # "pwl" is opt-in for cases where the logit range is wide.
            self.sigmoid_encoding = "envelope"

    @property
    def all_encodings(self) -> dict[str, str]:
        """Return a dictionary of all encodings used in the configuration."""
        return {
            "poisoning": self.poisoning_encoding,
            "label_poisoning": self.label_poisoning_encoding,
            "relu": self.relu_encoding,
            "heaviside": self.heaviside_encoding,
            "accuracy_loss": self.accuracy_loss_encoding,
            "relu_grad": self.relu_grad_encoding,
            "matmul": self.matmul_encoding,
            "linear_grad": self.linear_grad_encoding,
            "hinge_loss_prod": self.hinge_loss_prod_encoding,
            "sigmoid": self.sigmoid_encoding,
        }
