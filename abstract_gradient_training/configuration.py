"""Configuration class for the abstract gradient training module."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from collections.abc import Iterable
from typing import Literal

import pydantic
import pydantic.json
import torch

from abstract_gradient_training import bounded_losses
from abstract_gradient_training import bounded_models
from abstract_gradient_training import bounded_optimizers

LOGGER = logging.getLogger(__name__)

LOSS_FNS = ["mse", "binary_cross_entropy", "cross_entropy", "hinge"]

# fields that aren't used when serializing and hashing the configuration
EXCLUDE_FIELDS = [
    "fragsize",
    "device",
    "log_level",
    "early_stopping_callback",
    "on_iter_end_callback",
    "on_iter_start_callback",
    "on_epoch_start_callback",
    "batch_perturbation_callback",
    "dataset_perturbation_callback",
    "model_config",
    "bounded_loss_fn",
    "val_loss",
    "bounded_optimizer_fn",
]


class AGTConfig(
    pydantic.BaseModel, extra="forbid", arbitrary_types_allowed=True
):
    """Configuration class for the abstract gradient training module."""

    # optimizer parameters
    n_epochs: int = pydantic.Field(
        ..., gt=0, description="Number of epochs to train the model"
    )
    optimizer: str = pydantic.Field(
        default="SGD", description="Optimizer to use for training"
    )
    optimizer_kwargs: dict = pydantic.Field(
        default={}, description="Additional keyword arguments for the optimizer"
    )
    learning_rate: float = pydantic.Field(
        ..., gt=0, description="Initial learning rate"
    )
    l1_reg: float = pydantic.Field(
        default=0.0, ge=0, description="L1 regularization factor"
    )
    l2_reg: float = pydantic.Field(
        default=0.0, ge=0, description="L2 regularization factor"
    )
    lr_decay: float = pydantic.Field(
        default=0.0, ge=0, description="Learning rate decay factor"
    )
    lr_decay_type: Literal["batch", "epoch"] = pydantic.Field(
        default="batch",
        description="When to apply lr decay (per batch or per epoch)",
    )
    lr_min: float = pydantic.Field(
        default=0.0,
        ge=0,
        description="Minimum learning rate for learning rate scheduler",
    )
    loss: str = pydantic.Field(
        ..., pattern=f"^({'|'.join(LOSS_FNS)})$", description="Loss function"
    )
    device: str = pydantic.Field(
        default="cpu", description="Device to train the model on"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        pydantic.Field(default="INFO", description="Logging level")
    )
    fragsize: int = pydantic.Field(
        default=10000,
        gt=0,
        description="Size of fragments to split the batch into to avoid OOM",
    )
    multistep_domain: bool = pydantic.Field(
        default=False,
        description="Whether to use the multi-step domain (Privacy only).",
    )
    # poisoning parameters
    k_poison: int = pydantic.Field(
        default=0,
        ge=0,
        description="Number of samples whose features can be poisoned",
    )
    epsilon: float = pydantic.Field(
        default=0,
        ge=0,
        description="Maximum perturbation for feature poisoning",
    )
    poison_input_domain: tuple[torch.Tensor, torch.Tensor] | None = (
        pydantic.Field(
            default=None,
            description=(
                "If specified, the input domain for poisoning. Should be a "
                "tuple of (min, max) tensors."
            ),
        )
    )
    input_domain_poisoning: bool = pydantic.Field(
        default=False,
        description=(
            "Whether to use the input domain or epsilon for bounded feature "
            "poisoning."
        ),
    )
    label_k_poison: int = pydantic.Field(
        default=0,
        ge=0,
        description="Number of samples whose labels can be poisoned",
    )
    label_epsilon: float = pydantic.Field(
        default=0,
        ge=0,
        description="Maximum perturbation for label poisoning (in regression)",
    )
    poison_target_idx: int = pydantic.Field(
        default=-1,
        ge=0,
        description="If specified, labels can only flip to this class.",
    )
    paired_poison: bool = pydantic.Field(
        default=True,
        description=(
            "Whether the label and feature poisoning are paired"
            "(i.e. the same samples are poisoned)."
        ),
    )
    # unlearning parameters
    k_unlearn: int = pydantic.Field(
        default=0,
        ge=0,
        description="Number of removals per batch to be certified",
    )
    # privacy and dp-sgd parameters
    k_private: int = pydantic.Field(
        default=0,
        ge=0,
        description="Number of removals/insertions per batch to be certified",
    )
    clip_gamma: float = pydantic.Field(
        default=float("inf"), gt=0, description="Gradient clipping parameter"
    )
    clip_method: Literal["norm", "clamp"] = pydantic.Field(
        default="clamp", description="Method for clipping gradients"
    )
    metadata: dict = pydantic.Field(
        default={},
        description="Additional metadata to store with the configuration",
    )

    # callback functions
    early_stopping_callback: Callable[[bounded_models.BoundedModel], bool] = (
        pydantic.Field(
            default=lambda bounded_model: False,
            description="Callback function that takes a BoundedModel and"
            " returns True if training should terminate.",
        )
    )
    on_iter_end_callback: Callable[[bounded_models.BoundedModel], None] = (
        pydantic.Field(
            default=lambda bounded_model: None,
            description="Callback function that takes a BoundedModel and "
            "is called at the end of each batch.",
        )
    )
    on_iter_start_callback: Callable[[bounded_models.BoundedModel], None] = (
        pydantic.Field(
            default=lambda bounded_model: None,
            description="Callback function that takes a BoundedModel and is"
            " called at the start of each batch.",
        )
    )
    on_epoch_start_callback: Callable[[bounded_models.BoundedModel], None] = (
        pydantic.Field(
            default=lambda bounded_model: None,
            description="Callback function that takes a BoundedModel and is "
            "called at the start of each epoch.",
        )
    )
    batch_perturbation_callback: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = pydantic.Field(
        default=lambda batch, labels: (batch, labels),
        description="Callback function that takes a batch and applies "
        "poisoning to it.",
    )
    dataset_perturbation_callback: Callable[[Iterable], Iterable] = (
        pydantic.Field(
            default=lambda dataset: dataset,
            description="Callback function that takes a dataset and applies "
            "poisoning to it.",
        )
    )

    @property
    def val_loss(self) -> str:
        """Return the name of the validation loss function,."""
        if self.loss == "mse":
            return "mse"
        return "accuracy"

    def get_bounded_loss_fn(self) -> bounded_losses.BoundedLoss:
        """Return the bounded loss function."""
        if self.loss == "mse":
            return bounded_losses.BoundedMSELoss(reduction="none")
        if self.loss == "binary_cross_entropy":
            return bounded_losses.BoundedBCEWithLogitsLoss(reduction="none")
        if self.loss == "cross_entropy":
            return bounded_losses.BoundedCrossEntropyLoss(reduction="none")
        if self.loss == "hinge":
            margin = self.metadata.get("margin", 1.0)
            return bounded_losses.BoundedHingeLoss(
                reduction="none", margin=margin
            )
        raise ValueError(f"Unknown loss function {self.loss}")

    def get_val_loss_fn(self) -> bounded_losses.BoundedLoss:
        """Return the bounded validation loss function."""
        if self.loss == "mse":
            return bounded_losses.BoundedMSELoss(reduction="mean")
        return bounded_losses.BoundedAccuracy(reduction="mean")

    def get_bounded_optimizer(
        self, model: bounded_models.BoundedModel
    ) -> bounded_optimizers.BoundedOptimizer:
        """Return the bounded optimizer function."""
        optimizer_kwargs = dict(self.optimizer_kwargs)
        # add the standard optimizer kwargs to the optimizer_kwargs dict
        keys = (
            "learning_rate",
            "l1_reg",
            "l2_reg",
            "lr_decay",
            "lr_min",
            "lr_decay_type",
        )
        for key in keys:
            if key in optimizer_kwargs:
                raise ValueError(
                    f"{key} in optimizer_kwargs when in the default config."
                )
            optimizer_kwargs[key] = getattr(self, key)
        if self.optimizer == "SGDM" or "momentum" in optimizer_kwargs:
            return bounded_optimizers.BoundedSGDM(model, **optimizer_kwargs)
        if self.optimizer == "SGD":
            return bounded_optimizers.BoundedSGD(model, **optimizer_kwargs)
        raise ValueError(f"Unknown optimizer {self.optimizer}")

    def hash(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> str:  # pylint: disable=dangerous-default-value
        """Return a hash of the configuration, used for tracking experiments."""
        self_dict = self.asdict(drop_fields)
        return hashlib.md5(str(self_dict).encode()).hexdigest()

    def asdict(self, drop_fields: list[str] = EXCLUDE_FIELDS) -> dict:  # pylint: disable=dangerous-default-value
        """Return the configuration as a dictionary, excluding drop_fields."""
        self_dict = dict(self)
        # drop fields that don't change the results we are trying to store
        for field in drop_fields:
            self_dict.pop(field, None)
        return self_dict
