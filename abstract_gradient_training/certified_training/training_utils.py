"""Helper functions for certified training."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Literal
from typing import overload

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_losses import BoundedLoss
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.configuration import AGTConfig

LOGGER = logging.getLogger(__name__)


@overload
def compute_batch_gradients(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    config: AGTConfig,
    nominal: Literal[True],
    poisoned: bool = False,
) -> list[torch.Tensor]: ...


@overload
def compute_batch_gradients(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    config: AGTConfig,
    nominal: Literal[False],
    poisoned: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...


def compute_batch_gradients(
    bounded_model: BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    config: AGTConfig,
    nominal: bool,
    poisoned: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor]:
    """Helper function to calculate gradients of the loss function.

    If the `nominal` flag is set to True, the function computes gradients of
    the loss function with respect to the model parameters. Otherwise, compute
    bounds on the gradients of the loss function with respect to bounds on the
    model parameters. If the `poisoned` flag is set to True, the bounds are
    additionally computed wrt the poisoning adversary.

    Args:
        bounded_model: Bounded version of the pytorch model to train.
        batch: Batch of inputs.
        labels: Batch of labels.
        config: Configuration object for the abstract gradient training module.
        nominal: Whether to compute nominal gradients or bounds on the grads.
        poisoned: Whether to additionally compute bounds wrt the poisoning
            adversary stored in the config. Defaults to False.

    Returns:
        tuple: If `nominal` is True, return the list of gradients. Otherwise,
            return the list of lower and upper bounds on the gradients.
    """
    if nominal and poisoned:
        raise ValueError(
            "Nominal gradients cannot be computed wrt the poisoning adversary."
        )

    bounded_loss = config.get_bounded_loss_fn()
    batch, labels = (
        batch.to(bounded_model.device),
        labels.to(bounded_model.device),
    )

    if nominal:  # compute nominal gradients
        logit_n = bounded_model.forward(batch, retain_intermediate=True)
        dl = bounded_loss.backward(logit_n, labels)
        return bounded_model.backward(dl)

    # set poisoning parameters
    epsilon = config.epsilon if poisoned else 0.0
    label_epsilon = config.label_epsilon if poisoned else 0.0
    label_k_poison = config.label_k_poison if poisoned else 0
    poison_target_idx = config.poison_target_idx if poisoned else -1

    # handle the case of a specified input domain
    if config.input_domain_poisoning and poisoned:
        assert config.poison_input_domain is not None
        # compute the bounded gradients
        grads_l, grads_u = bounded_model.bound_backward_combined(
            config.poison_input_domain[0].to(bounded_model.device),
            config.poison_input_domain[1].to(bounded_model.device),
            labels[0],
            bounded_loss,
            label_k_poison=label_k_poison,
            label_epsilon=label_epsilon,
            poison_target_idx=poison_target_idx,
        )
        interval_arithmetic.validate_interval(
            grads_l, grads_u, msg="gradient bounds"
        )
        grads_l = [g.expand(len(labels), *g.shape).clone() for g in grads_l]
        grads_u = [g.expand(len(labels), *g.shape).clone() for g in grads_u]
        return grads_l, grads_u

    # compute the bounded gradients
    grads_l, grads_u = bounded_model.bound_backward_combined(
        batch - epsilon,
        batch + epsilon,
        labels,
        bounded_loss,
        label_k_poison=label_k_poison,
        label_epsilon=label_epsilon,
        poison_target_idx=poison_target_idx,
    )
    interval_arithmetic.validate_interval(
        grads_l, grads_u, msg="gradient bounds"
    )
    return grads_l, grads_u


def compute_loss(
    bounded_model: BoundedModel,
    bounded_loss: BoundedLoss,
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute the loss and bounds over the given batch of data.

    Args:
        bounded_model (BoundedModel): Bounded version of the pytorch model.
        bounded_loss (BoundedLoss): Bounded version of the loss function to use.
        batch (torch.Tensor): Batch of inputs.
        labels (torch.Tensor): Batch of labels.

    Returns:
        tuple[float, float, float]:
    """
    batch, labels = (
        batch.to(bounded_model.device),
        labels.to(bounded_model.device),
    )
    logits_val = bounded_model.forward(batch)
    val_loss_n = bounded_loss.forward(logits_val, labels)
    logits_val_l, logits_val_u = bounded_model.bound_forward(batch, batch)
    val_loss_l, val_loss_u = bounded_loss.bound_forward(
        logits_val_l, logits_val_u, labels
    )
    return val_loss_l.item(), val_loss_n.item(), val_loss_u.item()


def dataloader_wrapper(
    dl_train: Iterable, epoch: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Returns a generator that handles extra AGT logic for the dataloader.

    - Ensuring that each batch is full and ignoring any incomplete batches.
    - Setting the batchsize based on the first batch and comparing this with all
        subsequent batches.

    Args:
        dl_train: Training dataloader that returns (batch, labels)
            tuples at each iteration.
        epoch: Current epoch number.

    Yields:
        - batch, labels: Post-processed (batch, labels) tuples.
    """
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    t = 0  # possibly undefined loop variable
    for t, (batch, labels) in enumerate(dl_train, 1):
        # initialise the batchsize variable if this is the first iteration
        if full_batchsize is None:
            full_batchsize = batch.size(0)
            LOGGER.debug(
                "Initialising dataloader batchsize to %s", full_batchsize
            )
        # check the batch is the correct size, otherwise skip it
        if batch.size(0) != full_batchsize:
            LOGGER.debug(
                "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                t,
                epoch,
                full_batchsize,
                batch.size(0),
            )
            continue
        # return the batches for this iteration
        yield batch, labels
    # check the number of batches we have processed and report the appropriate
    # warnings
    assert t != 0, f"Dataloader is empty at epoch {epoch}!"
    if epoch == 1 and t == 1:
        LOGGER.warning(
            "Dataloader has only one batch per epoch, effective "
            "batchsize may be smaller than expected."
        )


def dataloader_pair_wrapper(
    dl_train: Iterable, dl_aux: Iterable | None, epoch: int
) -> Iterator[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
]:
    """Returns a generator that pairs the nominal and auxiliary training data.

    - Ensuring that each batch is full and ignoring any incomplete batches.
    - Setting the batchsize based on the first batch and comparing this with all
        subsequent batches.
    - Returning pairs of batches from the input dataloaders.

    In AGT, the auxiliary dataloader is used for poisoning or privacy, where the
    auxiliary data represents clean or public data, respectively.

    Args:
        dl_train (Iterable): Dataloader that returns (batch, labels) tuples at
            each iteration.
        dl_aux (Iterable | None): Optional additional dataloader for auxiliary
            data that returns (batch, labels) tuples at each iteration.
        epoch (int): Current epoch number.

    Yields:
        batch, labels, batch_aux, labels_aux: Tuples of post-processed (batch,
            labels, batch_aux, labels_aux) for each iteration.
    """
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # handle the case where there is no auxiliary dataloader by returning dummy
    # values
    if dl_aux is None:
        data_iterator: Iterable = (((b, l), (None, None)) for b, l in dl_train)
    else:
        data_iterator = zip(
            dl_train, dl_aux
        )  # note that zip will stop at the shortest iterator
    t = 0  # possibly undefined loop variable
    for t, ((batch, labels), (batch_aux, labels_aux)) in enumerate(
        data_iterator, 1
    ):
        batchsize = batch.size(0)
        if batch_aux is not None:
            batchsize += batch_aux.size(0)
        # initialise the batchsize variable if this is the first iteration
        if full_batchsize is None:
            full_batchsize = batchsize
            LOGGER.debug(
                "Initialising dataloader batchsize to %s", full_batchsize
            )
        # check the batch is the correct size, otherwise skip it
        if batchsize != full_batchsize:
            LOGGER.debug(
                "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                t,
                epoch,
                full_batchsize,
                batchsize,
            )
            continue
        # return the batches for this iteration
        yield batch, labels, batch_aux, labels_aux
    # check the number of batches we have processed and report the appropriate
    # warnings
    assert t != 0, f"Dataloader is empty at epoch {epoch}!"
    if t == 1 and epoch == 1:  # only report once
        LOGGER.debug(
            "Dataloader has only one batch per epoch, effective batchsize may"
            " be smaller than expected."
        )


def dataloader_cycle(
    dl: Iterable,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Return a new generator that cycles over the dataloader indefinitely.

    We use this for the 'validation' data, which we want to evaluate at every
    iteration without worrying about reaching the end of the data.
    """
    while True:
        yield from iter(dl)


def propagate_clipping(
    x_l: list[torch.Tensor],
    x: list[torch.Tensor],
    x_u: list[torch.Tensor],
    gamma: float,
    method: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Propagate the input through a clipping operation.

    This function is used to clip the gradients in the DP-SGD algorithm.

    Args:
        x_l (list[torch.Tensor]): Lower bound of the input tensors.
        x (list[torch.Tensor] | None): Optional nominal input tensor.
        x_u (list[torch.Tensor]): Upper bound of the input tensors.
        gamma (float): Clipping parameter.
        method (str): Clipping method, one of ["clamp", "norm"].

    Returns:
        list[torch.Tensor]: Lower bound on the clipped input tensor.
        list[torch.Tensor]: Nominal clipped input tensor.
        list[torch.Tensor]: Upper bound on the clipped input tensor.
    """
    if method == "clamp":
        interval_arithmetic.validate_interval(x_l, x_u, msg="input")
        x_l = [xi.clamp_(-gamma, gamma) for xi in x_l]
        x = [xi.clamp_(-gamma, gamma) for xi in x]
        x_u = [xi.clamp_(-gamma, gamma) for xi in x_u]
    elif method == "norm":
        interval_arithmetic.validate_interval(x_l, x_u, msg="input")
        for i in range(len(x_l)):
            # compute interval over the norm of the input interval
            norms = x[i].flatten(1).norm(2, dim=1)
            norms_l, norms_u = interval_arithmetic.propagate_norm(
                x_l[i], x_u[i], p=2
            )
            interval_arithmetic.validate_interval(norms_l, norms_u, msg="norm")
            # compute an interval over the clipping factor
            clip_factor = (gamma / (norms + 1e-6)).clamp(max=1.0)
            clip_factor_l = (gamma / (norms_u + 1e-6)).clamp(max=1.0)
            clip_factor_u = (gamma / (norms_l + 1e-6)).clamp(max=1.0)
            interval_arithmetic.validate_interval(
                clip_factor_l, clip_factor_u, msg="clip factor"
            )
            # compute an interval over the clipped input
            x_l[i], x_u[i] = interval_arithmetic.propagate_elementwise(
                x_l[i],
                x_u[i],
                clip_factor_l.view(-1, 1, 1),
                clip_factor_u.view(-1, 1, 1),
            )
            x[i] = x[i] * clip_factor.view(-1, 1, 1)
        interval_arithmetic.validate_interval(x_l, x_u, msg="clipped input")
    else:
        raise ValueError(f"Clipping method {method} not recognised.")
    return x_l, x, x_u


def validate_bounded_model(bounded_model: BoundedModel):
    """Validate that the final model returned by AGT has valid bounds."""
    # check that the parameters are within the certified bounds
    for i, (pl, pn, pu) in enumerate(
        zip(bounded_model.param_l, bounded_model.param_n, bounded_model.param_u)
    ):
        # total number of violations
        violations = (pl > pn).sum() + (pn > pu).sum()
        # maximum magnitude of the violation
        max_violation = max((pl - pn).max().item(), (pn - pu).max().item())
        if violations > 0 and max_violation > 1e-6:
            LOGGER.warning(
                "Nominal parameters not within certified bounds for param %s",
                i,
            )
            LOGGER.debug("\tNumber of violations: %s", violations.item())
            LOGGER.debug("\tMax violation: %.2e", max_violation)


def log_run_start(
    config: AGTConfig,
    agt_type: Literal["poison", "privacy", "unlearning", "nominal"],
) -> None:
    """Log the configuration at the start of an AGT run."""
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info(
        "================== Starting %s Certified Training ==================",
        agt_type.capitalize(),
    )
    LOGGER.debug(
        "\tOptimizer params: n_epochs=%s, learning_rate"
        "=%s, l1_reg=%s, l2_reg=%s",
        config.n_epochs,
        config.learning_rate,
        config.l1_reg,
        config.l2_reg,
    )
    LOGGER.debug(
        "\tLearning rate schedule: lr_decay=%s, lr_min=%s",
        config.lr_decay,
        config.lr_min,
    )

    # log gradient clipping and noise
    LOGGER.debug(
        "\tGradient clipping: gamma=%s, method=%s",
        config.clip_gamma,
        config.clip_method,
    )

    # log type-specific parameters
    if agt_type == "poison":
        LOGGER.debug(
            "\tAdversary feature-space budget: epsilon=%s, k_poison=%s",
            config.epsilon,
            config.k_poison,
        )
        LOGGER.debug(
            "\tAdversary label-space budget: label_epsilon=%s, "
            "label_k_poison=%s, poison_target_idx=%s",
            config.label_epsilon,
            config.label_k_poison,
            config.poison_target_idx,
        )
        LOGGER.debug(
            "\tPaired poisoning: %s, Input domain: %s",
            config.paired_poison,
            config.input_domain_poisoning,
        )
    elif agt_type == "privacy":
        LOGGER.debug("\tPrivacy parameter: k_private=%s", config.k_private)
    elif agt_type == "unlearning":
        LOGGER.debug("\tUnlearning parameter: k_unlearn=%s", config.k_unlearn)

    # warn against suspicious configurations
    if (
        not config.input_domain_poisoning
        and config.poison_input_domain is not None
    ):
        LOGGER.warning(
            "poison_input_domain is specified, but "
            "input_domain_poisoning is set to False. "
            "The input domain will be ignored."
        )
    if config.input_domain_poisoning and config.epsilon > 0:
        LOGGER.warning(
            "Both input_domain_poisoning and epsilon are set, "
            "epsilon will be ignored."
        )
    if config.input_domain_poisoning and config.poison_input_domain is None:
        LOGGER.warning(
            "input_domain_poisoning is set to True, but "
            "poison_input_domain is not specified."
        )
    if config.paired_poison and config.label_k_poison != config.k_poison:
        LOGGER.info(
            "Using paired poisoning with different "
            "feature and label poisoning budgets."
        )
    if config.k_private > 0 and agt_type not in ["privacy", "nominal"]:
        LOGGER.warning(
            "k_private is set to %s, but this is not a privacy run.",
            config.k_private,
        )
    if config.k_unlearn > 0 and agt_type not in ["unlearning", "nominal"]:
        LOGGER.warning(
            "k_unlearn is set to %s, but this is not an unlearning run.",
            config.k_unlearn,
        )
    if config.k_poison > 0 and agt_type not in ["poison", "nominal"]:
        LOGGER.warning(
            "k_poison is set to %s, but this is not a poisoning run.",
            config.k_poison,
        )
    if config.label_k_poison > 0 and agt_type not in ["poison", "nominal"]:
        LOGGER.warning(
            "label_k_poison is set to %s, but this is not a poisoning run.",
            config.label_k_poison,
        )
    if config.clip_gamma == float("inf") and agt_type == "privacy":
        LOGGER.error(
            "Gradient clipping parameter must be set "
            "for privacy certified training."
        )
    if (
        max(
            config.k_unlearn,
            config.k_private,
            config.k_poison,
            config.label_k_poison,
        )
        == 0
        and agt_type != "nominal"
    ):
        LOGGER.info(
            "k=0 suffers from numerical instability in the bounds, "
            "consider using k > 0 or dtype double."
        )
