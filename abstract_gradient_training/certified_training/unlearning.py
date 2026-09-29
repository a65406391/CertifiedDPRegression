"""Certified unlearning training."""

from __future__ import annotations

import gc
import itertools
import logging
from collections.abc import Iterable

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.certified_training import (
    UnlearningGradientAccumulator,
)
from abstract_gradient_training.certified_training import training_utils
from abstract_gradient_training.configuration import AGTConfig

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def unlearning_certified_training(
    bounded_model: BoundedModel,
    config: AGTConfig,
    dl_train: Iterable,
    dl_val: Iterable | None = None,
) -> BoundedModel:
    """Train the model with unlearning certified bounds.

    Args:
        bounded_model: Bounded version of the pytorch model to train.
        config: Configuration object for the abstract gradient training module.
            See the configuration module for more details.
        dl_train: Iterable of training data that returns(batch, labels) tuples
            at each iteration.
        dl_val: Iterable for the validation data that returns (batch, labels)
            tuples at each iteration. This data is used to log the performance
            of the model at each training iteration.

    Returns:
        BoundedModel: The trained model with the final bounds on the parameters.
    """
    # initialise hyperparameters, model, data, optimizer, logging
    bounded_model.to(config.device)
    optimizer = config.get_bounded_optimizer(bounded_model)
    training_utils.log_run_start(config, agt_type="unlearning")

    # initialise the gradient accumulation class, which handles the logic of
    # accumulating gradients across batch
    # fragments and computing the certified descent direction bounds.
    gradient_accumulator = UnlearningGradientAccumulator(
        config.k_unlearn, bounded_model.param_n
    )

    # construct an iterator for the validation data.
    val_iterator = (
        training_utils.dataloader_cycle(dl_val) if dl_val is not None else None
    )

    # apply dataset-level perturbation
    dl_train = config.dataset_perturbation_callback(dl_train)

    # early stopping flag
    stop_flag = False

    # main training loop
    for epoch in range(1, config.n_epochs + 1):
        # run callbacks and logging
        if stop_flag:
            break
        config.on_epoch_start_callback(bounded_model)
        optimizer.epoch_start()
        LOGGER.info(f"Starting epoch {epoch} of {config.n_epochs}.")

        # iterate over all batches
        for n, (batch, labels) in enumerate(
            training_utils.dataloader_wrapper(dl_train, epoch), 1
        ):
            # run callbacks, logging and check early stopping criteria
            if stop_flag := config.early_stopping_callback(bounded_model):
                break
            config.on_iter_start_callback(bounded_model)
            if val_iterator is not None:
                loss = training_utils.compute_loss(
                    bounded_model, config.get_val_loss_fn(), *next(val_iterator)
                )
                msg = (
                    f"Loss ({config.val_loss}) {loss[0]:.3f} "
                    f"<= {loss[1]:.3f} <= {loss[2]:.3f}"
                )
                if n == 1:
                    LOGGER.info(msg)
                else:
                    LOGGER.debug(f"Batch {n}. " + msg)
            batch, labels = config.batch_perturbation_callback(batch, labels)

            # split the batch into fragments to avoid running out of GPU memory
            batch_fragments = torch.split(batch, config.fragsize, dim=0)
            label_fragments = torch.split(labels, config.fragsize, dim=0)
            for batch_frag, label_frag in zip(batch_fragments, label_fragments):
                # compute nominal gradients
                frag_grads_n = training_utils.compute_batch_gradients(
                    bounded_model, batch_frag, label_frag, config, nominal=True
                )
                # compute weight perturbed gradients
                frag_grads_l, frag_grads_u = (
                    training_utils.compute_batch_gradients(
                        bounded_model,
                        batch_frag,
                        label_frag,
                        config,
                        nominal=False,
                    )
                )
                # apply gradient clipping
                frag_grads_l, frag_grads_n, frag_grads_u = (
                    training_utils.propagate_clipping(
                        frag_grads_l,
                        frag_grads_n,
                        frag_grads_u,
                        config.clip_gamma,
                        config.clip_method,
                    )
                )
                # accumulate the gradients
                gradient_accumulator.add_fragment_gradients(
                    frag_grads_n, frag_grads_l, frag_grads_u
                )
                # the gpu memory allocations at each loop are not always
                # collected, so we'll prompt pytorch to do so
                gc.collect()
                torch.cuda.empty_cache()

            # get the bounds on the descent direction, validate them, and apply
            # the optimizer update
            batchsize = batch.size(0)
            update_l, update_n, update_u = (
                gradient_accumulator.concretize_gradient_update(batchsize)
            )
            interval_arithmetic.validate_interval(
                update_l, update_u, update_n, "final_grad_bounds"
            )
            optimizer.step(update_l, update_n, update_u)
            config.on_iter_end_callback(bounded_model)

            # check for nan and inf values in the bounds
            if not all(
                p.isfinite().all()
                for p in itertools.chain(
                    bounded_model.param_l, bounded_model.param_u
                )
            ):
                LOGGER.warning(
                    f"NaN or Inf detected in bounds at {epoch=}, batch={n}."
                )

    if val_iterator is not None:
        loss = training_utils.compute_loss(
            bounded_model, config.get_val_loss_fn(), *next(val_iterator)
        )
        LOGGER.info(
            f"Final Eval. Loss ({config.val_loss}): {loss[0]:.3f} "
            f"<= {loss[1]:.3f} <= {loss[2]:.3f}"
        )

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info(
        "============== Finished Unlearning Certified Training =============="
    )

    return bounded_model
