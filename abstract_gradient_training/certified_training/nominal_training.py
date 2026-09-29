"""Uncertified training that matches each certified training nominal passes.

The purpose of this function is to provide a nominal model for the certified
training functions, where all the hyperparameter, batch, and training logic
are handled the same way as the actual AGT code.
"""

from __future__ import annotations

import gc
import itertools
import logging
from collections.abc import Iterable

import torch

from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.certified_training import training_utils
from abstract_gradient_training.configuration import AGTConfig

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def uncertified_training(
    bounded_model: BoundedModel,
    config: AGTConfig,
    dl_train: Iterable,
    dl_val: Iterable | None = None,
    dl_aux: Iterable | None = None,
) -> BoundedModel:
    """Train the model with the given config without certification.

    The behaviour should match the behaviour of certified training.
    NOTE: for the nominal training to match the certified training, the
    dataloaders must be seeded in the same way before passing them to this
    function.

    Args:
        bounded_model: Bounded version of the pytorch model to train.
        config: Configuration object (see configuration.py for details)
        dl_train: Iterable for the training data that returns (batch, labels)
            tuples at each iteration.
        dl_val: Iterable for the validation data that returns (batch, labels)
            tuples at each iteration. This data is used to log the performance
            of the model at each training iteration.
        dl_aux: Iterable for auxiliary training data, such as clean data for
            poisoning or public data for privacy. If provided, a batch will be
            taken from both dl_train and dl_aux for each training batch.

    Returns:
        BoundedModel: The trained model.
    """
    # initialise hyperparameters, model, data, optimizer, logging
    bounded_model.to(config.device)
    training_utils.log_run_start(config, agt_type="nominal")
    optimizer = config.get_bounded_optimizer(bounded_model)

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
        optimizer.epoch_start()
        config.on_epoch_start_callback(bounded_model)
        LOGGER.info(f"Starting epoch {epoch} of {config.n_epochs}.")

        # iterate over all batches
        for n, (batch, labels, batch_aux, labels_aux) in enumerate(
            training_utils.dataloader_pair_wrapper(dl_train, dl_aux, epoch), 1
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

            # NOTE: We have a lot of per-sample operations that must be
            # performed on the entire batchsize, with typically very large
            # batchsizes. To conserve memory, we split the batch into fragments
            # and process each fragment separately and accumulate the gradients
            # across fragments using the gradient_accumulator classes.

            # construct variables to accumulate the gradients across fragments
            grads_n = [torch.zeros_like(p) for p in bounded_model.param_n]

            # process the data
            batch_fragments = itertools.chain(
                torch.split(batch, config.fragsize, dim=0),
                torch.split(batch_aux, config.fragsize, dim=0)
                if batch_aux is not None
                else [],
            )
            label_fragments = itertools.chain(
                torch.split(labels, config.fragsize, dim=0),
                torch.split(labels_aux, config.fragsize, dim=0)
                if labels_aux is not None
                else [],
            )
            for batch_frag, label_frag in zip(batch_fragments, label_fragments):
                # compute nominal pass
                frag_grads_n = training_utils.compute_batch_gradients(
                    bounded_model,
                    batch_frag,
                    label_frag,
                    config,
                    nominal=True,
                    poisoned=False,
                )
                # clip the gradients
                frag_grads_n, _, _ = training_utils.propagate_clipping(
                    frag_grads_n,
                    frag_grads_n,
                    frag_grads_n,
                    config.clip_gamma,
                    config.clip_method,
                )
                # accumulate the gradients
                for i in range(len(grads_n)):
                    grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                # the gpu memory allocations at each loop are not always
                # collected, so we'll prompt pytorch to do so
                gc.collect()
                torch.cuda.empty_cache()

            # get the bounds on the descent direction, validate them, and apply
            # the optimizer update
            batchsize = (
                batch.size(0)
                if batch_aux is None
                else batch.size(0) + batch_aux.size(0)
            )
            for i in range(len(grads_n)):
                grads_n[i] /= batchsize
            optimizer.step(grads_n, grads_n, grads_n)
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
        LOGGER.info(f"Final Eval. Loss ({config.val_loss}): {loss[1]:.3f}")

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info(
        "================ Finished Nominal Certified Training ================"
    )

    return bounded_model
