"""Certified Training using Optimization-based Abstract Gradient Training."""

import copy
import logging
from collections.abc import Callable

import gurobipy as gp
import numpy as np
import torch
import torch.utils.data
import os


def get_gurobi_license_params() -> dict:
    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    license_file = os.path.join(root_folder, "gurobi.lic")
    params = {}
    with open(license_file, "r") as f:
        for line in f.readlines():
            if not line.strip() or "=" not in line:
                continue
            key, val = line.strip().split("=")
            if key == "LICENSEID":
                val = int(val)
            params[key] = val

    return params


import abstract_gradient_training as agt
from abstract_gradient_training import AGTConfig
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from abstract_gradient_training_opt import formulations
from abstract_gradient_training_opt import training_utils
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training_opt.formulations import BoundedVar
from abstract_gradient_training_opt.formulations import MIPSequential

LOGGER = logging.getLogger(__name__)
DEVICE = "cpu"

# Where per-sample gradient clipping is applied. "param" (default) clips
# dL/dtheta after backprop, matching propagate_clipping / DP-SGD / AGS. "loss"
# restores the pre-refactor behaviour of clipping dL/dlogit before backprop,
# retained only so the two can be compared without a checkout.
_CLIP_SITE = os.environ.get("AGT_OPT_CLIP_SITE", "param")
# Diagnostic knob. bound_parameters costs TWO solves per SCALAR parameter
# coordinate (min and max), so a d=5 model pays 10 timeouts per concretization.
# When only the SCALING of the envelope is of interest -- e.g. probing what
# governs MIP tractability -- capping the number of tightened coordinates cuts
# that cost proportionally. The cap counts scalars, not parameter tensors: a
# (1,4) weight is 4 coordinates, so capping at 1 slices into the tensor rather
# than bounding all of it. Un-tightened coordinates keep their (sound but loose)
# IBP bounds, so the result stays valid; it is just not uniformly tight. Unset
# means "tighten every coordinate", which is the only setting for real runs.
_MAX_BOUND_COORDS = int(os.environ.get("AGT_OPT_MAX_BOUND_COORDS", "0")) or None


def _slice_bounded(bv, idx):
    """Row-slice a BoundedVar."""
    return formulations.BoundedVar(
        n=bv.n[idx], l=bv.l[idx], u=bv.u[idx], v=bv.v[idx]
    )


def formulate_model_and_data(
    config: OPTConfig,
    ibp_model: BoundedModel,
    data: np.ndarray,
    labels: np.ndarray,
) -> tuple[gp.Model, MIPSequential, dict[str, BoundedVar]]:
    """Formulate the model and data for the optimization-based AGT.

    Args:
        config (OPTConfig): Configuration object for the optimization abstract
            gradient training module.
        ibp_model (BoundedModel): The model to be trained, which is a bounded
            model with interval bounds on the parameters.
        data (np.ndarray): Input data for the training.
        labels (np.ndarray): Labels for the training data.

    Returns:
        tuple[gp.Model, MIPSequential, dict[str, BoundedVar]]: Gurobi model,
            mip formulation of ibp_model, training data variables, training
            label variables.
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if config.loss == "hinge":
        if not ((labels != 1) + (labels != -1)).all():
            labels = 2 * labels - 1
        assert ((labels != 1) + (labels != -1)).all()
    LOGGER.info("Formulating model and data for certified training.")

    env = gp.Env(empty=True)
    try:
        params = get_gurobi_license_params()
        for key, val in params.items():
            env.setParam(key, val)
    except FileNotFoundError:
        pass
    env.start()

    formulation = gp.Model(env=env)
    for key, value in config.gurobi_kwargs.items():
        formulation.setParam(key, value)

    mip_model = MIPSequential.from_bounded_model(formulation, ibp_model, encodings=config.all_encodings)

    if config.k_removed:
        raise NotImplementedError("Certified training with removals is not yet supported.")
    if config.k_substitution:
        # k_substitution: model substitution of k samples
        # Hinge loss: use add_substitution_constraint (label flipping)
        # MSE loss: use add_substitution_constraint_regression (label in domain)

        if config.loss not in ("hinge", "mse", "binary_cross_entropy"):
            raise NotImplementedError(f"Loss '{config.loss}' not supported for k_substitution.")

        if config.input_domain is not None:
            # MIP-based substitution path
            if config.loss == "hinge":
                if config.perturbation_constraint == "input":
                    perturbed_data, perturbed_labels = formulations.functional.add_substitution_constraint(
                        formulation,
                        data,
                        labels,
                        config.k_substitution,
                        input_domain=config.input_domain,
                        vtype_encoding=config.poisoning_encoding,
                    )
                    variables = {
                        "data": perturbed_data,
                        "labels": perturbed_labels,
                    }
                    return formulation, mip_model, variables
                elif config.perturbation_constraint == "gradient":
                    data_variables = formulations.functional.add_substitution_constraint_disjoint(
                        formulation,
                        data,
                        labels,
                        k_substitution=config.k_substitution,
                        input_domain=config.input_domain,
                        n_batches=len(data) // config.batchsize,
                        batchsize=config.batchsize,
                        vtype_encoding=config.poisoning_encoding,
                    )
                    if config.encode_feature_transform:
                        formulations.functional.encode_feature_transform(
                            formulation,
                            data_variables["substituted_features"],
                            config.matmul_encoding,
                        )
                    return formulation, mip_model, data_variables
            elif config.loss == "mse":
                # MSE loss requires label_domain
                if config.label_domain is None:
                    raise ValueError("MSE loss with k_substitution requires label_domain to be set. " "Set label_domain=(y_min, y_max).")
                perturbed_data, perturbed_labels = formulations.functional.add_substitution_constraint_regression(
                    formulation,
                    data,
                    labels,
                    config.k_substitution,
                    input_domain=config.input_domain,
                    label_domain=config.label_domain,
                    vtype_encoding=config.poisoning_encoding,
                )
                variables = {
                    "data": perturbed_data,
                    "labels": perturbed_labels,
                }
                return formulation, mip_model, variables
        else:
            # k_substitution with input_domain=None: Privacy via gradient clipping
            # Create global substitution indicators for MIP-based bound tightening
            assert config.gradient_clipping is not None, "k_substitution with input_domain=None requires gradient_clipping to be set."
            LOGGER.info("Creating global substitution indicators for gradient clipping path.")

            # Total samples across all batches
            n_samples = len(data)

            # Create global substitution indicator variables
            substituted_idx = formulations.functional.add_privacy_substitution_indicators(
                formulation,
                n_samples=n_samples,
                k_substitution=config.k_substitution,
                vtype_encoding=config.poisoning_encoding,
                suffix="",
            )

            variables = {
                "data": formulations.BoundedVar(n=data),
                "labels": formulations.BoundedVar(n=labels),
                "substituted_idx": substituted_idx,  # Global indicators
            }
            return formulation, mip_model, variables
    if config.paired_perturbation:
        if config.loss != "hinge":
            raise NotImplementedError("Only hinge loss is currently supported.")
        if not (config.k_perturbed_features == config.k_perturbed_labels > 0):
            raise ValueError(
                "When using paired perturbation, both "
                "k_perturbed_features and k_perturbed_labels must be set."
                f"Got {config.k_perturbed_features} and "
                f"{config.k_perturbed_labels}."
            )
        perturbed_data, perturbed_labels = formulations.functional.add_paired_poison_constr(
            formulation,
            data,
            2 * labels - 1,
            config.feature_perturbation,
            config.k_perturbed_features,
            poisoning_vtype_encoding=config.poisoning_encoding,
        )
        variables = {
            "data": perturbed_data,
            "labels": perturbed_labels,
        }
        return formulation, mip_model, variables
    elif config.k_perturbed_features and config.k_perturbed_labels:
        if config.loss != "hinge":
            raise NotImplementedError("Only hinge loss is currently supported.")
        raise NotImplementedError("Non-paired perturbation of features and labels is not supported.")
    elif config.k_perturbed_features:
        LOGGER.info("Adding feature perturbation constraints.")
        perturbed_data = formulations.functional.add_feature_poisoning_constr(
            formulation,
            data,
            config.feature_perturbation,
            config.k_perturbed_features,
            poisoning_vtype_encoding=config.poisoning_encoding,
        )
        perturbed_labels = formulations.BoundedVar(n=labels)
        variables = {
            "data": perturbed_data,
            "labels": perturbed_labels,
        }
        return formulation, mip_model, variables
    elif config.k_perturbed_labels:
        LOGGER.info("Adding label perturbation constraints.")
        perturbed_data = formulations.BoundedVar(n=data)
        if config.loss == "hinge":
            perturbed_labels = formulations.functional.add_label_poisoning_constr(
                formulation,
                labels,
                label_k_poison=config.k_perturbed_labels,
                poisoning_vtype_encoding=config.poisoning_encoding,
                label_flipping_encoding=config.label_poisoning_encoding,
            )
        elif config.loss == "mse":
            perturbed_labels = formulations.functional.add_label_poisoning_constr_regression(
                formulation,
                labels,
                epsilon=config.label_perturbation,
                k_poison=config.k_perturbed_labels,
                poisoning_vtype_encoding=config.poisoning_encoding,
            )
        else:
            raise NotImplementedError("Only hinge loss and mse are currently supported.")
        variables = {
            "data": perturbed_data,
            "labels": perturbed_labels,
        }
        return formulation, mip_model, variables
    else:
        raise NotImplementedError("No perturbation constraints specified. " "Please specify either k_perturbed_features or k_perturbed_labels.")


def bound_parameters(
    mip_model: MIPSequential,
    callback: Callable[[gp.Model, int], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Bound the parameters of the mip model.

    Args:
        mip_model (MIPSequential): The mip model with parameters to be bounded.
        callback (Callable[[gp.Model, int], None], optional): A callback
    """
    params_l, params_n, params_u = [], [], []
    formulation = mip_model.formulation
    LOGGER.info("Concretizing model parameters.")
    formulation.update()
    LOGGER.info("Model stats: %s", training_utils.get_gurobi_model_stats(formulation))
    n_done = 0
    for i, param in enumerate(mip_model.get_parameters()):
        size = int(np.prod(param.n.shape))
        take = size if _MAX_BOUND_COORDS is None else max(
            0, min(size, _MAX_BOUND_COORDS - n_done)
        )
        params_n.append(param.n)
        if take == 0:
            # diagnostic mode: fall back to the IBP bounds, which are sound
            LOGGER.info("Skipping parameter %s (AGT_OPT_MAX_BOUND_COORDS)", i)
            params_l.append(param.l)
            params_u.append(param.u)
            continue
        LOGGER.info("Bounding parameter %s (%d/%d coords)", i, take, size)
        if take == size:
            p_l, p_u = training_utils.bound_objective(
                formulation,
                param.v,  # type: ignore
                callback,
            )
            params_l.append(np.maximum(p_l, param.l))
            params_u.append(np.minimum(p_u, param.u))
        else:
            # partial: tighten a prefix of the flattened tensor, keep IBP elsewhere
            p_l, p_u = training_utils.bound_objective(
                formulation,
                param.v.reshape(-1)[:take],  # type: ignore
                callback,
            )
            l_arr, u_arr = param.l.copy().ravel(), param.u.copy().ravel()
            l_arr[:take] = np.maximum(p_l, l_arr[:take])
            u_arr[:take] = np.minimum(p_u, u_arr[:take])
            params_l.append(l_arr.reshape(param.l.shape))
            params_u.append(u_arr.reshape(param.u.shape))
        n_done += take
    return params_l, params_n, params_u


def compute_batch_ibp(
    model: BoundedModel,
    perturbed_data: np.ndarray,
    perturbed_labels: np.ndarray,
    data_slice: slice,
    config: OPTConfig,
    learning_rate: float,
    dataset_val: torch.utils.data.TensorDataset | None,
) -> BoundedModel:
    """Compute the single iteration of the IBP training."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(perturbed_data[data_slice].copy(), dtype=torch.float64),
        torch.tensor(perturbed_labels[data_slice].copy()),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batchsize, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config.batchsize, shuffle=False) if dataset_val is not None else None
    if config.k_substitution:
        # For k_substitution with domain-based MIP + gradient clipping:
        # We use privacy_certified_training with clipping for IBP
        # The MIP handles data perturbation, IBP handles gradient clipping

        if config.gradient_clipping is not None:
            # Privacy model: gradient clipping
            # IBP uses per-batch worst-case k_private, MIP will tighten globally
            # When input_domain is None, MIP has global substitution indicators
            # that can tighten beyond per-batch worst-case
            k_priv = min(config.k_substitution, config.batchsize)

            agt_config = AGTConfig(
                loss=config.loss,
                n_epochs=1,
                learning_rate=learning_rate,
                device=DEVICE,
                k_private=k_priv,
                clip_gamma=config.gradient_clipping,
                log_level="WARNING",
                metadata={"margin": config.metadata.get("margin", 1.0)},
            )
            return agt.privacy_certified_training(model, agt_config, dataloader, dl_val=val_loader)
        elif config.input_domain is not None:
            # Poisoning model: input perturbation without clipping
            input_domain = (
                torch.from_numpy(config.input_domain[0]),
                torch.from_numpy(config.input_domain[1]),
            )
            k_poison = min(config.k_substitution, config.batchsize)
            agt_config = AGTConfig(
                loss=config.loss,
                n_epochs=1,
                learning_rate=learning_rate,
                device=DEVICE,
                k_poison=k_poison,
                label_k_poison=k_poison,
                paired_poison=True,
                poison_input_domain=input_domain,
                log_level="WARNING",
                epsilon=config.feature_perturbation,
                label_epsilon=config.label_perturbation,
                input_domain_poisoning=True,
                metadata={"margin": config.metadata.get("margin", 1.0)},
            )
            return agt.poison_certified_training(model, agt_config, dataloader, dl_val=val_loader)
        else:
            raise ValueError("k_substitution requires either gradient_clipping or input_domain.")
    else:
        agt_config = AGTConfig(
            loss=config.loss,
            n_epochs=1,
            learning_rate=learning_rate,
            device=DEVICE,
            k_poison=min(config.k_perturbed_features, config.batchsize),
            label_k_poison=min(config.k_perturbed_labels, config.batchsize),
            epsilon=config.feature_perturbation,
            label_epsilon=config.label_perturbation,
            log_level="WARNING",
            metadata={"margin": config.metadata.get("margin", 1.0)},
        )
        return agt.poison_certified_training(model, agt_config, dataloader, dl_val=val_loader)


def add_batch_to_formulation(
    formulation: gp.Model,
    mip_model: MIPSequential,
    perturbed_data: dict[str, BoundedVar],
    data_slice: slice,
    batch_no: int,
    config: OPTConfig,
    learning_rate: float,
) -> MIPSequential:
    """Add the constraints for the batch to the formulation."""
    LOGGER.info("Adding batch constraints to the formulation.")

    if config.metadata.get("ellide_model", False):
        assert config.loss == "hinge"
        assert config.perturbation_constraint == "input"
        x = perturbed_data["data"][data_slice]
        y = perturbed_data["labels"][data_slice]
        w, b = mip_model.get_parameters()
        loss, loss_grad = formulations.functional.ellide_linear_and_hinge_loss(
            formulation,
            x,
            y,
            w,
            b,
            relu_encoding=config.metadata.get("ellided_relu_encoding", "bigm"),
            matmul_encoding=config.matmul_encoding,
            suffix=f"ellision_step{mip_model.step_}_training_loss",
            margin=config.metadata.get("margin", 1.0),
        )
        _, dl_dw, dl_db = formulations.functional.ellide_linear_and_hinge_loss_grad(
            formulation,
            x,
            y,
            w,
            b,
            loss_grad,  # type: ignore
            mul_encoding=config.matmul_encoding,
            suffix=f"ellision_step{mip_model.step_}",
        )
        mip_model = mip_model.step(
            learning_rate,
            batchsize=config.batchsize,
            gradient_override=[dl_dw, dl_db],
            suffix=f"_step{mip_model.step_}",
        )
        return mip_model

    if config.perturbation_constraint == "input":
        x = perturbed_data["data"][data_slice]
        y = perturbed_data["labels"][data_slice]
        logits = mip_model.add_forward(x, suffix=f"_forward_step{mip_model.step_}")
        if config.loss == "hinge":
            loss, loss_z = formulations.functional.add_hinge_loss_constr(
                formulation,
                logits,
                y,
                hinge_prod_encoding=config.hinge_loss_prod_encoding,
                relu_encoding=config.relu_encoding,
                suffix=f"_forward_step{mip_model.step_}_training_loss",
                margin=config.metadata.get("margin", 1.0),
            )
            loss_grad = formulations.functional.add_hinge_loss_gradient_constr(
                formulation,
                loss_z,
                y,
                mul_encoding=config.matmul_encoding,
                relu_grad_encoding=config.relu_grad_encoding,
                suffix=f"_backward_step{mip_model.step_}",
            )
        elif config.loss == "mse":
            loss = formulations.functional.add_mse_loss_constr(
                formulation,
                logits,
                y,
                suffix="_training_loss",
            )
            loss_grad = formulations.functional.add_mse_loss_grad_constr(
                formulation,
                logits,
                y,
                reduction="none",
                suffix="_training_loss_grad",
            )
            # NOTE: clipping is applied AFTER add_backward (below), because
            # DP-SGD -- and propagate_clipping, and therefore AGS -- clips
            # dL/dtheta per sample, not dL/dlogit. Clipping here would bound the
            # parameter gradient only by gamma*|x_i|.
            # See AGS_Ablation/FIX_mip_clipping_site.md.
            pass
        elif config.loss == "binary_cross_entropy":
            loss = formulations.functional.add_logistic_loss_constr(
                formulation,
                logits,
                y,
                n_pieces=config.sigmoid_n_pieces,
                suffix=f"_step{mip_model.step_}_training_loss",
            )
            loss_grad = formulations.functional.add_logistic_loss_grad_constr(
                formulation,
                logits,
                y,
                sigmoid_encoding=config.sigmoid_encoding,
                n_pieces=config.sigmoid_n_pieces,
                reduction="none",
                suffix=f"_step{mip_model.step_}_training_loss_grad",
            )
            # clipping is applied AFTER add_backward, as for mse
        else:
            raise NotImplementedError("Only hinge, mse and binary_cross_entropy are supported.")

        if config.gradient_clipping is not None and _CLIP_SITE == "loss":
            # legacy behaviour, retained for comparison only (AGT_OPT_CLIP_SITE=loss)
            if "substituted_idx" in perturbed_data:
                loss_grad = formulations.functional.add_substitution_aware_gradient_clipping(
                    formulation, loss_grad,
                    _slice_bounded(perturbed_data["substituted_idx"], data_slice),
                    config.gradient_clipping, suffix=f"_step{mip_model.step_}",
                )
            else:
                loss_grad = formulations.functional.add_gradient_clipping_constr(
                    formulation, loss_grad, config.gradient_clipping,
                    suffix="_clipped_loss_grad",
                )

        mip_model.add_backward(loss_grad, suffix="_back")

        grad_override = None
        if config.gradient_clipping is not None and _CLIP_SITE == "param":
            per = mip_model.get_parameter_grads_per_sample()   # [(B,out,in), (B,out)]
            clipped = []
            for j, pg in enumerate(per):
                flat_shape = (pg.n.shape[0], int(np.prod(pg.n.shape[1:])))
                flat = formulations.functional.reshape_bounded(pg, flat_shape)
                if "substituted_idx" in perturbed_data:
                    flat = formulations.functional.add_substitution_aware_gradient_clipping(
                        formulation, flat,
                        _slice_bounded(perturbed_data["substituted_idx"], data_slice),
                        config.gradient_clipping,
                        suffix=f"_step{mip_model.step_}_p{j}",
                    )
                else:
                    flat = formulations.functional.add_gradient_clipping_constr(
                        formulation, flat, config.gradient_clipping,
                        suffix=f"_clipped_param_grad_step{mip_model.step_}_p{j}",
                    )
                clipped.append(formulations.functional.reshape_bounded(flat, pg.n.shape))
            grad_override = [
                formulations.functional.sum_over_batch(
                    formulation, c, suffix=f"_step{mip_model.step_}_p{j}")
                for j, c in enumerate(clipped)
            ]

        mip_model = mip_model.step(
            learning_rate,
            batchsize=config.batchsize,
            gradient_override=grad_override,
            suffix=f"_step{mip_model.step_}",
        )
        return mip_model
    elif config.perturbation_constraint == "gradient":
        assert config.metadata.get("shuffle", False) is False
        assert config.loss == "hinge"
        LOGGER.debug("Adding clean forward pass.")
        x_clean = perturbed_data["original_features"][data_slice]
        y_clean = perturbed_data["original_labels"][data_slice]
        subs_idx = perturbed_data["substituted_idx"][data_slice]
        logits_clean = mip_model.add_forward(x_clean, suffix=f"_forward_clean_step{mip_model.step_}")
        loss, loss_z = formulations.functional.add_hinge_loss_constr(
            formulation,
            logits_clean,
            y_clean,
            hinge_prod_encoding=config.hinge_loss_prod_encoding,
            relu_encoding=config.relu_encoding,
            suffix=f"_forward_clean_step{mip_model.step_}_training_loss",
            margin=config.metadata.get("margin", 1.0),
        )
        LOGGER.debug("Adding clean backward pass.")
        loss_grad = formulations.functional.add_hinge_loss_gradient_constr(
            formulation,
            loss_z,
            y_clean,
            mul_encoding=config.matmul_encoding,
            relu_grad_encoding=config.relu_grad_encoding,
            suffix=f"_backward_clean_step{mip_model.step_}",
        )
        loss_grad_masked = formulations.functional.add_masking_constraint(
            formulation,
            loss_grad,
            subs_idx[:, None],  # type: ignore
            negate_mask=True,
            matmul_encoding=config.matmul_encoding,
            suffix="",
        )
        assert loss_grad_masked.shape == loss_grad.shape
        mip_model.add_backward(loss_grad_masked, suffix=f"_back_clean_step{mip_model.step_}")
        grads_clean = mip_model.get_parameter_grads().copy()
        # perturbed data pass
        LOGGER.debug("Adding perturbed forward pass.")
        x_perturbed = perturbed_data["substituted_features"]
        y_perturbed = perturbed_data["substituted_labels"]
        subs_mask_perturbed = perturbed_data["batch_assignment"][batch_no]
        logits_perturbed = mip_model.add_forward(x_perturbed, suffix=f"_forward_perturbed_step{mip_model.step_}")
        _, loss_z_perturbed = formulations.functional.add_hinge_loss_constr(
            formulation,
            logits_perturbed,
            y_perturbed,
            hinge_prod_encoding=config.hinge_loss_prod_encoding,
            relu_encoding=config.relu_encoding,
            suffix=f"_forward_perturbed_step{mip_model.step_}",
            margin=config.metadata.get("margin", 1.0),
        )
        LOGGER.debug("Adding perturbed backward pass.")
        loss_grad_perturbed = formulations.functional.add_hinge_loss_gradient_constr(
            formulation,
            loss_z_perturbed,
            y_perturbed,
            mul_encoding=config.matmul_encoding,
            relu_grad_encoding=config.relu_grad_encoding,
            suffix=f"_backward_perturbed_step{mip_model.step_}",
        )
        loss_grad_perturbed_masked = formulations.functional.add_masking_constraint(
            formulation,
            loss_grad_perturbed,
            subs_mask_perturbed[:, None],  # type: ignore
            negate_mask=False,
            matmul_encoding=config.matmul_encoding,
            suffix="",
        )
        assert loss_grad_perturbed_masked.shape == loss_grad_perturbed.shape
        mip_model.add_backward(
            loss_grad_perturbed_masked,
            suffix=f"_back_perturbed_step{mip_model.step_}",
        )
        grads_perturbed = mip_model.get_parameter_grads().copy()
        # combine the gradients
        LOGGER.debug("Combining gradients.")
        actual_grads = []
        for j, (g_clean, g_perturbed) in enumerate(zip(grads_clean, grads_perturbed)):
            g_var = formulation.addMVar(
                shape=g_clean.n.shape,
                lb=-gp.GRB.INFINITY,
                name=f"actual_grad_step{mip_model.step_}_{j}",
            )
            formulation.addConstr(g_var == g_clean.v + g_perturbed.v)  # type: ignore
            actual_grads.append(
                BoundedVar(
                    n=g_clean.n + g_perturbed.n,
                    l=g_clean.l + g_perturbed.l,
                    u=g_clean.u + g_perturbed.u,
                    v=g_var,  # type: ignore
                )
            )
        LOGGER.debug("Stepping with combined gradients.")
        mip_model = mip_model.step(
            learning_rate,
            batchsize=config.batchsize,
            gradient_override=actual_grads,
            suffix=f"_step{mip_model.step_}",
        )
        return mip_model
    else:
        raise ValueError(f"Unknown perturbation constraint: {config.perturbation_constraint}")


@torch.no_grad()
def certified_training(
    model: torch.nn.Module,
    config: OPTConfig,
    dataset_train: torch.utils.data.TensorDataset,
    dataset_val: torch.utils.data.TensorDataset | None = None,
    solve: bool = True,
) -> tuple[BoundedModel, list[BoundedModel]]:
    """Create the mip formulation of the training pipeline.

    Formulate the given model and training configuration as a gurobi model, then
    use gurobi to bound the model parameter via optimization-based abstract
    gradient training.

    Args:
        model (torch.nn.Module): Pytorch model to train.
        config (OPTConfig): Configuration object for the optimization abstract
            gradient training module.
        dataset_train (torch.utils.data.TensorDataset): Training dataset.
        dataset_val (torch.utils.data.TensorDataset | None, optional):
            Validation dataset.
        solve (bool): Whether to solve the model or return the formulation.

    Returns:
        BoundedModel: The trained model with the final bounds on the parameters.
        list[BoundedModel]: List of concretized models at each
            concretization step.
    """
    # initialize gurobi model, config environment, and logging
    training_utils.log_run_start(config)
    model = model.to(torch.float64)
    concretized_models = []

    # set up the training data
    data, labels = dataset_train.tensors
    data, labels = (
        data.detach().cpu().clone().double().numpy(),
        labels.detach().cpu().clone().double().numpy(),
    )
    if config.loss == "hinge" and ((labels != 1) + (labels != 0)).all():
        labels = 2 * labels - 1
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels), torch.arange(len(data)))
    shuffle = config.metadata.get("shuffle", False)
    generator = torch.Generator().manual_seed(config.metadata.get("seed", 0))
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batchsize,
        shuffle=shuffle,
        generator=generator,
    )

    # set up the mip and ibp models
    ibp_model = IntervalBoundedModel(model, interval_matmul="exact")
    formulation, mip_model, perturbed_data = formulate_model_and_data(config, ibp_model, data, labels)

    # add the training loop
    n_batches = len(dataset_train) // config.batchsize

    LOGGER.info("Adding training loop constraints")
    iter_ = 0
    currently_opt = False
    obbt_frequency = config.metadata.get("obbt_frequency", 1)
    learning_rate_decay = config.metadata.get("lr_decay", 1.0)
    learning_rate = config.learning_rate
    for e in range(config.n_epochs):
        for b, (x, y, idxs) in enumerate(dataloader):
            LOGGER.info("Starting epoch %s, batch %s", e, b)
            currently_opt = False
            # perform the MIP pass
            data_slice = idxs.numpy()
            mip_model = add_batch_to_formulation(
                formulation,
                mip_model,
                perturbed_data,
                data_slice,
                b,
                config,
                learning_rate,
            )
            # add the ibp pass
            ibp_model = compute_batch_ibp(
                ibp_model,
                data,
                labels,
                data_slice,
                config,
                learning_rate,
                dataset_val=dataset_val,
            )
            mip_model.add_parameter_bounds(
                [p.cpu().numpy() for p in ibp_model.param_l],
                [p.cpu().numpy() for p in ibp_model.param_n],
                [p.cpu().numpy() for p in ibp_model.param_u],
            )
            iter_ += 1
            if config.concretization_freq != -1 and iter_ % config.concretization_freq == 0 and iter_ != config.n_epochs * n_batches:
                LOGGER.info("Concretizing model parameters at epoch %s, batch %s", e, b)
                params_l, params_n, params_u = bound_parameters(mip_model)
                ibp_model.apply_parameter_update(params_l, params_n, params_u)
                concretized_models.append(copy.deepcopy(ibp_model))
                formulation, mip_model, perturbed_data = formulate_model_and_data(config, ibp_model, data, labels)
                currently_opt = True
            elif config.obbt and not (e == config.n_epochs - 1 and b == n_batches - 1) and iter_ % obbt_frequency == 0:
                LOGGER.info("Applying OBBT at epoch %s, batch %s", e, b)
                old_timeout = formulation.Params.TimeLimit
                old_mipfocus = formulation.Params.MIPFocus
                formulation.setParam("TimeLimit", config.obbt_timeout)
                formulation.setParam("MIPFocus", 3)
                params_l, params_n, params_u = bound_parameters(mip_model)
                if not config.relax_binaries:
                    # for numerical stability, we add a small epsilon
                    params_l = [p - 0.0 for p in params_l]
                    params_u = [p + 0.0 for p in params_u]
                mip_model.add_parameter_bounds(params_l, params_n, params_u)
                ibp_model.apply_parameter_update(params_l, params_n, params_u)
                formulation.setParam("TimeLimit", old_timeout)
                formulation.setParam("MIPFocus", old_mipfocus)
        learning_rate *= learning_rate_decay

    # bound the parameters of the model
    if not currently_opt and solve:
        LOGGER.info("Concretizing model parameters at the end of training.")
        params_l, params_n, params_u = bound_parameters(mip_model)
        ibp_model.apply_parameter_update(params_l, params_n, params_u)
        concretized_models.append(copy.deepcopy(ibp_model))
    elif not solve:
        concretized_models.append(copy.deepcopy(ibp_model))
        return mip_model, concretized_models  # type: ignore

    LOGGER.info("=================== Finished Certified Training ===================")

    return ibp_model, concretized_models
