"""Utility functions for training models using optimization-based AGT."""

import logging
from collections.abc import Callable

import gurobipy as gp
import numpy as np
import tqdm

from abstract_gradient_training_opt.configuration import OPTConfig

LOGGER = logging.getLogger(__name__)


def bound_objective(
    model: gp.Model,
    objective: gp.MVar | gp.MLinExpr | gp.MQuadExpr,
    callback: Callable[[gp.Model, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize a vector of objectives in both directions.

    Given a gurobi model and a vector / matrix of objectives, compute the
    elementwise minimum and maximum value of the objective over the model.

    Args:
        model (gp.Model): Gurobi model
        objective: Objective to minimize/maximize over, either a gurobi variable
            or expression.
        callback (Callable, optional): Callback function to use during
            optimization. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays, the first
            containing the minimum of each objective and the second containing
            the maximum value.
    """
    model.update()
    l, u = np.zeros(objective.size), np.zeros(objective.size)
    for i in tqdm.trange(objective.size, disable=True):
        obj = objective[np.unravel_index(i, objective.shape)]  # type: ignore
        # lower bound
        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.reset()
        model.optimize(callback)
        try:
            max_vio = model.getAttr("MaxVio")
        except AttributeError:
            max_vio = 0.0
        if max_vio >= model.Params.FeasibilityTol:
            # numerical violations detected - use the dual bound (objBound)
            # which is guaranteed valid even with primal violations
            LOGGER.warning(
                "Numerical violations detected in model (MaxVio=%.2e), "
                "using dual bound for lb %d/%d",
                max_vio,
                i + 1,
                objective.size,
            )
            l[i] = getattr(model, "objBound", -np.inf)
        elif model.status == gp.GRB.OPTIMAL:
            # if model is solved, store the objective value
            l[i] = model.objVal
        elif model.status == gp.GRB.INFEASIBLE:
            # if model is infeasible, store infinity
            LOGGER.warning(
                "Objective infeasbile at %d/%d, setting lower bound to -inf",
                i + 1,
                objective.size,
            )
            l[i] = -np.inf
        else:  # otherwise use the best bound
            l[i] = getattr(model, "objBound", -np.inf)
            if l[i] == -np.inf:
                LOGGER.warning(
                    "Objective %d/%d has no lower bound, setting to -inf",
                    i + 1,
                    objective.size,
                )
        lb_status = model.status
        # upper bound
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.reset()
        model.optimize(callback)
        try:
            max_vio = model.getAttr("MaxVio")
        except AttributeError:
            max_vio = 0.0
        if max_vio >= model.Params.FeasibilityTol:
            # numerical violations detected - use the dual bound (objBound)
            # which is guaranteed valid even with primal violations
            LOGGER.warning(
                "Numerical violations detected in model (MaxVio=%.2e), "
                "using dual bound for ub %d/%d",
                max_vio,
                i + 1,
                objective.size,
            )
            u[i] = getattr(model, "objBound", np.inf)
        elif model.status == gp.GRB.OPTIMAL:
            u[i] = model.objVal
        elif model.status == gp.GRB.INFEASIBLE:
            LOGGER.warning(
                "Objective infeasbile at %d/%d, setting upper bound to inf",
                i + 1,
                objective.size,
            )
            u[i] = np.inf
        else:
            u[i] = getattr(model, "objBound", np.inf)
            if u[i] == np.inf:
                LOGGER.warning(
                    "Objective %d/%d has no upper bound, setting to inf",
                    i + 1,
                    objective.size,
                )
        LOGGER.debug(
            "Objective %d/%d: l=%f (%s), u=%f (%s)",
            i + 1,
            objective.size,
            l[i],
            lb_status,
            u[i],
            model.status,
        )
    return l.reshape(objective.shape), u.reshape(objective.shape)


def get_gurobi_model_stats(model: gp.Model) -> str:
    """Return a string with statistics about the Gurobi model.

    Args:
        model (gp.Model):Gurobi model

    Returns:
        str: Model statistics in a human-readable format.
    """
    return (
        f"Statistics for model {model.ModelName}:\n"
        f"  {'Linear constraint matrix':<30}: {model.NumConstrs} Constrs, "
        f"{model.NumVars} Vars, {model.NumNZs} NZs\n"
        f"  {'Quadratic constraints':<30}: {model.NumQConstrs} QConstrs, "
        f"{model.NumQCNZs} QNZs\n"
        f"  {'SOS constraints':<30}: {model.NumSOS} SOS\n"
        f"  {'General constraints':<30}: {model.NumGenConstrs} GenConstrs\n"
        f"  {'Quadratic objective':<30}: {model.NumQNZs} NZs, "
        f"{model.NumPWLObjVars} PWL \n"
        f"  {'Integer variables':<30}: {model.NumIntVars} IntVars,"
        f"{model.NumBinVars} BinVars\n"
        f"  {'Solve time':<30}: {model.Runtime:.2f}s\n"
        f"  {'Status':<30}: {model.Status}\n"
    )


def log_run_start(config: OPTConfig) -> None:
    """Log the configuration at the start of an AGT run."""
    logging.getLogger("abstract_gradient_training_opt").setLevel(
        config.log_level
    )

    LOGGER.info(
        "=================== Starting Certified Training ==================="
    )
    LOGGER.debug(
        "\tSGD params: n_epochs=%s, learning_rate=%s, batchsize=%s, loss=%s",
        config.n_epochs,
        config.learning_rate,
        config.batchsize,
        config.loss,
    )
    LOGGER.debug("\tPerturbation params:")
    LOGGER.debug(
        "\t\t removed=%s, substitution=%s",
        config.k_removed,
        config.k_substitution,
    )
    LOGGER.debug(
        "\t\t perturbed_features=%s, perturbed_labels=%s",
        config.k_perturbed_features,
        config.k_perturbed_labels,
    )
    LOGGER.debug(
        "\t\t feature_perturbation=%s, label_perturbation=%s",
        config.feature_perturbation,
        config.label_perturbation,
    )
    LOGGER.debug("\t\t paired_perturbation=%s", config.paired_perturbation)

    LOGGER.debug("\tFormulation params:")
    LOGGER.debug(
        "\t\t relax_binaries=%s, relax_bilinear=%s",
        config.relax_binaries,
        config.relax_bilinear,
    )
