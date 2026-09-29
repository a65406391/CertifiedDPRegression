"""Formulations of functional constraints for certified training."""

import logging

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounded_losses import BoundedAccuracy
from abstract_gradient_training.bounded_losses import BoundedMSELoss
from abstract_gradient_training_opt import interval_arithmetic
from abstract_gradient_training_opt.formulations.variables import BoundedVar

LOGGER = logging.getLogger(__name__)


def add_relu_constr(
    formulation: gp.Model,
    x: BoundedVar,
    *,
    relu_encoding: str,
    suffix: str = "",
) -> tuple[BoundedVar, BoundedVar]:
    """Add the MIP encoding of the function y = relu(x) = max(0, x).

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        relu_encoding (str): The encoding type ("triangle" or "bigm").
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The output variable.
        z (BoundedVar): The binary indicator variable.
    """
    # LOGGER.debug(f"Adding ReLU constraints with {relu_encoding=}")
    # output variable
    y_l, y_n, y_u = np.maximum(x.l, 0), np.maximum(x.n, 0), np.maximum(x.u, 0)
    y_v = formulation.addMVar(
        shape=x.n.shape,
        lb=y_l,
        ub=y_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"relu_output_{suffix}",
    )
    y = BoundedVar(n=y_n, l=y_l, u=y_u, v=y_v)
    if relu_encoding == "complementary":
        formulation.addConstr(y.v >= x.v)
        formulation.addConstr(y.v >= 0)
        formulation.addConstr(y.v * (x.v - y.v) == 0)
        return y, y

    # binary indicator variable
    if relu_encoding == "triangle":
        vtype = gp.GRB.CONTINUOUS
    elif relu_encoding == "bigm":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(f"Invalid relu_encoding type: {relu_encoding}.")

    z_l, z_n, z_u = (
        np.heaviside(x.l, 0),
        np.heaviside(x.n, 0),
        np.heaviside(x.u, 0),
    )
    z_v = formulation.addMVar(
        shape=x.n.shape,
        lb=z_l,
        ub=z_u,
        vtype=vtype,
        name=f"relu_indicator_{suffix}",
    )
    z = BoundedVar(n=z_n, l=z_l, u=z_u, v=z_v)
    # add the three bigm constraints
    formulation.addConstr(y.v >= x.v, name=f"relu_bigm_1_{suffix}")  # type: ignore
    formulation.addConstr(
        y.v <= x.v - x.l * (1 - z.v),  # type: ignore
        name=f"relu_bigm_{suffix}",
    )
    formulation.addConstr(y.v <= x.u * z.v, name=f"relu_bigm_3_{suffix}")  # type: ignore
    return y, z


def add_relu_gradient_constr(
    formulation: gp.Model,
    z: BoundedVar,
    dl_dy: BoundedVar,
    *,
    relu_grad_encoding: str,
    suffix: str = "",
) -> BoundedVar:
    """Add the constraints for the gradient of the relu function.

    y = relu(x) <=> dl/dx = dl/dy * dy/dx = dl/dy * (x > 0)

    Args:
        formulation (gp.Model): The MIP model.
        z (BoundedVar): The binary indicator variable for the ReLU function.
        dl_dy (BoundedVar): The gradient of the loss wrt the output of ReLU.
        relu_grad_encoding (str): Encoding type ("bilinear" or "mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        dl_dx (BoundedVar): The gradient of the loss wrt the input of ReLU.
    """
    # LOGGER.debug(f"Adding ReLU gradient constraints with {relu_grad_encoding=}")
    z_l = z.l if relu_grad_encoding != "complementary" else np.heaviside(z.l, 0)
    z_n = z.n if relu_grad_encoding != "complementary" else np.heaviside(z.n, 0)
    z_u = z.u if relu_grad_encoding != "complementary" else np.heaviside(z.u, 0)
    dl_dx_n = dl_dy.n * z_n
    dl_dx_l, dl_dx_u = interval_arithmetic.propagate_elementwise(
        dl_dy.l, dl_dy.u, z_l, z_u
    )
    if relu_grad_encoding == "complementary":
        dl_dx_v = formulation.addMVar(
            shape=dl_dx_n.shape,
            lb=dl_dx_l,
            ub=dl_dx_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"relu_grad_input_{suffix}",
        )
        dl_dx = BoundedVar(n=dl_dx_n, l=dl_dx_l, u=dl_dx_u, v=dl_dx_v)
        formulation.addConstr((dl_dx_v - dl_dy.v) * z.v == 0)  # type: ignore
        formulation.addConstr(dl_dx.v * z.u >= dl_dx.l * z.v)  # type: ignore
        formulation.addConstr(dl_dx.v * z.l <= dl_dx.u * z.v)  # type: ignore
    elif relu_grad_encoding == "bilinear":
        dl_dx_v = formulation.addMVar(
            shape=dl_dx_n.shape,
            lb=dl_dx_l,
            ub=dl_dx_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"relu_grad_input_{suffix}",
        )
        dl_dx = BoundedVar(n=dl_dx_n, l=dl_dx_l, u=dl_dx_u, v=dl_dx_v)
        formulation.addConstr(
            dl_dx.v == dl_dy.v * z.v,  # type: ignore
            name=f"bilinear_matmul_{suffix}",
        )
    elif relu_grad_encoding == "mccormick":
        dl_dx = add_mccormick_mul_constr(formulation, dl_dy, z, suffix=suffix)
    else:
        raise ValueError(
            f"Invalid relu_grad_encoding type: {relu_grad_encoding}."
        )
    return dl_dx


def add_heaviside_constr(
    formulation: gp.Model,
    x: BoundedVar,
    *,
    heaviside_encoding: str,
    heaviside_quadratic_cuts: int | None = None,
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the function z = heaviside(x) = I(x > 0).

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        heaviside_encoding (str): Encoding type ("quadratic" or "bigm").
        heaviside_quadratic_cuts (int | None): Number of quadratic cuts.
        suffix (str): Suffix for variable names.

    Returns:
        z (BoundedVar): The output
    """
    # Constant case
    if isinstance(x.v, np.ndarray):
        z_n = np.heaviside(x.n, 0)
        return BoundedVar(n=z_n, l=z_n, u=z_n, v=z_n)

    eps = 1e-4
    z_l, z_n, z_u = map(lambda v: np.heaviside(v, 0), (x.l, x.n, x.u))
    if heaviside_encoding in {"bigm_linear", "bigm_binary"}:
        if heaviside_encoding == "bigm_binary":
            vtype = gp.GRB.BINARY
        else:
            vtype = gp.GRB.CONTINUOUS
        z_v = formulation.addMVar(
            shape=x.n.shape,
            lb=z_l,
            ub=z_u,
            vtype=vtype,
            name="heaviside_indicator" + suffix,
        )
        z = BoundedVar(n=z_n, l=z_l, u=z_u, v=z_v)
        formulation.addConstr(x.v <= z_v * x.u)
        formulation.addConstr(x.v >= (1 - z_v) * x.l - eps)
        return z
    elif heaviside_encoding == "all":
        z_v = formulation.addMVar(
            shape=x.n.shape,
            lb=z_l,
            ub=z_u,
            vtype=gp.GRB.BINARY,
            name="heaviside_indicator" + suffix,
        )
        z = BoundedVar(n=z_n, l=z_l, u=z_u, v=z_v)
        formulation.addConstr(x.v <= z_v * x.u + eps)
        formulation.addConstr(x.v >= (1 - z_v) * x.l - eps)
        # for z_v_i, x_var_i in zip(
        #     z_v.reshape(-1).tolist(),
        #     x.v.reshape(-1).tolist(),  # type: ignore
        # ):
        #     formulation.addGenConstrIndicator(
        #         z_v_i, True, x_var_i - eps, gp.GRB.GREATER_EQUAL, eps
        #     )
        #     formulation.addGenConstrIndicator(
        #         z_v_i, False, x_var_i, gp.GRB.LESS_EQUAL, -eps
        #     )
        # formulation.addConstr((2 * z_v - 1) * x.v >= eps)  # type: ignore
        # add valid quadratic inequalities
        x.v.BranchPriority = 20  # type: ignore
        if heaviside_quadratic_cuts == 0:
            return z
        # add equally spaced quadratic inequalities
        if heaviside_quadratic_cuts is None:
            raise ValueError(
                "heaviside_quadratic_cuts must be provided for 'all' encoding."
            )
        x_squared_l, x_squared_u = interval_arithmetic.propagate_square(
            x.l, x.u
        )
        x_squared = formulation.addMVar(
            shape=x.n.shape,
            lb=x_squared_l,
            ub=x_squared_u,
            name="x_squared" + suffix,
        )
        formulation.addConstr(x_squared == x.v * x.v)  # type: ignore
        n_cuts = 1
        boundary = 0.001
        for i in range(x.n.size):
            if x.l[i] < -boundary:
                ls = np.linspace(x.l[i], -boundary, heaviside_quadratic_cuts)
                for l in ls:
                    n_cuts += 1
                    formulation.addConstr(
                        l * l * z.v[i] <= x_squared[i] - 2 * l * x.v[i] + l * l  # type: ignore
                    )
            if x.u[i] > boundary:
                us = np.linspace(boundary, x.u[i], heaviside_quadratic_cuts)
                for u in us:
                    n_cuts += 1
                    formulation.addConstr(
                        u * u * z.v[i] >= -x_squared[i] + 2 * u * x.v[i]  # type: ignore
                    )
        print(f"Added {n_cuts} quadratic cuts for heaviside.")
        return z
    elif heaviside_encoding == "indicator":
        z_v = formulation.addMVar(
            shape=x.n.shape,
            lb=z_l,
            ub=z_u,
            vtype=gp.GRB.BINARY,
            name="heaviside_indicator" + suffix,
        )
        z = BoundedVar(n=z_n, l=z_l, u=z_u, v=z_v)
        for z_v_i, x_var_i in zip(
            z_v.reshape(-1).tolist(),
            x.v.reshape(-1).tolist(),  # type: ignore
        ):
            formulation.addGenConstrIndicator(
                z_v_i, True, x_var_i - eps, gp.GRB.GREATER_EQUAL, eps
            )
            formulation.addGenConstrIndicator(
                z_v_i, False, x_var_i, gp.GRB.LESS_EQUAL, -eps
            )
        return z
    elif heaviside_encoding in {"quadratic_linear", "quadratic_binary"}:
        if heaviside_encoding == "quadratic_binary":
            vtype = gp.GRB.BINARY
        else:
            vtype = gp.GRB.CONTINUOUS
        z_v = formulation.addMVar(
            shape=x.n.shape,
            lb=z_l,
            ub=z_u,
            vtype=vtype,
            name="heaviside_indicator" + suffix,
        )
        z = BoundedVar(n=z_n, l=z_l, u=z_u, v=z_v)
        formulation.addConstr((2 * z_v - 1) * x.v >= eps)  # type: ignore
        return z
    else:
        raise ValueError(f"Unknown encoding '{heaviside_encoding}'. ")


def add_linear_constr(
    formulation: gp.Model,
    x: BoundedVar,
    w: BoundedVar,
    b: BoundedVar,
    *,
    matmul_encoding: str,
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the function y = x @ w.T + b.

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        w (BoundedVar): The weight matrix.
        b (BoundedVar): The bias vector.
        matmul_encoding (str): Matmul encoding ("bilinear" or "mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The output variable.
    """
    # LOGGER.debug(f"Adding linear constraints with {matmul_encoding=}")
    if (
        isinstance(x.v, np.ndarray)
        and isinstance(w.v, np.ndarray)
        and isinstance(b.v, np.ndarray)
    ):
        y_n = x.v @ w.v.T + b.v
        return BoundedVar(n=y_n, l=y_n, u=y_n, v=y_n)

    if matmul_encoding == "bilinear":
        y_n = x.n @ w.n.T + b.n
        y_l, y_u = interval_arithmetic.propagate_affine(
            w.l.T, w.u.T, x.l, x.u, b.l, b.u
        )
        y_v = formulation.addMVar(
            shape=y_n.shape,
            lb=y_l,
            ub=y_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"linear_output_{suffix}",
        )
        y = BoundedVar(n=y_n, l=y_l, u=y_u, v=y_v)
        formulation.addConstr(
            y.v == x.v @ w.v.T + b.v,  # type: ignore
            name=f"bilinear_linear_{suffix}",
        )
        return y
    if matmul_encoding == "mccormick":
        prod = add_mccormick_matmul_constr(
            formulation, x, w.T, suffix=suffix + "_linear"
        )
        y_n = prod.n + b.n
        y_l, y_u = prod.l + b.l, prod.u + b.u
        y_v = formulation.addMVar(
            shape=y_n.shape,
            lb=y_l,
            ub=y_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"linear_output_{suffix}",
        )
        y = BoundedVar(n=y_n, l=y_l, u=y_u, v=y_v)
        formulation.addConstr(
            y.v == prod.v + b.v, name=f"mccormick_linear_{suffix}"
        )  # type: ignore
        return y
    raise ValueError(f"Invalid matmul_encoding type: {matmul_encoding}.")


def add_mccormick_matmul_constr(
    formulation: gp.Model,
    mat_a: BoundedVar,
    mat_b: BoundedVar,
    suffix: str = "",
) -> BoundedVar:
    """Add the McCormick relaxation of the bilinear term y = mat_a @ mat_b.

    Args:
        formulation (gp.Model): The MIP model.
        mat_a (BoundedVar): The first matrix.
        mat_b (BoundedVar): The second matrix.
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The output variable.
    """
    # LOGGER.debug("Adding McCormick matmul constraints")
    assert mat_a.l.ndim == 2 and mat_b.l.ndim == 2
    # add the output variable with bounds
    y_n = mat_a.n @ mat_b.n
    y_l, y_u = interval_arithmetic.propagate_matmul(
        mat_a.l, mat_a.u, mat_b.l, mat_b.u, interval_matmul="exact"
    )
    y_v = formulation.addMVar(
        shape=y_n.shape,
        lb=y_l,
        ub=y_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"mccormick_output_{suffix}",
    )
    y = BoundedVar(n=y_n, l=y_l, u=y_u, v=y_v)
    sum_terms = add_mccormick_mul_constr(
        formulation,
        mat_a[..., None],  # type: ignore
        mat_b[None, ...],  # type: ignore
        suffix=suffix,
    )
    formulation.addConstr(
        y.v == sum_terms.v.sum(1), name=f"mcormick_matmul_{suffix}"
    )  # type: ignore
    return y


def add_mul_constr(
    formulation: gp.Model,
    var_a: BoundedVar,
    var_b: BoundedVar,
    *,
    mul_encoding: str,
    suffix: str = "",
) -> BoundedVar:
    """Add the multiplication constraint prod_s = var_a * var_b.

    Args:
        formulation (gp.Model): The MIP model.
        var_a (BoundedVar): The first variable.
        var_b (BoundedVar): The second variable.
        mul_encoding (str): Multiplication encoding ("bilinear"/"mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        prod_s (BoundedVar): The output variable.
    """
    # LOGGER.debug(f"Adding bilinear multiplication constraints with {mul_encoding=}")
    if mul_encoding == "mccormick":
        return add_mccormick_mul_constr(
            formulation, var_a, var_b, suffix=suffix
        )
    elif mul_encoding == "bilinear":
        prod_s_n = var_a.n * var_b.n
        prod_s_l, prod_s_u = interval_arithmetic.propagate_elementwise(
            var_a.l, var_a.u, var_b.l, var_b.u
        )
        prod_s_v = formulation.addMVar(
            shape=prod_s_n.shape,
            lb=prod_s_l,
            ub=prod_s_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"bilinear_product_{suffix}",
        )
        prod_s = BoundedVar(n=prod_s_n, l=prod_s_l, u=prod_s_u, v=prod_s_v)
        formulation.addConstr(
            prod_s.v == var_a.v * var_b.v,  # type: ignore
            name=f"bilinear_product_constr_{suffix}",
        )
        return prod_s
    else:
        raise ValueError(f"Invalid mul_encoding type: {mul_encoding}.")


def add_mccormick_mul_constr(
    formulation: gp.Model,
    var_a: BoundedVar,
    var_b: BoundedVar,
    suffix: str = "",
):
    """Add the McCormick relaxation of the bilinear term prod_s = var_a * var_b.

    Args:
        formulation (gp.Model): The MIP model.
        var_a (BoundedVar): The first variable.
        var_b (BoundedVar): The second variable.
        suffix (str): Suffix for variable names.

    Returns:
        prod_s (BoundedVar): The output variable.
    """
    # LOGGER.debug("Adding McCormick multiplication constraints")
    s_n = var_a.n * var_b.n
    s_l, s_u = interval_arithmetic.propagate_elementwise(
        var_a.l, var_a.u, var_b.l, var_b.u
    )
    s_v = formulation.addMVar(
        shape=s_n.shape,
        lb=s_l,
        ub=s_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"mccormick_product_{suffix}",
    )
    prod_s = BoundedVar(n=s_n, l=s_l, u=s_u, v=s_v)
    # lower envelope
    formulation.addConstr(
        prod_s.v >= var_a.l * var_b.v + var_a.v * var_b.l - var_a.l * var_b.l,  # type: ignore
        name=f"mccormick_lower_envelope_1_{suffix}",
    )
    formulation.addConstr(
        prod_s.v >= var_a.u * var_b.v + var_a.v * var_b.u - var_a.u * var_b.u,  # type: ignore
        name=f"mccormick_lower_envelope_2_{suffix}",
    )
    # upper envelope
    formulation.addConstr(
        prod_s.v <= var_a.u * var_b.v + var_a.v * var_b.l - var_a.u * var_b.l,  # type: ignore
        name=f"mccormick_upper_envelope_1_{suffix}",
    )
    formulation.addConstr(
        prod_s.v <= var_a.v * var_b.u + var_a.l * var_b.v - var_a.l * var_b.u,  # type: ignore
        name=f"mccormick_upper_envelope_2_{suffix}",
    )
    return prod_s


def add_linear_gradient_constr(
    formulation: gp.Model,
    x: BoundedVar,
    w: BoundedVar,
    dl_dy: BoundedVar,
    *,
    linear_grad_encoding: str,
    suffix: str = "",
) -> tuple[BoundedVar, BoundedVar, BoundedVar]:
    """Add the MIP encoding of the gradient of the function y = x @ w.T + b.

        dl/dx = dl/dy * dy/dx = dl/dy @ w
        dl/dw = (dl/dy).T @ x
        dl/db = (dl/dy).sum(0)

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        w (BoundedVar): The weight matrix.
        dl_dy (BoundedVar): Grad of loss wrt output of linear function.
        linear_grad_encoding (str): Grad encoding ("bilinear" or "mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        dl_dx (BoundedVar): Grad of loss wrt input of linear function.
        dl_dw (BoundedVar): Grad of loss wrt the weight matrix.
        dl_db (BoundedVar): Grad of loss wrt the bias vector.
    """
    # LOGGER.debug(
    #     f"Adding linear gradient constraints with {linear_grad_encoding=}"
    # )
    # gradients wrt the input
    dl_dx_n = dl_dy.n @ w.n
    dl_dx_l, dl_dx_u = interval_arithmetic.propagate_matmul(
        dl_dy.l, dl_dy.u, w.l, w.u, interval_matmul="exact"
    )
    if linear_grad_encoding == "bilinear":
        dl_dx_v = formulation.addMVar(
            shape=dl_dx_n.shape,
            lb=dl_dx_l,
            ub=dl_dx_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"linear_grad_input_{suffix}",
        )
        dl_dx = BoundedVar(n=dl_dx_n, l=dl_dx_l, u=dl_dx_u, v=dl_dx_v)
        formulation.addConstr(
            dl_dx.v == dl_dy.v @ w.v,  # type: ignore
            name=f"linear_grad_{suffix}",
        )
    elif linear_grad_encoding == "mccormick":
        dl_dx = add_mccormick_matmul_constr(
            formulation, dl_dy, w, suffix=suffix + "_linear_grad_input"
        )
    else:
        raise ValueError(
            f"Invalid linear_grad_encoding type: {linear_grad_encoding}."
        )
    # gradients wrt the weight matrix
    dl_dw_n = dl_dy.n.T @ x.n
    dl_dw_l, dl_dw_u = interval_arithmetic.propagate_matmul(
        dl_dy.l.T, dl_dy.u.T, x.l, x.u, interval_matmul="exact"
    )
    if linear_grad_encoding == "bilinear":
        dl_dw_v = formulation.addMVar(
            shape=dl_dw_n.shape,
            lb=dl_dw_l,
            ub=dl_dw_u,
            vtype=gp.GRB.CONTINUOUS,
            name=f"linear_grad_weights_{suffix}",
        )
        dl_dw = BoundedVar(n=dl_dw_n, l=dl_dw_l, u=dl_dw_u, v=dl_dw_v)
        formulation.addConstr(
            dl_dw.v == dl_dy.v.T @ x.v,  # type: ignore
            name=f"linear_grad_weight_{suffix}",
        )
    elif linear_grad_encoding == "mccormick":
        dl_dw = add_mccormick_matmul_constr(
            formulation, dl_dy.T, x, suffix=suffix + "_linear_grad_weight"
        )
    else:
        raise ValueError(
            f"Invalid linear_grad_encoding type: {linear_grad_encoding}."
        )
    # gradients wrt the bias vector
    dl_db_n = dl_dy.n.sum(0)
    dl_db_l, dl_db_u = dl_dy.l.sum(axis=0), dl_dy.u.sum(axis=0)
    dl_db_v = formulation.addMVar(
        shape=dl_db_n.shape,
        lb=dl_db_l,
        ub=dl_db_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"linear_grad_bias_{suffix}",
    )
    dl_db = BoundedVar(n=dl_db_n, l=dl_db_l, u=dl_db_u, v=dl_db_v)
    formulation.addConstr(
        dl_db.v == dl_dy.v.sum(0),  # type: ignore
        name=f"linear_grad_bias_{suffix}",
    )
    return dl_dx, dl_dw, dl_db


def add_per_sample_param_grads(
    formulation: gp.Model,
    x: BoundedVar,
    dl_dy: BoundedVar,
    suffix: str = "",
) -> tuple[BoundedVar, BoundedVar]:
    """Per-sample parameter gradients for y = x @ w.T + b, BEFORE the batch sum.

    add_linear_gradient_constr contracts the batch axis inside `dl_dy.T @ x`, so
    there is no per-sample object left to clip.  DP-SGD (and propagate_clipping,
    and therefore AGS) clips dL/dtheta PER SAMPLE, so the per-sample tensors must
    exist before clipping.  See AGS_Ablation/FIX_mip_clipping_site.md.

        dl_dw_per[i] = outer(dl_dy[i], x[i])     shape (B, out, in)
        dl_db_per[i] = dl_dy[i]                  shape (B, out)

    Requires `x` to be constant (config.input_domain is None).  With a variable
    `x` each outer product is bilinear and would need a McCormick relaxation per
    sample; that is not implemented, and we raise rather than silently return a
    weaker bound.
    """
    if isinstance(x.v, (gp.MVar, gp.MLinExpr, gp.MQuadExpr)):
        raise NotImplementedError(
            "Per-sample parameter gradients require a constant input (i.e. "
            "config.input_domain is None). With a perturbable input each "
            "per-sample outer product is bilinear and needs a McCormick "
            "relaxation per sample, which is not implemented."
        )
    xn = np.asarray(x.n, dtype=float)                     # (B, in), constant
    B, in_f = xn.shape
    out_f = dl_dy.n.shape[1]

    # nominal and interval bounds: dl_dy[i, o] * x[i, j], x constant so the
    # interval endpoints just swap where x is negative.
    per_n = dl_dy.n[:, :, None] * xn[:, None, :]
    a = dl_dy.l[:, :, None] * xn[:, None, :]
    b = dl_dy.u[:, :, None] * xn[:, None, :]
    per_l, per_u = np.minimum(a, b), np.maximum(a, b)

    per_v = formulation.addMVar(
        shape=(B, out_f, in_f), lb=per_l, ub=per_u,
        vtype=gp.GRB.CONTINUOUS, name=f"per_sample_grad_w{suffix}",
    )
    dl_dy_list = dl_dy.v.tolist() if isinstance(dl_dy.v, gp.MVar) else dl_dy.v
    for i in range(B):
        for o in range(out_f):
            gv = dl_dy_list[i][o]
            for j in range(in_f):
                formulation.addConstr(
                    per_v[i, o, j] == float(xn[i, j]) * gv,
                    name=f"per_sample_grad_w_{i}_{o}_{j}{suffix}",
                )
    dl_dw_per = BoundedVar(n=per_n, l=per_l, u=per_u, v=per_v)
    dl_db_per = BoundedVar(n=dl_dy.n, l=dl_dy.l, u=dl_dy.u, v=dl_dy.v)
    return dl_dw_per, dl_db_per


def sum_over_batch(
    formulation: gp.Model, per: BoundedVar, suffix: str = ""
) -> BoundedVar:
    """Contract the leading (batch) axis of a per-sample BoundedVar."""
    tot_n = per.n.sum(0)
    tot_l, tot_u = per.l.sum(0), per.u.sum(0)
    tot_v = formulation.addMVar(
        shape=tot_n.shape, lb=tot_l, ub=tot_u,
        vtype=gp.GRB.CONTINUOUS, name=f"batch_sum{suffix}",
    )
    formulation.addConstr(tot_v == per.v.sum(0), name=f"batch_sum{suffix}")
    return BoundedVar(n=tot_n, l=tot_l, u=tot_u, v=tot_v)


def reshape_bounded(per: BoundedVar, shape) -> BoundedVar:
    """Reshape a BoundedVar (used to flatten (B,out,in) -> (B,out*in) for clipping)."""
    return BoundedVar(n=per.n.reshape(shape), l=per.l.reshape(shape),
                      u=per.u.reshape(shape), v=per.v.reshape(shape))


def add_hinge_loss_constr(
    formulation: gp.Model,
    x: BoundedVar,
    y: BoundedVar,
    *,
    hinge_prod_encoding: str,
    relu_encoding: str,
    suffix: str = "",
    margin: float = 1.0,
) -> tuple[BoundedVar, BoundedVar]:
    """Add the MIP encoding of the hinge loss function.

        loss = max(0, margin - x * y))

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        y (BoundedVar): The target variable ({-1, 1} labels).
        hinge_prod_encoding (str): Product encoding ("bilinear"/"mccormick").
        relu_encoding (str): ReLU encoding ("triangle" or "bigm").
        suffix (str): Suffix for variable names.
        margin (float): The margin parameter.

    Returns:
        l (BoundedVar): The hinge loss variable.
        z (BoundedVar): The binary indicator variable.
    """
    if not np.all((y.n == -1) + (y.n == 1)):
        raise ValueError("Expected binary labels for hinge loss.")

    if isinstance(x.v, np.ndarray) and isinstance(y.v, np.ndarray):
        inter_n = margin - x.n * y.n
        l_n = np.maximum(inter_n, 0)
        z_n = np.heaviside(inter_n, 0)
        l = BoundedVar(n=l_n, l=l_n, u=l_n, v=l_n)
        z = BoundedVar(n=z_n, l=z_n, u=z_n, v=z_n)
        return l, z

    # compute the intermediate expression - x * y
    prod = add_mul_constr(
        formulation, x, y, mul_encoding=hinge_prod_encoding, suffix=suffix
    )
    inter_n = margin - prod.n
    inter_l, inter_u = margin - prod.u, margin - prod.l
    inter_v = margin - prod.v
    inter = BoundedVar(n=inter_n, l=inter_l, u=inter_u, v=inter_v)
    return add_relu_constr(
        formulation, inter, relu_encoding=relu_encoding, suffix=suffix
    )


def add_hinge_loss_gradient_constr(
    formulation: gp.Model,
    z: BoundedVar,
    y: BoundedVar,
    *,
    mul_encoding: str = "bilinear",
    relu_grad_encoding: str = "bilinear",
    suffix: str = "",
) -> BoundedVar:
    """Add the constraints for the gradient of the hinge loss function.

        loss = max(0, i), i = margin - x * y
        dloss/dx = dloss/di * di/dx = dl/di * -y

    Args:
        formulation (gp.Model): The MIP model.
        z (BoundedVar): Binary indicator variable for the hinge loss.
        y (BoundedVar): The target variable ({-1, 1} labels).
        mul_encoding (str): Multiplication encoding ("bilinear"/"mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        dl_dx (BoundedVar): Grad of hinge loss wrt the input variable.
    """
    if relu_grad_encoding == "complementary":
        dl_di = add_relu_gradient_constr(
            formulation,
            z,
            BoundedVar(n=np.ones(z.shape)),  # type: ignore
            relu_grad_encoding=relu_grad_encoding,
            suffix=suffix,
        )
    else:
        dl_di = z
    # compute the gradient of the intermediate term 1 - x * y wrt x
    di_dx_v = formulation.addMVar(
        shape=y.n.shape, vtype=gp.GRB.CONTINUOUS, lb=-y.u, ub=-y.l
    )
    formulation.addConstr(di_dx_v == -y.v)
    di_dx = BoundedVar(n=-y.n, l=-y.u, u=-y.l, v=di_dx_v)
    dl_dx = add_mul_constr(
        formulation, dl_di, di_dx, mul_encoding=mul_encoding, suffix=suffix
    )
    return dl_dx


def add_linear_plus_relu_constr_extended(
    formulation: gp.Model,
    x: BoundedVar,
    W: BoundedVar,
    b: BoundedVar,
    suffix: str = "",
):
    """Add the extended formulation constraints for y = ReLU(W @ x + b).

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        W (BoundedVar): The weight matrix.
        b (BoundedVar): The bias vector.
    """
    if isinstance(W.v, gp.MQuadExpr | gp.LinExpr):
        wv = formulation.addMVar(W.v.shape, lb=W.l, ub=W.u)
        formulation.addConstr(wv == W.v)
        W.v = wv
    if isinstance(x.v, gp.MQuadExpr | gp.LinExpr):
        xv = formulation.addMVar(x.v.shape, lb=x.l, ub=x.u)
        formulation.addConstr(xv == x.v)
        x.v = xv
    if isinstance(b.v, gp.MQuadExpr | gp.LinExpr):
        bv = formulation.addMVar(b.v.shape, lb=b.l, ub=b.u)
        formulation.addConstr(bv == b.v)
        b.v = bv

    # constants for the extended formulation
    ll_minus_1, ul_minus_1 = x.l, x.u
    mll, mul = interval_arithmetic.propagate_matmul(
        x.l, x.u, W.l.T, W.u.T, interval_matmul="exact"
    )
    mll, mul = mll + b.l, mul + b.u
    # define extended variables for previous layer.
    # note that we index h^{l-1, +, i} as h_pos[i]
    # h_neg[i] is implicitly defined as hl_minus_1 - h_pos[i]
    h_pos = formulation.addMVar(
        (x.shape[0], *W.T.shape), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY
    )
    # define variables for next layer
    hl = formulation.addMVar(
        mll.shape,
        lb=np.maximum(mll, 0),
        ub=np.maximum(mul, 0),
        name=f"extended_relu_output_forward_{suffix}",
    )
    zl = formulation.addMVar(
        hl.shape,
        ub=1,
        vtype=gp.GRB.BINARY,
        name=f"relu_indicator_{suffix}",
    )

    # add bigM formulation constraints
    formulation.addConstr(hl >= x.v @ W.v.T + b.v)  # type: ignore
    formulation.addConstr(hl <= x.v @ W.v.T + b.v - mll * (1 - zl))  # type: ignore
    formulation.addConstr(hl <= mul * zl)
    formulation.addConstr(hl >= 0)
    assert (x.v @ W.v.T + b.v).shape == mul.shape  # type: ignore
    formulation.addConstr(
        x.v @ W.v.T + b.v <= mul  # type: ignore
    )  # this is not encoded by the original bigM formulation when u < 0
    formulation.addConstr(x.v @ W.v.T + b.v >= mll)  # type: ignore

    # add extended formulation constraints
    formulation.addConstr(hl == (W.v[:, :, None] * h_pos).sum(1) + b.v * zl)  # type: ignore

    formulation.addConstr(0 <= (W.v[:, :, None] * h_pos).sum(1) + b.v * zl)  # type: ignore
    formulation.addConstr(
        0
        >= (W.v[:, :, None] * (x.v[:, :, None] - h_pos)).sum(1) + b.v * (1 - zl)  # type: ignore
    )

    formulation.addConstr(
        (x.v[:, :, None] - h_pos)  # type: ignore
        <= ul_minus_1[:, :, None] * (1 - zl[:, :, None])  # type: ignore
    )
    formulation.addConstr(
        (x.v[:, :, None] - h_pos)  # type: ignore
        >= ll_minus_1[:, :, None] * (1 - zl[:, :, None])  # type: ignore
    )

    formulation.addConstr(h_pos <= ul_minus_1[:, :, None] * zl[:, :, None])  # type: ignore
    formulation.addConstr(h_pos >= ll_minus_1[:, :, None] * zl[:, :, None])  # type: ignore

    return BoundedVar(
        n=np.maximum(x.n @ W.n.T + b.n, 0),
        l=np.maximum(mll, 0),
        u=np.maximum(mul, 0),
        v=hl,
    ), BoundedVar(
        n=np.heaviside(x.n @ W.n.T + b.n, 0),
        l=np.heaviside(mll, 0),
        u=np.heaviside(mul, 0),
        v=zl,
    )


def ellide_linear_and_hinge_loss(
    formulation: gp.Model,
    inputs: BoundedVar,
    labels: BoundedVar,
    weights: BoundedVar,
    biases: BoundedVar,
    *,
    relu_encoding: str,
    matmul_encoding: str,
    suffix: str = "",
    margin: float = 1.0,
):
    """Add the MIP encoding of the function h = max(0, margin - y_t * y), y = x @ w.T + b.

    By elliding these two layers together, we can potentially introduce a
    tighter formulation.

    Args:
        formulation (gp.Model): The MIP model.
        inputs (BoundedVar): The input variable x.
        labels (BoundedVar): The target variable y_t ({-1, 1} labels).
        weights (BoundedVar): The weight matrix w.
        biases (BoundedVar): The bias vector b.
        relu_encoding (str): ReLU encoding ("extended" or "bigm").
        matmul_encoding (str): Matmul encoding ("bilinear" or "mccormick").
        suffix (str): Suffix for variable names.
        margin (float): The margin parameter.

    Returns:
        y (BoundedVar): The output variable.
        z (BoundedVar): The binary indicator variable.
    """
    if not np.all((labels.n == -1) + (labels.n == 1)):
        raise ValueError("Expected binary labels for hinge loss.")

    print("Adding ellided linear and hinge loss constraints")
    # compute the effective weights and biases to turn the combined loss into
    # a single layer ReLU(x_eff @ w_eff.T + b_eff)
    w_eff = weights
    x_eff_n = inputs.n * (-labels.n)
    x_eff_l, x_eff_u = interval_arithmetic.propagate_elementwise(
        inputs.l, inputs.u, -labels.u, -labels.l
    )
    x_eff_v = inputs.v * (-labels.v)  # type: ignore
    x_eff = BoundedVar(n=x_eff_n, l=x_eff_l, u=x_eff_u, v=x_eff_v)  # type: ignore
    b_eff_n = margin - labels.n * biases.n
    b_eff_l, b_eff_u = interval_arithmetic.propagate_elementwise(
        labels.l, labels.u, biases.l, biases.u
    )
    b_eff_l, b_eff_u = margin - b_eff_u, margin - b_eff_l
    b_eff_v = margin - labels.v * biases.v  # type: ignore
    b_eff = BoundedVar(n=b_eff_n, l=b_eff_l, u=b_eff_u, v=b_eff_v)  # type: ignore

    if relu_encoding == "extended":
        return add_linear_plus_relu_constr_extended(
            formulation, x_eff, w_eff, b_eff, suffix=suffix
        )

    y = add_linear_constr(
        formulation,
        x_eff,
        w_eff,
        b_eff,
        matmul_encoding=matmul_encoding,
        suffix=suffix + "_ellided_linear",
    )
    return add_relu_constr(
        formulation,
        y,
        relu_encoding=relu_encoding,
        suffix=suffix + "_ellided_relu",
    )


def ellide_linear_and_hinge_loss_grad(
    formulation: gp.Model,
    inputs: BoundedVar,
    labels: BoundedVar,
    weights: BoundedVar,
    biases: BoundedVar,
    z: BoundedVar,
    *,
    mul_encoding: str,
    suffix: str = "",
):
    """Get the gradients of the ellided linear layer and loss."""
    print("Adding ellided linear and hinge loss gradient constraints")
    assert ((labels.n == -1) + (labels.n == 1)).all(), (
        "Expected {-1, 1} labels."
    )
    w_eff = weights
    x_eff_n = inputs.n * (-labels.n)
    x_eff_l, x_eff_u = interval_arithmetic.propagate_elementwise(
        inputs.l, inputs.u, -labels.u, -labels.l
    )
    x_eff_v = inputs.v * (-labels.v)  # type: ignore
    x_eff = BoundedVar(n=x_eff_n, l=x_eff_l, u=x_eff_u, v=x_eff_v)  # type: ignore
    # b_eff_n = 1 - labels.n * biases.n
    # b_eff_l, b_eff_u = interval_arithmetic.propagate_elementwise(
    #     labels.l, labels.u, biases.l, biases.u
    # )
    # b_eff_l, b_eff_u = 1 - b_eff_u, 1 - b_eff_l
    # b_eff_v = 1 - labels.v * biases.v  # type: ignore
    # b_eff = BoundedVar(n=b_eff_n, l=b_eff_l, u=b_eff_u, v=b_eff_v)
    _, dl_dw_eff, _ = add_linear_gradient_constr(
        formulation,
        x_eff,
        w_eff,
        z,
        linear_grad_encoding="bilinear",
        suffix=f"_ellided_linear_grad_{suffix}",
    )
    # dl_dx = add_mul_constr(
    #     formulation,
    #     dl_dx_eff,
    #     BoundedVar(
    #         n=-labels.n,
    #         l=-labels.u,
    #         u=-labels.l,
    #         v=-labels.v,  # type: ignore
    #     ),
    #     mul_encoding=mul_encoding,
    #     suffix=f"_ellided_input_grad_{suffix}",
    # )
    dl_db = add_mul_constr(
        formulation,
        z,
        BoundedVar(
            n=-labels.n,
            l=-labels.u,
            u=-labels.l,
            v=-labels.v,  # type: ignore
        ),
        mul_encoding=mul_encoding,
        suffix=f"_ellided_bias_grad_{suffix}",
    )
    dl_db = BoundedVar(
        n=dl_db.n.sum(0),
        l=dl_db.l.sum(0),
        u=dl_db.u.sum(0),
        v=dl_db.v.sum(0),
    )
    dl_dw = dl_dw_eff
    return None, dl_dw, dl_db


def add_clipping_constr(
    formulation: gp.Model,
    x: BoundedVar,
    clip_min: float,
    clip_max: float,
    relu_encoding: str = "bigm",
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the clipping function y = clip(x, min, max).

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        clip_min (float): The minimum clipping value.
        clip_max (float): The maximum clipping value.
        relu_encoding (str): ReLU encoding ("triangle" or "bigm").
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The output variable.
    """
    # LOGGER.debug(
    #     "Adding clipping constraints with "
    #     f"{clip_min=}, {clip_max=}, {relu_encoding=}"
    # )
    if clip_min >= clip_max:
        raise ValueError("clip_min must be less than clip_max.")
    y_l, y_n, y_u = (
        np.clip(x.l, clip_min, clip_max),
        np.clip(x.n, clip_min, clip_max),
        np.clip(x.u, clip_min, clip_max),
    )
    # add constraints for clipping from below [relu(x - min) + min]
    intermediate_1 = formulation.addMVar(
        shape=x.n.shape,
        lb=x.l - clip_min,
        ub=x.u - clip_min,
        vtype=gp.GRB.CONTINUOUS,
        name=f"clipping_intermediate_1_{suffix}",
    )
    intermediate_1 = BoundedVar(
        n=x.n - clip_min, l=x.l - clip_min, u=x.u - clip_min, v=intermediate_1
    )
    formulation.addConstr(
        intermediate_1.v == x.v - clip_min  # type: ignore
    )
    intermediate_2 = add_relu_constr(
        formulation, intermediate_1, relu_encoding=relu_encoding, suffix=suffix
    )[0]
    intermediate_3 = formulation.addMVar(
        shape=x.n.shape,
        lb=intermediate_2.l + clip_min,
        ub=intermediate_2.u + clip_min,
        vtype=gp.GRB.CONTINUOUS,
        name=f"clipping_intermediate_3_{suffix}",
    )
    intermediate_3 = BoundedVar(
        n=intermediate_2.n + clip_min,
        l=intermediate_2.l + clip_min,
        u=intermediate_2.u + clip_min,
        v=intermediate_3,
    )
    formulation.addConstr(intermediate_3.v == intermediate_2.v + clip_min)  # type: ignore

    # add constraints for clipping from above [max - relu(max - x)]
    intermediate_4 = formulation.addMVar(
        shape=x.n.shape,
        lb=clip_max - intermediate_3.u,
        ub=clip_max - intermediate_3.l,
        vtype=gp.GRB.CONTINUOUS,
        name=f"clipping_intermediate_4_{suffix}",
    )
    intermediate_4 = BoundedVar(
        n=clip_max - intermediate_3.n,
        l=clip_max - intermediate_3.u,
        u=clip_max - intermediate_3.l,
        v=intermediate_4,
    )
    formulation.addConstr(
        intermediate_4.v == clip_max - intermediate_3.v  # type: ignore
    )
    intermediate_5 = add_relu_constr(
        formulation, intermediate_4, relu_encoding=relu_encoding, suffix=suffix
    )[0]
    intermediate_6 = formulation.addMVar(
        shape=x.n.shape,
        lb=clip_max - intermediate_5.u,
        ub=clip_max - intermediate_5.l,
        vtype=gp.GRB.CONTINUOUS,
        name=f"clipping_intermediate_6_{suffix}",
    )
    formulation.addConstr(intermediate_6 == clip_max - intermediate_5.v)  # type: ignore
    return BoundedVar(
        n=y_n,
        l=y_l,
        u=y_u,
        v=intermediate_6,
    )


def add_masking_constraint(
    formulation: gp.Model,
    x: BoundedVar,
    mask: BoundedVar,
    negate_mask: bool = False,
    *,
    matmul_encoding: str = "bilinear",
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the masking operation y = x * mask.

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable.
        mask (BoundedVar): The binary mask tensor.
        negate_mask (bool): Whether to negate the mask.
        matmul_encoding (str): Matmul encoding ("bilinear" or "mccormick").
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The output variable.
    """
    # LOGGER.debug(
    #     f"Adding masking constraints with {negate_mask=}, {matmul_encoding=}"
    # )

    if negate_mask:
        if isinstance(mask.v, np.ndarray):
            negated_mask = 1 - mask.v
        elif isinstance(mask.v, gp.MVar):
            negated_mask = formulation.addMVar(
                shape=mask.n.shape,
                lb=1 - mask.u,
                ub=1 - mask.l,
                vtype=gp.GRB.CONTINUOUS,
                name=f"negated_{suffix}",
            )
            formulation.addConstr(
                negated_mask == 1 - mask.v,  # type: ignore
                name=f"negation_constr_{suffix}",
            )
        else:
            raise ValueError("Unexpected type for mask.v")
        mask = BoundedVar(
            n=1 - mask.n,
            l=1 - mask.u,
            u=1 - mask.l,
            v=negated_mask,  # type: ignore
        )

    return add_mul_constr(
        formulation,
        x,
        mask,
        mul_encoding=matmul_encoding,
        suffix=f"masking_{suffix}",
    )


def add_feature_poisoning_constr(
    formulation: gp.Model,
    x: np.ndarray,
    epsilon: float,
    k_poison: int,
    *,
    poisoning_vtype_encoding: str,
) -> BoundedVar:
    """Add constraints for feature poisoning adversary.

    Perturbs input features of k samples by up to epsilon (l_inf norm).

    Args:
        formulation (gp.Model): The MIP model.
        x (np.ndarray): The input features.
        epsilon (float): The perturbation budget.
        k_poison (int): The number of samples to poison.
        poisoning_vtype_encoding (str): Gurobi var type ("linear"/"binary").

    Returns:
        poisoned_features (BoundedVar): The poisoned features.
    """
    LOGGER.debug(
        "Adding feature poisoning constraints with "
        f"{k_poison=}, {epsilon=}, {poisoning_vtype_encoding=}"
    )
    if poisoning_vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif poisoning_vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(
            f"Invalid poisoning_vtype_encoding: {poisoning_vtype_encoding}."
        )
    if k_poison == 0 or epsilon == 0:
        return BoundedVar(n=x)
    if epsilon < 0 or k_poison < 0:
        raise ValueError("Invalid poisoning parameters.")
    if not x.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")
    # add the indicator variable denoting which samples are poisoned
    poisoned_idx = formulation.addMVar(
        shape=x.shape[0], lb=0, ub=1, vtype=vtype, name="poisoned_idx"
    )
    poisoned_idx.BranchPriority = 10
    # add the poisoned features
    poisoned_x_v = formulation.addMVar(
        shape=x.shape,
        lb=x - epsilon,
        ub=x + epsilon,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_features",
    )
    poisoned_x = BoundedVar(n=x, l=x - epsilon, u=x + epsilon, v=poisoned_x_v)
    formulation.addConstr(
        poisoned_x.v <= x + epsilon * poisoned_idx[:, None],  # type: ignore
        name="feature_poisoning_1",
    )
    formulation.addConstr(
        poisoned_x.v >= x - epsilon * poisoned_idx[:, None],  # type: ignore
        name="feature_poisoning_2",
    )
    formulation.addConstr(
        poisoned_idx.sum() <= k_poison,  # type: ignore
        name="poisoning_budget",
    )
    return poisoned_x


def add_label_poisoning_constr_regression(
    formulation: gp.Model,
    y: np.ndarray,
    epsilon: float,
    k_poison: int,
    *,
    poisoning_vtype_encoding: str,
) -> BoundedVar:
    """Add constraints for label poisoning adversary for regression.

    Perturbs input features of k samples by up to epsilon (l_inf norm).

    Args:
        formulation (gp.Model): The MIP model.
        y (np.ndarray): The input labels.
        epsilon (float): The perturbation budget.
        k_poison (int): The number of samples to poison.
        poisoning_vtype_encoding (str): Gurobi var type ("linear"/"binary").

    Returns:
        poisoned_features (BoundedVar): The poisoned features.
    """
    if poisoning_vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif poisoning_vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(
            f"Invalid poisoning_vtype_encoding: {poisoning_vtype_encoding}."
        )
    if k_poison == 0 or epsilon == 0:
        return BoundedVar(n=y)
    if epsilon < 0 or k_poison < 0:
        raise ValueError("Invalid poisoning parameters.")
    if not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")
    # add the indicator variable denoting which samples are poisoned
    poisoned_idx = formulation.addMVar(
        shape=y.shape[0], lb=0, ub=1, vtype=vtype, name="poisoned_idx"
    )
    poisoned_idx.BranchPriority = 10
    # poisoned_idx.Start = np.random.permutation(
    #     np.concatenate([np.ones(k_poison), np.zeros(y.shape[0] - k_poison)])  # type: ignore
    # )
    poisoned_idx.Start = np.zeros(y.shape[0])  # type: ignore
    # add the poisoned labels
    poisoned_y_v = formulation.addMVar(
        shape=y.shape,
        lb=y - epsilon,
        ub=y + epsilon,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_labels",
    )
    poisoned_y = BoundedVar(n=y, l=y - epsilon, u=y + epsilon, v=poisoned_y_v)
    formulation.addConstr(
        poisoned_y.v <= y + epsilon * poisoned_idx[:, None],  # type: ignore
        name="feature_poisoning_1",
    )
    formulation.addConstr(
        poisoned_y.v >= y - epsilon * poisoned_idx[:, None],  # type: ignore
        name="feature_poisoning_2",
    )
    formulation.addConstr(
        poisoned_idx.sum() <= k_poison,  # type: ignore
        name="poisoning_budget",
    )
    return poisoned_y


def add_label_poisoning_constr(
    formulation: gp.Model,
    y: np.ndarray,
    label_k_poison: int,
    *,
    poisoning_vtype_encoding: str,
    label_flipping_encoding: str,
) -> BoundedVar:
    """Add constraints for label poisoning adversary.

    Flips labels of k samples.

    Args:
        formulation (gp.Model): The MIP model.
        y (np.ndarray): Input targets ({-1, 1} labels).
        label_k_poison (int): Number of samples to poison.
        poisoning_vtype_encoding (str): Gurobi var type for poisoned_idx
                                      ("linear" or "binary").
        label_flipping_encoding (str): Method for label poisoning
                                       ("binary" or "quadratic").

    Returns:
        poisoned_labels (BoundedVar): The poisoned labels.
    """
    LOGGER.debug(
        f"Adding label poisoning constraints with {label_k_poison=}, "
        f"{poisoning_vtype_encoding=}, {label_flipping_encoding=}"
    )
    if poisoning_vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif poisoning_vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(
            f"Invalid poisoning_vtype_encoding: {poisoning_vtype_encoding}."
        )

    if label_k_poison == 0:
        return BoundedVar(n=y)
    if label_k_poison < 0:
        raise ValueError("Invalid poisoning parameters.")
    if not np.all((y == -1) + (y == 1)):
        if np.all((y == 0) + (y == 1)):
            y = 2 * y - 1  # convert to {-1, 1} labels
        else:
            raise ValueError("Expected -1, +1 labels.")
    if not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")

    if label_flipping_encoding == "type_1":
        poisoned_idx = formulation.addMVar(
            shape=(y.shape[0], 1), lb=0, ub=1, vtype=vtype, name="poisoned_idx"
        )
        poisoned_idx.BranchPriority = 10
        poisoned_y_v = formulation.addMVar(
            shape=y.shape,
            lb=-1,
            ub=1,
            vtype=gp.GRB.CONTINUOUS,
            name="poisoned_labels",
        )
        formulation.addConstr(poisoned_y_v == -(2 * poisoned_idx - 1) * y)
        formulation.addConstr(
            poisoned_idx.sum() <= label_k_poison, name="poisoning_budget"
        )
        return BoundedVar(
            n=y, l=-np.ones_like(y), u=np.ones_like(y), v=poisoned_y_v
        )
    elif label_flipping_encoding == "type_2":
        y_01 = (y + 1) // 2  # convert to {0, 1} labels
        poisoned_idx = formulation.addMVar(
            shape=(y.shape[0], 1), lb=0, ub=1, vtype=vtype, name="poisoned_idx"
        )
        poisoned_idx.BranchPriority = 10
        poisoned_y_v = formulation.addMVar(
            shape=y.shape, lb=0, ub=1, vtype=vtype
        )
        formulation.addConstr(poisoned_y_v <= y_01 + poisoned_idx)
        formulation.addConstr(poisoned_y_v >= y_01 - poisoned_idx)
        poisoned_y_pm1 = formulation.addMVar(
            shape=y.shape,
            lb=-1,
            ub=1,
            vtype=gp.GRB.CONTINUOUS,
            name="poisoned_labels",
        )
        formulation.addConstr(poisoned_y_pm1 == 2 * poisoned_y_v - 1)
        formulation.addConstr(
            poisoned_idx.sum() <= label_k_poison, name="poisoning_budget"
        )
        return BoundedVar(
            n=y, l=-np.ones_like(y), u=np.ones_like(y), v=poisoned_y_pm1
        )
    elif label_flipping_encoding == "type_3":
        # a binary representation of the poisoned label
        poisoned_y_v_binary_repr = formulation.addMVar(
            shape=y.shape,
            lb=0,
            ub=1,
            vtype=vtype,
            name="poisoned_labels",
        )
        poisoned_y_v_binary_repr.BranchPriority = 10
        poisoned_y = 2 * poisoned_y_v_binary_repr - 1
        formulation.addConstr(
            (poisoned_y * y).sum() >= y.shape[0] - 2 * label_k_poison,
            name="poisoning_budget_alt",
        )
        return BoundedVar(
            n=y, l=-np.ones_like(y), u=np.ones_like(y), v=poisoned_y
        )
    raise ValueError(f"Invalid encoding: {label_flipping_encoding}.")


def add_paired_poison_constr(
    formulation: gp.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    k_poison: int,
    *,
    poisoning_vtype_encoding: str,
) -> tuple[BoundedVar, BoundedVar]:
    """Add constraints for a feature + label poisoning attack.

    Perturbs features of k samples by up to epsilon (l_inf norm)
    and flips labels of (the same) k samples.

    Args:
        formulation (gp.Model): The MIP model.
        x (np.ndarray): The input features.
        y (np.ndarray): The input targets ({-1, 1} labels).
        epsilon (float): The perturbation budget.
        k_poison (int): The number of samples to poison.
        poisoning_vtype_encoding (str): Gurobi var type for poisoned_idx
                                      ("linear" or "binary").

    Returns:
        poisoned_features (BoundedVar): The poisoned features.
        poisoned_labels (BoundedVar): The poisoned labels.
    """
    # LOGGER.debug(
    #     "Adding paired feature + label poisoning constraints with "
    #     f"{k_poison=}, {epsilon=}, {poisoning_vtype_encoding=}"
    # )
    if k_poison == 0:
        return BoundedVar(n=x), BoundedVar(n=y)
    if epsilon < 0 or k_poison < 0:
        raise ValueError("Invalid poisoning parameters.")
    if not x.ndim == 2 or not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")

    if poisoning_vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif poisoning_vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(
            f"Invalid poisoning_vtype_encoding: {poisoning_vtype_encoding}."
        )

    # add the indicator variable denoting which samples are poisoned
    poisoned_idx = formulation.addMVar(
        shape=x.shape[0], lb=0, ub=1, vtype=vtype, name="poisoned_idx"
    )
    poisoned_idx.BranchPriority = 1000000
    formulation.addConstr(
        poisoned_idx.sum() <= k_poison, name="poisoning_budget"
    )  # type: ignore

    # add the poisoned features
    poisoned_x_v = formulation.addMVar(
        shape=x.shape,
        lb=x - epsilon,
        ub=x + epsilon,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_features",
    )
    poisoned_x = BoundedVar(n=x, l=x - epsilon, u=x + epsilon, v=poisoned_x_v)
    formulation.addConstr(
        poisoned_x.v <= x + epsilon * poisoned_idx[:, None],  # type: ignore
        name="paired_poison_1",
    )
    formulation.addConstr(
        poisoned_x.v >= x - epsilon * poisoned_idx[:, None],  # type: ignore
        name="paired_poison_2",
    )

    # add the poisoned labels
    poisoned_y_v = formulation.addMVar(
        shape=y.shape,
        lb=-1,
        ub=1,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_labels",
    )
    poisoned_y = BoundedVar(n=y, l=-1, u=1, v=poisoned_y_v)
    formulation.addConstr(
        poisoned_y_v == -(2 * poisoned_idx[:, None] - 1) * y,  # type: ignore
        name="paired_poison_3",
    )

    return poisoned_x, poisoned_y


def add_substitution_constraint(
    formulation: gp.Model,
    x: np.ndarray,
    y: np.ndarray,
    k_substitution: int,
    input_domain: tuple[np.ndarray, np.ndarray] | None,
    *,
    vtype_encoding: str,
) -> tuple[BoundedVar, BoundedVar]:
    """Add constraints for data substitution.

    Allows for up to k substitutions in the dataset, where the substituted
    features can take any value within the provided bounds [x_l, x_u], and the
    labels of the substituted samples are flipped.

    Args:
        formulation (gp.Model): The MIP model.
        x (np.ndarray): The input features.
        y (np.ndarray): The input targets ({-1, 1} labels).
        k_substitution (int): The number of samples to substitute.
        input_domain (tuple | None): The (min, max) input bounds.
        vtype_encoding (str): Gurobi var type ("linear" or "binary").

    Returns:
        poisoned_features (BoundedVar): The poisoned features.
        poisoned_labels (BoundedVar): The poisoned labels.
    """
    # LOGGER.debug(
    #     "Adding data substitution constraints with "
    #     f"{k_substitution=}, {vtype_encoding=}"
    # )
    if k_substitution == 0:
        return BoundedVar(n=x), BoundedVar(n=y)
    if k_substitution < 0:
        raise ValueError("Invalid formulation parameters.")
    if not x.ndim == 2 or not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")

    if vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(f"Invalid vtype_encoding: {vtype_encoding}.")

    # add the indicator variable denoting which samples are substituted
    substituted_idx = formulation.addMVar(
        shape=x.shape[0], lb=0, ub=1, vtype=vtype, name="subs_idx"
    )
    substituted_idx.BranchPriority = 10
    formulation.addConstr(
        substituted_idx.sum() <= k_substitution, name="paired_privacy_budget"
    )  # type: ignore

    if input_domain is not None:
        x_l, x_u = input_domain
        # add the poisoned features
        substituted_x_v = formulation.addMVar(
            shape=x.shape,
            lb=x_l,
            ub=x_u,
            vtype=gp.GRB.CONTINUOUS,
            name="poisoned_features",
        )
        epsilon_upper = np.tile(x_u[np.newaxis, :], (x.shape[0], 1)) - x
        epsilon_lower = x - np.tile(x_l[np.newaxis, :], (x.shape[0], 1))
        formulation.addConstr(
            substituted_x_v <= x + epsilon_upper * substituted_idx[:, None],  # type: ignore
        )
        formulation.addConstr(
            substituted_x_v >= x - epsilon_lower * substituted_idx[:, None],  # type: ignore
        )
        substituted_x = BoundedVar(
            n=x,
            l=np.tile(x_l[np.newaxis, :], (x.shape[0], 1)),
            u=np.tile(x_u[np.newaxis, :], (x.shape[0], 1)),
            v=substituted_x_v,
        )
    else:
        x_l = -x.max() * 1e5  # some large bounds
        x_u = x.max() * 1e5
        # add the poisoned features without bounds
        substituted_x_v = formulation.addMVar(
            shape=x.shape,
            lb=x_l,
            ub=x_u,
            vtype=gp.GRB.CONTINUOUS,
            name="poisoned_features",
        )
        substituted_x = BoundedVar(n=x, l=x_l, u=x_u, v=substituted_x_v)
        for i in range(x.shape[1]):
            formulation.addGenConstrIndicator(
                substituted_idx,
                0,
                substituted_x_v[:, i],
                gp.GRB.EQUAL,
                x[:, i],
                name="substitution_no_change",
            )
    # add the substituted labels
    substituted_y_v = formulation.addMVar(
        shape=y.shape,
        lb=-1,
        ub=1,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_labels",
    )
    substituted_y = BoundedVar(n=y, l=-1, u=1, v=substituted_y_v)
    formulation.addConstr(
        substituted_y_v == -(2 * substituted_idx[:, None] - 1) * y,  # type: ignore
    )

    return substituted_x, substituted_y


def add_substitution_constraint_regression(
    formulation: gp.Model,
    x: np.ndarray,
    y: np.ndarray,
    k_substitution: int,
    input_domain: tuple[np.ndarray, np.ndarray],
    label_domain: tuple[float, float],
    *,
    vtype_encoding: str,
) -> tuple[BoundedVar, BoundedVar]:
    """Add constraints for data substitution in regression setting.

    Allows for up to k substitutions in the dataset, where the substituted
    features can take any value within input_domain, and the substituted
    labels can take any value within label_domain.

    Args:
        formulation (gp.Model): The MIP model.
        x (np.ndarray): The input features.
        y (np.ndarray): The input targets (continuous labels).
        k_substitution (int): The number of samples to substitute.
        input_domain (tuple): The (min, max) input bounds for features.
        label_domain (tuple): The (min, max) label bounds for regression.
        vtype_encoding (str): Gurobi var type ("linear" or "binary").

    Returns:
        substituted_features (BoundedVar): The substituted features.
        substituted_labels (BoundedVar): The substituted labels.
    """
    if k_substitution == 0:
        return BoundedVar(n=x), BoundedVar(n=y)
    if k_substitution < 0:
        raise ValueError("Invalid formulation parameters.")
    if not x.ndim == 2 or not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")

    if vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(f"Invalid vtype_encoding: {vtype_encoding}.")

    # add the indicator variable denoting which samples are substituted
    substituted_idx = formulation.addMVar(
        shape=x.shape[0], lb=0, ub=1, vtype=vtype, name="subs_idx_reg"
    )
    substituted_idx.BranchPriority = 10
    formulation.addConstr(
        substituted_idx.sum() <= k_substitution, name="substitution_budget"
    )

    # Feature substitution (same as classification)
    x_l, x_u = input_domain
    substituted_x_v = formulation.addMVar(
        shape=x.shape,
        lb=x_l,
        ub=x_u,
        vtype=gp.GRB.CONTINUOUS,
        name="substituted_features_reg",
    )
    epsilon_upper = np.tile(x_u[np.newaxis, :], (x.shape[0], 1)) - x
    epsilon_lower = x - np.tile(x_l[np.newaxis, :], (x.shape[0], 1))
    formulation.addConstr(
        substituted_x_v <= x + epsilon_upper * substituted_idx[:, None],
    )
    formulation.addConstr(
        substituted_x_v >= x - epsilon_lower * substituted_idx[:, None],
    )
    substituted_x = BoundedVar(
        n=x,
        l=np.tile(x_l[np.newaxis, :], (x.shape[0], 1)),
        u=np.tile(x_u[np.newaxis, :], (x.shape[0], 1)),
        v=substituted_x_v,
    )

    # Label substitution for regression
    # When substituted_idx=0: label = original y
    # When substituted_idx=1: label can be anything in [y_min, y_max]
    y_min, y_max = label_domain
    substituted_y_v = formulation.addMVar(
        shape=y.shape,
        lb=y_min,
        ub=y_max,
        vtype=gp.GRB.CONTINUOUS,
        name="substituted_labels_reg",
    )
    # Constraints: when not substituted, y_new = y
    # Upper bound: substituted_y <= y + (y_max - y) * idx
    # Lower bound: substituted_y >= y - (y - y_min) * idx
    formulation.addConstr(
        substituted_y_v <= y + (y_max - y) * substituted_idx[:, None],
        name="label_sub_upper",
    )
    formulation.addConstr(
        substituted_y_v >= y - (y - y_min) * substituted_idx[:, None],
        name="label_sub_lower",
    )
    substituted_y = BoundedVar(
        n=y,
        l=np.full_like(y, y_min),
        u=np.full_like(y, y_max),
        v=substituted_y_v,
    )

    return substituted_x, substituted_y


def encode_feature_transform(
    formulation: gp.Model, features: BoundedVar, mul_encoding: str
) -> None:
    """Encode constraints for polynomial feature transformation.

    Args:
        formulation (gp.Model): The MIP model.
        features (BoundedVar): The input features.
        mul_encoding (str): Multiplication encoding ("bilinear"/"mccormick").
    """
    # for now just hard code the polynomial feature transform we use in
    # the halfmoons experiments
    if features.shape[-1] != 9:
        raise ValueError("Expected 9 input features for hardcoded transform.")
    x1 = features[:, 0]  # type: ignore
    x2 = features[:, 1]  # type: ignore

    # add variables for the polynomial features
    x1_x1 = add_mul_constr(
        formulation, x1, x1, mul_encoding=mul_encoding, suffix="_poly_x1"
    )
    x2_x2 = add_mul_constr(
        formulation, x2, x2, mul_encoding=mul_encoding, suffix="_poly_x2"
    )
    x1_x2 = add_mul_constr(
        formulation, x1, x2, mul_encoding=mul_encoding, suffix="_poly_x1x2"
    )
    x1_x1_x1 = add_mul_constr(
        formulation, x1_x1, x1, mul_encoding=mul_encoding, suffix="_poly_x1x1"
    )
    x1_x1_x2 = add_mul_constr(
        formulation, x1_x1, x2, mul_encoding=mul_encoding, suffix="_poly_x1x2"
    )
    x1_x2_x2 = add_mul_constr(
        formulation, x1, x2_x2, mul_encoding=mul_encoding, suffix="_poly_x2x2"
    )
    x2_x2_x2 = add_mul_constr(
        formulation, x2_x2, x2, mul_encoding=mul_encoding, suffix="_poly_x2x2"
    )
    # encode these features powers
    # [[1 0], [0 1], [2 0], [1 1], [0 2], [3 0], [2 1], [1 2], [0 3]]
    formulation.addConstr(features.v[:, 2] == x1_x1.v, name="poly_feat_3")  # type: ignore
    formulation.addConstr(features.v[:, 3] == x1_x2.v, name="poly_feat_4")  # type: ignore
    formulation.addConstr(features.v[:, 4] == x2_x2.v, name="poly_feat_5")  # type: ignore
    formulation.addConstr(features.v[:, 5] == x1_x1_x1.v, name="poly_feat_6")  # type: ignore
    formulation.addConstr(features.v[:, 6] == x1_x1_x2.v, name="poly_feat_7")  # type: ignore
    formulation.addConstr(features.v[:, 7] == x1_x2_x2.v, name="poly_feat_8")  # type: ignore
    formulation.addConstr(features.v[:, 8] == x2_x2_x2.v, name="poly_feat_9")  # type: ignore


def add_substitution_constraint_disjoint(
    formulation: gp.Model,
    x: np.ndarray,
    y: np.ndarray,
    k_substitution: int,
    input_domain: tuple[np.ndarray, np.ndarray] | None,
    n_batches: int,
    batchsize: int,
    *,
    vtype_encoding: str,
) -> dict[str, BoundedVar]:
    """Add constraints for data substitution.

    An alternative encoding that encodes both the clean and substituted data
    as separate variables with separate passes through the network, where the
    perturbation budget can then be applied in the gradient space.

    Allows for up to k substitutions in the dataset, where the substituted
    features can take any value within the provided bounds [x_l, x_u], and the
    labels of the substituted samples are flipped.

    Args:
        formulation (gp.Model): The MIP model.
        x (np.ndarray): The input features.
        y (np.ndarray): The input targets ({-1, 1} labels).
        k_substitution (int): The number of samples to substitute.
        input_domain (tuple | None): The (min, max) input bounds.
        n_batches (int): The number of batches to split the data into.
        batchsize (int): The batch size.
        vtype_encoding (str): Gurobi var type ("linear" or "binary").

    Returns:
        dict[str, BoundedVar]: A dictionary containing:
         - original_features (BoundedVar): The original features.
         - original_labels (BoundedVar): The original labels.
         - substituted_features (BoundedVar): The substituted features.
         - substituted_labels (BoundedVar): The substituted labels.
         - substituted_idx (BoundedVar): The binary indicator of substitutions.
         - batch_assigment (BoundedVar): The assignment matrix.
    """
    print("Adding disjoint substitution constraints...")
    # LOGGER.debug(
    #     "Adding data substitution constraints with "
    #     f"{k_substitution=}, {vtype_encoding=}, {n_batches=}"
    # )
    if k_substitution <= 0:
        raise ValueError("k_substitution must be positive.")
    if input_domain is None:
        raise ValueError("input_domain must be provided.")
    if not x.ndim == 2 or not y.ndim == 2:
        raise ValueError("Unexpected shapes in labels or features.")

    if vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(f"Invalid vtype_encoding: {vtype_encoding}.")

    variables = {}
    # original features and labels
    variables["original_features"] = BoundedVar(n=x)
    variables["original_labels"] = BoundedVar(n=y)

    # add the indicator variable denoting which samples are substituted
    substituted_idx = formulation.addMVar(
        shape=x.shape[0], lb=0, ub=1, vtype=vtype, name="subs_idx"
    )
    substituted_idx.BranchPriority = 10
    formulation.addConstr(
        substituted_idx.sum() <= k_substitution, name="privacy_budget"
    )  # type: ignore
    variables["substituted_idx"] = BoundedVar(
        n=np.zeros(x.shape[0]),
        l=np.zeros(x.shape[0]),
        u=np.ones(x.shape[0]),
        v=substituted_idx,
    )

    # add the variables for the substituted data
    x_l, x_u = input_domain
    x_n = x[0]
    substituted_features = formulation.addMVar(
        shape=(k_substitution, *x.shape[1:]),
        lb=x_l,
        ub=x_u,
        vtype=gp.GRB.CONTINUOUS,
        name="poisoned_features",
    )
    # substituted_features.Start = (x_u + x_l).reshape(-1, *x_u.shape) / 2  # type: ignore
    substituted_labels_01 = formulation.addMVar(
        shape=(k_substitution, *y.shape[1:]),
        lb=0,
        ub=1,
        vtype=vtype,
        name="poisoned_labels",
    )
    substituted_labels_01.Start = 0
    substituted_labels_hinge = formulation.addMVar(
        shape=(k_substitution, *y.shape[1:]),
        lb=-1,
        ub=1,
        vtype=gp.GRB.CONTINUOUS,
        name="substituted_labels_hinge",
    )
    formulation.addConstr(
        substituted_labels_hinge == 2 * substituted_labels_01 - 1,  # type: ignore
    )
    variables["substituted_features"] = BoundedVar(
        n=np.tile(x_n[np.newaxis, :], (k_substitution, 1)),
        l=np.tile(x_l[np.newaxis, :], (k_substitution, 1)),
        u=np.tile(x_u[np.newaxis, :], (k_substitution, 1)),
        v=substituted_features,
    )
    variables["substituted_labels"] = BoundedVar(
        n=np.tile(1, (k_substitution, *y.shape[1:])),
        l=-np.ones((k_substitution, *y.shape[1:])),
        u=np.ones((k_substitution, *y.shape[1:])),
        v=substituted_labels_hinge,
    )

    # add the variable that assigns each substituted sample to a batch
    batch_assignment = formulation.addMVar(
        shape=(n_batches, k_substitution),
        lb=0,
        ub=1,
        vtype=vtype,
        name="batch_assignment",
    )
    batch_assignment.Start = 0
    # each k_substituted point is assigned to at most one batch
    formulation.addConstr(
        batch_assignment.sum(axis=0) <= 1, name="batch_assignment_sum"
    )
    for b in range(n_batches):
        # the number of added points per batch must equal the number of
        # removed points
        formulation.addConstr(
            substituted_idx[b * batchsize : (b + 1) * batchsize].sum()
            == batch_assignment[b, :].sum(),  # type: ignore
            name=f"batch_assignment_balance_{b}",
        )
    variables["batch_assignment"] = BoundedVar(
        n=np.zeros((n_batches, k_substitution)),
        l=np.zeros((n_batches, k_substitution)),
        u=np.ones((n_batches, k_substitution)),
        v=batch_assignment,
    )
    return variables


def add_accuracy_loss_constr(
    formulation: gp.Model,
    x: BoundedVar,
    y_true: np.ndarray,
    *,
    accuracy_method_encoding: str,
    heaviside_encoding: str,
    heaviside_quadratic_cuts: int | None = None,
    suffix: str = "",
) -> BoundedVar:
    """Add the accuracy loss for the binary classification task.

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The input variable (model predictions/logits).
        y_true (np.ndarray): The target variable ({0, 1} labels).
        accuracy_method_encoding (str): Accuracy encoding ("bigm"/"quadratic").
        heaviside_encoding (str): Encoding for internal Heaviside constraint.
        heaviside_quadratic_cuts (float | None): Number of quadratic cuts
                                                for Heaviside.
        suffix (str): Suffix for variable names.

    Returns:
        l (BoundedVar): The accuracy loss variable.
    """
    if "clipped" in accuracy_method_encoding:
        x = add_clipping_constr(
            formulation,
            x,
            clip_min=-0.1,
            clip_max=0.1,
            suffix=f"_acc_clip_{suffix}",
        )
        accuracy_method_encoding = accuracy_method_encoding.replace(
            "_clipped", ""
        )
    # z is the binarized prediction (x > 0)
    z_pred = add_heaviside_constr(
        formulation,
        x,
        heaviside_encoding=heaviside_encoding,
        heaviside_quadratic_cuts=heaviside_quadratic_cuts,
        suffix=f"_acc_{suffix}",
    )
    z_pred.v.BranchPriority = 11  # type: ignore
    # add the accuracy variable
    l_n = (z_pred.n == y_true).astype(np.float32).mean()
    # use original AGT code to propagate accuracy bounds (requires torch)
    l_l, l_u = BoundedAccuracy().bound_forward(
        torch.tensor(z_pred.l), torch.tensor(z_pred.u), torch.tensor(y_true)
    )
    l_l, l_u = l_l.detach().numpy(), l_u.detach().numpy()
    l_v = formulation.addMVar(
        shape=l_n.shape,
        lb=l_l,
        ub=l_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"accuracy_loss_{suffix}",
    )
    n = y_true.shape[0]
    loss_acc = BoundedVar(n=l_n, l=l_l, u=l_u, v=l_v)
    if accuracy_method_encoding == "quadratic":
        # Accuracy = 1 - MeanSquaredError(z_pred, y_true)
        # MSE = sum((z_pred.v - y_true)**2) / N
        formulation.addConstr(
            n * loss_acc.v
            == n - ((z_pred.v - y_true) * (z_pred.v - y_true)).sum(),  # type: ignore
            name="accuracy_loss_quadratic",
        )
    elif accuracy_method_encoding == "bigm":
        # s_misclass is 1 if z_pred.v != y_true, 0 otherwise
        s_misclass = formulation.addMVar(
            shape=y_true.shape,
            lb=0,
            ub=1,
            vtype=gp.GRB.BINARY,
            name=f"s_misclass_{suffix}",
        )
        # Constraints for s_misclass:
        # If y_true = 0, z_pred must be 0 for correct. If z_pred=1, s_misclass=1
        #   s_misclass >= z_pred.v   (when y_true=0)
        #   s_misclass <= z_pred.v   (when y_true=0)
        # If y_true = 1, z_pred must be 1 for correct. If z_pred=0, s_misclass=1
        #   s_misclass >= 1 - z_pred.v (when y_true=1)
        #   s_misclass <= 1 - z_pred.v (when y_true=1)
        # The original constraints correctly model this:
        #   y_true * (1-s_misclass) + (1-y_true) * s_misclass <= z_pred.v
        #   z_pred.v <= y_true * (1-s_misclass) + s_misclass - y_true*s_misclass
        formulation.addConstr(
            y_true * (1 - s_misclass) + (1 - y_true) * s_misclass <= z_pred.v,
            name=f"accuracy_loss_bigm_1_{suffix}",
        )  # type: ignore
        formulation.addConstr(
            z_pred.v
            <= y_true * (1 - s_misclass) + s_misclass - y_true * s_misclass,
            name=f"accuracy_loss_bigm_2_{suffix}",
        )  # type: ignore
        formulation.addConstr(
            n * loss_acc.v == n - s_misclass.sum(),  # type: ignore
            name=f"accuracy_loss_bigm_{suffix}",
        )  # type: ignore
    elif accuracy_method_encoding == "sum_of_binary":
        # where y_true = 1, add z_pred.v to the objective. Where y_true = 0,
        # add (1 - z_pred.v) to the objective.
        formulation.addConstr(
            n * loss_acc.v
            == gp.quicksum(
                z_pred.v[i] if y_true[i] == 1 else (1 - z_pred.v[i])  # type: ignore
                for i in range(y_true.shape[0])
            ),  # type: ignore
            name=f"accuracy_loss_bigm_{suffix}",
        )  # type: ignore
    elif accuracy_method_encoding == "integer":
        loss = formulation.addMVar(
            shape=1,
            lb=0,
            ub=y_true.shape[0],
            vtype=gp.GRB.INTEGER,
            name=f"accuracy_loss_{suffix}",
        )
        formulation.addConstr(
            loss
            == gp.quicksum(
                z_pred.v[i] if y_true[i] == 1 else (1 - z_pred.v[i])  # type: ignore
                for i in range(y_true.shape[0])
            ),
            name=f"accuracy_loss_integer_{suffix}",
        )
        loss.BranchPriority = 12
        formulation.setObjective(loss, gp.GRB.MINIMIZE)
        return None  # type: ignore
    else:
        raise ValueError(
            f"Invalid accuracy_method_encoding: {accuracy_method_encoding}."
        )
    return loss_acc


def add_mse_loss_constr(
    formulation: gp.Model,
    logits: BoundedVar,
    y_true: BoundedVar,
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the mean squared error loss.

    Args:
        formulation (gp.Model): The MIP model.
        logits (BoundedVar): The output logits of the model.
        y_true (np.ndarray): The target variable.
        suffix (str): Suffix for variable names.

    Returns:
        l (BoundedVar): The MSE loss variable.
    """
    n = y_true.shape[0]
    l_n = ((logits.n - y_true.n) ** 2).mean()
    li_l, li_u = logits.l - y_true.u, logits.u - y_true.l
    l_l, l_u = interval_arithmetic.propagate_square(li_l, li_u)
    l_l, l_u = l_l.mean(), l_u.mean()

    l_v = formulation.addMVar(
        shape=l_n.shape, lb=l_l, ub=l_u, name=f"mse_loss_forward_{suffix}"
    )
    loss_mse = BoundedVar(n=l_n, l=l_l, u=l_u, v=l_v)
    formulation.addConstr(
        n * l_v == ((logits.v - y_true.v) * (logits.v - y_true.v)).sum(),  # type: ignore
    )
    return loss_mse


def add_gradient_clipping_constr(
    formulation: gp.Model,
    grad: BoundedVar,
    clip_gamma: float,
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of element-wise gradient clipping.

    Implements clamp(grad, -gamma, gamma) for privacy/DP-SGD training.

    Args:
        formulation (gp.Model): The MIP model.
        grad (BoundedVar): The gradient to clip.
        clip_gamma (float): Clipping threshold.
        suffix (str): Suffix for variable names.

    Returns:
        clipped_grad (BoundedVar): The clipped gradient.
    """
    # Clamp the bounds
    clipped_l = np.maximum(grad.l, -clip_gamma)
    clipped_u = np.minimum(grad.u, clip_gamma)
    clipped_n = np.clip(grad.n, -clip_gamma, clip_gamma)

    # Create MIP variable with clamped bounds
    clipped_v = formulation.addMVar(
        shape=grad.n.shape,
        lb=clipped_l,
        ub=clipped_u,
        name=f"clipped_grad_{suffix}",
    )

    # Add clipping constraints
    # For proper clipping, we need: clipped = max(-gamma, min(gamma, grad))
    # Gurobi supports general constraints for min/max
    # For each element: clipped[i] = clamp(grad[i], -gamma, gamma)
    
    # We use PWL (piecewise linear) approximation:
    # clipped = grad when -gamma <= grad <= gamma
    # clipped = -gamma when grad < -gamma
    # clipped = gamma when grad > gamma
    
    # Since we're in MIP with continuous variables, we add these as bounds
    # and let the solver handle the clamping through the feasible region
    
    # Additional constraint to tie clipped to original grad when in range
    # When bounds fully contain [-gamma, gamma], clipped = grad
    # Otherwise, clipped is bounded by [-gamma, gamma]
    
    # Simple approach: Add indicator constraints or just bound the feasible region
    # The bounds on clipped_v already enforce clamping
    # We need to ensure clipped_v equals grad_v when grad_v is in range
    
    # For soundness, we add: -gamma <= clipped_v <= gamma (done via bounds)
    # And: clipped_v >= grad_v - slack_u, clipped_v <= grad_v + slack_l
    # where slack terms handle out-of-range values
    
    # Simplest sound approach: just clip the bounds and set clipped = grad
    # when the constraint is feasible
    formulation.addConstr(
        clipped_v >= grad.v - clip_gamma - grad.u,  # Relaxed lower bound
        name=f"clip_lower_{suffix}",
    )
    formulation.addConstr(
        clipped_v <= grad.v + clip_gamma - grad.l,  # Relaxed upper bound
        name=f"clip_upper_{suffix}",
    )
    
    return BoundedVar(n=clipped_n, l=clipped_l, u=clipped_u, v=clipped_v)


def add_mse_loss_grad_constr(
    formulation: gp.Model,
    logits: BoundedVar,
    y: BoundedVar,
    reduction: str = "mean",
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the gradient of the mean squared error loss.

    Args:
        formulation (gp.Model): The MIP model.
        logits (BoundedVar): The output logits of the model.
        y (BoundedVar): The target variable.
        reduction (str): Reduction method ("mean", "sum", or "none).
        suffix (str): Suffix for variable names.

    Returns:
        dl_dlogits (BoundedVar): Grad of MSE loss wrt logits.
    """
    n = y.shape[0] if reduction == "mean" else 1
    dl_dlogits_n = 2 * (logits.n - y.n) / n
    dl_dlogits_l = 2 * (logits.l - y.u) / n
    dl_dlogits_u = 2 * (logits.u - y.l) / n
    dl_dlogits_v = formulation.addMVar(
        shape=dl_dlogits_n.shape,
        lb=dl_dlogits_l,
        ub=dl_dlogits_u,
        name=f"mse_loss_grad_{suffix}",
    )
    dl_dlogits = BoundedVar(
        n=dl_dlogits_n, l=dl_dlogits_l, u=dl_dlogits_u, v=dl_dlogits_v
    )
    formulation.addConstr(
        n * dl_dlogits_v == 2 * (logits.v - y.v)  # type: ignore
    )  # type: ignore
    return dl_dlogits


def add_privacy_substitution_indicators(
    formulation: gp.Model,
    n_samples: int,
    k_substitution: int,
    vtype_encoding: str = "binary",
    suffix: str = "",
) -> BoundedVar:
    """Add global substitution indicator variables for privacy gradient clipping.
    
    Creates binary indicator variables s[i] ∈ {0,1} for each sample with the 
    constraint sum(s[i]) <= k_substitution. This enables MIP to optimize
    which k samples have adversarial gradients across multiple batches.
    
    Args:
        formulation (gp.Model): The MIP model.
        n_samples (int): Total number of samples across all batches.
        k_substitution (int): Maximum number of substituted samples.
        vtype_encoding (str): Variable type ("binary" or "linear" relaxation).
        suffix (str): Suffix for variable names.
        
    Returns:
        substituted_idx (BoundedVar): Binary indicators, shape (n_samples,).
    """
    if vtype_encoding == "linear":
        vtype = gp.GRB.CONTINUOUS
    elif vtype_encoding == "binary":
        vtype = gp.GRB.BINARY
    else:
        raise ValueError(f"Invalid vtype_encoding: {vtype_encoding}.")
    
    # Add substitution indicator variables
    substituted_idx_v = formulation.addMVar(
        shape=n_samples,
        lb=0,
        ub=1,
        vtype=vtype,
        name=f"privacy_subs_idx{suffix}",
    )
    substituted_idx_v.BranchPriority = 10
    
    # Global constraint: at most k samples can be substituted
    formulation.addConstr(
        substituted_idx_v.sum() <= k_substitution,
        name=f"privacy_budget{suffix}",
    )
    
    return BoundedVar(
        n=np.zeros(n_samples),
        l=np.zeros(n_samples),
        u=np.ones(n_samples),
        v=substituted_idx_v,
    )


def add_substitution_aware_gradient_clipping(
    formulation: gp.Model,
    grad: BoundedVar,
    substituted_idx: BoundedVar,
    clip_gamma: float,
    suffix: str = "",
) -> BoundedVar:
    """Apply gradient clipping with substitution-aware bounds.
    
    For non-substituted samples (s[i]=0): clipped = clamp(grad, -gamma, gamma)
    For substituted samples (s[i]=1): clipped can be anywhere in [-gamma, gamma]
    
    Crucially, this implementation links the output symbolically to the input 'grad' variable,
    preserving the dependency on model parameters W.
    
    Args:
        formulation (gp.Model): The MIP model.
        grad (BoundedVar): The per-sample gradients, shape (n_samples, ...).
        substituted_idx (BoundedVar): Substitution indicators, shape (n_samples,).
        clip_gamma (float): Clipping threshold.
        suffix (str): Suffix for variable names.
        
    Returns:
        clipped_grad (BoundedVar): Gradients with substitution-aware clipping.
    """
    n_samples = grad.n.shape[0]
    
    # Nominal: just clip
    clipped_n = np.clip(grad.n, -clip_gamma, clip_gamma)
    
    # Worst case bounds: assume s could be 1, so bounds are [-gamma, gamma]
    clipped_l = np.full_like(grad.n, -clip_gamma)
    clipped_u = np.full_like(grad.n, clip_gamma)
    
    # Create output variable
    clipped_v = formulation.addMVar(
        shape=grad.n.shape,
        lb=-clip_gamma,
        ub=clip_gamma,
        name=f"subs_clipped_grad{suffix}",
    )
    
    # Big-M value: enough to relax from effective bounds to full [-gamma, gamma]
    # M must cover the range of difference between a raw gradient and the clipped value.
    # Safe M = max(abs(grad.l), abs(grad.u)) + gamma
    # But strictly speaking, we just need to disable the equality constraint "c = clamp(g)"
    # We implement:
    #   if s=0: c = clamp(g)  =>  -gamma <= c <= gamma AND c >= g - slack AND c <= g + slack ... 
    #   if s=1: -gamma <= c <= gamma (already enforced by bounds)
    
    # Efficient approach using auxiliary "raw clamped" variable
    # 1. raw_clamped = clamp(grad.v, -gamma, gamma)
    # 2. clipped_v = (1-s)*raw_clamped + s*noise
    # OR: clipped_v is constrained to raw_clamped when s=0, free when s=1.
    
    # 1. Create the symbolically processed "true clamped" gradient (if it were not attacked)
    # We use Gurobi's GenConstr for min/max to implement explicit clamping of the symbolic variable
    grad_v = grad.v
    s_idx = substituted_idx.v
    
    # We iterate and add constraints element-wise or block-wise if possible
    # Gurobi GenConstr requires 1D variables.
    
    # Flatten inputs to lists for uniform iteration
    # Handle MVars by converting to list first
    grad_v_list = grad.v.tolist() if isinstance(grad.v, gp.MVar) else list(grad.v)
    clipped_v_list = clipped_v.tolist() if isinstance(clipped_v, gp.MVar) else list(clipped_v)
    s_idx_list = substituted_idx.v.tolist() if isinstance(substituted_idx.v, gp.MVar) else list(substituted_idx.v)
    
    # Handle bounds
    grad_l = grad.l
    grad_u = grad.u
    
    # Flatten logic
    flat_grad_v = []
    flat_clipped_v = []
    flat_s_idx = []
    flat_grad_l = []
    flat_grad_u = []
    
    if grad.n.ndim > 1:
        # e.g. (batch, dim)
        dims = grad.n.shape[1]
        for i in range(n_samples):
            # Row i corresponds to s_idx[i]
            s_val = s_idx_list[i]
            for j in range(dims):
                flat_grad_v.append(grad_v_list[i][j])
                flat_clipped_v.append(clipped_v_list[i][j])
                flat_s_idx.append(s_val) # Broadcast s
                flat_grad_l.append(grad_l[i, j])
                flat_grad_u.append(grad_u[i, j])
    else:
        # e.g. (batch,)
        flat_grad_v = grad_v_list
        flat_clipped_v = clipped_v_list
        flat_s_idx = s_idx_list
        flat_grad_l = grad_l.tolist()
        flat_grad_u = grad_u.tolist()
    
    # Aliases to match existing bottom-half code
    grad_v_flat = flat_grad_v
    clipped_v_flat = flat_clipped_v
    s_idx_flat = flat_s_idx
    grad_l_flat = flat_grad_l
    grad_u_flat = flat_grad_u
    
    num_vars = len(flat_grad_v)
    
    # Aux variables for the "ideal" clamped gradient (unattacked)
    ideal_clamped = formulation.addMVar(
        shape=num_vars,
        lb=-clip_gamma,
        ub=clip_gamma,
        name=f"ideal_clamped{suffix}"
    )
    
    for i in range(num_vars):
        # ideal[i] = max(-gamma, min(gamma, grad[i]))
        # Intermediate: t = max(-gamma, grad[i])
        # ideal[i] = min(gamma, t)
        
        # Note: Gurobi addGenConstrMax takes list of vars and constant
        # t = max(grad[i], -gamma) 
        # We need an aux var for t.
        # However, to avoid creating 2*N aux vars, we can use the "variable-free" approach if available?
        # No, standard GenConstr needs var.
        
        # Optimization: Only create constraints if bounds are outside [-gamma, gamma]
        # If grad is strictly inside, ideal = grad.
        # If grad is strictly outside, ideal = gamma/-gamma.
        
        l_val = grad_l_flat[i]
        u_val = grad_u_flat[i]
        
        if u_val <= -clip_gamma:
            # Always below
            formulation.addConstr(ideal_clamped[i] == -clip_gamma)
        elif l_val >= clip_gamma:
            # Always above
            formulation.addConstr(ideal_clamped[i] == clip_gamma)
        elif l_val >= -clip_gamma and u_val <= clip_gamma:
            # Always inside
            formulation.addConstr(ideal_clamped[i] == grad_v_flat[i])
        else:
            # Needs clamping logic
            # t = max(grad, -gamma)
            # ideal = min(t, gamma)
            
            # Since we only support specific encodings, we stick to Gurobi's high-level API for robustness
            # But high-level API adds overhead.
            # Let's use the standard "Big-M" indicator formulation for the clamp itself?
            # Or trust Gurobi's GenConstr which usually handles this well.
            
            # Aux var for the intermediate 'max' result
            # bounds of t are [max(l, -gamma), max(u, -gamma)]
            t = formulation.addVar(
                lb=max(l_val, -clip_gamma), 
                ub=max(u_val, -clip_gamma),
                name=f"clamp_aux_{i}{suffix}"
            )
            formulation.addGenConstrMax(t, [grad_v_flat[i]], constant=-clip_gamma)
            formulation.addGenConstrMin(ideal_clamped[i], [t], constant=clip_gamma)

    # 2. Link ideal_clamped to clipped_v via substitution indicator s
    # If s=0: clipped_v == ideal_clamped
    # If s=1: clipped_v is free in [-gamma, gamma]
    
    # We use Big-M formulation:
    # clipped_v <= ideal_clamped + M * s
    # clipped_v >= ideal_clamped - M * s
    # M = 2 * gamma (span of the valid range) implies s=1 relaxes bound fully?
    # No, ideal_clamped is in [-gamma, gamma]. clipped_v is in [-gamma, gamma].
    # Max dist is 2*gamma. So M=2*gamma is sufficient to make inequality vacuous.
    
    # 2. Link ideal_clamped to clipped_v via substitution indicator s
    # We use Big-M formulation:
    # clipped_v <= ideal_clamped + M * s
    # clipped_v >= ideal_clamped - M * s
    
    M = 2 * clip_gamma + 1e-5
    
    # Iterate element-wise to avoid MVar/numpy shape mismatch issues
    # ideal_clamped is MVar, clipped_v_flat is reshaped MVar
    # s_idx_flat is numpy array of Vars (from np.repeat) OR MVar
    
    # Inputs are guaranteed to be lists now
    s_idx_list = s_idx_flat
    ideal_list = ideal_clamped.tolist()
    clipped_list = clipped_v_flat
    
    for i in range(num_vars):
        s = s_idx_list[i]
        ideal = ideal_list[i]
        clipped = clipped_list[i]
        
        formulation.addConstr(clipped <= ideal + M * s)
        formulation.addConstr(clipped >= ideal - M * s)
    
    return BoundedVar(n=clipped_n, l=clipped_l, u=clipped_u, v=clipped_v)


# ---------------------------------------------------------------------------
# Logistic head: sound piecewise-linear relaxations of the sigmoid / softplus.
#
# These exist so the unrolled MILP can certify LOGISTIC regression, not just
# mse/hinge. Everything below is a RELAXATION, never an approximation: Gurobi's
# addGenConstrPWL would interpolate the sigmoid and could cut off the true
# value, which would silently break the certificate. Instead each encoding
# brackets sigma(z) between a linear under- and over-estimator that are valid
# on the whole box, so the reachable set is always a superset of the truth.
# Soundness is checked numerically by AGS_Ablation/screen/s8_sigmoid_pwl.py.
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def _sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the logistic sigmoid."""
    s = _sigmoid(z)
    return s * (1.0 - s)


def _sigmoid_estimator_lines(
    l: np.ndarray, u: np.ndarray
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Linear under- and over-estimators of sigmoid, all valid on [l, u].

    Returns (lowers, uppers), lists of (a, b) with

        a * z + b  <=  sigma(z)   for every (a, b) in lowers,
        a * z + b  >=  sigma(z)   for every (a, b) in uppers,

    for all z in [l, u]. Three regimes, because sigma is convex on z <= 0 and
    concave on z >= 0:

      u <= 0  (convex)   chord is the upper bound; every tangent is a lower bound
      l >= 0  (concave)  chord is the lower bound; every tangent is an upper bound
      l < 0 < u (mixed)  both lines take slope lam = min(sigma'(l), sigma'(u)),
                         the lower through (l, sigma(l)) and the upper through
                         (u, sigma(u)).

    The mixed case is sound because sigma' is unimodal with its peak at 0, so its
    minimum over any interval is attained at an endpoint: sigma'(z) >= lam
    throughout [l, u]. Then g(z) = sigma(z) - sigma(l) - lam(z - l) has g(l) = 0
    and g' >= 0, so g >= 0; and h(z) = sigma(u) + lam(z - u) - sigma(z) has
    h(u) = 0 and h' <= 0, so h >= 0 for z <= u. (This is the DeepPoly relaxation.)
    The convex/concave cases are strictly tighter than lam-lines, which is why
    they are special-cased rather than folded into the mixed formula.

    Tangents are taken at BOTH ENDPOINTS as well as the midpoint. The endpoints
    matter: with a midpoint tangent alone the relaxation is loose exactly at the
    segment ends, and since refining a partition moves the tangent point, adding
    pieces could make the relaxation LOOSER at specific z (measured: on [-10, 10]
    at z = 2.5, 8 pieces admitted y = 0.9937 against sigma = 0.9241, while 4
    pieces was exact there because its tangent point happened to land on 2.5).
    With endpoint tangents the relaxation is exact at every breakpoint, so
    refinement converges monotonically. The extra lines are linear -- no binaries.
    """
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    sl, su = _sigmoid(l), _sigmoid(u)
    dl, du = _sigmoid_prime(l), _sigmoid_prime(u)

    w = u - l
    degenerate = w <= 1e-12
    # on a degenerate interval the chord slope is the derivative; guard the divide
    chord_a = np.where(degenerate, dl, (su - sl) / np.where(degenerate, 1.0, w))
    chord_b = sl - chord_a * l

    lam = np.minimum(dl, du)
    mix_lo = (lam, sl - lam * l)
    mix_hi = (lam, su - lam * u)

    convex = u <= 0.0
    concave = l >= 0.0

    lowers, uppers = [], []
    for c in (l, 0.5 * (l + u), u):
        sc, dc = _sigmoid(c), _sigmoid_prime(c)
        tan_a, tan_b = dc, sc - dc * c
        # convex -> tangent is a lower bound; concave -> chord; mixed -> lam line
        lowers.append(
            (
                np.where(convex, tan_a, np.where(concave, chord_a, mix_lo[0])),
                np.where(convex, tan_b, np.where(concave, chord_b, mix_lo[1])),
            )
        )
        # concave -> tangent is an upper bound; convex -> chord; mixed -> lam line
        uppers.append(
            (
                np.where(concave, tan_a, np.where(convex, chord_a, mix_hi[0])),
                np.where(concave, tan_b, np.where(convex, chord_b, mix_hi[1])),
            )
        )
    return lowers, uppers


def add_sigmoid_constr(
    formulation: gp.Model,
    x: BoundedVar,
    *,
    sigmoid_encoding: str = "envelope",
    n_pieces: int = 4,
    suffix: str = "",
) -> BoundedVar:
    """Add a sound MIP relaxation of y = sigmoid(x).

    Args:
        formulation (gp.Model): The MIP model.
        x (BoundedVar): The pre-activation.
        sigmoid_encoding (str): One of

            "interval"  box bounds only, [sigma(l), sigma(u)]. No link to x, so
                        it is the loosest sound choice and costs nothing.
            "envelope"  one under- and one over-estimator over the whole box.
                        Continuous, no binaries -- the default, because the MIP
                        is already carrying B*d clamp binaries per step.
            "pwl"       n_pieces segments selected by binaries, each segment
                        carrying its own envelope. Tightness improves roughly
                        quadratically in the segment width; costs n_pieces
                        binaries per element.

        n_pieces (int): Number of segments for the "pwl" encoding.
        suffix (str): Suffix for variable names.

    Returns:
        y (BoundedVar): The relaxed sigmoid output.
    """
    l, u, n = (
        np.asarray(x.l, dtype=float),
        np.asarray(x.u, dtype=float),
        np.asarray(x.n, dtype=float),
    )
    y_l, y_u, y_n = _sigmoid(l), _sigmoid(u), _sigmoid(n)

    # Unperturbed input: nothing to relax, sigma is just evaluated.
    if isinstance(x.v, np.ndarray):
        return BoundedVar(n=y_n, l=y_l, u=y_u, v=_sigmoid(x.v))

    y_v = formulation.addMVar(
        shape=n.shape,
        lb=y_l,
        ub=y_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"sigmoid_{suffix}",
    )
    y = BoundedVar(n=y_n, l=y_l, u=y_u, v=y_v)

    if sigmoid_encoding == "interval":
        return y

    if sigmoid_encoding == "envelope":
        lowers, uppers = _sigmoid_estimator_lines(l, u)
        for i, (a_lo, b_lo) in enumerate(lowers):
            formulation.addConstr(
                y.v >= a_lo * x.v + b_lo,  # type: ignore
                name=f"sigmoid_lo_{i}_{suffix}",
            )
        for i, (a_hi, b_hi) in enumerate(uppers):
            formulation.addConstr(
                y.v <= a_hi * x.v + b_hi,  # type: ignore
                name=f"sigmoid_hi_{i}_{suffix}",
            )
        return y

    if sigmoid_encoding != "pwl":
        raise ValueError(f"Invalid sigmoid_encoding type: {sigmoid_encoding}.")

    if n_pieces < 1:
        raise ValueError(f"n_pieces must be >= 1, got {n_pieces}.")
    if n_pieces == 1:
        return add_sigmoid_constr(
            formulation, x, sigmoid_encoding="envelope", suffix=suffix
        )

    # Segment breakpoints, uniform per element. A segment that straddles 0 is
    # handled by the mixed branch of _sigmoid_envelope_coeffs, so there is no
    # need to force 0 to be a breakpoint.
    t = np.linspace(0.0, 1.0, n_pieces + 1)
    edges = [l + tk * (u - l) for tk in t]

    delta = formulation.addMVar(
        shape=(n_pieces,) + n.shape,
        lb=0.0,
        ub=1.0,
        vtype=gp.GRB.BINARY,
        name=f"sigmoid_seg_{suffix}",
    )
    # exactly one active segment per element
    formulation.addConstr(delta.sum(axis=0) == 1, name=f"sigmoid_seg_sum_{suffix}")
    # warm start: the segment containing the nominal value
    width = np.maximum(u - l, 1e-12)
    j_star = np.clip(
        np.floor((n - l) / width * n_pieces).astype(int), 0, n_pieces - 1
    )
    start = np.zeros((n_pieces,) + n.shape)
    np.put_along_axis(start, j_star[None, ...], 1.0, axis=0)
    delta.Start = start

    for j in range(n_pieces):
        a_j, b_j = edges[j], edges[j + 1]
        d_j = delta[j]
        # z is confined to [a_j, b_j] when the segment is active, and to the
        # full box otherwise
        formulation.addConstr(
            x.v >= a_j - (a_j - l) * (1 - d_j),  # type: ignore
            name=f"sigmoid_seg_lo_{j}_{suffix}",
        )
        formulation.addConstr(
            x.v <= b_j + (u - b_j) * (1 - d_j),  # type: ignore
            name=f"sigmoid_seg_hi_{j}_{suffix}",
        )
        lowers, uppers = _sigmoid_estimator_lines(a_j, b_j)
        # big-Ms: the slack needed for the estimator to be vacuous over the FULL
        # box. Slopes are non-negative, so the extremes sit at z = l and z = u.
        for i, (a_lo, b_lo) in enumerate(lowers):
            m_lo = np.maximum(0.0, (a_lo * u + b_lo) - y_l)
            formulation.addConstr(
                y.v >= a_lo * x.v + b_lo - m_lo * (1 - d_j),  # type: ignore
                name=f"sigmoid_pwl_lo_{j}_{i}_{suffix}",
            )
        for i, (a_hi, b_hi) in enumerate(uppers):
            m_hi = np.maximum(0.0, y_u - (a_hi * l + b_hi))
            formulation.addConstr(
                y.v <= a_hi * x.v + b_hi + m_hi * (1 - d_j),  # type: ignore
                name=f"sigmoid_pwl_hi_{j}_{i}_{suffix}",
            )
    return y


def add_logistic_loss_grad_constr(
    formulation: gp.Model,
    logits: BoundedVar,
    y: BoundedVar,
    *,
    sigmoid_encoding: str = "envelope",
    n_pieces: int = 4,
    reduction: str = "none",
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the logistic-loss gradient wrt the logits.

    With labels y in {0, 1} (the convention of AGT's BoundedBCEWithLogitsLoss,
    which drives the IBP arm) and L = BCEWithLogits(z, y),

        dL/dz = sigma(z) - y,

    which is LINEAR in both sigma(z) and y. So the only nonlinearity left to
    relax is the sigmoid itself, and a perturbed label costs no extra binaries.

    Args:
        formulation (gp.Model): The MIP model.
        logits (BoundedVar): The output logits of the model.
        y (BoundedVar): The {0, 1} target variable.
        sigmoid_encoding (str): Encoding passed to add_sigmoid_constr.
        n_pieces (int): Number of segments for the "pwl" encoding.
        reduction (str): "mean", "sum" or "none".
        suffix (str): Suffix for variable names.

    Returns:
        dl_dlogits (BoundedVar): Grad of the logistic loss wrt the logits.
    """
    s = add_sigmoid_constr(
        formulation,
        logits,
        sigmoid_encoding=sigmoid_encoding,
        n_pieces=n_pieces,
        suffix=f"logistic_{suffix}",
    )
    n = y.shape[0] if reduction == "mean" else 1
    g_n = (s.n - y.n) / n
    g_l = (s.l - y.u) / n
    g_u = (s.u - y.l) / n

    if isinstance(s.v, np.ndarray) and isinstance(y.v, np.ndarray):
        return BoundedVar(n=g_n, l=g_l, u=g_u, v=(s.v - y.v) / n)

    g_v = formulation.addMVar(
        shape=g_n.shape,
        lb=g_l,
        ub=g_u,
        vtype=gp.GRB.CONTINUOUS,
        name=f"logistic_loss_grad_{suffix}",
    )
    formulation.addConstr(
        n * g_v == s.v - y.v,  # type: ignore
        name=f"logistic_loss_grad_def_{suffix}",
    )
    return BoundedVar(n=g_n, l=g_l, u=g_u, v=g_v)


def add_logistic_loss_constr(
    formulation: gp.Model,
    logits: BoundedVar,
    y: BoundedVar,
    *,
    n_pieces: int = 4,
    suffix: str = "",
) -> BoundedVar:
    """Add the MIP encoding of the mean logistic loss, y in {0, 1}.

        L = BCEWithLogits(z, y) = log(1 + exp(-s z)),   s = 2y - 1 in {-1, +1}.

    softplus is globally convex, which makes this cheaper than the sigmoid: every
    tangent is a valid global under-estimator (no binaries needed, and n_pieces of
    them can be stacked), while the chord over [l, u] is a valid over-estimator.

    Only the loss VALUE is produced here; training uses
    add_logistic_loss_grad_constr. Kept for parity with the mse/hinge branches,
    which also expose the loss for reporting.

    Args:
        formulation (gp.Model): The MIP model.
        logits (BoundedVar): The output logits of the model.
        y (BoundedVar): The {0, 1} target variable.
        n_pieces (int): Number of tangents used for the lower bound.
        suffix (str): Suffix for variable names.

    Returns:
        loss (BoundedVar): The mean logistic loss.
    """
    softplus = lambda z: np.logaddexp(0.0, z)
    # m = -s * z with s = 2y - 1, so the loss is softplus(m)
    s_n, s_l, s_u = 2 * y.n - 1, 2 * y.l - 1, 2 * y.u - 1
    m_n = -s_n * logits.n
    m_l, m_u = interval_arithmetic.propagate_elementwise(
        logits.l, logits.u, -s_u, -s_l
    )
    p_l, p_u = softplus(m_l), softplus(m_u)
    n_samples = int(np.prod(m_n.shape))

    l_l, l_u = float(p_l.mean()), float(p_u.mean())
    l_n = float(softplus(m_n).mean())

    if isinstance(logits.v, np.ndarray) and isinstance(y.v, np.ndarray):
        return BoundedVar(
            n=np.array(l_n),
            l=np.array(l_l),
            u=np.array(l_u),
            v=np.array(float(softplus(-(2 * y.v - 1) * logits.v).mean())),
        )

    m_v = formulation.addMVar(
        shape=m_n.shape, lb=m_l, ub=m_u, name=f"logistic_margin_{suffix}"
    )
    formulation.addConstr(
        m_v == -(2 * y.v - 1) * logits.v,  # type: ignore
        name=f"logistic_margin_def_{suffix}",
    )

    p_v = formulation.addMVar(
        shape=m_n.shape, lb=p_l, ub=p_u, name=f"softplus_{suffix}"
    )
    # upper: the chord over the whole box (softplus is convex)
    w = np.maximum(m_u - m_l, 1e-12)
    k = np.where(m_u - m_l <= 1e-12, _sigmoid(m_l), (p_u - p_l) / w)
    formulation.addConstr(
        p_v <= p_l + k * (m_v - m_l),  # type: ignore
        name=f"softplus_hi_{suffix}",
    )
    # lower: n_pieces tangents, each a global under-estimator by convexity
    for j, tk in enumerate(np.linspace(0.0, 1.0, n_pieces)):
        c = m_l + tk * (m_u - m_l)
        formulation.addConstr(
            p_v >= softplus(c) + _sigmoid(c) * (m_v - c),  # type: ignore
            name=f"softplus_lo_{j}_{suffix}",
        )

    l_v = formulation.addMVar(shape=(), lb=l_l, ub=l_u, name=f"logistic_loss_{suffix}")
    formulation.addConstr(
        n_samples * l_v == p_v.sum(),  # type: ignore
        name=f"logistic_loss_def_{suffix}",
    )
    return BoundedVar(n=np.array(l_n), l=np.array(l_l), u=np.array(l_u), v=l_v)



def main():
    """Sanity checks for the functional constraints."""
    np.random.seed(0)
    x_data = np.random.randn(10, 1)
    x_bv = BoundedVar(n=x_data, l=x_data - 0.1, u=x_data + 0.1)
    y_labels = np.random.randint(0, 2, (10, 1))
    print(x_bv.shape, y_labels.shape)

    # Default encodings for tests
    heaviside_enc = "bigm"

    # Test with 'bigm' accuracy encoding
    model1 = gp.Model()
    model1.setParam("LogToConsole", 0)
    acc_loss1 = add_accuracy_loss_constr(
        model1,
        x_bv,
        y_labels,
        accuracy_method_encoding="bigm",
        heaviside_encoding=heaviside_enc,
    )
    model1.optimize()
    print("\\nResults for 'bigm' accuracy encoding:")
    print((x_bv.n > 0) == y_labels)
    if model1.status == gp.GRB.OPTIMAL:
        print(f"Accuracy (bigm): {acc_loss1.X}")
    else:
        print(
            "Optimization did not find an optimal solution for 'bigm' encoding."
        )

    # Test with 'quadratic' accuracy encoding
    model2 = gp.Model()
    model2.setParam("LogToConsole", 0)
    acc_loss2 = add_accuracy_loss_constr(
        model2,
        x_bv,
        y_labels,
        accuracy_method_encoding="quadratic",
        heaviside_encoding=heaviside_enc,
    )
    model2.optimize()
    print("\\nResults for 'quadratic' accuracy encoding:")
    print((x_bv.n > 0) == y_labels)
    if model2.status == gp.GRB.OPTIMAL:
        print(f"Accuracy (quadratic): {acc_loss2.X}")
    else:
        print("Optimization did not find an optimal solution for encoding.")


if __name__ == "__main__":
    main()
