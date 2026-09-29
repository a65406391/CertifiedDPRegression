"""Per-sample gradient clipping on the disjoint-substitution path.

WRITTEN AS AN ADDITION, NOT AN EDIT. Nothing in abstract_gradient_training_opt is
modified; this file only composes its public primitives. Intended for review by the
package author before any of it is upstreamed.

--------------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------------
AGS needs a valid parameter-space bound for the algorithm it actually runs:
per-sample gradient clipping at gamma. Two encodings exist upstream, and neither
gives that under a *realizable* substitution:

  perturbation_constraint="input"     applies clipping, but with input_domain set
                                      add_substitution_constraint declares ALL N
                                      samples' features over the full domain box.
                                      McCormick tightness follows the DECLARED
                                      interval, not the incumbent value, so every
                                      sample's gradient product is relaxed loosely
                                      even when a binary pins it to its nominal.
                                      Measured 36x worse than IBP.

  perturbation_constraint="gradient"  uses add_substitution_constraint_disjoint,
                                      which keeps the clean data as CONSTANTS and
                                      introduces variables only for the k
                                      substituted records -- exactly the right
                                      structure -- but applies no clipping.

This file supplies the missing combination: the disjoint encoding *with* per-sample
clipping, so the MIP bound and the AGT/IBP bound describe the same algorithm and can
be compared.

--------------------------------------------------------------------------------
What it enables
--------------------------------------------------------------------------------
Under the disjoint encoding the substituted record is one datum whose gradient is
determined by (x', y', theta_t). It therefore cannot be re-drawn independently at each
step, which is the constraint the free-box encoding (input_domain=None) discards. With
a window of F updates the solver commits to a single substituted record across the
whole window; at F = n_s the entire trajectory is one problem and, for k=1, the solver
finds the single record that moves the trajectory most over the whole run.

That trajectory-global bound is the quantity this file exists to measure. Under the
free-box encoding it is unmeasurable: with the substituted gradient free in
[-gamma, gamma] at every step, fixing *which* record is substituted buys nothing when
that record appears in every batch (measured: ratio 1.000 at F = 1, 2, 4 at full batch).
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib

import gurobipy as gp
import numpy as np

from abstract_gradient_training_opt import certified_training as _ct
from abstract_gradient_training_opt.formulations import functional as F
from abstract_gradient_training_opt.formulations.modules import MIPSequential
from abstract_gradient_training_opt.formulations.variables import BoundedVar


# ----------------------------------------------------------------- per-sample grads
def per_sample_param_grads(formulation, x, dl_dy, suffix=""):
    """Per-sample dL/dtheta for y = x @ w.T + b, before the batch sum.

        dl_dw_per[i, o, j] = dl_dy[i, o] * x[i, j]
        dl_db_per[i, o]    = dl_dy[i, o]

    Constant x gives a linear constraint; variable x makes each element bilinear and
    it is relaxed by the four McCormick envelopes, which are the tightest linear
    relaxation of a bilinear term over a box.
    """
    variable_x = isinstance(x.v, (gp.MVar, gp.MLinExpr, gp.MQuadExpr))
    B, in_f = x.n.shape
    out_f = dl_dy.n.shape[1]

    a_l = np.broadcast_to(dl_dy.l[:, :, None], (B, out_f, in_f))
    a_u = np.broadcast_to(dl_dy.u[:, :, None], (B, out_f, in_f))
    a_n = np.broadcast_to(dl_dy.n[:, :, None], (B, out_f, in_f))
    b_l = np.broadcast_to(x.l[:, None, :], (B, out_f, in_f))
    b_u = np.broadcast_to(x.u[:, None, :], (B, out_f, in_f))
    b_n = np.broadcast_to(x.n[:, None, :], (B, out_f, in_f))

    per_n = a_n * b_n
    lo = np.minimum(np.minimum(a_l * b_l, a_l * b_u), np.minimum(a_u * b_l, a_u * b_u))
    hi = np.maximum(np.maximum(a_l * b_l, a_l * b_u), np.maximum(a_u * b_l, a_u * b_u))
    per_v = formulation.addMVar(
        shape=(B, out_f, in_f), lb=lo, ub=hi, vtype=gp.GRB.CONTINUOUS, name=f"ps_grad_w{suffix}"
    )
    for i in range(B):
        for o in range(out_f):
            av = dl_dy.v[i, o]
            for j in range(in_f):
                pv = per_v[i, o, j]
                if variable_x:
                    bv = x.v[i, j]
                    al, au = float(a_l[i, o, j]), float(a_u[i, o, j])
                    bl, bu = float(b_l[i, o, j]), float(b_u[i, o, j])
                    formulation.addConstr(pv >= al * bv + av * bl - al * bl)
                    formulation.addConstr(pv >= au * bv + av * bu - au * bu)
                    formulation.addConstr(pv <= au * bv + av * bl - au * bl)
                    formulation.addConstr(pv <= av * bu + al * bv - al * bu)
                else:
                    formulation.addConstr(pv == float(x.n[i, j]) * av)
    return (BoundedVar(n=per_n, l=lo, u=hi, v=per_v), BoundedVar(n=dl_dy.n, l=dl_dy.l, u=dl_dy.u, v=dl_dy.v))


def clamp(formulation, g, gamma, suffix=""):
    """Exact elementwise clamp(g, -gamma, gamma) via Gurobi general constraints.

    Uses addGenConstrMin/Max rather than a big-M encoding so there is no constant to
    tune and the relation is exact rather than a relaxation.
    """
    flat_n = g.n.ravel()
    n = flat_n.size
    gv = g.v.reshape(n)
    gl, gu = g.l.ravel(), g.u.ravel()
    # mid = max(g, -gamma) ranges over [max(gl,-gamma), max(gu,-gamma)]; declaring it
    # [-gamma, gamma] would impose g <= gamma on the UNCLIPPED gradient, which is a
    # spurious constraint on the model rather than on the clipped output.
    mid_l, mid_u = np.maximum(gl, -gamma), np.maximum(gu, -gamma)
    lo, hi = np.minimum(mid_l, gamma), np.minimum(mid_u, gamma)
    mid = formulation.addMVar(shape=(n,), lb=mid_l, ub=mid_u, name=f"clamp_mid{suffix}")
    out = formulation.addMVar(shape=(n,), lb=lo, ub=hi, name=f"clamp_out{suffix}")
    for i in range(n):
        formulation.addGenConstrMax(mid[i].item(), [gv[i].item()], constant=-gamma)
        formulation.addGenConstrMin(out[i].item(), [mid[i].item()], constant=gamma)
    return BoundedVar(
        n=np.clip(flat_n, -gamma, gamma).reshape(g.n.shape),
        l=lo.reshape(g.n.shape),
        u=hi.reshape(g.n.shape),
        v=out.reshape(g.n.shape),
    )


def _loss_grad(formulation, config, logits, y, margin, suffix):
    """dL/dlogits for the configured loss.

    The distinction matters for what a window of F updates can recover. For hinge,
    dL/dtheta = -y.x.1[active] depends on theta only through the active indicator, so a
    substituted record can saturate the clip at every step regardless of the
    trajectory and there is no cross-step correlation to exploit. For mse,
    dL/dtheta = 2(<theta,x> - y).x varies with theta, so a FIXED substituted record
    cannot independently attain its worst case at each step -- which is precisely the
    constraint a trajectory-global window is there to enforce.
    """
    if config.loss == "mse":
        return F.add_mse_loss_grad_constr(formulation, logits, y, reduction="none", suffix=suffix)
    _, lz = F.add_hinge_loss_constr(
        formulation,
        logits,
        y,
        hinge_prod_encoding=config.hinge_loss_prod_encoding,
        relu_encoding=config.relu_encoding,
        suffix=suffix,
        margin=margin,
    )
    return F.add_hinge_loss_gradient_constr(
        formulation,
        lz,
        y,
        mul_encoding=config.matmul_encoding,
        relu_grad_encoding=config.relu_grad_encoding,
        suffix=suffix,
    )


# ----------------------------------------------------------------- the batch step
def add_batch_clipped_disjoint(
    formulation, mip_model, perturbed_data, data_slice, batch_no, config, learning_rate
):
    """Drop-in replacement for certified_training.add_batch_to_formulation.

    Same signature and contract. Requires the disjoint substitution variables, i.e.
    config.perturbation_constraint == "gradient" with input_domain set. With
    ``use_deep_per_sample`` active it supports multiple trainable layers.
    """
    if config.loss not in ("hinge", "mse"):
        raise NotImplementedError("clipped disjoint path supports hinge and mse")
    gamma = config.gradient_clipping
    if gamma is None:
        raise ValueError("this path exists to apply clipping; gradient_clipping is None")
    margin = config.metadata.get("margin", 1.0)
    st = mip_model.step_

    x_clean = perturbed_data["original_features"][data_slice]
    y_clean = perturbed_data["original_labels"][data_slice]
    subs_idx = perturbed_data["substituted_idx"][data_slice]
    x_pert = perturbed_data["substituted_features"]
    y_pert = perturbed_data["substituted_labels"]
    assign = perturbed_data["batch_assignment"][batch_no]

    def pass_grads(x, y, mask, negate, tag):
        """Forward/backward one branch and return clipped, summed parameter grads."""
        logits = mip_model.add_forward(x, suffix=f"_fwd_{tag}{st}")
        dz = _loss_grad(formulation, config, logits, y, margin, f"_{tag}{st}")
        dz = F.add_masking_constraint(
            formulation,
            dz,
            mask[:, None],
            negate_mask=negate,
            matmul_encoding=config.matmul_encoding,
            suffix=f"_{tag}{st}",
        )
        mip_model.add_backward(dz, suffix=f"_back_{tag}{st}")
        per = mip_model.get_parameter_grads_per_sample()
        summed = []
        for j, pg in enumerate(per):
            flat_shape = (pg.n.shape[0], int(np.prod(pg.n.shape[1:])))
            flat = F.reshape_bounded(pg, flat_shape)
            clipped = clamp(formulation, flat, gamma, f"_{tag}{st}_p{j}")
            clipped = F.reshape_bounded(clipped, pg.n.shape)
            summed.append(F.sum_over_batch(formulation, clipped, suffix=f"_{tag}{st}_p{j}"))
        return summed

    grads_clean = pass_grads(x_clean, y_clean, subs_idx, True, "clean").copy()
    grads_pert = pass_grads(x_pert, y_pert, assign, False, "pert").copy()

    # ---- combine clean + perturbed and take the step
    combined = []
    if len(grads_clean) != len(grads_pert):
        raise RuntimeError("clean and perturbed branches returned different gradients")
    for idx, (ga, gb_) in enumerate(zip(grads_clean, grads_pert)):
        v = formulation.addMVar(shape=ga.n.shape, lb=-gp.GRB.INFINITY, name=f"grad_total_{idx}_{st}")
        formulation.addConstr(v == ga.v + gb_.v)
        combined.append(BoundedVar(n=ga.n + gb_.n, l=ga.l + gb_.l, u=ga.u + gb_.u, v=v))
    return mip_model.step(
        learning_rate, batchsize=config.batchsize, gradient_override=combined, suffix=f"_step{st}"
    )


def formulate_disjoint(config, ibp_model, data, labels):
    """Drop-in replacement for certified_training.formulate_model_and_data.

    Upstream builds the disjoint substitution variables only for loss="hinge"; mse
    goes to add_substitution_constraint_regression, which is the non-disjoint encoding.
    We need the disjoint variables for BOTH, because whether a window of F updates can
    recover anything depends on the loss (see _loss_grad), so the two must be
    comparable under one encoding.
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    env = gp.Env(empty=True)
    try:
        for key, val in _ct.get_gurobi_license_params().items():
            env.setParam(key, val)
    except FileNotFoundError:
        pass
    env.start()
    formulation = gp.Model(env=env)
    for key, value in config.gurobi_kwargs.items():
        formulation.setParam(key, value)
    mip_model = MIPSequential.from_bounded_model(formulation, ibp_model, encodings=config.all_encodings)
    if config.loss == "mse":
        if config.label_domain is None:
            raise ValueError("MSE disjoint substitution requires label_domain")
        variables = _add_regression_substitution_disjoint(
            formulation,
            data,
            labels,
            config.k_substitution,
            config.input_domain,
            config.label_domain,
            max(1, len(data) // config.batchsize),
            config.batchsize,
            config.poisoning_encoding,
        )
    else:
        variables = F.add_substitution_constraint_disjoint(
            formulation,
            data,
            labels,
            k_substitution=config.k_substitution,
            input_domain=config.input_domain,
            n_batches=max(1, len(data) // config.batchsize),
            batchsize=config.batchsize,
            vtype_encoding=config.poisoning_encoding,
        )
    return formulation, mip_model, variables


def _add_regression_substitution_disjoint(
    formulation, x, y, k, input_domain, label_domain, n_batches, batchsize, vtype_encoding
):
    """Regression counterpart of upstream's binary-label disjoint encoder."""
    if input_domain is None:
        raise ValueError("input_domain must be provided")
    vtype = gp.GRB.CONTINUOUS if vtype_encoding == "linear" else gp.GRB.BINARY
    out = {"original_features": BoundedVar(n=x), "original_labels": BoundedVar(n=y)}

    removed = formulation.addMVar(x.shape[0], lb=0, ub=1, vtype=vtype, name="subs_idx")
    formulation.addConstr(removed.sum() <= k, name="privacy_budget")
    out["substituted_idx"] = BoundedVar(
        n=np.zeros(x.shape[0]), l=np.zeros(x.shape[0]), u=np.ones(x.shape[0]), v=removed
    )

    x_l, x_u = input_domain
    y_l, y_u = label_domain
    x_var = formulation.addMVar(
        (k, *x.shape[1:]), lb=x_l, ub=x_u, vtype=gp.GRB.CONTINUOUS, name="poisoned_features"
    )
    y_var = formulation.addMVar(
        (k, *y.shape[1:]), lb=y_l, ub=y_u, vtype=gp.GRB.CONTINUOUS, name="poisoned_labels"
    )
    out["substituted_features"] = BoundedVar(
        n=np.tile(x[0][None, :], (k, 1)),
        l=np.tile(np.asarray(x_l)[None, :], (k, 1)),
        u=np.tile(np.asarray(x_u)[None, :], (k, 1)),
        v=x_var,
    )
    y_shape = (k, *y.shape[1:])
    out["substituted_labels"] = BoundedVar(
        n=np.full(y_shape, (y_l + y_u) / 2), l=np.full(y_shape, y_l), u=np.full(y_shape, y_u), v=y_var
    )

    assign = formulation.addMVar((n_batches, k), lb=0, ub=1, vtype=vtype, name="batch_assignment")
    formulation.addConstr(assign.sum(axis=0) <= 1, name="batch_assignment_sum")
    for b in range(n_batches):
        formulation.addConstr(
            removed[b * batchsize : (b + 1) * batchsize].sum() == assign[b, :].sum(),
            name=f"batch_assignment_balance_{b}",
        )
    out["batch_assignment"] = BoundedVar(
        n=np.zeros((n_batches, k)), l=np.zeros((n_batches, k)), u=np.ones((n_batches, k)), v=assign
    )
    return out


@contextlib.contextmanager
def use_clipped_disjoint():
    """Bind add_batch_clipped_disjoint into certified_training for the duration.

    certified_training calls both by module-level name, so these are the only two
    injection points needed. Explicit, scoped, and restored on exit -- no file in the
    package is changed.
    """
    orig_batch = _ct.add_batch_to_formulation
    orig_form = _ct.formulate_model_and_data
    _ct.add_batch_to_formulation = add_batch_clipped_disjoint
    _ct.formulate_model_and_data = formulate_disjoint
    try:
        yield
    finally:
        _ct.add_batch_to_formulation = orig_batch
        _ct.formulate_model_and_data = orig_form
