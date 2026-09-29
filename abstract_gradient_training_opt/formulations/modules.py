"""MIP encodings of torch.nn modules.

    - torch.nn.ReLU
    - torch.nn.Linear
    - torch.nn.Sequential
    - the Heaviside step function

The encodings are implemented in such a way that each module is typically
initialised once, then add_forward is called to add the forward pass constraints
to the MIP formulation. This should only occur once, as the internal state is
stored to also add backward pass constraints. Once the MIP formulation has been
solved, the forward method can be called to compute the output of the module for
a given input at the current primal value of the parameters.
"""

from __future__ import annotations

import abc
import itertools

import gurobipy as gp
import os

import numpy as np

# Set to '1' to restore the pre-audit (unsound) IBP tightening.
AGT_OPT_TIGHTEN_WITH_IBP = os.environ.get('AGT_OPT_TIGHTEN_WITH_IBP', '0') == '1'
import torch

from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training_opt.formulations import functional
from abstract_gradient_training_opt.formulations.variables import BoundedVar


class MIPModule(abc.ABC):
    """Abstract base class for MIP encodings of torch.nn modules."""

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the module using the primal parameters."""

    @abc.abstractmethod
    def add_forward(self, x: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the forward pass through the module.

        Args:
            x (BoundedVar): The module input decision variable.
            suffix (str): Suffix for variable names.

        Returns:
            y (BoundedVar): The module output decision variable.
        """

    @abc.abstractmethod
    def add_backward(self, dl_dy: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the backward pass through the module.

        Args:
            dl_dy (BoundMVar): The gradient of the loss wrt the module output.
            suffix (str): Suffix for variable names.

        Returns:
            dl_dx (BoundedVar): The gradient of the loss wrt the module input.
        """

    def get_parameter_grads(self) -> list[BoundedVar]:
        """Add constraints for the gradients of the loss.

        The gradients of the loss are given with respect to the parameters of
        the module. Must be called after the backward pass has been added. The
        order of the returned gradients must match the order of the parameters
        returned by the self.get_parameters().

        Returns:
            dl_dp (list[BoundedVar]): The gradients of the loss with respect to
                the parameters.
        """
        return []

    def get_parameters(self) -> list[BoundedVar]:
        """Return the parameters of the module as a list."""
        return []


class MIPReLU(MIPModule):
    """A MIP encoding of the torch.nn.ReLU layer y = relu(x)."""

    def __init__(
        self,
        formulation: gp.Model,
        encodings: dict[str, str],
    ):
        self.formulation = formulation
        self.z: BoundedVar | None = None
        self.forward_encoding = encodings["relu"]
        self.backward_encoding = encodings["relu_grad"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the primal output of the module for the given input."""
        return np.maximum(x, 0)

    def add_forward(self, x: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the forward pass through the module."""
        y, self.z = functional.add_relu_constr(
            self.formulation,
            x,
            relu_encoding=self.forward_encoding,
            suffix=suffix,
        )
        return y

    def add_backward(self, dl_dy: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the backward pass through the module."""
        if self.z is None:
            raise RuntimeError(
                "MIPModule.add_forward must be called before add_backward."
            )
        dl_dx = functional.add_relu_gradient_constr(
            self.formulation,
            self.z,
            dl_dy,
            relu_grad_encoding=self.backward_encoding,
            suffix=suffix,
        )
        return dl_dx


class MIPLinear(MIPModule):
    """A MIP encoding of the torch.nn.Linear layer."""

    def __init__(
        self,
        formulation: gp.Model,
        params: list[BoundedVar],
        encodings: dict[str, str],
    ):
        """Initialise the module."""
        assert len(params) == 2, "MIPLinear requires weight and bias."
        self.formulation = formulation
        self.w: BoundedVar = params[0]
        self.b: BoundedVar = params[1]
        self.x: BoundedVar | None = None
        self.dl_dw: BoundedVar | None = None
        self.dl_db: BoundedVar | None = None
        self.forward_encoding = encodings["matmul"]
        self.backward_encoding = encodings["linear_grad"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the output of the module for the primal parameters."""
        assert self.formulation.status == gp.GRB.OPTIMAL, (
            "Model must be solved before calling forward."
        )
        return x @ self.w.X.T + self.b.X  # type: ignore

    def add_forward(self, x: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the forward pass through the module."""
        self.x = x
        y = functional.add_linear_constr(
            self.formulation,
            x,
            self.w,
            self.b,
            matmul_encoding=self.forward_encoding,
            suffix=suffix,
        )
        return y

    def add_backward(self, dl_dy: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the backward pass through the module."""
        if self.x is None:
            raise ValueError(
                "MIPModule.add_forward must be called before add_backward."
            )
        dl_dx, dl_dw, dl_db = functional.add_linear_gradient_constr(
            self.formulation,
            self.x,
            self.w,
            dl_dy,
            linear_grad_encoding=self.backward_encoding,
            suffix=suffix,
        )
        self.dl_dw, self.dl_db = dl_dw, dl_db
        # Per-sample parameter gradients, kept alongside the batch-summed ones so
        # that clipping can be applied where DP-SGD applies it (to dL/dtheta, per
        # sample) rather than to dL/dlogit.  Only built when the input is constant;
        # see AGS_Ablation/FIX_mip_clipping_site.md.
        self.dl_dw_per = self.dl_db_per = None
        if not isinstance(
            self.x.v, (gp.MVar, gp.MLinExpr, gp.MQuadExpr)
        ):
            self.dl_dw_per, self.dl_db_per = functional.add_per_sample_param_grads(
                self.formulation, self.x, dl_dy, suffix=suffix
            )
        return dl_dx

    def get_parameter_grads(self) -> list[BoundedVar]:
        """Add constraints for the gradients of the loss."""
        if self.dl_dw is None or self.dl_db is None:
            raise ValueError(
                "add_backward must be called before get_parameter_grads."
            )
        return [self.dl_dw, self.dl_db]

    def get_parameter_grads_per_sample(self) -> list[BoundedVar]:
        """Per-sample parameter gradients, shapes (B, out, in) and (B, out)."""
        if self.dl_dw_per is None or self.dl_db_per is None:
            raise NotImplementedError(
                "Per-sample parameter gradients are unavailable. They require "
                "add_backward to have been called with a constant input "
                "(config.input_domain is None)."
            )
        return [self.dl_dw_per, self.dl_db_per]

    def get_parameters(self) -> list[BoundedVar]:
        """Return the parameters of the module as a list."""
        return [self.w, self.b]


class MIPSequential(MIPModule):
    """A MIP encoding of a torch.nn.Sequential model."""

    def __init__(
        self,
        formulation,
        params: list[BoundedVar],
        init_model: torch.nn.Sequential,
        encodings: dict[str, str],
        step_: int = 0,
    ):
        """Initialise the module.

        Construct a MIP encoding of the given torch.nn.Sequential model.
        NOTE: The initial model passed in serves only to determine the structure
        of the model, and its parameter values are not used in the MIP encoding.

        Args:
            formulation (gp.Model): The MIP formulation to add constraints to.
            params (list): The parameters of the model.
            init_model (torch.nn.Sequential): The model structure to encode.
            encodings (dict[str, str]): The encodings to use for each module.
            step_ (int): The current training step, used for naming variables.
        """
        if len(list(init_model.parameters())) != len(params):
            raise ValueError(
                f"Length: {len(list(init_model.parameters()))} != {len(params)}"
            )
        self.formulation = formulation
        self.init_model = init_model
        self.encodings = encodings
        # add all the layers to the MIP formulation
        self.modules: list[MIPModule] = []
        self.step_ = step_
        for module in init_model:
            if isinstance(module, torch.nn.Linear):
                w = params[0]
                b = params[1]
                self.modules.append(
                    MIPLinear(formulation, [w, b], encodings=encodings)
                )
                params = params[2:]
            elif isinstance(module, torch.nn.ReLU):
                self.modules.append(MIPReLU(formulation, encodings=encodings))
            else:
                raise ValueError(f"Unsupported module type: {module}")
        assert not params, "Not all parameters were used in the MIP encoding."

    @staticmethod
    def from_sequential(
        formulation: gp.Model,
        init_model: torch.nn.Module,
        encodings: dict[str, str],
    ) -> MIPSequential:
        """Initialise a MIPSequential model from a torch.nn.Sequential model."""
        if not isinstance(init_model, torch.nn.Sequential):
            raise ValueError(
                f"Expected torch.nn.Sequential, got {type(init_model)=}."
            )
        params: list[BoundedVar] = []
        for p in init_model.parameters():
            p_np = p.detach().cpu().clone().numpy()
            params.append(BoundedVar(n=p_np, l=p_np, u=p_np, v=p_np))
        return MIPSequential(
            formulation, params, init_model, encodings=encodings
        )

    @staticmethod
    def from_bounded_model(
        formulation: gp.Model,
        init_model: BoundedModel,
        encodings: dict[str, str],
    ) -> MIPSequential:
        """Initialise a MIPSequential model from a BoundedModel."""
        params: list[BoundedVar] = []
        for i, (p_l, p_n, p_u) in enumerate(
            zip(init_model.param_l, init_model.param_n, init_model.param_u)
        ):
            p_l_np = p_l.detach().cpu().clone().numpy()
            p_n_np = p_n.detach().cpu().clone().numpy()
            p_u_np = p_u.detach().cpu().clone().numpy()
            if np.all(p_l_np == p_u_np):
                p_v = p_n_np
            else:
                p_v = formulation.addMVar(
                    shape=p_l_np.shape,
                    lb=p_l_np,
                    ub=p_u_np,
                    vtype=gp.GRB.CONTINUOUS,
                    name=f"model_param_init_{i}",
                )
            params.append(BoundedVar(n=p_n_np, l=p_l_np, u=p_u_np, v=p_v))
        return MIPSequential(
            formulation,
            params,
            torch.nn.Sequential(*init_model.modules),
            encodings=encodings,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the primal output of the module."""
        for module in self.modules:
            x = module.forward(x)
        return x

    def add_forward(self, x: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the forward pass through the module."""
        for i, module in enumerate(self.modules):
            x = module.add_forward(
                x, suffix=f"{suffix}_layer{i}_step{self.step_}"
            )
        return x

    def add_backward(self, dl_dy: BoundedVar, suffix: str = "") -> BoundedVar:
        """Add the constraints for the backward pass through the module."""
        for i, module in enumerate(reversed(self.modules)):
            dl_dy = module.add_backward(
                dl_dy, suffix=f"{suffix}_layer{i}_step{self.step_}"
            )
        return dl_dy

    def get_parameter_grads(self) -> list[BoundedVar]:
        """Add constraints for the gradients of the loss wrt the parameters."""
        return list(
            itertools.chain.from_iterable(
                module.get_parameter_grads() for module in self.modules
            )
        )

    def get_parameter_grads_per_sample(self) -> list[BoundedVar]:
        """Per-sample parameter grads. Single trainable layer only.

        Per-sample gradients through a deeper stack would need per-sample
        backpropagation of dl_dx, which the module chain does not carry. Study 1
        (linear head on identity features) and the LLM-head setting are both
        single-layer; anything deeper raises rather than silently returning the
        wrong object.
        """
        trainable = [m for m in self.modules if m.get_parameters()]
        if len(trainable) != 1:
            raise NotImplementedError(
                f"Per-sample parameter gradients support exactly one trainable "
                f"layer, found {len(trainable)}."
            )
        return trainable[0].get_parameter_grads_per_sample()

    def get_parameters(self) -> list[BoundedVar]:
        """Return the parameters of the module as a list."""
        return list(
            itertools.chain.from_iterable(
                module.get_parameters() for module in self.modules
            )
        )

    def step(
        self,
        learning_rate: float,
        batchsize: int,
        gradient_override: list[BoundedVar] | None = None,
        suffix: str = "",
    ) -> MIPSequential:
        """Return a copy of this model with the parameters updated using SGD."""
        params = []
        self.step_ += 1
        if gradient_override is None:
            grads = self.get_parameter_grads()
        else:
            grads = gradient_override
        if grads is None or len(grads) != len(self.get_parameters()):
            raise ValueError(
                f"Expected {len(self.get_parameters())} gradients, "
                f"got {len(grads) if grads is not None else None}."
            )
        for i, (p, g) in enumerate(zip(self.get_parameters(), grads)):
            assert p.shape == g.shape
            p_next_l = p.l - learning_rate * g.u / batchsize
            p_next_n = p.n - learning_rate * g.n / batchsize
            p_next_u = p.u - learning_rate * g.l / batchsize
            p_next_v = self.formulation.addMVar(
                shape=p.n.shape,
                lb=p_next_l,
                ub=p_next_u,
                vtype=gp.GRB.CONTINUOUS,
                name=f"model_param_step{self.step_}_{i}",
            )
            self.formulation.addConstr(
                p_next_v == p.v - (learning_rate / batchsize) * g.v,  # type: ignore
                name=f"param_update_step_{suffix}",
            )
            p_next = BoundedVar(n=p_next_n, l=p_next_l, u=p_next_u, v=p_next_v)
            params.append(p_next)
        return MIPSequential(
            self.formulation,
            params,
            self.init_model,
            self.encodings,
            step_=self.step_,
        )

    def add_parameter_bounds(
        self,
        param_l: list[np.ndarray] | list[torch.Tensor],
        param_n: list[np.ndarray] | list[torch.Tensor],
        param_u: list[np.ndarray] | list[torch.Tensor],
    ) -> None:
        """Add bounds to the parameters of the model.

        The nominal values are used to validate that the variables match.

        Args:
            param_l (list[np.ndarray]): The lower bounds for the parameters.
            param_n (list[np.ndarray]): The nominal values for the parameters.
            param_u (list[np.ndarray]): The upper bounds for the parameters.
        """
        param_l = [np.asarray(p) for p in param_l]
        param_n = [np.asarray(p) for p in param_n]
        param_u = [np.asarray(p) for p in param_u]
        if len(param_l) != len(self.get_parameters()):
            raise ValueError(
                f"Expected {len(self.get_parameters())}, got {len(param_l)}."
            )
        if len(param_u) != len(self.get_parameters()):
            raise ValueError(
                f"Expected {len(self.get_parameters())}, got {len(param_u)}."
            )
        self.formulation.update()
        for p, l, n, u in zip(self.get_parameters(), param_l, param_n, param_u):
            # SOUNDNESS: tightening the MIP variables with externally supplied
            # (IBP) bounds is only valid if both refer to the same step of the
            # same trajectory. Empirically they do not: the intersection can
            # exclude the nominal trajectory (BoundedVar warns "Invalid bounds
            # for MVar", violated by up to 1.3e-2 on a 4-step problem), and the
            # resulting clamp cuts genuinely reachable parameters -- the solver
            # then reports a maximum BELOW the nominal.
            # See AGS_Ablation/screen/s7_milp_probe.py.
            # Default: update the nominal only, and leave the MIP's own bounds.
            p.n = n
            if AGT_OPT_TIGHTEN_WITH_IBP:
                l = np.maximum(l, p.l)
                u = np.minimum(u, p.u)
                p.v.lb = l  # type: ignore
                p.v.ub = u  # type: ignore
                p.l = l
                p.u = u


def main():
    """Sanity check for the MIPSequential class."""
    # create a simple torch.nn.Sequential model
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1),
    )
    enc = {
        "relu_encoding": "bigm",
        "relu_grad_encoding": "bilinear",
        "matmul_encoding": "bilinear",
        "linear_grad_encoding": "bilinear",
    }
    # create some random input data
    x = np.random.randn(10, 2).astype(np.float32)
    x_torch = torch.tensor(x, requires_grad=True)
    # create a gurobi model
    formulation = gp.Model()
    # create the MIP encoding of the model
    mip_model = MIPSequential.from_sequential(formulation, model, encodings=enc)
    # add the MVar for the model input
    x_v = formulation.addMVar(
        shape=x.shape, lb=x, ub=x, vtype=gp.GRB.CONTINUOUS
    )
    x = BoundedVar(n=x, l=x - 0.1, u=x + 0.1, v=x_v)
    # add the forward pass constraints
    y = mip_model.add_forward(x)
    # solve the model
    formulation.optimize()
    # check the output
    assert np.allclose(model(x_torch).detach().numpy(), y.X, atol=1e-5)
    assert np.allclose(
        model(x_torch).detach().numpy(), mip_model.forward(x.n), atol=1e-5
    )
    # pass through a loss function
    loss = 0.5 * model.forward(x_torch).square().sum()
    grad, *grads = torch.autograd.grad(
        loss, [x_torch, *list(model.parameters())]
    )  # type: ignore
    # add the backward pass constraints to the mip encoding
    dl_dx = mip_model.add_backward(y)
    dl_dp = mip_model.get_parameter_grads()
    formulation.optimize()
    # check the output
    assert np.allclose(grad, dl_dx.X, atol=1e-5)
    for grad, dp in zip(grads, dl_dp):
        assert np.allclose(grad.detach().numpy(), dp.X, atol=1e-5)


if __name__ == "__main__":
    main()
