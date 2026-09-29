"""Base class for pytorch models with bounded parameters."""

from __future__ import annotations

import abc
import itertools
import logging

import numpy as np
import torch

from abstract_gradient_training.bounded_losses import BoundedLoss
from abstract_gradient_training.bounded_models import nominal_modules

LOGGER = logging.getLogger(__name__)


class BoundedModel:
    """Base class for bounded pytorch models.

    Parent class for all bounded models. Sub-classes must implement the
    bound_forward and bound_backward methods.
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        *,
        transform: BoundedModel | None = None,
        trainable: bool = True,
    ):
        """Initialize the bounded model.

        Args:
        model (torch.nn.Sequential): Pytorch sequential model.
        transform (BoundedModel, optional): Fixed (non-trainable) transform
            which we pass the input through before the main model. This is
            useful for fine-tuning the last few layers of a pre-trained model.
        trainable (bool, optional): Flag to indicate if the model is trainable.
            If False, the model will not support gradient computation and all
            parameters will be fixed (non-interval).
        """
        self.modules = []
        for module in model:
            if isinstance(
                module, torch.nn.Dropout
            ):  # dropouts need a special wrapper that stores the mask
                self.modules.append(nominal_modules.DropoutWrapper(module))
            else:
                self.modules.append(module)
        # any dropout modules must be handled separately
        self.transform = transform
        self.trainable = trainable
        # store parameter lists for the nominal, upper and lower bounds.
        # These differ from the param_n, param_l and param_u properties in that
        # they are stored as a list of tuples of parameters corresponding to
        # each module, rather than a flat list.
        self._param_n = [
            [p.detach().clone() for p in list(module.parameters())]
            for module in model
        ]
        if (
            self.trainable
        ):  # trainable models need to support bounds on the parameters
            self._param_l = [
                [p.detach().clone() for p in list(module.parameters())]
                for module in model
            ]
            self._param_u = [
                [p.detach().clone() for p in list(module.parameters())]
                for module in model
            ]
        else:
            self._param_l = self._param_n
            self._param_u = self._param_n
        # attributes that cache intermediate values from the forward pass
        self._inter_n = self._inter_l = self._inter_u = None
        self._inter_grads_n = self._inter_grads_l = self._inter_grads_u = None

    @abc.abstractmethod
    def bound_forward(
        self,
        x_l: torch.Tensor,
        x_u: torch.Tensor,
        retain_intermediate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded forward pass of the model.

        Args:
            x_l (torch.Tensor): Lower bound of inputs to the model.
            x_u (torch.Tensor): Upper bound of inputs to the model.
            retain_intermediate (bool, optional): Flag to indicate if
                intermediate values should be cached by the model. If True,
                the model stores the intermediate bounds of the model which are
                required for the backward pass.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the
                output of the model.
        """

    @abc.abstractmethod
    def bound_backward(
        self,
        dl_dy_l: torch.Tensor,
        dl_dy_u: torch.Tensor,
        retain_intermediate: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Bounded backward pass of the model.

        Note that the gradient bounds are computed with respect to the last
        forward pass call with retain_intermediate=True.

        Args:
            dl_dy_l (torch.Tensor): Lower bound of the gradient of the loss with
                respect to the output of the model.
            dl_dy_u (torch.Tensor): Upper bound of the gradient of the loss with
                respect to the output of the model.
            retain_intermediate (bool, optional): Flag to indicate if the
                gradients with respect to the intermediate variables should be
                stored.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Lower and upper
                bounds on the gradients of the loss with respect to the
                parameters of the model. The lists of the gradient tensors
                should be in the same order as that returned by the param_n,
                param_l, and param_u properties.
        """

    def forward(
        self, x: torch.Tensor, retain_intermediate: bool = False
    ) -> torch.Tensor:
        """Nominal forward pass of the model.

        Args:
            x (torch.Tensor): Inputs to the model.
            retain_intermediate (bool, optional): Flag to indicate if
                intermediate values should be cached by the model. If True, the
                model stores the intermediate values of the model which are
                required for the backward pass.

        Returns:
            torch.Tensor: Output of the model.
        """
        if retain_intermediate and not self.trainable:
            raise ValueError(
                f"Intermediate values shouldn't be retained {self.trainable=}."
            )
        # pass the input through the fixed transform if required
        if self.transform is not None:
            x = self.transform.forward(x, retain_intermediate=False)
        # list to store the intermediate values from the forward pass
        inter = [x]
        # loop over the modules in the model and propagate the input interval
        for module, params in zip(self.modules, self._param_n):
            x = nominal_modules.module_forward_pass(
                x, params, module, retain_intermediate
            )
            inter.append(x)
        # cache the intermediate values if required
        if retain_intermediate:
            self._inter_n = inter

        return x

    def backward(
        self, dl_dy: torch.Tensor, retain_intermediate: bool = False
    ) -> list[torch.Tensor]:
        """Nominal backward pass of the model.

        Unlike standard pytorch autograd, we compute *per sample* gradients,
        i.e. we don't reduce over the batch dimension. Note that the gradients
        are computed with respect to the last forward pass call with
        retain_intermediate=True.

        Args:
            dl_dy (torch.Tensor): Gradient of the loss with respect to the
                output of the model.
            retain_intermediate (bool, optional): Flag to indicate if the
                gradients with respect to the intermediate variables should be
                stored.

        Returns:
            list[torch.Tensor]: List of gradient tensors of the loss with
                respect to the parameters of the model. The list of the gradient
                tensors should be in the same order as that returned by the
                param_n, param_l, and param_u properties.
        """
        if not self.trainable:
            raise ValueError(
                "Model not trainable, so backward computations not supported."
            )
        if self._inter_n is None:
            raise ValueError(
                "Forward pass with retain_intermediate=True must"
                "be called before backward pass."
            )

        # compare the output of the forward pass with the grads passed to the
        # backward pass
        output = self._inter_n.pop()
        if output.shape != dl_dy.shape:
            raise ValueError(
                f"Unexpected gradient shape: {output.shape=} != {dl_dy.shape=}"
            )

        # list to store intermediate gradients
        inter_grads = []

        # list to store the grads wrt the parameters
        grads_params = []

        # loop over the modules in reverse order and back-propagate gradients
        for module, params, inter in zip(
            reversed(self.modules),
            reversed(self._param_n),
            reversed(self._inter_n),
        ):
            grads = nominal_modules.module_backward_pass(
                dl_dy, inter, params, module
            )
            dl_dy = grads.pop(0)
            inter_grads.append(dl_dy)
            grads_params.extend(grads)

        if retain_intermediate:
            # save the gradients wrt the intermediate values
            inter_grads.reverse()
            self._inter_grads_n = inter_grads
        else:
            # delete the intermediate values from the previous forward pass
            self._inter_n = None

        # reverse and return the list of gradients wrt the parameters
        grads_params.reverse()
        return grads_params

    def bound_backward_combined(
        self,
        x_l: torch.Tensor,
        x_u: torch.Tensor,
        labels: torch.Tensor,
        loss: BoundedLoss,
        *,
        label_k_poison: int = 0,
        label_epsilon: float = 0.0,
        poison_target_idx: int = -1,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute bounds on the backward pass (gradients) of the model.

        This combines the forward pass in a single call. This has no impact on
        models that consider the bounding passes separately, but can be
        over-written for bounding methods that require combining the forward and
        backward bounds. The bounds are computed wrt the input bounds and the
        specified poisoning attack. See BoundedLoss for details of the poisoning
        attack parameters.

        Args:
            x_l (torch.Tensor): Lower bound of the inputs to the model.
            x_u (torch.Tensor): Upper bound of the inputs to the model.
            labels (torch.Tensor): Target labels for the loss function.
            loss (BoundedLoss): Loss function that supports bounds.
            label_k_poison (int, optional): Maximum number of data-points with
                poisoned targets.
            label_epsilon (float, optional): Maximum perturbation of the targets
                (in the inf norm).
            poison_target_idx (int, optional): Target class for the poisoning
                attack.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Lower and upper
                bounds on the gradients of the loss with respect to the
                parameters of the model. The lists of the gradient tensors
                should be in the same order as that returned by the param_n,
                param_l, and param_u properties.
        """
        logits_l, logits_u = self.bound_forward(
            x_l, x_u, retain_intermediate=True
        )
        dl_l, dl_u = loss.bound_backward(
            logits_l,
            logits_u,
            labels,
            label_k_poison=label_k_poison,
            label_epsilon=label_epsilon,
            poison_target_idx=poison_target_idx,
        )
        return self.bound_backward(dl_l, dl_u)

    def to(self, destination: torch.device | torch.dtype | str) -> BoundedModel:
        """Move the model to the specified device or dtype.

        Note that unlike standard torch this is an in-place operation.

        Args:
            destination (torch.device | torch.dtype | str): Device or dtype to
                which the model should be moved.
        """
        self._param_n = [
            [p.to(destination) for p in module] for module in self._param_n
        ]
        self._param_l = [
            [p.to(destination) for p in module] for module in self._param_l
        ]
        self._param_u = [
            [p.to(destination) for p in module] for module in self._param_u
        ]
        if self._inter_n is not None:
            self._inter_n = [i.to(destination) for i in self._inter_n]
        if self._inter_l is not None:
            self._inter_l = [i.to(destination) for i in self._inter_l]
        if self._inter_u is not None:
            self._inter_u = [i.to(destination) for i in self._inter_u]
        if self.transform is not None:
            self.transform = self.transform.to(destination)
        return self

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.param_n[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Return the device of the model."""
        return self.param_n[0].dtype

    @property
    def param_l(self) -> list[torch.Tensor]:
        """Flatten the list of parameter lower bounds."""
        if not self.trainable:
            raise ValueError(
                "Model is not trainable, so parameter bounds are not available."
            )
        return list(itertools.chain.from_iterable(self._param_l))

    @property
    def param_n(self) -> list[torch.Tensor]:
        """Flatten the list of parameter lower bounds."""
        return list(itertools.chain.from_iterable(self._param_n))

    @property
    def param_u(self) -> list[torch.Tensor]:
        """Flatten the list of parameter lower bounds."""
        if not self.trainable:
            raise ValueError(
                "Model is not trainable, so parameter bounds are not available."
            )
        return list(itertools.chain.from_iterable(self._param_u))

    def save_params(self, filename: str) -> None:
        """Save the model parameters to a pytorch file.

        Args:
            filename (str): Path to the file where the parameters are saved.
        """
        torch.save(
            {
                "param_n": self._param_n,
                "param_l": self._param_l,
                "param_u": self._param_u,
            },
            filename,
        )

    def load_params(self, filename: str) -> None:
        """Load the parameters from the file.

        Args:
            filename (str): Path to a file where the parameters are saved.
        """
        params = torch.load(filename, weights_only=True)
        # validate shapes
        if len(params["param_n"]) != len(self._param_n):
            raise RuntimeError(
                f"Loaded parameters do not match the model,"
                f"{len(params['param_n'])} != {len(self._param_n)}."
            )
        for p, p_n in zip(self._param_n, params["param_n"]):
            for p1, p2 in zip(p, p_n):
                if p1.shape != p2.shape:
                    raise RuntimeError(
                        "Loaded parameters do not match the model."
                    )

        self._param_n = params["param_n"]
        self._param_l = params["param_l"]
        self._param_u = params["param_u"]

    def apply_parameter_update(
        self,
        param_l: list[torch.Tensor] | list[np.ndarray],
        param_n: list[torch.Tensor] | list[np.ndarray],
        param_u: list[torch.Tensor] | list[np.ndarray],
    ) -> None:
        """Apply the parameter update to the model parameters.

        Args:
            param_l (list[torch.Tensor]): Lower bounds of the parameters.
            param_n (list[torch.Tensor]): Nominal values of the parameters.
            param_u (list[torch.Tensor]): Upper bounds of the parameters.
        """
        param_l = [
            p if isinstance(p, torch.Tensor) else torch.tensor(p)
            for p in param_l
        ]
        param_n = [
            p if isinstance(p, torch.Tensor) else torch.tensor(p)
            for p in param_n
        ]
        param_u = [
            p if isinstance(p, torch.Tensor) else torch.tensor(p)
            for p in param_u
        ]
        for i in range(len(self.param_n)):
            if param_n[i].shape != self.param_n[i].shape:
                raise RuntimeError("Loaded parameters do not match the model.")
            if not torch.all(param_l[i] <= param_n[i]):
                LOGGER.warning(
                    "Loaded parameters are not valid bounds"
                    f"{(param_n[i] - param_l[i]).min()=}"
                )
                param_l[i] = torch.minimum(param_l[i], param_n[i])
            if not torch.all(param_n[i] <= param_u[i]):
                LOGGER.warning(
                    "Loaded parameters are not valid bounds"
                    f"{(param_u[i] - param_n[i]).min()=}"
                )
                param_u[i] = torch.maximum(param_u[i], param_n[i])
            self.param_l[i].data.copy_(param_l[i].to(self.device))
            self.param_n[i].data.copy_(param_n[i].to(self.device))
            self.param_u[i].data.copy_(param_u[i].to(self.device))
