"""An implementation of a multi-step interval domain for certified training."""

import torch

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounded_models import BoundedModel
from abstract_gradient_training.bounded_optimizers import BoundedOptimizer
from abstract_gradient_training.configuration import AGTConfig


class MultiStepPrivacyAccumulator(BoundedOptimizer):
    """This class serves as a tighter abstract domain for parameter domains.

    It is used in privacy certified training to compute guarantees on the
    reachable parameter space with respect to k_private removals / additions
    to the training data jointly over the entire dataset.

    This differs from the standard interval domain, that considers the effect of
    k_private removals / additions to each previously seen batch independently.

    This aim of this class is for it to be used in place of both the
    BoundedOptimizer and the PrivacyGradientAccumulator classes within the
    privacy_certified_training function. It should be able to handle the logic
    of accumulating gradients across batch fragments, computing the certified
    descent direction bounds, and concretizing the updated
    parameter interval bounds at each iteration.

    To simplify implementation, for now we will consider only the simplest
    optimizer set-up with a fixed learning rate and no momentum.

    The domain accumulates gradient updates to track a parameter domain of the
    following form:

        theta^0 - alpha / B sum_i grads_1[i]
                - alpha / B sum_i grads_2[i] - ...
                - alpha / B sum_i grads_n[i]

    and concretizes the domain to the interval:

        dtheta_l = SEMin_{N - k} (grads_l_1 U ... U grads_l_n) - k * gamma
        dtheta_u = SEMax_{N - k} (grads_u_1 U ... U grads_u_n) + k * gamma

        theta_l = theta^0 - alpha / B dtheta_u
        theta_u = theta^0 - alpha / B dtheta_l
    """

    def __init__(
        self,
        bounded_model: BoundedModel,
        config: AGTConfig,
    ):
        """Initialise the multi-step domain.

        Args:
            bounded_model (BoundedModel): The bounded model the parameter domain
                is tracking.
            config (AGTConfig): Training configuration object.
        """
        # disable most of the optimizer options for now
        if config.optimizer != "SGD":
            raise NotImplementedError(
                "MultiStepDomain only supports the SGD optimizer."
            )
        if config.optimizer_kwargs:
            raise NotImplementedError(
                "MultiStepDomain does not support optimizer settings."
            )
        if config.l1_reg != 0 or config.l2_reg != 0:
            raise NotImplementedError(
                "MultiStepDomain does not support L1 or L2 regularization."
            )
        if config.lr_decay != 0 or config.lr_min != 0:
            raise NotImplementedError(
                "MultiStepDomain does not support learning rate decay."
            )

        init_params = bounded_model.param_n
        self.bounded_model = bounded_model
        self.gamma = config.clip_gamma
        self.k_private = config.k_private
        self.lr = config.learning_rate
        self.n_samples = 0
        self.epoch_count = -1
        # we'll calculate this once we've accumulated the first batch
        self.batchsize = None
        self.theta_l = [param.clone().detach() for param in init_params]
        self.theta_n = [param.clone().detach() for param in init_params]
        self.theta_u = [param.clone().detach() for param in init_params]

        # variables to store the accumulated updates to the parameters
        self.update_l = [torch.zeros_like(param) for param in init_params]
        self.update_n = [torch.zeros_like(param) for param in init_params]
        self.update_u = [torch.zeros_like(param) for param in init_params]

        # we also want to store the largest k lower bounds and the smallest k
        # upper bounds. These are the ones that are
        # replaced by +/- gamma when concretizing our parameter interval
        self.grads_l_max_k = [
            torch.zeros(0, *p.shape, dtype=p.dtype, device=p.device)
            for p in init_params
        ]
        self.grads_u_min_k = [
            torch.zeros(0, *p.shape, dtype=p.dtype, device=p.device)
            for p in init_params
        ]

    def add_public_fragment_gradients(
        self,
        frag_grads_n: list[torch.Tensor],
        frag_grads_l: list[torch.Tensor],
        frag_grads_u: list[torch.Tensor],
    ) -> None:
        """Accumulate the gradients for a fragment of public data."""
        raise NotImplementedError(
            "MultiStepDomain does not support separate public data."
        )

    def add_private_fragment_gradients(
        self,
        frag_grads_n: list[torch.Tensor],
        frag_grads_l: list[torch.Tensor],
        frag_grads_u: list[torch.Tensor],
    ) -> None:
        """Accumulate the gradients for a fragment of private data.

        Args:
            frag_grads_n: The nominal gradients for the fragment.
            frag_grads_l: The lower bound gradients with weight perturbation.
            frag_grads_u: The upper bound gradients with weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_l,
            frag_grads_u,
            frag_grads_n,
            msg="grad bounds, private fragment",
        )
        # update the number of samples we've seen in this batch
        self.n_samples += frag_grads_n[0].size(0)
        # for each parameter
        for i in range(len(self.theta_n)):
            # accumulate the nominal gradients
            self.update_n[i] += frag_grads_n[i].sum(dim=0)
            # take the top k lower and bottom k upper bounds from the fragment
            k = min(frag_grads_l[i].size(0), self.k_private)
            grad_l_max_k = torch.topk(
                frag_grads_l[i], k, largest=True, dim=0, sorted=False
            ).values
            grad_u_min_k = torch.topk(
                frag_grads_u[i], k, largest=False, dim=0, sorted=False
            ).values
            # accumulate the rest of the gradients from the fragment
            self.update_l[i] += frag_grads_l[i].sum(dim=0) - grad_l_max_k.sum(
                dim=0
            )
            self.update_u[i] += frag_grads_u[i].sum(dim=0) - grad_u_min_k.sum(
                dim=0
            )
            # now we concatenate the top/bottom k gradients from the fragment
            # to the top/bottom k gradients we have seen
            frag_grads_l[i] = torch.cat(
                [grad_l_max_k, self.grads_l_max_k[i]], dim=0
            )
            frag_grads_u[i] = torch.cat(
                [grad_u_min_k, self.grads_u_min_k[i]], dim=0
            )
            # take the new top k lower and bottom k upper bound gradients
            k = min(frag_grads_l[i].size(0), self.k_private)
            self.grads_l_max_k[i] = torch.topk(
                frag_grads_l[i], k, largest=True, dim=0, sorted=False
            ).values
            self.grads_u_min_k[i] = torch.topk(
                frag_grads_u[i], k, largest=False, dim=0, sorted=False
            ).values
            # accumulate the required sums to the bounds
            self.update_l[i] += frag_grads_l[i].sum(dim=0) - self.grads_l_max_k[
                i
            ].sum(dim=0)
            self.update_u[i] += frag_grads_u[i].sum(dim=0) - self.grads_u_min_k[
                i
            ].sum(dim=0)

        interval_arithmetic.validate_interval(
            self.update_l,
            self.update_u,
            self.update_n,
            msg="grad update bounds",
        )

    def concretize_gradient_update(
        self, batchsize: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Compute the gradient update for the current batch.

        This function is called at the end of the batch, but doesn't have much
        significance for this multi-step domain. Here it only serves to indicate
        the end of the batch and to validate the batchsize. We return a dummy
        value that is not used in any computations involving the multi-step
        domain.
        """
        assert self.n_samples == batchsize, (
            "Batchsize does not match the number of samples seen in the batch."
        )
        if self.batchsize is None:
            self.batchsize = batchsize
        assert self.batchsize == batchsize, (
            f"Size of batch does not match. "
            f"Got {batchsize}, expected {self.batchsize}."
        )
        # reset the number of samples seen for the next batch
        self.n_samples = 0
        # this is a dummy return value not used in the multi-step domain
        e = [torch.zeros(1)]
        return e, e, e

    def step(
        self,
        mean_grads_l: list[torch.Tensor],
        mean_grads_n: list[torch.Tensor],
        mean_grads_u: list[torch.Tensor],
    ) -> None:
        """Apply the optimizer update to the parameter values and bounds.

        The update should be applied in-place with
        respect to the BoundedModel that is tracked by this class.

        The gradients passed in here are dummy values and aren't used, since the
        actual gradients are accumulated internally in this class.
        """
        # dummy values
        assert mean_grads_l == mean_grads_n == mean_grads_u == [torch.zeros(1)]
        assert self.batchsize is not None, "Batchsize has not been set."

        # for each parameter
        for i in range(len(self.theta_n)):
            # compute new parameter values
            p_n = self.theta_n[i] - self.lr * self.update_n[i] / self.batchsize
            p_u = (
                self.theta_u[i]
                - self.lr
                * (self.update_l[i] - self.gamma * self.k_private)
                / self.batchsize
            )
            p_l = (
                self.theta_l[i]
                - self.lr
                * (self.update_u[i] + self.gamma * self.k_private)
                / self.batchsize
            )
            # update the values in-place
            self.bounded_model.param_n[i].copy_(p_n)
            self.bounded_model.param_l[i].copy_(p_l)
            self.bounded_model.param_u[i].copy_(p_u)

    def epoch_start(self) -> None:
        """Called at the start of each epoch to update any optimizer state.

        Here, we must concretize the updates from the previous epoch, as we must
        take the worst case k_private at each epoch independently.
        """
        self.epoch_count += 1

        if self.epoch_count == 0:
            return

        for i in range(len(self.theta_n)):
            self.theta_n[i] -= self.lr * self.update_n[i] / self.batchsize
            self.theta_l[i] -= (
                self.lr
                * (self.update_u[i] + self.gamma * self.k_private)
                / self.batchsize
            )
            self.theta_u[i] -= (
                self.lr
                * (self.update_l[i] - self.gamma * self.k_private)
                / self.batchsize
            )
            assert torch.allclose(
                self.theta_n[i], self.bounded_model.param_n[i]
            )
            assert torch.allclose(
                self.theta_l[i], self.bounded_model.param_l[i]
            )
            assert torch.allclose(
                self.theta_u[i], self.bounded_model.param_u[i]
            )

        # reset the accumulated updates for the past epoch
        self.update_l = [torch.zeros_like(param) for param in self.theta_n]
        self.update_n = [torch.zeros_like(param) for param in self.theta_n]
        self.update_u = [torch.zeros_like(param) for param in self.theta_n]
        self.grads_l_max_k = [
            torch.zeros(0, *p.shape, dtype=p.dtype, device=p.device)
            for p in self.theta_n
        ]
        self.grads_u_min_k = [
            torch.zeros(0, *p.shape, dtype=p.dtype, device=p.device)
            for p in self.theta_n
        ]


class MultiStepPoisoningAccumulator(BoundedOptimizer):
    """A multi-step interval domain for poison certified training.

    This class serves as a tighter abstract domain for propagating parameter
    domains through multiple time-steps of training. It is used in poison
    certified training to compute guarantees on the reachable parameter space
    with respect to k_poison modifications to the training data jointly over
    the entire dataset.

    This differs from the standard interval domain, that considers the effect of
    k_poison modifications to each previously seen batch independently.

    This aim of this class is for it to be used in place of both the
    BoundedOptimizer and PoisoningGradientAccumulator classes within the
    poison_certified_training function. It should be able to handle the logic
    of accumulating gradients across batch fragments, computing the certified
    descent direction bounds, and concretizing the updated
    parameter interval bounds at each iteration.

    To simplify implementation, for now we will consider only the simplest
    optimizer set-up with a fixed learning rate and no momentum.
    """

    def __init__(
        self,
        bounded_model: BoundedModel,
        config: AGTConfig,
        k_poison: int,
    ):
        """Initialise the multi-step domain.

        Args:
            bounded_model: The bounded model the parameter domain is tracking.
            config (AGTConfig): Training configuration object.
            k_poison (int): The number of poisoned samples to consider.
        """
        # disable most of the optimizer options for now
        if config.optimizer != "SGD":
            raise NotImplementedError(
                "MultiStepDomain only supports the SGD optimizer."
            )
        if config.optimizer_kwargs:
            raise NotImplementedError(
                "MultiStepDomain does not support optimizer settings."
            )
        if config.l1_reg != 0 or config.l2_reg != 0:
            raise NotImplementedError(
                "MultiStepDomain does not support L1 or L2 regularization."
            )
        if config.lr_decay != 0 or config.lr_min != 0:
            raise NotImplementedError(
                "MultiStepDomain does not support learning rate decay."
            )

        init_params = bounded_model.param_n
        self.bounded_model = bounded_model
        self.k_poison = k_poison
        self.lr = config.learning_rate
        self.n_samples = 0
        self.epoch_count = -1
        # we'll calculate this once we've accumulated the first batch
        self.batchsize = None
        self.theta_l = [param.clone().detach() for param in init_params]
        self.theta_n = [param.clone().detach() for param in init_params]
        self.theta_u = [param.clone().detach() for param in init_params]

        # variables to store the accumulated updates to the parameters
        self.update_l = [torch.zeros_like(param) for param in init_params]
        self.update_n = [torch.zeros_like(param) for param in init_params]
        self.update_u = [torch.zeros_like(param) for param in init_params]

        # store the min-k and max-k differences between the input+weight
        # perturbed and weight perturbed bounds that we have seen so far
        self.diffs_l = [
            torch.empty(0, *p.size(), dtype=p.dtype, device=p.device)
            for p in init_params
        ]
        self.diffs_u = [
            torch.empty(0, *p.size(), dtype=p.dtype, device=p.device)
            for p in init_params
        ]

    def add_clean_fragment_gradients(
        self,
        frag_grads_n: list[torch.Tensor],
        frag_grads_wp_l: list[torch.Tensor],
        frag_grads_wp_u: list[torch.Tensor],
    ) -> None:
        """Accumulate the gradients for a fragment of clean data."""
        raise NotImplementedError(
            "MultiStepDomain does not support separate clean data."
        )

    def add_poisoned_fragment_gradients(
        self,
        frag_grads_n: list[torch.Tensor],
        frag_grads_wp_l: list[torch.Tensor],
        frag_grads_wp_u: list[torch.Tensor],
        frag_grads_iwp_l: list[torch.Tensor],
        frag_grads_iwp_u: list[torch.Tensor],
    ) -> None:
        """Accumulate the gradients for a fragment of potentially poisoned data.

        Args:
            frag_grads_n: The nominal gradients for the fragment.
            frag_grads_wp_l: Lower bound grads with weight perturbation.
            frag_grads_wp_u: Upper bound grads with weight perturbation.
            frag_grads_iwp_l: Lower bound grads with input+weight perturbation.
            frag_grads_iwp_u: Upper bound grads with input+weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_wp_l,
            frag_grads_wp_u,
            frag_grads_n,
            msg="wp grad bounds, poisoned fragment",
        )
        interval_arithmetic.validate_interval(
            frag_grads_iwp_l,
            frag_grads_iwp_u,
            frag_grads_n,
            msg="i+wp grad bounds, poisoned fragment",
        )

        self.n_samples += frag_grads_n[0].size(
            0
        )  # update the number of samples we've seen in this batch
        for i in range(len(self.theta_n)):
            # add weight perturbed bounds directly
            self.update_l[i] += frag_grads_wp_l[i].sum(dim=0)
            self.update_n[i] += frag_grads_n[i].sum(dim=0)
            self.update_u[i] += frag_grads_wp_u[i].sum(dim=0)
            # calculate the differences beetween the input+weight perturbed and
            # weight perturbed bounds
            frag_diff_l = frag_grads_iwp_l[i] - frag_grads_wp_l[i]
            frag_diff_u = frag_grads_iwp_u[i] - frag_grads_wp_u[i]
            # calculate the bottom/top-k differences for each parameter
            k = min(frag_diff_l.size(0), self.k_poison)
            frag_diff_l = torch.topk(
                frag_diff_l, k, dim=0, largest=False, sorted=False
            ).values
            frag_diff_u = torch.topk(frag_diff_u, k, dim=0, sorted=False).values
            # concatenate with the bottom/top-k differences from previous
            # fragments
            frag_diff_l = torch.cat([self.diffs_l[i], frag_diff_l], dim=0)
            frag_diff_u = torch.cat([self.diffs_u[i], frag_diff_u], dim=0)
            # take the new top/bottom k
            k = min(frag_diff_l.size(0), self.k_poison)
            self.diffs_l[i] = torch.topk(
                frag_diff_l, k, dim=0, largest=False, sorted=False
            ).values
            self.diffs_u[i] = torch.topk(
                frag_diff_u, k, dim=0, sorted=False
            ).values

    def concretize_gradient_update(
        self, batchsize: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Compute the gradient update for the current batch.

        This function is called at the end of the batch, but doesn't have much
        significance for this multi-step domain. Here it only serves to indicate
        the end of the batch and to validate the batchsize. We return a dummy
        value that is not used in any computations involving the multi-step
        domain.
        """
        assert self.n_samples == batchsize, (
            "Batchsize does not match the number of samples seen in the batch."
        )
        if self.batchsize is None:
            self.batchsize = batchsize
        assert self.batchsize == batchsize, (
            "Size of batch does not match. "
            f"Got {batchsize}, expected {self.batchsize}."
        )
        # reset the number of samples seen for the next batch
        self.n_samples = 0
        # this is a dummy return value not used in the multi-step domain
        e = [torch.zeros(1)]
        return e, e, e

    def step(
        self,
        mean_grads_l: list[torch.Tensor],
        mean_grads_n: list[torch.Tensor],
        mean_grads_u: list[torch.Tensor],
    ) -> None:
        """Apply the optimizer update to the parameter values and bounds.

        The update should be applied in-place with respect to the BoundedModel
        that is tracked by this class.

        The gradients passed in here are dummy values and aren't used, since
        the actual gradients are accumulated internally in this class.
        """
        # check dummy values
        assert mean_grads_l == mean_grads_n == mean_grads_u == [torch.zeros(1)]
        assert self.batchsize is not None, "Batchsize has not been set."

        # for each parameter
        for i in range(len(self.theta_n)):
            # compute new parameter values
            p_n = self.theta_n[i] - self.lr * self.update_n[i] / self.batchsize
            p_u = (
                self.theta_u[i]
                - self.lr
                * (self.update_l[i] + self.diffs_l[i].sum(dim=0))
                / self.batchsize
            )
            p_l = (
                self.theta_l[i]
                - self.lr
                * (self.update_u[i] + self.diffs_u[i].sum(dim=0))
                / self.batchsize
            )
            # update the values in-place
            self.bounded_model.param_n[i].copy_(p_n)
            self.bounded_model.param_l[i].copy_(p_l)
            self.bounded_model.param_u[i].copy_(p_u)

    def epoch_start(self) -> None:
        """Called at the start of each epoch to update any optimizer state.

        Here, we must concretize the updates from the previous epoch, as we must
        take the worst case k_private at each epoch independently.
        """
        self.epoch_count += 1

        if self.epoch_count == 0:
            return

        for i in range(len(self.theta_n)):
            self.theta_n[i] -= self.lr * self.update_n[i] / self.batchsize
            self.theta_l[i] -= (
                self.lr
                * (self.update_u[i] + self.diffs_u[i].sum(dim=0))
                / self.batchsize
            )
            self.theta_u[i] -= (
                self.lr
                * (self.update_l[i] + self.diffs_l[i].sum(dim=0))
                / self.batchsize
            )
            assert torch.allclose(
                self.theta_n[i], self.bounded_model.param_n[i]
            )
            assert torch.allclose(
                self.theta_l[i], self.bounded_model.param_l[i]
            )
            assert torch.allclose(
                self.theta_u[i], self.bounded_model.param_u[i]
            )

        # reset the accumulated updates for the past epoch
        self.update_l = [torch.zeros_like(param) for param in self.theta_n]
        self.update_n = [torch.zeros_like(param) for param in self.theta_n]
        self.update_u = [torch.zeros_like(param) for param in self.theta_n]
        self.diffs_l = [
            torch.empty(0, *p.size(), dtype=p.dtype, device=p.device)
            for p in self.theta_n
        ]
        self.diffs_u = [
            torch.empty(0, *p.size(), dtype=p.dtype, device=p.device)
            for p in self.theta_n
        ]
