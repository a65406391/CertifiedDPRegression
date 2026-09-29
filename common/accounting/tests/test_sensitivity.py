"""Mandatory finite-domain validation gate for cross-radius sensitivity."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from common.accounting.sensitivity import ReachableEnvelopes
from common.accounting.sensitivity import cross_radius_a_bar
from common.accounting.sensitivity import clipped_mean_smooth_sensitivity
from common.accounting.sensitivity import one_step_envelopes
from common.accounting.sensitivity import smooth_sensitivity
from common.accounting.toy_oracle import all_datasets
from common.accounting.toy_oracle import exact_a
from common.accounting.toy_oracle import exact_envelopes


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_cross_radius_dominates_literal_a_and_is_below_nested_diameter(m: int) -> None:
    domain = (-1, 0, 1)
    datasets = all_datasets(domain, m)

    def output(dataset: tuple[int, ...]) -> torch.Tensor:
        x = torch.tensor(dataset, dtype=torch.float64)
        # Two coordinates, including a nonlinear symmetric statistic.
        return torch.stack((x.mean(), (x.square().sum() + x.prod()) / m))

    # Exhaustive nominal datasets become costly at m=5; these representatives
    # cover boundary, homogeneous, and mixed batches while E and E' remain fully
    # enumerated for every selected nominal batch.
    nominals = (
        datasets
        if m <= 3
        else (
            tuple([-1] * m),
            tuple([0] * m),
            tuple([1] * m),
            tuple(itertools.islice(itertools.cycle(domain), m)),
        )
    )
    for nominal in nominals:
        lower, upper = exact_envelopes(nominal, datasets, output)
        boxes = ReachableEnvelopes(lower, upper)
        a_bar = cross_radius_a_bar(boxes)
        literal = exact_a(nominal, datasets, output)
        assert torch.all(literal <= a_bar + 1e-12)

        next_lower = torch.cat((lower[1:], lower[-1:]))
        next_upper = torch.cat((upper[1:], upper[-1:]))
        next_diameter = next_upper - next_lower
        assert torch.all(a_bar <= next_diameter + 1e-12)


def test_order_statistic_boxes_equal_finite_replacement_extrema() -> None:
    gradients = torch.tensor([[-0.7, 0.2], [0.1, -0.4], [0.8, 0.5]], dtype=torch.float64)
    caps = torch.tensor([1.0, 0.5], dtype=torch.float64)
    boxes = one_step_envelopes(gradients, caps, learning_rate=0.3)
    # k=0 is the nominal update and k=m spans the complete clipping box.
    nominal = -0.3 * gradients.mean(dim=0)
    assert torch.allclose(boxes.lower[0], nominal)
    assert torch.allclose(boxes.upper[0], nominal)
    assert torch.allclose(boxes.lower[-1], -0.3 * caps)
    assert torch.allclose(boxes.upper[-1], 0.3 * caps)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_cross_radius_smooth_scale_is_bidirectionally_beta_smooth(m: int) -> None:
    domain = (-1, 0, 1)
    datasets = all_datasets(domain, m)
    beta = torch.tensor([0.13, 0.27], dtype=torch.float64)

    def output(dataset: tuple[int, ...]) -> torch.Tensor:
        x = torch.tensor(dataset, dtype=torch.float64)
        return torch.stack((x.mean(), x.abs().mean() + 0.1 * x.prod()))

    scales = {}
    for nominal in datasets:
        lower, upper = exact_envelopes(nominal, datasets, output)
        scales[nominal] = smooth_sensitivity(cross_radius_a_bar(ReachableEnvelopes(lower, upper)), beta)

    for left in datasets:
        for index in range(m):
            for value in domain:
                if value == left[index]:
                    continue
                right = left[:index] + (value,) + left[index + 1 :]
                assert torch.all(scales[left] <= torch.exp(beta) * scales[right] + 1e-12)
                assert torch.all(scales[right] <= torch.exp(beta) * scales[left] + 1e-12)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_direct_clipped_mean_is_smooth_and_bounded_by_gs(m: int) -> None:
    domain = (-1.0, 0.0, 1.0)
    datasets = tuple(itertools.product(domain, repeat=m))
    beta = 0.2
    scales = {}
    for dataset in datasets:
        gradients = torch.tensor(dataset, dtype=torch.float64)[:, None]
        scale = clipped_mean_smooth_sensitivity(gradients, torch.ones(1, dtype=torch.float64), beta)[0]
        local = max(
            abs(sum(dataset) / m - sum(neighbor) / m)
            for neighbor in datasets
            if sum(x != y for x, y in zip(dataset, neighbor, strict=True)) == 1
        )
        assert local <= scale + 1e-12
        assert scale <= 2.0 / m + 1e-12
        expected = max(1.0 + max(abs(x) for x in dataset), 2.0 * math.exp(-beta)) / m
        assert float(scale) == pytest.approx(expected)
        scales[dataset] = scale
    for left in datasets:
        for right in datasets:
            distance = sum(x != y for x, y in zip(left, right, strict=True))
            if distance == 1:
                assert scales[left] <= math.exp(beta) * scales[right] + 1e-12
                assert scales[right] <= math.exp(beta) * scales[left] + 1e-12
