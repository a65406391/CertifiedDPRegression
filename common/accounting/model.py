"""The models the accounting experiments train.

LeNet-5 (44,426 parameters, asserted), multinomial logistic regression, the Opacus sample CNN and a
compact SELU CNN, plus `build_model` to select one by name from a config and `parameter_count`.

Parameter count is not incidental here: the coordinate-wise release splits the budget across every
coordinate, so P appears directly in the noise scale. That is why the PCA-MNIST head has 650
parameters and not more.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class LogisticRegression(nn.Module):
    """Multinomial logistic regression on raw 28x28 MNIST pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(1))


class SampleConvNet(nn.Module):
    """The 26,010-parameter MNIST CNN used by Opacus/TF Privacy."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = nn.functional.relu(self.conv1(inputs))
        values = nn.functional.max_pool2d(values, 2, 1)
        values = nn.functional.relu(self.conv2(values))
        values = nn.functional.max_pool2d(values, 2, 1)
        values = values.flatten(1)
        values = nn.functional.relu(self.fc1(values))
        return self.fc2(values)


class CompactSeluConvNet(nn.Module):
    """Half-width AGS candidate retaining the standard CNN receptive fields."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(8, 16, 4, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = nn.functional.selu(self.conv1(inputs))
        values = nn.functional.max_pool2d(values, 2, 1)
        values = nn.functional.selu(self.conv2(values))
        values = nn.functional.max_pool2d(values, 2, 1)
        values = values.flatten(1)
        values = nn.functional.selu(self.fc1(values))
        return self.fc2(values)


def build_model(name: str) -> nn.Module:
    if name == "lenet5":
        return LeNet5()
    if name == "logistic":
        return LogisticRegression()
    if name == "sample_cnn":
        return SampleConvNet()
    if name == "compact_selu_cnn":
        return CompactSeluConvNet()
    raise ValueError(f"unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def flatten_parameters(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in parameters.values()])


def unflatten_parameters(
    vector: torch.Tensor, template: dict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for name, value in template.items():
        size = value.numel()
        result[name] = vector[offset : offset + size].reshape(value.shape)
        offset += size
    if offset != vector.numel():
        raise ValueError("parameter vector has an incorrect length")
    return result


def layer_slices(model: nn.Module) -> dict[str, slice]:
    result = {}
    offset = 0
    for name, value in model.named_parameters():
        result[name] = slice(offset, offset + value.numel())
        offset += value.numel()
    return result
