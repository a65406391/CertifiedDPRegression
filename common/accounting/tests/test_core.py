from collections import OrderedDict

import torch

from common.accounting.accounting import amplify_pair
from common.accounting.gradients import coordinate_clip
from common.accounting.gradients import l2_clip
from common.accounting.model import LeNet5
from common.accounting.model import flatten_parameters
from common.accounting.model import parameter_count
from common.accounting.model import unflatten_parameters
from common.accounting.config import ExperimentConfig
from common.accounting.model import LogisticRegression
from common.accounting.model import CompactSeluConvNet
from common.accounting.model import SampleConvNet
from common.accounting.training import public_coordinate_caps
from common.accounting.training import ags_release_count
from common.accounting.training import ags_release_scale
from common.accounting.training import _noise_from_uniform


def test_lenet_parameter_count_and_parameter_roundtrip() -> None:
    model = LeNet5()
    params = OrderedDict(model.named_parameters())
    vector = flatten_parameters(params)
    restored = unflatten_parameters(vector, params)
    assert parameter_count(model) == 44_426
    assert list(restored) == list(params)
    assert torch.equal(flatten_parameters(restored), vector)


def test_candidate_cnn_shapes_and_parameter_counts() -> None:
    inputs = torch.zeros(2, 1, 28, 28)
    full = SampleConvNet()
    compact = CompactSeluConvNet()
    assert full(inputs).shape == (2, 10)
    assert compact(inputs).shape == (2, 10)
    assert parameter_count(full) == 26_010
    assert parameter_count(compact) == 11_138


def test_clipping_contracts() -> None:
    grads = torch.tensor([[3.0, 4.0], [-2.0, 0.0]])
    coordinate, _ = coordinate_clip(grads, 1.5)
    euclidean, _ = l2_clip(grads, 1.0)
    assert torch.all(coordinate.abs() <= 1.5)
    assert torch.all(torch.linalg.vector_norm(euclidean, dim=1) <= 1.0 + 1e-6)


def test_coordinate_clipping_accepts_public_vector_caps() -> None:
    grads = torch.tensor([[3.0, -4.0], [-2.0, 0.25]])
    clipped, rate = coordinate_clip(grads, torch.tensor([1.0, 0.5]))
    assert torch.equal(clipped, torch.tensor([[1.0, -0.5], [-1.0, 0.25]]))
    assert rate == 0.75


def test_public_coordinate_caps_follow_parameter_roles() -> None:
    model = LogisticRegression()
    config = ExperimentConfig(
        model="logistic",
        coordinate_clip=0.1,
        coordinate_clip_weight=0.03,
        coordinate_clip_bias=0.2,
    )
    caps = public_coordinate_caps(model, config)
    sections = dict(model.named_parameters())
    weight_size = sections["linear.weight"].numel()
    assert torch.all(caps[:weight_size] == 0.03)
    assert torch.all(caps[weight_size:] == 0.2)


def test_l1_release_scales_sum_coordinate_sensitivities_by_block() -> None:
    model = LogisticRegression()
    sensitivity = torch.ones(sum(p.numel() for p in model.parameters()))
    sections = list(model.parameters())
    coordinate = ags_release_scale(model, sensitivity, 2.0, "coordinate", "laplace")
    tensor = ags_release_scale(model, sensitivity, 2.0, "tensor_l1", "laplace")
    joint = ags_release_scale(model, sensitivity, 2.0, "joint_l1", "laplace")
    assert torch.all(coordinate == 1.0)
    assert torch.all(tensor[: sections[0].numel()] == sections[0].numel())
    assert torch.all(tensor[sections[0].numel() :] == sections[1].numel())
    assert torch.all(joint == sensitivity.numel())
    assert ags_release_count(model, "coordinate") == sensitivity.numel()
    assert ags_release_count(model, "tensor_l1") == 2
    assert ags_release_count(model, "joint_l1") == 1


def test_config_accepts_only_laplace_for_l1_block_release() -> None:
    ExperimentConfig(model="logistic", ags_mechanism="laplace", ags_release_mode="joint_l1")
    try:
        ExperimentConfig(model="logistic", ags_mechanism="cauchy", ags_release_mode="joint_l1")
    except ValueError as error:
        assert "only for Laplace" in str(error)
    else:
        raise AssertionError("Cauchy block mode should be rejected")


def test_inverse_cdf_noise_is_finite_at_float32_rng_endpoints() -> None:
    uniform = torch.tensor([0.0, 1.0], dtype=torch.float32)
    scale = torch.ones(2, dtype=torch.float32)
    assert torch.isfinite(_noise_from_uniform(uniform, scale, "laplace")).all()
    assert torch.isfinite(_noise_from_uniform(uniform, scale, "cauchy")).all()


def test_wor_amplification_identity() -> None:
    assert amplify_pair(2.3, 1e-5, 1.0) == (2.3, 1e-5)
