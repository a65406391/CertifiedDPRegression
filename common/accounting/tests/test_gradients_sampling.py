import torch
from torch import nn

from common.accounting.gradients import per_example_gradients


def test_average_per_example_gradient_equals_batch_gradient() -> None:
    torch.manual_seed(3)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    inputs = torch.randn(5, 1, 2, 2)
    targets = torch.tensor([0, 1, 2, 1, 0])
    per_example = per_example_gradients(model, inputs, targets, chunk_size=2)
    model.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(inputs), targets, reduction="mean").backward()
    ordinary = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])
    assert torch.allclose(per_example.mean(0), ordinary, atol=1e-6, rtol=1e-5)


def test_wor_is_unique_and_poisson_can_be_empty_without_resampling() -> None:
    generator = torch.Generator().manual_seed(8)
    for _ in range(100):
        batch = torch.randperm(50, generator=generator)[:17]
        assert len(batch) == 17
        assert len(batch.unique()) == 17
    empty_seen = False
    selected = 0
    trials = 20_000
    for _ in range(trials):
        mask = torch.rand(4, generator=generator) < 0.1
        selected += int(mask.sum())
        empty_seen |= not bool(mask.any())
    assert empty_seen
    assert abs(selected / (trials * 4) - 0.1) < 0.01
