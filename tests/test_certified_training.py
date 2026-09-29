"""
End to end tests for certified training that check if the nominal value of the parameters trained using AGT are correct.
These tests are not exhaustive and only check the very basic configurations of AGT.
"""

from itertools import product
import copy
import pytest
import torch
import torch.utils.data
import abstract_gradient_training as agt


@pytest.mark.parametrize(
    "hidden_layers, hidden_dim, in_dim, seed",
    product([1, 2], [1, 10], [1, 10], [0, 1]),
)
def test_certified_training(hidden_layers, hidden_dim, in_dim, seed):
    """Test that the nominal value of the model returned by AGT is correct."""
    # set the config of the training run
    torch.manual_seed(seed)
    batchsize = 100  # samples per batch
    n_batches = 2  # batches per epoch
    n_epochs = 2  # number of epochs
    lr = 0.1  # learning rate
    config = agt.AGTConfig(learning_rate=lr, n_epochs=n_epochs, loss="mse")
    # construct the dummy dataset
    X = torch.randn(batchsize * n_batches, in_dim, dtype=torch.float64)
    y = torch.randn(batchsize * n_batches, 1, dtype=torch.float64)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batchsize, shuffle=False)
    # construct the test model
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_dim, 1))
    test_model = torch.nn.Sequential(*layers).double()
    test_model_copy = copy.deepcopy(test_model)
    # train the model normally
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(test_model.parameters(), lr=lr)  # type: ignore
    for _ in range(n_epochs):
        for x, y in dataloader:
            y_pred = test_model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # test that the uncertified training returns the correct nominal parameter result
    bounded_model = agt.bounded_models.IntervalBoundedModel(test_model_copy)
    agt.uncertified_training(bounded_model, config, dataloader)
    for p1, p2 in zip(test_model.parameters(), bounded_model.param_n):
        assert torch.allclose(p1, p2), f"{(p1 - p2).abs().max()}"

    # test that poison certified training returns the correct nominal parameter result
    bounded_model = agt.bounded_models.IntervalBoundedModel(test_model_copy)
    agt.poison_certified_training(bounded_model, config, dataloader)
    for p1, p2 in zip(test_model.parameters(), bounded_model.param_n):
        assert torch.allclose(p1, p2)

    # test that privacy certified training returns the correct nominal parameter result
    bounded_model = agt.bounded_models.IntervalBoundedModel(test_model_copy)
    agt.privacy_certified_training(bounded_model, config, dataloader)
    for p1, p2 in zip(test_model.parameters(), bounded_model.param_n):
        assert torch.allclose(p1, p2)

    # test that unlearning certified training returns the correct nominal parameter result
    bounded_model = agt.bounded_models.IntervalBoundedModel(test_model_copy)
    agt.unlearning_certified_training(bounded_model, config, dataloader)
    for p1, p2 in zip(test_model.parameters(), bounded_model.param_n):
        assert torch.allclose(p1, p2)
