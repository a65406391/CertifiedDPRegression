import numpy as np

from ags.mnist.torch_sgd import train


def test_tail_window_one_equals_final_iterate():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(8, 3))
    y = np.arange(8) % 2
    final, averages = train(
        x,
        y,
        n_s=4,
        lr=0.1,
        gamma=0.2,
        lam=0,
        b=8,
        n_classes=2,
        device="cpu",
        tail_average_windows=[1, 4],
    )
    np.testing.assert_allclose(averages[1], final)
    assert averages[4].shape == final.shape
