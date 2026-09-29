import pytest

from ags.blobs.run_repeated_release_milp import (
    clipping_tail,
    record_touching_releases,
)


def test_fixed_cyclic_parallel_composition_counts_epochs():
    assert record_touching_releases(20) == 20


def test_one_step_clipping_tail_retains_batch_average():
    assert clipping_tail(1, lr=0.5, gamma=0.05, batch_size=20) == pytest.approx(0.0025)


def test_multistep_clipping_only_tail_is_full_output_diameter():
    # Without a non-expansiveness proof, shared examples in later steps may
    # react to the first changed update, so no 1/b factor is available.
    assert clipping_tail(10, lr=0.5, gamma=0.05, batch_size=20) == pytest.approx(0.5)
