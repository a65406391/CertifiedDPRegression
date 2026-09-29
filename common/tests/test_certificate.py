"""Pin common.certificate to the implementations it replaces.

Each test transcribes one of the four independent implementations that existed before this module
and asserts bit-equality on random data. This is the technique that caught the AGS release drift
(see ags/blobs/test_ss_reference.py): abstract property tests did not catch a 71x error, tests
against the original logic did.
"""

import numpy as np
import pytest

from common.certificate import affine_output_interval, backfill, cross_radius, nest, smooth_sensitivity

RADII = [[1, 2, 3, 4], [1, 2, 4, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
BETAS = [0.01, 1 / 6, 0.5, 2.0]


def _nested_boxes(radii, n=25, seed=0):
    """Nominal plus a nested box per radius, widening with k."""
    rng = np.random.RandomState(seed)
    nominal = rng.randn(n)
    boxes, half = {}, np.zeros(n)
    for k in sorted(radii):
        half = half + np.abs(rng.rand(n)) * 0.4
        boxes[k] = (nominal - half, nominal + half)
    return nominal, boxes


# --- transcriptions of the four originals -------------------------------------------------------


def _linear_recompute(nominal, boxes, ceil, k_max):
    """agtr/linear/release.py::cross_radius_profile, ceiling box, no cap."""
    lo = {0: nominal}
    hi = {0: nominal}
    for k in range(1, k_max + 1):
        lo[k], hi[k] = boxes[k]
    A = np.zeros((k_max + 1, len(nominal)))
    for k in range(0, k_max + 1):
        nl, nu = (lo[k + 1], hi[k + 1]) if k < k_max else ceil
        A[k] = np.maximum(hi[k] - nl, nu - lo[k])
    return A


def _pate(nominal, boxes, y_min, y_max, gs, k_max):
    """agtr/*/pate/*::certificate, output-range tail and a GS cap, (n, k+1) oriented."""
    ivs = (
        [(nominal, nominal)]
        + [boxes[k] for k in range(1, k_max + 1)]
        + [(np.full_like(nominal, y_min), np.full_like(nominal, y_max))]
    )
    A = np.stack(
        [np.maximum(ivs[k][1] - ivs[k + 1][0], ivs[k + 1][1] - ivs[k][0]) for k in range(k_max + 1)],
        axis=1,
    )
    return np.minimum(A, gs)


def _staircase(A_k_first, beta, tail_width, k_max):
    """The staircase, identical in all three originals up to array orientation."""
    cand = (np.exp(-beta * np.arange(k_max + 1))[:, None] * A_k_first).max(axis=0)
    return np.maximum(cand, np.exp(-beta * (k_max + 1)) * tail_width)


# --- equivalence --------------------------------------------------------------------------------


@pytest.mark.parametrize("radii", RADII)
def test_matches_linear_recompute(radii):
    radii = list(range(1, max(radii) + 1))  # that implementation requires a dense grid
    nominal, boxes = _nested_boxes(radii)
    k_max = max(radii)
    ceil = (nominal - 6.0, nominal + 6.0)
    got = cross_radius(nominal, boxes, tail=ceil)
    want = _linear_recompute(nominal, boxes, ceil, k_max)
    assert np.allclose(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("radii", RADII)
def test_matches_pate(radii):
    radii = list(range(1, max(radii) + 1))
    nominal, boxes = _nested_boxes(radii, seed=1)
    k_max, y_min, y_max, gs = max(radii), -5.0, 5.0, 10.0
    tail = (np.full_like(nominal, y_min), np.full_like(nominal, y_max))
    got = cross_radius(nominal, boxes, tail=tail, cap=gs)
    want = _pate(nominal, boxes, y_min, y_max, gs, k_max)
    assert np.allclose(got, want.T, rtol=0, atol=0)


@pytest.mark.parametrize("radii", RADII)
@pytest.mark.parametrize("beta", BETAS)
def test_staircase_matches(radii, beta):
    radii = list(range(1, max(radii) + 1))
    nominal, boxes = _nested_boxes(radii, seed=2)
    ceil = (nominal - 6.0, nominal + 6.0)
    A = cross_radius(nominal, boxes, tail=ceil)
    tail_width = ceil[1] - ceil[0]
    assert np.allclose(
        smooth_sensitivity(A, beta, tail_width),
        _staircase(A, beta, tail_width, max(radii)),
        rtol=0,
        atol=0,
    )


# --- properties ---------------------------------------------------------------------------------


def test_backfill_uses_the_next_computed_radius():
    nominal, boxes = _nested_boxes([1, 2, 5])
    filled = backfill(boxes, 5)
    for gap in (3, 4):
        assert filled[gap] is boxes[5]
    assert filled[1] is boxes[1] and filled[2] is boxes[2]


def test_backfilled_certificate_dominates_the_dense_one():
    """Sound direction: a sparse grid may only over-state the certificate, never under-state it."""
    radii = list(range(1, 9))
    nominal, boxes = _nested_boxes(radii, seed=3)
    ceil = (nominal - 6.0, nominal + 6.0)
    dense = cross_radius(nominal, boxes, tail=ceil)
    sparse_boxes = {k: boxes[k] for k in (1, 2, 5, 8)}
    with pytest.warns(UserWarning, match="only .beta.gap.-smooth"):
        sparse = cross_radius(nominal, sparse_boxes, tail=ceil)
    assert np.all(sparse >= dense - 1e-12)


def test_nest_restores_nesting_without_tightening():
    nominal = np.zeros(4)
    boxes = {1: (np.full(4, -1.0), np.full(4, 1.0)), 2: (np.full(4, -0.5), np.full(4, 0.5))}
    out = nest(boxes)
    assert np.all(out[2][0] <= out[1][0]) and np.all(out[2][1] >= out[1][1])
    assert np.all(out[1][0] == boxes[1][0]) and np.all(out[1][1] == boxes[1][1])


def test_dense_grid_does_not_warn(recwarn):
    nominal, boxes = _nested_boxes([1, 2, 3])
    cross_radius(nominal, boxes, tail=(nominal - 1, nominal + 1))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_k0_must_not_be_supplied_as_a_box():
    nominal, boxes = _nested_boxes([1, 2])
    boxes[0] = (nominal, nominal)
    with pytest.raises(ValueError, match="k = 0"):
        cross_radius(nominal, boxes, tail=(nominal - 1, nominal + 1))


def test_affine_output_interval_is_exact_and_contains_the_nominal():
    x = np.linspace(-2, 2, 7)
    box = ((0.9, -0.1), (1.1, 0.1))
    lo, hi = affine_output_interval(box, x)
    for w, b in [(0.9, -0.1), (1.1, 0.1), (1.0, 0.0), (0.9, 0.1), (1.1, -0.1)]:
        y = w * x + b
        assert np.all(lo <= y + 1e-12) and np.all(y <= hi + 1e-12)
    assert np.allclose(lo, np.minimum(0.9 * x, 1.1 * x) - 0.1)
