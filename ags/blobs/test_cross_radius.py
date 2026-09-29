"""The cross-radius certificate against the reference on synthetic nested envelopes.

Three facts the module rests on: W(k) <= Abar(k) <= W(k+1) for nested boxes; Abar(0) is the radius
of the 1-box about the nominal; and where every envelope already sits at the GS cap (the F = 1
rung of fig 1) the two certificates give the same smooth sensitivity, so the switch cannot move a
saturated curve.
"""

import numpy as np

from ags.blobs import cross_radius as CR
from ags.blobs import ss_reference as R

P = 5
RNG = np.random.RandomState(0)
THETA = RNG.normal(size=P)


def nested_boxes(k_max, step=0.1, skew=0.3):
    """Growing, nested, asymmetric boxes about THETA for k = 1 .. k_max."""
    boxes = {}
    for k in range(1, k_max + 1):
        lo = THETA - step * k * (1 - skew)
        hi = THETA + step * k * (1 + skew)
        boxes[k] = (lo, hi)
    return boxes


def test_sandwich_between_consecutive_widths():
    boxes = nested_boxes(6)
    gs = np.full(P, 1e9)  # cap inactive
    A = CR.certificate_grid(boxes, THETA, gs)
    for k in range(1, 6):
        w_k = boxes[k][1] - boxes[k][0]
        w_next = boxes[k + 1][1] - boxes[k + 1][0]
        assert np.all(A[k] >= w_k - 1e-12)
        assert np.all(A[k] <= w_next + 1e-12)


def test_k0_is_the_radius_of_the_one_box():
    boxes = nested_boxes(3)
    gs = np.full(P, 1e9)
    A = CR.certificate_grid(boxes, THETA, gs)
    lo1, hi1 = boxes[1]
    np.testing.assert_allclose(A[0], np.maximum(hi1 - THETA, THETA - lo1))


def test_agrees_with_reference_when_every_envelope_is_capped():
    gs = np.full(P, 0.5)
    # every box at least 2 GS wide and symmetric, so widths, radii and cross-radii all cap at GS
    boxes = {k: (THETA - 2.0 * gs * k, THETA + 2.0 * gs * k) for k in (1, 2, 3, 4, 8)}
    ks = sorted(boxes)
    ls = {k: R.local_sens(boxes[k][0], boxes[k][1], gs, k) for k in ks}
    for beta in (0.05, 0.5, 3.0):
        ss_ref = R.smooth_sensitivity(ls, ks, gs, beta, P)
        ss_cr = CR.smooth_sensitivity(boxes, THETA, gs, beta, P)
        np.testing.assert_allclose(ss_cr, ss_ref)


def test_report_names_the_gaps():
    boxes = nested_boxes(4)
    boxes[8] = (THETA - 1.0, THETA + 1.0)
    text = CR.report(boxes, THETA, np.full(P, 1e9))
    assert "back-filled [5, 6, 7]" in text
