import numpy as np

import ags.blobs.accounting.run_rdp_comparison as blob_rdp
from common.accounting.laplace_rdp import SmoothLaplaceRdpInterval
from ags.blobs.accounting.run_cross_radius_comparison import (
    complete_profile,
    cross_radius_profile,
    intersect_public_and_nested,
)
from ags.blobs.accounting.run_rdp_comparison import public_bias_recalibration


def test_public_and_suffix_intersection_is_nested():
    boxes = {
        1: (np.array([-2.0, -0.5]), np.array([2.0, 0.5])),
        4: (np.array([-0.8, -2.0]), np.array([0.9, 2.0])),
        8: (np.array([-1.5, -1.5]), np.array([1.5, 1.5])),
    }
    result = intersect_public_and_nested(boxes, absolute_radius=1.0)
    for smaller, larger in ((1, 4), (4, 8)):
        assert np.all(result[larger][0] <= result[smaller][0])
        assert np.all(result[smaller][1] <= result[larger][1])
    np.testing.assert_allclose(result[1][0], [-0.8, -0.5])
    np.testing.assert_allclose(result[1][1], [0.9, 0.5])


def test_sparse_radius_profile_uses_next_enclosing_diameter():
    boxes = {
        0: (np.array([0.0]), np.array([0.0])),
        1: (np.array([-0.1]), np.array([0.2])),
        4: (np.array([-0.5]), np.array([0.7])),
    }
    certificates, _ = cross_radius_profile(boxes, diameter=4)
    profile, provenance = complete_profile(boxes, certificates, np.array([10.0]), diameter=4)
    np.testing.assert_allclose(profile[0], [0.2])
    np.testing.assert_allclose(profile[1], [1.2])
    np.testing.assert_allclose(profile[2], [1.2])
    np.testing.assert_allclose(profile[3], [1.2])
    np.testing.assert_allclose(profile[4], [1.2])
    assert provenance[1] == "diameter_T_4"
    assert provenance[4] == "cross_radius"


def test_public_bias_recalibration_finds_separating_threshold():
    design = np.c_[np.array([-3.0, -2.0, 1.0, 2.0]), np.ones(4)]
    labels = np.array([-1.0, -1.0, 1.0, 1.0])
    adjusted = public_bias_recalibration(np.array([1.0, 100.0]), design, labels)
    predictions = np.sign(design @ adjusted)
    np.testing.assert_array_equal(predictions, labels)


def test_coordinate_certification_groups_only_identical_release_laws(monkeypatch):
    calls = []

    def fake_certificate(order, epsilon, beta):
        calls.append((order, epsilon, beta))
        return SmoothLaplaceRdpInterval(order, epsilon + beta, epsilon + beta + 1e-4, 17)

    monkeypatch.setattr(blob_rdp, "_certified_coordinate_law", fake_certificate)
    result = blob_rdp.certify_coordinate_accounting([0.2, 0.2, 0.4], [0.01, 0.01, 0.02], 3.0, 1e-5)
    assert calls == [(3.0, 0.2, 0.01), (3.0, 0.4, 0.02)]
    assert result["unique_coordinate_laws"] == 2
    assert result["coordinate_laws_composed"] == 3
    assert result["interval_boxes_evaluated"] == 34
    assert np.isclose(result["rdp_lower"], 2 * 0.21 + 0.42)
    assert np.isclose(result["rdp_upper"], 2 * 0.2101 + 0.4201)
