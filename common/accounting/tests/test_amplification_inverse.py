import math

import pytest

from common.accounting.accounting import amplify_pair, base_budget_for_outer


@pytest.mark.parametrize("eps", [0.1, 1.0, 5.0, 20.0, 45.0])
@pytest.mark.parametrize("q", [1.0, 0.5, 0.1, 0.01])
def test_base_budget_inverts_amplify_pair(eps, q):
    delta = 1e-6
    eps_b, delta_b = base_budget_for_outer(eps, delta, q)
    eps_out, delta_out = amplify_pair(eps_b, delta_b, q)
    assert math.isclose(eps_out, eps, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(delta_out, delta, rel_tol=1e-12)


def test_base_budget_matches_paper_formula():
    eps, delta, q = 2.0, 1e-5, 0.25
    eps_b, delta_b = base_budget_for_outer(eps, delta, q)
    assert math.isclose(eps_b, math.log(1 + (math.exp(eps) - 1) / q))
    assert math.isclose(delta_b, delta / q)


def test_base_budget_rejects_delta_above_q():
    with pytest.raises(ValueError):
        base_budget_for_outer(1.0, 0.5, 0.1)
