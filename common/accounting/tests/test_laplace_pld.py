import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from common.accounting.laplace_pld import (
    continuous_ratio_laplace_pld_envelope,
    fixed_pair_laplace_pld,
)


def test_fixed_pair_matches_equal_scale_library_conservatively():
    interval = 0.002
    for shift in (0.1, 1.0, 2.0):
        actual = (
            fixed_pair_laplace_pld(shift, 1.0, interval=interval, tail_mass_truncation=1e-14)
            .self_compose(9)
            .get_epsilon_for_delta(1e-5)
        )
        expected = (
            privacy_loss_distribution.from_laplace_mechanism(
                parameter=1.0 / shift,
                sensitivity=1.0,
                pessimistic_estimate=True,
                value_discretization_interval=interval,
            )
            .self_compose(9)
            .get_epsilon_for_delta(1e-5)
        )
        assert actual >= expected
        assert actual - expected < 3e-5


def test_fixed_pair_cdf_is_monotone_probability():
    from common.accounting.laplace_pld import (
        laplace_pair_privacy_loss_cdf,
    )

    for ratio in (0.5, 1.0, 2.0):
        cdf = laplace_pair_privacy_loss_cdf(0.7, ratio)
        values = np.asarray([cdf(x) for x in np.linspace(-20, 20, 2001)])
        assert np.all(np.diff(values) >= -1e-14)
        assert 0 <= values.min() <= values.max() <= 1


def test_continuous_envelope_dominates_fixed_ratio_privacy_profiles():
    epsilon_coordinate, beta = 1.3, 0.4
    envelope = continuous_ratio_laplace_pld_envelope(
        epsilon_coordinate,
        beta,
        privacy_loss_interval=0.002,
        output_interval=0.005,
        ratio_intervals=200,
        tail_mass_truncation=1e-14,
    )
    for ratio in np.exp(np.linspace(-beta, beta, 9)):
        for fraction in (0.0, 0.5, 1.0):
            shift = 0.5 * epsilon_coordinate * min(1.0, ratio) * fraction
            fixed = fixed_pair_laplace_pld(
                shift,
                ratio,
                interval=0.002,
                tail_mass_truncation=1e-14,
            )
            for privacy_epsilon in np.linspace(0, 8, 33):
                assert envelope.get_delta_for_epsilon(privacy_epsilon) >= (
                    fixed.get_delta_for_epsilon(privacy_epsilon) - 1e-12
                )


def test_continuous_envelope_supports_constant_scale():
    envelope = continuous_ratio_laplace_pld_envelope(
        1.0,
        0.0,
        privacy_loss_interval=0.002,
        output_interval=0.005,
        ratio_intervals=20,
    )
    fixed = fixed_pair_laplace_pld(0.5, 1.0, interval=0.002)
    for privacy_epsilon in np.linspace(0, 4, 17):
        assert envelope.get_delta_for_epsilon(privacy_epsilon) >= (
            fixed.get_delta_for_epsilon(privacy_epsilon) - 1e-12
        )
