"""Numerical identities do not validate an astronomical null model."""
import math

import pytest
from scipy import stats

from aneos_core.utils.statistical_utils import (
    sigma_to_p_value, p_value_to_sigma, fisher_log_p_value,
    log_p_value_to_sigma, finite_probability,
)


@pytest.mark.parametrize("sigma", [0, 1, 5, 8, 12, 20, 30])
def test_gaussian_tail_roundtrip_without_cdf_cancellation(sigma):
    p = sigma_to_p_value(sigma)
    assert 0 < p <= 1
    assert p_value_to_sigma(p) == pytest.approx(sigma, abs=1e-12)


@pytest.mark.parametrize("p", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        p_value_to_sigma(p)


def test_fisher_matches_scipy_in_representable_range():
    ps = [.05, .01, .1]
    assert math.exp(fisher_log_p_value(ps)) == pytest.approx(stats.combine_pvalues(ps).pvalue)


def test_fisher_preserves_log_tail_when_float_underflows():
    logp = fisher_log_p_value([1e-300] * 4)
    assert logp < -1000
    assert math.isfinite(log_p_value_to_sigma(logp))
    assert finite_probability(logp) > 0


def test_endpoint_zero_is_retained_as_bound_not_missing_evidence():
    assert fisher_log_p_value([0, .05]) < fisher_log_p_value([.05])
    assert p_value_to_sigma(0) == float("inf")
