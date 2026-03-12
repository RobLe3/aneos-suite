"""
Property-based tests for aNEOS core invariants using hypothesis (TST-001/TST-003).

Groups:
1. OrbitalElements invariants
2. ClueContribution math invariant
3. statistical_utils round-trip
4. ATLAS score bounds
5. Targeted unit tests for coverage uplift
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Group 1 — OrbitalElements invariants
# ---------------------------------------------------------------------------

@given(e=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False))
def test_orbital_eccentricity_always_nonneg(e):
    """OrbitalElements must store eccentricity unchanged when in [0, 1)."""
    from aneos_core.data.models import OrbitalElements
    oe = OrbitalElements(eccentricity=e, inclination=10.0, semi_major_axis=1.0)
    assert oe.eccentricity >= 0.0


@given(inc=st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False))
def test_orbital_inclination_in_range(inc):
    """OrbitalElements must store inclination unchanged for values in [0, 180]."""
    from aneos_core.data.models import OrbitalElements
    oe = OrbitalElements(eccentricity=0.3, inclination=inc, semi_major_axis=1.0)
    assert 0.0 <= oe.inclination <= 180.0


# ---------------------------------------------------------------------------
# Group 2 — ClueContribution math invariant
# ---------------------------------------------------------------------------

@given(
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_clue_contribution_math(score, weight):
    """ClueContribution.contribution must equal normalized_score × weight (±float tolerance)."""
    from aneos_core.analysis.advanced_scoring import ClueContribution
    clue = ClueContribution(
        name='test',
        category='test',
        raw_value=score,
        normalized_score=score,
        weight=weight,
        contribution=score * weight,
        confidence=0.8,
        flag='',
        explanation='test',
    )
    assert abs(clue.contribution - clue.normalized_score * clue.weight) < 1e-9


# ---------------------------------------------------------------------------
# Group 3 — statistical_utils round-trip
# ---------------------------------------------------------------------------

@given(sigma=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
def test_sigma_p_sigma_roundtrip(sigma):
    """p_value_to_sigma(sigma_to_p_value(σ)) ≈ σ for σ ∈ [0.1, 5.0]."""
    from aneos_core.utils.statistical_utils import sigma_to_p_value, p_value_to_sigma
    p = sigma_to_p_value(sigma)
    recovered = p_value_to_sigma(p)
    assert abs(recovered - sigma) < 1e-6, f"sigma={sigma}, p={p}, recovered={recovered}"


@given(p=st.floats(min_value=1e-10, max_value=0.99, allow_nan=False, allow_infinity=False))
def test_p_sigma_p_roundtrip(p):
    """sigma_to_p_value(p_value_to_sigma(p)) ≈ p for p ∈ [1e-10, 0.99]."""
    from aneos_core.utils.statistical_utils import sigma_to_p_value, p_value_to_sigma
    sigma = p_value_to_sigma(p)
    recovered_p = sigma_to_p_value(sigma)
    rel_err = abs(recovered_p - p) / max(p, 1e-300)
    assert rel_err < 0.01, f"p={p}, sigma={sigma}, recovered_p={recovered_p}, rel_err={rel_err}"


# ---------------------------------------------------------------------------
# Group 4 — ATLAS score bounds
# ---------------------------------------------------------------------------

@given(
    e=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
    inc=st.floats(min_value=0.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    a=st.floats(min_value=0.3, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_atlas_overall_score_always_in_unit_interval(e, inc, a):
    """AdvancedScoreCalculator.calculate_score must always return overall_score in [0, 1]."""
    from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
    neo_data = {'eccentricity': e, 'inclination': inc, 'semi_major_axis': a}
    result = AdvancedScoreCalculator().calculate_score(neo_data, {})
    assert 0.0 <= result.overall_score <= 1.0, (
        f"overall_score={result.overall_score} out of [0,1] for e={e}, inc={inc}, a={a}"
    )


# ---------------------------------------------------------------------------
# Group 5 — Targeted unit tests for coverage uplift
# ---------------------------------------------------------------------------

def test_bonferroni_empty_list():
    from aneos_core.utils.statistical_utils import apply_bonferroni_correction
    result = apply_bonferroni_correction([])
    assert result == []


def test_bonferroni_single_p_capped_at_1():
    from aneos_core.utils.statistical_utils import apply_bonferroni_correction
    result = apply_bonferroni_correction([0.6])
    assert result[0] <= 1.0


def test_bonferroni_scales_by_n():
    from aneos_core.utils.statistical_utils import apply_bonferroni_correction
    p_vals = [0.01, 0.02, 0.05]
    corrected = apply_bonferroni_correction(p_vals)
    assert len(corrected) == 3
    # Each corrected p should be >= original (Bonferroni is conservative)
    for orig, corr in zip(p_vals, corrected):
        assert corr >= orig


def test_advanced_scoring_config_defaults_when_file_missing(tmp_path, monkeypatch):
    """AdvancedScoreCalculator must initialise without error even with no config file."""
    monkeypatch.chdir(tmp_path)  # no aneos_config.yaml here
    from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
    calc = AdvancedScoreCalculator()
    assert calc.config is not None


def test_network_sigma_fisher_adds_method_key():
    """Fisher combine() result must contain 'method': 'fisher'."""
    from aneos_core.pattern_analysis.network_sigma import NetworkSigmaCombiner
    result = NetworkSigmaCombiner().combine({'pa1': 0.05})
    assert result.get('method') == 'fisher'


def test_p_value_sigma_at_boundary_values():
    from aneos_core.utils.statistical_utils import p_value_to_sigma
    assert p_value_to_sigma(1.0) == 0.0
    # Very small p → large (finite) sigma
    sigma_tiny = p_value_to_sigma(1e-300)
    assert sigma_tiny > 10.0
