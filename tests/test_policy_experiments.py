"""Synthetic arithmetic/integration controls, not independent NEO validation."""
import numpy as np
import pytest

POSITIVE={'a':1.23,'e':.2,'i':180.}
NEGATIVE={'a':1.23,'e':.2,'i':1.49}


def test_existing_batch_helper_measures_candidate_not_unavailable_origin():
    from aneos_core.validation.monte_carlo_false_positive_validation import run_detector_on_batch
    assert run_detector_on_batch([POSITIVE,NEGATIVE],{})==[True,False]


def test_bad_row_never_becomes_a_successful_natural_negative():
    from aneos_core.validation.monte_carlo_false_positive_validation import run_detector_on_batch
    assert run_detector_on_batch([{'a':1.,'e':-1.,'i':1.}],{})==[None]


def test_empty_mc_has_no_fabricated_zero_error_rate_or_zero_width_interval():
    from aneos_core.validation.monte_carlo_false_positive_validation import MonteCarloValidator
    result=MonteCarloValidator(n_cores=1).validate_false_positive_rate([])
    assert result.empirical_false_positive_rate is None
    assert result.false_positives is None and result.confidence_interval_95 is None


def test_generator_zero_seed_reproduces_records_and_preserves_global_rng():
    from aneos_core.validation.monte_carlo_false_positive_validation import SyntheticNEOGenerator
    before=np.random.get_state()
    first=SyntheticNEOGenerator(seed=0).generate_realistic_neo_population(5)
    second=SyntheticNEOGenerator(seed=0).generate_realistic_neo_population(5)
    assert len(first)==5 and first==second
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
