"""Legacy-named Monte Carlo facades for uncalibrated candidate experiments.

Synthetic candidate yield is not an artificial-origin false-positive rate. Legacy
FP/uncertainty aliases are nullable; all inputs and abstentions remain visible.
"""
from dataclasses import dataclass, asdict
from typing import Optional

from .policy_evaluation import OrbitGenerator, evaluate_population, binomial_interval, _integer


@dataclass
class MonteCarloResult:
    total_synthetic_objects: int
    false_positives: Optional[int]
    empirical_false_positive_rate: Optional[float]
    theoretical_false_positive_rate: Optional[float]
    confidence_interval_95: Optional[tuple]
    bootstrap_confidence_intervals: dict
    statistical_comparison: dict
    limitations: list
    candidate_count: int = 0
    assessed_count: int = 0
    abstained_count: int = 0
    failed_count: int = 0
    candidate_fraction: Optional[float] = None
    candidate_interval_95: Optional[list] = None
    evaluation: Optional[dict] = None
    validation_status: str = 'synthetic_or_supplied_screening_experiment_not_independent_validation'
    five_sigma_claim_permitted: bool = False


class SyntheticNEOGenerator:
    """Compatibility name for a declared, unqualified toy orbital generator."""
    def __init__(self, seed=None):
        self._generator=OrbitGenerator(seed)
        self.limitations=['Unvalidated clipped toy recipe, not an empirically established natural NEO population.']

    def generate_realistic_neo_population(self, size=1000):
        """Retained name; returned list carries explicit synthetic replay context."""
        return self._generator.generate(size)


def run_detector_on_batch(neo_batch, detector_params=None):
    """Return candidate flags; None means unassessed/invalid/failed, never natural.

    Not origin labels. Consumers must not bool-coerce None or divide by all rows.
    Unsupported custom detector parameters are refused, not silently ignored.
    """
    report=_evaluate(neo_batch,detector_params,workers=1)
    return [row['candidate'] for row in report['outcomes']]


def _evaluate(rows,params,*,workers=1,chunk_size=1000):
    if params is not None and params!={}:raise ValueError('Only the deployed canonical screening policy is supported')
    return evaluate_population(rows,workers=workers,chunk_size=chunk_size)


class MonteCarloValidator:
    """Measure orbital-only candidate selections, abstentions and errors."""
    def __init__(self, n_cores=None):
        self.n_cores=_integer(1 if n_cores is None else n_cores,'process budget',1,8)
        self.limitations=[]

    def validate_false_positive_rate(self, synthetic_neos, detector_params=None):
        """Retained method name; no artificial-origin FPR is estimated."""
        report=_evaluate(synthetic_neos,detector_params,workers=self.n_cores)
        counts=report['counts']
        self.limitations=report['limitations'].copy()
        return MonteCarloResult(counts['total'],None,None,None,None,{},
            {'status':'unavailable_no_validated_origin_target_or_theoretical_error_rate'},self.limitations.copy(),
            candidate_count=counts['candidates'],assessed_count=counts['assessed'],
            abstained_count=counts['abstained'],failed_count=counts['invalid_input']+counts['screening_error'],
            candidate_fraction=report['candidate_fraction_assessed'],candidate_interval_95=report['candidate_interval_95'],
            evaluation=report)

    def _calculate_confidence_intervals(self,false_positives,total,empirical_rate):
        """Legacy supplied-count arithmetic only, not automatic MC uncertainty.

        The supplied number is an event count, not certified origin false positives.
        Public experiments require verified IID generator context for an interval.
        """
        expected=false_positives/total if total else None
        if empirical_rate!=expected:raise ValueError('Supplied rate does not match counts or is unavailable')
        return {'95_percent':binomial_interval(false_positives,total,.95),
                '99_percent':binomial_interval(false_positives,total,.99)}

    def _bootstrap_confidence_intervals(self,detection_results,n_bootstrap=1000):
        raise NotImplementedError('Unqualified bootstrap error-rate estimates withdrawn; specify an independently justified target and resampling design')

    def _compare_with_theoretical_rate(self,false_positives,total,theoretical_rate):
        raise NotImplementedError('No validated origin-FPR target or five-sigma calibration is installed')


def demonstrate_monte_carlo_validation(sample_size=1000):
    """Offline synthetic mechanics demonstration, not a discovery certification."""
    population=SyntheticNEOGenerator(seed=42).generate_realistic_neo_population(sample_size)
    return asdict(MonteCarloValidator(n_cores=1).validate_false_positive_rate(population))
