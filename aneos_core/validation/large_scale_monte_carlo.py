"""Legacy large-scale harness sharing canonical candidate experiment evaluation.

No comparison between two names of one detector and no sigma/FPR certification.
Large sample counts do not make an unqualified toy population independently valid.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Optional

from .policy_evaluation import OrbitGenerator, evaluate_population, _integer


@dataclass
class MonteCarloValidationResult:
    total_samples: int
    false_positives: Optional[int]
    empirical_false_positive_rate: Optional[float]
    theoretical_false_positive_rate: Optional[float]
    sigma_level_equivalent: Optional[float]
    confidence_interval_95: Optional[tuple]
    processing_time_seconds: float
    detector_type: str
    parameter_source: str
    validation_timestamp: datetime
    statistical_significance_test: dict
    candidate_count: int = 0
    assessed_count: int = 0
    abstained_count: int = 0
    failed_count: int = 0
    candidate_fraction: Optional[float] = None
    candidate_interval_95: Optional[list] = None
    evaluation: Optional[dict] = None
    scientific_validation_status: str = 'not_independently_validated'
    five_sigma_claim_permitted: bool = False


@dataclass
class DetectorPerformanceComparison:
    """Historical transport type; automatic two-name comparison is withdrawn."""
    original_detector_results: Optional[MonteCarloValidationResult]
    corrected_detector_results: Optional[MonteCarloValidationResult]
    performance_improvement: dict
    recommendation: str


class LargeScaleMonteCarloValidator:
    def __init__(self,cache_dir='monte_carlo_cache'):
        self.cache_dir=Path(cache_dir)  # No implicit file creation or cleanup.
        # Historical reference constants retained, never used as measured validation.
        self.SIGMA_5_THEORETICAL_FP_RATE=5.7e-7
        self.SIGMA_5_THRESHOLD=5.
        self.chunk_size=1000;self.n_workers=1

    def generate_synthetic_natural_neos(self,count,seed=None):
        """Retained name: independent draws from an explicit unqualified mixture."""
        return OrbitGenerator(seed,profile='mixture-orbits-v2').generate(count)

    @staticmethod
    def _configuration(detector_type,use_corrected_params):
        if detector_type!='canonical' or use_corrected_params is not None:
            raise ValueError('Only canonical policy is installed; original/corrected configuration comparison is unavailable')

    def test_detector_chunk(self,args):
        """Return a versioned experiment, not a misleading (n, origin-FP, sigma) tuple."""
        rows,kind,corrected,chunk_id=args
        self._configuration(kind,corrected);_integer(chunk_id,'chunk ID',0,1_000_000)
        return evaluate_population(rows,workers=1,chunk_size=self.chunk_size)

    async def validate_detector_false_positive_rate(self,detector_type='canonical',use_corrected_params=None,
                                                    sample_size=1000,seed=42):
        """Offline candidate-yield experiment; legacy FPR/sigma fields stay null."""
        self._configuration(detector_type,use_corrected_params)
        _integer(self.n_workers,'process budget',1,8);_integer(self.chunk_size,'chunk budget',1,10000)
        population=self.generate_synthetic_natural_neos(sample_size,seed)
        start=time.perf_counter()
        report=await asyncio.to_thread(evaluate_population,population,workers=self.n_workers,chunk_size=self.chunk_size)
        counts=report['counts']
        return MonteCarloValidationResult(counts['total'],None,None,None,None,None,
            time.perf_counter()-start,'canonical','frozen_deployed_policy_orbital_only',datetime.now(timezone.utc),
            {'status':'unavailable_no_validated_origin_target_or_theoretical_error_rate'},
            candidate_count=counts['candidates'],assessed_count=counts['assessed'],abstained_count=counts['abstained'],
            failed_count=counts['invalid_input']+counts['screening_error'],
            candidate_fraction=report['candidate_fraction_assessed'],candidate_interval_95=report['candidate_interval_95'],
            evaluation=report)

    async def compare_detector_performance(self,sample_size=1000):
        raise NotImplementedError('Original/corrected names invoked the same model; provide genuinely distinct frozen policies and a matched independent evaluation protocol')

    def save_validation_results(self,results,filename):
        """Exclusive verified experiment export through the existing reporting owner."""
        from aneos_core.reporting.exporters import Exporter
        if not isinstance(results,MonteCarloValidationResult) or results.evaluation is None:
            raise ValueError('A completed versioned experiment is required; legacy recommendation artifacts are not regenerated')
        return Exporter(str(self.cache_dir)).export_policy_experiment(results.evaluation,filename)


async def test_monte_carlo_validation(sample_size=1000):
    """Explicit script entry: candidate counts and unavailable origin inference."""
    validator=LargeScaleMonteCarloValidator()
    result=await validator.validate_detector_false_positive_rate(sample_size=sample_size)
    print('Synthetic orbital screening experiment — not independent scientific validation')
    print(f'Requested {result.total_samples}; assessed {result.assessed_count}; candidates {result.candidate_count}; '
          f'abstained {result.abstained_count}; failed {result.failed_count}.')
    print('Origin FPR, discovery significance and calibrated probability: unavailable.')
    return result


if __name__=='__main__':asyncio.run(test_monte_carlo_validation())
