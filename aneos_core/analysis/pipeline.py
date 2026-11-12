"""Typed acquisition and shared uncalibrated candidate screening.

Legacy indicator utilities remain available for research, outside deployed
classification. No calibrated significance or physical-origin model is installed.
"""

from typing import Dict, List, Optional, Any, Set, Callable
import asyncio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
import time

from .indicators.base import AnomalyIndicator, IndicatorResult, IndicatorConfig
from .indicators.orbital import (
    EccentricityIndicator, InclinationIndicator, SemiMajorAxisIndicator,
    OrbitalResonanceIndicator, OrbitalStabilityIndicator
)
from .indicators.velocity import (
    VelocityShiftIndicator, AccelerationIndicator, VelocityConsistencyIndicator,
    InfinityVelocityIndicator
)
from .indicators.temporal import (
    CloseApproachRegularityIndicator, ObservationGapIndicator,
    PeriodicityIndicator, TemporalInertiaIndicator
)
from .indicators.geographic import (
    SubpointClusteringIndicator, GeographicBiasIndicator
)
from .indicators.physical import (
    DiameterAnomalyIndicator, AlbedoAnomalyIndicator, SpectralAnomalyIndicator
)
from .scoring import ScoreCalculator, StatisticalAnalyzer, AnomalyScore
from .screening import ScreeningAnalysisService
from ..utils.errors import DataSourceUnavailableError
from ..data.models import NEOData, AnalysisResult
from ..data.sources.base import DataSourceManager
from ..data.sources.sbdb import SBDBSource
from ..data.sources.neodys import NEODySSource
from ..data.sources.mpc import MPCSource
from ..config.settings import ANEOSConfig, get_config
from ..data.cache import get_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    max_workers: int = 10
    timeout_seconds: int = 300
    retry_failed: bool = True
    enable_caching: bool = True
    indicator_configs: Dict[str, IndicatorConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default indicator configurations."""
        if not self.indicator_configs:
            # Default configurations for all indicators
            default_config = IndicatorConfig(weight=1.0, enabled=True, confidence_threshold=0.5)
            
            indicator_names = [
                'eccentricity', 'inclination', 'semi_major_axis', 'orbital_resonance', 'orbital_stability',
                'velocity_shifts', 'acceleration_anomalies', 'velocity_consistency', 'infinity_velocity',
                'approach_regularity', 'observation_gaps', 'periodicity', 'temporal_inertia',
                'subpoint_clustering', 'geographic_bias',
                'diameter_anomalies', 'albedo_anomalies', 'spectral_anomalies',
            ]
            
            for name in indicator_names:
                self.indicator_configs[name] = default_config

PIPELINE_VERSION = 'canonical-analysis-v1'
SCORE_SEMANTICS = 'uncalibrated_screening_rank'


@dataclass
class CanonicalScoreView:
    """Compatibility view: overall_score is ranking units, never a probability."""
    designation: str
    overall_score: Optional[float] = None
    confidence: Optional[float] = None
    classification: str = 'INCONCLUSIVE'
    indicator_scores: Dict[str, Any] = field(default_factory=dict)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    score_semantics: str = SCORE_SEMANTICS

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineResult:
    """Result from the analysis pipeline."""
    designation: str
    anomaly_score: CanonicalScoreView
    analysis_metadata: Dict[str, Any]
    processing_time: float
    data_quality: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    neo_data: Optional[NEOData] = None
    screening_assessment: Optional[Dict[str, Any]] = None
    status: str = 'inconclusive'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'designation': self.designation,
            'anomaly_score': self.anomaly_score.to_dict(),
            'analysis_metadata': self.analysis_metadata,
            'processing_time': self.processing_time,
            'data_quality': self.data_quality,
            'errors': self.errors,
            'warnings': self.warnings,
            'pipeline_version': PIPELINE_VERSION,
            'status': self.status,
            'screening_assessment': self.screening_assessment
        }

class AnalysisPipeline:
    """Acquire typed evidence and delegate decisions to the shared screening service.

    Legacy indicator calculators remain callable research utilities; they do not
    define this deployed decision policy or calibrated probabilities.
    """
    score_contract = SCORE_SEMANTICS
    
    def __init__(self, config: Optional[ANEOSConfig] = None, 
                 pipeline_config: Optional[PipelineConfig] = None):
        """Initialize the analysis pipeline."""
        self.config = config or get_config()
        self.pipeline_config = pipeline_config or PipelineConfig()

        # Align pipeline execution characteristics with configuration
        self.pipeline_config.max_workers = getattr(self.config, 'max_workers', self.pipeline_config.max_workers)
        self.pipeline_config.timeout_seconds = getattr(
            self.config, 'analysis_timeout', self.pipeline_config.timeout_seconds
        )
        self.analysis_parallel = getattr(self.config, 'analysis_parallel', True)
        self.analysis_queue_size = max(1, getattr(self.config, 'analysis_queue_size', 1))
        
        # Initialize components
        self.cache_manager = get_cache_manager()
        self.data_source_manager = None  # legacy manager is not an acquisition path
        self.data_fetcher = None  # lazy real DataFetcher, or explicit test injection
        self.screening_service = ScreeningAnalysisService()
        self.score_calculator = ScoreCalculator(self.config.weights, self.config.thresholds)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Initialize indicators
        self.indicators: Dict[str, AnomalyIndicator] = {}
        self._initialize_indicators()
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'indicator_performance': {}
        }
        
        logger.info(f"AnalysisPipeline initialized with {len(self.indicators)} indicators")
    
    def _create_data_source_manager(self):
        """Controlled deprecation: use typed DataFetcher, never mock sources."""
        raise NotImplementedError('Legacy source manager withdrawn; use DataFetcher')

    def _initialize_indicators(self) -> None:
        """Initialize all anomaly indicators."""
        indicator_classes = {
            # Orbital mechanics indicators
            'eccentricity': EccentricityIndicator,
            'inclination': InclinationIndicator,
            'semi_major_axis': SemiMajorAxisIndicator,
            'orbital_resonance': OrbitalResonanceIndicator,
            'orbital_stability': OrbitalStabilityIndicator,
            
            # Velocity indicators
            'velocity_shifts': VelocityShiftIndicator,
            'acceleration_anomalies': AccelerationIndicator,
            'velocity_consistency': VelocityConsistencyIndicator,
            'infinity_velocity': InfinityVelocityIndicator,
            
            # Temporal indicators
            'approach_regularity': CloseApproachRegularityIndicator,
            'observation_gaps': ObservationGapIndicator,
            'periodicity': PeriodicityIndicator,
            'temporal_inertia': TemporalInertiaIndicator,
            
            # Geographic indicators
            'subpoint_clustering': SubpointClusteringIndicator,
            'geographic_bias': GeographicBiasIndicator,

            # Physical indicators
            'diameter_anomalies': DiameterAnomalyIndicator,
            'albedo_anomalies': AlbedoAnomalyIndicator,
            'spectral_anomalies': SpectralAnomalyIndicator,
        }
        
        for name, indicator_class in indicator_classes.items():
            try:
                config = self.pipeline_config.indicator_configs.get(name, IndicatorConfig())
                self.indicators[name] = indicator_class(config)
                logger.debug(f"Initialized indicator: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize indicator {name}: {e}")
    
    async def analyze_neo(self, designation: str, neo_data: Optional[NEOData] = None,
                          force_refresh: bool = False) -> PipelineResult:
        """Screen real/provided evidence; failures abstain, never classify natural."""
        start = time.monotonic()
        try:
            if neo_data is None:
                neo_data = await self._fetch_neo_data(designation, force_refresh=force_refresh)
            if neo_data is None:
                raise DataSourceUnavailableError('No source evidence available')
            if not isinstance(neo_data, NEOData) or neo_data.designation != designation:
                raise ValueError('Acquired evidence type or object identity mismatch')
            assessment = self.screening_service.analyze_neo(neo_data)
            payload = assessment.to_payload()
            # Descriptive proxy diagnostics, not independent measured indicators.
            contributions = {e['type']: {
                'indicator_name': e['type'], 'raw_score': e['effect_size'],
                'weighted_score': None, 'confidence': None,
                'metadata': dict(e), 'contributing_factors': []}
                for e in assessment.evidence_sources if e['analyzed']}
            score = CanonicalScoreView(designation, assessment.screening_score,
                classification=assessment.classification, indicator_scores=contributions,
                risk_factors=list(contributions), statistical_summary={
                    'significance_status': assessment.significance_status,
                    'five_sigma_claim_permitted': False})
            quality = self._assess_data_quality(neo_data)
            elapsed = time.monotonic() - start
            self._update_performance_metrics(elapsed, True)
            return PipelineResult(designation, score, {
                **payload['analysis_metadata'], 'pipeline_version': PIPELINE_VERSION,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_sources_used': list(neo_data.sources_used),
                'legacy_indicators_run': [],
            }, elapsed, quality,
                warnings=list(assessment.limitations), neo_data=neo_data,
                screening_assessment=payload,
                status='completed' if assessment.screening_score is not None else 'inconclusive')
        except DataSourceUnavailableError as exc:
            return self._failure_result(designation, 'source_unavailable', str(exc), start)
        except ValueError as exc:
            return self._failure_result(designation, 'invalid_evidence', str(exc), start)
        except Exception as exc:
            logger.exception('Analysis failed for %s', designation)
            return self._failure_result(designation, 'failed', str(exc), start)

    def _failure_result(self, designation, status, reason, start):
        elapsed = time.monotonic() - start
        self._update_performance_metrics(elapsed, False)
        return PipelineResult(designation, CanonicalScoreView(designation),
            {'pipeline_version': PIPELINE_VERSION, 'five_sigma_claim_permitted': False},
            elapsed, {'reliability_score': None, 'error': reason},
            errors=[reason], status=status)

    async def analyze_batch(self, designations: List[str],
                           progress_callback: Optional[Callable[[int, int], None]] = None,
                           force_refresh: bool = False) -> List[PipelineResult]:
        """Bounded async scheduling; preserve order/identity including failed items.

        Cancellation stops awaiting a blocking provider call, not its worker thread.
        Provider timeouts and global rate limiting remain acquisition responsibilities.
        """
        semaphore = asyncio.Semaphore(max(1, min(self.pipeline_config.max_workers,
            self.analysis_queue_size)) if self.analysis_parallel else 1)
        completed = 0

        async def run(designation):
            nonlocal completed
            async with semaphore:
                start = time.monotonic()
                try:
                    result = await asyncio.wait_for(self.analyze_neo(designation,
                        force_refresh=force_refresh), self.pipeline_config.timeout_seconds)
                except asyncio.TimeoutError:
                    result = self._failure_result(designation, 'failed', 'Analysis timeout', start)
                except Exception as exc:
                    result = self._failure_result(designation, 'failed', str(exc), start)
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, len(designations))
                    except Exception:
                        logger.exception('Progress callback failed; scientific result retained')
                return result

        return list(await asyncio.gather(*(run(d) for d in designations)))

    async def _fetch_neo_data(self, designation: str, force_refresh: bool = False) -> Optional[NEOData]:
        """Use the existing typed/versioned DataFetcher; no duplicate legacy cache."""
        if self.data_fetcher is None:
            from ..data.fetcher import DataFetcher
            self.data_fetcher = DataFetcher(cache_manager=self.cache_manager)
        return await asyncio.to_thread(self.data_fetcher.fetch_neo_data, designation,
            force_refresh=force_refresh or not self.pipeline_config.enable_caching)

    async def _run_indicators(self, neo_data: NEOData) -> Dict[str, IndicatorResult]:
        """Run all enabled anomaly indicators."""
        indicator_results = {}
        
        # Run indicators in parallel where possible
        with ThreadPoolExecutor(max_workers=min(len(self.indicators), 5)) as executor:
            future_to_indicator = {
                executor.submit(indicator.safe_evaluate, neo_data): name
                for name, indicator in self.indicators.items()
                if indicator.is_enabled()
            }
            
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                
                try:
                    result = future.result()
                    indicator_results[indicator_name] = result
                    
                    logger.debug(f"Indicator {indicator_name} completed: "
                               f"score={result.raw_score:.3f}, confidence={result.confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"Indicator {indicator_name} failed: {e}")
                    
                    # Create error result
                    error_result = IndicatorResult(
                        indicator_name=indicator_name,
                        raw_score=0.0,
                        weighted_score=0.0,
                        confidence=0.0,
                        metadata={'error': str(e)}
                    )
                    indicator_results[indicator_name] = error_result
        
        return indicator_results
    
    def _assess_data_quality(self, neo_data: NEOData) -> Dict[str, Any]:
        """Count field presence; completeness is not measurement reliability."""
        orbital_fields = ['eccentricity', 'inclination', 'semi_major_axis', 'ascending_node',
            'argument_of_perihelion', 'mean_anomaly', 'epoch', 'orbital_period']
        physical_fields = ['diameter_km', 'albedo', 'absolute_magnitude_h', 'rotation_period_hours', 'spectral_type']
        missing = [f'orbital.{k}' for k in orbital_fields
            if getattr(neo_data.orbital_elements, k, None) is None]
        physical_missing = [f'physical.{k}' for k in physical_fields
            if getattr(neo_data.physical_properties, k, None) is None]
        count = len(orbital_fields) + len(physical_fields)
        return {
            'completeness_score': (count - len(missing) - len(physical_missing)) / count,
            'completeness_semantics': 'presence_of_13_catalogue_fields_not_reliability',
            'orbital_completeness': (len(orbital_fields) - len(missing)) / len(orbital_fields),
            'physical_fields_present': len(physical_fields) - len(physical_missing),
            'missing_elements': missing + physical_missing,
            'close_approaches_count': len(neo_data.close_approaches or []),
            'data_sources': list(neo_data.sources_used),
            'retrieved_at': neo_data.fetched_at.isoformat() if neo_data.fetched_at else None,
            'data_age_days': None, 'reliability_score': None,
        }

    def _update_performance_metrics(self, processing_time: float, success: bool) -> None:
        """Update pipeline performance metrics."""
        self.performance_metrics['total_analyses'] += 1
        
        if success:
            self.performance_metrics['successful_analyses'] += 1
        else:
            self.performance_metrics['failed_analyses'] += 1
        
        # Update average processing time
        total = self.performance_metrics['total_analyses']
        current_avg = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def _calculate_config_hash(self) -> str:
        """Compatibility accessor for the actual ranking policy ID, not legacy weights."""
        return self.screening_service.analyze_raw('policy-only', {}).policy_id

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        indicator_metrics = {}
        for name, indicator in self.indicators.items():
            indicator_metrics[name] = indicator.get_performance_metrics()
        
        return {
            'pipeline_metrics': self.performance_metrics,
            'indicator_metrics': indicator_metrics,
            'data_source_metrics': self.data_fetcher.get_fetch_statistics() if self.data_fetcher is not None else {},
            'cache_metrics': self.cache_manager.get_stats()
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        return {
            'performance_metrics': self.get_performance_metrics(),
            'statistical_summary': {'status': 'unavailable_uncalibrated_model',
                'five_sigma_claim_permitted': False},
            'configuration': {
                'pipeline_config': self.pipeline_config.__dict__,
                'weights': self.config.weights.__dict__,
                'thresholds': self.config.thresholds.__dict__
            },
            'indicators': {
                name: {
                    'enabled': indicator.is_enabled(),
                    'weight': indicator.get_weight(),
                    'description': indicator.get_description()
                }
                for name, indicator in self.indicators.items()
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'indicator_performance': {}
        }
        
        for indicator in self.indicators.values():
            indicator.reset_metrics()
        
        logger.info("Pipeline metrics reset")

# Factory function for easy pipeline creation
def create_analysis_pipeline(config_path: Optional[str] = None, 
                           custom_indicators: Optional[Dict[str, AnomalyIndicator]] = None) -> AnalysisPipeline:
    """Create and configure an analysis pipeline."""
    from ..config.settings import ConfigManager
    
    # Load configuration
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.config
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = AnalysisPipeline(config)
    
    # Add custom indicators if provided
    if custom_indicators:
        for name, indicator in custom_indicators.items():
            pipeline.indicators[name] = indicator
            logger.info(f"Added custom indicator: {name}")
    
    return pipeline