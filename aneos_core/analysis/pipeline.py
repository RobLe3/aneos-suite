"""
Analysis pipeline orchestrator for aNEOS anomaly detection.

This module provides the main pipeline that coordinates all anomaly indicators,
data sources, and scoring systems to produce comprehensive NEO analysis results.
"""

from typing import Dict, List, Optional, Any, Set, Callable
import asyncio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from .scoring import ScoreCalculator, StatisticalAnalyzer, AnomalyScore
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
                'subpoint_clustering', 'geographic_bias'
            ]
            
            for name in indicator_names:
                self.indicator_configs[name] = default_config

@dataclass
class PipelineResult:
    """Result from the analysis pipeline."""
    designation: str
    anomaly_score: AnomalyScore
    analysis_metadata: Dict[str, Any]
    processing_time: float
    data_quality: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
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
            'pipeline_version': '2.0.0'
        }

class AnalysisPipeline:
    """Main analysis pipeline orchestrator."""
    
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
        self.data_source_manager = self._create_data_source_manager()
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
    
    def _create_data_source_manager(self) -> DataSourceManager:
        """Create data source manager with default sources."""
        try:
            sources = []
            
            # Try to create SBDB source (most reliable)
            try:
                sources.append(SBDBSource(self.config.api))
            except Exception as e:
                logger.warning(f"Failed to initialize SBDB source: {e}")
            
            # Try to create NEODyS source
            try:
                sources.append(NEODySSource(self.config.api))
            except Exception as e:
                logger.warning(f"Failed to initialize NEODyS source: {e}")
            
            # Try to create MPC source
            try:
                sources.append(MPCSource(self.config.api))
            except Exception as e:
                logger.warning(f"Failed to initialize MPC source: {e}")
            
            if not sources:
                logger.warning("No data sources available, creating dummy source")
                # Create a minimal mock source for testing
                from ..data.sources.base import DataSourceBase
                
                class MockSource(DataSourceBase):
                    def __init__(self):
                        # Mock minimal initialization
                        self.name = "mock"
                        
                    async def fetch_orbital_elements(self, designation: str) -> Dict[str, Any]:
                        return {"designation": designation, "mock": True}
                
                sources.append(MockSource())
            
            return DataSourceManager(sources, self.cache_manager)
            
        except Exception as e:
            logger.error(f"Failed to create data source manager: {e}")
            # Create an empty manager as fallback
            from ..data.sources.base import DataSourceBase
            
            class EmptySource(DataSourceBase):
                def __init__(self):
                    self.name = "empty"
                    
                async def fetch_orbital_elements(self, designation: str) -> Dict[str, Any]:
                    return {"designation": designation, "error": "No data sources available"}
            
            return DataSourceManager([EmptySource()], self.cache_manager)
    
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
            'geographic_bias': GeographicBiasIndicator
        }
        
        for name, indicator_class in indicator_classes.items():
            try:
                config = self.pipeline_config.indicator_configs.get(name, IndicatorConfig())
                self.indicators[name] = indicator_class(config)
                logger.debug(f"Initialized indicator: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize indicator {name}: {e}")
    
    async def analyze_neo(self, designation: str, neo_data: Optional[NEOData] = None) -> PipelineResult:
        """Analyze a single NEO."""
        start_time = time.time()
        
        try:
            # Fetch data if not provided
            if neo_data is None:
                neo_data = await self._fetch_neo_data(designation)
            
            if neo_data is None:
                return PipelineResult(
                    designation=designation,
                    anomaly_score=AnomalyScore(designation, 0.0, 0.0, 'natural'),
                    analysis_metadata={},
                    processing_time=time.time() - start_time,
                    data_quality={'error': 'No data available'},
                    errors=['Failed to fetch NEO data']
                )
            
            # Validate data quality
            data_quality = self._assess_data_quality(neo_data)
            
            # Run anomaly indicators
            indicator_results = await self._run_indicators(neo_data)
            
            # Calculate anomaly score
            anomaly_score = self.score_calculator.calculate_score(designation, indicator_results)
            
            # Update statistical analyzer
            self.statistical_analyzer.add_score(anomaly_score)
            
            # Generate analysis metadata
            analysis_metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_sources_used': getattr(neo_data, 'data_sources', []),
                'indicators_run': list(indicator_results.keys()),
                'pipeline_version': '2.0.0',
                'config_hash': self._calculate_config_hash()
            }
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            result = PipelineResult(
                designation=designation,
                anomaly_score=anomaly_score,
                analysis_metadata=analysis_metadata,
                processing_time=processing_time,
                data_quality=data_quality
            )
            
            # Cache result if enabled
            if self.pipeline_config.enable_caching:
                cache_key = f"analysis_result_{designation}_{self._calculate_config_hash()}"
                self.cache_manager.set(cache_key, result.to_dict(), ttl=3600)
            
            logger.info(f"Analysis completed for {designation}: {anomaly_score.classification} "
                       f"(score: {anomaly_score.overall_score:.3f}, time: {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {designation}: {e}")
            
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, False)
            
            return PipelineResult(
                designation=designation,
                anomaly_score=AnomalyScore(designation, 0.0, 0.0, 'natural'),
                analysis_metadata={'error_timestamp': datetime.now().isoformat()},
                processing_time=processing_time,
                data_quality={'error': 'Analysis failed'},
                errors=[str(e)]
            )
    
    async def analyze_batch(self, designations: List[str], 
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[PipelineResult]:
        """Analyze a batch of NEOs."""
        logger.info(f"Starting batch analysis of {len(designations)} NEOs")
        
        results = []
        completed = 0
        total = len(designations)

        if not self.analysis_parallel:
            logger.info("Parallel analysis disabled; processing sequentially")
            for designation in designations:
                result = await self.analyze_neo(designation)
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
            return results

        queue_size = self.analysis_queue_size if self.analysis_queue_size else total
        if queue_size <= 0:
            queue_size = total
        batches = [designations[i:i + queue_size] for i in range(0, total, queue_size)]

        for batch in batches:
            with ThreadPoolExecutor(max_workers=self.pipeline_config.max_workers) as executor:
                future_to_designation = {
                    executor.submit(asyncio.run, self.analyze_neo(designation)): designation
                    for designation in batch
                }

                for future in as_completed(future_to_designation, timeout=self.pipeline_config.timeout_seconds):
                    designation = future_to_designation[future]

                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total)

                        logger.debug(f"Completed analysis {completed}/{total}: {designation}")

                    except Exception as e:
                        logger.error(f"Failed to analyze {designation}: {e}")

                        error_result = PipelineResult(
                            designation=designation,
                            anomaly_score=AnomalyScore(designation, 0.0, 0.0, 'natural'),
                            analysis_metadata={},
                            processing_time=0.0,
                            data_quality={'error': 'Processing failed'},
                            errors=[str(e)]
                        )
                        results.append(error_result)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total)

        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    async def _fetch_neo_data(self, designation: str) -> Optional[NEOData]:
        """Fetch NEO data from available sources."""
        try:
            # Check cache first
            if self.pipeline_config.enable_caching:
                cache_key = f"neo_data_{designation}"
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    logger.debug(f"Using cached data for {designation}")
                    return NEOData(**cached_data)
            
            # Fetch from data sources
            neo_data = await self.data_source_manager.get_neo_data(designation)
            
            if neo_data and self.pipeline_config.enable_caching:
                # Cache the data
                cache_key = f"neo_data_{designation}"
                self.cache_manager.set(cache_key, neo_data.__dict__, ttl=7200)  # 2 hours
            
            return neo_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {designation}: {e}")
            return None
    
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
        """Assess the quality and completeness of NEO data."""
        quality_metrics = {
            'completeness_score': 0.0,
            'data_sources': getattr(neo_data, 'data_sources', []),
            'missing_elements': [],
            'data_age_days': None,
            'reliability_score': 1.0
        }
        
        # Check orbital elements completeness
        if neo_data.orbital_elements:
            orbital_completeness = 0
            orbital_total = 8  # Total expected orbital elements
            
            elements_to_check = [
                'eccentricity', 'inclination', 'semi_major_axis', 'ascending_node',
                'argument_of_perihelion', 'mean_anomaly', 'epoch', 'orbital_period'
            ]
            
            for element in elements_to_check:
                if hasattr(neo_data.orbital_elements, element) and getattr(neo_data.orbital_elements, element) is not None:
                    orbital_completeness += 1
                else:
                    quality_metrics['missing_elements'].append(f'orbital.{element}')
            
            quality_metrics['orbital_completeness'] = orbital_completeness / orbital_total
        else:
            quality_metrics['missing_elements'].append('orbital_elements')
            quality_metrics['orbital_completeness'] = 0.0
        
        # Check close approaches data
        if neo_data.close_approaches:
            approach_quality = 0
            for approach in neo_data.close_approaches:
                if approach.distance_au and approach.close_approach_date:
                    approach_quality += 1
            
            quality_metrics['close_approaches_quality'] = approach_quality / len(neo_data.close_approaches)
            quality_metrics['close_approaches_count'] = len(neo_data.close_approaches)
        else:
            quality_metrics['missing_elements'].append('close_approaches')
            quality_metrics['close_approaches_quality'] = 0.0
            quality_metrics['close_approaches_count'] = 0
        
        # Calculate overall completeness
        completeness_factors = [
            quality_metrics.get('orbital_completeness', 0),
            quality_metrics.get('close_approaches_quality', 0),
            1.0 if neo_data.designation else 0.0,
            1.0 if getattr(neo_data, 'physical_parameters', None) else 0.0
        ]
        
        quality_metrics['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        
        # Assess data age (if available)
        if neo_data.orbital_elements and hasattr(neo_data.orbital_elements, 'epoch') and neo_data.orbital_elements.epoch:
            try:
                if isinstance(neo_data.orbital_elements.epoch, datetime):
                    age = (datetime.now() - neo_data.orbital_elements.epoch).days
                    quality_metrics['data_age_days'] = age
                    
                    # Reduce reliability for very old data
                    if age > 365:  # More than 1 year old
                        quality_metrics['reliability_score'] *= 0.8
                    elif age > 30:  # More than 1 month old
                        quality_metrics['reliability_score'] *= 0.9
            except Exception:
                pass
        
        return quality_metrics
    
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
        """Calculate hash of current configuration for caching."""
        import hashlib
        
        config_str = f"{self.config.weights.__dict__}{self.config.thresholds.__dict__}{self.pipeline_config.__dict__}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        indicator_metrics = {}
        for name, indicator in self.indicators.items():
            indicator_metrics[name] = indicator.get_performance_metrics()
        
        return {
            'pipeline_metrics': self.performance_metrics,
            'indicator_metrics': indicator_metrics,
            'data_source_metrics': self.data_source_manager.get_performance_metrics(),
            'cache_metrics': self.cache_manager.get_stats()
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        return {
            'performance_metrics': self.get_performance_metrics(),
            'statistical_summary': self.statistical_analyzer.get_summary_statistics(),
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