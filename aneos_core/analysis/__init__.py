"""
aNEOS Analysis Module - Advanced anomaly detection for Near Earth Objects.

This module provides a comprehensive suite of anomaly detection indicators,
scoring systems, and analysis pipelines for identifying potentially artificial
Near Earth Objects through statistical and scientific analysis.
"""

from typing import List

# Core analysis components
from .pipeline import AnalysisPipeline, PipelineConfig, PipelineResult, create_analysis_pipeline
from .scoring import ScoreCalculator, StatisticalAnalyzer, AnomalyScore

# Base indicator framework
from .indicators.base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    NumericRangeIndicator, StatisticalIndicator, TemporalIndicator, GeographicIndicator
)

# Orbital mechanics indicators
from .indicators.orbital import (
    EccentricityIndicator, InclinationIndicator, SemiMajorAxisIndicator,
    OrbitalResonanceIndicator, OrbitalStabilityIndicator
)

# Velocity analysis indicators
from .indicators.velocity import (
    VelocityShiftIndicator, AccelerationIndicator, VelocityConsistencyIndicator,
    InfinityVelocityIndicator
)

# Temporal pattern indicators
from .indicators.temporal import (
    CloseApproachRegularityIndicator, ObservationGapIndicator,
    PeriodicityIndicator, TemporalInertiaIndicator
)

# Geographic clustering indicators
from .indicators.geographic import (
    SubpointClusteringIndicator, GeographicBiasIndicator
)

__version__ = "2.0.0"
__author__ = "aNEOS Project"

# All available indicators for easy access
ALL_INDICATORS = {
    # Orbital mechanics
    'eccentricity': EccentricityIndicator,
    'inclination': InclinationIndicator,
    'semi_major_axis': SemiMajorAxisIndicator,
    'orbital_resonance': OrbitalResonanceIndicator,
    'orbital_stability': OrbitalStabilityIndicator,
    
    # Velocity analysis
    'velocity_shifts': VelocityShiftIndicator,
    'acceleration_anomalies': AccelerationIndicator,
    'velocity_consistency': VelocityConsistencyIndicator,
    'infinity_velocity': InfinityVelocityIndicator,
    
    # Temporal patterns
    'approach_regularity': CloseApproachRegularityIndicator,
    'observation_gaps': ObservationGapIndicator,
    'periodicity': PeriodicityIndicator,
    'temporal_inertia': TemporalInertiaIndicator,
    
    # Geographic clustering
    'subpoint_clustering': SubpointClusteringIndicator,
    'geographic_bias': GeographicBiasIndicator
}

def get_indicator_by_name(name: str, config: IndicatorConfig) -> AnomalyIndicator:
    """Get an indicator instance by name."""
    if name not in ALL_INDICATORS:
        raise ValueError(f"Unknown indicator: {name}. Available: {list(ALL_INDICATORS.keys())}")
    
    return ALL_INDICATORS[name](config)

def list_available_indicators() -> List[str]:
    """List all available indicator names."""
    return list(ALL_INDICATORS.keys())

def create_default_pipeline() -> AnalysisPipeline:
    """Create a pipeline with default configuration."""
    return create_analysis_pipeline()

# Quick analysis function for single NEO
async def analyze_neo_quick(designation: str) -> PipelineResult:
    """Quick analysis of a single NEO with default settings."""
    pipeline = create_default_pipeline()
    return await pipeline.analyze_neo(designation)

__all__ = [
    # Main pipeline
    'AnalysisPipeline', 'PipelineConfig', 'PipelineResult', 'create_analysis_pipeline',
    
    # Scoring and analysis
    'ScoreCalculator', 'StatisticalAnalyzer', 'AnomalyScore',
    
    # Base framework
    'AnomalyIndicator', 'IndicatorResult', 'IndicatorConfig',
    'NumericRangeIndicator', 'StatisticalIndicator', 'TemporalIndicator', 'GeographicIndicator',
    
    # Orbital indicators
    'EccentricityIndicator', 'InclinationIndicator', 'SemiMajorAxisIndicator',
    'OrbitalResonanceIndicator', 'OrbitalStabilityIndicator',
    
    # Velocity indicators
    'VelocityShiftIndicator', 'AccelerationIndicator', 'VelocityConsistencyIndicator',
    'InfinityVelocityIndicator',
    
    # Temporal indicators
    'CloseApproachRegularityIndicator', 'ObservationGapIndicator',
    'PeriodicityIndicator', 'TemporalInertiaIndicator',
    
    # Geographic indicators
    'SubpointClusteringIndicator', 'GeographicBiasIndicator',
    
    # Utility functions
    'ALL_INDICATORS', 'get_indicator_by_name', 'list_available_indicators',
    'create_default_pipeline', 'analyze_neo_quick'
]