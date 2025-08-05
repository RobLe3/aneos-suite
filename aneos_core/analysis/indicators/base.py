"""
Base classes for anomaly indicators in aNEOS.

This module provides the foundation for all anomaly detection indicators,
enabling a pluggable system for extensible scientific analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np

from ...data.models import NEOData, OrbitalElements, CloseApproach
from ...config.settings import ThresholdConfig, WeightConfig

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    """Result from an anomaly indicator evaluation."""
    indicator_name: str
    raw_score: float
    weighted_score: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'indicator_name': self.indicator_name,
            'raw_score': self.raw_score,
            'weighted_score': self.weighted_score,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'contributing_factors': self.contributing_factors
        }

@dataclass
class IndicatorConfig:
    """Configuration for an anomaly indicator."""
    weight: float = 1.0
    enabled: bool = True
    confidence_threshold: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)

class AnomalyIndicator(ABC):
    """Abstract base class for all anomaly indicators."""
    
    def __init__(self, name: str, config: IndicatorConfig):
        self.name = name
        self.config = config
        self._evaluation_count = 0
        self._total_score = 0.0
        self._performance_metrics = {
            'evaluations': 0,
            'average_score': 0.0,
            'max_score': 0.0,
            'min_score': float('inf'),
            'anomalies_detected': 0
        }
    
    @abstractmethod
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """
        Evaluate anomaly score for given NEO data.
        
        Args:
            neo_data: Complete NEO data including orbital elements and close approaches
            
        Returns:
            IndicatorResult with raw score, weighted score, and metadata
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of what this indicator measures."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if this indicator is enabled."""
        return self.config.enabled
    
    def get_weight(self) -> float:
        """Get the weight for this indicator."""
        return self.config.weight
    
    def calculate_weighted_score(self, raw_score: float, confidence: float = 1.0) -> float:
        """Calculate weighted score based on raw score and confidence."""
        return raw_score * self.config.weight * confidence
    
    def update_performance_metrics(self, result: IndicatorResult) -> None:
        """Update performance tracking metrics."""
        self._evaluation_count += 1
        self._total_score += result.raw_score
        
        self._performance_metrics['evaluations'] = self._evaluation_count
        self._performance_metrics['average_score'] = self._total_score / self._evaluation_count
        self._performance_metrics['max_score'] = max(self._performance_metrics['max_score'], result.raw_score)
        self._performance_metrics['min_score'] = min(self._performance_metrics['min_score'], result.raw_score)
        
        # Count as anomaly if weighted score is above threshold
        if result.weighted_score > 1.0:  # Arbitrary threshold for anomaly
            self._performance_metrics['anomalies_detected'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this indicator."""
        return {
            'name': self.name,
            'config': {
                'weight': self.config.weight,
                'enabled': self.config.enabled,
                'confidence_threshold': self.config.confidence_threshold
            },
            'metrics': self._performance_metrics.copy()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._evaluation_count = 0
        self._total_score = 0.0
        self._performance_metrics = {
            'evaluations': 0,
            'average_score': 0.0,
            'max_score': 0.0,
            'min_score': float('inf'),
            'anomalies_detected': 0
        }
    
    def validate_neo_data(self, neo_data: NEOData) -> bool:
        """Validate that NEO data contains required information for this indicator."""
        return neo_data is not None and neo_data.designation is not None
    
    def safe_evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Safely evaluate with error handling and performance tracking."""
        if not self.is_enabled():
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'disabled': True}
            )
        
        if not self.validate_neo_data(neo_data):
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'validation_failed': True}
            )
        
        try:
            result = self.evaluate(neo_data)
            
            # Ensure weighted score is calculated
            if result.weighted_score == 0.0 and result.raw_score != 0.0:
                result.weighted_score = self.calculate_weighted_score(result.raw_score, result.confidence)
            
            # Update performance metrics
            self.update_performance_metrics(result)
            
            logger.debug(f"Indicator {self.name} evaluated {neo_data.designation}: raw={result.raw_score:.3f}, weighted={result.weighted_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in indicator {self.name} for {neo_data.designation}: {e}")
            
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'error': str(e)}
            )

class NumericRangeIndicator(AnomalyIndicator):
    """Base class for indicators that check numeric values against ranges."""
    
    def __init__(self, name: str, config: IndicatorConfig, 
                 normal_range: Tuple[float, float], extreme_threshold: float):
        super().__init__(name, config)
        self.normal_range = normal_range
        self.extreme_threshold = extreme_threshold
    
    def calculate_range_anomaly(self, value: Optional[float], 
                              context: str = "") -> Tuple[float, List[str]]:
        """
        Calculate anomaly score based on how far a value is from normal range.
        
        Returns:
            Tuple of (anomaly_score, contributing_factors)
        """
        if value is None:
            return 0.0, [f"No {context} data available"]
        
        factors = []
        
        # Check if within normal range
        if self.normal_range[0] <= value <= self.normal_range[1]:
            return 0.0, [f"{context} within normal range ({value:.3f})"]
        
        # Calculate how far outside normal range
        if value < self.normal_range[0]:
            deviation = (self.normal_range[0] - value) / (self.normal_range[1] - self.normal_range[0])
            factors.append(f"{context} below normal range: {value:.3f} < {self.normal_range[0]}")
        else:
            deviation = (value - self.normal_range[1]) / (self.normal_range[1] - self.normal_range[0])
            factors.append(f"{context} above normal range: {value:.3f} > {self.normal_range[1]}")
        
        # Check for extreme values
        if abs(value) > self.extreme_threshold:
            deviation *= 2.0  # Double the score for extreme values
            factors.append(f"{context} extreme value: {value:.3f}")
        
        # Cap the score at 1.0 to prevent one indicator from dominating
        score = min(deviation, 1.0)
        
        return score, factors

class StatisticalIndicator(AnomalyIndicator):
    """Base class for indicators that use statistical analysis."""
    
    def __init__(self, name: str, config: IndicatorConfig):
        super().__init__(name, config)
        self._data_history: List[float] = []
        self._statistics_cache = {}
        self._cache_valid = False
    
    def add_data_point(self, value: float) -> None:
        """Add a data point to the statistical history."""
        if value is not None and not np.isnan(value):
            self._data_history.append(value)
            self._cache_valid = False
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical measures for the data history."""
        if not self._cache_valid and self._data_history:
            data = np.array(self._data_history)
            
            self._statistics_cache = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'count': len(data)
            }
            self._cache_valid = True
        
        return self._statistics_cache.copy()
    
    def calculate_z_score(self, value: float) -> float:
        """Calculate z-score for a value against the data history."""
        if not self._data_history:
            return 0.0
        
        stats = self.get_statistics()
        if stats['std'] == 0:
            return 0.0
        
        return abs(value - stats['mean']) / stats['std']
    
    def is_outlier(self, value: float, z_threshold: float = 2.0) -> bool:
        """Check if a value is a statistical outlier."""
        return self.calculate_z_score(value) > z_threshold
    
    def calculate_percentile_rank(self, value: float) -> float:
        """Calculate percentile rank of a value (0-100)."""
        if not self._data_history:
            return 50.0  # Default to median
        
        data = np.array(self._data_history)
        return (np.sum(data <= value) / len(data)) * 100.0

class TemporalIndicator(AnomalyIndicator):
    """Base class for indicators that analyze temporal patterns."""
    
    def __init__(self, name: str, config: IndicatorConfig):
        super().__init__(name, config)
    
    def extract_time_series(self, close_approaches: List[CloseApproach]) -> List[Tuple[datetime, float]]:
        """Extract time series data from close approaches."""
        time_series = []
        
        for approach in close_approaches:
            if approach.close_approach_date and approach.distance_au:
                time_series.append((approach.close_approach_date, approach.distance_au))
        
        # Sort by date
        time_series.sort(key=lambda x: x[0])
        
        return time_series
    
    def calculate_time_intervals(self, dates: List[datetime]) -> List[float]:
        """Calculate time intervals between dates in days."""
        if len(dates) < 2:
            return []
        
        intervals = []
        sorted_dates = sorted(dates)
        
        for i in range(1, len(sorted_dates)):
            delta = (sorted_dates[i] - sorted_dates[i-1]).total_seconds() / 86400.0  # Convert to days
            intervals.append(delta)
        
        return intervals
    
    def analyze_periodicity(self, intervals: List[float]) -> Dict[str, float]:
        """Analyze periodicity in time intervals."""
        if len(intervals) < 3:
            return {'regularity': 0.0, 'period_estimate': 0.0, 'variance': 0.0}
        
        intervals_array = np.array(intervals)
        
        mean_interval = np.mean(intervals_array)
        variance = np.var(intervals_array)
        
        # Calculate regularity (inverse of coefficient of variation)
        cv = np.sqrt(variance) / mean_interval if mean_interval > 0 else float('inf')
        regularity = 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
        
        return {
            'regularity': regularity,
            'period_estimate': mean_interval,
            'variance': variance,
            'coefficient_of_variation': cv
        }

class GeographicIndicator(AnomalyIndicator):
    """Base class for indicators that analyze geographic patterns."""
    
    def __init__(self, name: str, config: IndicatorConfig, regions_of_interest: List[Dict[str, Any]]):
        super().__init__(name, config)
        self.regions_of_interest = regions_of_interest
    
    def extract_subpoints(self, close_approaches: List[CloseApproach]) -> List[Tuple[float, float]]:
        """Extract geographic subpoints from close approaches."""
        subpoints = []
        
        for approach in close_approaches:
            if approach.subpoint and len(approach.subpoint) == 2:
                lat, lon = approach.subpoint
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    subpoints.append((lat, lon))
        
        return subpoints
    
    def calculate_distance_to_region(self, point: Tuple[float, float], 
                                   region: Dict[str, Any]) -> float:
        """Calculate distance from a point to a region center."""
        from geopy.distance import geodesic
        
        try:
            region_center = (region['lat'], region['lon'])
            return geodesic(point, region_center).kilometers
        except Exception as e:
            logger.warning(f"Error calculating distance to region: {e}")
            return float('inf')
    
    def count_region_passes(self, subpoints: List[Tuple[float, float]]) -> Dict[str, int]:
        """Count how many subpoints pass over regions of interest."""
        region_counts = {}
        
        for region in self.regions_of_interest:
            region_name = region.get('name', 'Unknown')
            region_radius = region.get('radius_km', 50)
            count = 0
            
            for point in subpoints:
                distance = self.calculate_distance_to_region(point, region)
                if distance <= region_radius:
                    count += 1
            
            region_counts[region_name] = count
        
        return region_counts