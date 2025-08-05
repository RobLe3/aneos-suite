"""
Feature engineering for aNEOS machine learning models.

This module provides comprehensive feature extraction and engineering
capabilities for NEO anomaly detection models.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from ..data.models import NEOData, OrbitalElements, CloseApproach
from ..analysis.indicators.base import IndicatorResult

logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:
    """Complete feature vector for a NEO."""
    designation: str
    features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'designation': self.designation,
            'features': self.features.tolist(),
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = dict(zip(self.feature_names, self.features))
        data['designation'] = self.designation
        return pd.DataFrame([data])

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract(self, neo_data: NEOData) -> Dict[str, float]:
        """Extract features from NEO data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of features extracted by this extractor."""
        pass

class OrbitalFeatureExtractor(FeatureExtractor):
    """Extract features from orbital elements."""
    
    def extract(self, neo_data: NEOData) -> Dict[str, float]:
        """Extract orbital features."""
        features = {}
        
        if not neo_data.orbital_elements:
            # Return zeros for missing orbital elements
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        elements = neo_data.orbital_elements
        
        # Basic orbital elements
        features['eccentricity'] = elements.eccentricity or 0.0
        features['inclination'] = elements.inclination or 0.0
        features['semi_major_axis'] = elements.semi_major_axis or 0.0
        features['ascending_node'] = elements.ascending_node or 0.0
        features['argument_of_perihelion'] = elements.argument_of_perihelion or 0.0
        features['mean_anomaly'] = elements.mean_anomaly or 0.0
        
        # Derived orbital features
        if elements.semi_major_axis and elements.eccentricity is not None:
            # Perihelion and aphelion distances
            features['perihelion_distance'] = elements.semi_major_axis * (1 - elements.eccentricity)
            features['aphelion_distance'] = elements.semi_major_axis * (1 + elements.eccentricity)
            
            # Orbital period (Kepler's third law)
            features['orbital_period'] = elements.semi_major_axis ** 1.5
            
            # Earth crossing indicators
            features['earth_crossing'] = 1.0 if (features['perihelion_distance'] < 1.3 and features['aphelion_distance'] > 0.7) else 0.0
            
            # Potentially hazardous asteroid criteria
            features['potentially_hazardous'] = 1.0 if (features['perihelion_distance'] < 1.3 and 
                                                       getattr(neo_data, 'diameter_km', 0) > 0.14) else 0.0
        else:
            features['perihelion_distance'] = 0.0
            features['aphelion_distance'] = 0.0
            features['orbital_period'] = 0.0
            features['earth_crossing'] = 0.0
            features['potentially_hazardous'] = 0.0
        
        # Angular momentum components (if available)
        if (elements.inclination is not None and elements.ascending_node is not None and
            elements.argument_of_perihelion is not None):
            
            # Convert to radians for calculations
            i = np.radians(elements.inclination)
            omega = np.radians(elements.ascending_node)
            w = np.radians(elements.argument_of_perihelion)
            
            # Angular momentum vector components (normalized)
            features['angular_momentum_x'] = np.sin(i) * np.sin(omega + w)
            features['angular_momentum_y'] = -np.sin(i) * np.cos(omega + w)
            features['angular_momentum_z'] = np.cos(i)
        else:
            features['angular_momentum_x'] = 0.0
            features['angular_momentum_y'] = 0.0
            features['angular_momentum_z'] = 0.0
        
        # Orbital energy
        if elements.semi_major_axis:
            features['specific_orbital_energy'] = -1.0 / (2 * elements.semi_major_axis)
        else:
            features['specific_orbital_energy'] = 0.0
        
        # Eccentricity-based classifications
        features['circular_orbit'] = 1.0 if (elements.eccentricity or 0) < 0.1 else 0.0
        features['elliptical_orbit'] = 1.0 if 0.1 <= (elements.eccentricity or 0) < 1.0 else 0.0
        features['parabolic_orbit'] = 1.0 if abs((elements.eccentricity or 0) - 1.0) < 0.01 else 0.0
        features['hyperbolic_orbit'] = 1.0 if (elements.eccentricity or 0) > 1.0 else 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get orbital feature names."""
        return [
            'eccentricity', 'inclination', 'semi_major_axis', 'ascending_node',
            'argument_of_perihelion', 'mean_anomaly', 'perihelion_distance',
            'aphelion_distance', 'orbital_period', 'earth_crossing',
            'potentially_hazardous', 'angular_momentum_x', 'angular_momentum_y',
            'angular_momentum_z', 'specific_orbital_energy', 'circular_orbit',
            'elliptical_orbit', 'parabolic_orbit', 'hyperbolic_orbit'
        ]

class VelocityFeatureExtractor(FeatureExtractor):
    """Extract features from velocity and close approach data."""
    
    def extract(self, neo_data: NEOData) -> Dict[str, float]:
        """Extract velocity features."""
        features = {}
        
        if not neo_data.close_approaches:
            # Return zeros for missing close approach data
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        # Extract velocity data
        velocities = [approach.relative_velocity_km_s for approach in neo_data.close_approaches 
                     if approach.relative_velocity_km_s is not None]
        
        if not velocities:
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        velocities = np.array(velocities)
        
        # Statistical features
        features['velocity_mean'] = np.mean(velocities)
        features['velocity_std'] = np.std(velocities)
        features['velocity_min'] = np.min(velocities)
        features['velocity_max'] = np.max(velocities)
        features['velocity_median'] = np.median(velocities)
        features['velocity_range'] = np.max(velocities) - np.min(velocities)
        
        # Coefficient of variation
        features['velocity_cv'] = features['velocity_std'] / features['velocity_mean'] if features['velocity_mean'] > 0 else 0.0
        
        # Velocity change features
        if len(velocities) > 1:
            velocity_diffs = np.diff(velocities)
            features['velocity_change_mean'] = np.mean(np.abs(velocity_diffs))
            features['velocity_change_max'] = np.max(np.abs(velocity_diffs))
            features['velocity_change_std'] = np.std(velocity_diffs)
        else:
            features['velocity_change_mean'] = 0.0
            features['velocity_change_max'] = 0.0
            features['velocity_change_std'] = 0.0
        
        # Infinity velocity features
        v_inf_values = [approach.infinity_velocity_km_s for approach in neo_data.close_approaches 
                       if approach.infinity_velocity_km_s is not None]
        
        if v_inf_values:
            v_inf_array = np.array(v_inf_values)
            features['v_inf_mean'] = np.mean(v_inf_array)
            features['v_inf_std'] = np.std(v_inf_array)
            features['v_inf_min'] = np.min(v_inf_array)
            features['v_inf_max'] = np.max(v_inf_array)
        else:
            features['v_inf_mean'] = 0.0
            features['v_inf_std'] = 0.0
            features['v_inf_min'] = 0.0
            features['v_inf_max'] = 0.0
        
        # Acceleration features (if we have time-ordered data)
        sorted_approaches = sorted(
            [a for a in neo_data.close_approaches if a.close_approach_date and a.relative_velocity_km_s],
            key=lambda x: x.close_approach_date
        )
        
        if len(sorted_approaches) >= 2:
            accelerations = []
            for i in range(1, len(sorted_approaches)):
                prev_approach = sorted_approaches[i-1]
                curr_approach = sorted_approaches[i]
                
                time_diff = (curr_approach.close_approach_date - prev_approach.close_approach_date).total_seconds()
                if time_diff > 0:
                    velocity_diff = curr_approach.relative_velocity_km_s - prev_approach.relative_velocity_km_s
                    acceleration = abs(velocity_diff) / time_diff
                    accelerations.append(acceleration)
            
            if accelerations:
                features['acceleration_mean'] = np.mean(accelerations)
                features['acceleration_max'] = np.max(accelerations)
                features['acceleration_std'] = np.std(accelerations)
            else:
                features['acceleration_mean'] = 0.0
                features['acceleration_max'] = 0.0
                features['acceleration_std'] = 0.0
        else:
            features['acceleration_mean'] = 0.0
            features['acceleration_max'] = 0.0
            features['acceleration_std'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get velocity feature names."""
        return [
            'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max',
            'velocity_median', 'velocity_range', 'velocity_cv',
            'velocity_change_mean', 'velocity_change_max', 'velocity_change_std',
            'v_inf_mean', 'v_inf_std', 'v_inf_min', 'v_inf_max',
            'acceleration_mean', 'acceleration_max', 'acceleration_std'
        ]

class TemporalFeatureExtractor(FeatureExtractor):
    """Extract features from temporal patterns."""
    
    def extract(self, neo_data: NEOData) -> Dict[str, float]:
        """Extract temporal features."""
        features = {}
        
        if not neo_data.close_approaches:
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        # Extract dates
        dates = [approach.close_approach_date for approach in neo_data.close_approaches 
                if approach.close_approach_date is not None]
        
        if len(dates) < 2:
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        # Sort dates
        dates = sorted(dates)
        
        # Time span features
        total_span = (dates[-1] - dates[0]).total_seconds() / 86400.0  # Convert to days
        features['observation_span_days'] = total_span
        features['observation_count'] = len(dates)
        features['observation_density'] = len(dates) / max(total_span, 1.0)  # Observations per day
        
        # Time interval features
        intervals = [(dates[i] - dates[i-1]).total_seconds() / 86400.0 for i in range(1, len(dates))]
        
        if intervals:
            intervals = np.array(intervals)
            features['interval_mean'] = np.mean(intervals)
            features['interval_std'] = np.std(intervals)
            features['interval_min'] = np.min(intervals)
            features['interval_max'] = np.max(intervals)
            features['interval_cv'] = features['interval_std'] / features['interval_mean'] if features['interval_mean'] > 0 else 0.0
            
            # Regularity measure (inverse of coefficient of variation)
            features['temporal_regularity'] = 1.0 / (1.0 + features['interval_cv'])
        else:
            features['interval_mean'] = 0.0
            features['interval_std'] = 0.0
            features['interval_min'] = 0.0
            features['interval_max'] = 0.0
            features['interval_cv'] = 0.0
            features['temporal_regularity'] = 0.0
        
        # Seasonal features (based on day of year)
        days_of_year = [date.timetuple().tm_yday for date in dates]
        
        if len(days_of_year) >= 3:
            # Convert to circular coordinates for seasonal analysis
            angles = [2 * np.pi * day / 365.25 for day in days_of_year]
            sin_sum = np.sum(np.sin(angles))
            cos_sum = np.sum(np.cos(angles))
            
            # Resultant length (measure of seasonal concentration)
            features['seasonal_concentration'] = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
            
            # Mean seasonal angle
            mean_angle = np.arctan2(sin_sum, cos_sum)
            features['seasonal_phase'] = mean_angle / (2 * np.pi)  # Normalize to [0, 1]
        else:
            features['seasonal_concentration'] = 0.0
            features['seasonal_phase'] = 0.0
        
        # Gap analysis
        large_gaps = [interval for interval in intervals if interval > 365]  # Gaps > 1 year
        features['large_gap_count'] = len(large_gaps)
        features['large_gap_fraction'] = len(large_gaps) / len(intervals) if intervals else 0.0
        
        # Periodicity detection (simplified)
        if len(intervals) >= 4:
            # Look for repeating patterns
            pattern_scores = []
            for period_length in range(2, min(5, len(intervals) // 2)):
                score = self._calculate_periodicity_score(intervals, period_length)
                pattern_scores.append(score)
            
            features['max_periodicity_score'] = max(pattern_scores) if pattern_scores else 0.0
        else:
            features['max_periodicity_score'] = 0.0
        
        return features
    
    def _calculate_periodicity_score(self, intervals: List[float], period_length: int) -> float:
        """Calculate periodicity score for a given period length."""
        if len(intervals) < period_length * 2:
            return 0.0
        
        # Compare patterns across periods
        patterns = []
        for start in range(0, len(intervals) - period_length + 1, period_length):
            pattern = intervals[start:start + period_length]
            if len(pattern) == period_length:
                patterns.append(pattern)
        
        if len(patterns) < 2:
            return 0.0
        
        # Calculate similarity between patterns
        similarities = []
        for i in range(len(patterns) - 1):
            pattern1 = np.array(patterns[i])
            pattern2 = np.array(patterns[i + 1])
            
            # Normalized cross-correlation
            if np.std(pattern1) > 0 and np.std(pattern2) > 0:
                correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                similarities.append(abs(correlation))
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_feature_names(self) -> List[str]:
        """Get temporal feature names."""
        return [
            'observation_span_days', 'observation_count', 'observation_density',
            'interval_mean', 'interval_std', 'interval_min', 'interval_max',
            'interval_cv', 'temporal_regularity', 'seasonal_concentration',
            'seasonal_phase', 'large_gap_count', 'large_gap_fraction',
            'max_periodicity_score'
        ]

class GeographicFeatureExtractor(FeatureExtractor):
    """Extract features from geographic patterns."""
    
    def extract(self, neo_data: NEOData) -> Dict[str, float]:
        """Extract geographic features."""
        features = {}
        
        # Extract subpoints
        subpoints = []
        for approach in neo_data.close_approaches:
            if hasattr(approach, 'subpoint') and approach.subpoint and len(approach.subpoint) == 2:
                lat, lon = approach.subpoint
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    subpoints.append((lat, lon))
        
        if not subpoints:
            for name in self.get_feature_names():
                features[name] = 0.0
            return features
        
        lats = [point[0] for point in subpoints]
        lons = [point[1] for point in subpoints]
        
        # Basic statistical features
        features['latitude_mean'] = np.mean(lats)
        features['latitude_std'] = np.std(lats)
        features['latitude_range'] = np.max(lats) - np.min(lats)
        features['longitude_mean'] = np.mean(lons)
        features['longitude_std'] = np.std(lons)
        features['longitude_range'] = np.max(lons) - np.min(lons)
        
        # Hemisphere distribution
        northern_count = sum(1 for lat in lats if lat > 0)
        eastern_count = sum(1 for lon in lons if lon > 0)
        
        features['northern_hemisphere_fraction'] = northern_count / len(lats)
        features['southern_hemisphere_fraction'] = 1.0 - features['northern_hemisphere_fraction']
        features['eastern_hemisphere_fraction'] = eastern_count / len(lons)
        features['western_hemisphere_fraction'] = 1.0 - features['eastern_hemisphere_fraction']
        
        # Hemisphere bias (deviation from 50/50)
        features['north_south_bias'] = abs(features['northern_hemisphere_fraction'] - 0.5)
        features['east_west_bias'] = abs(features['eastern_hemisphere_fraction'] - 0.5)
        
        # Land vs water bias (simplified)
        land_count = sum(1 for lat, lon in subpoints if self._is_likely_land(lat, lon))
        features['land_fraction'] = land_count / len(subpoints)
        features['water_fraction'] = 1.0 - features['land_fraction']
        
        # Expected land fraction is ~29%
        features['land_bias'] = abs(features['land_fraction'] - 0.29)
        
        # Clustering features
        if len(subpoints) >= 3:
            clustering_metrics = self._calculate_clustering_metrics(subpoints)
            features.update(clustering_metrics)
        else:
            features['clustering_coefficient'] = 0.0
            features['max_cluster_size'] = 0.0
            features['cluster_density'] = 0.0
        
        # Geographic dispersion
        if len(subpoints) >= 2:
            distances = []
            for i in range(len(subpoints)):
                for j in range(i + 1, len(subpoints)):
                    distance = self._calculate_great_circle_distance(subpoints[i], subpoints[j])
                    distances.append(distance)
            
            features['mean_pairwise_distance'] = np.mean(distances)
            features['max_pairwise_distance'] = np.max(distances)
            features['min_pairwise_distance'] = np.min(distances)
        else:
            features['mean_pairwise_distance'] = 0.0
            features['max_pairwise_distance'] = 0.0
            features['min_pairwise_distance'] = 0.0
        
        return features
    
    def _is_likely_land(self, lat: float, lon: float) -> bool:
        """Simplified land detection."""
        # Major continental masses (very simplified)
        
        # North America
        if 25 <= lat <= 70 and -170 <= lon <= -50:
            return True
        
        # South America
        if -55 <= lat <= 15 and -85 <= lon <= -35:
            return True
        
        # Europe
        if 35 <= lat <= 75 and -10 <= lon <= 50:
            return True
        
        # Asia
        if 10 <= lat <= 80 and 50 <= lon <= 180:
            return True
        
        # Africa
        if -35 <= lat <= 35 and -20 <= lon <= 55:
            return True
        
        # Australia
        if -45 <= lat <= -10 and 110 <= lon <= 155:
            return True
        
        return False
    
    def _calculate_great_circle_distance(self, point1: Tuple[float, float], 
                                       point2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two points in km."""
        import math
        
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in km
        R = 6371
        return R * c
    
    def _calculate_clustering_metrics(self, subpoints: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate clustering metrics."""
        # Simple density-based clustering
        cluster_threshold = 500  # km
        clusters = []
        processed = set()
        
        for i, point in enumerate(subpoints):
            if i in processed:
                continue
            
            # Find all points within threshold distance
            cluster = [i]
            for j, other_point in enumerate(subpoints):
                if j != i and j not in processed:
                    distance = self._calculate_great_circle_distance(point, other_point)
                    if distance < cluster_threshold:
                        cluster.append(j)
            
            if len(cluster) >= 2:  # At least 2 points for a cluster
                clusters.append(cluster)
                processed.update(cluster)
        
        # Calculate metrics
        if clusters:
            max_cluster_size = max(len(cluster) for cluster in clusters)
            total_clustered = sum(len(cluster) for cluster in clusters)
            clustering_coefficient = total_clustered / len(subpoints)
            
            # Average cluster density
            cluster_densities = []
            for cluster in clusters:
                if len(cluster) > 1:
                    cluster_points = [subpoints[i] for i in cluster]
                    # Calculate cluster area (simplified as bounding box)
                    lats = [p[0] for p in cluster_points]
                    lons = [p[1] for p in cluster_points]
                    
                    lat_range = max(lats) - min(lats)
                    lon_range = max(lons) - min(lons)
                    
                    # Approximate area in km² (very rough)
                    area = lat_range * 111.0 * lon_range * 111.0 * np.cos(np.radians(np.mean(lats)))
                    density = len(cluster) / max(area, 1.0)
                    cluster_densities.append(density)
            
            cluster_density = np.mean(cluster_densities) if cluster_densities else 0.0
        else:
            max_cluster_size = 0.0
            clustering_coefficient = 0.0
            cluster_density = 0.0
        
        return {
            'clustering_coefficient': clustering_coefficient,
            'max_cluster_size': max_cluster_size,
            'cluster_density': cluster_density
        }
    
    def get_feature_names(self) -> List[str]:
        """Get geographic feature names."""
        return [
            'latitude_mean', 'latitude_std', 'latitude_range',
            'longitude_mean', 'longitude_std', 'longitude_range',
            'northern_hemisphere_fraction', 'southern_hemisphere_fraction',
            'eastern_hemisphere_fraction', 'western_hemisphere_fraction',
            'north_south_bias', 'east_west_bias',
            'land_fraction', 'water_fraction', 'land_bias',
            'clustering_coefficient', 'max_cluster_size', 'cluster_density',
            'mean_pairwise_distance', 'max_pairwise_distance', 'min_pairwise_distance'
        ]

class IndicatorFeatureExtractor(FeatureExtractor):
    """Extract features from anomaly indicator results."""
    
    def extract(self, indicator_results: Dict[str, IndicatorResult]) -> Dict[str, float]:
        """Extract features from indicator results."""
        features = {}
        
        # Individual indicator scores
        for indicator_name, result in indicator_results.items():
            features[f'{indicator_name}_raw_score'] = result.raw_score
            features[f'{indicator_name}_weighted_score'] = result.weighted_score
            features[f'{indicator_name}_confidence'] = result.confidence
        
        # Aggregate features
        raw_scores = [result.raw_score for result in indicator_results.values()]
        weighted_scores = [result.weighted_score for result in indicator_results.values()]
        confidences = [result.confidence for result in indicator_results.values()]
        
        if raw_scores:
            features['indicator_raw_mean'] = np.mean(raw_scores)
            features['indicator_raw_std'] = np.std(raw_scores)
            features['indicator_raw_max'] = np.max(raw_scores)
            features['indicator_weighted_mean'] = np.mean(weighted_scores)
            features['indicator_weighted_std'] = np.std(weighted_scores)
            features['indicator_confidence_mean'] = np.mean(confidences)
            
            # Count of active indicators
            features['active_indicator_count'] = sum(1 for score in raw_scores if score > 0)
            features['high_score_indicator_count'] = sum(1 for score in raw_scores if score > 0.5)
            features['high_confidence_indicator_count'] = sum(1 for conf in confidences if conf > 0.8)
        else:
            for name in ['indicator_raw_mean', 'indicator_raw_std', 'indicator_raw_max',
                        'indicator_weighted_mean', 'indicator_weighted_std', 'indicator_confidence_mean',
                        'active_indicator_count', 'high_score_indicator_count', 'high_confidence_indicator_count']:
                features[name] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get indicator feature names (dynamic based on available indicators)."""
        # Base names - actual names will be generated dynamically
        return [
            'indicator_raw_mean', 'indicator_raw_std', 'indicator_raw_max',
            'indicator_weighted_mean', 'indicator_weighted_std', 'indicator_confidence_mean',
            'active_indicator_count', 'high_score_indicator_count', 'high_confidence_indicator_count'
        ]

class FeatureEngineer:
    """Main feature engineering coordinator."""
    
    def __init__(self):
        """Initialize feature engineer with all extractors."""
        self.extractors = {
            'orbital': OrbitalFeatureExtractor(),
            'velocity': VelocityFeatureExtractor(),
            'temporal': TemporalFeatureExtractor(),
            'geographic': GeographicFeatureExtractor(),
            'indicator': IndicatorFeatureExtractor()
        }
        
        # Cache feature names
        self._feature_names = None
        
        logger.info(f"FeatureEngineer initialized with {len(self.extractors)} extractors")
    
    def extract_features(self, neo_data: NEOData, 
                        indicator_results: Optional[Dict[str, IndicatorResult]] = None) -> FeatureVector:
        """Extract complete feature vector from NEO data."""
        all_features = {}
        
        # Extract features from each extractor
        for extractor_name, extractor in self.extractors.items():
            try:
                if extractor_name == 'indicator' and indicator_results:
                    features = extractor.extract(indicator_results)
                elif extractor_name != 'indicator':
                    features = extractor.extract(neo_data)
                else:
                    features = {}
                
                # Add prefix to avoid name collisions
                prefixed_features = {f'{extractor_name}_{name}': value 
                                   for name, value in features.items()}
                all_features.update(prefixed_features)
                
            except Exception as e:
                logger.warning(f"Failed to extract {extractor_name} features for {neo_data.designation}: {e}")
                
                # Add zero features for failed extractors
                for name in extractor.get_feature_names():
                    all_features[f'{extractor_name}_{name}'] = 0.0
        
        # Create feature vector
        feature_names = sorted(all_features.keys())
        feature_values = np.array([all_features[name] for name in feature_names])
        
        # Add metadata
        metadata = {
            'data_quality': self._assess_feature_quality(all_features),
            'extraction_timestamp': datetime.now().isoformat(),
            'extractor_versions': {name: '2.0.0' for name in self.extractors.keys()}
        }
        
        return FeatureVector(
            designation=neo_data.designation,
            features=feature_values,
            feature_names=feature_names,
            metadata=metadata
        )
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        if self._feature_names is None:
            all_names = []
            for extractor_name, extractor in self.extractors.items():
                if extractor_name != 'indicator':  # Indicator features are dynamic
                    names = [f'{extractor_name}_{name}' for name in extractor.get_feature_names()]
                    all_names.extend(names)
            
            # Add base indicator feature names
            indicator_names = [f'indicator_{name}' for name in self.extractors['indicator'].get_feature_names()]
            all_names.extend(indicator_names)
            
            self._feature_names = sorted(all_names)
        
        return self._feature_names
    
    def _assess_feature_quality(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Assess quality of extracted features."""
        total_features = len(features)
        zero_features = sum(1 for value in features.values() if value == 0.0)
        nan_features = sum(1 for value in features.values() if np.isnan(value))
        inf_features = sum(1 for value in features.values() if np.isinf(value))
        
        return {
            'total_features': total_features,
            'zero_features': zero_features,
            'nan_features': nan_features,
            'inf_features': inf_features,
            'completeness': (total_features - zero_features) / total_features if total_features > 0 else 0.0,
            'validity': (total_features - nan_features - inf_features) / total_features if total_features > 0 else 0.0
        }
    
    def transform_features(self, feature_vector: FeatureVector, 
                          transformations: Optional[List[str]] = None) -> FeatureVector:
        """Apply transformations to feature vector."""
        if transformations is None:
            transformations = ['normalize', 'handle_missing']
        
        features = feature_vector.features.copy()
        
        for transformation in transformations:
            if transformation == 'normalize':
                # Z-score normalization
                mean = np.mean(features)
                std = np.std(features)
                if std > 0:
                    features = (features - mean) / std
            
            elif transformation == 'handle_missing':
                # Replace NaN and Inf with zeros
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            elif transformation == 'log_transform':
                # Log transform for skewed features (add 1 to handle zeros)
                features = np.log1p(np.abs(features))
        
        return FeatureVector(
            designation=feature_vector.designation,
            features=features,
            feature_names=feature_vector.feature_names,
            metadata={
                **feature_vector.metadata,
                'transformations_applied': transformations
            }
        )
    
    def create_feature_matrix(self, feature_vectors: List[FeatureVector]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create feature matrix from multiple feature vectors."""
        if not feature_vectors:
            return np.array([]), [], []
        
        # Ensure all feature vectors have the same features
        reference_names = feature_vectors[0].feature_names
        
        feature_matrix = []
        designations = []
        
        for fv in feature_vectors:
            if fv.feature_names == reference_names:
                feature_matrix.append(fv.features)
                designations.append(fv.designation)
            else:
                logger.warning(f"Feature vector for {fv.designation} has mismatched features")
        
        return np.array(feature_matrix), reference_names, designations