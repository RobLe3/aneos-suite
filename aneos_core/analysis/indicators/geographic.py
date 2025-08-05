"""
Geographic clustering anomaly indicators for aNEOS.

This module implements indicators that detect anomalies in the geographic
distribution of NEO close approaches that might suggest artificial targeting.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import logging
from collections import defaultdict
import math

from .base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    GeographicIndicator, StatisticalIndicator
)
from ...data.models import NEOData, CloseApproach

logger = logging.getLogger(__name__)

class SubpointClusteringIndicator(GeographicIndicator):
    """Detects suspicious clustering of subpoints over specific regions."""
    
    def __init__(self, config: IndicatorConfig):
        # Define regions of interest (major cities, military bases, space facilities)
        regions_of_interest = [
            {"name": "Washington_DC", "lat": 38.9072, "lon": -77.0369, "radius_km": 100},
            {"name": "Moscow", "lat": 55.7558, "lon": 37.6176, "radius_km": 100},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074, "radius_km": 100},
            {"name": "Kennedy_Space_Center", "lat": 28.5721, "lon": -80.6480, "radius_km": 50},
            {"name": "Baikonur", "lat": 45.6, "lon": 63.3, "radius_km": 50},
            {"name": "Vandenberg", "lat": 34.7420, "lon": -120.5724, "radius_km": 50},
            {"name": "Cape_Canaveral", "lat": 28.3922, "lon": -80.6077, "radius_km": 50},
            {"name": "Plesetsk", "lat": 62.7, "lon": 40.3, "radius_km": 50},
            {"name": "Jiuquan", "lat": 40.96, "lon": 100.29, "radius_km": 50},
            {"name": "Kourou", "lat": 5.1642, "lon": -52.6816, "radius_km": 50},
            {"name": "NORAD", "lat": 38.7441, "lon": -104.8242, "radius_km": 75},
            {"name": "Pentagon", "lat": 38.8719, "lon": -77.0563, "radius_km": 25},
            {"name": "Area_51", "lat": 37.2431, "lon": -115.7930, "radius_km": 50},
        ]
        
        super().__init__(name="subpoint_clustering", config=config, regions_of_interest=regions_of_interest)
        self.clustering_threshold = 0.3  # 30% of approaches in regions is suspicious
        self.density_threshold = 0.1  # Density threshold for DBSCAN-like clustering
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate subpoint clustering anomaly."""
        subpoints = self.extract_subpoints(neo_data.close_approaches)
        
        if len(subpoints) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_subpoints': len(subpoints)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Count region passes
        region_counts = self.count_region_passes(subpoints)
        total_approaches = len(subpoints)
        
        metadata = {
            'total_subpoints': total_approaches,
            'region_counts': region_counts,
            'subpoint_coordinates': subpoints
        }
        
        # Check for suspicious concentration in regions of interest
        total_region_passes = sum(region_counts.values())
        region_concentration = total_region_passes / total_approaches if total_approaches > 0 else 0
        
        if region_concentration > self.clustering_threshold:
            concentration_score = (region_concentration - self.clustering_threshold) / (1.0 - self.clustering_threshold)
            score += concentration_score * 0.8
            
            factors.append(f"High concentration in regions of interest: {region_concentration:.2%}")
            confidence = 0.9
            
            # Identify specific regions with high activity
            for region_name, count in region_counts.items():
                if count > 0:
                    region_ratio = count / total_approaches
                    if region_ratio > 0.2:  # More than 20% in one region
                        score = max(score, 0.7)
                        factors.append(f"High activity over {region_name}: {count}/{total_approaches} approaches")
                        confidence = 0.95
        
        # Perform general clustering analysis
        cluster_analysis = self._analyze_geographic_clusters(subpoints)
        
        if cluster_analysis['significant_clusters'] > 0:
            cluster_score = min(cluster_analysis['significant_clusters'] * 0.2, 0.6)
            score += cluster_score
            
            factors.append(f"Detected {cluster_analysis['significant_clusters']} significant geographic clusters")
            metadata['cluster_analysis'] = cluster_analysis
        
        # Check for suspicious patterns
        pattern_score = self._analyze_geographic_patterns(subpoints)
        if pattern_score > 0:
            score += pattern_score
            factors.append("Detected suspicious geographic patterns")
        
        # Check for land vs water distribution bias
        land_bias_score = self._analyze_land_water_bias(subpoints)
        if land_bias_score > 0:
            score += land_bias_score
            factors.append("Suspicious bias toward populated land areas")
        
        score = min(score, 1.0)  # Cap at 1.0
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def _analyze_geographic_clusters(self, subpoints: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze geographic clustering using density-based approach."""
        if len(subpoints) < 3:
            return {'significant_clusters': 0, 'cluster_details': []}
        
        # Simple density-based clustering
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
                    if distance < 500:  # 500 km threshold
                        cluster.append(j)
            
            if len(cluster) >= 2:  # At least 2 points for a cluster
                clusters.append({
                    'size': len(cluster),
                    'center': self._calculate_cluster_center([subpoints[idx] for idx in cluster]),
                    'max_distance': max(
                        self._calculate_great_circle_distance(subpoints[cluster[0]], subpoints[idx])
                        for idx in cluster[1:]
                    ) if len(cluster) > 1 else 0
                })
                processed.update(cluster)
        
        # Identify significant clusters (unusually dense)
        significant_clusters = 0
        for cluster in clusters:
            expected_density = len(subpoints) / (4 * math.pi * 6371**2)  # Points per km²
            cluster_area = math.pi * (cluster['max_distance']**2)
            actual_density = cluster['size'] / max(cluster_area, 1)
            
            if actual_density > expected_density * 10:  # 10x expected density
                significant_clusters += 1
        
        return {
            'significant_clusters': significant_clusters,
            'total_clusters': len(clusters),
            'cluster_details': clusters
        }
    
    def _analyze_geographic_patterns(self, subpoints: List[Tuple[float, float]]) -> float:
        """Analyze for artificial geographic patterns."""
        if len(subpoints) < 4:
            return 0.0
        
        score = 0.0
        
        # Check for grid-like patterns
        lats = [point[0] for point in subpoints]
        lons = [point[1] for point in subpoints]
        
        # Check for suspicious regularity in coordinates
        lat_diffs = [abs(lats[i] - lats[j]) for i in range(len(lats)) for j in range(i+1, len(lats))]
        lon_diffs = [abs(lons[i] - lons[j]) for i in range(len(lons)) for j in range(i+1, len(lons))]
        
        # Look for repeated coordinate differences (suggesting grid pattern)
        if lat_diffs:
            lat_diff_counts = defaultdict(int)
            for diff in lat_diffs:
                rounded_diff = round(diff, 1)  # Round to 0.1 degree
                if rounded_diff > 0:
                    lat_diff_counts[rounded_diff] += 1
            
            max_lat_repeats = max(lat_diff_counts.values()) if lat_diff_counts else 0
            if max_lat_repeats > len(subpoints) * 0.3:  # More than 30% show same difference
                score += 0.3
        
        if lon_diffs:
            lon_diff_counts = defaultdict(int)
            for diff in lon_diffs:
                rounded_diff = round(diff, 1)  # Round to 0.1 degree
                if rounded_diff > 0:
                    lon_diff_counts[rounded_diff] += 1
            
            max_lon_repeats = max(lon_diff_counts.values()) if lon_diff_counts else 0
            if max_lon_repeats > len(subpoints) * 0.3:  # More than 30% show same difference
                score += 0.3
        
        # Check for linear arrangements
        if len(subpoints) >= 3:
            linear_score = self._check_linear_arrangement(subpoints)
            score += linear_score
        
        return min(score, 0.6)  # Cap pattern score at 0.6
    
    def _analyze_land_water_bias(self, subpoints: List[Tuple[float, float]]) -> float:
        """Analyze bias toward land areas vs water."""
        if len(subpoints) < 5:
            return 0.0
        
        # Simplified land detection (rough approximation)
        # Earth is ~71% water, so random distribution should reflect this
        land_points = 0
        
        for lat, lon in subpoints:
            # Very rough land detection based on coordinate ranges
            # This is simplified - real implementation would use geographic databases
            if self._is_likely_land(lat, lon):
                land_points += 1
        
        land_ratio = land_points / len(subpoints)
        expected_land_ratio = 0.29  # ~29% land on Earth
        
        # If significantly more points are over land than expected
        if land_ratio > expected_land_ratio * 2:  # More than double expected
            bias_strength = (land_ratio - expected_land_ratio) / (1.0 - expected_land_ratio)
            return min(bias_strength * 0.4, 0.4)
        
        return 0.0
    
    def _is_likely_land(self, lat: float, lon: float) -> bool:
        """Simplified land detection (rough approximation)."""
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
    
    def _check_linear_arrangement(self, subpoints: List[Tuple[float, float]]) -> float:
        """Check if points are arranged in linear patterns."""
        if len(subpoints) < 3:
            return 0.0
        
        # Try all combinations of 3+ points and check linearity
        max_linear_fraction = 0.0
        
        for i in range(len(subpoints)):
            for j in range(i+1, len(subpoints)):
                # Define line between points i and j
                p1 = subpoints[i]
                p2 = subpoints[j]
                
                # Count how many other points are close to this line
                linear_count = 2  # p1 and p2 are on the line
                
                for k in range(len(subpoints)):
                    if k != i and k != j:
                        distance_to_line = self._point_to_line_distance(subpoints[k], p1, p2)
                        if distance_to_line < 100:  # Within 100 km of line
                            linear_count += 1
                
                linear_fraction = linear_count / len(subpoints)
                max_linear_fraction = max(max_linear_fraction, linear_fraction)
        
        # If more than 60% of points are linear
        if max_linear_fraction > 0.6:
            return min((max_linear_fraction - 0.4) / 0.4, 0.3)
        
        return 0.0
    
    def _calculate_great_circle_distance(self, point1: Tuple[float, float], 
                                       point2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two points in km."""
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
    
    def _calculate_cluster_center(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate center of a cluster of points."""
        if not points:
            return (0.0, 0.0)
        
        # Simple centroid calculation
        mean_lat = sum(point[0] for point in points) / len(points)
        mean_lon = sum(point[1] for point in points) / len(points)
        
        return (mean_lat, mean_lon)
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line_p1: Tuple[float, float], 
                              line_p2: Tuple[float, float]) -> float:
        """Calculate approximate distance from point to line on sphere (simplified)."""
        # Simplified calculation - not geodesically accurate but sufficient for detection
        
        # Convert to Cartesian for easier calculation
        def to_cartesian(lat, lon):
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            x = math.cos(lat_rad) * math.cos(lon_rad)
            y = math.cos(lat_rad) * math.sin(lon_rad)
            z = math.sin(lat_rad)
            return (x, y, z)
        
        p = to_cartesian(point[0], point[1])
        a = to_cartesian(line_p1[0], line_p1[1])
        b = to_cartesian(line_p2[0], line_p2[1])
        
        # Vector from a to b
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        
        # Vector from a to p
        ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
        
        # Project ap onto ab
        ab_length_sq = ab[0]**2 + ab[1]**2 + ab[2]**2
        if ab_length_sq == 0:
            return self._calculate_great_circle_distance(point, line_p1)
        
        t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab_length_sq
        t = max(0, min(1, t))  # Clamp to line segment
        
        # Closest point on line
        closest = (a[0] + t*ab[0], a[1] + t*ab[1], a[2] + t*ab[2])
        
        # Distance (simplified - using Euclidean distance then scaling)
        distance = math.sqrt((p[0] - closest[0])**2 + (p[1] - closest[1])**2 + (p[2] - closest[2])**2)
        
        # Convert back to km (very approximate)
        return distance * 6371
    
    def get_description(self) -> str:
        return "Detects suspicious clustering of subpoints over specific regions"

class GeographicBiasIndicator(StatisticalIndicator):
    """Detects geographic bias in NEO close approaches."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="geographic_bias", config=config)
        self.hemisphere_bias_threshold = 0.8  # 80% in one hemisphere is suspicious
        self.timezone_bias_threshold = 0.4  # 40% in one timezone region
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate geographic bias anomaly."""
        subpoints = self.extract_subpoints(neo_data.close_approaches)
        
        if len(subpoints) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_subpoints': len(subpoints)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Analyze hemisphere distribution
        northern_count = sum(1 for lat, lon in subpoints if lat > 0)
        southern_count = len(subpoints) - northern_count
        eastern_count = sum(1 for lat, lon in subpoints if lon > 0)
        western_count = len(subpoints) - eastern_count
        
        metadata = {
            'total_subpoints': len(subpoints),
            'northern_hemisphere': northern_count,
            'southern_hemisphere': southern_count,
            'eastern_hemisphere': eastern_count,
            'western_hemisphere': western_count
        }
        
        # Check for hemisphere bias
        north_south_bias = max(northern_count, southern_count) / len(subpoints)
        east_west_bias = max(eastern_count, western_count) / len(subpoints)
        
        if north_south_bias > self.hemisphere_bias_threshold:
            bias_score = (north_south_bias - 0.5) / 0.5  # Normalize excess above 50%
            score += bias_score * 0.4
            
            dominant_hemisphere = "Northern" if northern_count > southern_count else "Southern"
            factors.append(f"Strong {dominant_hemisphere} hemisphere bias: {north_south_bias:.1%}")
            confidence = 0.8
        
        if east_west_bias > self.hemisphere_bias_threshold:
            bias_score = (east_west_bias - 0.5) / 0.5
            score += bias_score * 0.4
            
            dominant_hemisphere = "Eastern" if eastern_count > western_count else "Western"
            factors.append(f"Strong {dominant_hemisphere} hemisphere bias: {east_west_bias:.1%}")
            confidence = 0.8
        
        # Analyze timezone distribution
        timezone_analysis = self._analyze_timezone_distribution(subpoints)
        
        if timezone_analysis['max_timezone_fraction'] > self.timezone_bias_threshold:
            timezone_score = (timezone_analysis['max_timezone_fraction'] - 0.2) / 0.6  # Normalize
            score += min(timezone_score * 0.5, 0.5)
            
            factors.append(f"Timezone bias: {timezone_analysis['max_timezone_fraction']:.1%} in one region")
            metadata['timezone_analysis'] = timezone_analysis
        
        # Check for latitude band clustering
        latitude_bias = self._analyze_latitude_clustering(subpoints)
        if latitude_bias > 0:
            score += latitude_bias
            factors.append("Suspicious clustering in specific latitude bands")
        
        # Add statistical analysis
        lats = [lat for lat, lon in subpoints]
        lons = [lon for lat, lon in subpoints]
        
        for lat in lats:
            self.add_data_point(lat + 90)  # Normalize to 0-180 range
        
        # Check for statistical outliers in distribution
        if len(self._data_history) > 20:
            outlier_count = sum(1 for lat in lats if self.is_outlier(lat + 90, 2.0))
            if outlier_count > 0:
                outlier_score = min(outlier_count / len(lats) * 0.3, 0.3)
                score += outlier_score
                factors.append(f"{outlier_count} statistically anomalous latitudes")
        
        score = min(score, 1.0)  # Cap at 1.0
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def extract_subpoints(self, close_approaches: List[CloseApproach]) -> List[Tuple[float, float]]:
        """Extract geographic subpoints from close approaches."""
        subpoints = []
        
        for approach in close_approaches:
            if hasattr(approach, 'subpoint') and approach.subpoint and len(approach.subpoint) == 2:
                lat, lon = approach.subpoint
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    subpoints.append((lat, lon))
        
        return subpoints
    
    def _analyze_timezone_distribution(self, subpoints: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze distribution across timezone regions."""
        timezone_counts = defaultdict(int)
        
        for lat, lon in subpoints:
            # Divide into 12 timezone regions (30° each)
            timezone_region = int((lon + 180) // 30)
            timezone_counts[timezone_region] += 1
        
        max_count = max(timezone_counts.values()) if timezone_counts else 0
        max_fraction = max_count / len(subpoints) if subpoints else 0
        
        return {
            'timezone_distribution': dict(timezone_counts),
            'max_timezone_count': max_count,
            'max_timezone_fraction': max_fraction,
            'unique_timezones': len(timezone_counts)
        }
    
    def _analyze_latitude_clustering(self, subpoints: List[Tuple[float, float]]) -> float:
        """Analyze clustering in latitude bands."""
        if len(subpoints) < 4:
            return 0.0
        
        # Define latitude bands (15° each)
        band_counts = defaultdict(int)
        
        for lat, lon in subpoints:
            band = int((lat + 90) // 15)  # 0-11 bands
            band_counts[band] += 1
        
        # Check for excessive clustering in specific bands
        max_band_count = max(band_counts.values()) if band_counts else 0
        max_band_fraction = max_band_count / len(subpoints)
        
        # Expected fraction for random distribution across 12 bands is ~8.3%
        expected_fraction = 1.0 / 12
        
        if max_band_fraction > expected_fraction * 4:  # 4x expected
            excess = (max_band_fraction - expected_fraction) / (1.0 - expected_fraction)
            return min(excess * 0.3, 0.3)
        
        return 0.0
    
    def get_description(self) -> str:
        return "Detects geographic bias in NEO close approach distributions"