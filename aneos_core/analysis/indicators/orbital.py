"""
Orbital mechanics anomaly indicators for aNEOS.

This module implements indicators that detect anomalies in orbital
mechanics parameters that might suggest artificial influence.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from .base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    NumericRangeIndicator, StatisticalIndicator
)
from ...data.models import NEOData, OrbitalElements

logger = logging.getLogger(__name__)

class EccentricityIndicator(NumericRangeIndicator):
    """Detects anomalous orbital eccentricity values."""
    
    def __init__(self, config: IndicatorConfig):
        # Normal NEO eccentricity range: 0.0 to 0.8
        # Extreme threshold: > 0.95 (highly unusual)
        super().__init__(
            name="eccentricity",
            config=config,
            normal_range=(0.0, 0.8),
            extreme_threshold=0.95
        )
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate eccentricity anomaly."""
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True}
            )
        
        eccentricity = neo_data.orbital_elements.eccentricity
        
        if eccentricity is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_eccentricity_data': True}
            )
        
        # Calculate anomaly score
        score, factors = self.calculate_range_anomaly(eccentricity, "eccentricity")
        
        # Special cases for artificial signatures
        confidence = 1.0
        metadata = {'eccentricity_value': eccentricity}
        
        # Perfect circular orbits are highly suspicious (artificial satellites)
        if abs(eccentricity) < 0.001:
            score = max(score, 0.8)
            factors.append("Near-perfect circular orbit (highly artificial)")
            confidence = 0.95
        
        # Very high eccentricity suggests possible artificial adjustment
        elif eccentricity > 0.9:
            score = max(score, 0.9)
            factors.append("Extremely high eccentricity (possible artificial)")
            confidence = 0.9
        
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def get_description(self) -> str:
        return "Detects anomalous orbital eccentricity that may indicate artificial influence"

class InclinationIndicator(NumericRangeIndicator):
    """Detects anomalous orbital inclination values."""
    
    def __init__(self, config: IndicatorConfig):
        # Normal NEO inclination range: 0° to 45°
        # Extreme threshold: > 90° (retrograde or polar orbits)
        super().__init__(
            name="inclination",
            config=config,
            normal_range=(0.0, 45.0),
            extreme_threshold=90.0
        )
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate inclination anomaly."""
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True}
            )
        
        inclination = neo_data.orbital_elements.inclination
        
        if inclination is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_inclination_data': True}
            )
        
        # Calculate anomaly score
        score, factors = self.calculate_range_anomaly(inclination, "inclination")
        
        confidence = 1.0
        metadata = {'inclination_value': inclination}
        
        # Special cases for artificial signatures
        # Perfect equatorial orbits (0°) are suspicious
        if abs(inclination) < 0.1:
            score = max(score, 0.7)
            factors.append("Near-perfect equatorial orbit (artificial signature)")
            confidence = 0.9
        
        # Polar orbits (90°) are unusual for natural NEOs
        elif abs(inclination - 90.0) < 5.0:
            score = max(score, 0.8)
            factors.append("Near-polar orbit (unusual for natural NEOs)")
            confidence = 0.85
        
        # Retrograde orbits (> 90°) are extremely rare naturally
        elif inclination > 90.0:
            score = max(score, 0.95)
            factors.append("Retrograde orbit (extremely rare naturally)")
            confidence = 0.95
        
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def get_description(self) -> str:
        return "Detects anomalous orbital inclination suggesting artificial control"

class SemiMajorAxisIndicator(StatisticalIndicator):
    """Detects anomalous semi-major axis values."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="semi_major_axis", config=config)
        
        # Typical NEO semi-major axis range (AU)
        self.normal_range = (0.8, 3.0)  # Earth-crossing to Mars-crossing
        self.suspicious_values = [1.0, 1.5, 2.0]  # Round numbers suggesting artificial placement
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate semi-major axis anomaly."""
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True}
            )
        
        semi_major_axis = neo_data.orbital_elements.semi_major_axis
        
        if semi_major_axis is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_semi_major_axis_data': True}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        metadata = {'semi_major_axis_value': semi_major_axis}
        
        # Check if outside normal range
        if not (self.normal_range[0] <= semi_major_axis <= self.normal_range[1]):
            if semi_major_axis < self.normal_range[0]:
                score += 0.3
                factors.append(f"Semi-major axis below typical range: {semi_major_axis:.3f} AU")
            else:
                score += 0.4
                factors.append(f"Semi-major axis above typical range: {semi_major_axis:.3f} AU")
        
        # Check for suspiciously round numbers (suggesting artificial placement)
        for suspicious_value in self.suspicious_values:
            if abs(semi_major_axis - suspicious_value) < 0.05:  # Within 0.05 AU
                score += 0.5
                factors.append(f"Semi-major axis suspiciously close to round number: {suspicious_value} AU")
                confidence = 0.8
                break
        
        # Perfect 1 AU orbit (Earth-like) is highly suspicious
        if abs(semi_major_axis - 1.0) < 0.01:
            score = max(score, 0.9)
            factors.append("Semi-major axis very close to 1 AU (Earth-like, highly artificial)")
            confidence = 0.95
        
        # Add to statistical history
        self.add_data_point(semi_major_axis)
        
        # Check if statistical outlier
        if len(self._data_history) > 10:
            z_score = self.calculate_z_score(semi_major_axis)
            if z_score > 2.0:
                score += min(z_score / 10.0, 0.3)  # Add up to 0.3 for statistical outliers
                factors.append(f"Statistical outlier (z-score: {z_score:.2f})")
        
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
    
    def get_description(self) -> str:
        return "Detects anomalous semi-major axis values suggesting artificial orbital placement"

class OrbitalResonanceIndicator(AnomalyIndicator):
    """Detects orbital resonances that might indicate artificial control."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="orbital_resonance", config=config)
        
        # Known resonances with Earth (periods in years)
        self.earth_resonances = {
            "1:1": 1.0,      # Same period as Earth
            "2:1": 1.587,    # 2:1 resonance
            "3:2": 1.31,     # 3:2 resonance  
            "4:3": 1.2,      # 4:3 resonance
            "1:2": 0.794,    # 1:2 resonance
            "2:3": 0.87      # 2:3 resonance
        }
        
        self.resonance_tolerance = 0.05  # 5% tolerance for resonance detection
    
    def calculate_orbital_period(self, semi_major_axis: float) -> float:
        """Calculate orbital period using Kepler's third law (in years)."""
        # P² = a³ (with P in years and a in AU)
        return semi_major_axis ** 1.5
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate orbital resonance anomaly."""
        if not neo_data.orbital_elements or neo_data.orbital_elements.semi_major_axis is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_data': True}
            )
        
        semi_major_axis = neo_data.orbital_elements.semi_major_axis
        period = self.calculate_orbital_period(semi_major_axis)
        
        score = 0.0
        factors = []
        confidence = 1.0
        metadata = {
            'orbital_period_years': period,
            'semi_major_axis': semi_major_axis
        }
        
        # Check for resonances with Earth
        for resonance_name, resonance_period in self.earth_resonances.items():
            period_diff = abs(period - resonance_period) / resonance_period
            
            if period_diff < self.resonance_tolerance:
                # Found a resonance
                resonance_score = 1.0 - (period_diff / self.resonance_tolerance)
                
                if resonance_name == "1:1":
                    # 1:1 resonance with Earth is extremely suspicious
                    score = max(score, 0.95)
                    factors.append(f"1:1 orbital resonance with Earth (period: {period:.3f} years)")
                    confidence = 0.98
                elif resonance_name in ["2:1", "1:2"]:
                    # Simple integer resonances are suspicious
                    score = max(score, 0.8)
                    factors.append(f"{resonance_name} orbital resonance with Earth (period: {period:.3f} years)")
                    confidence = 0.9
                else:
                    # Other resonances are moderately suspicious
                    score = max(score, 0.6)
                    factors.append(f"{resonance_name} orbital resonance with Earth (period: {period:.3f} years)")
                    confidence = 0.8
                
                metadata[f'resonance_{resonance_name}'] = {
                    'detected': True,
                    'period_difference': period_diff,
                    'strength': resonance_score
                }
        
        # Check for perfect round-number periods (artificial signatures)
        round_periods = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for round_period in round_periods:
            if abs(period - round_period) < 0.01:  # Within 0.01 years
                score = max(score, 0.7)
                factors.append(f"Period suspiciously close to round number: {round_period} years")
                confidence = 0.85
        
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def get_description(self) -> str:
        return "Detects orbital resonances with Earth that may indicate artificial control"

class OrbitalStabilityIndicator(AnomalyIndicator):
    """Detects unnaturally stable orbits that might indicate artificial maintenance."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="orbital_stability", config=config)
    
    def calculate_hill_sphere_ratio(self, semi_major_axis: float, eccentricity: float) -> float:
        """Calculate ratio of perihelion distance to Earth's Hill sphere."""
        if semi_major_axis is None or eccentricity is None:
            return 0.0
        
        # Earth's Hill sphere radius ~ 0.01 AU
        earth_hill_sphere = 0.01
        
        # Perihelion distance
        perihelion = semi_major_axis * (1 - eccentricity)
        
        return perihelion / earth_hill_sphere
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate orbital stability anomaly."""
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True}
            )
        
        elements = neo_data.orbital_elements
        
        if elements.semi_major_axis is None or elements.eccentricity is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_orbital_data': True}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        metadata = {
            'semi_major_axis': elements.semi_major_axis,
            'eccentricity': elements.eccentricity
        }
        
        # Check for suspiciously stable Earth-crossing orbits
        perihelion = elements.semi_major_axis * (1 - elements.eccentricity)
        aphelion = elements.semi_major_axis * (1 + elements.eccentricity)
        
        # Earth's orbit is at 1 AU
        if perihelion < 1.3 and aphelion > 0.7:  # Earth-crossing
            # Calculate Minimum Orbit Intersection Distance (MOID) approximation
            earth_distance_factor = min(abs(perihelion - 1.0), abs(aphelion - 1.0))
            
            if earth_distance_factor < 0.05:  # Very close to Earth's orbit
                score += 0.6
                factors.append("Orbit extremely close to Earth's orbit (artificially maintained)")
                confidence = 0.9
            
            # Check for circular Earth-crossing orbit (highly artificial)
            if elements.eccentricity < 0.1 and 0.9 < elements.semi_major_axis < 1.1:
                score = max(score, 0.95)
                factors.append("Nearly circular orbit close to Earth's orbit (artificial satellite signature)")
                confidence = 0.98
        
        # Check for Lagrange point orbits (L4, L5 positions)
        if elements.inclination is not None:
            # L4/L5 points are at 60° ahead/behind Earth
            if (abs(elements.semi_major_axis - 1.0) < 0.05 and  # Same distance as Earth
                elements.eccentricity < 0.1 and               # Nearly circular
                elements.inclination < 10.0):                 # Low inclination
                
                score = max(score, 0.8)
                factors.append("Potential Lagrange point orbit (artificial positioning)")
                confidence = 0.85
        
        # Check for unrealistically low eccentricity for the distance
        expected_min_eccentricity = max(0.01, (elements.semi_major_axis - 1.0) * 0.1)
        if elements.eccentricity < expected_min_eccentricity:
            score += 0.3
            factors.append(f"Unusually low eccentricity for distance (expected > {expected_min_eccentricity:.3f})")
        
        weighted_score = self.calculate_weighted_score(score, confidence)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata=metadata,
            contributing_factors=factors
        )
    
    def get_description(self) -> str:
        return "Detects unnaturally stable orbits that may indicate artificial maintenance"