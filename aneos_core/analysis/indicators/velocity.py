"""
Velocity analysis anomaly indicators for aNEOS.

This module implements indicators that detect anomalies in velocity
patterns that might suggest artificial propulsion or control.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

from .base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    StatisticalIndicator, TemporalIndicator
)
from ...data.models import NEOData, CloseApproach

logger = logging.getLogger(__name__)

class VelocityShiftIndicator(StatisticalIndicator):
    """Detects anomalous shifts in relative velocity between observations."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="velocity_shifts", config=config)
        self.normal_velocity_range = (5.0, 50.0)  # km/s for typical NEOs
        self.significant_shift_threshold = 5.0    # km/s change considered significant
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate velocity shift anomaly."""
        if len(neo_data.close_approaches) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_observations': len(neo_data.close_approaches)}
            )
        
        # Extract velocity data
        velocities = []
        valid_approaches = []
        
        for approach in neo_data.close_approaches:
            if approach.relative_velocity_km_s is not None:
                velocities.append(approach.relative_velocity_km_s)
                valid_approaches.append(approach)
                self.add_data_point(approach.relative_velocity_km_s)
        
        if len(velocities) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_velocity_data': len(velocities)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        metadata = {
            'velocity_count': len(velocities),
            'velocity_range': (min(velocities), max(velocities)),
            'mean_velocity': np.mean(velocities)
        }
        
        # Calculate velocity shifts between consecutive observations
        velocity_shifts = []
        for i in range(1, len(velocities)):
            shift = abs(velocities[i] - velocities[i-1])
            velocity_shifts.append(shift)
        
        if velocity_shifts:
            max_shift = max(velocity_shifts)
            mean_shift = np.mean(velocity_shifts)
            
            metadata['max_velocity_shift'] = max_shift
            metadata['mean_velocity_shift'] = mean_shift
            metadata['velocity_shifts'] = velocity_shifts
            
            # Significant sudden velocity changes are suspicious
            if max_shift > self.significant_shift_threshold:
                shift_score = min(max_shift / 20.0, 1.0)  # Normalize by 20 km/s max expected
                score += shift_score
                factors.append(f"Large velocity shift detected: {max_shift:.2f} km/s")
                
                # Very large shifts are extremely suspicious
                if max_shift > 15.0:
                    score = max(score, 0.8)
                    factors.append("Extremely large velocity shift (possible propulsion)")
                    confidence = 0.9
            
            # Consistent velocity shifts might indicate periodic propulsion
            if len(velocity_shifts) >= 3:
                shift_variance = np.var(velocity_shifts)
                shift_mean = np.mean(velocity_shifts)
                
                if shift_mean > 2.0 and shift_variance < 1.0:  # Consistent moderate shifts
                    score += 0.4
                    factors.append("Consistent velocity shifts (possible periodic propulsion)")
                    confidence = 0.8
        
        # Check for velocities outside normal range
        abnormal_velocities = [v for v in velocities if not (self.normal_velocity_range[0] <= v <= self.normal_velocity_range[1])]
        
        if abnormal_velocities:
            abnormal_ratio = len(abnormal_velocities) / len(velocities)
            score += abnormal_ratio * 0.5
            factors.append(f"{len(abnormal_velocities)} velocities outside normal range")
        
        # Check for impossibly precise velocities (artificial signatures)
        rounded_velocities = [v for v in velocities if abs(v - round(v)) < 0.01]
        if len(rounded_velocities) > len(velocities) * 0.5:  # More than half are suspiciously round
            score += 0.3
            factors.append("Suspiciously precise velocity values")
            confidence = 0.7
        
        # Statistical analysis if we have enough data
        if len(self._data_history) > 10:
            for velocity in velocities:
                z_score = self.calculate_z_score(velocity)
                if z_score > 3.0:  # Very unusual velocity
                    score += min(z_score / 20.0, 0.2)
                    factors.append(f"Statistical velocity outlier: {velocity:.2f} km/s (z={z_score:.2f})")
        
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
        return "Detects anomalous velocity shifts that may indicate artificial propulsion"

class AccelerationIndicator(TemporalIndicator):
    """Detects non-gravitational acceleration that might indicate propulsion."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="acceleration_anomalies", config=config)
        self.acceleration_threshold = 0.0005  # km/s² - threshold for significant acceleration
        self.max_natural_acceleration = 0.001  # km/s² - maximum expected from natural forces
    
    def calculate_acceleration(self, approach1: CloseApproach, approach2: CloseApproach) -> Optional[float]:
        """Calculate acceleration between two close approaches."""
        if (not approach1.relative_velocity_km_s or not approach2.relative_velocity_km_s or
            not approach1.close_approach_date or not approach2.close_approach_date):
            return None
        
        # Time difference in seconds
        time_diff = (approach2.close_approach_date - approach1.close_approach_date).total_seconds()
        
        if time_diff <= 0:
            return None
        
        # Velocity difference in km/s
        velocity_diff = approach2.relative_velocity_km_s - approach1.relative_velocity_km_s
        
        # Acceleration in km/s²
        acceleration = abs(velocity_diff) / time_diff
        
        return acceleration
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate acceleration anomaly."""
        if len(neo_data.close_approaches) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_observations': len(neo_data.close_approaches)}
            )
        
        # Sort approaches by date
        sorted_approaches = sorted(
            [a for a in neo_data.close_approaches if a.close_approach_date and a.relative_velocity_km_s],
            key=lambda x: x.close_approach_date
        )
        
        if len(sorted_approaches) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_valid_data': len(sorted_approaches)}
            )
        
        accelerations = []
        acceleration_events = []
        
        # Calculate accelerations between consecutive observations
        for i in range(1, len(sorted_approaches)):
            acceleration = self.calculate_acceleration(sorted_approaches[i-1], sorted_approaches[i])
            
            if acceleration is not None:
                accelerations.append(acceleration)
                acceleration_events.append({
                    'time1': sorted_approaches[i-1].close_approach_date,
                    'time2': sorted_approaches[i].close_approach_date,
                    'velocity1': sorted_approaches[i-1].relative_velocity_km_s,
                    'velocity2': sorted_approaches[i].relative_velocity_km_s,
                    'acceleration': acceleration
                })
        
        if not accelerations:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_calculable_accelerations': True}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        metadata = {
            'acceleration_count': len(accelerations),
            'max_acceleration': max(accelerations),
            'mean_acceleration': np.mean(accelerations),
            'acceleration_events': acceleration_events
        }
        
        # Check for significant accelerations
        significant_accelerations = [a for a in accelerations if a > self.acceleration_threshold]
        
        if significant_accelerations:
            max_acceleration = max(significant_accelerations)
            
            # Score based on magnitude of acceleration
            if max_acceleration > self.max_natural_acceleration:
                # Acceleration beyond natural forces
                excess_acceleration = max_acceleration - self.max_natural_acceleration
                acceleration_score = min(excess_acceleration / self.max_natural_acceleration, 1.0)
                score += acceleration_score
                
                factors.append(f"Non-gravitational acceleration detected: {max_acceleration:.6f} km/s²")
                
                # Very high accelerations are extremely suspicious
                if max_acceleration > 0.005:  # 10x natural threshold
                    score = max(score, 0.9)
                    factors.append("Extremely high acceleration (strong artificial signature)")
                    confidence = 0.95
                elif max_acceleration > 0.002:  # 4x natural threshold
                    score = max(score, 0.7)
                    factors.append("High acceleration (possible artificial propulsion)")
                    confidence = 0.85
            
            # Multiple significant acceleration events
            if len(significant_accelerations) > 1:
                score += min(len(significant_accelerations) * 0.1, 0.3)
                factors.append(f"Multiple acceleration events detected ({len(significant_accelerations)})")
        
        # Check for periodic acceleration patterns (course corrections)
        if len(accelerations) >= 3:
            acceleration_intervals = []
            significant_events = [event for event in acceleration_events 
                                if event['acceleration'] > self.acceleration_threshold]
            
            if len(significant_events) >= 2:
                for i in range(1, len(significant_events)):
                    interval = (significant_events[i]['time2'] - significant_events[i-1]['time2']).days
                    acceleration_intervals.append(interval)
                
                if acceleration_intervals:
                    # Check for regular intervals (suggesting planned course corrections)
                    mean_interval = np.mean(acceleration_intervals)
                    interval_variance = np.var(acceleration_intervals)
                    
                    if interval_variance < (mean_interval * 0.1) ** 2:  # Low variance = regular
                        score += 0.4
                        factors.append(f"Regular acceleration intervals (mean: {mean_interval:.1f} days)")
                        confidence = 0.8
        
        # Check for instantaneous velocity changes (impossible naturally)
        for event in acceleration_events:
            time_diff_hours = (event['time2'] - event['time1']).total_seconds() / 3600
            
            if time_diff_hours < 24 and event['acceleration'] > self.acceleration_threshold:
                # Very quick acceleration
                score += 0.3
                factors.append(f"Rapid acceleration over {time_diff_hours:.1f} hours")
                confidence = 0.9
        
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
        return "Detects non-gravitational accelerations that may indicate artificial propulsion"

class VelocityConsistencyIndicator(StatisticalIndicator):
    """Detects suspiciously consistent velocities that might indicate artificial control."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="velocity_consistency", config=config)
        self.consistency_threshold = 0.05  # 5% variation considered too consistent
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate velocity consistency anomaly."""
        if len(neo_data.close_approaches) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_observations': len(neo_data.close_approaches)}
            )
        
        # Extract velocity data
        velocities = [approach.relative_velocity_km_s for approach in neo_data.close_approaches 
                     if approach.relative_velocity_km_s is not None]
        
        if len(velocities) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_velocity_data': len(velocities)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        velocities_array = np.array(velocities)
        mean_velocity = np.mean(velocities_array)
        std_velocity = np.std(velocities_array)
        coefficient_of_variation = std_velocity / mean_velocity if mean_velocity > 0 else 0
        
        metadata = {
            'velocity_count': len(velocities),
            'mean_velocity': mean_velocity,
            'std_velocity': std_velocity,
            'coefficient_of_variation': coefficient_of_variation,
            'velocity_range': (float(np.min(velocities_array)), float(np.max(velocities_array)))
        }
        
        # Check for suspiciously low variation
        if coefficient_of_variation < self.consistency_threshold:
            consistency_score = 1.0 - (coefficient_of_variation / self.consistency_threshold)
            score += consistency_score * 0.8
            
            factors.append(f"Suspiciously consistent velocities (CV: {coefficient_of_variation:.4f})")
            confidence = 0.8
            
            # Perfect consistency is extremely suspicious
            if coefficient_of_variation < 0.001:
                score = max(score, 0.95)
                factors.append("Near-perfect velocity consistency (artificial control)")
                confidence = 0.95
        
        # Check for identical velocities (impossible naturally)
        unique_velocities = len(set(velocities))
        if unique_velocities < len(velocities) * 0.7:  # Less than 70% unique
            score += 0.5
            factors.append(f"Many identical velocity values ({unique_velocities}/{len(velocities)} unique)")
        
        # Check for velocities that are suspiciously round numbers
        round_velocities = [v for v in velocities if abs(v - round(v)) < 0.1]
        if len(round_velocities) > len(velocities) * 0.6:  # More than 60% are round
            score += 0.3
            factors.append("Many suspiciously round velocity values")
            confidence = 0.7
        
        # Add data to statistical history
        for v in velocities:
            self.add_data_point(v)
        
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
        return "Detects suspiciously consistent velocities that may indicate artificial control"

class InfinityVelocityIndicator(AnomalyIndicator):
    """Detects anomalous hyperbolic excess velocities (v_infinity)."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="infinity_velocity", config=config)
        self.normal_v_inf_range = (0.5, 20.0)  # km/s typical range for NEOs
        self.zero_v_inf_threshold = 0.1  # km/s - suspiciously low v_infinity
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate infinity velocity anomaly."""
        v_inf_values = [approach.infinity_velocity_km_s for approach in neo_data.close_approaches 
                       if approach.infinity_velocity_km_s is not None]
        
        if not v_inf_values:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_v_inf_data': True}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        v_inf_array = np.array(v_inf_values)
        mean_v_inf = np.mean(v_inf_array)
        min_v_inf = np.min(v_inf_array)
        max_v_inf = np.max(v_inf_array)
        
        metadata = {
            'v_inf_count': len(v_inf_values),
            'mean_v_inf': mean_v_inf,
            'min_v_inf': min_v_inf,
            'max_v_inf': max_v_inf,
            'v_inf_values': v_inf_values
        }
        
        # Check for suspiciously low v_infinity (bound orbit signatures)
        low_v_inf_count = sum(1 for v in v_inf_values if v < self.zero_v_inf_threshold)
        if low_v_inf_count > 0:
            low_ratio = low_v_inf_count / len(v_inf_values)
            score += low_ratio * 0.8
            factors.append(f"{low_v_inf_count} observations with near-zero v_infinity")
            
            if min_v_inf < 0.01:  # Essentially zero
                score = max(score, 0.9)
                factors.append("Near-zero v_infinity (artificial bound orbit)")
                confidence = 0.9
        
        # Check for values outside normal range
        abnormal_v_inf = [v for v in v_inf_values 
                         if not (self.normal_v_inf_range[0] <= v <= self.normal_v_inf_range[1])]
        
        if abnormal_v_inf:
            abnormal_ratio = len(abnormal_v_inf) / len(v_inf_values)
            score += abnormal_ratio * 0.4
            factors.append(f"{len(abnormal_v_inf)} v_infinity values outside normal range")
        
        # Check for impossible negative v_infinity
        negative_v_inf = [v for v in v_inf_values if v < 0]
        if negative_v_inf:
            score = max(score, 0.8)
            factors.append("Negative v_infinity values (physically impossible)")
            confidence = 0.95
        
        # Check for suspiciously consistent v_infinity
        if len(v_inf_values) > 2:
            std_v_inf = np.std(v_inf_array)
            cv = std_v_inf / mean_v_inf if mean_v_inf > 0 else 0
            
            if cv < 0.05:  # Very low variation
                score += 0.4
                factors.append("Suspiciously consistent v_infinity values")
                confidence = 0.8
        
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
        return "Detects anomalous hyperbolic excess velocities suggesting artificial orbits"