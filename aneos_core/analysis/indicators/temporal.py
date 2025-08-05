"""
Temporal pattern anomaly indicators for aNEOS.

This module implements indicators that detect anomalies in temporal
patterns that might suggest artificial scheduling or control.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from .base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    TemporalIndicator, StatisticalIndicator
)
from ...data.models import NEOData, CloseApproach

logger = logging.getLogger(__name__)

class CloseApproachRegularityIndicator(TemporalIndicator):
    """Detects suspiciously regular patterns in close approach timing."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="approach_regularity", config=config)
        self.regularity_threshold = 0.8  # Threshold for detecting artificial regularity
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate close approach regularity anomaly."""
        if len(neo_data.close_approaches) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_approaches': len(neo_data.close_approaches)}
            )
        
        # Extract dates from close approaches
        dates = [approach.close_approach_date for approach in neo_data.close_approaches 
                if approach.close_approach_date is not None]
        
        if len(dates) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_valid_dates': len(dates)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Calculate time intervals between approaches
        intervals = self.calculate_time_intervals(dates)
        periodicity = self.analyze_periodicity(intervals)
        
        metadata = {
            'approach_count': len(dates),
            'time_intervals_days': intervals,
            'periodicity_analysis': periodicity,
            'date_range': (min(dates).isoformat(), max(dates).isoformat())
        }
        
        # Check for suspicious regularity
        regularity = periodicity['regularity']
        if regularity > self.regularity_threshold:
            regularity_score = (regularity - self.regularity_threshold) / (1.0 - self.regularity_threshold)
            score += regularity_score * 0.8
            
            factors.append(f"Highly regular close approach intervals (regularity: {regularity:.3f})")
            confidence = 0.9
            
            # Perfect regularity is extremely suspicious
            if regularity > 0.95:
                score = max(score, 0.9)
                factors.append("Near-perfect regularity in close approaches (artificial scheduling)")
                confidence = 0.95
        
        # Check for specific suspicious patterns
        if intervals:
            mean_interval = np.mean(intervals)
            
            # Check for round-number intervals (suggesting artificial scheduling)
            round_intervals = [30, 60, 90, 120, 180, 365, 730]  # Common scheduling intervals
            for round_interval in round_intervals:
                if abs(mean_interval - round_interval) < 5:  # Within 5 days
                    score += 0.4
                    factors.append(f"Mean interval close to round number: {round_interval} days")
                    confidence = 0.8
                    break
            
            # Check for intervals that are exact multiples
            if len(intervals) >= 2:
                # Check if intervals are multiples of each other
                sorted_intervals = sorted(intervals)
                for i in range(len(sorted_intervals)-1):
                    for j in range(i+1, len(sorted_intervals)):
                        ratio = sorted_intervals[j] / sorted_intervals[i]
                        if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                            score += 0.3
                            factors.append(f"Intervals show integer ratio pattern: {ratio:.1f}")
                            confidence = 0.85
                            break
        
        # Check for clustering in specific time periods
        seasonal_score = self._analyze_seasonal_clustering(dates)
        if seasonal_score > 0:
            score += seasonal_score
            factors.append("Suspicious seasonal clustering of close approaches")
        
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
    
    def _analyze_seasonal_clustering(self, dates: List[datetime]) -> float:
        """Analyze if dates cluster around specific times of year."""
        if len(dates) < 4:
            return 0.0
        
        # Convert to day of year
        day_of_year = [date.timetuple().tm_yday for date in dates]
        
        # Check for clustering using circular statistics
        # Convert to radians (day of year as angle)
        angles = [2 * np.pi * day / 365.25 for day in day_of_year]
        
        # Calculate circular mean and concentration
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        
        # Resultant length (measure of concentration)
        resultant_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        
        # High concentration suggests seasonal clustering
        if resultant_length > 0.7:  # Threshold for significant clustering
            return min(resultant_length - 0.5, 0.4)  # Score up to 0.4
        
        return 0.0
    
    def get_description(self) -> str:
        return "Detects suspiciously regular patterns in close approach timing"

class ObservationGapIndicator(StatisticalIndicator):
    """Detects unusual patterns in observation gaps."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="observation_gaps", config=config)
        self.normal_gap_range = (1, 365)  # Days between observations
        self.suspicious_gap_threshold = 1000  # Days - very long gaps
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate observation gap anomaly."""
        # Use close approach dates as proxy for observation dates
        dates = [approach.close_approach_date for approach in neo_data.close_approaches 
                if approach.close_approach_date is not None]
        
        if len(dates) < 2:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_observations': len(dates)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Calculate gaps between observations
        gaps = self.calculate_time_intervals(dates)
        
        if not gaps:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_calculable_gaps': True}
            )
        
        metadata = {
            'observation_count': len(dates),
            'gaps_days': gaps,
            'min_gap': min(gaps),
            'max_gap': max(gaps),
            'mean_gap': np.mean(gaps),
            'gap_std': np.std(gaps)
        }
        
        # Add gaps to statistical history
        for gap in gaps:
            self.add_data_point(gap)
        
        # Check for suspiciously long gaps
        long_gaps = [gap for gap in gaps if gap > self.suspicious_gap_threshold]
        if long_gaps:
            long_gap_score = min(len(long_gaps) / len(gaps), 0.5)
            score += long_gap_score
            factors.append(f"{len(long_gaps)} suspiciously long observation gaps (>{self.suspicious_gap_threshold} days)")
        
        # Check for gaps outside normal range
        abnormal_gaps = [gap for gap in gaps if not (self.normal_gap_range[0] <= gap <= self.normal_gap_range[1])]
        if abnormal_gaps:
            abnormal_ratio = len(abnormal_gaps) / len(gaps)
            score += abnormal_ratio * 0.3
            factors.append(f"{len(abnormal_gaps)} gaps outside normal range")
        
        # Check for suspiciously consistent gaps
        if len(gaps) >= 3:
            gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
            
            if gap_cv < 0.1:  # Very low variation
                score += 0.6
                factors.append("Suspiciously consistent observation gaps")
                confidence = 0.8
            
            # Check for identical gaps (impossible naturally)
            unique_gaps = len(set(int(gap) for gap in gaps))
            if unique_gaps < len(gaps) * 0.7:  # Less than 70% unique
                score += 0.4
                factors.append("Many identical observation gaps")
        
        # Statistical analysis if we have enough historical data
        if len(self._data_history) > 20:
            outlier_count = 0
            for gap in gaps:
                z_score = self.calculate_z_score(gap)
                if z_score > 3.0:  # Statistical outlier
                    outlier_count += 1
            
            if outlier_count > 0:
                outlier_score = min(outlier_count / len(gaps), 0.3)
                score += outlier_score
                factors.append(f"{outlier_count} statistically anomalous gaps")
        
        # Check for perfect scheduling patterns
        if len(gaps) >= 4:
            # Check if gaps follow arithmetic progression
            diffs = [gaps[i+1] - gaps[i] for i in range(len(gaps)-1)]
            if all(abs(diff - diffs[0]) < 1 for diff in diffs):  # Nearly constant differences
                score += 0.5
                factors.append("Gaps follow arithmetic progression (scheduled pattern)")
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
        return "Detects unusual patterns in observation timing gaps"

class PeriodicityIndicator(TemporalIndicator):
    """Detects artificial periodicity in NEO behavior."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="periodicity", config=config)
        self.known_periods = {
            "daily": 1.0,
            "weekly": 7.0,
            "monthly": 30.0,
            "quarterly": 90.0,
            "semi_annual": 180.0,
            "annual": 365.0,
            "biennial": 730.0
        }
        self.tolerance = 0.05  # 5% tolerance for period matching
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate periodicity anomaly."""
        dates = [approach.close_approach_date for approach in neo_data.close_approaches 
                if approach.close_approach_date is not None]
        
        if len(dates) < 4:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_data': len(dates)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Calculate intervals and analyze periodicity
        intervals = self.calculate_time_intervals(dates)
        periodicity = self.analyze_periodicity(intervals)
        
        metadata = {
            'observation_count': len(dates),
            'periodicity_analysis': periodicity,
            'detected_periods': []
        }
        
        # Check for known artificial periods
        mean_interval = periodicity['period_estimate']
        for period_name, period_days in self.known_periods.items():
            relative_error = abs(mean_interval - period_days) / period_days
            
            if relative_error < self.tolerance:
                period_score = 1.0 - (relative_error / self.tolerance)
                
                if period_name in ["daily", "weekly", "monthly"]:
                    # Very suspicious periods
                    score = max(score, 0.8)
                    factors.append(f"Detected {period_name} periodicity (period: {mean_interval:.1f} days)")
                    confidence = 0.95
                elif period_name in ["quarterly", "semi_annual", "annual"]:
                    # Moderately suspicious periods
                    score = max(score, 0.6)
                    factors.append(f"Detected {period_name} periodicity (period: {mean_interval:.1f} days)")
                    confidence = 0.85
                else:
                    # Less suspicious but still notable
                    score = max(score, 0.4)
                    factors.append(f"Detected {period_name} periodicity (period: {mean_interval:.1f} days)")
                    confidence = 0.75
                
                metadata['detected_periods'].append({
                    'name': period_name,
                    'expected_days': period_days,
                    'observed_days': mean_interval,
                    'relative_error': relative_error,
                    'strength': period_score
                })
        
        # Check for other suspicious patterns
        if periodicity['regularity'] > 0.9 and not metadata['detected_periods']:
            # High regularity but not matching known periods
            score += 0.5
            factors.append(f"Unknown but highly regular periodicity (period: {mean_interval:.1f} days)")
            confidence = 0.8
        
        # Check for harmonic relationships (multiples/fractions of basic periods)
        if not metadata['detected_periods'] and mean_interval > 1:
            for period_name, period_days in self.known_periods.items():
                # Check for integer multiples
                for multiplier in [2, 3, 4, 5, 0.5, 0.33, 0.25, 0.2]:
                    expected = period_days * multiplier
                    if abs(mean_interval - expected) / expected < self.tolerance:
                        score += 0.3
                        factors.append(f"Harmonic of {period_name} period ({multiplier}x)")
                        confidence = 0.7
                        break
        
        # Analyze sub-patterns within the data
        if len(intervals) >= 6:
            # Look for repeated sub-sequences
            subsequence_score = self._analyze_subsequences(intervals)
            if subsequence_score > 0:
                score += subsequence_score
                factors.append("Detected repeating subsequences in timing")
        
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
    
    def _analyze_subsequences(self, intervals: List[float]) -> float:
        """Analyze for repeating subsequences in interval patterns."""
        if len(intervals) < 6:
            return 0.0
        
        # Look for repeating patterns of length 2-4
        for pattern_length in range(2, min(5, len(intervals) // 2)):
            patterns = {}
            
            # Extract all possible patterns of this length
            for i in range(len(intervals) - pattern_length + 1):
                pattern = tuple(round(intervals[i+j]) for j in range(pattern_length))
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            # Check if any pattern repeats significantly
            max_repeats = max(patterns.values()) if patterns else 0
            if max_repeats >= 3:  # Pattern repeats at least 3 times
                repeat_ratio = max_repeats / (len(intervals) - pattern_length + 1)
                if repeat_ratio > 0.5:  # More than half the positions show this pattern
                    return min(repeat_ratio * 0.4, 0.4)
        
        return 0.0
    
    def get_description(self) -> str:
        return "Detects artificial periodicity in NEO timing patterns"

class TemporalInertiaIndicator(AnomalyIndicator):
    """Detects temporal inertia - resistance to expected orbital changes."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(name="temporal_inertia", config=config)
        self.inertia_threshold = 100.0  # Threshold for detecting temporal inertia
    
    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        """Evaluate temporal inertia anomaly."""
        if len(neo_data.close_approaches) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_approaches': len(neo_data.close_approaches)}
            )
        
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Sort approaches by date
        sorted_approaches = sorted(
            [a for a in neo_data.close_approaches if a.close_approach_date and a.distance_au],
            key=lambda x: x.close_approach_date
        )
        
        if len(sorted_approaches) < 3:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'insufficient_valid_data': len(sorted_approaches)}
            )
        
        # Calculate expected vs actual changes in approach distance
        distance_changes = []
        time_spans = []
        
        for i in range(1, len(sorted_approaches)):
            prev_approach = sorted_approaches[i-1]
            curr_approach = sorted_approaches[i]
            
            distance_change = abs(curr_approach.distance_au - prev_approach.distance_au)
            time_span = (curr_approach.close_approach_date - prev_approach.close_approach_date).days
            
            if time_span > 0:
                distance_changes.append(distance_change)
                time_spans.append(time_span)
        
        if not distance_changes:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_calculable_changes': True}
            )
        
        # Calculate temporal inertia metric
        mean_distance_change = np.mean(distance_changes)
        mean_time_span = np.mean(time_spans)
        
        # Expected change should increase with time
        expected_change_rate = 0.001  # AU per day (rough estimate)
        expected_total_change = expected_change_rate * mean_time_span
        
        metadata = {
            'approach_count': len(sorted_approaches),
            'mean_distance_change_au': mean_distance_change,
            'mean_time_span_days': mean_time_span,
            'expected_change_au': expected_total_change,
            'distance_changes': distance_changes,
            'time_spans': time_spans
        }
        
        # Check for suspiciously low change rates
        if mean_distance_change < expected_total_change * 0.1:  # Less than 10% of expected
            inertia_score = 1.0 - (mean_distance_change / (expected_total_change * 0.1))
            score += inertia_score * 0.8
            
            factors.append(f"Extremely low orbital change rate: {mean_distance_change:.6f} AU over {mean_time_span:.1f} days")
            confidence = 0.9
            
            # Perfect stability is impossible naturally
            if mean_distance_change < expected_total_change * 0.01:  # Less than 1% of expected
                score = max(score, 0.95)
                factors.append("Near-perfect orbital stability (artificial maintenance)")
                confidence = 0.98
        
        # Check for identical distances (impossible naturally)
        unique_distances = len(set(round(a.distance_au, 6) for a in sorted_approaches))
        if unique_distances < len(sorted_approaches) * 0.8:  # Less than 80% unique
            score += 0.5
            factors.append("Many identical approach distances")
            confidence = 0.85
        
        # Check for linear progression (artificial pattern)
        if len(distance_changes) >= 3:
            distance_array = np.array([a.distance_au for a in sorted_approaches])
            
            # Fit linear trend
            time_array = np.array([(a.close_approach_date - sorted_approaches[0].close_approach_date).days 
                                  for a in sorted_approaches])
            
            if len(time_array) > 1 and np.std(time_array) > 0:
                correlation = np.corrcoef(time_array, distance_array)[0, 1]
                
                if abs(correlation) > 0.95:  # Very high correlation
                    score += 0.4
                    factors.append(f"Highly linear distance progression (r={correlation:.3f})")
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
        return "Detects temporal inertia - resistance to expected orbital changes"