"""
Scoring and statistical analysis for aNEOS anomaly detection.

This module provides comprehensive scoring calculation and statistical
analysis of anomaly indicator results.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
import math

from .indicators.base import IndicatorResult, AnomalyIndicator
from ..data.models import NEOData
from ..config.settings import WeightConfig, ThresholdConfig

logger = logging.getLogger(__name__)

@dataclass
class AnomalyScore:
    """Complete anomaly score for a NEO."""
    designation: str
    overall_score: float
    confidence: float
    classification: str  # 'natural', 'suspicious', 'highly_suspicious', 'artificial'
    indicator_scores: Dict[str, IndicatorResult] = field(default_factory=dict)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'designation': self.designation,
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'classification': self.classification,
            'indicator_scores': {k: v.to_dict() for k, v in self.indicator_scores.items()},
            'statistical_summary': self.statistical_summary,
            'risk_factors': self.risk_factors,
            'created_at': self.created_at.isoformat(),
            'analysis_version': '2.0.0'
        }

class ScoreCalculator:
    """Calculates composite anomaly scores from indicator results."""
    
    def __init__(self, weights: WeightConfig, thresholds: ThresholdConfig):
        self.weights = weights
        self.thresholds = thresholds
        
        # Classification thresholds
        self.classification_thresholds = {
            'natural': 0.0,
            'suspicious': 0.3,
            'highly_suspicious': 0.6,
            'artificial': 0.8
        }
        
        # Indicator category mappings
        self.indicator_categories = {
            'orbital': ['eccentricity', 'inclination', 'semi_major_axis', 'orbital_resonance', 'orbital_stability'],
            'velocity': ['velocity_shifts', 'acceleration_anomalies', 'velocity_consistency', 'infinity_velocity'],
            'temporal': ['approach_regularity', 'observation_gaps', 'periodicity', 'temporal_inertia'],
            'geographic': ['subpoint_clustering', 'geographic_bias'],
            'physical': ['diameter_anomalies', 'albedo_anomalies', 'spectral_anomalies'],
            'behavioral': ['detection_history', 'observation_history']
        }
        
        logger.info("ScoreCalculator initialized with category-based scoring")
    
    def calculate_score(self, designation: str, indicator_results: Dict[str, IndicatorResult]) -> AnomalyScore:
        """Calculate comprehensive anomaly score."""
        if not indicator_results:
            return AnomalyScore(
                designation=designation,
                overall_score=0.0,
                confidence=0.0,
                classification='natural'
            )
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(indicator_results)
        
        # Calculate overall weighted score
        overall_score = self._calculate_overall_score(category_scores, indicator_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicator_results)
        
        # Determine classification
        classification = self._classify_anomaly(overall_score, confidence)
        
        # Generate statistical summary
        statistical_summary = self._generate_statistical_summary(indicator_results, category_scores)
        
        # Identify primary risk factors
        risk_factors = self._identify_risk_factors(indicator_results, category_scores)
        
        return AnomalyScore(
            designation=designation,
            overall_score=overall_score,
            confidence=confidence,
            classification=classification,
            indicator_scores=indicator_results,
            statistical_summary=statistical_summary,
            risk_factors=risk_factors
        )
    
    def _calculate_category_scores(self, indicator_results: Dict[str, IndicatorResult]) -> Dict[str, float]:
        """Calculate scores for each indicator category."""
        category_scores = {}
        
        for category, indicators in self.indicator_categories.items():
            category_results = []
            
            for indicator_name in indicators:
                if indicator_name in indicator_results:
                    result = indicator_results[indicator_name]
                    if result.confidence > 0.1:  # Only include results with meaningful confidence
                        category_results.append(result)
            
            if category_results:
                # Calculate weighted average for category
                total_weight = sum(r.confidence for r in category_results)
                if total_weight > 0:
                    weighted_score = sum(r.weighted_score * r.confidence for r in category_results) / total_weight
                    category_scores[category] = weighted_score
                else:
                    category_scores[category] = 0.0
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[str, float], 
                               indicator_results: Dict[str, IndicatorResult]) -> float:
        """Calculate overall anomaly score."""
        # Category weights (can be made configurable)
        category_weights = {
            'orbital': self.weights.orbital_mechanics,
            'velocity': (self.weights.velocity_shifts + self.weights.acceleration_anomalies) / 2,
            'temporal': (self.weights.temporal_anomalies + self.weights.close_approach_regularity) / 2,
            'geographic': self.weights.geographic_clustering,
            'physical': self.weights.physical_anomalies,
            'behavioral': (self.weights.observation_history + self.weights.detection_history) / 2
        }
        
        # Calculate weighted category score
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            if category in category_weights:
                weight = category_weights[category]
                weighted_sum += score * weight
                total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply boosting for high-confidence anomalies
        high_confidence_indicators = [
            r for r in indicator_results.values() 
            if r.confidence > 0.8 and r.weighted_score > 0.5
        ]
        
        if len(high_confidence_indicators) >= 2:
            # Multiple high-confidence anomalies boost the score
            boost_factor = 1.0 + (len(high_confidence_indicators) - 1) * 0.1
            base_score *= min(boost_factor, 1.5)  # Cap boost at 50%
        
        # Apply penalty for low data quality
        low_confidence_count = sum(1 for r in indicator_results.values() if r.confidence < 0.3)
        total_indicators = len(indicator_results)
        
        if low_confidence_count > total_indicators * 0.5:  # More than half low confidence
            penalty_factor = 0.8
            base_score *= penalty_factor
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _calculate_confidence(self, indicator_results: Dict[str, IndicatorResult]) -> float:
        """Calculate overall confidence in the anomaly score."""
        if not indicator_results:
            return 0.0
        
        # Base confidence is average of indicator confidences
        confidences = [r.confidence for r in indicator_results.values()]
        base_confidence = np.mean(confidences)
        
        # Boost confidence if multiple indicators agree
        high_score_indicators = [r for r in indicator_results.values() if r.raw_score > 0.5]
        agreement_boost = min(len(high_score_indicators) * 0.1, 0.3)
        
        # Reduce confidence if results are inconsistent
        score_variance = np.var([r.raw_score for r in indicator_results.values()])
        if score_variance > 0.2:  # High variance in scores
            variance_penalty = 0.1
        else:
            variance_penalty = 0.0
        
        final_confidence = base_confidence + agreement_boost - variance_penalty
        return max(0.0, min(final_confidence, 1.0))
    
    def _classify_anomaly(self, overall_score: float, confidence: float) -> str:
        """Classify anomaly based on score and confidence."""
        # Adjust thresholds based on confidence
        confidence_factor = confidence
        
        adjusted_thresholds = {
            'artificial': self.classification_thresholds['artificial'] * (2.0 - confidence_factor),
            'highly_suspicious': self.classification_thresholds['highly_suspicious'] * (2.0 - confidence_factor),
            'suspicious': self.classification_thresholds['suspicious'] * (2.0 - confidence_factor)
        }
        
        if overall_score >= adjusted_thresholds['artificial']:
            return 'artificial'
        elif overall_score >= adjusted_thresholds['highly_suspicious']:
            return 'highly_suspicious'
        elif overall_score >= adjusted_thresholds['suspicious']:
            return 'suspicious'
        else:
            return 'natural'
    
    def _generate_statistical_summary(self, indicator_results: Dict[str, IndicatorResult],
                                    category_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate statistical summary of results."""
        scores = [r.raw_score for r in indicator_results.values()]
        weights = [r.weighted_score for r in indicator_results.values()]
        confidences = [r.confidence for r in indicator_results.values()]
        
        return {
            'indicator_statistics': {
                'total_indicators': len(indicator_results),
                'active_indicators': sum(1 for r in indicator_results.values() if r.raw_score > 0),
                'high_confidence_indicators': sum(1 for r in indicator_results.values() if r.confidence > 0.8),
                'mean_raw_score': np.mean(scores) if scores else 0.0,
                'std_raw_score': np.std(scores) if scores else 0.0,
                'mean_weighted_score': np.mean(weights) if weights else 0.0,
                'mean_confidence': np.mean(confidences) if confidences else 0.0,
                'max_score': max(scores) if scores else 0.0,
                'min_score': min(scores) if scores else 0.0
            },
            'category_scores': category_scores,
            'top_indicators': sorted(
                [(name, result.weighted_score) for name, result in indicator_results.items()],
                key=lambda x: x[1], reverse=True
            )[:5]  # Top 5 indicators
        }
    
    def _identify_risk_factors(self, indicator_results: Dict[str, IndicatorResult],
                             category_scores: Dict[str, float]) -> List[str]:
        """Identify primary risk factors contributing to anomaly score."""
        risk_factors = []
        
        # Category-based risk factors
        for category, score in category_scores.items():
            if score > 0.5:
                if category == 'orbital':
                    risk_factors.append("Anomalous orbital mechanics parameters")
                elif category == 'velocity':
                    risk_factors.append("Suspicious velocity patterns")
                elif category == 'temporal':
                    risk_factors.append("Artificial temporal patterns")
                elif category == 'geographic':
                    risk_factors.append("Geographic targeting patterns")
                elif category == 'physical':
                    risk_factors.append("Unusual physical characteristics")
                elif category == 'behavioral':
                    risk_factors.append("Anomalous observation patterns")
        
        # Specific high-impact indicators
        high_impact_indicators = {
            'eccentricity': "Anomalous orbital eccentricity",
            'orbital_resonance': "Artificial orbital resonance",
            'acceleration_anomalies': "Non-gravitational acceleration detected",
            'velocity_consistency': "Artificially consistent velocities",
            'approach_regularity': "Suspiciously regular close approaches",
            'subpoint_clustering': "Geographic clustering over sensitive areas"
        }
        
        for indicator_name, description in high_impact_indicators.items():
            if (indicator_name in indicator_results and 
                indicator_results[indicator_name].weighted_score > 0.6):
                risk_factors.append(description)
        
        # Aggregate multiple weak signals
        moderate_indicators = [
            name for name, result in indicator_results.items()
            if 0.3 < result.weighted_score <= 0.6
        ]
        
        if len(moderate_indicators) >= 4:
            risk_factors.append("Multiple moderate anomaly indicators")
        
        return risk_factors[:10]  # Limit to top 10 risk factors

class StatisticalAnalyzer:
    """Provides statistical analysis and comparison capabilities."""
    
    def __init__(self):
        self.score_history: List[AnomalyScore] = []
        self.population_statistics = {}
        
        logger.info("StatisticalAnalyzer initialized")
    
    def add_score(self, score: AnomalyScore) -> None:
        """Add a score to the statistical history."""
        self.score_history.append(score)
        self._update_population_statistics()
    
    def analyze_population(self, scores: List[AnomalyScore]) -> Dict[str, Any]:
        """Analyze a population of NEO scores."""
        if not scores:
            return {'error': 'No scores provided'}
        
        overall_scores = [s.overall_score for s in scores]
        confidences = [s.confidence for s in scores]
        
        # Classification distribution
        classifications = [s.classification for s in scores]
        classification_counts = {
            'natural': classifications.count('natural'),
            'suspicious': classifications.count('suspicious'), 
            'highly_suspicious': classifications.count('highly_suspicious'),
            'artificial': classifications.count('artificial')
        }
        
        # Statistical measures
        analysis = {
            'population_size': len(scores),
            'score_statistics': {
                'mean': np.mean(overall_scores),
                'median': np.median(overall_scores),
                'std': np.std(overall_scores),
                'min': np.min(overall_scores),
                'max': np.max(overall_scores),
                'q25': np.percentile(overall_scores, 25),
                'q75': np.percentile(overall_scores, 75)
            },
            'confidence_statistics': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences)
            },
            'classification_distribution': classification_counts,
            'classification_percentages': {
                k: v / len(scores) * 100 for k, v in classification_counts.items()
            },
            'anomaly_rate': (classification_counts['suspicious'] + 
                           classification_counts['highly_suspicious'] + 
                           classification_counts['artificial']) / len(scores) * 100
        }
        
        # Indicator analysis
        analysis['indicator_analysis'] = self._analyze_indicator_patterns(scores)
        
        # Outlier detection
        analysis['outliers'] = self._detect_statistical_outliers(scores)
        
        return analysis
    
    def compare_to_population(self, score: AnomalyScore) -> Dict[str, Any]:
        """Compare a single score to the population statistics."""
        if not self.score_history:
            return {'error': 'No population data available'}
        
        population_scores = [s.overall_score for s in self.score_history]
        
        # Calculate percentile rank
        percentile_rank = (np.sum(np.array(population_scores) <= score.overall_score) / 
                          len(population_scores) * 100)
        
        # Z-score calculation
        mean_score = np.mean(population_scores)
        std_score = np.std(population_scores)
        z_score = (score.overall_score - mean_score) / std_score if std_score > 0 else 0
        
        # Similar objects (within 1 std dev)
        similar_threshold = std_score
        similar_objects = [
            s for s in self.score_history 
            if abs(s.overall_score - score.overall_score) <= similar_threshold
        ]
        
        return {
            'percentile_rank': percentile_rank,
            'z_score': z_score,
            'population_mean': mean_score,
            'population_std': std_score,
            'similar_objects_count': len(similar_objects),
            'rarity_assessment': self._assess_rarity(percentile_rank, z_score),
            'comparison_context': self._generate_comparison_context(score, similar_objects)
        }
    
    def generate_trend_analysis(self, window_days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis over time."""
        if len(self.score_history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Sort by creation time
        sorted_scores = sorted(self.score_history, key=lambda x: x.created_at)
        
        # Recent window analysis
        cutoff_time = datetime.now() - timedelta(days=window_days)
        recent_scores = [s for s in sorted_scores if s.created_at >= cutoff_time]
        
        if not recent_scores:
            return {'error': f'No data in recent {window_days} days'}
        
        # Calculate trends
        recent_mean = np.mean([s.overall_score for s in recent_scores])
        
        # Compare to historical average
        if len(sorted_scores) > len(recent_scores):
            historical_scores = sorted_scores[:-len(recent_scores)]
            historical_mean = np.mean([s.overall_score for s in historical_scores])
            trend_direction = "increasing" if recent_mean > historical_mean else "decreasing"
            trend_magnitude = abs(recent_mean - historical_mean)
        else:
            historical_mean = recent_mean
            trend_direction = "stable"
            trend_magnitude = 0.0
        
        return {
            'window_days': window_days,
            'recent_objects': len(recent_scores),
            'recent_mean_score': recent_mean,
            'historical_mean_score': historical_mean,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude,
            'recent_anomaly_rate': len([s for s in recent_scores if s.classification != 'natural']) / len(recent_scores) * 100,
            'classification_trends': self._analyze_classification_trends(sorted_scores, window_days)
        }
    
    def _update_population_statistics(self) -> None:
        """Update cached population statistics."""
        if not self.score_history:
            return
        
        scores = [s.overall_score for s in self.score_history]
        
        self.population_statistics = {
            'count': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'last_updated': datetime.now()
        }
    
    def _analyze_indicator_patterns(self, scores: List[AnomalyScore]) -> Dict[str, Any]:
        """Analyze patterns across indicators."""
        indicator_scores = defaultdict(list)
        
        # Collect all indicator scores
        for score in scores:
            for indicator_name, result in score.indicator_scores.items():
                indicator_scores[indicator_name].append(result.weighted_score)
        
        # Analyze each indicator
        indicator_analysis = {}
        for indicator_name, values in indicator_scores.items():
            if values:
                indicator_analysis[indicator_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'active_fraction': np.mean([v > 0 for v in values]),
                    'high_score_fraction': np.mean([v > 0.5 for v in values])
                }
        
        # Identify most discriminating indicators
        discriminating_indicators = sorted(
            [(name, stats['std']) for name, stats in indicator_analysis.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            'indicator_statistics': indicator_analysis,
            'most_discriminating': discriminating_indicators,
            'total_indicators': len(indicator_analysis)
        }
    
    def _detect_statistical_outliers(self, scores: List[AnomalyScore]) -> List[Dict[str, Any]]:
        """Detect statistical outliers in the score population."""
        if len(scores) < 10:  # Need reasonable sample size
            return []
        
        overall_scores = [s.overall_score for s in scores]
        mean_score = np.mean(overall_scores)
        std_score = np.std(overall_scores)
        
        outliers = []
        for score in scores:
            z_score = abs(score.overall_score - mean_score) / std_score if std_score > 0 else 0
            
            if z_score > 2.5:  # More than 2.5 standard deviations
                outliers.append({
                    'designation': score.designation,
                    'overall_score': score.overall_score,
                    'z_score': z_score,
                    'classification': score.classification,
                    'outlier_type': 'high' if score.overall_score > mean_score else 'low'
                })
        
        return sorted(outliers, key=lambda x: x['z_score'], reverse=True)
    
    def _assess_rarity(self, percentile_rank: float, z_score: float) -> str:
        """Assess the rarity of a score."""
        if percentile_rank >= 99:
            return "extremely_rare"
        elif percentile_rank >= 95:
            return "very_rare"
        elif percentile_rank >= 90:
            return "rare"
        elif percentile_rank >= 75:
            return "uncommon"
        else:
            return "common"
    
    def _generate_comparison_context(self, score: AnomalyScore, 
                                   similar_objects: List[AnomalyScore]) -> Dict[str, Any]:
        """Generate context for comparison."""
        if not similar_objects:
            return {'message': 'No similar objects in database'}
        
        similar_classifications = [s.classification for s in similar_objects]
        most_common_classification = max(set(similar_classifications), 
                                       key=similar_classifications.count)
        
        return {
            'similar_objects_count': len(similar_objects),
            'most_common_classification': most_common_classification,
            'classification_agreement': similar_classifications.count(score.classification) / len(similar_classifications),
            'typical_score_range': (
                min(s.overall_score for s in similar_objects),
                max(s.overall_score for s in similar_objects)
            )
        }
    
    def _analyze_classification_trends(self, sorted_scores: List[AnomalyScore], 
                                     window_days: int) -> Dict[str, Any]:
        """Analyze trends in classifications over time."""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=window_days)
        recent_scores = [s for s in sorted_scores if s.created_at >= cutoff_time]
        historical_scores = [s for s in sorted_scores if s.created_at < cutoff_time]
        
        if not historical_scores:
            return {'message': 'No historical data for comparison'}
        
        def get_classification_rates(scores):
            if not scores:
                return {}
            total = len(scores)
            return {
                'natural': scores.count('natural') / total,
                'suspicious': scores.count('suspicious') / total,
                'highly_suspicious': scores.count('highly_suspicious') / total,
                'artificial': scores.count('artificial') / total
            }
        
        recent_classifications = [s.classification for s in recent_scores]
        historical_classifications = [s.classification for s in historical_scores]
        
        recent_rates = get_classification_rates(recent_classifications)
        historical_rates = get_classification_rates(historical_classifications)
        
        trends = {}
        for classification in ['natural', 'suspicious', 'highly_suspicious', 'artificial']:
            recent_rate = recent_rates.get(classification, 0)
            historical_rate = historical_rates.get(classification, 0)
            change = recent_rate - historical_rate
            
            trends[classification] = {
                'recent_rate': recent_rate,
                'historical_rate': historical_rate,
                'change': change,
                'trend': 'increasing' if change > 0.05 else 'decreasing' if change < -0.05 else 'stable'
            }
        
        return trends
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the entire analyzed population."""
        if not self.score_history:
            return {'message': 'No data available'}
        
        return {
            'total_objects_analyzed': len(self.score_history),
            'population_statistics': self.population_statistics,
            'overall_analysis': self.analyze_population(self.score_history),
            'last_analysis': max(s.created_at for s in self.score_history).isoformat() if self.score_history else None
        }