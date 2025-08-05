#!/usr/bin/env python3
"""
Advanced Analytics and Classification System for aNEOS Core

Provides mission priority ranking, anomaly categorization, and advanced
classification capabilities with academic rigor for NEO analysis.
"""

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class NEOClassificationSystem:
    """
    Advanced NEO classification and categorization system.
    
    Provides comprehensive classification based on orbital mechanics,
    anomaly detection, and mission priority assessment.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize classification system.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.has_sklearn = HAS_SKLEARN
        self.has_pandas = HAS_PANDAS
        
        # Classification thresholds and weights
        self.config = {
            "thresholds": {
                "eccentricity_hyperbolic": 1.0,
                "eccentricity_stable": 0.8,
                "inclination_high": 45.0,
                "velocity_shift_significant": 5.0,
                "anomaly_verification": 10.0,
                "delta_v_high": 2.0,
                "priority_high": 5.0,
                "priority_medium": 2.0,
                "albedo_artificial": 0.6,
                "observation_gap_suspicious": 100.0
            },
            "weights": {
                "orbital_mechanics": 1.5,
                "velocity_anomalies": 2.0,
                "close_approach_regularity": 2.0,
                "physical_characteristics": 1.0,
                "temporal_patterns": 1.0,
                "acceleration_anomalies": 2.0,
                "spectral_characteristics": 1.5,
                "observation_consistency": 1.0
            },
            "categories": {
                "iso_candidate": "Interstellar Object Candidate",
                "true_anomaly": "True Anomaly (ΔV)",
                "neo_reclassified": "NEO (Reclassified)",
                "neo_stable": "NEO (Stable)",
                "highly_anomalous": "Extremely Anomalous / Potentially Artificial",
                "moderately_anomalous": "Moderately Anomalous",
                "slightly_anomalous": "Slightly Anomalous",
                "normal_range": "Within Normal Range",
                "uncategorized": "Uncategorized"
            }
        }
    
    def calculate_escape_velocity(self, neo_data: Dict[str, Any]) -> float:
        """
        Calculate escape velocity for the object.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Escape velocity in km/s
        """
        # Standard Earth escape velocity - can be enhanced with object-specific calculations
        return 11.2
    
    def assess_orbital_mechanics(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess orbital mechanics characteristics.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Orbital mechanics assessment
        """
        assessment = {
            "is_hyperbolic": False,
            "is_highly_eccentric": False,
            "is_highly_inclined": False,
            "orbital_stability": "unknown",
            "classification_confidence": 0.0
        }
        
        # Eccentricity assessment
        eccentricity = neo_data.get("eccentricity", 0)
        if eccentricity > self.config["thresholds"]["eccentricity_hyperbolic"]:
            assessment["is_hyperbolic"] = True
            assessment["orbital_stability"] = "hyperbolic"
        elif eccentricity > self.config["thresholds"]["eccentricity_stable"]:
            assessment["is_highly_eccentric"] = True
            assessment["orbital_stability"] = "unstable"
        else:
            assessment["orbital_stability"] = "stable"
        
        # Inclination assessment
        inclination = neo_data.get("inclination", 0)
        if inclination > self.config["thresholds"]["inclination_high"]:
            assessment["is_highly_inclined"] = True
        
        # Calculate confidence based on data availability
        available_params = sum([
            1 for param in ["eccentricity", "inclination", "semi_major_axis"]
            if neo_data.get(param) is not None and neo_data.get(param) != 0
        ])
        assessment["classification_confidence"] = available_params / 3.0
        
        return assessment
    
    def assess_velocity_anomalies(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess velocity-related anomalies.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Velocity anomaly assessment
        """
        assessment = {
            "has_velocity_anomaly": False,
            "delta_v_score": 0.0,
            "expected_velocity": 0.0,
            "velocity_deviation": 0.0,
            "anomaly_significance": "none"
        }
        
        # Delta-V analysis
        delta_v = neo_data.get("delta_v", 0)
        expected_delta_v = neo_data.get("expected_delta_v", 0)
        anomaly_confidence = neo_data.get("anomaly_confidence", 0)
        
        if delta_v > 0 and expected_delta_v > 0:
            assessment["delta_v_score"] = delta_v
            assessment["expected_velocity"] = expected_delta_v
            assessment["velocity_deviation"] = abs(delta_v - expected_delta_v)
            
            if anomaly_confidence > self.config["thresholds"]["anomaly_verification"]:
                assessment["has_velocity_anomaly"] = True
                assessment["anomaly_significance"] = "high"
            elif anomaly_confidence > 5.0:
                assessment["anomaly_significance"] = "moderate"
            elif anomaly_confidence > 1.0:
                assessment["anomaly_significance"] = "low"
        
        return assessment
    
    def assess_physical_characteristics(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess physical characteristics that might indicate artificial nature.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Physical characteristics assessment
        """
        assessment = {
            "unusual_albedo": False,
            "unusual_size": False,
            "spectral_anomaly": False,
            "artificial_likelihood": "low",
            "physical_confidence": 0.0
        }
        
        # Albedo analysis
        albedo = neo_data.get("albedo", 0)
        if albedo > self.config["thresholds"]["albedo_artificial"]:
            assessment["unusual_albedo"] = True
            assessment["artificial_likelihood"] = "moderate"
        
        # Size analysis (if available)
        diameter = neo_data.get("diameter", 0)
        if diameter > 0:
            if diameter < 0.1 or diameter > 10.0:  # Unusual size ranges
                assessment["unusual_size"] = True
        
        # Calculate confidence based on available physical data
        physical_params = ["albedo", "diameter", "absolute_magnitude"]
        available_physical = sum([
            1 for param in physical_params
            if neo_data.get(param) is not None and neo_data.get(param) != 0
        ])
        assessment["physical_confidence"] = available_physical / len(physical_params)
        
        return assessment
    
    def categorize_by_dynamic_tas(self, neo_data: Dict[str, Any]) -> str:
        """
        Categorize NEO based on Dynamic TAS values.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Category string
        """
        dyn_tas = neo_data.get("Dynamic TAS") or neo_data.get("dynamic_tas", 0)
        dyn_cat = neo_data.get("Dynamic Category") or neo_data.get("dynamic_category")
        
        # Use existing category if available and valid
        if dyn_cat and str(dyn_cat).lower() not in ["unknown", "uncategorized"]:
            return str(dyn_cat)
        
        # Categorize based on Dynamic TAS value
        if dyn_tas is not None and dyn_tas != 0:
            if dyn_tas < 0.5:
                return self.config["categories"]["normal_range"]
            elif dyn_tas < 1.0:
                return self.config["categories"]["slightly_anomalous"]
            elif dyn_tas < 2.0:
                return self.config["categories"]["moderately_anomalous"]
            elif dyn_tas < 3.0:
                return "Highly Anomalous"
            else:
                return self.config["categories"]["highly_anomalous"]
        
        return self.config["categories"]["uncategorized"]
    
    def classify_neo_comprehensive(self, neo_data: Dict[str, Any], 
                                 previous_classification: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive NEO classification.
        
        Args:
            neo_data: NEO data dictionary
            previous_classification: Previous classification if available
            
        Returns:
            Comprehensive classification results
        """
        # Perform individual assessments
        orbital_assessment = self.assess_orbital_mechanics(neo_data)
        velocity_assessment = self.assess_velocity_anomalies(neo_data)
        physical_assessment = self.assess_physical_characteristics(neo_data)
        
        # Determine primary classification
        primary_category = self._determine_primary_category(
            neo_data, orbital_assessment, velocity_assessment, physical_assessment
        )
        
        # Calculate classification confidence
        confidence_factors = [
            orbital_assessment["classification_confidence"],
            physical_assessment["physical_confidence"],
            1.0 if velocity_assessment["has_velocity_anomaly"] else 0.5
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
        
        # Generate reclassification reasons
        reclassification_reasons = self._generate_reclassification_reasons(
            neo_data, primary_category, previous_classification,
            orbital_assessment, velocity_assessment
        )
        
        # Verification status
        is_verified = neo_data.get("anomaly_confidence", 0) > self.config["thresholds"]["anomaly_verification"]
        
        classification = {
            "primary_category": primary_category,
            "previous_classification": previous_classification or "Unknown",
            "classification_confidence": overall_confidence,
            "is_verified_anomaly": is_verified,
            "verification_status": "[Verified]" if is_verified else "[Unverified]",
            "reclassification_reasons": reclassification_reasons,
            "orbital_assessment": orbital_assessment,
            "velocity_assessment": velocity_assessment,
            "physical_assessment": physical_assessment,
            "classification_timestamp": datetime.now().isoformat()
        }
        
        return classification
    
    def _determine_primary_category(self, neo_data: Dict[str, Any],
                                  orbital_assessment: Dict[str, Any],
                                  velocity_assessment: Dict[str, Any],
                                  physical_assessment: Dict[str, Any]) -> str:
        """Determine the primary classification category."""
        
        # Check for ISO candidate (highest priority)
        eccentricity = neo_data.get("eccentricity", 0)
        v_inf = neo_data.get("v_inf", 0)
        escape_vel = self.calculate_escape_velocity(neo_data)
        
        if eccentricity > 1 and v_inf > escape_vel:
            return self.config["categories"]["iso_candidate"]
        
        # Check for true velocity anomaly
        if velocity_assessment["has_velocity_anomaly"]:
            delta_v_score = velocity_assessment["delta_v_score"]
            if delta_v_score > self.config["thresholds"]["delta_v_high"]:
                return self.config["categories"]["true_anomaly"]
        
        # Check for reclassification from previous NEO status
        previous_class = neo_data.get("previous_classification", "")
        if ("NEO" in previous_class and 
            orbital_assessment["orbital_stability"] == "stable"):
            return self.config["categories"]["neo_reclassified"]
        
        # Use Dynamic TAS categorization as fallback
        return self.categorize_by_dynamic_tas(neo_data)
    
    def _generate_reclassification_reasons(self, neo_data: Dict[str, Any],
                                         current_category: str,
                                         previous_category: Optional[str],
                                         orbital_assessment: Dict[str, Any],
                                         velocity_assessment: Dict[str, Any]) -> List[str]:
        """Generate list of reclassification reasons."""
        reasons = []
        
        # Orbital changes
        if orbital_assessment["is_hyperbolic"]:
            reasons.append("Eccentricity exceeded 1 (Hyperbolic orbit)")
        
        # Velocity changes
        if velocity_assessment["has_velocity_anomaly"]:
            deviation = velocity_assessment["velocity_deviation"]
            reasons.append(f"ΔV anomaly detected (deviation: {deviation:.2f} km/s)")
        
        # Stability assessment
        if orbital_assessment["orbital_stability"] == "stable":
            reasons.append("Now following a stable Keplerian orbit")
        
        # Category-specific reasons
        if current_category == self.config["categories"]["iso_candidate"]:
            reasons.append("Classified as potential interstellar object")
        elif current_category == self.config["categories"]["true_anomaly"]:
            reasons.append("Significant velocity anomaly confirmed")
        
        # Compare with previous classification
        if previous_category and previous_category != current_category:
            reasons.append(f"Reclassified from '{previous_category}' to '{current_category}'")
        
        return reasons


class MissionPriorityCalculator:
    """
    Advanced mission priority calculation system.
    
    Calculates mission priority scores based on multiple factors including
    anomaly significance, accessibility, scientific value, and risk assessment.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize priority calculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Priority weights for different factors
        self.priority_weights = {
            "anomaly_significance": 3.0,
            "accessibility": 2.0,
            "scientific_value": 2.5,
            "verification_status": 1.5,
            "temporal_urgency": 1.0,
            "observability": 1.0,
            "uniqueness": 2.0
        }
        
        # Scoring thresholds
        self.scoring_thresholds = {
            "high_priority": 8.0,
            "medium_priority": 4.0,
            "low_priority": 1.0
        }
    
    def calculate_anomaly_significance_score(self, neo_data: Dict[str, Any]) -> float:
        """Calculate score based on anomaly significance."""
        anomaly_confidence = neo_data.get("anomaly_confidence", 0)
        delta_v_score = neo_data.get("delta_v_anomaly_score", 0) or anomaly_confidence
        
        # Base score from anomaly confidence
        base_score = min(delta_v_score / 10.0, 5.0)  # Cap at 5.0
        
        # Bonus for verified anomalies
        if anomaly_confidence > 10:
            base_score *= 1.5
        
        # Bonus for specific categories
        category = neo_data.get("category", "")
        if "ISO Candidate" in category:
            base_score *= 2.0
        elif "True Anomaly" in category:
            base_score *= 1.8
        elif "Extremely Anomalous" in category:
            base_score *= 1.6
        
        return min(base_score, 10.0)  # Cap at 10.0
    
    def calculate_accessibility_score(self, neo_data: Dict[str, Any]) -> float:
        """Calculate score based on mission accessibility."""
        # Basic accessibility based on orbital parameters
        score = 5.0  # Base score
        
        # Distance factor
        semi_major_axis = neo_data.get("semi_major_axis", 1.0)
        if semi_major_axis > 0:
            # Closer to Earth's orbit is more accessible
            distance_factor = 1.0 / (abs(semi_major_axis - 1.0) + 0.1)
            score *= min(distance_factor, 2.0)
        
        # Eccentricity factor - less eccentric orbits are more accessible
        eccentricity = neo_data.get("eccentricity", 0)
        if eccentricity < 0.3:
            score *= 1.5
        elif eccentricity > 0.8:
            score *= 0.7
        
        # Inclination factor - lower inclination is more accessible
        inclination = neo_data.get("inclination", 0)
        if inclination < 10:
            score *= 1.3
        elif inclination > 45:
            score *= 0.8
        
        return min(score, 10.0)
    
    def calculate_scientific_value_score(self, neo_data: Dict[str, Any]) -> float:
        """Calculate score based on scientific value."""
        score = 5.0  # Base scientific value
        
        # Anomaly types add scientific value
        if neo_data.get("ai_validated_anomaly", False):
            score += 2.0
        
        # Physical characteristics
        if neo_data.get("albedo", 0) > 0.6:  # Unusual albedo
            score += 1.5
        
        # Orbital uniqueness
        eccentricity = neo_data.get("eccentricity", 0)
        if eccentricity > 1.0:  # Hyperbolic orbit
            score += 3.0
        elif eccentricity > 0.9:  # Highly eccentric
            score += 1.5
        
        # High inclination objects
        inclination = neo_data.get("inclination", 0)
        if inclination > 60:
            score += 1.0
        
        # Multiple close approaches (regular visitor)
        close_approaches = neo_data.get("Close Approaches", 0)
        if close_approaches > 5:
            score += 1.0
        
        return min(score, 10.0)
    
    def calculate_verification_score(self, neo_data: Dict[str, Any]) -> float:
        """Calculate score based on verification status and confidence."""
        anomaly_confidence = neo_data.get("anomaly_confidence", 0)
        
        if anomaly_confidence > 15:
            return 10.0
        elif anomaly_confidence > 10:
            return 8.0
        elif anomaly_confidence > 5:
            return 5.0
        elif anomaly_confidence > 1:
            return 3.0
        else:
            return 1.0
    
    def calculate_temporal_urgency_score(self, neo_data: Dict[str, Any]) -> float:
        """Calculate score based on temporal factors."""
        # Base urgency
        score = 5.0
        
        # Objects with recent observations are more urgent
        obs_end = neo_data.get("Observation End", "")
        if obs_end:
            try:
                # Simple check if observation is recent (placeholder logic)
                if "2024" in obs_end or "2025" in obs_end:
                    score += 2.0
            except:
                pass
        
        # Objects with many close approaches need timely study
        close_approaches = neo_data.get("Close Approaches", 0)
        if close_approaches > 3:
            score += 1.0
        
        return min(score, 10.0)
    
    def calculate_comprehensive_priority_score(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive mission priority score.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Priority scoring breakdown
        """
        # Calculate individual scores
        scores = {
            "anomaly_significance": self.calculate_anomaly_significance_score(neo_data),
            "accessibility": self.calculate_accessibility_score(neo_data),
            "scientific_value": self.calculate_scientific_value_score(neo_data),
            "verification_status": self.calculate_verification_score(neo_data),
            "temporal_urgency": self.calculate_temporal_urgency_score(neo_data),
            "observability": 5.0,  # Placeholder - could be enhanced
            "uniqueness": 5.0      # Placeholder - could be enhanced
        }
        
        # Calculate weighted total
        weighted_total = sum(
            scores[factor] * self.priority_weights[factor]
            for factor in scores.keys()
        )
        
        # Normalize to 0-10 scale
        max_possible = sum(10.0 * weight for weight in self.priority_weights.values())
        normalized_score = (weighted_total / max_possible) * 10.0
        
        # Determine priority tier
        if normalized_score >= self.scoring_thresholds["high_priority"]:
            priority_tier = "HIGH"
        elif normalized_score >= self.scoring_thresholds["medium_priority"]:
            priority_tier = "MEDIUM"
        else:
            priority_tier = "LOW"
        
        priority_result = {
            "overall_score": normalized_score,
            "priority_tier": priority_tier,
            "individual_scores": scores,
            "scoring_breakdown": {
                factor: scores[factor] * self.priority_weights[factor]
                for factor in scores.keys()
            },
            "calculation_timestamp": datetime.now().isoformat()
        }
        
        return priority_result
    
    def rank_neos_by_priority(self, neo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank a list of NEOs by mission priority.
        
        Args:
            neo_list: List of NEO data dictionaries
            
        Returns:
            Sorted list with priority scores added
        """
        enhanced_neos = []
        
        for neo in neo_list:
            neo_copy = neo.copy()
            priority_data = self.calculate_comprehensive_priority_score(neo)
            
            # Add priority information to NEO data
            neo_copy["priority_score"] = priority_data["overall_score"]
            neo_copy["priority_tier"] = priority_data["priority_tier"]
            neo_copy["priority_breakdown"] = priority_data
            
            enhanced_neos.append(neo_copy)
        
        # Sort by priority score (highest first)
        enhanced_neos.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        # Add ranking information
        for i, neo in enumerate(enhanced_neos, 1):
            neo["priority_rank"] = i
        
        self.logger.info(f"Ranked {len(enhanced_neos)} NEOs by mission priority")
        
        return enhanced_neos


def create_classification_system(logger: Optional[logging.Logger] = None) -> NEOClassificationSystem:
    """Create a NEO classification system instance."""
    return NEOClassificationSystem(logger)


def create_priority_calculator(logger: Optional[logging.Logger] = None) -> MissionPriorityCalculator:
    """Create a mission priority calculator instance."""
    return MissionPriorityCalculator(logger)