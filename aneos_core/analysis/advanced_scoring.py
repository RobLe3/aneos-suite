"""
Advanced Anomaly Scoring System for aNEOS - ATLAS Implementation

This module implements a sophisticated per-object anomaly scoring system based on 
multi-indicator analysis with continuous scoring, weighted importance, and human-readable flags.

Conceptual blueprint implementation:
- Multi-indicator blend with 6 core clue categories
- Continuous scoring (0→1) instead of binary classification
- Weighted importance with configurable parameters
- Human-origin veto system for space debris
- Human-readable flag reporting
- Transparent threshold system
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import json
import math

logger = logging.getLogger(__name__)

@dataclass
class AdvancedScoringConfig:
    """Configuration for advanced scoring system."""
    # Core indicator weights (6 main categories)
    encounter_geometry_weight: float = 0.15  # Distance & relative speed
    orbit_behavior_weight: float = 0.25      # Repeat passes, accelerations
    physical_traits_weight: float = 0.20     # Area-to-mass, radar, thermal
    spectral_identity_weight: float = 0.20   # Color curve anomalies
    dynamical_sanity_weight: float = 0.15    # Yarkovsky drift
    human_origin_weight: float = 0.05        # Space debris correlation
    
    # Transparent thresholds
    ordinary_threshold: float = 0.30         # Below: "ordinary"
    suspicious_threshold: float = 0.60       # 0.30-0.59: "suspicious"
    highly_suspicious_threshold: float = 1.0 # ≥0.60: "highly suspicious"
    
    # Human debris penalty
    debris_penalty: float = 0.4              # Fixed penalty for likely debris
    debris_confidence_threshold: float = 0.8 # Confidence needed for penalty
    
    # Continuous scoring parameters
    distance_scale: float = 0.1              # AU scale for distance scoring
    velocity_scale: float = 2.0              # km/s scale for velocity scoring
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'AdvancedScoringConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Filter out metadata and other non-parameter keys
            valid_params = {}
            # Get valid field names from dataclass
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(cls)}
            
            for key, value in config_data.items():
                if not key.startswith('_') and key in valid_fields:
                    valid_params[key] = value
            
            return cls(**valid_params)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # EMERGENCY: Suppress configuration loading warnings
            return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_data = self.__dict__
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

@dataclass
class ClueContribution:
    """Individual clue contribution to overall score."""
    name: str
    category: str
    raw_value: float         # Raw measurement/indicator value
    normalized_score: float  # Continuous score 0→1
    weight: float           # Importance weight
    contribution: float     # weighted score * weight
    confidence: float       # Confidence in this measurement
    flag: str              # Single-character flag for this clue
    explanation: str       # Human-readable explanation

@dataclass
class AdvancedAnomalyScore:
    """Complete advanced anomaly score with detailed breakdown."""
    designation: str
    overall_score: float                    # Final capped score 0→1
    raw_weighted_sum: float                # Before capping
    confidence: float                      # Overall confidence
    classification: str                    # ordinary/suspicious/highly_suspicious
    
    # Detailed breakdown
    clue_contributions: List[ClueContribution] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    flag_string: str = ""                  # Compact flag string (e.g., "d,v,Δ,μc")
    
    # Penalties and adjustments
    debris_penalty_applied: float = 0.0
    debris_match_info: Optional[Dict[str, Any]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    analysis_version: str = "ATLAS-1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'designation': self.designation,
            'overall_score': self.overall_score,
            'raw_weighted_sum': self.raw_weighted_sum,
            'confidence': self.confidence,
            'classification': self.classification,
            'clue_contributions': [
                {
                    'name': c.name,
                    'category': c.category,
                    'raw_value': c.raw_value,
                    'normalized_score': c.normalized_score,
                    'weight': c.weight,
                    'contribution': c.contribution,
                    'confidence': c.confidence,
                    'flag': c.flag,
                    'explanation': c.explanation
                } for c in self.clue_contributions
            ],
            'category_scores': self.category_scores,
            'flag_string': self.flag_string,
            'debris_penalty_applied': self.debris_penalty_applied,
            'debris_match_info': self.debris_match_info,
            'created_at': self.created_at.isoformat(),
            'analysis_version': self.analysis_version
        }

class AdvancedScoreCalculator:
    """Advanced multi-indicator scoring calculator."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with configuration."""
        if config_path and config_path.exists():
            self.config = AdvancedScoringConfig.load_from_file(config_path)
        else:
            self.config = AdvancedScoringConfig()
            
        # Define clue processors for each category
        self.clue_processors = {
            'encounter_geometry': self._process_encounter_geometry,
            'orbit_behavior': self._process_orbit_behavior,
            'physical_traits': self._process_physical_traits,
            'spectral_identity': self._process_spectral_identity,
            'dynamical_sanity': self._process_dynamical_sanity,
            'human_origin': self._process_human_origin
        }
        
        # Flag mappings for human-readable output
        self.flag_mappings = {
            # Encounter geometry flags
            'close_distance': 'd',     # Close miss distance
            'high_velocity': 'v',      # High relative velocity
            
            # Orbit behavior flags  
            'repeat_passes': 'r',      # Repeated close approaches
            'acceleration': 'Δ',       # Non-gravitational acceleration
            'resonance': 'ω',          # Orbital resonance
            
            # Physical traits flags
            'area_mass_ratio': 'μ',    # Unusual area-to-mass ratio
            'radar_smooth': 's',       # Radar smoothness
            'thermal_anomaly': 't',    # Thermal cooling anomaly
            
            # Spectral identity flags
            'spectral_outlier': 'c',   # Color curve outlier
            'material_match': 'm',     # Artificial material signature
            
            # Dynamical sanity flags
            'yarkovsky': 'y',          # Yarkovsky drift excess
            'stability': 'u',          # Orbital instability
            
            # Human origin flags (penalties)
            'debris_match': 'D',       # Space debris catalog match
            'launch_correlation': 'L'  # Launch correlation
        }
        
        # EMERGENCY: Suppress initialization logging
        # logger.info(f"AdvancedScoreCalculator initialized with {len(self.clue_processors)} clue categories")
    
    def calculate_score(self, neo_data: Dict[str, Any], 
                       indicator_results: Dict[str, Any]) -> AdvancedAnomalyScore:
        """Calculate advanced anomaly score with multi-indicator blend."""
        designation = neo_data.get('designation', 'Unknown')
        
        # Process each clue category
        all_clue_contributions = []
        category_scores = {}
        
        for category, processor in self.clue_processors.items():
            try:
                clues = processor(neo_data, indicator_results)
                all_clue_contributions.extend(clues)
                
                # Calculate category score
                if clues:
                    category_score = sum(c.contribution for c in clues) / len(clues)
                    category_scores[category] = min(category_score, 1.0)
                else:
                    category_scores[category] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error processing {category} for {designation}: {e}")
                category_scores[category] = 0.0
        
        # Calculate raw weighted sum
        raw_weighted_sum = sum(c.contribution for c in all_clue_contributions)
        
        # Apply human debris penalty
        debris_penalty, debris_info = self._calculate_debris_penalty(all_clue_contributions)
        adjusted_score = max(0.0, raw_weighted_sum - debris_penalty)
        
        # Cap at 1.0 (additive but capped)
        overall_score = min(adjusted_score, 1.0)
        
        # Calculate overall confidence
        if all_clue_contributions:
            confidence = np.mean([c.confidence for c in all_clue_contributions])
        else:
            confidence = 0.0
        
        # Determine classification
        classification = self._classify_score(overall_score)
        
        # Generate flag string
        flag_string = self._generate_flag_string(all_clue_contributions)
        
        return AdvancedAnomalyScore(
            designation=designation,
            overall_score=overall_score,
            raw_weighted_sum=raw_weighted_sum,
            confidence=confidence,
            classification=classification,
            clue_contributions=all_clue_contributions,
            category_scores=category_scores,
            flag_string=flag_string,
            debris_penalty_applied=debris_penalty,
            debris_match_info=debris_info
        )
    
    def _process_encounter_geometry(self, neo_data: Dict[str, Any], 
                                  indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process encounter geometry clues (distance & relative speed)."""
        clues = []
        
        # Distance-based scoring
        miss_distance = neo_data.get('miss_distance_au', None)
        if miss_distance is not None:
            # Continuous scoring: closer = higher score
            distance_score = np.exp(-miss_distance / self.config.distance_scale)
            
            clues.append(ClueContribution(
                name='close_approach_distance',
                category='encounter_geometry',
                raw_value=miss_distance,
                normalized_score=distance_score,
                weight=self.config.encounter_geometry_weight * 0.6,
                contribution=distance_score * self.config.encounter_geometry_weight * 0.6,
                confidence=0.9,  # Distance measurements are usually reliable
                flag='d' if distance_score > 0.5 else '',
                explanation=f"Miss distance: {miss_distance:.4f} AU"
            ))
        
        # Velocity-based scoring
        relative_velocity = neo_data.get('relative_velocity_km_s', None)
        if relative_velocity is not None:
            # High velocities can be suspicious for artificial objects
            velocity_score = min(relative_velocity / 50.0, 1.0)  # Cap at 50 km/s
            
            clues.append(ClueContribution(
                name='relative_velocity',
                category='encounter_geometry',
                raw_value=relative_velocity,
                normalized_score=velocity_score,
                weight=self.config.encounter_geometry_weight * 0.4,
                contribution=velocity_score * self.config.encounter_geometry_weight * 0.4,
                confidence=0.8,
                flag='v' if velocity_score > 0.6 else '',
                explanation=f"Relative velocity: {relative_velocity:.2f} km/s"
            ))
        
        return clues
    
    def _process_orbit_behavior(self, neo_data: Dict[str, Any], 
                              indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process orbit behavior clues (repeat passes, accelerations)."""
        clues = []
        
        # Repeat passes indicator
        approach_regularity = indicator_results.get('approach_regularity', {})
        if approach_regularity.get('raw_score', 0) > 0:
            regularity_score = approach_regularity['raw_score']
            
            clues.append(ClueContribution(
                name='repeat_approaches',
                category='orbit_behavior',
                raw_value=regularity_score,
                normalized_score=regularity_score,
                weight=self.config.orbit_behavior_weight * 0.4,
                contribution=regularity_score * self.config.orbit_behavior_weight * 0.4,
                confidence=approach_regularity.get('confidence', 0.5),
                flag='r' if regularity_score > 0.5 else '',
                explanation="Suspiciously regular close approaches detected"
            ))
        
        # Non-gravitational acceleration (from ΔBIC analysis)
        delta_bic_result = indicator_results.get('delta_bic_analysis', {})
        if delta_bic_result.get('weighted_score', 0) > 0:
            acceleration_score = delta_bic_result['weighted_score']
            
            clues.append(ClueContribution(
                name='non_gravitational_acceleration',
                category='orbit_behavior',
                raw_value=delta_bic_result.get('raw_score', 0),
                normalized_score=acceleration_score,
                weight=self.config.orbit_behavior_weight * 0.6,
                contribution=acceleration_score * self.config.orbit_behavior_weight * 0.6,
                confidence=delta_bic_result.get('confidence', 0.7),
                flag='Δ' if acceleration_score > 0.4 else '',
                explanation="Non-gravitational acceleration detected via ΔBIC analysis"
            ))
        
        return clues
    
    def _process_physical_traits(self, neo_data: Dict[str, Any], 
                               indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process physical traits clues (area-to-mass, radar, thermal)."""
        clues = []
        
        # Area-to-mass ratio anomaly
        if 'diameter' in neo_data and 'mass_estimate' in neo_data:
            diameter = neo_data['diameter']
            mass = neo_data['mass_estimate']
            
            # Calculate area-to-mass ratio
            area = np.pi * (diameter/2)**2
            area_mass_ratio = area / mass if mass > 0 else 0
            
            # High area-to-mass ratios suggest artificial structures
            ratio_score = min(area_mass_ratio / 10.0, 1.0)  # Normalize
            
            clues.append(ClueContribution(
                name='area_mass_ratio',
                category='physical_traits',
                raw_value=area_mass_ratio,
                normalized_score=ratio_score,
                weight=self.config.physical_traits_weight * 0.4,
                contribution=ratio_score * self.config.physical_traits_weight * 0.4,
                confidence=0.6,
                flag='μ' if ratio_score > 0.6 else '',
                explanation=f"Area-to-mass ratio: {area_mass_ratio:.2f} m²/kg"
            ))
        
        # Radar polarization analysis
        radar_result = indicator_results.get('radar_polarization', {})
        if radar_result.get('weighted_score', 0) > 0:
            radar_score = radar_result['weighted_score']
            
            clues.append(ClueContribution(
                name='radar_polarization',
                category='physical_traits',
                raw_value=radar_result.get('raw_score', 0),
                normalized_score=radar_score,
                weight=self.config.physical_traits_weight * 0.3,
                contribution=radar_score * self.config.physical_traits_weight * 0.3,
                confidence=radar_result.get('confidence', 0.7),
                flag='s' if radar_score > 0.5 else '',
                explanation="Radar polarization indicates artificial surface properties"
            ))
        
        # Thermal-IR analysis
        thermal_result = indicator_results.get('thermal_ir_analysis', {})
        if thermal_result.get('weighted_score', 0) > 0:
            thermal_score = thermal_result['weighted_score']
            
            clues.append(ClueContribution(
                name='thermal_signature',
                category='physical_traits',
                raw_value=thermal_result.get('raw_score', 0),
                normalized_score=thermal_score,
                weight=self.config.physical_traits_weight * 0.3,
                contribution=thermal_score * self.config.physical_traits_weight * 0.3,
                confidence=thermal_result.get('confidence', 0.6),
                flag='t' if thermal_score > 0.5 else '',
                explanation="Thermal signature suggests artificial materials"
            ))
        
        return clues
    
    def _process_spectral_identity(self, neo_data: Dict[str, Any], 
                                 indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process spectral identity clues (color curve anomalies)."""
        clues = []
        
        # Spectral outlier analysis
        spectral_result = indicator_results.get('spectral_outlier', {})
        if spectral_result.get('weighted_score', 0) > 0:
            spectral_score = spectral_result['weighted_score']
            
            clues.append(ClueContribution(
                name='spectral_anomaly',
                category='spectral_identity',
                raw_value=spectral_result.get('raw_score', 0),
                normalized_score=spectral_score,
                weight=self.config.spectral_identity_weight,
                contribution=spectral_score * self.config.spectral_identity_weight,
                confidence=spectral_result.get('confidence', 0.7),
                flag='c' if spectral_score > 0.4 else '',
                explanation="Spectral signature unlike any known natural asteroid"
            ))
        
        return clues
    
    def _process_dynamical_sanity(self, neo_data: Dict[str, Any], 
                                indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process dynamical sanity clues (Yarkovsky drift)."""
        clues = []
        
        # Yarkovsky effect analysis
        # This would need actual orbital evolution data
        # For now, use a placeholder based on orbital characteristics
        
        eccentricity = neo_data.get('eccentricity', 0)
        inclination = neo_data.get('inclination', 0)
        
        # High eccentricity + inclination might suggest artificial insertion
        dynamical_anomaly = min((eccentricity * inclination) / 10.0, 1.0)
        
        if dynamical_anomaly > 0.1:
            clues.append(ClueContribution(
                name='orbital_dynamics',
                category='dynamical_sanity',
                raw_value=dynamical_anomaly,
                normalized_score=dynamical_anomaly,
                weight=self.config.dynamical_sanity_weight,
                contribution=dynamical_anomaly * self.config.dynamical_sanity_weight,
                confidence=0.5,  # Lower confidence without proper evolution data
                flag='y' if dynamical_anomaly > 0.5 else '',
                explanation=f"Orbital dynamics suggest artificial trajectory"
            ))
        
        return clues
    
    def _process_human_origin(self, neo_data: Dict[str, Any], 
                            indicator_results: Dict[str, Any]) -> List[ClueContribution]:
        """Process human origin clues (space debris correlation)."""
        clues = []
        
        # Human hardware analysis
        hardware_result = indicator_results.get('human_hardware', {})
        if hardware_result.get('weighted_score', 0) > 0:
            hardware_score = hardware_result['weighted_score']
            
            # This contributes to scoring but also triggers penalty
            clues.append(ClueContribution(
                name='space_debris_correlation',
                category='human_origin',
                raw_value=hardware_result.get('raw_score', 0),
                normalized_score=hardware_score,
                weight=self.config.human_origin_weight,
                contribution=hardware_score * self.config.human_origin_weight,
                confidence=hardware_result.get('confidence', 0.8),
                flag='D' if hardware_score > 0.7 else '',
                explanation="Correlation with space debris catalog detected"
            ))
        
        return clues
    
    def _calculate_debris_penalty(self, clue_contributions: List[ClueContribution]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Calculate penalty for likely human debris."""
        debris_clues = [c for c in clue_contributions if c.category == 'human_origin']
        
        if not debris_clues:
            return 0.0, None
        
        # Find highest confidence debris match
        best_debris_clue = max(debris_clues, key=lambda c: c.confidence * c.normalized_score)
        
        # Apply penalty if confidence is high enough
        if (best_debris_clue.confidence >= self.config.debris_confidence_threshold and 
            best_debris_clue.normalized_score > 0.5):
            
            penalty_info = {
                'matched_clue': best_debris_clue.name,
                'confidence': best_debris_clue.confidence,
                'match_score': best_debris_clue.normalized_score,
                'penalty_applied': self.config.debris_penalty
            }
            
            return self.config.debris_penalty, penalty_info
        
        return 0.0, None
    
    def _classify_score(self, overall_score: float) -> str:
        """Classify score using transparent thresholds."""
        if overall_score < self.config.ordinary_threshold:
            return 'ordinary'
        elif overall_score < self.config.suspicious_threshold:
            return 'suspicious'
        else:
            return 'highly_suspicious'
    
    def _generate_flag_string(self, clue_contributions: List[ClueContribution]) -> str:
        """Generate compact human-readable flag string."""
        active_flags = []
        
        for clue in clue_contributions:
            if clue.flag and clue.flag not in active_flags:
                active_flags.append(clue.flag)
        
        # Sort flags for consistent output
        active_flags.sort()
        
        return ','.join(active_flags)
    
    def explain_score(self, score: AdvancedAnomalyScore) -> str:
        """Generate human-readable explanation of the score."""
        explanation_lines = [
            f"Object: {score.designation}",
            f"Overall Score: {score.overall_score:.3f} ({score.classification})",
            f"Confidence: {score.confidence:.3f}",
            f"Flags: {score.flag_string}",
            "",
            "Contributing factors:"
        ]
        
        # Sort contributions by impact
        sorted_contributions = sorted(score.clue_contributions, 
                                    key=lambda c: c.contribution, reverse=True)
        
        for clue in sorted_contributions[:5]:  # Top 5 contributors
            impact = clue.contribution
            if impact > 0.05:  # Only show meaningful contributions
                explanation_lines.append(
                    f"  • {clue.explanation} (impact: {impact:.3f})"
                )
        
        if score.debris_penalty_applied > 0:
            explanation_lines.append(
                f"  • Space debris penalty applied: -{score.debris_penalty_applied:.3f}"
            )
        
        return "\n".join(explanation_lines)
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update configuration parameters."""
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
    
    def save_config(self, config_path: Path) -> None:
        """Save current configuration to file."""
        self.config.save_to_file(config_path)
        logger.info(f"Configuration saved to {config_path}")