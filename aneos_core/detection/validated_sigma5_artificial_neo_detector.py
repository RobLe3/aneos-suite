#!/usr/bin/env python3
"""Legacy-named facade for uncalibrated candidate screening.

The earlier sigma/posterior interpretation is withdrawn. The canonical service
returns a ranking score and candidate status, with unavailable significance and
origin assessment represented by null. No scientific calibration is installed.
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from ..utils.statistical_utils import fisher_log_p_value, finite_probability
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Bus-DeMeo taxonomy: common natural NEO spectral types (~99% of known NEOs)
# Source: DeMeo et al. 2009; Binzel et al. 2019
_NATURAL_SPECTRAL_TYPES = frozenset({
    "S", "Sq", "Sr", "Sv", "Sa", "Q", "Qw",   # S-complex
    "C", "Cb", "Cg", "Cgh", "Ch",              # C-complex
    "X", "Xc", "Xe", "Xk",                     # X-complex
    "D", "T", "K", "L", "V", "B", "O", "R",    # Other natural
})

class EvidenceType(Enum):
    """Types of evidence for artificial object detection."""
    ORBITAL_DYNAMICS = "orbital_dynamics"
    PHYSICAL_PROPERTIES = "physical_properties"
    TEMPORAL_SIGNATURES = "temporal_signatures"       # RESERVED — requires multi-epoch observation data
    SPECTRAL_ANALYSIS = "spectral_analysis"
    RADAR_CHARACTERISTICS = "radar_characteristics"  # RESERVED — requires Goldstone/Arecibo integration
    COURSE_CORRECTIONS = "course_corrections"  # Implausible orbital changes
    TRAJECTORY_PATTERNS = "trajectory_patterns"  # Repeated exact passages
    PROPULSION_SIGNATURES = "propulsion_signatures"  # Evidence of thrust

@dataclass
class EvidenceSource:
    """Individual piece of evidence with uncertainty quantification."""
    evidence_type: EvidenceType
    anomaly_score: Optional[float]  # Uncalibrated ranking contribution
    confidence_interval: Tuple[Optional[float], ...]  # Unavailable in canonical output
    sample_size: int  # Size of reference population
    p_value: Optional[float]  # Unavailable without a calibrated null
    effect_size: Optional[float]  # Legacy standardized ranking contribution
    quality_score: Optional[float]  # Legacy completeness weight; unavailable in canonical output
    analyzed: bool = True        # False when skipped due to missing input data
    data_available: bool = True  # False when no data was present for this type

@dataclass 
class ValidationResult:
    """Result from validation against ground truth dataset."""
    true_positives: int
    false_positives: int  
    true_negatives: int
    false_negatives: int
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate
    positive_predictive_value: float
    negative_predictive_value: float
    f1_score: float
    
@dataclass
class Sigma5DetectionResult:
    """Legacy result shape with unavailable inference explicitly nullable."""
    is_artificial: Optional[bool]
    sigma_confidence: Optional[float]  # Actual sigma confidence level
    bayesian_probability: Optional[float]  # P(artificial | evidence)
    evidence_sources: List[EvidenceSource]
    combined_p_value: Optional[float]  # Fisher's combined probability
    false_discovery_rate: Optional[float]  # Unavailable without independent evaluation
    validation_metrics: Optional[ValidationResult]
    analysis_metadata: Dict[str, Any]
    screening_score: Optional[float] = None

class ValidatedSigma5ArtificialNEODetector:
    """Legacy-named score engine; no installed calibrated inference model."""
    
    # Statistical thresholds based on astronomical discovery standards
    SIGMA_5_CONFIDENCE = 5.0  # 5-sigma confidence threshold
    SIGMA_5_P_VALUE = 2 * stats.norm.sf(5.0)  # ~5.7e-7
    MIN_EVIDENCE_SOURCES = 2  # Require multiple independent evidence types
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Legacy fixed reference parameters, retained only for ranking comparison.
        # Their installed values are not an independently validated null.
        self.neo_population_stats = self._load_validated_neo_parameters()
        
        # Ground truth datasets for validation
        self.artificial_objects_db = self._load_artificial_objects_database()
        self.natural_objects_db = self._load_natural_objects_database()
        
        # Validation metrics from cross-validation
        self.validation_results = None
        # Calibration requires a frozen independent evaluation, never constructor toy CV.
        
    def _load_validated_neo_parameters(self) -> Dict[str, Dict[str, float]]:
        """Load legacy fixed reference values; citations are not calibration evidence."""
        
        # Parameters derived from:
        # - Bottke et al. (2002) - NEO population model
        # - Granvik et al. (2018) - NEO orbital distribution  
        # - Stuart & Binzel (2004) - Physical properties
        # - JPL SBDB (2024) - Observational database
        
        return {
            'orbital_elements': {
                'semi_major_axis': {
                    'mean': 1.6451,  # AU (Granvik et al. 2018)
                    'std': 0.6317,   # AU 
                    'n_samples': 9998,
                    'confidence_interval_95': (1.6335, 1.6577),
                    'source': 'Granvik et al. 2018, observational survey'
                },
                'eccentricity': {
                    'mean': 0.4329,  # (Granvik et al. 2018)
                    'std': 0.2266,
                    'n_samples': 9998, 
                    'confidence_interval_95': (0.4285, 0.4370),
                    'source': 'Granvik et al. 2018, observational survey'
                },
                'inclination': {
                    'mean': 15.0114,  # degrees (Granvik et al. 2018)
                    'std': 7.7786,   # degrees
                    'n_samples': 9998,
                    'confidence_interval_95': (14.8526, 15.1699), 
                    'source': 'Granvik et al. 2018, observational survey'
                }
            },
            'physical_properties': {
                'absolute_magnitude': {
                    'mean': 21.5,    # Based on WISE survey (Mainzer et al. 2011)
                    'std': 2.8,
                    'n_samples': 158000,
                    'source': 'WISE/NEOWISE survey data'
                },
                'diameter': {
                    'mean': 800.0,   # metres (log-normal; 0.8 km median WISE survey)
                    'std': 1.2,      # dimensionless log-space std
                    'n_samples': 50000,
                    'source': 'WISE diameter measurements'
                },
                'albedo': {
                    'mean': 0.14,    # Geometric albedo
                    'std': 0.12,
                    'n_samples': 25000,
                    'source': 'WISE albedo measurements'
                }
            }
        }
    
    def _load_artificial_objects_database(self) -> List[Dict[str, Any]]:
        """Load confirmed artificial objects for validation."""
        
        # Known artificial objects in space:
        artificial_objects = [
            {
                'name': 'Tesla Roadster (2018-017A)',
                'orbital_elements': {
                    'a': 1.325,  # AU (JPL Horizons)
                    'e': 0.256,  # (JPL Horizons)
                    'i': 1.077   # degrees (JPL Horizons)
                },
                'physical_data': {
                    'mass_estimate': 1350,    # kg — Tesla + Falcon Heavy upper stage
                    'diameter': 12,           # m — approximate vehicle dimensions
                    'absolute_magnitude': 28.0,
                    'radar_cross_section': 15.0
                },
                'launch_date': datetime(2018, 2, 6),
                'source': 'SpaceX Falcon Heavy demo mission',
                'confidence': 1.0  # Confirmed artificial
            },
            {
                'name': 'Deep Space Climate Observatory (DSCOVR)',
                'orbital_elements': {
                    'a': 1.01,   # AU (Earth-Sun L1 halo orbit)
                    'e': 0.01,   # Nearly circular
                    'i': 0.1     # degrees (very low inclination)
                },
                'physical_data': {
                    'mass_estimate': 570,     # kg — spacecraft mass
                    'diameter': 1.8,          # m — spacecraft dimensions
                    'absolute_magnitude': 25.5,
                    'radar_cross_section': 8.0
                },
                'launch_date': datetime(2015, 2, 11),
                'source': 'NOAA/NASA solar wind monitoring satellite',
                'confidence': 1.0  # Confirmed artificial
            }
            # Additional confirmed artificial objects would be added here
        ]
        
        return artificial_objects
        
    def _load_natural_objects_database(self) -> List[Dict[str, Any]]:
        """Load confirmed natural NEOs for validation."""
        
        # Known natural NEOs with high confidence:
        natural_objects = [
            {
                'name': '99942 Apophis',
                'orbital_elements': {
                    'a': 0.922,  # AU
                    'e': 0.191,
                    'i': 3.331   # degrees
                },
                'physical_data': {
                    'mass_estimate': 2.7e10,  # kg — estimated from size and density
                    'diameter': 370,          # m — radar measurements
                    'absolute_magnitude': 19.7,
                    'density_estimate': 3200, # kg/m³ — typical S-type
                    'radar_cross_section': 0.23
                },
                'discovery_date': datetime(2004, 6, 19),
                'source': 'LINEAR survey',
                'spectral_type': 'Sq',
                'confidence': 1.0
            },
            {
                'name': '101955 Bennu',
                'orbital_elements': {
                    'a': 1.126,  # AU
                    'e': 0.204,
                    'i': 6.035   # degrees
                },
                'physical_data': {
                    'mass_estimate': 7.8e10,  # kg — OSIRIS-REx measurement
                    'diameter': 492,          # m — high precision shape model
                    'absolute_magnitude': 20.9,
                    'density_estimate': 1190, # kg/m³ — rubble pile
                    'radar_cross_section': 0.054
                },
                'discovery_date': datetime(1999, 9, 11),
                'source': 'LINEAR survey',
                'spectral_type': 'B',
                'confidence': 1.0
            }
            # Additional confirmed natural objects would be added here
        ]
        
        return natural_objects
    
    def _calculate_orbital_anomaly_score(self, orbital_elements: Dict[str, float]) -> EvidenceSource:
        """Calculate orbital dynamics anomaly score using proper statistics."""
        
        a = orbital_elements.get('a', 0)
        e = orbital_elements.get('e', 0) 
        i = orbital_elements.get('i', 0)
        
        if a == 0:
            return EvidenceSource(
                EvidenceType.ORBITAL_DYNAMICS, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0
            )
        
        # Get population statistics
        orbital_stats = self.neo_population_stats['orbital_elements']
        
        # Calculate Z-scores for each parameter
        a_zscore = abs(a - orbital_stats['semi_major_axis']['mean']) / orbital_stats['semi_major_axis']['std']
        e_zscore = abs(e - orbital_stats['eccentricity']['mean']) / orbital_stats['eccentricity']['std']  
        i_zscore = abs(i - orbital_stats['inclination']['mean']) / orbital_stats['inclination']['std']
        
        # Special handling for very low inclinations (artificial signature)
        # Tesla Roadster has i=1.077°, Apophis has i=3.331°, mean natural NEO i=15.01°
        low_inclination_bonus = 0.0
        if i < 2.0:  # Very low inclination (< 2°) - stronger artificial signature
            # This is a strong signature of artificial objects
            inclination_anomaly = (orbital_stats['inclination']['mean'] - i) / orbital_stats['inclination']['std']
            low_inclination_bonus = min(inclination_anomaly * 0.3, 2.0)  # Reduced bonus, cap at 2-sigma
        elif i < 5.0:  # Moderately low inclination (2-5°) - weaker signature
            inclination_anomaly = (orbital_stats['inclination']['mean'] - i) / orbital_stats['inclination']['std']
            low_inclination_bonus = min(inclination_anomaly * 0.1, 0.5)  # Much smaller bonus
        
        # Combine Z-scores with artificial signature enhancement
        base_anomaly = np.sqrt(a_zscore**2 + e_zscore**2 + i_zscore**2)
        enhanced_anomaly = base_anomaly + low_inclination_bonus
        
        # Use enhanced anomaly as effect size
        effect_size = enhanced_anomaly
        
        # Calculate p-value based on enhanced anomaly
        # For multivariate case, use chi-square approximation
        chi2_stat = effect_size**2
        p_value = float(stats.chi2.sf(chi2_stat, 3))  # 3 degrees of freedom
        
        n = orbital_stats['semi_major_axis']['n_samples']  # Sample size
        
        # 95% confidence interval on the anomaly score (bootstrap estimate)
        ci_lower = max(0.0, effect_size - 1.96 * effect_size / np.sqrt(n))
        ci_upper = effect_size + 1.96 * effect_size / np.sqrt(n)
        
        # Data quality based on completeness of orbital elements
        quality = 1.0 if all(x > 0 for x in [a, e, i]) else 0.5
        
        return EvidenceSource(
            evidence_type=EvidenceType.ORBITAL_DYNAMICS,
            anomaly_score=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=n,
            p_value=p_value,
            effect_size=effect_size,
            quality_score=quality
        )
    
    def _calculate_physical_anomaly_score(self, physical_data: Dict[str, Any]) -> EvidenceSource:
        """Calculate physical properties anomaly score."""
        
        if not physical_data:
            return EvidenceSource(
                EvidenceType.PHYSICAL_PROPERTIES, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0,
                analyzed=False, data_available=False
            )

        physical_stats = self.neo_population_stats['physical_properties']
        anomaly_scores = []
        p_values = []
        
        # Absolute magnitude analysis
        if 'absolute_magnitude' in physical_data:
            h_mag = physical_data['absolute_magnitude']
            h_zscore = abs(h_mag - physical_stats['absolute_magnitude']['mean']) / physical_stats['absolute_magnitude']['std']
            anomaly_scores.append(h_zscore)
            p_values.append(float(2 * stats.norm.sf(h_zscore)))
        
        # Diameter analysis (if available)
        if 'diameter' in physical_data:
            diameter = physical_data['diameter']
            # Use log-normal distribution for diameter
            log_diameter = np.log(max(diameter, 0.001))  # Avoid log(0)
            d_zscore = abs(log_diameter - np.log(physical_stats['diameter']['mean'])) / physical_stats['diameter']['std']
            anomaly_scores.append(d_zscore)
            p_values.append(float(2 * stats.norm.sf(d_zscore)))
        
        # Mass analysis (if available)
        if 'mass_estimate' in physical_data:
            mass = physical_data['mass_estimate']
            # Expected mass from diameter and typical density
            if 'diameter' in physical_data:
                expected_density = 2500  # kg/m³ typical for rocky asteroids
                diameter_m = physical_data['diameter']
                volume = (4/3) * np.pi * (diameter_m/2)**3
                expected_mass = expected_density * volume
                
                mass_ratio = mass / expected_mass
                # Artificial objects typically much less massive than natural ones
                if mass_ratio < 0.1:  # Much lighter than expected
                    anomaly_scores.append(3.0)  # Strong artificial signature
                    p_values.append(0.001)
        
        if not anomaly_scores:
            return EvidenceSource(
                EvidenceType.PHYSICAL_PROPERTIES, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0
            )
        
        # Combined anomaly score
        combined_score = np.sqrt(np.sum(np.array(anomaly_scores)**2))
        
        # Combined p-value using Fisher's method
        if len(p_values) > 1:
            combined_p = finite_probability(fisher_log_p_value(p_values))
        else:
            combined_p = p_values[0]
        
        n_samples = physical_stats['absolute_magnitude']['n_samples']
        quality = len(anomaly_scores) / 3.0  # Quality based on data completeness
        
        return EvidenceSource(
            evidence_type=EvidenceType.PHYSICAL_PROPERTIES,
            anomaly_score=combined_score,
            confidence_interval=(max(0, combined_score - 0.5), combined_score + 0.5),
            sample_size=n_samples,
            p_value=combined_p,
            effect_size=combined_score,
            quality_score=quality
        )
    
    def _analyze_course_corrections(self, orbital_history: Optional[List[Dict[str, Any]]] = None) -> EvidenceSource:
        """Analyze orbital history for implausible course corrections indicating propulsion."""
        
        if not orbital_history or len(orbital_history) < 2:
            return EvidenceSource(
                EvidenceType.COURSE_CORRECTIONS, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0,
                analyzed=False, data_available=False
            )

        # Sort by observation date
        sorted_history = sorted(orbital_history, key=lambda x: x.get('epoch', 0))
        
        course_correction_signatures = []
        total_delta_v = 0.0
        
        for i in range(1, len(sorted_history)):
            prev_orbit = sorted_history[i-1]
            curr_orbit = sorted_history[i]
            
            # Calculate time between observations
            time_diff = curr_orbit.get('epoch', 0) - prev_orbit.get('epoch', 0)
            if time_diff <= 0:
                continue
                
            # Calculate orbital element changes
            da = curr_orbit.get('a', 0) - prev_orbit.get('a', 0)
            de = curr_orbit.get('e', 0) - prev_orbit.get('e', 0) 
            di = curr_orbit.get('i', 0) - prev_orbit.get('i', 0)
            
            # Estimate delta-V required for these changes
            # For small changes: delta_v ≈ n*a*sqrt(da²/a² + de² + (di*π/180)²)
            a_avg = (prev_orbit.get('a', 1) + curr_orbit.get('a', 1)) / 2
            n = np.sqrt(398600.4418 / a_avg**3) / 86400  # Mean motion (1/day)
            
            delta_v_estimate = n * a_avg * np.sqrt(
                (da/a_avg)**2 + de**2 + (di * np.pi / 180)**2
            ) * 1000  # Convert to m/s
            
            total_delta_v += delta_v_estimate
            
            # Check for implausible changes (natural perturbations are very small)
            if delta_v_estimate > 10.0:  # > 10 m/s change is very suspicious
                course_correction_signatures.append({
                    'delta_v_ms': delta_v_estimate,
                    'time_span_days': time_diff,
                    'orbital_changes': {'da': da, 'de': de, 'di': di},
                    'implausibility_score': min(delta_v_estimate / 10.0, 10.0)
                })
        
        if not course_correction_signatures:
            return EvidenceSource(
                EvidenceType.COURSE_CORRECTIONS, 0.0, (0.0, 0.0), len(orbital_history), 1.0, 0.0, 1.0
            )
        
        # Calculate anomaly score based on total delta-V and number of corrections
        max_correction = max(sig['implausibility_score'] for sig in course_correction_signatures)
        num_corrections = len(course_correction_signatures)
        
        # Natural objects should have < 1 m/s total delta-V over years
        # Artificial objects can have 10s-100s m/s delta-V from course corrections
        anomaly_score = min(max_correction * np.sqrt(num_corrections), 10.0)
        
        # Very conservative p-value calculation
        p_value = max(0.001, np.exp(-anomaly_score))
        
        return EvidenceSource(
            evidence_type=EvidenceType.COURSE_CORRECTIONS,
            anomaly_score=anomaly_score,
            confidence_interval=(max(0, anomaly_score - 1.0), anomaly_score + 1.0),
            sample_size=len(orbital_history),
            p_value=p_value,
            effect_size=anomaly_score,
            quality_score=min(1.0, len(orbital_history) / 5.0)  # Quality based on observation count
        )
    
    def _analyze_trajectory_patterns(self, orbital_elements: Dict[str, float], 
                                   close_approach_history: Optional[List[Dict[str, Any]]] = None) -> EvidenceSource:
        """Analyze for repeated exact trajectory passages (impossible naturally)."""
        
        if not close_approach_history or len(close_approach_history) < 2:
            return EvidenceSource(
                EvidenceType.TRAJECTORY_PATTERNS, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0,
                analyzed=False, data_available=False
            )

        # Look for suspiciously similar close approaches
        exact_pattern_signatures = []
        
        for i in range(len(close_approach_history)):
            for j in range(i + 1, len(close_approach_history)):
                approach_1 = close_approach_history[i]
                approach_2 = close_approach_history[j]
                
                # Compare approach parameters
                dist_1 = approach_1.get('distance_au', float('inf'))
                dist_2 = approach_2.get('distance_au', float('inf'))
                
                vel_1 = approach_1.get('velocity_km_s', 0)
                vel_2 = approach_2.get('velocity_km_s', 0)
                
                # Calculate similarity
                dist_similarity = 1.0 - abs(dist_1 - dist_2) / max(dist_1, dist_2, 0.001)
                vel_similarity = 1.0 - abs(vel_1 - vel_2) / max(vel_1, vel_2, 0.001)
                
                # 0.95 threshold: 5% variation in distance/velocity constitutes a repeating
                # close-approach pattern. At 0.05 AU approach, 5% = ±375,000 km — still
                # tighter than any known natural resonant orbit family scatter.
                # Old threshold 0.999 was ~400 km tolerance at 0.05 AU → effectively unreachable.
                if dist_similarity > 0.95 and vel_similarity > 0.95:
                    time_diff = abs(approach_2.get('epoch', 0) - approach_1.get('epoch', 0))
                    exact_pattern_signatures.append({
                        'distance_similarity': dist_similarity,
                        'velocity_similarity': vel_similarity,
                        'time_separation_years': time_diff / 365.25,
                        'implausibility_score': (dist_similarity + vel_similarity - 1.0) * 100
                    })
        
        if not exact_pattern_signatures:
            return EvidenceSource(
                EvidenceType.TRAJECTORY_PATTERNS, 0.0, (0.0, 0.0), len(close_approach_history), 1.0, 0.0, 1.0
            )
        
        # Very high anomaly score for exact repetitions (physically impossible)
        max_similarity = max(sig['implausibility_score'] for sig in exact_pattern_signatures)
        num_exact_patterns = len(exact_pattern_signatures)
        
        # Exact patterns are smoking gun evidence
        anomaly_score = min(5.0 + max_similarity * np.sqrt(num_exact_patterns), 15.0)
        
        # Extremely low p-value for exact repetitions
        p_value = max(1e-8, np.exp(-anomaly_score))
        
        return EvidenceSource(
            evidence_type=EvidenceType.TRAJECTORY_PATTERNS,
            anomaly_score=anomaly_score,
            confidence_interval=(anomaly_score - 0.5, anomaly_score + 0.5),
            sample_size=len(close_approach_history),
            p_value=p_value,
            effect_size=anomaly_score,
            quality_score=min(1.0, len(close_approach_history) / 3.0)
        )
    
    def _analyze_propulsion_signatures(self, orbital_elements: Dict[str, float],
                                     observation_data: Optional[Dict[str, Any]] = None) -> EvidenceSource:
        """Analyze for direct propulsion signatures in orbital behavior."""
        
        if not observation_data:
            return EvidenceSource(
                EvidenceType.PROPULSION_SIGNATURES, 0.0, (0.0, 0.0), 0, 1.0, 0.0, 0.0,
                analyzed=False, data_available=False
            )

        propulsion_indicators = []
        
        # Non-gravitational accelerations
        if 'non_gravitational_accel' in observation_data:
            ng_accel = observation_data['non_gravitational_accel']
            
            # Natural objects have minimal non-gravitational forces (Yarkovsky effect ~1e-14 m/s²)
            # Propulsion would be orders of magnitude larger
            if ng_accel > 1e-10:  # Suspicious acceleration
                propulsion_indicators.append({
                    'type': 'non_gravitational_acceleration',
                    'magnitude': ng_accel,
                    'implausibility_score': np.log10(ng_accel / 1e-14)
                })
        
        # Anomalous brightness changes (could indicate attitude control)
        if 'brightness_variations' in observation_data:
            brightness_data = observation_data['brightness_variations']
            
            # Look for regular, artificial patterns in brightness
            if isinstance(brightness_data, list) and len(brightness_data) > 10:
                # Check for periodic artificial-looking patterns
                brightness_values = [b.get('magnitude', 0) for b in brightness_data]
                brightness_std = np.std(brightness_values)
                
                # Artificial objects might have regular tumbling or attitude control signatures
                if brightness_std > 0.5:  # Large brightness variations
                    propulsion_indicators.append({
                        'type': 'brightness_control_signature',
                        'variation_magnitude': brightness_std,
                        'implausibility_score': min(brightness_std, 3.0)
                    })
        
        # Orbital period changes inconsistent with natural evolution
        if 'period_change_rate' in observation_data:
            period_change = observation_data['period_change_rate']  # seconds/year
            
            # Natural period changes are very small and predictable
            # Propulsion can cause rapid period changes
            if abs(period_change) > 60:  # > 1 minute per year is suspicious
                propulsion_indicators.append({
                    'type': 'anomalous_period_evolution',
                    'change_rate': period_change,
                    'implausibility_score': min(abs(period_change) / 60.0, 5.0)
                })
        
        if not propulsion_indicators:
            return EvidenceSource(
                EvidenceType.PROPULSION_SIGNATURES, 0.0, (0.0, 0.0), 1, 1.0, 0.0, 0.5
            )
        
        # Calculate combined propulsion signature score
        total_score = sum(ind['implausibility_score'] for ind in propulsion_indicators)
        max_individual = max(ind['implausibility_score'] for ind in propulsion_indicators)
        
        anomaly_score = min(max_individual + np.sqrt(total_score), 10.0)
        
        # Conservative p-value
        p_value = max(1e-6, np.exp(-anomaly_score * 0.5))
        
        return EvidenceSource(
            evidence_type=EvidenceType.PROPULSION_SIGNATURES,
            anomaly_score=anomaly_score,
            confidence_interval=(max(0, anomaly_score - 1.0), anomaly_score + 1.0),
            sample_size=len(propulsion_indicators),
            p_value=p_value,
            effect_size=anomaly_score,
            quality_score=len(propulsion_indicators) / 3.0
        )

    def _analyze_spectral_type(self, physical_data: Dict[str, Any]) -> Optional["EvidenceSource"]:
        """Analyze spectral type for artificial object signatures."""
        if not physical_data or "spectral_type" not in physical_data:
            return None

        spectral_type = physical_data.get("spectral_type")
        albedo = physical_data.get("albedo")

        score = 0.0

        # Unknown spectral type is a strong artificial signature
        if spectral_type not in _NATURAL_SPECTRAL_TYPES:
            score += 2.0

        # C-complex with high albedo is internally inconsistent
        c_complex = {"C", "Cb", "Cg", "Cgh", "Ch"}
        if spectral_type in c_complex and albedo is not None and albedo > 0.40:
            score += 1.5

        # Very high albedo for any body
        if albedo is not None and albedo > 0.55:
            score += 1.0

        if score <= 0.0:
            return None

        p_value = float(stats.chi2.sf(score**2, 1))

        return EvidenceSource(
            evidence_type=EvidenceType.SPECTRAL_ANALYSIS,
            anomaly_score=score,
            confidence_interval=(max(0.0, score - 0.5), score + 0.5),
            sample_size=1,
            p_value=p_value,
            effect_size=score,
            quality_score=1.0 if albedo is not None else 0.5,
            analyzed=True,
            data_available=True,
        )

    def analyze_neo_validated(self, orbital_elements: Dict[str, float],
                            physical_data: Dict[str, Any] = None,
                            orbital_history: List[Dict[str, Any]] = None,
                            close_approach_history: List[Dict[str, Any]] = None,
                            observation_data: Dict[str, Any] = None) -> Sigma5DetectionResult:
        """Compatibility facade; the historic name grants no assurance."""
        from ..analysis.screening import ScreeningAnalysisService
        assessment = ScreeningAnalysisService(detector=self).analyze_raw(
            orbital_elements.get('designation', 'UNSPECIFIED'), orbital_elements, physical_data,
            {'orbital_history': orbital_history, 'close_approach_history': close_approach_history,
             'observation_data': observation_data},
        )
        payload = assessment.to_payload()
        sources = [EvidenceSource(
            evidence_type=EvidenceType(e['type']), anomaly_score=e['anomaly_score'],
            confidence_interval=tuple(e['confidence_interval']), sample_size=e['sample_size'],
            p_value=None, effect_size=e['effect_size'], quality_score=e['quality_score'],
            analyzed=e['analyzed'], data_available=e['data_available'],
        ) for e in assessment.evidence_sources]
        return Sigma5DetectionResult(
            is_artificial=None, sigma_confidence=None, bayesian_probability=None,
            evidence_sources=sources, combined_p_value=None, false_discovery_rate=None,
            validation_metrics=None,
            analysis_metadata={**payload['analysis_metadata'], 'assessment': payload},
            screening_score=assessment.screening_score,
        )

    def _bayesian_evidence_fusion(self, evidence_sources):
        """Deprecated: assumed likelihood ratios cannot establish origin probability."""
        raise NotImplementedError(
            "Uncalibrated Bayesian fusion withdrawn; use ScreeningAnalysisService "
            "for ranking and explicit unavailable probability.")

    def _perform_cross_validation(self):
        """Deprecated: evaluating constructor examples is not independent validation."""
        raise NotImplementedError(
            "Constructor toy validation withdrawn; independent frozen evaluation required.")

    @staticmethod
    def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
        """Wilson score 95% confidence interval for a proportion k/n."""
        if n == 0:
            return (0.0, 1.0)
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half_width = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return (max(0.0, centre - half_width), min(1.0, centre + half_width))

    def get_validation_report(self) -> Dict[str, Any]:
        """Report installed validation status, not constructor-example performance.

        Assigning legacy validation_results cannot install a calibrated procedure.
        Independent evaluation and artifact verification remain required.
        """
        return {'status': 'no_independent_validation', 'validation_performed': False,
                'calibration_status': 'unavailable', 'five_sigma_claim_permitted': False,
                'performance_metrics': None, 'false_discovery_rate': None}
