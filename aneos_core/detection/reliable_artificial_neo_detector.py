"""
Reliable Artificial NEO Detector - Focused on proven indicators.

This module implements a reliable artificial NEO detection system based on
well-validated indicators that can distinguish genuine human-made objects
from natural NEOs with high confidence and low false positive rates.

Focus Areas:
1. Heliocentric trajectories from Earth (major indicator)
2. Launch-favorable orbital inclinations
3. Transfer orbit characteristics
4. Physical size/brightness inconsistencies  
5. Temporal correlation with known launches

The goal is reliable detection with proven indicators rather than
comprehensive analysis with uncertain results.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReliableArtificialResult:
    """Result from reliable artificial NEO detection."""
    is_artificial: bool
    confidence: float
    primary_evidence: str
    contributing_factors: List[str]
    reliability_score: float
    processing_notes: str

class ReliableArtificialNEODetector:
    """
    Reliable artificial NEO detector focused on proven indicators.
    
    This detector uses a conservative approach with well-validated
    indicators to minimize false positives while catching genuine
    artificial NEOs with high confidence.
    """
    
    def __init__(self):
        """Initialize reliable detector with validated thresholds."""
        self.logger = logging.getLogger(__name__)
        
        # Validated detection thresholds (refined to reduce false positives)
        self.thresholds = {
            'heliocentric_min_a': 1.3,         # AU - more conservative, clearly artificial
            'launch_inclination_max': 25.0,    # degrees - more restrictive for launches
            'transfer_eccentricity_min': 0.25, # Higher minimum for transfer orbits
            'disposal_eccentricity_min': 0.5,  # Higher threshold for disposal orbits
            'confidence_threshold': 0.7,       # Higher threshold to reduce false positives
            'max_natural_size': 1000           # meters - objects larger than this likely natural
        }
        
        # Evidence weights (conservative approach)
        self.evidence_weights = {
            'heliocentric_trajectory': 0.8,    # Strongest indicator
            'launch_inclination': 0.3,         # Supporting evidence
            'transfer_orbit': 0.5,             # Moderate indicator
            'disposal_orbit': 0.4,             # Moderate indicator
            'size_brightness_anomaly': 0.2     # Supporting evidence
        }
    
    async def detect_artificial_neo(
        self, 
        neo_data: Any, 
        orbital_elements: Dict[str, float]
    ) -> ReliableArtificialResult:
        """
        Detect artificial NEO using reliable indicators.
        
        Args:
            neo_data: NEO data object
            orbital_elements: Orbital elements dictionary
            
        Returns:
            ReliableArtificialResult with detection analysis
        """
        try:
            evidence_scores = {}
            contributing_factors = []
            processing_notes = []
            
            # Extract orbital parameters
            a = orbital_elements.get('a', 1.0)      # Semi-major axis (AU)
            e = orbital_elements.get('e', 0.0)      # Eccentricity
            i = orbital_elements.get('i', 0.0)      # Inclination (degrees)
            
            processing_notes.append(f"Orbital: a={a:.3f}AU, e={e:.3f}, i={i:.1f}°")
            
            # PRE-FILTER: Natural Object Indicators (Reduces False Positives)
            natural_score = self._calculate_natural_probability(neo_data, orbital_elements)
            if natural_score > 0.7:
                processing_notes.append(f"❌ Strong natural object indicators (score: {natural_score:.3f})")
                return ReliableArtificialResult(
                    is_artificial=False,
                    confidence=0.0,
                    primary_evidence='natural_object_filter',
                    contributing_factors=['natural_indicators'],
                    reliability_score=1.0 - natural_score,
                    processing_notes=' | '.join(processing_notes)
                )
            
            # INDICATOR 1: Heliocentric Trajectory (Strongest Evidence)
            if a > self.thresholds['heliocentric_min_a']:
                # Object is clearly in heliocentric space
                heliocentric_score = min((a - 1.0) / 2.0, 1.0)  # Scale 0-1 over 1-3 AU
                evidence_scores['heliocentric_trajectory'] = heliocentric_score
                contributing_factors.append('heliocentric_orbit')
                processing_notes.append(f"✓ Heliocentric trajectory detected (a={a:.3f}AU)")
                
                # Bonus for typical disposal orbit distances
                if 1.2 < a < 2.5:  # Typical rocket stage disposal range
                    evidence_scores['heliocentric_trajectory'] *= 1.2
                    contributing_factors.append('disposal_distance')
                    processing_notes.append("✓ Typical disposal orbit distance")
            
            # INDICATOR 2: Launch-Favorable Inclination
            if i < self.thresholds['launch_inclination_max']:
                inclination_score = (self.thresholds['launch_inclination_max'] - i) / self.thresholds['launch_inclination_max']
                evidence_scores['launch_inclination'] = inclination_score
                contributing_factors.append('launch_inclination')
                processing_notes.append(f"✓ Launch-favorable inclination ({i:.1f}°)")
                
                # Special bonus for very low inclinations (equatorial launches)
                if i < 10:
                    evidence_scores['launch_inclination'] *= 1.3
                    contributing_factors.append('equatorial_launch')
                    processing_notes.append("✓ Equatorial launch signature")
            
            # INDICATOR 3: Transfer Orbit Characteristics
            if (1.0 < a < 3.0 and 
                e > self.thresholds['transfer_eccentricity_min']):
                
                transfer_score = 0.6  # Base score for transfer orbit
                
                # Enhance score based on eccentricity
                if e > 0.3:
                    transfer_score *= (1 + e * 0.5)
                
                evidence_scores['transfer_orbit'] = min(transfer_score, 1.0)
                contributing_factors.append('transfer_orbit')
                processing_notes.append(f"✓ Transfer orbit characteristics (e={e:.3f})")
            
            # INDICATOR 4: High Eccentricity Disposal Orbit
            if e > self.thresholds['disposal_eccentricity_min']:
                disposal_score = min(e, 1.0)
                evidence_scores['disposal_orbit'] = disposal_score
                contributing_factors.append('disposal_orbit')
                processing_notes.append(f"✓ High eccentricity disposal orbit (e={e:.3f})")
            
            # INDICATOR 5: Size/Brightness Anomaly (Supporting Evidence)
            size_brightness_score = self._analyze_size_brightness_anomaly(neo_data)
            if size_brightness_score > 0:
                evidence_scores['size_brightness_anomaly'] = size_brightness_score
                contributing_factors.append('size_brightness_anomaly')
                processing_notes.append("✓ Size/brightness anomaly detected")
            
            # Calculate overall confidence using weighted evidence
            total_confidence = 0.0
            for evidence_type, score in evidence_scores.items():
                weight = self.evidence_weights.get(evidence_type, 0.1)
                total_confidence += score * weight
                processing_notes.append(f"  {evidence_type}: {score:.3f} * {weight} = {score*weight:.3f}")
            
            # Cap confidence at 1.0
            total_confidence = min(total_confidence, 1.0)
            
            # Determine if artificial based on threshold
            is_artificial = total_confidence >= self.thresholds['confidence_threshold']
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability(evidence_scores, contributing_factors)
            
            # Determine primary evidence
            if evidence_scores:
                primary_evidence = max(evidence_scores.keys(), 
                                     key=lambda k: evidence_scores[k] * self.evidence_weights.get(k, 0))
            else:
                primary_evidence = 'no_evidence'
            
            processing_notes.append(f"Total confidence: {total_confidence:.3f}")
            processing_notes.append(f"Classification: {'ARTIFICIAL' if is_artificial else 'NATURAL'}")
            
            self.logger.info(
                f"Reliable detection: {'ARTIFICIAL' if is_artificial else 'NATURAL'} "
                f"(confidence: {total_confidence:.3f}, primary: {primary_evidence})"
            )
            
            return ReliableArtificialResult(
                is_artificial=is_artificial,
                confidence=total_confidence,
                primary_evidence=primary_evidence,
                contributing_factors=contributing_factors,
                reliability_score=reliability_score,
                processing_notes=' | '.join(processing_notes)
            )
            
        except Exception as e:
            self.logger.error(f"Reliable artificial NEO detection failed: {e}")
            return ReliableArtificialResult(
                is_artificial=False,
                confidence=0.0,
                primary_evidence='error',
                contributing_factors=['error'],
                reliability_score=0.0,
                processing_notes=f"Error: {str(e)}"
            )
    
    def _analyze_size_brightness_anomaly(self, neo_data: Any) -> float:
        """Analyze size/brightness relationship for artificial object signatures."""
        try:
            # Get physical properties
            diameter = getattr(neo_data, 'diameter', None)
            magnitude = getattr(neo_data, 'absolute_magnitude', None)
            
            if not (diameter and magnitude):
                return 0.0
            
            # Artificial objects often have high area-to-mass ratios
            # leading to brightness that doesn't match natural asteroids
            
            # Expected magnitude for rocky asteroid (simplified H-D relationship)
            # H = 5*log10(D_km) + albedo_term, typical albedo ~0.1
            expected_magnitude = 5 * np.log10(diameter / 1000) + 15.0
            
            magnitude_difference = abs(magnitude - expected_magnitude)
            
            anomaly_score = 0.0
            
            # Significantly brighter than expected (high albedo/area)
            if magnitude < expected_magnitude - 2:
                anomaly_score = 0.4
            elif magnitude < expected_magnitude - 1:
                anomaly_score = 0.2
            
            # Small size but observed (detection bias toward artificial)
            if diameter < 50:  # meters
                anomaly_score += 0.1
            
            return min(anomaly_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_reliability(
        self, 
        evidence_scores: Dict[str, float], 
        contributing_factors: List[str]
    ) -> float:
        """Calculate reliability of the detection."""
        try:
            reliability_factors = []
            
            # Number of independent evidence types
            num_evidence = len(evidence_scores)
            if num_evidence > 0:
                reliability_factors.append(min(num_evidence / 3.0, 1.0))
            
            # Strength of best evidence
            if evidence_scores:
                best_evidence = max(evidence_scores.values())
                reliability_factors.append(best_evidence)
            
            # Presence of primary indicators
            if 'heliocentric_orbit' in contributing_factors:
                reliability_factors.append(0.8)  # High reliability for heliocentric
            
            # Consistency check
            if len(reliability_factors) > 1:
                consistency = 1.0 - np.std(reliability_factors)
                reliability_factors.append(max(0.0, consistency))
            
            return np.mean(reliability_factors) if reliability_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_natural_probability(
        self, 
        neo_data: Any, 
        orbital_elements: Dict[str, float]
    ) -> float:
        """Calculate probability that object is natural (to reduce false positives)."""
        try:
            natural_indicators = []
            
            # SIZE INDICATOR: Large objects likely natural
            diameter = getattr(neo_data, 'diameter', 100)
            if diameter > self.thresholds['max_natural_size']:
                natural_indicators.append(0.9)  # Strong natural indicator
            elif diameter > 500:
                natural_indicators.append(0.6)  # Moderate natural indicator
            elif diameter > 100:
                natural_indicators.append(0.3)  # Weak natural indicator
            
            # ORBITAL INDICATORS: Natural asteroid patterns
            a = orbital_elements.get('a', 1.0)
            e = orbital_elements.get('e', 0.0)
            i = orbital_elements.get('i', 0.0)
            
            # High inclination suggests natural origin (not launch-favorable)
            if i > 45:
                natural_indicators.append(0.7)
            elif i > 30:
                natural_indicators.append(0.4)
            
            # Very eccentric orbits can be natural (comets, scattered disk objects)
            if e > 0.8:
                natural_indicators.append(0.6)
            
            # Interior orbits (Atira class) - natural population exists
            if a < 1.0:
                natural_indicators.append(0.5)
            
            # Traditional asteroid belt proximity
            if 2.0 < a < 3.5:
                natural_indicators.append(0.8)
            
            # If no natural indicators, return low probability
            if not natural_indicators:
                return 0.1
            
            # Return average of natural indicators
            return min(np.mean(natural_indicators), 1.0)
            
        except Exception:
            return 0.1  # Low natural probability on error