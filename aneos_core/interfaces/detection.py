"""
Unified Detection Interface for aNEOS Suite

Provides standardized interfaces for all artificial NEO detection systems
to ensure compatibility and interoperability across the suite.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol, Union
from dataclasses import dataclass
from enum import Enum


class DetectionResult:
    """Standardized detection result format for all detectors."""
    
    def __init__(self, 
                 is_artificial: bool,
                 confidence: float,
                 sigma_level: Optional[float] = None,
                 artificial_probability: Optional[float] = None,
                 classification: Optional[str] = None,
                 analysis: Optional[Dict[str, Any]] = None,
                 risk_factors: Optional[list] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.is_artificial = is_artificial
        self.confidence = confidence
        self.sigma_level = sigma_level
        self.artificial_probability = artificial_probability or (confidence if is_artificial else 1 - confidence)
        self.classification = classification or self._determine_classification()
        self.analysis = analysis or {}
        self.risk_factors = risk_factors or []
        self.metadata = metadata or {}
    
    def _determine_classification(self) -> str:
        """Determine classification based on confidence and artificial probability."""
        if self.is_artificial and self.confidence >= 0.8:
            return "artificial"
        elif self.artificial_probability >= 0.7:
            return "highly_suspicious"
        elif self.artificial_probability >= 0.4:
            return "suspicious"
        else:
            return "natural"


class OrbitalElementsNormalizer:
    """Normalizes orbital elements between different naming conventions."""
    
    @staticmethod
    def normalize(orbital_elements: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert orbital elements to standardized format with dual compatibility.
        
        Supports both astronomical notation (a, e, i) and full names
        (semi_major_axis, eccentricity, inclination).
        """
        normalized = {}
        
        # Semi-major axis
        normalized['a'] = normalized['semi_major_axis'] = orbital_elements.get(
            'a', orbital_elements.get('semi_major_axis', 0.0)
        )
        
        # Eccentricity
        normalized['e'] = normalized['eccentricity'] = orbital_elements.get(
            'e', orbital_elements.get('eccentricity', 0.0)
        )
        
        # Inclination
        normalized['i'] = normalized['inclination'] = orbital_elements.get(
            'i', orbital_elements.get('inclination', 0.0)
        )
        
        # Optional elements
        optional_mappings = {
            ('q', 'perihelion_distance'): 'q',
            ('Q', 'aphelion_distance'): 'Q',
            ('om', 'ascending_node', 'longitude_of_ascending_node'): 'om',
            ('w', 'argument_of_perihelion'): 'w',
            ('ma', 'mean_anomaly'): 'ma',
            ('tp', 'time_of_perihelion'): 'tp',
            ('epoch',): 'epoch'
        }
        
        for keys, standard_key in optional_mappings.items():
            for key in keys:
                if key in orbital_elements:
                    normalized[standard_key] = orbital_elements[key]
                    # Also add all alternative names
                    for alt_key in keys:
                        normalized[alt_key] = orbital_elements[key]
                    break
        
        return normalized


class ArtificialNEODetector(ABC):
    """Abstract base class for all artificial NEO detectors."""
    
    @abstractmethod
    def analyze_neo(self, 
                   orbital_elements: Dict[str, Any], 
                   physical_data: Optional[Dict[str, Any]] = None,
                   additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """
        Analyze a NEO for artificial characteristics.
        
        Args:
            orbital_elements: Orbital parameters (supports both naming conventions)
            physical_data: Physical properties (size, mass, etc.)
            additional_data: Additional context data
            
        Returns:
            DetectionResult with standardized format
        """
        pass
    
    def preprocess_orbital_elements(self, orbital_elements: Dict[str, Any]) -> Dict[str, float]:
        """Preprocess orbital elements to ensure compatibility."""
        return OrbitalElementsNormalizer.normalize(orbital_elements)


class EnhancedDetector(ArtificialNEODetector):
    """Enhanced detector interface with additional capabilities."""
    
    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """Return detector metadata and capabilities."""
        pass
    
    @abstractmethod
    def validate_input_data(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input data quality and completeness."""
        pass


class MultiModalDetector(EnhancedDetector):
    """Multi-modal detector interface for advanced analysis."""
    
    @abstractmethod
    def analyze_orbital_dynamics(self, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze orbital dynamics component."""
        pass
    
    @abstractmethod
    def analyze_physical_properties(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze physical properties component."""
        pass
    
    def combine_analyses(self, analyses: Dict[str, Dict[str, Any]]) -> DetectionResult:
        """Combine multiple analysis results into final detection."""
        # Default implementation - should be overridden by specific detectors
        combined_confidence = sum(a.get('confidence', 0) for a in analyses.values()) / len(analyses)
        is_artificial = combined_confidence > 0.5
        
        return DetectionResult(
            is_artificial=is_artificial,
            confidence=combined_confidence,
            analysis=analyses,
            metadata={'fusion_method': 'simple_average'}
        )


# Legacy compatibility aliases
Sigma5DetectionResult = DetectionResult  # For backward compatibility
MultiModalDetectionResult = DetectionResult
ProductionDetectionResult = DetectionResult