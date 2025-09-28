"""
Detection Manager - Unified Interface for All Artificial NEO Detectors

This manager provides a single entry point for all detection systems,
automatically handles data format compatibility, and provides fallback
detection if primary systems fail.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Union
from enum import Enum

from ..interfaces.detection import (
    ArtificialNEODetector, EnhancedDetector, MultiModalDetector,
    DetectionResult, OrbitalElementsNormalizer
)

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Available detector types."""
    BASIC = "basic"
    CORRECTED = "corrected"
    MULTIMODAL = "multimodal"
    PRODUCTION = "production"
    VALIDATED = "validated"  # New scientifically validated detector
    AUTO = "auto"  # Automatically select best detector


class DetectionManager:
    """
    Manages all artificial NEO detection systems with unified interface.
    
    Provides automatic detector selection, data format compatibility,
    and fallback mechanisms for robust detection capabilities.
    """
    
    def __init__(self, preferred_detector: DetectorType = DetectorType.AUTO):
        self.logger = logging.getLogger(__name__)
        self.preferred_detector = preferred_detector
        self._detectors = {}
        self._load_available_detectors()
    
    def _load_available_detectors(self):
        """Load all available detectors with error handling."""
        detector_configs = [
            {
                'type': DetectorType.VALIDATED,
                'class_path': 'aneos_core.detection.validated_sigma5_artificial_neo_detector',
                'class_name': 'ValidatedSigma5ArtificialNEODetector',
                'priority': 0,  # Highest priority - scientifically validated
                'wrapper': self._wrap_validated_detector
            },
            {
                'type': DetectorType.MULTIMODAL,
                'class_path': 'aneos_core.detection.multimodal_sigma5_artificial_neo_detector',
                'class_name': 'MultiModalSigma5ArtificialNEODetector',
                'priority': 1,  # Second priority - experimental multimodal
                'wrapper': self._wrap_multimodal_detector
            },
            {
                'type': DetectorType.PRODUCTION,
                'class_path': 'aneos_core.detection.production_artificial_neo_detector',
                'class_name': 'ProductionArtificialNEODetector',
                'priority': 2,  # Second priority - production calibrated
                'wrapper': self._wrap_production_detector
            },
            {
                'type': DetectorType.CORRECTED,
                'class_path': 'aneos_core.detection.corrected_sigma5_artificial_neo_detector',
                'class_name': 'CorrectedSigma5ArtificialNEODetector',
                'priority': 3,  # Third priority - corrected version
                'wrapper': self._wrap_corrected_detector
            },
            {
                'type': DetectorType.BASIC,
                'class_path': 'aneos_core.detection.sigma5_artificial_neo_detector',
                'class_name': 'Sigma5ArtificialNEODetector',
                'priority': 4,  # Lowest priority - basic implementation
                'wrapper': self._wrap_basic_detector
            }
        ]
        
        for config in sorted(detector_configs, key=lambda x: x['priority']):
            try:
                module = __import__(config['class_path'], fromlist=[config['class_name']])
                detector_class = getattr(module, config['class_name'])
                detector_instance = detector_class()
                
                # Wrap the detector to provide unified interface
                wrapped_detector = config['wrapper'](detector_instance)
                self._detectors[config['type']] = wrapped_detector
                
                self.logger.info(f"Loaded detector: {config['type'].value}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load detector {config['type'].value}: {e}")
    
    def _wrap_validated_detector(self, detector):
        """Wrap ValidatedSigma5ArtificialNEODetector to unified interface."""
        class ValidatedWrapper(MultiModalDetector):
            def __init__(self, wrapped_detector):
                self.detector = wrapped_detector
            
            def analyze_neo(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
                
                # Normalize orbital elements
                normalized_elements = self.preprocess_orbital_elements(orbital_elements)
                
                # Call validated detector
                result = self.detector.analyze_neo_validated(
                    orbital_elements=normalized_elements,
                    physical_data=physical_data or {}
                )
                
                # Convert to unified format
                return DetectionResult(
                    is_artificial=result.is_artificial,
                    confidence=result.bayesian_probability,
                    sigma_level=result.sigma_confidence,
                    artificial_probability=result.bayesian_probability,
                    classification=self._map_validated_classification(result),
                    analysis=result.analysis_metadata,
                    risk_factors=[e.evidence_type.value for e in result.evidence_sources],
                    metadata={
                        'detector_type': 'validated',
                        'evidence_count': len(result.evidence_sources),
                        'combined_p_value': result.combined_p_value,
                        'false_discovery_rate': result.false_discovery_rate,
                        'validation_available': result.validation_metrics is not None
                    }
                )
            
            def _map_validated_classification(self, result):
                """Map validated result to classification categories."""
                if result.is_artificial and result.sigma_confidence >= 5.0:
                    return "🛸 ARTIFICIAL (VALIDATED σ≥5)"
                elif result.sigma_confidence >= 3.0:
                    return "⚠️ SUSPICIOUS (σ≥3)"
                elif result.sigma_confidence >= 2.0:
                    return "❓ EDGE CASE (σ≥2)"
                else:
                    return "🌍 NATURAL"
            
            def analyze_orbital_dynamics(self, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
                normalized = self.preprocess_orbital_elements(orbital_elements)
                evidence = self.detector._calculate_orbital_anomaly_score(normalized)
                return {
                    'anomaly_score': evidence.effect_size,
                    'p_value': evidence.p_value,
                    'confidence_interval': evidence.confidence_interval,
                    'quality_score': evidence.quality_score
                }
            
            def analyze_physical_properties(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
                """Analyze physical properties for anomalies."""
                evidence = self.detector._calculate_physical_anomaly_score(physical_data)
                return {
                    'anomaly_score': evidence.effect_size,
                    'p_value': evidence.p_value,
                    'confidence_interval': evidence.confidence_interval,
                    'quality_score': evidence.quality_score
                }
            
            def get_detector_info(self) -> Dict[str, Any]:
                """Get detector information and capabilities."""
                validation_report = self.detector.get_validation_report()
                return {
                    'name': 'Validated Sigma 5 Artificial NEO Detector',
                    'version': '1.0',
                    'type': 'validated',
                    'capabilities': ['orbital_analysis', 'physical_analysis', 'bayesian_fusion'],
                    'validation_status': validation_report.get('validation_performed', False),
                    'ground_truth_samples': validation_report.get('sample_sizes', {}),
                    'performance_metrics': validation_report.get('performance_metrics', {}),
                    'scientific_rigor': 'peer_review_ready'
                }
            
            def validate_input_data(self, orbital_elements: Dict[str, Any], 
                                  physical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                """Validate input data quality and completeness."""
                issues = []
                quality_score = 1.0
                
                # Check orbital elements
                required_orbital = ['a', 'e', 'i']
                missing_orbital = [param for param in required_orbital if param not in orbital_elements or orbital_elements[param] == 0]
                
                if missing_orbital:
                    issues.append(f"Missing orbital elements: {missing_orbital}")
                    quality_score *= 0.7
                
                # Check physical data
                if physical_data:
                    recommended_physical = ['mass_estimate', 'diameter', 'absolute_magnitude']
                    available_physical = [param for param in recommended_physical if param in physical_data]
                    
                    if len(available_physical) < 2:
                        issues.append("Limited physical data available")
                        quality_score *= 0.8
                else:
                    issues.append("No physical data provided")
                    quality_score *= 0.6
                
                return {
                    'valid': len(issues) == 0 or quality_score > 0.5,
                    'quality_score': quality_score,
                    'issues': issues,
                    'recommendations': [
                        "Provide complete orbital elements (a, e, i)",
                        "Include physical properties (mass, diameter, magnitude)",
                        "Add spectral or radar data if available"
                    ]
                }
            
        return ValidatedWrapper(detector)
    
    def _wrap_multimodal_detector(self, detector):
        """Wrap MultiModalSigma5ArtificialNEODetector to unified interface."""
        class MultiModalWrapper(MultiModalDetector):
            def __init__(self, wrapped_detector):
                self.detector = wrapped_detector
            
            def analyze_neo(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
                
                # Normalize orbital elements
                normalized_elements = self.preprocess_orbital_elements(orbital_elements)
                
                # Call original detector with correct parameter names
                observation_date = None
                if additional_data:
                    observation_date = additional_data.get('observation_date')
                
                result = self.detector.analyze_neo_multimodal(
                    orbital_elements=normalized_elements,
                    physical_data=physical_data or {},
                    observation_date=observation_date
                )
                
                # Convert to unified format
                return DetectionResult(
                    is_artificial=result.is_artificial,
                    confidence=result.confidence,
                    sigma_level=result.sigma_level,
                    artificial_probability=result.statistical_certainty,
                    classification=self._map_multimodal_classification(result),
                    analysis=result.analysis,
                    risk_factors=result.evidence_sources,
                    metadata={
                        'detector_type': 'multimodal',
                        'fusion_method': result.fusion_method,
                        'individual_scores': result.individual_scores,
                        'false_positive_rate': result.false_positive_rate
                    }
                )
            
            def analyze_orbital_dynamics(self, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
                normalized = self.preprocess_orbital_elements(orbital_elements)
                return self.detector.analyze_orbital_dynamics(normalized)
            
            def analyze_physical_properties(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
                return self.detector.analyze_physical_properties(physical_data)
            
            def get_detector_info(self) -> Dict[str, Any]:
                return {
                    'name': 'Multi-Modal Sigma5 Detector',
                    'version': '1.0',
                    'capabilities': ['orbital', 'physical', 'temporal', 'statistical'],
                    'sigma_level': 5.0,
                    'confidence_threshold': self.detector.SIGMA_5_CERTAINTY
                }
            
            def validate_input_data(self, orbital_elements: Dict[str, Any], 
                                  physical_data: Optional[Dict[str, Any]] = None) -> bool:
                required_elements = ['a', 'e', 'i']
                normalized = self.preprocess_orbital_elements(orbital_elements)
                return all(elem in normalized and normalized[elem] is not None 
                          for elem in required_elements)
            
            def _map_multimodal_classification(self, result) -> str:
                if result.sigma_level >= 5.0:
                    return "artificial"
                elif result.statistical_certainty >= 0.95:
                    return "highly_suspicious"
                elif result.statistical_certainty >= 0.7:
                    return "suspicious"
                else:
                    return "natural"
        
        return MultiModalWrapper(detector)
    
    def _wrap_production_detector(self, detector):
        """Wrap ProductionArtificialNEODetector to unified interface."""
        class ProductionWrapper(EnhancedDetector):
            def __init__(self, wrapped_detector):
                self.detector = wrapped_detector
            
            def analyze_neo(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
                
                normalized_elements = self.preprocess_orbital_elements(orbital_elements)
                result = self.detector.analyze_neo(normalized_elements, physical_data)
                
                return DetectionResult(
                    is_artificial=result.is_artificial,
                    confidence=result.confidence,
                    artificial_probability=result.confidence if result.is_artificial else 1 - result.confidence,
                    classification=result.detection_certainty.lower(),
                    analysis=result.analysis,
                    metadata={
                        'detector_type': 'production',
                        'false_positive_risk': result.false_positive_risk
                    }
                )
            
            def get_detector_info(self) -> Dict[str, Any]:
                return {
                    'name': 'Production Calibrated Detector',
                    'version': '1.0',
                    'capabilities': ['orbital', 'physical'],
                    'optimized_for': 'low_false_positives'
                }
            
            def validate_input_data(self, orbital_elements: Dict[str, Any], 
                                  physical_data: Optional[Dict[str, Any]] = None) -> bool:
                required_elements = ['a', 'e', 'i']
                normalized = self.preprocess_orbital_elements(orbital_elements)
                return all(elem in normalized for elem in required_elements)
        
        return ProductionWrapper(detector)
    
    def _wrap_corrected_detector(self, detector):
        """Wrap CorrectedSigma5ArtificialNEODetector to unified interface."""
        class CorrectedWrapper(EnhancedDetector):
            def __init__(self, wrapped_detector):
                self.detector = wrapped_detector
            
            def analyze_neo(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
                
                normalized_elements = self.preprocess_orbital_elements(orbital_elements)
                result = self.detector.analyze_neo(normalized_elements, physical_data)
                
                return DetectionResult(
                    is_artificial=result.is_artificial,
                    confidence=result.confidence,
                    sigma_level=getattr(result, 'sigma_level', None),
                    analysis=result.analysis,
                    metadata={'detector_type': 'corrected'}
                )
            
            def get_detector_info(self) -> Dict[str, Any]:
                return {
                    'name': 'Corrected Sigma5 Detector',
                    'version': '1.1',
                    'capabilities': ['orbital', 'physical']
                }
            
            def validate_input_data(self, orbital_elements: Dict[str, Any], 
                                  physical_data: Optional[Dict[str, Any]] = None) -> bool:
                required_elements = ['a', 'e', 'i']
                normalized = self.preprocess_orbital_elements(orbital_elements)
                return all(elem in normalized for elem in required_elements)
        
        return CorrectedWrapper(detector)
    
    def _wrap_basic_detector(self, detector):
        """Wrap Sigma5ArtificialNEODetector to unified interface."""
        class BasicWrapper(ArtificialNEODetector):
            def __init__(self, wrapped_detector):
                self.detector = wrapped_detector
            
            def analyze_neo(self, orbital_elements: Dict[str, Any], 
                          physical_data: Optional[Dict[str, Any]] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
                
                normalized_elements = self.preprocess_orbital_elements(orbital_elements)
                result = self.detector.analyze_neo(normalized_elements, physical_data)
                
                return DetectionResult(
                    is_artificial=result.is_artificial,
                    confidence=result.confidence,
                    analysis=result.analysis,
                    metadata={'detector_type': 'basic'}
                )
        
        return BasicWrapper(detector)
    
    def get_available_detectors(self) -> List[DetectorType]:
        """Get list of successfully loaded detectors."""
        return list(self._detectors.keys())
    
    def analyze_neo(self, 
                   orbital_elements: Dict[str, Any],
                   physical_data: Optional[Dict[str, Any]] = None,
                   additional_data: Optional[Dict[str, Any]] = None,
                   detector_type: Optional[DetectorType] = None) -> DetectionResult:
        """
        Analyze NEO using specified or best available detector.
        
        Args:
            orbital_elements: Orbital parameters (any naming convention)
            physical_data: Physical properties
            additional_data: Additional context data
            detector_type: Specific detector to use (None for auto-selection)
            
        Returns:
            DetectionResult with unified format
        """
        # Select detector
        chosen_detector = detector_type or self.preferred_detector
        
        if chosen_detector == DetectorType.AUTO:
            chosen_detector = self._select_best_detector(orbital_elements, physical_data)
        
        if chosen_detector not in self._detectors:
            # Fallback to any available detector
            if self._detectors:
                chosen_detector = next(iter(self._detectors.keys()))
                self.logger.warning(f"Requested detector not available, using {chosen_detector.value}")
            else:
                raise RuntimeError("No detection systems available")
        
        detector = self._detectors[chosen_detector]
        
        try:
            # Validate input data if possible
            if hasattr(detector, 'validate_input_data'):
                if not detector.validate_input_data(orbital_elements, physical_data):
                    self.logger.warning("Input data validation failed, proceeding with analysis")
            
            result = detector.analyze_neo(orbital_elements, physical_data, additional_data)
            result.metadata['detector_used'] = chosen_detector.value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Detection failed with {chosen_detector.value}: {e}")
            
            # Try fallback detector
            if chosen_detector != DetectorType.BASIC and DetectorType.BASIC in self._detectors:
                self.logger.info("Attempting fallback to basic detector")
                return self.analyze_neo(orbital_elements, physical_data, additional_data, DetectorType.BASIC)
            
            raise
    
    def _select_best_detector(self, orbital_elements: Dict[str, Any], 
                            physical_data: Optional[Dict[str, Any]]) -> DetectorType:
        """Select the best detector based on available data."""
        # Always prefer validated detector if available (highest scientific rigor)
        if DetectorType.VALIDATED in self._detectors:
            return DetectorType.VALIDATED
        
        # Prefer multimodal if we have rich data
        if (DetectorType.MULTIMODAL in self._detectors and 
            physical_data and len(physical_data) > 2):
            return DetectorType.MULTIMODAL
        
        # Use production detector for robust analysis
        if DetectorType.PRODUCTION in self._detectors:
            return DetectorType.PRODUCTION
        
        # Fall back to corrected or basic
        for detector_type in [DetectorType.CORRECTED, DetectorType.BASIC]:
            if detector_type in self._detectors:
                return detector_type
        
        # Should not reach here if detectors are loaded
        raise RuntimeError("No suitable detector found")
    
    def get_detector_info(self, detector_type: DetectorType = None) -> Dict[str, Any]:
        """Get information about specified or all detectors."""
        if detector_type:
            if detector_type in self._detectors:
                detector = self._detectors[detector_type]
                if hasattr(detector, 'get_detector_info'):
                    return detector.get_detector_info()
                else:
                    return {'name': detector_type.value, 'capabilities': ['orbital']}
            else:
                raise ValueError(f"Detector {detector_type.value} not available")
        else:
            return {
                dt.value: (d.get_detector_info() if hasattr(d, 'get_detector_info') 
                          else {'name': dt.value})
                for dt, d in self._detectors.items()
            }


# Global detection manager instance
_detection_manager = None

def get_detection_manager(preferred_detector: DetectorType = DetectorType.AUTO) -> DetectionManager:
    """Get global detection manager instance."""
    global _detection_manager
    if _detection_manager is None:
        _detection_manager = DetectionManager(preferred_detector)
    return _detection_manager


# Convenience function for direct analysis
def analyze_neo(orbital_elements: Dict[str, Any],
               physical_data: Optional[Dict[str, Any]] = None,
               additional_data: Optional[Dict[str, Any]] = None,
               detector_type: Optional[DetectorType] = None) -> DetectionResult:
    """
    Convenience function for NEO analysis using unified detection system.
    
    Args:
        orbital_elements: Orbital parameters (any naming convention supported)
        physical_data: Physical properties (optional)
        additional_data: Additional context data (optional)
        detector_type: Specific detector to use (None for auto-selection)
        
    Returns:
        DetectionResult with standardized format
    """
    manager = get_detection_manager()
    return manager.analyze_neo(orbital_elements, physical_data, additional_data, detector_type)