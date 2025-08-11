"""
Spectral Outlier Analysis for aNEOS Material Detection Validation.

This module implements advanced spectroscopic analysis for detecting artificial
materials and spectral anomalies that may indicate human-made objects.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import mahalanobis
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class SpectralOutlierResult:
    """Results from spectral outlier analysis."""
    is_outlier: bool
    outlier_score: float
    confidence: float
    spectral_class: str
    material_classification: Dict[str, float]
    artificial_likelihood: float
    outlier_details: Dict[str, Any]
    processing_time_ms: float

class SpectralOutlierAnalyzer:
    """
    Advanced spectral analysis for detecting artificial materials and anomalies.
    
    Implements SMASS classification, PCA outlier detection, and artificial
    material identification for enhanced validation.
    """
    
    def __init__(self):
        """Initialize spectral analyzer with reference libraries."""
        self.natural_spectra_library = self._load_natural_spectra_library()
        self.artificial_spectra_library = self._load_artificial_spectra_library()
        self.pca_model = None
        self.scaler = StandardScaler()
        self._initialize_models()
        
    def _load_natural_spectra_library(self) -> Dict[str, np.ndarray]:
        """Load natural asteroid spectral library."""
        # Mock natural spectral library (in production would load from database)
        return {
            'C-type': np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),  # Dark carbonaceous
            'S-type': np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40]),  # Silicate
            'M-type': np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20]),  # Metallic
            'X-type': np.array([0.08, 0.10, 0.12, 0.14, 0.16, 0.18]),  # Complex
            'V-type': np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),  # Basaltic
        }
        
    def _load_artificial_spectra_library(self) -> Dict[str, np.ndarray]:
        """Load artificial material spectral library."""
        return {
            'aluminum': np.array([0.70, 0.72, 0.74, 0.76, 0.78, 0.80]),
            'titanium': np.array([0.45, 0.47, 0.49, 0.51, 0.53, 0.55]),
            'steel': np.array([0.25, 0.27, 0.29, 0.31, 0.33, 0.35]),
            'solar_panel': np.array([0.10, 0.15, 0.20, 0.85, 0.90, 0.95]),
            'thermal_coating': np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
        }
        
    def _initialize_models(self):
        """Initialize PCA and outlier detection models."""
        # Create combined training data from natural spectra
        training_data = []
        for spectrum in self.natural_spectra_library.values():
            # Add noise variations for robust model
            for i in range(10):
                noisy_spectrum = spectrum + np.random.normal(0, 0.01, len(spectrum))
                training_data.append(noisy_spectrum)
        
        if training_data:
            training_array = np.array(training_data)
            self.scaler.fit(training_array)
            scaled_data = self.scaler.transform(training_array)
            
            # Initialize PCA
            self.pca_model = PCA(n_components=min(4, scaled_data.shape[1]))
            self.pca_model.fit(scaled_data)
            
            # EMERGENCY: Suppressed initialization logging
    
    async def analyze_spectral_outliers(
        self,
        designation: str,
        spectrum_data: Optional[np.ndarray] = None,
        neo_data: Optional[Any] = None
    ) -> SpectralOutlierResult:
        """
        Perform comprehensive spectral outlier analysis.
        
        Args:
            designation: NEO designation
            spectrum_data: Spectral data array (wavelength, reflectance)
            neo_data: Additional NEO data for context
            
        Returns:
            SpectralOutlierResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Generate mock spectrum if none provided (in production would retrieve actual data)
            if spectrum_data is None:
                spectrum_data = self._generate_mock_spectrum(designation)
            
            # Ensure spectrum has correct format
            if len(spectrum_data.shape) == 2:
                reflectance = spectrum_data[:, 1]  # Use reflectance values
            else:
                reflectance = spectrum_data
                
            # Spectral classification
            spectral_class = self._classify_spectral_type(reflectance)
            
            # Outlier detection using multiple methods
            outlier_results = await self._detect_spectral_outliers(reflectance)
            
            # Material classification
            material_probs = self._classify_materials(reflectance)
            
            # Calculate artificial likelihood
            artificial_likelihood = self._calculate_artificial_likelihood(
                reflectance, material_probs, outlier_results
            )
            
            # Determine if this is an outlier
            is_outlier = (
                outlier_results['pca_outlier'] or 
                outlier_results['isolation_outlier'] or
                artificial_likelihood > 0.7
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SpectralOutlierResult(
                is_outlier=is_outlier,
                outlier_score=max(
                    outlier_results['pca_score'],
                    outlier_results['isolation_score'],
                    artificial_likelihood
                ),
                confidence=min(0.95, outlier_results['confidence']),
                spectral_class=spectral_class,
                material_classification=material_probs,
                artificial_likelihood=artificial_likelihood,
                outlier_details=outlier_results,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Spectral outlier analysis failed for {designation}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return safe fallback result
            return SpectralOutlierResult(
                is_outlier=False,
                outlier_score=0.0,
                confidence=0.0,
                spectral_class='unknown',
                material_classification={},
                artificial_likelihood=0.0,
                outlier_details={'error': str(e)},
                processing_time_ms=processing_time
            )
    
    def _generate_mock_spectrum(self, designation: str) -> np.ndarray:
        """Generate mock spectrum for testing (in production would query database)."""
        # Create wavelength array (0.4 to 2.5 micrometers)
        wavelengths = np.linspace(0.4, 2.5, 6)
        
        # Generate realistic asteroid spectrum based on designation hash
        np.random.seed(hash(designation) % 10000)
        base_reflectance = 0.05 + np.random.random() * 0.3
        
        # Add some spectral features
        reflectance = np.array([
            base_reflectance,
            base_reflectance * 1.1,
            base_reflectance * 1.2,
            base_reflectance * 1.15,
            base_reflectance * 1.1,
            base_reflectance * 1.05
        ])
        
        # Add noise
        reflectance += np.random.normal(0, 0.01, len(reflectance))
        
        return np.column_stack([wavelengths, reflectance])
    
    def _classify_spectral_type(self, reflectance: np.ndarray) -> str:
        """Classify spectrum using Bus-DeMeo taxonomic system."""
        try:
            best_match = 'unknown'
            best_score = float('inf')
            
            for spec_type, reference_spectrum in self.natural_spectra_library.items():
                if len(reference_spectrum) == len(reflectance):
                    # Calculate spectral similarity (chi-squared)
                    chi2 = np.sum((reflectance - reference_spectrum)**2 / (reference_spectrum + 1e-10))
                    
                    if chi2 < best_score:
                        best_score = chi2
                        best_match = spec_type
            
            return best_match
            
        except Exception as e:
            logger.warning(f"Spectral classification failed: {e}")
            return 'unknown'
    
    async def _detect_spectral_outliers(self, reflectance: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple statistical methods."""
        try:
            results = {
                'pca_outlier': False,
                'pca_score': 0.0,
                'isolation_outlier': False,
                'isolation_score': 0.0,
                'mahalanobis_distance': 0.0,
                'confidence': 0.0
            }
            
            # PCA-based outlier detection
            if self.pca_model is not None:
                scaled_spectrum = self.scaler.transform([reflectance])
                pca_transformed = self.pca_model.transform(scaled_spectrum)
                pca_reconstructed = self.pca_model.inverse_transform(pca_transformed)
                reconstruction_error = np.mean((scaled_spectrum - pca_reconstructed)**2)
                
                # Threshold based on training data (95th percentile)
                pca_threshold = 0.05  # Would be calculated from training data
                results['pca_outlier'] = reconstruction_error > pca_threshold
                results['pca_score'] = min(1.0, reconstruction_error / pca_threshold)
            
            # Isolation Forest outlier detection
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                # Fit on natural spectra
                training_spectra = np.array(list(self.natural_spectra_library.values()))
                iso_forest.fit(training_spectra)
                
                outlier_prediction = iso_forest.predict([reflectance])[0]
                outlier_score = iso_forest.decision_function([reflectance])[0]
                
                results['isolation_outlier'] = outlier_prediction == -1
                results['isolation_score'] = max(0.0, -outlier_score)  # Convert to positive score
                
            except Exception as e:
                logger.debug(f"Isolation Forest failed: {e}")
            
            # Calculate overall confidence
            results['confidence'] = min(0.95, max(results['pca_score'], results['isolation_score']))
            
            return results
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return {
                'pca_outlier': False,
                'pca_score': 0.0,
                'isolation_outlier': False,
                'isolation_score': 0.0,
                'mahalanobis_distance': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_materials(self, reflectance: np.ndarray) -> Dict[str, float]:
        """Classify materials using spectral matching."""
        try:
            material_probs = {}
            
            # Calculate similarity to artificial materials
            for material, reference_spectrum in self.artificial_spectra_library.items():
                if len(reference_spectrum) == len(reflectance):
                    # Calculate spectral correlation
                    correlation = np.corrcoef(reflectance, reference_spectrum)[0, 1]
                    correlation = max(0.0, correlation)  # Ensure positive
                    material_probs[material] = correlation
            
            # Normalize probabilities
            total_prob = sum(material_probs.values())
            if total_prob > 0:
                material_probs = {k: v/total_prob for k, v in material_probs.items()}
            
            return material_probs
            
        except Exception as e:
            logger.warning(f"Material classification failed: {e}")
            return {}
    
    def _calculate_artificial_likelihood(
        self,
        reflectance: np.ndarray,
        material_probs: Dict[str, float],
        outlier_results: Dict[str, Any]
    ) -> float:
        """Calculate likelihood that spectrum indicates artificial object."""
        try:
            # High artificial material probability
            max_artificial_prob = max(material_probs.values()) if material_probs else 0.0
            
            # Strong outlier indicators
            outlier_strength = max(
                outlier_results.get('pca_score', 0.0),
                outlier_results.get('isolation_score', 0.0)
            )
            
            # Unusual spectral characteristics
            mean_reflectance = np.mean(reflectance)
            unusual_brightness = 1.0 if mean_reflectance > 0.7 else 0.0  # Very bright = artificial
            
            # Combine factors
            artificial_likelihood = (
                0.4 * max_artificial_prob +
                0.4 * outlier_strength +
                0.2 * unusual_brightness
            )
            
            return min(1.0, artificial_likelihood)
            
        except Exception as e:
            logger.warning(f"Artificial likelihood calculation failed: {e}")
            return 0.0
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get current analyzer status and capabilities."""
        return {
            'spectral_library_loaded': len(self.natural_spectra_library) > 0,
            'artificial_library_loaded': len(self.artificial_spectra_library) > 0,
            'pca_model_trained': self.pca_model is not None,
            'natural_classes': list(self.natural_spectra_library.keys()),
            'artificial_materials': list(self.artificial_spectra_library.keys()),
            'analyzer_version': '1.0.0'
        }

# Integration function for MultiStageValidator
def enhance_stage3_with_spectral_analysis(
    validator_instance,
    designation: str,
    neo_data: Any,
    stage_result: Any,
    spectral_analyzer: Optional[SpectralOutlierAnalyzer] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Enhance Stage 3 physical plausibility with spectral outlier analysis.
    
    This function integrates spectral analysis into the existing Stage 3
    without modifying the core validation logic.
    """
    enhanced_details = stage_result.details.copy()
    
    try:
        if spectral_analyzer is None:
            spectral_analyzer = SpectralOutlierAnalyzer()
        
        # Run spectral analysis
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            spectral_result = loop.run_until_complete(
                spectral_analyzer.analyze_spectral_outliers(designation, neo_data=neo_data)
            )
        except RuntimeError:
            # If no event loop, create one
            spectral_result = asyncio.run(
                spectral_analyzer.analyze_spectral_outliers(designation, neo_data=neo_data)
            )
        
        # Enhance stage result with spectral information
        enhanced_details['spectral_analysis'] = {
            'is_outlier': spectral_result.is_outlier,
            'outlier_score': spectral_result.outlier_score,
            'spectral_class': spectral_result.spectral_class,
            'artificial_likelihood': spectral_result.artificial_likelihood,
            'material_classification': spectral_result.material_classification
        }
        
        # Adjust stage score based on spectral analysis
        spectral_factor = 1.0
        if spectral_result.is_outlier:
            if spectral_result.artificial_likelihood > 0.7:
                spectral_factor = 0.3  # Strong artificial signature reduces natural likelihood
            elif spectral_result.artificial_likelihood > 0.5:
                spectral_factor = 0.6  # Moderate artificial signature
        
        # Update stage result
        enhanced_stage_result = type(stage_result)(
            stage_number=stage_result.stage_number,
            stage_name=stage_result.stage_name + " + Spectral",
            passed=stage_result.passed and not spectral_result.is_outlier,
            score=stage_result.score * spectral_factor,
            confidence=min(stage_result.confidence, spectral_result.confidence),
            false_positive_reduction=min(0.95, stage_result.false_positive_reduction + 0.05),
            details=enhanced_details,
            processing_time_ms=stage_result.processing_time_ms + spectral_result.processing_time_ms
        )
        
        spectral_metadata = {
            'spectral_outlier_result': spectral_result,
            'spectral_enhancement_applied': True
        }
        
        logger.info(f"Stage 3 enhanced with spectral analysis for {designation}")
        return enhanced_stage_result, spectral_metadata
        
    except Exception as e:
        logger.error(f"Spectral enhancement failed for {designation}: {e}")
        enhanced_details['spectral_analysis_error'] = str(e)
        
        # Return original stage result with error info
        enhanced_stage_result = type(stage_result)(
            stage_number=stage_result.stage_number,
            stage_name=stage_result.stage_name,
            passed=stage_result.passed,
            score=stage_result.score,
            confidence=stage_result.confidence,
            false_positive_reduction=stage_result.false_positive_reduction,
            details=enhanced_details,
            processing_time_ms=stage_result.processing_time_ms
        )
        
        return enhanced_stage_result, {'spectral_enhancement_applied': False, 'error': str(e)}