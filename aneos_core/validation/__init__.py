"""
Scientific rigor validation module for aNEOS.

This module provides comprehensive validation capabilities including:
- Multi-stage validation pipeline for false positive prevention
- Statistical significance testing with multiple testing corrections
- Uncertainty quantification through Monte Carlo methods
- False positive prevention through external catalog cross-matching
- Advanced spectroscopic material analysis and outlier detection
- SMASS/Bus-DeMeo asteroid spectral classification
- Artificial material signature identification
- Principal Component Analysis based outlier detection

The validation system is designed as an additive layer that enhances
existing analysis results without modifying the core analysis pipeline.
"""

from .multi_stage_validator import MultiStageValidator, EnhancedAnalysisResult
from .statistical_testing import StatisticalTesting, StatisticalTestResult
from .false_positive_prevention import FalsePositivePrevention, FalsePositiveResult
from .uncertainty_analysis import UncertaintyAnalysis, UncertaintyResult, SensitivityResult, MonteCarloResult
from .human_hardware_analysis import HumanHardwareAnalyzer, HumanHardwareMatch, MaterialSignature, ConstellationPattern
from .spectral_outlier_analysis import (
    SpectralOutlierAnalyzer, 
    SpectralOutlierResult,
    enhance_stage3_with_spectral_analysis
)
from .radar_polarization_analysis import (
    RadarPolarizationAnalyzer,
    RadarPolarizationResult, 
    RadarSignature,
    StokesParameters,
    PolarizationRatios,
    SurfaceProperties,
    RadarDatabase,
    enhance_stage3_with_radar_polarization,
    create_radar_performance_tester,
    RadarPerformanceTester
)
from .gaia_astrometric_calibration import (
    GaiaAstrometricCalibrator,
    GaiaAstrometricResult,
    GaiaSource,
    ProperMotionAnalysis,
    ParallaxAnalysis,
    AstrometricPrecision,
    ArtificialObjectSignature,
    enhance_stage2_with_gaia_precision,
    create_gaia_performance_tester
)
from .physical_sanity import (
    PhysicalSanityValidator,
    PhysicalValidationResult,
    ValidationResult,
    validate_neo_analysis_output
)

__all__ = [
    'MultiStageValidator',
    'EnhancedAnalysisResult', 
    'StatisticalTesting',
    'StatisticalTestResult',
    'FalsePositivePrevention',
    'FalsePositiveResult',
    'UncertaintyAnalysis',
    'UncertaintyResult',
    'SensitivityResult',
    'MonteCarloResult',
    'HumanHardwareAnalyzer',
    'HumanHardwareMatch',
    'MaterialSignature',
    'ConstellationPattern',
    'SpectralOutlierAnalyzer',
    'SpectralOutlierResult',
    'enhance_stage3_with_spectral_analysis',
    'RadarPolarizationAnalyzer',
    'RadarPolarizationResult',
    'RadarSignature',
    'StokesParameters',
    'PolarizationRatios',
    'SurfaceProperties',
    'RadarDatabase',
    'enhance_stage3_with_radar_polarization',
    'create_radar_performance_tester',
    'RadarPerformanceTester',
    'GaiaAstrometricCalibrator',
    'GaiaAstrometricResult',
    'GaiaSource',
    'ProperMotionAnalysis',
    'ParallaxAnalysis',
    'AstrometricPrecision',
    'ArtificialObjectSignature',
    'enhance_stage2_with_gaia_precision',
    'create_gaia_performance_tester',
    'PhysicalSanityValidator',
    'PhysicalValidationResult',
    'ValidationResult',
    'validate_neo_analysis_output'
]

# Version information
__version__ = "0.7.0"
__author__ = "aNEOS Scientific Rigor Enhancement Team"
__description__ = "Scientific validation and false positive prevention for artificial NEO detection"
