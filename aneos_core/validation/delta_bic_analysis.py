"""
Delta BIC Analysis for aNEOS Enhanced Validation Pipeline.

This module implements Bayesian Information Criterion (BIC) delta analysis
for model comparison in artificial NEO detection, specifically designed to
distinguish between natural and artificial object orbital characteristics.
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeltaBICResult:
    """Result from Delta BIC analysis."""
    delta_bic: float
    preferred_model: str  # 'artificial' or 'natural'
    model_confidence: float
    natural_model_bic: float
    artificial_model_bic: float
    model_parameters: Dict[str, Any]
    analysis_timestamp: datetime
    data_quality_score: float

@dataclass
class ModelComparisonResult:
    """Results from comparing natural vs artificial models."""
    natural_likelihood: float
    artificial_likelihood: float
    evidence_ratio: float
    model_weights: Dict[str, float]
    parameter_estimates: Dict[str, Dict[str, float]]

class DeltaBICAnalyzer:
    """
    Delta BIC analyzer for distinguishing artificial from natural NEOs.
    
    Uses Bayesian Information Criterion to compare goodness-of-fit between
    orbital models optimized for natural vs artificial objects.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Delta BIC analyzer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model selection thresholds
        self.strong_evidence_threshold = self.config.get('strong_evidence_threshold', 6.0)
        self.moderate_evidence_threshold = self.config.get('moderate_evidence_threshold', 2.0)
        self.weak_evidence_threshold = self.config.get('weak_evidence_threshold', 0.5)
        
        # Model parameters
        self.natural_model_params = self.config.get('natural_model_params', {
            'eccentricity_prior': {'mean': 0.2, 'std': 0.15},
            'inclination_prior': {'mean': 10.0, 'std': 15.0},
            'semimajor_axis_prior': {'mean': 1.5, 'std': 0.8}
        })
        
        self.artificial_model_params = self.config.get('artificial_model_params', {
            'eccentricity_prior': {'mean': 0.6, 'std': 0.25},
            'inclination_prior': {'mean': 25.0, 'std': 20.0},
            'semimajor_axis_prior': {'mean': 1.2, 'std': 1.0}
        })
    
    async def analyze_delta_bic(self, orbital_elements: Dict[str, float], 
                                observation_data: Optional[Dict[str, Any]] = None) -> DeltaBICResult:
        """
        Perform Delta BIC analysis on orbital elements.
        
        Args:
            orbital_elements: Dictionary containing orbital elements
            observation_data: Optional observation metadata
            
        Returns:
            DeltaBICResult with model comparison results
        """
        try:
            # Extract key orbital parameters
            eccentricity = orbital_elements.get('e', 0.0)
            inclination = orbital_elements.get('i', 0.0)
            semimajor_axis = orbital_elements.get('a', 1.0)
            
            # Calculate likelihoods for both models
            natural_likelihood = self._calculate_natural_likelihood(
                eccentricity, inclination, semimajor_axis
            )
            artificial_likelihood = self._calculate_artificial_likelihood(
                eccentricity, inclination, semimajor_axis
            )
            
            # Calculate BIC for both models
            natural_bic = self._calculate_bic(natural_likelihood, n_params=3, n_observations=1)
            artificial_bic = self._calculate_bic(artificial_likelihood, n_params=3, n_observations=1)
            
            # Calculate Delta BIC (artificial - natural)
            delta_bic = artificial_bic - natural_bic
            
            # Determine preferred model and confidence
            if delta_bic > self.strong_evidence_threshold:
                preferred_model = 'natural'
                confidence = min(0.95, 0.5 + (delta_bic / 20.0))
            elif delta_bic < -self.strong_evidence_threshold:
                preferred_model = 'artificial'
                confidence = min(0.95, 0.5 + (abs(delta_bic) / 20.0))
            elif abs(delta_bic) > self.moderate_evidence_threshold:
                preferred_model = 'natural' if delta_bic > 0 else 'artificial'
                confidence = 0.7 + (abs(delta_bic) / 40.0)
            else:
                preferred_model = 'uncertain'
                confidence = 0.5 + (abs(delta_bic) / 40.0)
            
            # Calculate data quality score
            data_quality = self._assess_data_quality(orbital_elements, observation_data)
            
            return DeltaBICResult(
                delta_bic=delta_bic,
                preferred_model=preferred_model,
                model_confidence=confidence,
                natural_model_bic=natural_bic,
                artificial_model_bic=artificial_bic,
                model_parameters={
                    'eccentricity': eccentricity,
                    'inclination': inclination,
                    'semimajor_axis': semimajor_axis,
                    'natural_likelihood': natural_likelihood,
                    'artificial_likelihood': artificial_likelihood
                },
                analysis_timestamp=datetime.now(),
                data_quality_score=data_quality
            )
            
        except Exception as e:
            self.logger.error(f"Delta BIC analysis failed: {e}")
            return DeltaBICResult(
                delta_bic=0.0,
                preferred_model='uncertain',
                model_confidence=0.5,
                natural_model_bic=0.0,
                artificial_model_bic=0.0,
                model_parameters={},
                analysis_timestamp=datetime.now(),
                data_quality_score=0.0
            )
    
    def _calculate_natural_likelihood(self, eccentricity: float, inclination: float, 
                                      semimajor_axis: float) -> float:
        """Calculate likelihood under natural NEO model."""
        # Gaussian likelihood for natural orbital distribution
        ecc_like = self._gaussian_likelihood(
            eccentricity, 
            self.natural_model_params['eccentricity_prior']['mean'],
            self.natural_model_params['eccentricity_prior']['std']
        )
        
        inc_like = self._gaussian_likelihood(
            inclination,
            self.natural_model_params['inclination_prior']['mean'],
            self.natural_model_params['inclination_prior']['std']
        )
        
        sma_like = self._gaussian_likelihood(
            semimajor_axis,
            self.natural_model_params['semimajor_axis_prior']['mean'],
            self.natural_model_params['semimajor_axis_prior']['std']
        )
        
        return ecc_like * inc_like * sma_like
    
    def _calculate_artificial_likelihood(self, eccentricity: float, inclination: float,
                                         semimajor_axis: float) -> float:
        """Calculate likelihood under artificial NEO model."""
        # Gaussian likelihood for artificial orbital distribution
        ecc_like = self._gaussian_likelihood(
            eccentricity,
            self.artificial_model_params['eccentricity_prior']['mean'],
            self.artificial_model_params['eccentricity_prior']['std']
        )
        
        inc_like = self._gaussian_likelihood(
            inclination,
            self.artificial_model_params['inclination_prior']['mean'],
            self.artificial_model_params['inclination_prior']['std']
        )
        
        sma_like = self._gaussian_likelihood(
            semimajor_axis,
            self.artificial_model_params['semimajor_axis_prior']['mean'],
            self.artificial_model_params['semimajor_axis_prior']['std']
        )
        
        return ecc_like * inc_like * sma_like
    
    def _gaussian_likelihood(self, value: float, mean: float, std: float) -> float:
        """Calculate Gaussian likelihood."""
        if std <= 0:
            return 1e-10
        
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((value - mean) / std) ** 2
        )
    
    def _calculate_bic(self, likelihood: float, n_params: int, n_observations: int) -> float:
        """Calculate Bayesian Information Criterion."""
        if likelihood <= 0:
            likelihood = 1e-10
        
        log_likelihood = np.log(likelihood)
        bic = -2 * log_likelihood + n_params * np.log(n_observations)
        return bic
    
    def _assess_data_quality(self, orbital_elements: Dict[str, float],
                             observation_data: Optional[Dict[str, Any]]) -> float:
        """Assess quality of input data for BIC analysis."""
        quality_score = 1.0
        
        # Check for missing critical orbital elements
        required_elements = ['e', 'i', 'a']
        missing_elements = [elem for elem in required_elements if elem not in orbital_elements]
        if missing_elements:
            quality_score *= 0.5
        
        # Check for reasonable orbital element ranges
        eccentricity = orbital_elements.get('e', 0.0)
        if eccentricity < 0 or eccentricity > 1:
            quality_score *= 0.7
        
        inclination = orbital_elements.get('i', 0.0)
        if inclination < 0 or inclination > 180:
            quality_score *= 0.7
        
        semimajor_axis = orbital_elements.get('a', 1.0)
        if semimajor_axis <= 0:
            quality_score *= 0.5
        
        # Factor in observation data quality if available
        if observation_data:
            obs_count = observation_data.get('observation_count', 1)
            if obs_count > 10:
                quality_score *= 1.1
            elif obs_count < 3:
                quality_score *= 0.8
        
        return min(1.0, quality_score)

async def enhance_stage3_with_delta_bic(stage3_result: Dict[str, Any],
                                        orbital_elements: Dict[str, float],
                                        observation_data: Optional[Dict[str, Any]] = None,
                                        config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhance Stage 3 physical plausibility analysis with Delta BIC results.
    
    Args:
        stage3_result: Results from Stage 3 analysis
        orbital_elements: Orbital elements for BIC analysis
        observation_data: Optional observation metadata
        config: Configuration parameters
        
    Returns:
        Enhanced Stage 3 results with Delta BIC analysis
    """
    try:
        analyzer = DeltaBICAnalyzer(config)
        bic_result = await analyzer.analyze_delta_bic(orbital_elements, observation_data)
        
        # Enhance the stage 3 result
        enhanced_result = stage3_result.copy()
        enhanced_result['delta_bic_analysis'] = {
            'delta_bic': bic_result.delta_bic,
            'preferred_model': bic_result.preferred_model,
            'model_confidence': bic_result.model_confidence,
            'natural_bic': bic_result.natural_model_bic,
            'artificial_bic': bic_result.artificial_model_bic,
            'data_quality': bic_result.data_quality_score
        }
        
        # Adjust confidence based on BIC analysis
        original_confidence = stage3_result.get('confidence', 0.5)
        bic_weight = 0.3  # Weight for BIC contribution
        
        if bic_result.preferred_model == 'artificial':
            enhanced_confidence = original_confidence * (1 - bic_weight) + bic_result.model_confidence * bic_weight
        elif bic_result.preferred_model == 'natural':
            enhanced_confidence = original_confidence * (1 - bic_weight) + (1 - bic_result.model_confidence) * bic_weight
        else:
            enhanced_confidence = original_confidence
        
        enhanced_result['enhanced_confidence'] = enhanced_confidence
        enhanced_result['bic_enhancement_applied'] = True
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Delta BIC enhancement failed: {e}")
        # Return original result if enhancement fails
        return stage3_result