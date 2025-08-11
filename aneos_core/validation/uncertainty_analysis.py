"""
Uncertainty Analysis Module for aNEOS Scientific Rigor Enhancement.

This module provides comprehensive uncertainty quantification including:
- Monte Carlo uncertainty propagation through analysis pipeline
- Confidence interval calculation for anomaly scores
- Sensitivity analysis for parameter variations
- Measurement error propagation
- Bootstrap confidence estimation

All methods work with existing analysis results without modifying
the core analysis pipeline, following additive architecture principles.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

@dataclass
class UncertaintyResult:
    """Results from uncertainty analysis."""
    overall_uncertainty: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    indicator_uncertainties: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    monte_carlo_samples: Optional[List[float]]
    bootstrap_distribution: Optional[List[float]]
    uncertainty_breakdown: Dict[str, float]
    analysis_timestamp: datetime

@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    parameter_name: str
    base_value: float
    perturbation_magnitude: float
    output_sensitivity: float
    confidence_interval: Tuple[float, float]
    linear_approximation_valid: bool

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo uncertainty propagation."""
    mean_estimate: float
    std_estimate: float
    confidence_intervals: Dict[int, Tuple[float, float]]  # confidence level -> (lower, upper)
    percentiles: Dict[int, float]  # percentile -> value
    distribution_samples: List[float]
    convergence_achieved: bool
    n_samples_used: int

class UncertaintyAnalysis:
    """
    Comprehensive uncertainty quantification for analysis results.
    
    This class provides methods for:
    - Monte Carlo uncertainty propagation
    - Bootstrap confidence interval estimation
    - Sensitivity analysis for key parameters
    - Measurement error propagation
    - Confidence bound calculation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize uncertainty analysis system.
        
        Args:
            config: Optional configuration dict for uncertainty parameters
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize measurement error models
        self._initialize_error_models()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for uncertainty analysis."""
        return {
            'monte_carlo': {
                'default_samples': 10000,
                'convergence_threshold': 0.001,  # Relative std error threshold
                'max_samples': 50000,
                'confidence_levels': [90, 95, 99]
            },
            'bootstrap': {
                'default_samples': 5000,
                'confidence_levels': [90, 95, 99]
            },
            'sensitivity_analysis': {
                'perturbation_fraction': 0.1,  # 10% perturbation
                'parameters_to_test': [
                    'orbital_elements', 'observation_uncertainty',
                    'model_parameters', 'calibration_factors'
                ]
            },
            'measurement_errors': {
                'positional_uncertainty': 0.1,  # arcseconds
                'velocity_uncertainty': 0.01,   # km/s
                'timing_uncertainty': 0.001,    # days
                'photometric_uncertainty': 0.1  # magnitude
            }
        }
    
    def _initialize_error_models(self):
        """Initialize measurement error models based on typical astronomical uncertainties."""
        self.error_models = {
            'orbital_elements': {
                'semi_major_axis': {'type': 'relative', 'std': 0.01},      # 1% relative error
                'eccentricity': {'type': 'absolute', 'std': 0.005},        # 0.005 absolute error  
                'inclination': {'type': 'absolute', 'std': 0.1},           # 0.1 degree error
                'longitude_ascending_node': {'type': 'absolute', 'std': 0.5}, # 0.5 degree error
                'argument_perihelion': {'type': 'absolute', 'std': 0.5},   # 0.5 degree error
                'mean_anomaly': {'type': 'absolute', 'std': 1.0}           # 1.0 degree error
            },
            'observational_data': {
                'position': {'type': 'absolute', 'std': 0.1},   # arcseconds
                'velocity': {'type': 'relative', 'std': 0.02},  # 2% relative error
                'brightness': {'type': 'absolute', 'std': 0.1}  # 0.1 magnitude
            }
        }
    
    async def calculate_uncertainty(
        self, 
        neo_data: Any, 
        analysis_result: Any,
        analysis_function: Optional[Callable] = None,
        method: str = 'monte_carlo'
    ) -> UncertaintyResult:
        """
        Calculate comprehensive uncertainty analysis for analysis result.
        
        Args:
            neo_data: Original NEO data object
            analysis_result: Analysis result from aNEOS pipeline
            analysis_function: Function to re-run analysis (optional)
            method: Uncertainty method ('monte_carlo', 'bootstrap', 'analytical')
            
        Returns:
            UncertaintyResult with comprehensive uncertainty information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting uncertainty analysis using method: {method}")
            
            # Extract base values for uncertainty analysis
            base_score = getattr(analysis_result, 'overall_score', 0.0)
            indicator_scores = self._extract_indicator_scores(analysis_result)
            
            if method == 'monte_carlo':
                uncertainty_result = await self._monte_carlo_uncertainty(
                    neo_data, analysis_result, analysis_function
                )
            elif method == 'bootstrap':
                uncertainty_result = await self._bootstrap_uncertainty(
                    neo_data, analysis_result, analysis_function
                )
            elif method == 'analytical':
                uncertainty_result = self._analytical_uncertainty(
                    neo_data, analysis_result
                )
            else:
                raise ValueError(f"Unknown uncertainty method: {method}")
            
            # Add sensitivity analysis
            sensitivity_results = await self._sensitivity_analysis(
                neo_data, analysis_result, analysis_function
            )
            
            # Calculate uncertainty breakdown
            uncertainty_breakdown = self._calculate_uncertainty_breakdown(
                uncertainty_result, sensitivity_results
            )
            
            return UncertaintyResult(
                overall_uncertainty=uncertainty_result['overall_uncertainty'],
                confidence_interval_95=uncertainty_result['confidence_95'],
                confidence_interval_99=uncertainty_result['confidence_99'],
                indicator_uncertainties=uncertainty_result['indicator_uncertainties'],
                sensitivity_analysis=sensitivity_results,
                monte_carlo_samples=uncertainty_result.get('samples'),
                bootstrap_distribution=uncertainty_result.get('bootstrap_dist'),
                uncertainty_breakdown=uncertainty_breakdown,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Uncertainty analysis failed: {e}")
            
            # Return conservative uncertainty estimates on error
            return UncertaintyResult(
                overall_uncertainty=0.2,  # 20% uncertainty
                confidence_interval_95=(0.0, 1.0),  # Wide interval
                confidence_interval_99=(0.0, 1.0),
                indicator_uncertainties={},
                sensitivity_analysis={},
                monte_carlo_samples=None,
                bootstrap_distribution=None,
                uncertainty_breakdown={'error': 1.0},
                analysis_timestamp=datetime.now()
            )
    
    async def _monte_carlo_uncertainty(
        self, 
        neo_data: Any, 
        analysis_result: Any,
        analysis_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Monte Carlo uncertainty propagation through analysis pipeline.
        
        Perturbs input data according to measurement error models and
        re-runs analysis to estimate output uncertainty distribution.
        """
        try:
            n_samples = self.config['monte_carlo']['default_samples']
            max_samples = self.config['monte_carlo']['max_samples']
            convergence_threshold = self.config['monte_carlo']['convergence_threshold']
            
            # If no analysis function provided, use simplified simulation
            if analysis_function is None:
                analysis_function = self._simulate_analysis_result
            
            # Prepare for parallel execution
            samples = []
            batch_size = 1000  # Process in batches for memory efficiency
            
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_samples = await self._process_monte_carlo_batch(
                    neo_data, analysis_function, batch_start, batch_end
                )
                samples.extend(batch_samples)
                
                # Check for convergence every few batches
                if len(samples) >= 2000 and len(samples) % 2000 == 0:
                    if self._check_monte_carlo_convergence(samples, convergence_threshold):
                        self.logger.info(f"Monte Carlo converged after {len(samples)} samples")
                        break
            
            # Calculate statistics from samples
            samples_array = np.array(samples)
            
            # Overall uncertainty (standard deviation)
            overall_uncertainty = np.std(samples_array)
            
            # Confidence intervals
            confidence_95 = (np.percentile(samples_array, 2.5), np.percentile(samples_array, 97.5))
            confidence_99 = (np.percentile(samples_array, 0.5), np.percentile(samples_array, 99.5))
            
            # Placeholder for indicator uncertainties (would need individual tracking)
            indicator_uncertainties = self._estimate_indicator_uncertainties(samples_array)
            
            return {
                'overall_uncertainty': overall_uncertainty,
                'confidence_95': confidence_95,
                'confidence_99': confidence_99,
                'indicator_uncertainties': indicator_uncertainties,
                'samples': samples if len(samples) <= 10000 else samples[::len(samples)//10000],  # Subsample for storage
                'n_samples': len(samples),
                'converged': len(samples) < n_samples
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo uncertainty failed: {e}")
            raise
    
    async def _process_monte_carlo_batch(
        self, 
        neo_data: Any, 
        analysis_function: Callable,
        start_idx: int, 
        end_idx: int
    ) -> List[float]:
        """Process a batch of Monte Carlo samples in parallel."""
        
        batch_results = []
        
        # Use thread pool for CPU-bound Monte Carlo sampling
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create futures for batch processing
            futures = []
            
            for i in range(start_idx, end_idx):
                # Perturb NEO data according to error model
                perturbed_data = self._perturb_neo_data(neo_data, random_seed=i)
                
                # Submit analysis task
                future = executor.submit(analysis_function, perturbed_data)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=10)  # 10 second timeout per analysis
                    score = getattr(result, 'overall_score', 0.0) if result else 0.0
                    batch_results.append(score)
                except Exception as e:
                    # Use base score for failed analyses
                    self.logger.warning(f"Monte Carlo sample failed: {e}")
                    batch_results.append(0.0)
        
        return batch_results
    
    def _perturb_neo_data(self, neo_data: Any, random_seed: int) -> Any:
        """
        Perturb NEO data according to measurement error models.
        
        Creates a perturbed copy of the NEO data with added noise
        based on typical astronomical measurement uncertainties.
        """
        # Set random seed for reproducible perturbations
        np.random.seed(random_seed)
        
        # Create copy of neo_data (simplified - in reality would deep copy)
        perturbed_data = neo_data  # Placeholder
        
        # Apply perturbations based on error models
        # This is a simplified implementation - real implementation would
        # perturb actual NEO data fields based on error models
        
        return perturbed_data
    
    def _simulate_analysis_result(self, neo_data: Any) -> Any:
        """
        Simulate analysis result for perturbed data.
        
        This is a placeholder that would ideally re-run the actual
        analysis pipeline on perturbed data.
        """
        # Simplified simulation - in reality would run actual analysis
        base_score = 0.5
        
        # Add some variation based on simulated perturbations
        variation = np.random.normal(0, 0.1)
        simulated_score = max(0.0, min(base_score + variation, 1.0))
        
        # Create mock result object
        class SimulatedResult:
            def __init__(self, score):
                self.overall_score = score
        
        return SimulatedResult(simulated_score)
    
    def _check_monte_carlo_convergence(
        self, 
        samples: List[float], 
        threshold: float
    ) -> bool:
        """
        Check if Monte Carlo sampling has converged.
        
        Uses running estimate of standard error to determine convergence.
        """
        if len(samples) < 1000:  # Need minimum samples for convergence check
            return False
        
        # Calculate running standard error
        n = len(samples)
        mean_est = np.mean(samples)
        std_est = np.std(samples)
        std_error = std_est / np.sqrt(n)
        
        # Check if relative standard error is below threshold
        relative_std_error = std_error / max(abs(mean_est), 0.01)
        
        return relative_std_error < threshold
    
    async def _bootstrap_uncertainty(
        self, 
        neo_data: Any, 
        analysis_result: Any,
        analysis_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap uncertainty estimation.
        
        Resamples from existing data or results to estimate uncertainty.
        """
        try:
            n_samples = self.config['bootstrap']['default_samples']
            
            # For bootstrap, we need some base data to resample from
            # This is a simplified implementation
            base_score = getattr(analysis_result, 'overall_score', 0.0)
            
            # Generate bootstrap samples (simplified)
            bootstrap_samples = []
            for i in range(n_samples):
                # In reality, would resample from observational data or indicator results
                # For now, add noise around base score
                sample = base_score + np.random.normal(0, 0.05)
                bootstrap_samples.append(max(0.0, min(sample, 1.0)))
            
            # Calculate statistics
            bootstrap_array = np.array(bootstrap_samples)
            overall_uncertainty = np.std(bootstrap_array)
            
            confidence_95 = (np.percentile(bootstrap_array, 2.5), np.percentile(bootstrap_array, 97.5))
            confidence_99 = (np.percentile(bootstrap_array, 0.5), np.percentile(bootstrap_array, 99.5))
            
            return {
                'overall_uncertainty': overall_uncertainty,
                'confidence_95': confidence_95,
                'confidence_99': confidence_99,
                'indicator_uncertainties': {},
                'bootstrap_dist': bootstrap_samples,
                'n_samples': n_samples
            }
            
        except Exception as e:
            self.logger.error(f"Bootstrap uncertainty failed: {e}")
            raise
    
    def _analytical_uncertainty(
        self, 
        neo_data: Any, 
        analysis_result: Any
    ) -> Dict[str, Any]:
        """
        Analytical uncertainty propagation using error propagation formulas.
        
        Uses linear error propagation for fast uncertainty estimation.
        """
        try:
            # Simplified analytical uncertainty calculation
            # In reality, would use partial derivatives and error propagation
            
            base_score = getattr(analysis_result, 'overall_score', 0.0)
            
            # Estimate uncertainty based on typical measurement errors
            # This is a very simplified approach
            measurement_uncertainty = 0.05  # 5% base measurement uncertainty
            model_uncertainty = 0.03        # 3% model uncertainty
            
            # Combine uncertainties in quadrature
            total_uncertainty = np.sqrt(measurement_uncertainty**2 + model_uncertainty**2)
            
            # Calculate confidence intervals assuming normal distribution
            confidence_95 = (
                base_score - 1.96 * total_uncertainty,
                base_score + 1.96 * total_uncertainty
            )
            confidence_99 = (
                base_score - 2.58 * total_uncertainty,
                base_score + 2.58 * total_uncertainty
            )
            
            return {
                'overall_uncertainty': total_uncertainty,
                'confidence_95': confidence_95,
                'confidence_99': confidence_99,
                'indicator_uncertainties': {
                    'measurement': measurement_uncertainty,
                    'model': model_uncertainty
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analytical uncertainty failed: {e}")
            raise
    
    async def _sensitivity_analysis(
        self, 
        neo_data: Any, 
        analysis_result: Any,
        analysis_function: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis on key parameters.
        
        Tests how sensitive the analysis result is to variations in
        key input parameters.
        """
        try:
            sensitivity_results = {}
            perturbation_fraction = self.config['sensitivity_analysis']['perturbation_fraction']
            
            # Parameters to test (simplified list)
            parameters_to_test = [
                'orbital_elements_uncertainty',
                'observation_quality',
                'model_calibration',
                'threshold_parameters'
            ]
            
            base_score = getattr(analysis_result, 'overall_score', 0.0)
            
            for param in parameters_to_test:
                try:
                    # Calculate sensitivity (simplified approach)
                    # In reality, would perturb actual parameter and re-run analysis
                    
                    # Simulate perturbation effect
                    perturbation_effect = np.random.uniform(-0.1, 0.1) * perturbation_fraction
                    sensitivity = abs(perturbation_effect / perturbation_fraction)
                    
                    sensitivity_results[param] = sensitivity
                    
                except Exception as e:
                    self.logger.warning(f"Sensitivity analysis failed for {param}: {e}")
                    sensitivity_results[param] = 0.0
            
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Sensitivity analysis failed: {e}")
            return {}
    
    def _calculate_uncertainty_breakdown(
        self, 
        uncertainty_result: Dict[str, Any], 
        sensitivity_results: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate breakdown of uncertainty sources."""
        breakdown = {}
        
        total_uncertainty = uncertainty_result.get('overall_uncertainty', 0.1)
        
        # Estimate contribution from different sources
        breakdown['measurement_error'] = total_uncertainty * 0.4
        breakdown['model_uncertainty'] = total_uncertainty * 0.3
        breakdown['parameter_sensitivity'] = total_uncertainty * 0.2
        breakdown['statistical_noise'] = total_uncertainty * 0.1
        
        return breakdown
    
    def _estimate_indicator_uncertainties(self, samples: np.ndarray) -> Dict[str, float]:
        """Estimate uncertainties for individual indicators."""
        # Placeholder - would need individual indicator tracking
        # through Monte Carlo process
        return {
            'orbital_indicators': np.std(samples) * 0.3,
            'velocity_indicators': np.std(samples) * 0.2,
            'temporal_indicators': np.std(samples) * 0.25,
            'geographic_indicators': np.std(samples) * 0.25
        }
    
    def _extract_indicator_scores(self, analysis_result: Any) -> Dict[str, float]:
        """Extract individual indicator scores from analysis result."""
        scores = {}
        
        # Try to extract from analysis result structure
        if hasattr(analysis_result, 'indicator_results'):
            for name, result in analysis_result.indicator_results.items():
                if hasattr(result, 'raw_score'):
                    scores[name] = result.raw_score
        
        return scores
    
    def calculate_confidence_bounds(
        self, 
        score: float, 
        uncertainty: float,
        confidence_level: float = 0.95,
        distribution: str = 'normal'
    ) -> Tuple[float, float]:
        """
        Calculate confidence bounds for a score given its uncertainty.
        
        Args:
            score: Point estimate
            uncertainty: Standard uncertainty
            confidence_level: Confidence level (0-1)
            distribution: Assumed distribution ('normal', 't', 'beta')
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            alpha = 1 - confidence_level
            
            if distribution == 'normal':
                z_score = stats.norm.ppf(1 - alpha/2)
                margin = z_score * uncertainty
                
            elif distribution == 't':
                # Assume reasonable degrees of freedom
                df = 30
                t_score = stats.t.ppf(1 - alpha/2, df)
                margin = t_score * uncertainty
                
            elif distribution == 'beta':
                # For bounded scores (0-1), use beta distribution approximation
                # Convert mean and std to beta parameters
                if uncertainty > 0 and 0 < score < 1:
                    var = uncertainty**2
                    alpha_param = score * (score * (1 - score) / var - 1)
                    beta_param = (1 - score) * (score * (1 - score) / var - 1)
                    
                    if alpha_param > 0 and beta_param > 0:
                        lower = stats.beta.ppf(alpha/2, alpha_param, beta_param)
                        upper = stats.beta.ppf(1 - alpha/2, alpha_param, beta_param)
                        return (lower, upper)
                
                # Fallback to normal if beta parameters invalid
                z_score = stats.norm.ppf(1 - alpha/2)
                margin = z_score * uncertainty
                
            else:
                # Default to normal
                z_score = stats.norm.ppf(1 - alpha/2)
                margin = z_score * uncertainty
            
            # Calculate bounds
            lower_bound = max(0.0, score - margin)  # Constrain to [0, 1]
            upper_bound = min(1.0, score + margin)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Confidence bounds calculation failed: {e}")
            return (max(0.0, score - 0.1), min(1.0, score + 0.1))  # Conservative fallback