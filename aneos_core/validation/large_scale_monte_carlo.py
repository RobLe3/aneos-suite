#!/usr/bin/env python3
"""
Large-Scale Monte Carlo False Positive Validation

This module executes comprehensive Monte Carlo simulation to validate false positive
rates for the corrected sigma 5 artificial NEO detection system. Tests with 1M+
synthetic natural NEO objects to empirically measure false positive rate and compare
against theoretical 5.7×10⁻⁷ sigma 5 claim.

Addresses critical requirement:
- Empirical false positive rate ≤5.7×10⁻⁷ (sigma 4.9+ equivalent)
- Statistical validation of sigma 5 claims with large sample sizes
- Performance analysis and confidence interval calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import stats
import asyncio
import multiprocessing as mp
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os

# Add project root to path
sys.path.append('/Users/roble/Documents/Python/claude_flow/aneos-project')

from aneos_core.detection.corrected_sigma5_artificial_neo_detector import CorrectedSigma5ArtificialNEODetector
from aneos_core.detection.sigma5_artificial_neo_detector import Sigma5ArtificialNEODetector

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloValidationResult:
    """Results from large-scale Monte Carlo false positive validation."""
    total_samples: int
    false_positives: int
    empirical_false_positive_rate: float
    theoretical_false_positive_rate: float
    sigma_level_equivalent: float
    confidence_interval_95: Tuple[float, float]
    processing_time_seconds: float
    detector_type: str
    parameter_source: str
    validation_timestamp: datetime
    statistical_significance_test: Dict[str, Any]
    
@dataclass
class DetectorPerformanceComparison:
    """Comparison of detector performance across different configurations."""
    original_detector_results: MonteCarloValidationResult
    corrected_detector_results: MonteCarloValidationResult
    performance_improvement: Dict[str, Any]
    recommendation: str

class LargeScaleMonteCarloValidator:
    """Large-scale Monte Carlo validator for artificial NEO detection systems."""
    
    def __init__(self, cache_dir: str = "monte_carlo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Theoretical sigma 5 false positive rate
        self.SIGMA_5_THEORETICAL_FP_RATE = 5.7e-7
        self.SIGMA_5_THRESHOLD = 5.0
        
        # Monte Carlo simulation parameters
        self.chunk_size = 10000  # Process in chunks for memory efficiency
        self.n_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    def generate_synthetic_natural_neos(self, count: int, seed: int = None) -> List[Dict[str, float]]:
        """Generate large population of synthetic natural NEOs based on empirical distributions."""
        
        if seed:
            np.random.seed(seed)
        
        logger.info(f"Generating {count} synthetic natural NEOs")
        
        # Use empirically validated parameters from literature
        synthetic_neos = []
        
        # Semi-major axis distribution (AU) - realistic NEO population
        # Composite distribution: Atira (2%), Aten (8%), Apollo (60%), Amor (30%)
        
        n_atira = int(0.02 * count)
        n_aten = int(0.08 * count) 
        n_apollo = int(0.60 * count)
        n_amor = count - n_atira - n_aten - n_apollo
        
        # Generate orbital elements for each population
        for i in range(count):
            if i < n_atira:
                # Atira group
                a = np.random.uniform(0.7, 1.0)
                e = np.random.beta(1.2, 2.0) * 0.6
                i = np.random.rayleigh(8.0)
            elif i < n_atira + n_aten:
                # Aten group  
                a = np.random.uniform(0.8, 1.0)
                e = np.random.beta(1.5, 1.5) * 0.8
                i = np.random.rayleigh(10.0)
            elif i < n_atira + n_aten + n_apollo:
                # Apollo group (majority)
                a = np.random.lognormal(np.log(1.5), 0.4)
                a = np.clip(a, 1.0, 4.0)
                e = np.random.beta(1.3, 1.2) * 0.95
                i = np.random.rayleigh(12.0)
            else:
                # Amor group
                a = np.random.lognormal(np.log(1.8), 0.3)
                a = np.clip(a, 1.2, 3.5)
                e = np.random.beta(1.4, 1.8) * 0.9
                i = np.random.rayleigh(15.0)
            
            # Clip inclination to reasonable values
            i = np.clip(i, 0, 45)
            
            # Additional orbital elements (not used in detection but included for completeness)
            omega = np.random.uniform(0, 360)
            Omega = np.random.uniform(0, 360) 
            M = np.random.uniform(0, 360)
            
            synthetic_neos.append({
                'a': a,
                'e': e,
                'i': i,
                'omega': omega,
                'Omega': Omega,
                'M': M
            })
        
        logger.info(f"Generated {len(synthetic_neos)} synthetic natural NEOs")
        return synthetic_neos
    
    def test_detector_chunk(self, args: Tuple[List[Dict], str, bool, int]) -> Tuple[int, int, List[float]]:
        """Test detector on chunk of synthetic NEOs (for multiprocessing)."""
        
        neo_chunk, detector_type, use_corrected_params, chunk_id = args
        
        # Initialize detector in worker process
        if detector_type == "corrected":
            detector = CorrectedSigma5ArtificialNEODetector()
        else:
            detector = Sigma5ArtificialNEODetector()
        
        false_positives = 0
        sigma_levels = []
        
        for neo in neo_chunk:
            try:
                if detector_type == "corrected":
                    result = detector.analyze_neo(neo, use_corrected_params=use_corrected_params)
                else:
                    result = detector.analyze_neo(neo)
                
                sigma_levels.append(result.sigma_level)
                
                # Count false positives (natural objects classified as artificial)
                if result.is_artificial:
                    false_positives += 1
                    
            except Exception as e:
                logger.warning(f"Error processing NEO in chunk {chunk_id}: {e}")
                continue
        
        return len(neo_chunk), false_positives, sigma_levels
    
    async def validate_detector_false_positive_rate(self, 
                                                   detector_type: str = "corrected",
                                                   use_corrected_params: bool = True,
                                                   sample_size: int = 1000000,
                                                   seed: int = 42) -> MonteCarloValidationResult:
        """Validate detector false positive rate with large-scale Monte Carlo simulation."""
        
        logger.info(f"Starting large-scale Monte Carlo validation")
        logger.info(f"Detector: {detector_type}, Corrected params: {use_corrected_params}")
        logger.info(f"Sample size: {sample_size:,}, Workers: {self.n_workers}")
        
        start_time = time.time()
        
        # Generate synthetic natural NEO population
        synthetic_neos = self.generate_synthetic_natural_neos(sample_size, seed)
        
        # Split into chunks for parallel processing
        chunks = []
        for i in range(0, len(synthetic_neos), self.chunk_size):
            chunk = synthetic_neos[i:i+self.chunk_size]
            chunks.append((chunk, detector_type, use_corrected_params, i//self.chunk_size))
        
        logger.info(f"Processing {len(chunks)} chunks with {self.n_workers} workers")
        
        # Process chunks in parallel
        total_processed = 0
        total_false_positives = 0
        all_sigma_levels = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(self.test_detector_chunk, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    processed, false_positives, sigma_levels = future.result()
                    total_processed += processed
                    total_false_positives += false_positives
                    all_sigma_levels.extend(sigma_levels)
                    
                    if chunk_id % 10 == 0:  # Progress update every 10 chunks
                        current_fp_rate = total_false_positives / total_processed if total_processed > 0 else 0.0
                        logger.info(f"Progress: {total_processed:,}/{sample_size:,} "
                                  f"({100*total_processed/sample_size:.1f}%), "
                                  f"FP rate: {current_fp_rate:.2e}")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate empirical false positive rate
        empirical_fp_rate = total_false_positives / total_processed
        
        # Calculate 95% confidence interval using Wilson score interval
        if total_false_positives == 0:
            # Special case: no false positives observed
            # Use rule of three: upper 95% CI ≈ 3/n
            ci_lower = 0.0
            ci_upper = 3.0 / total_processed
        else:
            # Wilson score interval for binomial proportion
            z = 1.96  # 95% confidence
            p = empirical_fp_rate
            n = total_processed
            
            denominator = 1 + z*z/n
            center = (p + z*z/(2*n)) / denominator
            margin = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
            
            ci_lower = max(0.0, center - margin)
            ci_upper = center + margin
        
        # Calculate equivalent sigma level
        if empirical_fp_rate > 0:
            # Two-tailed test: p = 2 * (1 - Φ(σ))
            # Solve for σ: σ = Φ⁻¹(1 - p/2)
            sigma_equivalent = stats.norm.ppf(1 - empirical_fp_rate/2)
        else:
            # Use upper confidence interval for conservative estimate
            sigma_equivalent = stats.norm.ppf(1 - ci_upper/2)
        
        # Statistical significance test vs theoretical rate
        # Binomial test: H₀: p = 5.7×10⁻⁷, H₁: p ≠ 5.7×10⁻⁷
        expected_fp_count = self.SIGMA_5_THEORETICAL_FP_RATE * total_processed
        if expected_fp_count < 5:  # Use exact binomial test
            try:
                # Try new scipy interface first
                result = stats.binomtest(total_false_positives, total_processed, self.SIGMA_5_THEORETICAL_FP_RATE)
                p_value = result.pvalue
            except AttributeError:
                # Fall back to deprecated binom_test if available
                p_value = 2 * stats.binom.pmf(total_false_positives, total_processed, self.SIGMA_5_THEORETICAL_FP_RATE)
        else:  # Use normal approximation
            z_stat = (total_false_positives - expected_fp_count) / np.sqrt(expected_fp_count * (1 - self.SIGMA_5_THEORETICAL_FP_RATE))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        statistical_test = {
            'test_type': 'binomial_test_vs_theoretical',
            'null_hypothesis': f'False positive rate = {self.SIGMA_5_THEORETICAL_FP_RATE:.2e}',
            'observed_fps': total_false_positives,
            'expected_fps': expected_fp_count,
            'p_value': p_value,
            'significant_difference': p_value < 0.05,
            'interpretation': 'Empirical rate significantly different from theoretical' if p_value < 0.05 else 'No significant difference from theoretical rate'
        }
        
        logger.info(f"Monte Carlo validation completed")
        logger.info(f"Processed: {total_processed:,} objects")
        logger.info(f"False positives: {total_false_positives}")
        logger.info(f"Empirical FP rate: {empirical_fp_rate:.2e}")
        logger.info(f"95% CI: [{ci_lower:.2e}, {ci_upper:.2e}]")
        logger.info(f"Equivalent sigma level: {sigma_equivalent:.2f}σ")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        
        return MonteCarloValidationResult(
            total_samples=total_processed,
            false_positives=total_false_positives,
            empirical_false_positive_rate=empirical_fp_rate,
            theoretical_false_positive_rate=self.SIGMA_5_THEORETICAL_FP_RATE,
            sigma_level_equivalent=sigma_equivalent,
            confidence_interval_95=(ci_lower, ci_upper),
            processing_time_seconds=processing_time,
            detector_type=detector_type,
            parameter_source="corrected" if use_corrected_params else "original",
            validation_timestamp=datetime.now(),
            statistical_significance_test=statistical_test
        )
    
    async def compare_detector_performance(self, sample_size: int = 100000) -> DetectorPerformanceComparison:
        """Compare performance between original and corrected detectors."""
        
        logger.info("Starting detector performance comparison")
        
        # Test original detector with hardcoded parameters
        logger.info("Testing original sigma5 detector...")
        original_results = await self.validate_detector_false_positive_rate(
            detector_type="original",
            use_corrected_params=False,
            sample_size=sample_size,
            seed=42
        )
        
        # Test corrected detector with empirical parameters
        logger.info("Testing corrected sigma5 detector...")
        corrected_results = await self.validate_detector_false_positive_rate(
            detector_type="corrected", 
            use_corrected_params=True,
            sample_size=sample_size,
            seed=42  # Same seed for fair comparison
        )
        
        # Calculate performance improvements
        fp_rate_improvement = ((original_results.empirical_false_positive_rate - corrected_results.empirical_false_positive_rate) 
                              / original_results.empirical_false_positive_rate * 100 
                              if original_results.empirical_false_positive_rate > 0 else 0.0)
        
        sigma_improvement = corrected_results.sigma_level_equivalent - original_results.sigma_level_equivalent
        
        # Determine recommendation
        if corrected_results.empirical_false_positive_rate <= self.SIGMA_5_THEORETICAL_FP_RATE * 2:  # Within 2x theoretical
            if corrected_results.sigma_level_equivalent >= 4.5:
                recommendation = "ACCEPTABLE - Corrected detector meets near-sigma 5 performance"
            else:
                recommendation = "MARGINAL - Additional improvements needed for sigma 5"
        else:
            recommendation = "INADEQUATE - False positive rate too high, major redesign needed"
        
        performance_improvement = {
            'false_positive_rate_change_percent': fp_rate_improvement,
            'sigma_level_improvement': sigma_improvement,
            'processing_time_ratio': corrected_results.processing_time_seconds / original_results.processing_time_seconds,
            'empirical_vs_theoretical_ratio_original': original_results.empirical_false_positive_rate / original_results.theoretical_false_positive_rate,
            'empirical_vs_theoretical_ratio_corrected': corrected_results.empirical_false_positive_rate / corrected_results.theoretical_false_positive_rate
        }
        
        return DetectorPerformanceComparison(
            original_detector_results=original_results,
            corrected_detector_results=corrected_results,
            performance_improvement=performance_improvement,
            recommendation=recommendation
        )
    
    def save_validation_results(self, results: Any, filename: str):
        """Save validation results to JSON file."""
        
        filepath = self.cache_dir / filename
        
        # Convert dataclass to dict for JSON serialization
        if isinstance(results, MonteCarloValidationResult):
            results_dict = {
                'total_samples': results.total_samples,
                'false_positives': results.false_positives,
                'empirical_false_positive_rate': results.empirical_false_positive_rate,
                'theoretical_false_positive_rate': results.theoretical_false_positive_rate,
                'sigma_level_equivalent': results.sigma_level_equivalent,
                'confidence_interval_95': results.confidence_interval_95,
                'processing_time_seconds': results.processing_time_seconds,
                'detector_type': results.detector_type,
                'parameter_source': results.parameter_source,
                'validation_timestamp': results.validation_timestamp.isoformat(),
                'statistical_significance_test': results.statistical_significance_test
            }
        elif isinstance(results, DetectorPerformanceComparison):
            results_dict = {
                'original_detector': {
                    'total_samples': results.original_detector_results.total_samples,
                    'false_positives': results.original_detector_results.false_positives,
                    'empirical_false_positive_rate': results.original_detector_results.empirical_false_positive_rate,
                    'sigma_level_equivalent': results.original_detector_results.sigma_level_equivalent,
                    'confidence_interval_95': results.original_detector_results.confidence_interval_95,
                    'processing_time_seconds': results.original_detector_results.processing_time_seconds
                },
                'corrected_detector': {
                    'total_samples': results.corrected_detector_results.total_samples,
                    'false_positives': results.corrected_detector_results.false_positives,
                    'empirical_false_positive_rate': results.corrected_detector_results.empirical_false_positive_rate,
                    'sigma_level_equivalent': results.corrected_detector_results.sigma_level_equivalent,
                    'confidence_interval_95': results.corrected_detector_results.confidence_interval_95,
                    'processing_time_seconds': results.corrected_detector_results.processing_time_seconds
                },
                'performance_improvement': results.performance_improvement,
                'recommendation': results.recommendation
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")

# Test and validation functions
async def test_monte_carlo_validation():
    """Test Monte Carlo validation system."""
    
    validator = LargeScaleMonteCarloValidator()
    
    print("🎲 LARGE-SCALE MONTE CARLO FALSE POSITIVE VALIDATION")
    print("=" * 70)
    print("Empirically measuring false positive rates for sigma 5 artificial NEO detection")
    print()
    
    # Start with smaller sample for testing, then scale up
    test_sample_size = 50000  # Reduced for testing
    
    print(f"Running detector performance comparison (n={test_sample_size:,})")
    print("This may take several minutes...")
    print()
    
    comparison = await validator.compare_detector_performance(test_sample_size)
    
    print("DETECTOR PERFORMANCE COMPARISON RESULTS")
    print("=" * 50)
    
    # Original detector results
    orig = comparison.original_detector_results
    print("Original Detector (Hardcoded Parameters):")
    print(f"  Sample size:           {orig.total_samples:,}")
    print(f"  False positives:       {orig.false_positives}")
    print(f"  Empirical FP rate:     {orig.empirical_false_positive_rate:.2e}")
    print(f"  95% CI:                [{orig.confidence_interval_95[0]:.2e}, {orig.confidence_interval_95[1]:.2e}]")
    print(f"  Equivalent sigma:      {orig.sigma_level_equivalent:.2f}σ")
    print(f"  Processing time:       {orig.processing_time_seconds:.1f}s")
    print()
    
    # Corrected detector results
    corr = comparison.corrected_detector_results
    print("Corrected Detector (Empirical Parameters):")
    print(f"  Sample size:           {corr.total_samples:,}")
    print(f"  False positives:       {corr.false_positives}")
    print(f"  Empirical FP rate:     {corr.empirical_false_positive_rate:.2e}")
    print(f"  95% CI:                [{corr.confidence_interval_95[0]:.2e}, {corr.confidence_interval_95[1]:.2e}]")
    print(f"  Equivalent sigma:      {corr.sigma_level_equivalent:.2f}σ")
    print(f"  Processing time:       {corr.processing_time_seconds:.1f}s")
    print()
    
    # Performance comparison
    perf = comparison.performance_improvement
    print("PERFORMANCE IMPROVEMENT:")
    print(f"  FP rate change:        {perf['false_positive_rate_change_percent']:+.1f}%")
    print(f"  Sigma level change:    {perf['sigma_level_improvement']:+.2f}σ")
    print(f"  Processing time ratio: {perf['processing_time_ratio']:.2f}x")
    print()
    
    # Theoretical comparison
    theoretical_fp_rate = validator.SIGMA_5_THEORETICAL_FP_RATE
    print("THEORETICAL SIGMA 5 COMPARISON:")
    print(f"  Theoretical FP rate:   {theoretical_fp_rate:.2e}")
    print(f"  Original vs theoretical: {perf['empirical_vs_theoretical_ratio_original']:.1f}x")
    print(f"  Corrected vs theoretical: {perf['empirical_vs_theoretical_ratio_corrected']:.1f}x")
    print()
    
    # Statistical significance
    orig_test = orig.statistical_significance_test
    corr_test = corr.statistical_significance_test
    
    print("STATISTICAL SIGNIFICANCE TESTS:")
    print(f"Original detector p-value: {orig_test['p_value']:.4f} ({'significant' if orig_test['significant_difference'] else 'not significant'})")
    print(f"Corrected detector p-value: {corr_test['p_value']:.4f} ({'significant' if corr_test['significant_difference'] else 'not significant'})")
    print()
    
    print("RECOMMENDATION:")
    print(f"  {comparison.recommendation}")
    print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validator.save_validation_results(comparison, f"monte_carlo_comparison_{timestamp}.json")
    
    return comparison

if __name__ == "__main__":
    asyncio.run(test_monte_carlo_validation())