#!/usr/bin/env python3
"""
Statistical utilities for aNEOS analysis.

Provides centralized statistical functions including sigma-to-p-value mapping,
multiple testing corrections, and other statistical calculations.
"""

import numpy as np
from scipy import stats
from typing import List, Union
import logging
import math
from scipy.special import ndtri_exp, gammaln, logsumexp

logger = logging.getLogger(__name__)

def sigma_to_p_value(sigma_level: float) -> float:
    """
    Convert sigma level to two-sided p-value.
    
    Args:
        sigma_level: Statistical significance in sigma units
        
    Returns:
        Two-sided p-value
        
    Examples:
        >>> sigma_to_p_value(2.0)  # ~4.6%
        0.045399929762484854
        >>> sigma_to_p_value(3.0)  # ~0.27%
        0.002699796063260207
        >>> sigma_to_p_value(5.0)  # ~5.7e-7
        5.733031438470705e-07
    """
    if not math.isfinite(sigma_level):
        raise ValueError("sigma_level must be finite")
    return float(2 * stats.norm.sf(abs(sigma_level)))

def sigma_to_confidence_level(sigma_level: float) -> float:
    """
    Convert sigma level to confidence level percentage.
    
    Args:
        sigma_level: Statistical significance in sigma units
        
    Returns:
        Confidence level as percentage (0-100)
        
    Examples:
        >>> sigma_to_confidence_level(2.8)  # ~99.5%
        99.49...
        >>> sigma_to_confidence_level(3.0)  # ~99.73%
        99.72...
    """
    p_value = sigma_to_p_value(sigma_level)
    return (1 - p_value) * 100

def p_value_to_sigma(p_value: float) -> float:
    """
    Convert two-sided p-value to sigma level.
    
    Args:
        p_value: Two-sided p-value
        
    Returns:
        Sigma level
    """
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise ValueError("p_value must be finite and in [0, 1]")
    if p_value == 0:
        return float('inf')
    if p_value >= 1:
        return 0.0
    
    # Convert two-sided p-value to sigma
    return log_p_value_to_sigma(math.log(p_value))


def log_p_value_to_sigma(log_p_value: float) -> float:
    """Two-sided Gaussian equivalent without subtracting tiny tails from one.

    This conversion does not validate the null model or establish artificial
    origin. Retain the log probability when the ordinary probability underflows.
    """
    if math.isnan(log_p_value) or log_p_value > 0:
        raise ValueError("log_p_value must be <= 0")
    return float(-ndtri_exp(log_p_value - math.log(2.0)))


def finite_probability(log_p_value: float) -> float:
    """Serialize a tail, flooring underflow to the smallest positive float.

    Callers must retain log_p_value and mark a floored value as a numerical
    bound, not an exact measured probability.
    """
    if math.isnan(log_p_value) or log_p_value > 0:
        raise ValueError("log_p_value must be <= 0")
    return max(float(np.nextafter(0.0, 1.0)), math.exp(log_p_value))


def fisher_log_p_value(p_values: List[float]) -> float:
    """Fisher tail under its independence assumption, using a stable finite sum.

    A zero input is interpreted as below floating-point resolution, never
    discarded. Callers must disclose that endpoint as a numerical bound.
    """
    if not p_values:
        return 0.0
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    half = -sum(math.log(max(p, float(np.nextafter(0.0, 1.0)))) for p in p_values)
    if half == 0:
        return 0.0
    j = np.arange(len(p_values), dtype=float)
    return min(0.0, float(-half + logsumexp(j * math.log(half) - gammaln(j + 1))))

def apply_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni multiple testing correction.
    
    Args:
        p_values: List of uncorrected p-values
        
    Returns:
        List of Bonferroni-corrected p-values
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []
    
    corrected = [min(p * n_tests, 1.0) for p in p_values]
    return corrected

def apply_benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: List of uncorrected p-values
        alpha: False discovery rate threshold
        
    Returns:
        List of booleans indicating significance after correction
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []
    
    # Sort p-values with original indices
    sorted_indices = sorted(range(n_tests), key=lambda i: p_values[i])
    sorted_p_values = [p_values[i] for i in sorted_indices]
    
    # Apply BH procedure
    significant = [False] * n_tests
    for i in reversed(range(n_tests)):
        threshold = alpha * (i + 1) / n_tests
        if sorted_p_values[i] <= threshold:
            # Mark all up to this point as significant
            for j in range(i + 1):
                original_idx = sorted_indices[j]
                significant[original_idx] = True
            break
    
    return significant

def get_sigma_interpretation(sigma_level: float) -> dict:
    """
    Get standardized interpretation of sigma level.
    
    Args:
        sigma_level: Statistical significance in sigma units
        
    Returns:
        Dictionary with interpretation details
    """
    p_value = sigma_to_p_value(sigma_level)
    confidence = sigma_to_confidence_level(sigma_level)
    
    if sigma_level >= 5.0:
        interpretation = "discovery"
        strength = "very_strong"
    elif sigma_level >= 3.0:
        interpretation = "evidence"  
        strength = "strong"
    elif sigma_level >= 2.0:
        interpretation = "indication"
        strength = "moderate"
    elif sigma_level >= 1.0:
        interpretation = "weak_indication"
        strength = "weak"
    else:
        interpretation = "not_significant"
        strength = "none"
    
    return {
        'sigma': sigma_level,
        'p_value': p_value,
        'confidence_percent': confidence,
        'interpretation': interpretation,
        'strength': strength,
        'description': f"{sigma_level:.1f}σ ({confidence:.1f}% confidence, p={p_value:.2e})"
    }

# Standard sigma levels for reference
STANDARD_SIGMA_LEVELS = {
    '1sigma': {'sigma': 1.0, 'p_value': 0.3173, 'confidence': 68.27},
    '2sigma': {'sigma': 2.0, 'p_value': 0.0455, 'confidence': 95.45},
    '3sigma': {'sigma': 3.0, 'p_value': 0.0027, 'confidence': 99.73},
    '4sigma': {'sigma': 4.0, 'p_value': 6.3e-5, 'confidence': 99.994},
    '5sigma': {'sigma': 5.0, 'p_value': 5.7e-7, 'confidence': 99.99994}
}

def validate_statistical_consistency(sigma_level: float, reported_significance: float, 
                                   tolerance: float = 0.05) -> bool:
    """
    Validate consistency between sigma level and reported significance.
    
    Args:
        sigma_level: Reported sigma level
        reported_significance: Reported statistical significance (as fraction or percentage)
        tolerance: Tolerance for mismatch (default 5%)
        
    Returns:
        True if consistent, False otherwise
    """
    expected_p = sigma_to_p_value(sigma_level)
    expected_confidence = sigma_to_confidence_level(sigma_level)
    
    # Check if reported as p-value
    if reported_significance <= 1.0 and reported_significance > 0:
        return abs(reported_significance - expected_p) / expected_p <= tolerance
    
    # Check if reported as confidence percentage  
    elif reported_significance > 1.0:
        return abs(reported_significance - expected_confidence) / expected_confidence <= tolerance
    
    return False
