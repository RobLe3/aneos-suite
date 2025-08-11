"""
Statistical Testing Framework for aNEOS Scientific Rigor Enhancement.

This module provides comprehensive statistical testing capabilities including:
- Formal hypothesis testing for indicator significance
- Multiple testing corrections (Benjamini-Hochberg, Bonferroni)
- Effect size calculations and confidence intervals
- Monte Carlo uncertainty propagation
- Power analysis for detection capability assessment

All functions are designed to work with existing analysis results without
modifying the core analysis pipeline.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing."""
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_statistic: float
    degrees_of_freedom: Optional[int] = None
    test_type: str = "z_test"
    alpha_level: float = 0.05

@dataclass
class MultipleTestingResult:
    """Results from multiple testing correction."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    significant_indicators: List[bool]
    correction_method: str
    family_wise_error_rate: float
    false_discovery_rate: float

@dataclass
class PowerAnalysisResult:
    """Results from statistical power analysis."""
    statistical_power: float
    effect_size: float
    sample_size: int
    alpha_level: float
    beta_error: float
    minimum_detectable_effect: float

class StatisticalTesting:
    """
    Comprehensive statistical testing framework for scientific rigor.
    
    This class provides methods for:
    - Hypothesis testing for individual indicators
    - Multiple testing corrections for family-wise error control
    - Effect size calculations and confidence intervals
    - Power analysis for detection capability assessment
    """
    
    def __init__(self, alpha_level: float = 0.05):
        """
        Initialize statistical testing framework.
        
        Args:
            alpha_level: Significance level for hypothesis testing (default: 0.05)
        """
        self.alpha_level = alpha_level
        self.logger = logging.getLogger(__name__)
        
    def formal_hypothesis_test(
        self, 
        indicator_score: float, 
        null_distribution: Dict[str, float],
        test_type: str = "z_test"
    ) -> StatisticalTestResult:
        """
        Perform formal hypothesis testing for indicator significance.
        
        Tests the null hypothesis that the indicator score comes from
        the null distribution (natural NEO population).
        
        Args:
            indicator_score: Observed indicator score
            null_distribution: Dict with 'mean' and 'std' of null distribution
            test_type: Type of statistical test ('z_test', 't_test', 'wilcoxon')
            
        Returns:
            StatisticalTestResult with test statistics and significance
        """
        try:
            if test_type == "z_test":
                return self._z_test(indicator_score, null_distribution)
            elif test_type == "t_test":
                return self._t_test(indicator_score, null_distribution)
            elif test_type == "wilcoxon":
                return self._wilcoxon_test(indicator_score, null_distribution)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
                
        except Exception as e:
            self.logger.error(f"Statistical test failed: {e}")
            # Return conservative result on error
            return StatisticalTestResult(
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(indicator_score, indicator_score),
                is_significant=False,
                test_statistic=0.0,
                test_type=test_type,
                alpha_level=self.alpha_level
            )
    
    def _z_test(
        self, 
        indicator_score: float, 
        null_distribution: Dict[str, float]
    ) -> StatisticalTestResult:
        """Perform Z-test for normally distributed indicator scores."""
        mean = null_distribution.get('mean', 0.0)
        std = null_distribution.get('std', 1.0)
        
        if std <= 0:
            std = 1.0  # Prevent division by zero
            
        # Calculate Z-score
        z_score = (indicator_score - mean) / std
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Effect size (standardized effect size)
        effect_size = abs(z_score)
        
        # Confidence interval for the score
        margin_error = stats.norm.ppf(1 - self.alpha_level/2) * std
        ci_lower = indicator_score - margin_error
        ci_upper = indicator_score + margin_error
        
        return StatisticalTestResult(
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha_level,
            test_statistic=z_score,
            test_type="z_test",
            alpha_level=self.alpha_level
        )
    
    def _t_test(
        self, 
        indicator_score: float, 
        null_distribution: Dict[str, float]
    ) -> StatisticalTestResult:
        """Perform t-test for small samples or unknown population variance."""
        mean = null_distribution.get('mean', 0.0)
        std = null_distribution.get('std', 1.0)
        df = null_distribution.get('degrees_of_freedom', 30)
        
        if std <= 0:
            std = 1.0
            
        # Calculate t-statistic
        t_statistic = (indicator_score - mean) / std
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # Effect size (Cohen's d)
        effect_size = abs(t_statistic)
        
        # Confidence interval
        margin_error = stats.t.ppf(1 - self.alpha_level/2, df) * std
        ci_lower = indicator_score - margin_error
        ci_upper = indicator_score + margin_error
        
        return StatisticalTestResult(
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha_level,
            test_statistic=t_statistic,
            degrees_of_freedom=df,
            test_type="t_test",
            alpha_level=self.alpha_level
        )
    
    def _wilcoxon_test(
        self, 
        indicator_score: float, 
        null_distribution: Dict[str, float]
    ) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test for non-parametric testing."""
        # For single value, use approximation based on null distribution
        median = null_distribution.get('median', 0.0)
        mad = null_distribution.get('mad', 1.0)  # Median Absolute Deviation
        
        # Approximate z-score using median and MAD
        z_score = (indicator_score - median) / (1.4826 * mad)  # 1.4826 normalizes MAD to std
        
        # Approximate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Effect size (approximation)
        effect_size = abs(z_score)
        
        # Confidence interval (approximate)
        margin_error = stats.norm.ppf(1 - self.alpha_level/2) * (1.4826 * mad)
        ci_lower = indicator_score - margin_error
        ci_upper = indicator_score + margin_error
        
        return StatisticalTestResult(
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha_level,
            test_statistic=z_score,
            test_type="wilcoxon",
            alpha_level=self.alpha_level
        )
    
    def benjamini_hochberg_correction(
        self, 
        p_values: List[float], 
        alpha: float = None
    ) -> MultipleTestingResult:
        """
        Apply Benjamini-Hochberg (FDR) correction for multiple testing.
        
        Controls the False Discovery Rate (FDR) rather than family-wise error rate.
        Less conservative than Bonferroni correction.
        
        Args:
            p_values: List of p-values from individual tests
            alpha: Desired FDR level (default: uses instance alpha_level)
            
        Returns:
            MultipleTestingResult with corrected p-values and significance decisions
        """
        if alpha is None:
            alpha = self.alpha_level
            
        try:
            p_array = np.array(p_values)
            m = len(p_array)
            
            if m == 0:
                return MultipleTestingResult(
                    original_p_values=[],
                    corrected_p_values=[],
                    significant_indicators=[],
                    correction_method="benjamini_hochberg",
                    family_wise_error_rate=0.0,
                    false_discovery_rate=0.0
                )
            
            # Sort p-values and keep track of original indices
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            # Calculate BH critical values
            critical_values = np.array([(i + 1) / m * alpha for i in range(m)])
            
            # Find the largest k such that p(k) <= (k/m) * alpha
            significant_sorted = sorted_p <= critical_values
            
            # Find the cutoff point
            if np.any(significant_sorted):
                cutoff_idx = np.where(significant_sorted)[0][-1]
                corrected_p = np.minimum(sorted_p * m / np.arange(1, m + 1), 1.0)
            else:
                cutoff_idx = -1
                corrected_p = sorted_p * m / np.arange(1, m + 1)
            
            # Enforce monotonicity constraint
            for i in range(m - 2, -1, -1):
                corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])
            
            # Map back to original order
            corrected_p_values = np.zeros(m)
            corrected_p_values[sorted_indices] = corrected_p
            
            # Determine significance
            significant_indicators = [False] * m
            if cutoff_idx >= 0:
                for i in range(cutoff_idx + 1):
                    original_idx = sorted_indices[i]
                    significant_indicators[original_idx] = True
            
            # Calculate error rates
            n_significant = sum(significant_indicators)
            fdr = min(alpha, n_significant / max(m, 1))
            fwer = 1 - (1 - alpha) ** m if m > 0 else 0.0
            
            return MultipleTestingResult(
                original_p_values=p_values,
                corrected_p_values=corrected_p_values.tolist(),
                significant_indicators=significant_indicators,
                correction_method="benjamini_hochberg",
                family_wise_error_rate=fwer,
                false_discovery_rate=fdr
            )
            
        except Exception as e:
            self.logger.error(f"Benjamini-Hochberg correction failed: {e}")
            # Return conservative result
            return MultipleTestingResult(
                original_p_values=p_values,
                corrected_p_values=[1.0] * len(p_values),
                significant_indicators=[False] * len(p_values),
                correction_method="benjamini_hochberg",
                family_wise_error_rate=1.0,
                false_discovery_rate=1.0
            )
    
    def bonferroni_correction(
        self, 
        p_values: List[float], 
        alpha: float = None
    ) -> MultipleTestingResult:
        """
        Apply Bonferroni correction for multiple testing.
        
        Controls the Family-Wise Error Rate (FWER). More conservative
        than Benjamini-Hochberg but provides stronger error control.
        
        Args:
            p_values: List of p-values from individual tests
            alpha: Significance level (default: uses instance alpha_level)
            
        Returns:
            MultipleTestingResult with corrected p-values and significance decisions
        """
        if alpha is None:
            alpha = self.alpha_level
            
        try:
            corrected_p_values = [min(p * len(p_values), 1.0) for p in p_values]
            significant_indicators = [p < alpha for p in corrected_p_values]
            
            n_significant = sum(significant_indicators)
            fwer = alpha
            fdr = min(alpha * len(p_values) / max(n_significant, 1), 1.0) if n_significant > 0 else 0.0
            
            return MultipleTestingResult(
                original_p_values=p_values,
                corrected_p_values=corrected_p_values,
                significant_indicators=significant_indicators,
                correction_method="bonferroni",
                family_wise_error_rate=fwer,
                false_discovery_rate=fdr
            )
            
        except Exception as e:
            self.logger.error(f"Bonferroni correction failed: {e}")
            return MultipleTestingResult(
                original_p_values=p_values,
                corrected_p_values=[1.0] * len(p_values),
                significant_indicators=[False] * len(p_values),
                correction_method="bonferroni",
                family_wise_error_rate=1.0,
                false_discovery_rate=1.0
            )
    
    def calculate_power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = None,
        test_type: str = "z_test"
    ) -> PowerAnalysisResult:
        """
        Calculate statistical power for given parameters.
        
        Power is the probability of correctly rejecting a false null hypothesis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size for the test
            alpha: Significance level (default: uses instance alpha_level)
            test_type: Type of test for power calculation
            
        Returns:
            PowerAnalysisResult with power analysis metrics
        """
        if alpha is None:
            alpha = self.alpha_level
            
        try:
            if test_type == "z_test":
                # For z-test
                critical_value = stats.norm.ppf(1 - alpha / 2)
                power = 1 - stats.norm.cdf(critical_value - effect_size * np.sqrt(sample_size))
                power += stats.norm.cdf(-critical_value - effect_size * np.sqrt(sample_size))
                
            elif test_type == "t_test":
                # For t-test (approximation)
                df = sample_size - 1
                critical_value = stats.t.ppf(1 - alpha / 2, df)
                ncp = effect_size * np.sqrt(sample_size)  # Non-centrality parameter
                power = 1 - stats.nct.cdf(critical_value, df, ncp)
                power += stats.nct.cdf(-critical_value, df, ncp)
                
            else:
                # Default to z-test approximation
                critical_value = stats.norm.ppf(1 - alpha / 2)
                power = 1 - stats.norm.cdf(critical_value - effect_size * np.sqrt(sample_size))
                power += stats.norm.cdf(-critical_value - effect_size * np.sqrt(sample_size))
            
            # Beta error (Type II error rate)
            beta_error = 1 - power
            
            # Minimum detectable effect (80% power)
            target_power = 0.8
            if test_type == "z_test":
                z_beta = stats.norm.ppf(target_power)
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                mde = (z_alpha + z_beta) / np.sqrt(sample_size)
            else:
                mde = effect_size  # Approximation
            
            return PowerAnalysisResult(
                statistical_power=min(max(power, 0.0), 1.0),
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=alpha,
                beta_error=beta_error,
                minimum_detectable_effect=mde
            )
            
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            return PowerAnalysisResult(
                statistical_power=0.5,  # Conservative estimate
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=alpha,
                beta_error=0.5,
                minimum_detectable_effect=1.0
            )
    
    def calculate_confidence_interval(
        self,
        score: float,
        std_error: float,
        confidence_level: float = 0.95,
        distribution: str = "normal"
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a score.
        
        Args:
            score: Observed score
            std_error: Standard error of the score
            confidence_level: Confidence level (default: 0.95)
            distribution: Distribution type ('normal', 't')
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            alpha = 1 - confidence_level
            
            if distribution == "normal":
                critical_value = stats.norm.ppf(1 - alpha / 2)
            elif distribution == "t":
                # Assume reasonable degrees of freedom if not specified
                df = 30
                critical_value = stats.t.ppf(1 - alpha / 2, df)
            else:
                critical_value = stats.norm.ppf(1 - alpha / 2)
            
            margin_error = critical_value * std_error
            
            return (score - margin_error, score + margin_error)
            
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return (score, score)  # Fallback to point estimate