"""Supplied-model statistical arithmetic; not deployed scientific calibration.

Canonical screening does not obtain discovery significance from these utilities.
Callers own the population model, observation independence, sampling/selection,
valid marginal p-values and the complete search family. Invalid prerequisites raise
ValueError, never fabricated neutral evidence. See ADR-096 and the utility guide.
"""
from dataclasses import dataclass, field
import math
from numbers import Real, Integral
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import brentq

STATUS = 'supplied_model_not_independently_validated'
METHOD_VERSION = 'statistical-utility-v2'


@dataclass
class StatisticalTestResult:
    p_value: float
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]]
    is_significant: bool
    test_statistic: float
    degrees_of_freedom: Optional[float] = None
    test_type: str = 'z_test'
    alpha_level: float = .05
    log_p_value: Optional[float] = None
    p_value_underflow: bool = False
    effect_size_basis: str = 'absolute_standardized_test_statistic_not_origin_evidence'
    validation_status: str = STATUS
    five_sigma_claim_permitted: bool = False
    method_version: str = METHOD_VERSION
    assumptions: List[str] = field(default_factory=list)


@dataclass
class MultipleTestingResult:
    original_p_values: List[float]
    corrected_p_values: List[float]
    significant_indicators: List[bool]
    correction_method: str
    # These fields are retained for readers, not misused as nominal control levels.
    family_wise_error_rate: Optional[float] = None
    false_discovery_rate: Optional[float] = None
    control_level: float = .05
    control_target: str = ''
    assumptions: List[str] = field(default_factory=list)
    error_rate_status: str = 'not_empirically_estimated'
    validation_status: str = STATUS
    five_sigma_claim_permitted: bool = False
    method_version: str = METHOD_VERSION


@dataclass
class PowerAnalysisResult:
    statistical_power: float
    effect_size: float
    sample_size: int
    alpha_level: float
    beta_error: float
    minimum_detectable_effect: float
    test_type: str = 'z_test'
    target_power: float = .8
    assumptions: List[str] = field(default_factory=list)
    validation_status: str = 'hypothetical_planning_not_empirical_performance'
    five_sigma_claim_permitted: bool = False
    method_version: str = METHOD_VERSION


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name} is outside numeric range') from exc
    if not math.isfinite(result) or positive and result <= 0:
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


def _level(value, name='alpha'):
    result = _number(value, name)
    if not 0 < result < 1:
        raise ValueError(f'{name} must be strictly between 0 and 1')
    return result


def _critical(alpha, distribution, df=None):
    # isf avoids cancellation in 1-alpha/2; reject unrepresentable critical values.
    tail = alpha / 2
    value = stats.norm.isf(tail) if distribution == 'normal' else stats.t.isf(tail, df)
    return _number(value, 'critical value', positive=True)


class StatisticalTesting:
    """Explicit normal/Student calculations and p-value adjustment.

    Significance flags are conditional mathematical comparisons, not validation of
    any aNEOS anomaly, artificial-origin hypothesis or search-wide discovery.
    """
    def __init__(self, alpha_level=.05):
        self.alpha_level = _level(alpha_level)

    def formal_hypothesis_test(self, indicator_score, null_distribution, test_type='z_test'):
        """Two-sided tail under an explicitly supplied normal or Student model.

        z_test: `mean`, `std` describe the null distribution of this statistic.
        t_test: the input is an estimated mean; supply null `mean`, actual
        `standard_error`, and `degrees_of_freedom`. No default variance or df.
        """
        if test_type == 'wilcoxon':
            return self._wilcoxon_test(indicator_score, null_distribution)
        if test_type not in ('z_test', 't_test'):
            raise ValueError('Unsupported hypothesis test')
        return (self._z_test if test_type == 'z_test' else self._t_test)(indicator_score, null_distribution)

    def _test(self, score, model, kind):
        score = _number(score, 'statistic')
        if not isinstance(model, dict):
            raise ValueError('Explicit null model required')
        try:
            mean = _number(model['mean'], 'null mean')
            scale = _number(model['std' if kind == 'z_test' else 'standard_error'],
                            'null scale', positive=True)
            df = _number(model['degrees_of_freedom'], 'degrees of freedom', positive=True) if kind == 't_test' else None
        except KeyError as exc:
            raise ValueError('Required null-model parameter missing') from exc
        alpha = _level(self.alpha_level)
        statistic = _number((score - mean) / scale, 'standardized test statistic')
        distribution = stats.norm if df is None else stats.t(df)
        p = _number(2 * distribution.sf(abs(statistic)), 'tail probability')
        if not 0 <= p <= 1:
            raise ValueError('Tail probability outside [0,1]')
        raw_log_p = float(math.log(2) + distribution.logsf(abs(statistic)))
        if math.isnan(raw_log_p) or raw_log_p == math.inf:
            raise ValueError('Log-tail calculation failed')
        # Extreme finite statistics can exhaust even log-tail precision. Preserve
        # unavailable explicitly instead of emitting invalid JSON Infinity/NaN.
        log_p = min(0., raw_log_p) if math.isfinite(raw_log_p) else None
        interval = None
        if df is not None:
            margin = _number(_critical(alpha, 't', df)*scale, 'interval margin')
            interval = (_number(score-margin, 'lower interval bound'),
                        _number(score+margin, 'upper interval bound'))
        assumptions = (['Supplied normal null distribution is valid for this statistic; not fitted here.']
                       if df is None else ['Estimated mean and supplied standard error obey a Student sampling model with the supplied df.'])
        return StatisticalTestResult(p, abs(statistic), interval, p <= alpha, statistic,
            degrees_of_freedom=df, test_type=kind, alpha_level=alpha, log_p_value=log_p,
            p_value_underflow=p == 0., assumptions=assumptions + [
                'No population calibration, selection correction or origin inference is performed.'])

    def _z_test(self, indicator_score, null_distribution):
        return self._test(indicator_score, null_distribution, 'z_test')

    def _t_test(self, indicator_score, null_distribution):
        return self._test(indicator_score, null_distribution, 't_test')

    def _wilcoxon_test(self, indicator_score, null_distribution):
        raise NotImplementedError('Scalar median/MAD is not a Wilcoxon signed-rank test; paired observations and a specified test are required')

    def _adjust(self, p_values, alpha, method):
        alpha = _level(self.alpha_level if alpha is None else alpha)
        if not isinstance(p_values, (list, tuple, np.ndarray)):
            raise ValueError('A one-dimensional finite family of p-values is required')
        if isinstance(p_values, np.ndarray) and p_values.ndim != 1:
            raise ValueError('P-value family must be one-dimensional')
        values = [_number(p, 'p-value') for p in p_values]
        if any(not 0 <= p <= 1 for p in values):
            raise ValueError('P-values must be within [0,1]')
        m = len(values)
        if method == 'bonferroni':
            # Multiply only below 1/m, avoiding overflow for large families.
            adjusted = [1. if p >= 1/m else p*m for p in values] if m else []
            target = 'family_wise_error_rate_bound'
            assumptions = ['Valid marginal p-values; arbitrary dependence is permitted.']
        else:
            order = np.argsort(values, kind='stable')
            sorted_values = np.asarray(values, dtype=float)[order]
            factor = math.fsum(1/i for i in range(1, m+1)) if method == 'benjamini_yekutieli' else 1.
            raw = sorted_values * (m / np.arange(1, m+1)) * factor if m else np.array([])
            monotone = np.minimum.accumulate(raw[::-1])[::-1] if m else raw
            output = np.empty(m)
            output[order] = np.minimum(monotone, 1.)
            adjusted = output.tolist()
            target = 'false_discovery_rate_bound'
            assumptions = (['Valid marginal p-values; arbitrary dependence is permitted.'] if factor != 1. or method == 'benjamini_yekutieli'
                           else ['Valid marginal p-values; independence or positive regression dependence on true nulls is required.'])
        return MultipleTestingResult(values, adjusted, [p <= alpha for p in adjusted], method,
            control_level=alpha, control_target=target, assumptions=assumptions + [
                'The family and selection procedure must be specified; no omitted trials are corrected.',
                'Nominal control is conditional on assumptions, not an empirical error-rate estimate.'])

    def benjamini_hochberg_correction(self, p_values, alpha=None):
        return self._adjust(p_values, alpha, 'benjamini_hochberg')

    def benjamini_yekutieli_correction(self, p_values, alpha=None):
        return self._adjust(p_values, alpha, 'benjamini_yekutieli')

    def bonferroni_correction(self, p_values, alpha=None):
        return self._adjust(p_values, alpha, 'bonferroni')

    def calculate_power_analysis(self, effect_size, sample_size, alpha=None, test_type='z_test'):
        """Hypothetical two-sided one-sample normal planning; MDE targets 80%.

        effect_size=(alternative mean-null mean)/population standard deviation.
        n is the number of independent observations, not catalogue feature count.
        """
        effect = _number(effect_size, 'standardized effect')
        if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, Integral):
            raise ValueError('Sample size must be an integer')
        n = int(sample_size)
        if test_type not in ('z_test', 't_test') or n < (2 if test_type == 't_test' else 1):
            raise ValueError('Unsupported test or insufficient sample size')
        alpha = _level(self.alpha_level if alpha is None else alpha)
        df = n-1 if test_type == 't_test' else None
        critical = _critical(alpha, 't' if df is not None else 'normal', df)
        root_n = math.sqrt(_number(n, 'sample size', positive=True))
        def power(d):
            noncentrality = _number(abs(d)*root_n, 'noncentrality')
            if df is None:
                value = stats.norm.sf(critical-noncentrality) + stats.norm.cdf(-critical-noncentrality)
            else:
                value = stats.nct.sf(critical, df, noncentrality) + stats.nct.cdf(-critical, df, noncentrality)
            value = _number(value, 'power')
            if not -1e-12 <= value <= 1+1e-12:
                raise ValueError('Power outside numeric probability range')
            return min(1., max(0., value))
        observed = power(effect)
        target = .8
        if power(0.) >= target:
            mde = 0.
        else:
            # Solve in noncentrality units so large n does not collapse MDE to
            # zero through an absolute root tolerance in effect-size units.
            upper = 1.
            for _ in range(64):
                if power(upper/root_n) >= target:
                    break
                upper *= 2
            else:
                raise ValueError('Minimum detectable effect could not be bracketed')
            mde = float(brentq(lambda nc: power(nc/root_n)-target, 0., upper, xtol=1e-12))/root_n
            if abs(power(mde)-target) > 1e-8:
                raise ValueError('Minimum detectable effect failed its numerical target')
        return PowerAnalysisResult(observed, effect, n, alpha, 1-observed, mde,
            test_type=test_type, target_power=target, assumptions=[
                'Independent normal observations; specified standardized alternative and fixed two-sided alpha.',
                'Population variance is known.' if df is None else 'Variance is estimated from the same independent normal sample, df=n-1.',
                'Hypothetical planning, not measured sensitivity, survey power or origin accuracy.'])

    def calculate_confidence_interval(self, score, std_error, confidence_level=.95,
                                      distribution='normal', *, degrees_of_freedom=None):
        """Location interval conditional on a real supplied standard error/model."""
        score = _number(score, 'location estimate')
        se = _number(std_error, 'standard error', positive=True)
        confidence = _level(confidence_level, 'confidence level')
        if distribution not in ('normal', 't'):
            raise ValueError('Unsupported confidence-interval distribution')
        df = _number(degrees_of_freedom, 'degrees of freedom', positive=True) if distribution == 't' else None
        critical = _critical(1-confidence, distribution, df)
        margin = _number(critical*se, 'interval margin')
        return (_number(score-margin, 'lower interval bound'), _number(score+margin, 'upper interval bound'))
