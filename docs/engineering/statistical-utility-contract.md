# Statistical utilities: explicit inputs, limited conclusions

`aneos_core.validation.statistical_testing.StatisticalTesting` is a mathematical
utility, not the deployed detector or an independently calibrated scientific model.
ADR-096 records the contract correction; results identify `statistical-utility-v2`. No canonical score, policy threshold,
archived result or API field is recalculated by this change.

## Confirmed defects and corrections

The previous implementation silently supplied null mean 0, standard deviation 1
and Student degrees of freedom 30. Invalid tests returned p=1/zero effect, failed
power calculations returned 50%, and invalid intervals could become exact points.
A scalar median/MAD normal approximation was labelled Wilcoxon. Multiplicity
helpers inferred FDR/FWER from alpha and rejection counts, without outcome labels.
Student minimum detectable effect simply echoed the supplied effect. These were
not defensible uncertainty or performance estimates.

Invalid/missing/nonfinite inputs now raise `ValueError`. The retained scalar
Wilcoxon entry point raises `NotImplementedError`: it cannot perform a signed-rank
test without observation differences. No old fabricated result is silently mapped
to a successful or natural scientific assessment. Existing callable names remain,
with stricter prerequisites documented below. Old persisted artifacts are untouched.

## Supplied null tests

```python
from dataclasses import asdict
from aneos_core.validation.statistical_testing import StatisticalTesting

s = StatisticalTesting(alpha_level=0.05)
# Synthetic arithmetic example, not astronomical observations or calibration.
normal = s.formal_hypothesis_test(10.0, {"mean": 0.0, "std": 1.0})
assert 0 < normal.p_value < 1e-22
assert normal.confidence_interval is None
assert normal.five_sigma_claim_permitted is False

student = s.formal_hypothesis_test(2.0, {
    "mean": 0.0, "standard_error": 0.5, "degrees_of_freedom": 9,
}, "t_test")
print(asdict(student))
```

Normal `mean`/`std` must describe the null distribution of the supplied statistic;
no defaults or scale repair are allowed. Its two-sided tail is not automatically
an uncertainty interval for an estimated parameter. Student input is an estimated
mean with an actual standard error and positive degrees of freedom, not a single
population draw divided by population standard deviation. See SciPy's
[one-sample test definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
The retained `effect_size` is explicitly the absolute standardized test statistic,
not Cohen's d, significance calibration or evidence of technological origin.

Survival functions avoid cancellation in `1-cdf`; see SciPy's
[normal distribution methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).
A representable 10-standard-deviation normal tail remains nonzero. At floating-point
underflow, `p_value_underflow=True` distinguishes numeric zero from mathematical
zero; `log_p_value` is retained when representable, otherwise null. This does not
license discovery language. JSON serialization never needs nonstandard Infinity.

`calculate_confidence_interval` requires positive finite standard error, a confidence
level strictly between zero and one, and `normal` or `t`. Student intervals also
require explicit keyword `degrees_of_freedom`. The caller owns sampling assumptions;
an interval cannot be fabricated from an anomaly score or unknown measurement error.

## Multiple testing: control target is not measured error

BH, BY and Bonferroni preserve input order, validate every p-value in [0,1], and
compare adjusted values **inclusively** to the supplied alpha. Empty families return
empty arrays, not measured zero error. Each result preserves the legacy fields
`false_discovery_rate` and `family_wise_error_rate` as **null**. Separate
`control_level`, `control_target`, `assumptions` and `error_rate_status` describe the
conditional procedure. Consumers that assumed numeric error rates must handle null.

BH requires independent or appropriate positively dependent valid marginal p-values.
BY is more conservative and supports arbitrary dependence of valid marginal p-values.
See SciPy's [false-discovery adjustment documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).
Bonferroni permits arbitrary dependence. None establishes valid null p-values,
fixes omitted searches/selection bias or estimates deployed false discoveries from
rejection counts. Passing an aNEOS heuristic score to these helpers is invalid.

## Power is hypothetical planning

`calculate_power_analysis(effect_size, sample_size, test_type='z_test')` models a
fixed two-sided one-sample normal experiment. Standardized effect is the mean
shift divided by population standard deviation; sample size counts independent
observations, not correlated features/providers. z assumes known variance; t uses
estimated variance with df=n-1. Student power uses the
[noncentral t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html).
Both tails are included. Minimum detectable effect solves the 80%-power target;
when alpha already exceeds 80%, the minimum is zero. Signed effects have symmetric
power under this two-sided model. No unknown test silently falls back to normal.

This calculation does not measure the power of aNEOS screening, survey completeness,
artificial-origin accuracy or sensitivity on independent NEOs. Those require
appropriate real populations, frozen protocols and external evaluation (#10/#11).
No Brier/artificiality prediction is added here.

## Verification and integration limits

```sh
python -m pytest tests/test_statistical_utility_contract.py -q \
  --cov=aneos_core --cov-config=tests/coverage-statistical-utilities.ini \
  --cov-report=term-missing --cov-fail-under=95
python -m pytest tests/test_ky26_baseline.py tests/test_core_analysis_contract.py -q
python -m pytest tests/ aneos_core/tests/ -m 'not network' \
  --cov=aneos_core --cov-fail-under=30
```

Tests compare adjusted values with SciPy, Student summaries with the actual
one-sample test, and Student planning power with a separate normal/chi-squared
variance-mixture integral. Invalid inputs, overflow/underflow and corrupted numeric
outputs have denial controls alongside valid positive cases. These are deterministic
mathematical controls, not independent scientific validation. No network is needed.

Current canonical screening and required review do not invoke these calculations;
retained wrappers instantiate the utility but do not gain a calibrated significance
model from it. The original 4.142 display and rounded-input KY26 reproduction remain
historical outputs. Source validity, population selection, covariance, search family,
physical explanations and independent evaluation remain separate obligations.
