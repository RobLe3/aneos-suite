"""Deterministic mathematical controls, not independent astronomical validation."""
from dataclasses import asdict
import math
import numpy as np
import pytest
from scipy import stats
from aneos_core.validation.statistical_testing import StatisticalTesting


def test_extreme_normal_tail_remains_representable_and_no_fake_interval():
    result=StatisticalTesting().formal_hypothesis_test(10,{'mean':0,'std':1})
    assert result.p_value==pytest.approx(1.523970604832094e-23,rel=1e-12,abs=0)
    assert result.confidence_interval is None
    assert result.validation_status=='supplied_model_not_independently_validated'
    assert result.five_sigma_claim_permitted is False


@pytest.mark.parametrize('model',[{}, {'mean':0}, {'mean':0,'std':0}, {'mean':0,'std':-1},
    {'mean':0,'std':float('nan')},{'mean':False,'std':1}])
def test_missing_or_invalid_normal_model_is_not_default_standard_normal(model):
    with pytest.raises(ValueError):StatisticalTesting().formal_hypothesis_test(1,model)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'2',None])
def test_invalid_measurement_does_not_turn_into_p_one(value):
    with pytest.raises(ValueError):StatisticalTesting().formal_hypothesis_test(value,{'mean':0,'std':1})


def test_student_summary_matches_actual_sample_test():
    sample=np.array([1.,2.,3.,5.,7.]);mean=float(sample.mean());se=float(stats.sem(sample))
    result=StatisticalTesting().formal_hypothesis_test(mean,{'mean':0.,'standard_error':se,
        'degrees_of_freedom':4},'t_test')
    reference=stats.ttest_1samp(sample,0.)
    assert result.test_statistic==pytest.approx(reference.statistic)
    assert result.p_value==pytest.approx(reference.pvalue)
    assert result.confidence_interval==pytest.approx(tuple(reference.confidence_interval()))


@pytest.mark.parametrize('model',[{'mean':0,'std':1},{'mean':0,'standard_error':1},
    {'mean':0,'standard_error':0,'degrees_of_freedom':4},
    {'mean':0,'standard_error':1,'degrees_of_freedom':0}])
def test_student_requires_actual_standard_error_and_df(model):
    with pytest.raises(ValueError):StatisticalTesting().formal_hypothesis_test(3,model,'t_test')


def test_scalar_mad_is_not_wilcoxon_and_unknown_test_not_normal():
    with pytest.raises(NotImplementedError):
        StatisticalTesting().formal_hypothesis_test(3,{'median':0,'mad':1},'wilcoxon')
    with pytest.raises(ValueError):StatisticalTesting().formal_hypothesis_test(3,{},'unknown')


@pytest.mark.parametrize('method',['benjamini_hochberg_correction','bonferroni_correction'])
def test_multiplicity_does_not_invent_observed_error_rates(method):
    result=getattr(StatisticalTesting(),method)([.8,.9])
    assert result.false_discovery_rate is None and result.family_wise_error_rate is None
    assert result.error_rate_status=='not_empirically_estimated'
    assert result.control_level==.05 and result.assumptions
    assert result.five_sigma_claim_permitted is False


@pytest.mark.parametrize('ps',[[.01,.04,.03,.8],[0.,0.,1.],[.9,.8,.7],[],[.05],[.01,.01,.02]])
@pytest.mark.parametrize('method,scipy_method',[('benjamini_hochberg_correction','bh'),
                                            ('benjamini_yekutieli_correction','by')])
def test_adjusted_p_values_match_scipy_and_preserve_order(ps,method,scipy_method):
    result=getattr(StatisticalTesting(),method)(ps)
    expected=stats.false_discovery_control(ps,method=scipy_method) if ps else []
    assert result.corrected_p_values==pytest.approx(expected)
    assert result.significant_indicators==[p<=.05 for p in expected]
    assert result.original_p_values==ps


@pytest.mark.parametrize('ps',[[float('nan')],[float('inf')],[-.1],[1.1],[True],['.1'],[[.1]],None])
@pytest.mark.parametrize('method',['benjamini_hochberg_correction','benjamini_yekutieli_correction','bonferroni_correction'])
def test_invalid_p_values_are_not_fabricated_adjustments(ps,method):
    with pytest.raises(ValueError):getattr(StatisticalTesting(),method)(ps)


@pytest.mark.parametrize('alpha',[0,1,-.1,float('nan'),float('inf'),True,'0.05'])
def test_alpha_is_strict_at_initialization_and_per_call(alpha):
    with pytest.raises(ValueError):StatisticalTesting(alpha)
    with pytest.raises(ValueError):StatisticalTesting().bonferroni_correction([.1],alpha)


def test_bonferroni_inclusive_boundary_and_empty_family():
    result=StatisticalTesting().bonferroni_correction([.025,.9])
    assert result.corrected_p_values==[.05,1.]
    assert result.significant_indicators==[True,False]
    assert StatisticalTesting().bonferroni_correction([]).false_discovery_rate is None


@pytest.mark.parametrize('kind',['z_test','t_test'])
def test_power_mde_has_actual_target_power_and_signed_effect_symmetry(kind):
    s=StatisticalTesting();positive=s.calculate_power_analysis(.5,25,test_type=kind)
    negative=s.calculate_power_analysis(-.5,25,test_type=kind)
    assert positive.statistical_power==pytest.approx(negative.statistical_power)
    at_mde=s.calculate_power_analysis(positive.minimum_detectable_effect,25,test_type=kind)
    assert at_mde.statistical_power==pytest.approx(.8,abs=1e-10)
    assert positive.beta_error==pytest.approx(1-positive.statistical_power)
    assert positive.target_power==.8 and positive.assumptions
    assert s.calculate_power_analysis(0,25,test_type=kind).statistical_power==pytest.approx(.05)


@pytest.mark.parametrize('effect,n,kind',[(float('nan'),20,'z_test'),(.2,0,'z_test'),(.2,True,'z_test'),
    (.2,2.5,'z_test'),(.2,1,'t_test'),(.2,10,'unknown')])
def test_invalid_planning_input_not_fifty_percent_power(effect,n,kind):
    with pytest.raises(ValueError):StatisticalTesting().calculate_power_analysis(effect,n,test_type=kind)


def test_interval_requires_student_degrees_of_freedom_and_no_distribution_fallback():
    s=StatisticalTesting()
    with pytest.raises(ValueError):s.calculate_confidence_interval(1,1,distribution='t')
    with pytest.raises(ValueError):s.calculate_confidence_interval(1,1,distribution='other')
    assert s.calculate_confidence_interval(2,1,distribution='t',degrees_of_freedom=4)==pytest.approx(
        (2-stats.t.isf(.025,4),2+stats.t.isf(.025,4)))


@pytest.mark.parametrize('score,se,level',[(1,0,.95),(1,-1,.95),(1,float('nan'),.95),
    (float('nan'),1,.95),(1,1,1),(1,1,0),(True,1,.95)])
def test_bad_interval_inputs_never_fallback_to_point(score,se,level):
    with pytest.raises(ValueError):StatisticalTesting().calculate_confidence_interval(score,se,level)


def test_underflow_is_distinguished_from_mathematical_zero():
    result=StatisticalTesting().formal_hypothesis_test(40,{'mean':0,'std':1})
    assert result.p_value==0 and result.p_value_underflow is True
    assert result.log_p_value==pytest.approx(math.log(2)+stats.norm.logsf(40))
    assert math.isfinite(result.log_p_value)
    assert result.five_sigma_claim_permitted is False


def test_student_power_matches_independent_variance_mixture_integral():
    from scipy.integrate import quad
    df=24;critical=stats.t.isf(.025,df);shift=.5*5
    def integrand(v):
        cutoff=critical*math.sqrt(v/df)
        return (stats.norm.sf(cutoff-shift)+stats.norm.cdf(-cutoff-shift))*stats.chi2.pdf(v,df)
    expected,error=quad(integrand,0,np.inf,epsabs=1e-11)
    assert error<1e-8
    assert StatisticalTesting().calculate_power_analysis(.5,25,test_type='t_test').statistical_power==pytest.approx(expected,abs=1e-10)


def test_large_sample_mde_not_lost_to_absolute_root_tolerance():
    s=StatisticalTesting();result=s.calculate_power_analysis(0,10**30)
    assert result.minimum_detectable_effect>0
    assert s.calculate_power_analysis(result.minimum_detectable_effect,10**30).statistical_power==pytest.approx(.8,abs=1e-10)


def test_student_small_alpha_does_not_round_interval_confidence_to_one():
    result=StatisticalTesting(1e-20).formal_hypothesis_test(0,{
        'mean':0,'standard_error':1,'degrees_of_freedom':10},'t_test')
    assert result.p_value==1 and all(math.isfinite(v) for v in result.confidence_interval)


def test_nominal_alpha_already_exceeds_planning_target():
    result=StatisticalTesting(.9).calculate_power_analysis(.1,20)
    assert result.minimum_detectable_effect==0 and result.target_power==.8


@pytest.mark.parametrize('family',[np.array(.5), np.ones((2,2)), [complex(.1)], [None]])
def test_non_scalar_or_nonreal_p_inputs_rejected(family):
    with pytest.raises(ValueError):StatisticalTesting().benjamini_hochberg_correction(family)


def test_numpy_scalar_inputs_and_one_dimensional_families_supported():
    s=StatisticalTesting(np.float64(.05))
    assert s.formal_hypothesis_test(np.int64(0),{'mean':0,'std':1}).p_value==1
    assert s.bonferroni_correction(np.array([0.,.2])).corrected_p_values==[0.,.4]
    assert s.calculate_power_analysis(np.float64(.2),np.int64(20)).sample_size==20


def test_calls_do_not_mutate_inputs_and_results_remain_json_serializable():
    import json
    values=[.02,.001,.3];before=values.copy()
    result=StatisticalTesting().benjamini_hochberg_correction(values)
    result.original_p_values[0]=1
    assert values==before
    payload=asdict(StatisticalTesting().formal_hypothesis_test(10,{'mean':0,'std':1}))
    assert json.loads(json.dumps(payload,allow_nan=False))['p_value']>0


@pytest.mark.parametrize('action',[
    lambda s:s.formal_hypothesis_test(1e308,{'mean':-1e308,'std':1}),
    lambda s:s.calculate_confidence_interval(1e308,1e308),
    lambda s:s.calculate_power_analysis(1e308,100),
    lambda s:s.calculate_power_analysis(.1,10**1000),
    lambda s:s.formal_hypothesis_test(1,None),
    lambda s:s._z_test(1,{'mean':0,'std':False}),
])
def test_numeric_overflow_or_direct_invalid_call_is_explicit(action):
    with pytest.raises(ValueError):action(StatisticalTesting())


def test_log_tail_precision_exhaustion_is_json_safe_and_not_discovery():
    import json
    result=StatisticalTesting().formal_hypothesis_test(1e200,{'mean':0,'std':1})
    assert result.p_value_underflow and result.log_p_value is None
    assert result.five_sigma_claim_permitted is False
    json.dumps(asdict(result),allow_nan=False)


def test_invalid_numeric_power_from_distribution_not_reported_as_probability(monkeypatch):
    monkeypatch.setattr(stats.nct,'sf',lambda *a:1.)
    monkeypatch.setattr(stats.nct,'cdf',lambda *a:1.)
    with pytest.raises(ValueError,match='probability range'):
        StatisticalTesting().calculate_power_analysis(.5,25,test_type='t_test')


def test_unsolvable_mde_not_substituted_with_requested_effect(monkeypatch):
    monkeypatch.setattr(stats.norm,'sf',lambda *a:.25)
    monkeypatch.setattr(stats.norm,'cdf',lambda *a:.25)
    with pytest.raises(ValueError,match='bracketed'):
        StatisticalTesting().calculate_power_analysis(.5,25)


def test_bad_root_solver_result_fails_numerical_target(monkeypatch):
    import aneos_core.validation.statistical_testing as module
    monkeypatch.setattr(module,'brentq',lambda *a,**k:0.)
    with pytest.raises(ValueError,match='numerical target'):
        StatisticalTesting().calculate_power_analysis(.5,25)


def test_invalid_numeric_tail_not_silently_reported(monkeypatch):
    monkeypatch.setattr(stats.norm,'sf',lambda *a:1.)
    with pytest.raises(ValueError,match='probability outside'):
        StatisticalTesting().formal_hypothesis_test(1,{'mean':0,'std':1})


def test_failed_log_tail_not_silently_reported(monkeypatch):
    monkeypatch.setattr(stats.norm,'logsf',lambda *a:float('nan'))
    with pytest.raises(ValueError,match='Log-tail'):
        StatisticalTesting().formal_hypothesis_test(1,{'mean':0,'std':1})


def test_every_result_identifies_corrected_method_version_without_promoting_calibration():
    s=StatisticalTesting()
    for result in (s.formal_hypothesis_test(0,{'mean':0,'std':1}),
                   s.bonferroni_correction([]),s.calculate_power_analysis(.2,10)):
        assert result.method_version=='statistical-utility-v2'
        assert result.five_sigma_claim_permitted is False


def test_positive_infinite_log_tail_is_an_error_not_precision_underflow(monkeypatch):
    monkeypatch.setattr(stats.norm,'logsf',lambda *a:float('inf'))
    with pytest.raises(ValueError,match='Log-tail'):
        StatisticalTesting().formal_hypothesis_test(1,{'mean':0,'std':1})
