"""Standalone/simple/advanced real-service controls, not scientific validation."""
import asyncio
import json
from unittest.mock import Mock
import pytest
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',Mock(side_effect=AssertionError('external network forbidden')))


def neo(d='marker',candidate=True):
    return NEOData(d,orbital_elements=OrbitalElements(semi_major_axis=1.,eccentricity=.2,
        inclination=52.23568469321602 if candidate else 15.),
        physical_properties=PhysicalProperties(diameter_km=.37,absolute_magnitude_h=19.7))


@pytest.fixture
def pipeline():
    p=AnalysisPipeline();p.data_fetcher=Mock()
    p.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:neo(d)
    return p


@pytest.mark.parametrize('method',['analyze_neo_comprehensive','analyze_single_neo','detect_artificial_signatures'])
@pytest.mark.parametrize('candidate',[True,False])
def test_simple_library_aliases_use_actual_typed_evidence(pipeline,method,candidate):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:neo(d,candidate)
    analyzer=SimpleNEOAnalyzer(pipeline=pipeline)
    report=getattr(analyzer,method)('marker')
    assert report['screening_assessment']==ScreeningAnalysisService().analyze_neo(neo(candidate=candidate)).to_payload()
    assert report['is_artificial'] is None
    assert report['artificial_probability'] is None
    assert report['confidence_level'] is None
    assert report['impact_assessment']['collision_probability'] is None
    assert report['impact_assessment']['risk_level']=='not_assessed'
    pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker',force_refresh=False)
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('d',['test','tesla','roadster','1998 KY26','2024 PLACEHOLDER'])
def test_names_and_failed_sources_never_make_probability_or_orbits(pipeline,d):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    pipeline.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('offline')
    report=SimpleNEOAnalyzer(pipeline=pipeline).analyze_neo_comprehensive(d)
    assert report['status']=='source_unavailable'
    assert report['classification']=='INCONCLUSIVE'
    assert report['artificial_probability'] is None
    assert report['is_artificial'] is None
    assert report['neo_data'] is None
    assert report['errors']


def test_scalar_probability_and_mock_contracts_are_controlled_deprecations(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    analyzer=SimpleNEOAnalyzer(pipeline=pipeline)
    for name,args in [('calculate_artificial_probability',('test',)),('_fallback_calculation',('roadster',)),
        ('_generate_test_result',()),('_get_mock_orbital_elements',('marker',)),
        ('_create_mock_orbital_elements_for_impact',('marker',))]:
        with pytest.raises(NotImplementedError):getattr(analyzer,name)(*args)
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_calibration_helper_cannot_create_fixed_prior_or_zero_probability(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    result=SimpleNEOAnalyzer(pipeline=pipeline)._calculate_calibrated_probabilities(Mock())
    assert result['status']=='not_calibrated'
    assert result['calibrated_artificial_probability'] is None
    assert result['sigma_level'] is None


def test_real_data_accessor_preserves_native_fields_and_does_not_calculate_probability(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    data=SimpleNEOAnalyzer(pipeline=pipeline).get_neo_data('marker')
    assert data['designation']=='marker'
    assert data['orbital_elements']['semi_major_axis']==1.
    assert data['physical_properties']['diameter_km']==.37
    assert data['data_source']!='mock_for_compatibility'


def test_advanced_wrapper_and_report_are_shared_core_not_a_second_score(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    result=analyzer.analyze_neo_comprehensive('marker')
    assert result.screening_assessment==ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    assert result.is_artificial is None and result.artificial_probability is None
    assert result.confidence_level is None and result.sigma_statistical_level is None
    assert result.risk_assessment=='not_assessed'
    assert result.analysis_quality is None
    assert analyzer.assess_threat_level(result)=='not_assessed'
    assert analyzer.validate_analysis_quality(result)['status']=='unavailable'
    report=analyzer.generate_comprehensive_report(result)
    assert report['screening_assessment']==result.screening_assessment
    assert report['neo_data']==result.neo_data.to_dict()


def test_advanced_caller_raw_data_is_not_mixed_with_a_fetched_default(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    result=analyzer.analyze_neo_comprehensive('marker',orbital_elements={'a':1.,'e':0.,'i':0.},physical_data={})
    assert result.screening_assessment['evidence_snapshot']['orbital']=={'a':1.,'e':0.,'i':0.}
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()
    result=analyzer.analyze_neo_comprehensive('marker',orbital_elements={})
    assert result.screening_assessment['screening_score'] is None
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_capability_count_or_imports_cannot_mean_calibrated_or_production_ready():
    from aneos_core.interfaces.unified_analysis import AdvancedaNEOSCapabilities,AnalysisCapability,assess_aneos_system_maturity
    caps=AdvancedaNEOSCapabilities()
    assert not caps.is_capability_available(AnalysisCapability.STATISTICAL_VALIDATION)
    assert not caps.is_capability_available(AnalysisCapability.ARTIFICIAL_DETECTION)
    result=assess_aneos_system_maturity()
    assert result['maturity_level']=='research_unvalidated'
    assert result['capability_coverage'] is None
    assert result['five_sigma_claim_permitted'] is False


def test_standalone_script_main_uses_actual_shared_cli_and_nonzero_errors(pipeline,monkeypatch,capsys):
    import simple_neo_analyzer as simple
    from aneos_core.analysis import cli
    monkeypatch.setattr(cli,'create_analysis_pipeline',lambda:pipeline)
    assert simple.main(['single','marker','--json'])==0
    assert json.loads(capsys.readouterr().out)['screening_assessment']['candidate']
    pipeline.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('offline')
    assert simple.main(['analyze','test','--json'])==1
    assert json.loads(capsys.readouterr().out)['screening_assessment'] is None


@pytest.mark.asyncio
async def test_async_entrypoints_preserve_refresh_and_reject_nested_sync(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    simple=SimpleNEOAnalyzer(pipeline=pipeline)
    advanced=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    for analyzer in [simple,advanced]:
        result=await analyzer.analyze_neo_async('marker',force_refresh=True)
        payload=result['screening_assessment'] if isinstance(result,dict) else result.screening_assessment
        assert payload['candidate']
        with pytest.raises(RuntimeError,match='active event loop'):
            analyzer.analyze_neo_comprehensive('marker')
    assert all(call.kwargs['force_refresh'] is True for call in pipeline.data_fetcher.fetch_neo_data.call_args_list)


def test_core_initialization_failure_has_no_dummy_fallback(monkeypatch):
    import simple_neo_analyzer as simple
    monkeypatch.setattr(simple,'create_analysis_pipeline',Mock(side_effect=RuntimeError('core unavailable')))
    analyzer=simple.SimpleNEOAnalyzer()
    assert analyzer.get_status()['initialized'] is False
    report=analyzer.analyze_single_neo('test')
    assert report['status']=='service_unavailable'
    assert report['artificial_probability'] is None
    assert report['classification']=='INCONCLUSIVE'


@pytest.mark.parametrize('bad',[None,NEOData('other')])
def test_missing_or_wrong_source_is_not_a_default_orbit(pipeline,bad):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    pipeline.data_fetcher.fetch_neo_data.side_effect=None
    pipeline.data_fetcher.fetch_neo_data.return_value=bad
    report=SimpleNEOAnalyzer(pipeline=pipeline).analyze_single_neo('marker')
    assert report['screening_assessment'] is None
    assert report['is_artificial'] is None
    assert report['neo_data'] is None
    assert report['errors']


def test_advanced_native_nongrav_uncertainty_and_correlations_survive(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    from aneos_core.data.models import NonGravitationalParameters
    from aneos_core.data.orbit_evidence import OrbitModelParameter,OrbitalCovariance
    from aneos_core.analysis.reporting import verify_screening_report
    n=neo();n.nongrav=NonGravitationalParameters(parameters={
        'A1':OrbitModelParameter('A1',3e-11,1e-11,'au/d^2','EST',raw_value='3e-11',raw_sigma='1e-11'),
        'A3':OrbitModelParameter('A3',1.2e-12,2e-13,'au/d^2','EST',raw_value='1.2e-12',raw_sigma='2e-13')},
        covariance=OrbitalCovariance(['A1','A3'],[[1e-22,2e-25],[2e-25,4e-26]],2451545.))
    pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:n
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    result=analyzer.analyze_neo_comprehensive('marker')
    report=analyzer.generate_comprehensive_report(result)
    assert report['neo_data']['nongrav']==n.nongrav.to_dict()
    assert report['neo_data']['nongrav']['covariance']['data'][0][1]==2e-25
    assert report['screening_assessment']['non_gravitational_evidence']==n.nongrav.evidence_report()
    assert report['impact_assessment']['input_inventory']['covariance_epoch_jd_tdb']==2451545.
    assert report['impact_assessment']['collision_probability'] is None
    assert verify_screening_report(report)


@pytest.mark.parametrize('bad_orbit',[{'a':float('nan'),'e':.2,'i':15}, {'a':1,'e':.2,'i':15,'epoch':2451545.},
    {'a':1,'e':.2,'i':15,'epoch':'2000-01-01'}, {'a':1,'e':.2,'i':15,'mean_anomaly':float('inf')}])
def test_advanced_invalid_raw_input_never_fetches_or_silently_defaults(pipeline,bad_orbit):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    result=AdvancedaNEOSAnalyzer(pipeline=pipeline).analyze_neo_comprehensive('marker',orbital_elements=bad_orbit)
    assert result.status=='invalid_evidence'
    assert result.screening_assessment is None
    assert result.artificial_probability is None
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_provided_raw_unit_normalization_and_preserved_epoch(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    result=analyzer.analyze_neo_comprehensive('marker',orbital_elements={
        'a':1.,'e':.2,'i':15.,'epoch':'2000-01-01T00:00:00Z'},physical_data={'diameter':370.,'rotation_period':.1})
    assert result.neo_data.physical_properties.diameter_km==.37
    assert result.neo_data.physical_properties.rotation_period_hours==.1
    assert result.neo_data.orbital_elements.epoch.isoformat()=='2000-01-01T00:00:00+00:00'
    assert result.metadata['caller_supplied_input_record']['evidence_basis']=='caller_supplied_not_independently_verified'
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_unsupported_source_selection_cannot_silently_claim_success(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    with pytest.raises(NotImplementedError,match='source filtering'):
        analyzer.analyze_neo_comprehensive('marker',enrichment_sources=['made-up-source'])
    with pytest.raises(NotImplementedError,match='source filtering'):
        analyzer.fetch_neo_data_multi_source('marker',sources=[])
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_enrichment_preserves_originals_without_invented_measurements(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    original={'designation':'marker','orbital_elements':{'a':1.}}
    result=analyzer.enrich_neo_data_comprehensive(original)
    assert 'enrichment_status' not in original
    assert result['enrichment_status']=='not_performed_no_validated_model'
    assert 'physical_data' not in result and 'computed_characteristics' not in result
    result['orbital_elements']['a']=2.
    assert original['orbital_elements']['a']==1.


@pytest.mark.parametrize('designation,return_code',[('marker',0),('offline',1)])
def test_actual_standalone_process_with_network_denied(designation,return_code,tmp_path):
    import os,subprocess,sys
    from pathlib import Path
    script=r'''
import sys,socket,runpy
from unittest.mock import Mock
socket.socket.connect=Mock(side_effect=AssertionError('external network forbidden'))
from aneos_core.data.fetcher import DataFetcher
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError
def fetch(self,d,**kw):
    if d=='offline': raise DataSourceUnavailableError('offline synthetic control')
    return NEOData(d,orbital_elements=OrbitalElements(semi_major_axis=1.,eccentricity=.2,inclination=52.23568469321602),physical_properties=PhysicalProperties(diameter_km=.37,absolute_magnitude_h=19.7))
DataFetcher.fetch_neo_data=fetch
sys.argv=['simple_neo_analyzer.py']+sys.argv[1:]
runpy.run_path('simple_neo_analyzer.py',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',script,'single',designation,'--json'],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=30,
        env={**os.environ,'ANEOS_DATABASE_URL':f'sqlite:///{tmp_path / "standalone.db"}','PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==return_code,result.stderr
    data=json.loads(result.stdout)
    assert data['designation']==designation
    assert data['artificial_probability'] is None
    assert data['screening_assessment']['candidate'] if return_code==0 else data['screening_assessment'] is None


def test_actual_arc_inventory_is_not_a_quality_or_impact_probability(pipeline):
    from datetime import datetime,timezone,timedelta
    from simple_neo_analyzer import SimpleNEOAnalyzer
    n=neo();n.first_observation=datetime(2026,1,1,tzinfo=timezone.utc);n.last_observation=n.first_observation+timedelta(days=12)
    pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:n
    report=SimpleNEOAnalyzer(pipeline=pipeline).analyze_neo_comprehensive('marker')
    impact=report['impact_assessment']
    assert impact['input_inventory']['observation_arc_days']==12.
    assert impact['calculation_confidence'] is None
    assert impact['collision_probability'] is None


def test_timezone_aware_datetime_raw_epoch_remains_json_serializable(pipeline):
    from datetime import datetime,timezone
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    analyzer=AdvancedaNEOSAnalyzer(pipeline=pipeline)
    result=analyzer.analyze_neo_comprehensive('marker',orbital_elements={
        'a':1.,'e':.2,'i':15.,'epoch':datetime(2000,1,1,tzinfo=timezone.utc)})
    report=analyzer.generate_comprehensive_report(result)
    json.dumps(report,allow_nan=False)
    assert report['neo_data']['orbital_elements']['epoch']=='2000-01-01T00:00:00+00:00'


def test_simple_status_and_impact_helpers_do_not_invent_probability_or_arc(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    from types import SimpleNamespace
    analyzer=SimpleNEOAnalyzer(pipeline=pipeline)
    assert analyzer.get_status()['origin_model_available'] is False
    assert analyzer._extract_orbital_elements_for_impact(SimpleNamespace(designation='tesla')) is None
    assert analyzer._estimate_observation_arc_quality(SimpleNamespace(designation='tesla')) is None
    impact=analyzer._calculate_impact_probability(SimpleNamespace(designation='tesla'),'tesla')
    assert impact['collision_probability'] is None and impact['risk_level']=='not_assessed'
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


@pytest.mark.parametrize('orbit,physical',[(0,None),([],None),({},[]),({},0)])
def test_falsey_invalid_containers_are_not_missing_measurement_defaults(pipeline,orbit,physical):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    result=AdvancedaNEOSAnalyzer(pipeline=pipeline).analyze_neo_comprehensive('marker',
        orbital_elements=orbit,physical_data=physical)
    assert result.status=='invalid_evidence'
    assert result.screening_assessment is None
    assert result.artificial_probability is None
    pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_legacy_detection_confidence_key_is_explicitly_null_not_missing(pipeline):
    from simple_neo_analyzer import SimpleNEOAnalyzer
    result=SimpleNEOAnalyzer(pipeline=pipeline).detect_artificial_signatures('marker')
    assert result.get('confidence',0) is None
    assert result['sigma_statistical_level'] is None
    assert result['risk_assessment']=='not_assessed'


def test_advanced_wrapper_cannot_claim_cache_miss_without_request_metadata(pipeline):
    from aneos_core.analysis.advanced_aneos_analyzer import AdvancedaNEOSAnalyzer
    result=AdvancedaNEOSAnalyzer(pipeline=pipeline).analyze_neo_comprehensive('marker')
    assert result.cache_status=='not_reported'
