"""Existing enhanced REST adapters must use actual evidence/shared decisions."""
import asyncio
import importlib
import json
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient

from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError


def endpoint():
    return importlib.import_module('aneos_api.endpoints.enhanced_analysis')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('external network forbidden')))


def neo(name='marker', unusual=True):
    return NEOData(name, orbital_elements=OrbitalElements(semi_major_axis=1., eccentricity=.2,
        inclination=52.23568469321602 if unusual else 15.),
        physical_properties=PhysicalProperties(diameter_km=.37, absolute_magnitude_h=19.7))


@pytest.fixture
def pipeline():
    original = AnalysisPipeline()
    original.data_fetcher = Mock()
    original.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: neo(d)
    return EnhancedAnalysisPipeline(original)


@pytest.mark.parametrize('candidate', [True, False])
def test_existing_enhanced_single_uses_same_science_and_null_inference(pipeline, candidate):
    api = endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: neo(d, candidate)
    result = asyncio.run(api.analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker',
        force_refresh=True, include_raw_data=True), BackgroundTasks(), pipeline, None))
    data = result.model_dump(mode='json')
    expected = ScreeningAnalysisService().analyze_neo(neo(unusual=candidate)).to_payload()
    assert data['screening_assessment'] == expected
    assert data['overall_score'] == expected['screening_score']
    assert data['confidence'] is None
    assert data['validation_result']['overall_validation_passed'] is False
    assert data['validation_result']['overall_false_positive_probability'] is None
    assert len(data['validation_result']['stage_results']) == 5
    assert data['validation_result']['total_processing_time_ms'] is None
    assert all(s['processing_time_ms'] is None for s in data['validation_result']['stage_results'])
    assert data['uncertainty_analysis']['overall_uncertainty'] is None
    assert data['uncertainty_analysis']['analysis_method'] is None
    assert data['raw_neo_data']['designation'] == 'marker'
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker', force_refresh=True)
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize('exception,code', [(DataSourceUnavailableError('offline'),503), (ValueError('invalid'),422), (RuntimeError('broken'),500)])
def test_source_failures_never_produce_mock_results(pipeline, exception, code):
    api = endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect = exception
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None))
    assert exc.value.status_code == code


def test_get_pipeline_delegates_to_actual_application_service(monkeypatch, pipeline):
    api = endpoint()
    from aneos_api import app
    service = Mock(analysis_pipeline=pipeline.original_pipeline)
    monkeypatch.setattr(app, 'get_aneos_app', lambda: service)
    assert asyncio.run(api.get_analysis_pipeline()) is pipeline.original_pipeline
    wrapper = asyncio.run(api.get_enhanced_analysis_pipeline())
    assert wrapper.original_pipeline is pipeline.original_pipeline
    replacement = AnalysisPipeline()
    service.analysis_pipeline = replacement
    assert asyncio.run(api.get_enhanced_analysis_pipeline()).original_pipeline is replacement


def test_request_options_do_not_claim_unperformed_sampling_or_bypass_review(pipeline):
    api = endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    request = EnhancedAnalysisRequest(designation='marker', validation_method='fast',
        uncertainty_method='bootstrap', monte_carlo_samples=123)
    result = asyncio.run(api.analyze_neo_enhanced(request, BackgroundTasks(), pipeline, None))
    assert len(result.validation_result.stage_results) == 5
    assert result.execution_options['requested']['validation_method'] == 'fast'
    assert result.execution_options['uncertainty_computation_performed'] is False
    assert result.uncertainty_analysis.analysis_method is None
    assert result.validation_result.overall_validation_passed is False


def test_disabled_review_is_not_a_pass_or_mutation_of_other_requests(pipeline):
    api = endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    disabled = asyncio.run(api.analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker',
        enable_validation=False), BackgroundTasks(), pipeline, None))
    assert disabled.validation_result.validation_status == 'not_requested'
    assert disabled.validation_result.overall_validation_passed is False
    assert disabled.screening_assessment['candidate'] is True
    enabled = asyncio.run(api.analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None))
    assert len(enabled.validation_result.stage_results) == 5
    assert pipeline.enable_validation is True


def test_mixed_batch_retains_all_ordered_outcomes_and_forwards_flags(pipeline):
    api = endpoint()
    def fetch(d, **kw):
        if d == 'offline':
            raise DataSourceUnavailableError('offline')
        return NEOData(d) if d == 'missing' else neo(d)
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect = fetch
    result = asyncio.run(api.batch_analyze_enhanced(['first', 'offline', 'missing', 'first'],
        BackgroundTasks(), True, False, pipeline, None))
    data = result.model_dump(mode='json')
    assert [r['designation'] for r in data['results']] == ['first', 'offline', 'missing', 'first']
    assert [r['status'] for r in data['results']] == ['completed', 'source_unavailable', 'inconclusive', 'completed']
    assert data['results'][1]['overall_score'] is None
    assert data['results'][1]['classification'] == 'INCONCLUSIVE'
    assert data['summary_statistics']['failed'] == 1
    for call in pipeline.original_pipeline.data_fetcher.fetch_neo_data.call_args_list:
        assert call.kwargs['force_refresh'] is True
    assert data['results'][0]['validation_result']['validation_status'] == 'not_requested'


def test_app_registers_existing_enhanced_routes_and_http_serializes(pipeline):
    api = endpoint()
    from aneos_api.app import create_app
    app = create_app()
    paths = set(app.openapi()['paths'])
    assert '/api/v1/analysis/enhanced' in paths
    assert '/api/v1/analysis/enhanced/batch' in paths
    app.dependency_overrides[api.get_enhanced_analysis_pipeline] = lambda: pipeline
    app.dependency_overrides[api.get_current_user] = lambda: None
    # No lifespan startup: only real routing/serialization, not network/service boot.
    client = TestClient(app)
    response = client.post('/api/v1/analysis/enhanced', json={'designation':'marker'})
    assert response.status_code == 200, response.text
    assert response.json()['screening_assessment']['candidate'] is True
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect = DataSourceUnavailableError('offline')
    assert client.post('/api/v1/analysis/enhanced', json={'designation':'marker'}).status_code == 503


def test_compatibility_acquisition_helper_has_real_refresh_path(monkeypatch,pipeline):
    api = endpoint()
    async def actual():
        return pipeline.original_pipeline
    monkeypatch.setattr(api,'get_analysis_pipeline',actual)
    data = asyncio.run(api.get_neo_data('marker',force_refresh=True))
    assert isinstance(data,NEOData)
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker',force_refresh=True)


def test_concurrent_review_flags_do_not_share_mutation(pipeline):
    api = endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    async def run():
        return await asyncio.gather(*(api.analyze_neo_enhanced(EnhancedAnalysisRequest(
            designation='marker',enable_validation=flag),BackgroundTasks(),pipeline,None)
            for flag in [False,True]))
    disabled,enabled = asyncio.run(run())
    assert disabled.validation_result.validation_status == 'not_requested'
    assert len(enabled.validation_result.stage_results) == 5
    assert not disabled.validation_result.overall_validation_passed
    assert not enabled.validation_result.overall_validation_passed


def test_empty_summary_has_no_zero_probability_or_perfect_performance():
    result = endpoint()._summarize_rows([],7)
    assert result.false_positive_statistics['mean_probability'] is None
    assert result.overall_score_statistics['mean'] is None
    assert result.overall_score_statistics['available_count'] == 0
    assert not result.validation_system_status['validation_available']


def test_persisted_summary_replays_current_policy_and_excludes_legacy_and_corruption():
    from aneos_core.analysis.screening import ScreeningPolicy
    api = endpoint()
    service = ScreeningAnalysisService()
    positives = service.analyze_neo(neo()).to_payload()
    negative = service.analyze_neo(neo('ordinary',False)).to_payload()
    abstain = service.analyze_neo(NEOData('missing')).to_payload()
    other_policy = ScreeningAnalysisService(ScreeningPolicy(candidate_threshold=10)).analyze_neo(neo()).to_payload()
    tampered = dict(positives,artificial_probability=.99)
    rows = [{'anomaly_score_data':d} for d in [positives,negative,abstain,
        {'overall_score':.9,'classification':'natural'},tampered,other_policy]]
    result = api._summarize_rows(rows,7)
    assert result.total_analyses == 6
    assert result.included_screening_assessments == 3
    assert result.excluded_rows == {'legacy_or_missing_contract':1,'invalid_or_unsupported_contract':2}
    assert result.screening_counts == {'FOLLOW_UP_CANDIDATE':1,'NOT_FLAGGED':1,'INCONCLUSIVE':1}
    assert result.overall_score_statistics['available_count'] == 2
    assert result.false_positive_statistics['mean_probability'] is None
    assert result.false_positive_statistics['high_fp_risk_count'] is None
    assert result.recommendations == {'unknown':6}
    assert result.validation_system_status['unique_included_objects'] == 3
    assert len(result.policy_groups) == 1


def test_summary_actual_sql_query_period_and_http(monkeypatch,tmp_path):
    from datetime import datetime,timedelta,timezone
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from aneos_api import database
    from aneos_api.app import create_app
    api = endpoint()
    engine = create_engine(f'sqlite:///{tmp_path / "summary.db"}',connect_args={'check_same_thread':False})
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    now = datetime.now(timezone.utc)
    payload = ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    with factory() as db:
        for age,data in [(1,payload),(1,{'overall_score':.99,'classification':'natural'}),(40,payload)]:
            db.add(database.AnalysisResult(designation='marker',analysis_date=now-timedelta(days=age),
                anomaly_score_data=data,overall_score=.99,classification='natural'))
        db.commit()
    monkeypatch.setattr(database,'SessionLocal',factory)
    app = create_app()
    app.dependency_overrides[api.get_current_user] = lambda: None
    client = TestClient(app)
    result = client.get('/api/v1/analysis/enhanced/summary?days=7')
    assert result.status_code == 200,result.text
    data = result.json()
    assert data['total_analyses'] == 2
    assert data['included_screening_assessments'] == 1
    assert data['overall_score_statistics']['mean'] == payload['screening_score']
    assert data['false_positive_statistics']['mean_probability'] is None
    assert data['recommendations'] == {'unknown':2}
    assert client.get('/api/v1/analysis/enhanced/summary?days=0').status_code == 422
    engine.dispose()


def test_unavailable_summary_database_remains_503(monkeypatch):
    from aneos_api import database
    monkeypatch.setattr(database,'HAS_SQLALCHEMY',False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(endpoint().get_validation_summary(7,None))
    assert exc.value.status_code == 503


@pytest.mark.parametrize('body',[[],[' '],['marker']*101])
def test_batch_rejects_invalid_requests_before_acquisition(pipeline,body):
    api = endpoint()
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: None
    app.dependency_overrides[api.get_enhanced_analysis_pipeline] = lambda: pipeline
    response = TestClient(app).post('/enhanced/batch',json=body)
    assert response.status_code == 422,response.text
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.assert_not_called()


def test_batch_http_uses_existing_bare_list_body_and_query_flags(pipeline):
    api = endpoint()
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: None
    app.dependency_overrides[api.get_enhanced_analysis_pipeline] = lambda: pipeline
    response = TestClient(app).post('/enhanced/batch?force_refresh=true&enable_validation=false',json=['marker','marker'])
    assert response.status_code == 200,response.text
    results = response.json()['results']
    assert len(results)==2
    assert results[0]['screening_assessment']==results[1]['screening_assessment']
    assert all(r['validation_result']['validation_status']=='not_requested' for r in results)
    assert all(c.kwargs['force_refresh'] for c in pipeline.original_pipeline.data_fetcher.fetch_neo_data.call_args_list)


def test_batch_timeout_and_request_unavailable_validation_keep_other_candidate(pipeline,monkeypatch):
    api = endpoint()
    original = pipeline.analyze_neo_with_validation
    async def slow(d,**kw):
        if d=='slow':
            await asyncio.sleep(1)
        return await original(d,**kw)
    monkeypatch.setattr(pipeline,'analyze_neo_with_validation',slow)
    pipeline.original_pipeline.pipeline_config.timeout_seconds=.01
    pipeline.validation_available=False
    result = asyncio.run(api.batch_analyze_enhanced(['slow','marker'],BackgroundTasks(),False,True,pipeline,None))
    assert result.results[0].status=='failed'
    assert result.results[0].overall_score is None
    assert result.results[1].screening_assessment['candidate'] is True
    assert result.results[1].validation_result.validation_status=='unavailable'
    assert result.summary_statistics['required_review_passed']==0


def test_mixed_batch_json_retains_null_diagnostics_not_fake_inference(pipeline):
    api=endpoint()
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw: NEOData(d)
    response=asyncio.run(api.analyze_neo_enhanced(EnhancedAnalysisRequest(designation='missing',
        include_raw_data=False,include_indicators=False),BackgroundTasks(),pipeline,None))
    data=response.model_dump(mode='json')
    assert data['status']=='inconclusive'
    assert data['classification']=='INCONCLUSIVE'
    assert data['overall_score'] is None and data['confidence'] is None
    assert data['raw_neo_data'] is None and data['indicator_scores']=={}
    assert data['screening_assessment']['analysis_metadata']['five_sigma_claim_permitted'] is False
    assert all(s['score'] is None and s['confidence'] is None for s in data['validation_result']['stage_results'])
    json.dumps(data,allow_nan=False)


def test_complete_review_transport_positive_is_not_origin_or_sigma(pipeline,monkeypatch):
    from datetime import datetime
    from aneos_core.validation.multi_stage_validator import ValidationStageResult,EnhancedAnalysisResult
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    async def transport_control(data,original):
        # Explicit synthetic gate/transport fixture: no real scientific model pass claimed.
        stages=[ValidationStageResult(i,'synthetic prerequisite transport control',True,None,None,None,
            {'fixture_scope':'transport_only_not_scientific_validation'},0.,validation_status='passed')
            for i in range(1,6)]
        report=pipeline.validator._aggregate_validation_stages(stages)
        report.evidence_id=original.screening_assessment['evidence_id']
        report.policy_id=original.screening_assessment['policy_id']
        from aneos_core.validation.review_evidence import freeze_review_inputs
        _,report.validation_evidence_id=freeze_review_inputs(data)
        return EnhancedAnalysisResult(original,report,{}, {'status':'unavailable'},datetime.now(),
            screening_assessment=original.screening_assessment)
    monkeypatch.setattr(pipeline.validator,'validate_analysis_result',transport_control)
    response=asyncio.run(endpoint().analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'),
        BackgroundTasks(),pipeline,None))
    assert response.validation_result.overall_validation_passed
    assert response.execution_options['required_review_pass_granted'] is True
    assert response.screening_assessment['artificial_probability'] is None
    assert response.screening_assessment['is_artificial'] is None
    assert response.screening_assessment['sigma_confidence'] is None
    assert response.screening_assessment['analysis_metadata']['five_sigma_claim_permitted'] is False


def test_typed_a3_uncertainty_covariance_and_raw_provenance_survive_rest(pipeline):
    from aneos_core.data.models import NonGravitationalParameters
    from aneos_core.data.orbit_evidence import OrbitModelParameter,OrbitalCovariance
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    data=neo()
    data.nongrav=NonGravitationalParameters(parameters={
        'A1':OrbitModelParameter('A1',3e-11,1e-11,'au/d^2','EST',raw_value='3e-11',raw_sigma='1e-11'),
        'A3':OrbitModelParameter('A3',1e-12,2e-13,'au/d^2','EST',raw_value='1e-12',raw_sigma='2e-13')},
        covariance=OrbitalCovariance(['A1','A3'],[[1e-22,2e-25],[2e-25,4e-26]],2451545.0),
        solution={'source':'synthetic transport control, not a KY26 observation'})
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:data
    response=asyncio.run(endpoint().analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker',
        include_raw_data=True),BackgroundTasks(),pipeline,None))
    expected=data.nongrav.evidence_report()
    assert response.screening_assessment['non_gravitational_evidence']==expected
    assert response.validation_result.non_gravitational_evidence==expected
    assert response.raw_neo_data['nongrav']==data.nongrav.to_dict()
    assert response.validation_result.stage_results[2].details['non_gravitational_evidence']==expected
    json.dumps(response.model_dump(mode='json'),allow_nan=False)


def test_documented_enhanced_routes_and_models_match_actual_app():
    from pathlib import Path
    from aneos_api.app import create_app
    actual=create_app().openapi()
    documented=json.loads((Path(__file__).parents[1]/'docs/api/openapi.json').read_text())
    for path in ['/api/v1/analysis/enhanced','/api/v1/analysis/enhanced/batch','/api/v1/analysis/enhanced/summary']:
        assert documented['paths'][path]==actual['paths'][path]
    for name in ['EnhancedAnalysisRequest','EnhancedAnalysisResponse','EnhancedBatchAnalysisResponse',
        'ValidationResultModel','ValidationStageResultModel','UncertaintyAnalysisModel','ValidationSummaryResponse']:
        assert documented['components']['schemas'][name]==actual['components']['schemas'][name]


def test_missing_source_returns_503_instead_of_zero_or_mock(pipeline):
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect=None
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.return_value=None
    with pytest.raises(HTTPException) as exc:
        asyncio.run(endpoint().analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'),
            BackgroundTasks(),pipeline,None))
    assert exc.value.status_code==503
    assert exc.value.detail['screening_score'] is None


def test_current_missing_application_pipeline_cannot_use_previous_cached_wrapper(monkeypatch,pipeline):
    from aneos_api import app
    api=endpoint()
    monkeypatch.setattr(api,'_enhanced_pipeline',pipeline)
    monkeypatch.setattr(app,'get_aneos_app',lambda:Mock(analysis_pipeline=None))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.get_enhanced_analysis_pipeline())
    assert exc.value.status_code==503


def test_batch_invalid_and_wrong_identity_remain_visible(pipeline):
    def fetch(d,**kw):
        data=neo(d)
        if d=='invalid':
            data.orbital_elements.eccentricity=float('nan')
        if d=='wrong':
            data.designation='other-object'
        return data
    pipeline.original_pipeline.data_fetcher.fetch_neo_data.side_effect=fetch
    result=asyncio.run(endpoint().batch_analyze_enhanced(['invalid','wrong','marker'],
        BackgroundTasks(),False,True,pipeline,None))
    assert [r.status for r in result.results]==['invalid_evidence','invalid_evidence','completed']
    assert [r.designation for r in result.results]==['invalid','wrong','marker']
    assert all(r.overall_score is None and r.classification=='INCONCLUSIVE' for r in result.results[:2])
    assert result.summary_statistics['failed']==2
    assert result.results[2].screening_assessment['candidate'] is True


def test_malformed_review_pass_cannot_advance_or_erase_candidate(pipeline,monkeypatch):
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    original=pipeline.validator.validate_analysis_result
    async def malformed(data,assessment):
        result=await original(data,assessment)
        result.validation_result.overall_validation_passed=True
        result.validation_result.validation_status='passed'
        result.validation_result.stage_results=[]
        return result
    monkeypatch.setattr(pipeline.validator,'validate_analysis_result',malformed)
    response=asyncio.run(endpoint().analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'),
        BackgroundTasks(),pipeline,None))
    assert response.screening_assessment['candidate'] is True
    assert response.validation_result.overall_validation_passed is False
    assert response.validation_result.validation_status=='unavailable'
    assert response.execution_options['required_review_pass_granted'] is False


def test_unsupported_probability_claims_cannot_leak_through_review_transport(pipeline,monkeypatch):
    from aneos_api.enhanced_models import EnhancedAnalysisRequest
    original=pipeline.validator.validate_analysis_result
    async def unsupported(data,assessment):
        result=await original(data,assessment)
        result.validation_result.overall_false_positive_probability=.01
        result.validation_result.overall_confidence=.99
        for stage in result.validation_result.stage_results:
            stage.confidence=.9
            stage.false_positive_reduction=.9
        return result
    monkeypatch.setattr(pipeline.validator,'validate_analysis_result',unsupported)
    response=asyncio.run(endpoint().analyze_neo_enhanced(EnhancedAnalysisRequest(designation='marker'),
        BackgroundTasks(),pipeline,None))
    assert response.validation_result.overall_false_positive_probability is None
    assert response.validation_result.overall_confidence is None
    assert all(s.confidence is None and s.false_positive_reduction is None for s in response.validation_result.stage_results)


@pytest.mark.parametrize('defect',['missing','duplicate','error','truthy','evidence','policy','review_inputs'])
def test_review_transport_rejects_each_incomplete_or_stale_binding(pipeline,defect):
    from aneos_core.validation.multi_stage_validator import ValidationStageResult
    from aneos_core.validation.review_evidence import freeze_review_inputs
    original=asyncio.run(pipeline.original_pipeline.analyze_neo('marker'))
    stages=[ValidationStageResult(i,'synthetic binding control',True,None,None,None,{},0.,
        validation_status='passed') for i in range(1,6)]
    report=pipeline.validator._aggregate_validation_stages(stages)
    report.evidence_id=original.screening_assessment['evidence_id']
    report.policy_id=original.screening_assessment['policy_id']
    _,report.validation_evidence_id=freeze_review_inputs(original.neo_data)
    if defect=='missing': report.stage_results.pop()
    if defect=='duplicate': report.stage_results[-1].stage_number=1
    if defect=='error': report.stage_results[0].details['error']='failed prerequisite'
    if defect=='truthy': report.stage_results[0].passed=1
    if defect=='evidence': report.evidence_id='stale'
    if defect=='policy': report.policy_id='stale'
    if defect=='review_inputs': original.neo_data.physical_properties.rotation_period_hours=.1
    output=endpoint()._validation_transport(report,original)
    assert not output.overall_validation_passed
    assert output.validation_status=='unavailable'
    assert output.overall_false_positive_probability is None
    assert output.transport_limitations
