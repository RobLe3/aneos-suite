"""Offline core/REST parity and failure controls; not detector calibration."""
import asyncio
import json
from unittest.mock import Mock, AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException

from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('External network access forbidden in core contract controls')
    monkeypatch.setattr(socket.socket, 'connect', denied)


def neo(name='marker', unusual=True):
    return NEOData(name, orbital_elements=OrbitalElements(
        semi_major_axis=1., eccentricity=.2, inclination=52.23568469321602 if unusual else 15.),
        physical_properties=PhysicalProperties(diameter_km=.37, absolute_magnitude_h=19.7))


@pytest.fixture
def pipeline(monkeypatch):
    # Real initialized orchestrator; injected acquisition, never a network request.
    monkeypatch.setattr(AnalysisPipeline, '_create_data_source_manager', lambda self: Mock())
    p = AnalysisPipeline()
    p.data_fetcher = Mock()
    p.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: neo(d)
    return p


@pytest.mark.parametrize('unusual', [True, False])
def test_core_shared_payload_and_no_legacy_scoring(pipeline, unusual):
    n = neo(unusual=unusual)
    pipeline._run_indicators = AsyncMock(side_effect=AssertionError('legacy path invoked'))
    pipeline.score_calculator.calculate_score = Mock(side_effect=AssertionError('legacy score invoked'))
    result = asyncio.run(pipeline.analyze_neo(n.designation, n))
    assert not result.errors
    assert result.screening_assessment == ScreeningAnalysisService().analyze_neo(n).to_payload()
    assert result.neo_data is n
    assert result.anomaly_score.overall_score == result.screening_assessment['screening_score']
    assert result.anomaly_score.confidence is None
    assert result.anomaly_score.score_semantics == 'uncalibrated_screening_rank'
    assert result.anomaly_score.classification == ('FOLLOW_UP_CANDIDATE' if unusual else 'NOT_FLAGGED')
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize('failure', [None, DataSourceUnavailableError('offline'), RuntimeError('broken')])
def test_acquisition_failure_is_not_a_natural_negative(pipeline, failure):
    pipeline.data_fetcher.fetch_neo_data.side_effect = failure
    pipeline.data_fetcher.fetch_neo_data.return_value = None
    result = asyncio.run(pipeline.analyze_neo('marker'))
    assert result.errors
    assert result.status in {'source_unavailable', 'failed'}
    assert result.anomaly_score.classification == 'INCONCLUSIVE'
    assert result.anomaly_score.overall_score is None
    assert result.anomaly_score.confidence is None
    assert result.screening_assessment is None
    assert pipeline.performance_metrics['failed_analyses'] == 1


@pytest.mark.parametrize('case', ['missing', 'invalid', 'wrong_identity'])
def test_incomplete_or_invalid_evidence_never_natural(pipeline, case):
    n = NEOData('marker') if case == 'missing' else neo()
    if case == 'invalid':
        n.orbital_elements.eccentricity = float('nan')
    if case == 'wrong_identity':
        n.designation = 'other-marker'
    result = asyncio.run(pipeline.analyze_neo('marker', n))
    assert result.anomaly_score.overall_score is None
    assert result.anomaly_score.classification == 'INCONCLUSIVE'
    assert result.status == ('inconclusive' if case == 'missing' else 'invalid_evidence')


def test_canonical_acquisition_force_refresh_and_no_old_result_cache(pipeline):
    first = asyncio.run(pipeline.analyze_neo('marker', force_refresh=True))
    pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker', force_refresh=True)
    pipeline.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: neo(d, unusual=False)
    second = asyncio.run(pipeline.analyze_neo('marker'))
    assert first.screening_assessment['evidence_id'] != second.screening_assessment['evidence_id']
    assert not second.screening_assessment['candidate']


def test_quality_is_descriptive_not_invented_reliability(pipeline):
    n = neo()
    n.orbital_elements.eccentricity = n.orbital_elements.inclination = 0
    quality = pipeline._assess_data_quality(n)
    assert quality['reliability_score'] is None
    assert quality['data_age_days'] is None
    assert 'orbital.eccentricity' not in quality['missing_elements']
    assert quality['physical_fields_present'] == 2


def test_batch_preserves_identity_order_and_failure(pipeline):
    def fetch(d, **kw):
        if d == 'offline':
            raise DataSourceUnavailableError('offline')
        return neo(d)
    pipeline.data_fetcher.fetch_neo_data.side_effect = fetch
    progress = []
    result = asyncio.run(pipeline.analyze_batch(['first', 'offline', 'first'], lambda n, total: progress.append((n, total))))
    assert [r.designation for r in result] == ['first', 'offline', 'first']
    assert [r.status for r in result] == ['completed', 'source_unavailable', 'completed']
    assert progress == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.parametrize('raw,indicators', [(False, False), (True, True)])
def test_analyze_route_real_result_shape_and_parity(pipeline, raw, indicators):
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest
    result = asyncio.run(analysis.analyze_neo(AnalysisRequest(designation='marker', force_refresh=True,
        include_raw_data=raw, include_indicators=indicators), BackgroundTasks(), pipeline, None))
    payload = result.model_dump(mode='json')
    expected = ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    assert payload['screening_assessment'] == expected
    assert payload['anomaly_score']['overall_score'] == expected['screening_score']
    assert payload['anomaly_score']['confidence'] is None
    assert bool(payload['raw_neo_data']) is raw
    assert bool(payload['anomaly_score']['indicator_scores']) is indicators
    pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker', force_refresh=True)
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize('error,code', [(DataSourceUnavailableError('offline'), 503), (ValueError('invalid'), 422)])
def test_analyze_route_preserves_typed_error(pipeline, error, code):
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest
    pipeline.data_fetcher.fetch_neo_data.side_effect = error
    with pytest.raises(HTTPException) as exc:
        asyncio.run(analysis.analyze_neo(AnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None))
    assert exc.value.status_code == code


def test_ml_cannot_consume_new_rank_or_old_prediction_cache(pipeline):
    from aneos_core.ml.prediction import RealTimePredictor, PredictionConfig
    # Constructor bypass avoids model file access; contract check must precede cache/model access.
    predictor = RealTimePredictor.__new__(RealTimePredictor)
    predictor.analysis_pipeline = pipeline
    predictor.config = PredictionConfig()
    predictor.cache_manager = Mock()
    predictor.cache_manager.get.return_value = {
        'designation': 'marker', 'anomaly_score': 0., 'anomaly_probability': 0.,
        'is_anomaly': False, 'confidence': 0., 'model_id': 'legacy-cache',
        'predicted_at': '2026-11-12T00:00:00'}
    predictor.prediction_stats = {'cache_hits': 0, 'total_predictions': 0, 'failed_predictions': 0}
    for method, args in [(predictor.predict_anomaly, ('marker',)), (predictor.predict_batch, (['marker'],))]:
        with pytest.raises(NotImplementedError, match='calibrat|contract|screening'):
            asyncio.run(method(*args))
    predictor.cache_manager.get.assert_not_called()


def test_existing_rest_route_http_serialization_and_unavailable(pipeline):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aneos_api.endpoints import analysis
    app = FastAPI()
    app.include_router(analysis.router)
    app.dependency_overrides[analysis.get_analysis_pipeline] = lambda: pipeline
    app.dependency_overrides[analysis.get_current_user] = lambda: None
    with TestClient(app) as client:
        response = client.post('/analyze', json={'designation': 'marker', 'force_refresh': True})
        assert response.status_code == 200, response.text
        assert response.json()['screening_assessment']['candidate'] is True
        pipeline.data_fetcher.fetch_neo_data.side_effect = DataSourceUnavailableError('offline')
        response = client.post('/analyze', json={'designation': 'marker'})
        assert response.status_code == 503
        assert response.json()['detail']['screening_score'] is None


def test_request_cannot_reuse_previous_designation_only_result(pipeline):
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest
    async def run():
        first = await analysis.analyze_neo(AnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None)
        pipeline.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: neo(d, unusual=False)
        second = await analysis.analyze_neo(AnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None)
        assert first.screening_assessment['candidate'] is True
        assert second.screening_assessment['candidate'] is False
    asyncio.run(run())


def test_batch_timeout_and_callback_error_preserve_results(pipeline):
    pipeline.pipeline_config.timeout_seconds = .01
    original = pipeline.analyze_neo
    async def slow(d, **kw):
        if d == 'slow':
            await asyncio.sleep(1)
        return await original(d, **kw)
    pipeline.analyze_neo = slow
    def broken_callback(*args):
        raise RuntimeError('UI disconnected')
    results = asyncio.run(pipeline.analyze_batch(['slow', 'marker'], broken_callback))
    assert [r.designation for r in results] == ['slow', 'marker']
    assert results[0].status == 'failed'
    assert results[0].anomaly_score.overall_score is None
    assert results[1].screening_assessment['candidate'] is True


@pytest.mark.parametrize('classification', ['FOLLOW_UP_CANDIDATE', 'NOT_FLAGGED', 'INCONCLUSIVE', 'natural'])
def test_screening_and_legacy_classifications_cannot_be_training_truth(classification):
    import numpy as np
    from aneos_core.ml.features import FeatureVector
    from aneos_core.ml.training import TrainingDataManager, TrainingConfig
    manager = TrainingDataManager.__new__(TrainingDataManager)
    vector = FeatureVector('marker', np.array([1.]), ['a'], metadata={'classification': classification})
    with pytest.raises(ValueError, match='Independent'):
        manager.prepare_training_data([vector], TrainingConfig())


def test_independent_training_labels_positive_control_aligns_object_order():
    import numpy as np
    from aneos_core.ml.features import FeatureVector
    from aneos_core.ml.training import TrainingDataManager, TrainingConfig
    manager = TrainingDataManager.__new__(TrainingDataManager)
    manager.feature_engineer = Mock()
    manager.feature_engineer.create_feature_matrix.return_value = (np.array([[2.], [1.]]), ['a'], ['b', 'a'])
    vectors = [FeatureVector(d, np.array([i]), ['a'], metadata={
        'independent_label': label, 'label_provenance': 'synthetic label-contract control, not observations'})
        for d, i, label in [('a', 1., 0), ('b', 2., 1)]]
    X, y, names = manager.prepare_training_data(vectors, TrainingConfig(feature_selection=False))
    assert list(y) == [1, 0]
    assert names == ['a']


def test_direct_indicator_prediction_cannot_relabel_rank_as_probability(pipeline):
    from aneos_core.ml.prediction import RealTimePredictor
    result = asyncio.run(pipeline.analyze_neo('marker', neo()))
    predictor = RealTimePredictor.__new__(RealTimePredictor)
    with pytest.raises(NotImplementedError, match='probability'):
        predictor._create_indicator_prediction('marker', result.anomaly_score)


def test_ml_api_guard_precedes_cache_and_job_submission(pipeline):
    from aneos_api.endpoints import prediction
    from aneos_api.models import PredictionRequest, BatchPredictionRequest
    predictor = Mock(analysis_pipeline=pipeline)
    prediction._prediction_cache['marker_None'] = object()
    try:
        for method, request in [(prediction.predict_anomaly, PredictionRequest(designation='marker')),
                                (prediction.predict_batch, BatchPredictionRequest(designations=['marker']))]:
            with pytest.raises(HTTPException) as exc:
                asyncio.run(method(request, BackgroundTasks(), predictor, None))
            assert exc.value.status_code == 503
        predictor.predict_anomaly.assert_not_called()
    finally:
        prediction._prediction_cache.pop('marker_None', None)


def test_menu_null_safe_core_result_plain_and_rich(pipeline, capsys):
    from aneos_menu import ANEOSMenu
    pipeline.data_fetcher.fetch_neo_data.side_effect = DataSourceUnavailableError('offline')
    result = asyncio.run(pipeline.analyze_neo('marker'))
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    menu.display_analysis_result(result)
    text = capsys.readouterr().out
    assert 'unavailable' in text and 'INCONCLUSIVE' in text
    assert 'natural' not in text.lower()
    menu.console = Mock()
    menu.display_analysis_result(result)
    menu.console.print.assert_called_once()


def test_json_csv_export_preserves_contract_and_replay(pipeline):
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest, ExportRequest
    analysis._analysis_cache.clear()
    asyncio.run(analysis.analyze_neo(AnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None))
    for fmt in ['json', 'csv']:
        analysis._export_jobs['contract-export'] = {}
        asyncio.run(analysis._process_export('contract-export', ExportRequest(format=fmt)))
        job = analysis._export_jobs['contract-export']
        assert job['status'] == 'completed'
        text = job['content'].decode()
        assert 'uncalibrated_screening_rank' in text
        assert 'policy_id' in text and 'evidence_id' in text
        if fmt == 'json':
            data = json.loads(text)[0]['screening_assessment']
            replay = ScreeningAnalysisService().replay(data['evidence_snapshot'], data['evidence_id'])
            assert replay.to_payload() == data
    analysis._export_jobs.pop('contract-export', None)
    analysis._analysis_cache.clear()


def test_documented_core_analysis_openapi_contract():
    from pathlib import Path
    from aneos_api.models import AnalysisResponse
    documented = json.loads((Path(__file__).parents[1] / 'docs/api/openapi.json').read_text())
    actual = AnalysisResponse.model_json_schema(ref_template='#/components/schemas/{model}')
    actual.pop('$defs', None)
    assert documented['components']['schemas']['AnalysisResponse'] == actual


def test_core_result_keeps_candidate_through_validation_and_fallback(pipeline):
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    wrapper = EnhancedAnalysisPipeline(pipeline)
    result = asyncio.run(wrapper.analyze_neo_with_validation('marker'))
    assert result.screening_assessment['candidate'] is True
    assert result.validation_result.overall_validation_passed is False
    assert result.validation_result.overall_false_positive_probability is None
    assert len(result.validation_result.stage_results) == 5
    wrapper.validation_available = False
    result = asyncio.run(wrapper.analyze_neo_with_validation('marker'))
    assert result.screening_assessment['candidate'] is True
    assert result.validation_result.validation_status == 'unavailable'


def test_native_acceleration_survives_core_and_api_report(pipeline):
    from aneos_core.data.models import NonGravitationalParameters, CloseApproach
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest
    n = neo()
    n.nongrav = NonGravitationalParameters(a3=1.23e-12)
    n.close_approaches = [CloseApproach('marker', epoch_jd_tdb=2460000.5, time_scale='TDB',
        body='Earth', distance_au=.01, evidence_metadata={'marker': 'computed, not astrometry'})]
    pipeline.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: n
    response = asyncio.run(analysis.analyze_neo(AnalysisRequest(designation='marker', include_raw_data=True),
        BackgroundTasks(), pipeline, None))
    assert response.raw_neo_data['nongrav']['a3'] == n.nongrav.a3
    assert response.screening_assessment['non_gravitational_evidence'] == n.nongrav.evidence_report()
    assert response.close_approaches[0].epoch_jd_tdb == 2460000.5
    assert response.close_approaches[0].time_scale == 'TDB'
    assert response.close_approaches[0].evidence_metadata == n.close_approaches[0].evidence_metadata


def test_existing_batch_route_forwards_refresh_and_raw_data(monkeypatch):
    from aneos_api.endpoints import analysis
    from aneos_api.models import BatchAnalysisRequest
    from aneos_core.data.fetcher import DataFetcher
    fetch = Mock(side_effect=lambda d, **kw: neo(d))
    monkeypatch.setattr(DataFetcher, 'fetch_neo_data', fetch)
    monkeypatch.setattr(analysis, '_persist_detection_result', Mock())
    tasks = BackgroundTasks()
    response = asyncio.run(analysis.analyze_batch(BatchAnalysisRequest(
        designations=['marker'], force_refresh=True, include_raw_data=True), tasks, None))
    try:
        asyncio.run(tasks())
        fetch.assert_called_once_with('marker', force_refresh=True)
        result = analysis._batch_store[response['batch_id']]['results'][0]
        assert result['candidate'] is True
        assert result['raw_neo_data']['designation'] == 'marker'
        assert result['sigma_confidence'] is None
    finally:
        analysis._batch_store.pop(response['batch_id'], None)
        analysis._detection_cache.pop('marker', None)


def test_search_uses_ranking_units_and_latest_result_not_stale_detect_cache(pipeline):
    from aneos_api.endpoints import analysis
    from aneos_api.models import AnalysisRequest, SearchRequest, AnalysisFilter
    analysis._analysis_cache.clear()
    analysis._detection_cache['marker'] = {'classification': 'stale-marker'}
    response = asyncio.run(analysis.analyze_neo(AnalysisRequest(designation='marker'), BackgroundTasks(), pipeline, None))
    latest = asyncio.run(analysis.get_analysis_result('marker', None))
    assert latest == response.screening_assessment
    search = asyncio.run(analysis.search_analyses(SearchRequest(filters=AnalysisFilter(
        min_score=5, max_score=100, classification=['FOLLOW_UP_CANDIDATE'])), None))
    assert search.total == 1
    assert search.items[0].screening_assessment == response.screening_assessment
    analysis._analysis_cache.clear()
    analysis._detection_cache.pop('marker', None)


def test_core_lazily_constructs_typed_fetcher_once(pipeline, monkeypatch):
    from aneos_core.data import fetcher
    factory = Mock(return_value=pipeline.data_fetcher)
    pipeline.data_fetcher = None
    monkeypatch.setattr(fetcher, 'DataFetcher', factory)
    first = asyncio.run(pipeline.analyze_neo('marker'))
    second = asyncio.run(pipeline.analyze_neo('marker', force_refresh=True))
    assert first.screening_assessment == second.screening_assessment
    factory.assert_called_once_with(cache_manager=pipeline.cache_manager)
    assert factory.return_value.fetch_neo_data.call_args_list[0].kwargs == {'force_refresh': False}
    assert factory.return_value.fetch_neo_data.call_args_list[1].kwargs == {'force_refresh': True}
