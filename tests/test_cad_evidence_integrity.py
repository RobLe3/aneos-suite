"""Synthetic CAD contract controls; no live providers or scientific accuracy claim."""
import copy
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aneos_core.polling.historical_chunked_poller import HistoricalChunkedPoller, ChunkConfig
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties


@pytest.fixture
def payload():
    return {'signature': {'source': 'NASA/JPL SBDB Close Approach Data API', 'version': '1.5'},
        'count': 1, 'fields': ['des', 'orbit_id', 'jd', 'cd', 'dist', 'dist_min', 'dist_max',
            'v_rel', 'v_inf', 't_sigma_f', 'h', 'diameter', 'diameter_sigma'],
        'data': [['synthetic-marker', '42', '2460000.5', '2023-Feb-25 00:00', '.03', '.02', '.04',
            '12', '11', '00:01', '25.9', '.011', '.002']]}


def poller(tmp_path):
    p = HistoricalChunkedPoller.__new__(HistoricalChunkedPoller)
    p.config = ChunkConfig(enable_caching=True, rate_limit_delay=0)
    p.cache_dir = tmp_path / 'cache'
    p.results_dir = tmp_path / 'results'
    p.cache_dir.mkdir(exist_ok=True)
    p.results_dir.mkdir(exist_ok=True)
    p.logger = logging.getLogger(__name__)
    p.console = None
    p.base_poller = None
    p.xviii_swarm_scorer = None
    p.data_fetcher = None
    return p


def parsed(payload):
    from aneos_core.data.sources.cad import parse_cad_payload
    return parse_cad_payload(payload, query={'body': 'Earth'}, retrieved_at='2026-11-12T00:00:00Z')


def test_label_order_native_times_units_and_no_fake_orbit(payload):
    original = parsed(payload)
    indices = list(reversed(range(len(payload['fields']))))
    payload['fields'] = [payload['fields'][i] for i in indices]
    payload['data'] = [[row[i] for i in indices] for row in payload['data']]
    result = parsed(payload)
    item = result['records'][0]
    assert item['designation'] == 'synthetic-marker'
    assert item['miss_distance_au'] == .03
    assert item['relative_velocity_km_s'] == 12
    assert item['diameter_km'] == .011
    assert item['diameter_uncertainty'] == .002
    assert item['absolute_magnitude_h'] == 25.9
    assert item['orbital_elements'] is None
    assert item['discovery_date'] is None
    assert item['close_approach']['epoch_jd_tdb'] == 2460000.5
    assert item['close_approach']['calendar_tdb'] == '2023-Feb-25 00:00'
    assert item['encounter_id'] == original['records'][0]['encounter_id']
    assert item['cad_evidence']['raw_fields'] == payload['fields']
    assert item['cad_evidence']['raw_row'] == payload['data'][0]
    assert item['cad_evidence']['evidence_basis'] == 'catalogue_prediction'
    assert result['metadata']['canonical_response_sha256']


@pytest.mark.parametrize('name', ['jd', 'cd', 'dist', 'v_rel', 'h', 'diameter', 'diameter_sigma'])
def test_missing_measurements_stay_unavailable(payload, name):
    payload['data'][0][payload['fields'].index(name)] = None
    item = parsed(payload)['records'][0]
    mapping = {'dist': ('miss_distance_au',), 'v_rel': ('relative_velocity_km_s',),
               'h': ('absolute_magnitude_h',), 'diameter': ('diameter_km',),
               'diameter_sigma': ('diameter_uncertainty',)}
    if name in mapping:
        assert item[mapping[name][0]] is None
    else:
        key = 'epoch_jd_tdb' if name == 'jd' else 'calendar_tdb'
        assert item['close_approach'][key] is None
    assert item['discovery_date'] is None


@pytest.mark.parametrize('mode', ['version', 'count', 'fields', 'duplicate', 'row', 'error', 'truncated'])
def test_invalid_or_incomplete_payloads_fail_explicitly(payload, mode):
    if mode == 'version': payload['signature']['version'] = 'future'
    elif mode == 'count': payload['count'] = 2
    elif mode == 'fields': payload.pop('fields')
    elif mode == 'duplicate': payload['fields'][1] = 'des'
    elif mode == 'row': payload['data'][0].pop()
    elif mode == 'error': payload['error'] = 'synthetic provider failure'
    elif mode == 'truncated': payload['total'] = 100
    with pytest.raises(ValueError):
        parsed(payload)


@pytest.mark.parametrize('field,value', [('dist', 'nan'), ('v_rel', 'inf'), ('diameter', '0'),
                                         ('dist', '-1'), ('v_rel', True)])
def test_invalid_numbers_cannot_become_observations(payload, field, value):
    payload['data'][0][payload['fields'].index(field)] = value
    with pytest.raises(ValueError):
        parsed(payload)


def test_documented_empty_query_is_distinct_from_missing_source_response():
    from aneos_core.data.sources.cad import parse_cad_payload
    assert parse_cad_payload({'signature': {'version': '1.5'}, 'count': 0})['records'] == []
    for invalid in [None, {}, {'count': 0}, {'signature': {'version': '1.5'}, 'data': []}]:
        with pytest.raises(ValueError):
            parse_cad_payload(invalid)


@pytest.mark.asyncio
async def test_provider_outage_does_not_become_successful_empty_survey(tmp_path):
    p = poller(tmp_path)
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(side_effect=RuntimeError('synthetic outage')))
    result = await p.process_single_chunk(datetime(2020, 1, 1), datetime(2021, 1, 1), 0)
    assert result.success is False
    assert 'synthetic outage' in result.error_message
    assert result.intake_metrics['acquisition_status'] == 'unavailable'


@pytest.mark.asyncio
async def test_valid_empty_query_is_success_not_outage(tmp_path):
    p = poller(tmp_path)
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(return_value={'signature': {'version': '1.5'}, 'count': 0}))
    result = await p.process_single_chunk(datetime(2020, 1, 1), datetime(2021, 1, 1), 0)
    assert result.success is True
    assert result.object_count == 0
    assert result.intake_metrics['acquisition_status'] == 'complete'


@pytest.mark.asyncio
async def test_no_orbit_is_kept_unscored_and_positive_separate_orbit_advances(tmp_path, payload):
    from aneos_core.analysis.screening import ScreeningAnalysisService
    item = parsed(payload)['records'][0]
    p = poller(tmp_path)
    p.data_fetcher = SimpleNamespace(fetch_neo_data=Mock(side_effect=RuntimeError('synthetic orbit outage')))
    failed = await p._process_object_batch([copy.deepcopy(item)])
    assert len(failed['processed']) == 1
    assert failed['candidates'] == []
    kept = failed['processed'][0]
    assert kept['screening_assessment']['classification'] == 'INCONCLUSIVE'
    assert kept['screening_assessment']['screening_score'] is None
    assert kept['orbit_acquisition']['status'] == 'unavailable'
    neo = NEOData(item['designation'], orbital_elements=OrbitalElements(
        semi_major_axis=1, eccentricity=.2, inclination=80),
        physical_properties=PhysicalProperties(diameter_km=.011, diameter_uncertainty=.002))
    p.data_fetcher = SimpleNamespace(fetch_neo_data=Mock(return_value=neo))
    success = await p._process_object_batch([copy.deepcopy(item)])
    assert len(success['candidates']) == 1
    acquired = success['processed'][0]['neo_data']
    expected = ScreeningAnalysisService().analyze_neo(acquired).to_payload()
    assert success['processed'][0]['screening_assessment'] == expected
    assert acquired.first_observation is None  # Query window is not discovery/observation arc.
    assert acquired.nongrav is None
    assert neo.close_approaches == []  # No mutation of shared cached source object.
    assert acquired.close_approaches[0].epoch_jd_tdb == 2460000.5
    assert acquired.close_approaches[0].time_scale == 'TDB'


def test_event_merge_keeps_multiple_encounters_and_deduplicates_only_same_event(tmp_path, payload):
    p = poller(tmp_path)
    one = parsed(payload)['records'][0]
    payload['data'][0][payload['fields'].index('jd')] = '2460100.5'
    two = parsed(payload)['records'][0]
    merged = p._merge_chunks([[one], [copy.deepcopy(one), two]])
    assert len(merged) == 2
    assert len({m['encounter_id'] for m in merged}) == 2


@pytest.mark.asyncio
async def test_chunk_cache_preserves_all_encounters_and_refuses_legacy_cache(tmp_path, payload):
    p = poller(tmp_path)
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(return_value=payload))
    p.data_fetcher = SimpleNamespace(fetch_neo_data=Mock(return_value=NEOData('synthetic-marker')))
    start, end = datetime(2020, 1, 1), datetime(2021, 1, 1)
    result = await p.process_single_chunk(start, end, 0)
    assert result.object_count == 1
    assert len(result.chunk_data) == 1  # Unscored records cannot disappear from the next stage.
    assert result.candidate_count == 0
    cache_path = next(p.cache_dir.glob('*.json'))
    data = json.loads(cache_path.read_text())
    assert data['schema_version'] == 'cad-intake-v1'
    restored = p._load_cached_chunk(cache_path.stem)
    assert restored.chunk_data[0]['cad_evidence'] == result.chunk_data[0]['cad_evidence']
    assert isinstance(restored.start_date, datetime)
    legacy_path = p.cache_dir / 'legacy.json'
    legacy_path.write_text(json.dumps({'start_date': start.isoformat(), 'end_date': end.isoformat(),
        'object_count': 1, 'processed_count': 1, 'candidate_count': 1, 'processing_time_ms': 0,
        'success': True, 'chunk_data': [{'orbital_elements': {'a': 1, 'e': .1, 'i': 10}}]}))
    before = legacy_path.read_bytes()
    assert p._load_cached_chunk('legacy') is None
    assert legacy_path.read_bytes() == before


@pytest.fixture(autouse=True)
def no_unmarked_network(request, monkeypatch):
    if request.node.get_closest_marker('network'):
        yield
        return
    import socket
    attempts = []
    def deny(*args, **kwargs):
        attempts.append('network attempt')
        raise AssertionError('Offline CAD controls must not use live providers')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    yield
    assert attempts == []


@pytest.mark.asyncio
async def test_direct_request_and_real_base_adapter_paths_use_same_parser(tmp_path, payload, monkeypatch):
    import requests
    p = poller(tmp_path)
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = b"synthetic buffered response"
    response.json.return_value = payload
    request = Mock(return_value=response)
    monkeypatch.setattr(requests, 'get', request)
    direct = await p._fetch_nasa_cad_data(datetime(2020, 1, 1), datetime(2021, 1, 1))
    assert direct[0]['orbital_elements'] is None
    assert direct[0]['close_approach']['orbit_id'] == '42'
    assert 'aNEOS/' in request.call_args.kwargs['headers']['User-Agent']
    assert request.call_args.kwargs['params']['diameter'] == 'true'
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(return_value=payload))
    base = await p._call_enhanced_poller(datetime(2020, 1, 1), datetime(2021, 1, 1))
    assert base['neos'][0]['close_approach'] == direct[0]['close_approach']
    assert base['metadata']['retrieved_at'] is None  # A base cache does not identify HTTP retrieval time.
    assert direct.metadata['retrieved_at']


def test_data_fetcher_uses_labels_preserves_native_tdb_and_reports_query_coverage(payload, monkeypatch):
    import requests
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.models import CloseApproach
    payload['total'] = 20
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = b"synthetic buffered response"
    response.json.return_value = payload
    monkeypatch.setattr(requests, 'get', Mock(return_value=response))
    fetcher = DataFetcher.__new__(DataFetcher)
    approaches = fetcher._fetch_close_approaches('synthetic-marker')
    assert approaches.query_metadata['completeness_status'] == 'partial'
    ca = approaches[0]
    assert ca.epoch_jd_tdb == 2460000.5
    assert ca.time_scale == 'TDB'
    assert ca.close_approach_date.tzinfo is None  # Never silently retagged UTC.
    assert ca.body == 'Earth'
    assert CloseApproach.from_dict(ca.to_dict()).to_dict() == ca.to_dict()
    response.raise_for_status.side_effect = requests.HTTPError('synthetic 503')
    from aneos_core.utils.errors import DataSourceUnavailableError
    with pytest.raises(DataSourceUnavailableError, match='synthetic 503'):
        fetcher._fetch_close_approaches('synthetic-marker')


@pytest.mark.asyncio
async def test_repeated_encounters_share_orbit_fetch_not_source_mutation(tmp_path, payload):
    p = poller(tmp_path)
    neo = NEOData('synthetic-marker', orbital_elements=OrbitalElements(semi_major_axis=1, eccentricity=.2, inclination=80))
    fetch = Mock(return_value=neo)
    p.data_fetcher = SimpleNamespace(fetch_neo_data=fetch)
    first = parsed(payload)['records'][0]
    payload['data'][0][2] = '2460100.5'
    second = parsed(payload)['records'][0]
    result = await p._process_object_batch([first, second])
    fetch.assert_called_once_with('synthetic-marker')
    assert len(result['processed']) == 2
    assert neo.close_approaches == []
    assert first['neo_data'] is not second['neo_data']


@pytest.mark.asyncio
async def test_orbit_outage_is_not_retried_for_every_encounter(tmp_path, payload):
    p = poller(tmp_path)
    fetch = Mock(side_effect=RuntimeError('synthetic outage'))
    p.data_fetcher = SimpleNamespace(fetch_neo_data=fetch)
    record = parsed(payload)['records'][0]
    result = await p._process_object_batch([copy.deepcopy(record), copy.deepcopy(record)])
    fetch.assert_called_once()
    assert len(result['processed']) == 2
    assert all(r['orbit_acquisition']['status'] == 'unavailable' for r in result['processed'])
    assert all(r['screening_assessment']['screening_score'] is None for r in result['processed'])


@pytest.mark.asyncio
async def test_chunk_retry_backoff_does_not_hide_failure_and_keeps_positive_recovery(tmp_path, monkeypatch):
    from aneos_core.polling.historical_chunked_poller import ChunkResult
    import aneos_core.polling.historical_chunked_poller as module
    p = poller(tmp_path)
    start, end = datetime(2020, 1, 1), datetime(2021, 1, 1)
    failure = ChunkResult(start, end, 0, 0, 0, 0, False, error_message='synthetic outage')
    success = ChunkResult(start, end, 1, 1, 0, 0, True)
    p.process_single_chunk = AsyncMock(side_effect=[failure, success])
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    assert (await p._process_chunk_with_retries(start, end, 0)).success
    sleep.assert_awaited_once_with(1.)
    assert p.process_single_chunk.await_count == 2


def test_all_chunk_outages_cannot_become_empty_successful_pipeline():
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline
    from aneos_core.utils.errors import DataSourceUnavailableError
    p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    p.logger = logging.getLogger(__name__)
    with pytest.raises(DataSourceUnavailableError, match='not an empty successful survey'):
        p._extract_raw_objects(SimpleNamespace(chunk_results=[SimpleNamespace(success=False)]))


@pytest.mark.asyncio
async def test_current_chunk_cache_roundtrip_and_corruption_refusal(tmp_path, payload):
    p = poller(tmp_path)
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(return_value=payload))
    p.data_fetcher = SimpleNamespace(fetch_neo_data=Mock(return_value=NEOData('synthetic-marker',
        orbital_elements=OrbitalElements(semi_major_axis=1, eccentricity=.2, inclination=80),
        physical_properties=PhysicalProperties(diameter_km=.011, diameter_uncertainty=.002))))
    result = await p.process_single_chunk(datetime(2020, 1, 1), datetime(2021, 1, 1), 0)
    path = next(p.cache_dir.glob('*.json'))
    restored = p._load_cached_chunk(path.stem)
    assert restored.chunk_data[0]['screening_assessment'] == result.chunk_data[0]['screening_assessment']
    assert restored.chunk_data[0]['neo_data'].physical_properties.diameter_uncertainty == .002
    data = json.loads(path.read_text())
    data['chunk_data'][0]['screening_assessment']['candidate'] = False
    path.write_text(json.dumps(data))
    assert p._load_cached_chunk(path.stem) is None


@pytest.mark.asyncio
async def test_no_random_scoring_and_same_data_matches_canonical_service(tmp_path):
    from aneos_core.analysis.screening import ScreeningAnalysisService
    p = poller(tmp_path)
    item = {'designation': 'synthetic-control', 'orbital_elements': {
        'semi_major_axis': 1, 'eccentricity': .2, 'inclination': 80}}
    first = await p._apply_xviii_swarm_first_stage(item)
    second = await p._apply_xviii_swarm_first_stage(item)
    assert first == second
    assert first['candidate'] is True
    assert first['overall_score'] > 5
    assert first['artificial_probability'] is None
    assert first['confidence'] is None
    assert p._simple_artificial_screening(item) is True


@pytest.mark.network
def test_live_ky26_bounded_cad_contract():
    import requests
    from aneos_core.data.sources.cad import parse_cad_payload
    from aneos_core.data.sources.base import _APPLICATION_HEADERS
    query = {'des': '1998 KY26', 'date-min': '1900-01-01', 'date-max': '2100-01-01',
             'limit': 1, 'diameter': 'true', 'body': 'Earth'}
    response = requests.get('https://ssd-api.jpl.nasa.gov/cad.api', params=query, headers=_APPLICATION_HEADERS, timeout=30)
    response.raise_for_status()
    batch = parse_cad_payload(response.json(), query=query, require_complete=False)
    assert batch['metadata']['api_version'] == '1.5'
    assert batch['records'], 'Positive live KY26 encounter control unavailable'
    for item in batch['records']:
        assert item['orbital_elements'] is None
        assert item['cad_evidence']['evidence_basis'] == 'catalogue_prediction'
        assert item['close_approach']['time_scale'] == 'TDB'


def test_model_and_fetcher_cache_preserve_all_available_physical_fields():
    from aneos_core.data.fetcher import DataFetcher
    physical = PhysicalProperties(diameter_km=.011, diameter_uncertainty=.002,
        mass_kg=1000, density_g_cm3=2., albedo=.52, albedo_uncertainty=.08,
        rotation_period_hours=.09, rotation_period_uncertainty=.0001,
        pole_ecliptic_lat=45, pole_ecliptic_lon=100, color_indices={'marker': .1},
        thermal_inertia=100, emissivity=.9, reference_source='synthetic-source')
    neo = NEOData('synthetic-control', physical_properties=physical)
    fetcher = DataFetcher.__new__(DataFetcher)
    serialized = fetcher._serialize_neo_data(neo)
    assert serialized['physical_properties'] == physical.to_dict()
    restored = fetcher._deserialize_neo_data(serialized)
    assert restored.physical_properties.to_dict() == physical.to_dict()
    restored.physical_properties.color_indices['marker'] = .9
    assert physical.color_indices['marker'] == .1
    assert serialized['physical_properties']['color_indices']['marker'] == .1


@pytest.mark.asyncio
async def test_intake_to_actual_first_stage_and_required_scientific_gate(tmp_path, payload):
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline, PipelineConfig
    from aneos_core.validation.multi_stage_validator import MultiStageValidator
    p = poller(tmp_path)
    p.base_poller = SimpleNamespace(fetch_cad_data=Mock(return_value=payload))
    p.data_fetcher = SimpleNamespace(fetch_neo_data=Mock(return_value=NEOData('synthetic-marker',
        orbital_elements=OrbitalElements(semi_major_axis=1, eccentricity=.2, inclination=80))))
    chunk = await p.process_single_chunk(datetime(2020, 1, 1), datetime(2021, 1, 1), 0)
    pipeline = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    pipeline.config = PipelineConfig()
    pipeline.logger = logging.getLogger(__name__)
    pipeline.xviii_swarm_scorer = None
    pipeline.multi_stage_validator = MultiStageValidator.__new__(MultiStageValidator)
    pipeline.multi_stage_validator.logger = logging.getLogger(__name__)
    raw = pipeline._extract_raw_objects(SimpleNamespace(chunk_results=[chunk]))
    first = await pipeline._stage_first_review(raw)
    assert first.output_count == 1  # Real shared rank positive control, not all-denial success.
    gate = await pipeline._stage_multi_validation(first.candidates)
    assert gate.output_count == 0  # No physical/calibrated model is installed.
    assert first.candidates[0]['multi_stage_validation']['validation_status'] == 'inconclusive'
    assert first.candidates[0]['neo_data'].close_approaches[0].epoch_jd_tdb == 2460000.5


def test_historical_reports_keep_all_outage_reasons_and_preserve_prior_writes(tmp_path):
    from aneos_core.polling.historical_chunked_poller import ChunkResult, HistoricalPollingResult
    p = poller(tmp_path)
    start, end = datetime(2020, 1, 1), datetime(2021, 1, 1)
    chunks = [ChunkResult(start, end, 0, 0, 0, 0, False, error_message=f'synthetic outage{i}',
        intake_metrics={'acquisition_status': 'unavailable'}) for i in range(12)]
    result = HistoricalPollingResult(1, 12, 0, 0, start, end, chunks, {'failed_chunks': 12})
    p._save_historical_results(result)
    first = next(p.results_dir.glob('*.json'))
    before = first.read_bytes()
    p._save_historical_results(result)
    assert len(list(p.results_dir.glob('*.json'))) == 2
    assert first.read_bytes() == before
    saved = json.loads(before)
    assert len(saved['chunk_summaries']) == 12
    assert saved['chunk_summaries'][11]['error_message'] == 'synthetic outage11'
    assert saved['evidence_storage']['note'].startswith('Operational')


def test_cad_field_removal_and_ambiguous_limit_are_explicit_failures(payload):
    from aneos_core.data.sources.cad import parse_cad_payload
    fields, row = payload['fields'], payload['data'][0]
    index = fields.index('jd')
    fields.pop(index)
    row.pop(index)
    with pytest.raises(ValueError, match='schema fields'):
        parsed(payload)


def test_event_identity_does_not_round_native_julian_dates(payload):
    first = parsed(payload)['records'][0]
    payload['data'][0][2] = '2460000.50000000001'
    second = parsed(payload)['records'][0]
    assert first['encounter_id'] != second['encounter_id']
    assert second['cad_evidence']['raw_row'][2] == '2460000.50000000001'


def test_validation_import_is_offline_in_a_fresh_process():
    import subprocess
    import sys
    code = '''import socket
attempts = []
def deny(*a, **kw):
    attempts.append(True)
    raise AssertionError("unexpected import-time network")
socket.socket.connect = deny
import aneos_core.validation
assert not attempts
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_lazy_gaia_query_keeps_authorized_client_positive_control(monkeypatch):
    from aneos_core.validation import gaia_astrometric_calibration as module
    row = dict(source_id=1, ra=10., dec=20., ra_error=1., dec_error=1., pmra=None, pmdec=None,
        pmra_error=None, pmdec_error=None, parallax=None, parallax_error=None,
        phot_g_mean_mag=19., bp_rp=None, ruwe=None, astrometric_excess_noise=None)
    client = SimpleNamespace(launch_job_async=Mock(return_value=SimpleNamespace(get_results=lambda: [row])))
    monkeypatch.setattr(module, 'Gaia', client)
    calibrator = module.GaiaAstrometricCalibrator.__new__(module.GaiaAstrometricCalibrator)
    calibrator.config = {'gaia_table': 'synthetic_table', 'reference_epoch': 2016., 'max_sources': 1}
    calibrator.logger = logging.getLogger(__name__)
    calibrator._get_cached_result = Mock(return_value=None)
    calibrator._cache_result = Mock()
    calibrator._query_times = []
    calibrator._cache_misses = 0
    result = await calibrator._query_gaia_catalog(10., 20., 5.)
    assert len(result) == 1
    assert result[0].source_id == 1
    assert client.ROW_LIMIT == 1
    assert 'FROM synthetic_table' in client.launch_job_async.call_args.args[0]
    calibrator._cache_result.assert_called_once()


def test_missing_total_at_limit_does_not_claim_complete_query(payload):
    from aneos_core.data.sources.cad import parse_cad_payload
    with pytest.raises(ValueError, match='completeness unverified'):
        parse_cad_payload(payload, query={'limit': 1})


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,expected', [('error', 'error'), ('partial', 'partial'), ('complete', 'success')])
async def test_workflow_wrapper_does_not_relabel_pipeline_failures_as_success(mode, expected):
    from aneos_core.integration.pipeline_integration import PipelineIntegration
    now = datetime.now()
    wrapper = PipelineIntegration.__new__(PipelineIntegration)
    wrapper.components_initialized = True
    wrapper.console = None
    wrapper.logger = logging.getLogger(__name__)
    metrics = {'error': 'synthetic source outage'} if mode == 'error' else {}
    stage = SimpleNamespace(success=mode != 'partial', stage_metrics={})
    result = SimpleNamespace(total_input_objects=0, final_candidates=0, pipeline_metrics=metrics,
        processing_start_time=now, processing_end_time=now, stage_results={'stage': stage})
    wrapper.automatic_pipeline = SimpleNamespace(run_complete_pipeline=AsyncMock(return_value=result))
    output = await wrapper.run_historical_polling_workflow(1, interactive=False)
    assert output['status'] == expected
    assert output['scientific_validation_status'] == 'not_established'
    if mode == 'error':
        assert output['error_message'] == 'synthetic source outage'


def test_provider_payload_and_metadata_mutation_cannot_change_frozen_record(payload):
    batch = parsed(payload)
    item = batch['records'][0]
    before = copy.deepcopy(item)
    payload['signature']['version'] = 'changed'
    payload['data'][0][4] = '999'
    batch['metadata']['signature']['source'] = 'changed'
    assert item == before
