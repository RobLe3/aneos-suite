"""Synthetic Sentry controls; no live provider, origin or impact-model validation."""
import copy
import json
import socket
from unittest.mock import Mock

import pytest


def module():
    from aneos_core.data.sources import sentry
    return sentry


def payload(method='IOBS'):
    diagnostics = {'IOBS': {'sigma_vi': '1.2'}, 'MC': {'sigma_mc': '1.3'},
        'LOV': {'dist': '.8', 'width': '.1', 'sigma_imp': '2', 'sigma_lov': '-1', 'stretch': '4'}}
    return {'signature': {'version': '2.0', 'source': 'NASA/JPL Sentry Data API'},
        'summary': {'des': '1998 KY26', 'fullname': '(1998 KY26)', 'method': method,
            'ip': '3e-7', 'n_imp': '2', 'first_obs': '1998-05-31', 'last_obs': '2024-05-06',
            'cdate': '2024-05-07 13:14:15', 'pdate': '2024-05-07 20:14:15',
            'diameter': '.011', 'mass': '1000', 'energy': '.02', 'ts_max': '0', 'ps_cum': '-6'},
        'data': [{'date': '2050-01-01.25', 'ip': '1e-7', 'energy': '.01', 'ps': '-7', 'ts': '0', **diagnostics[method]},
                 {'date': '2060-02-29.50', 'ip': '2e-7', 'energy': '.03', 'ps': '-6', 'ts': '0', **diagnostics[method]}]}


def encode(data):
    return json.dumps(data, separators=(',', ':')).encode()


def report(data=None, designation='1998 KY26'):
    return module().parse_sentry_response(encode(payload() if data is None else data),
        designation=designation, retrieved_at='2026-11-12T20:30:00+00:00')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('External provider forbidden')))


@pytest.mark.parametrize('method', ['IOBS', 'MC', 'LOV'])
def test_real_source_contract_positive_preserves_method_events_and_limits(method):
    data = payload(method)
    result = report(data)
    assert result['status'] == 'available'
    assert result['collision_probability'] == 3e-7
    assert result['assessment_basis'] == 'authoritative_external_model_output'
    assert result['source']['method'] == method
    assert result['source']['retrieved_at'] == '2026-11-12T20:30:00+00:00'
    assert result['source']['computation_time']['time_scale'] == 'Pacific Local'
    assert result['time_scope']['last_listed_event_utc'] == '2060-02-29.50'
    assert result['time_scope']['semantics'] == 'tabulated_virtual_impactors_only'
    assert result['probability_uncertainty'] is None and result['calculation_confidence'] is None
    assert result['moon_collision_probability'] is None and result['risk_level'] == 'not_classified_by_aneos'
    assert 'factor of ten' in ' '.join(result['limitations'])
    assert result['physical_estimate_basis']['diameter'] == 'source_estimate_not_assumed_independent_measurement'
    assert module().verify_sentry_report(result)
    assert json.loads(result['source_snapshot']['raw_utf8']) == data


@pytest.mark.parametrize('error,status,extra', [
    ('specified object not found', 'not_listed', {}),
    ('specified object removed', 'removed', {'removed': '2020-01-01 12:00:00'})])
def test_absent_removed_are_distinct_not_zero_or_negligible(error, status, extra):
    data = {'signature': payload()['signature'], 'error': error, **extra}
    result = report(data)
    assert result['status'] == status and result['collision_probability'] is None
    assert result['risk_level'] == 'not_assessed'
    assert result['identity']['verification'] == 'query_bound_no_object_row'
    assert module().verify_sentry_report(result)


@pytest.mark.parametrize('mutate', [
    lambda p: p.pop('signature'), lambda p: p['signature'].update(version='3.0'),
    lambda p: p['signature'].update(source='unrelated'),
    lambda p: p['summary'].update(des='2020 SO'), lambda p: p['summary'].update(method='new-method'),
    lambda p: p['summary'].pop('method'), lambda p: p['summary'].pop('ip'),
    lambda p: p['summary'].update(ip='nan'), lambda p: p['summary'].update(ip=True),
    lambda p: p['summary'].update(ip='1.1'), lambda p: p['summary'].update(ip='1e-9999'),
    lambda p: p['summary'].update(n_imp='3'), lambda p: p.update(data=[]),
    lambda p: p['data'][0].update(date='2050-02-30.2'),
    lambda p: p['data'][0].pop('sigma_vi'), lambda p: p['data'][0].update(method='MC'),
    lambda p: p['data'][0].update(ip='-1'), lambda p: p['summary'].update(first_obs='2025-01-01'),
])
def test_invalid_schema_identity_method_numbers_counts_or_dates_fail_closed(mutate):
    data = payload(); mutate(data)
    with pytest.raises(ValueError): report(data)


@pytest.mark.parametrize('data', [
    {'error': 'unspecified database failure'}, {'error': 'specified object removed'},
    {'error': 'specified object not found', 'summary': {'des': '1998 KY26'}}])
def test_error_payload_does_not_launder_inconsistent_or_unknown_state(data):
    data['signature'] = payload()['signature']
    with pytest.raises(ValueError): report(data)


def test_duplicate_keys_and_nonstandard_json_are_rejected():
    for raw in [b'{"signature":{},"signature":{}}', b'{"bad":NaN}']:
        with pytest.raises(ValueError):
            module().parse_sentry_response(raw, designation='1998 KY26', retrieved_at='2026-11-12T00:00:00Z')


@pytest.mark.parametrize('field', ['collision_probability', 'status', 'source', 'virtual_impactors', 'limitations'])
def test_report_cannot_be_tampered_into_scientific_evidence(field):
    result = report(); result[field] = 'tampered'
    assert not module().verify_sentry_report(result)


def test_unknown_or_alias_identity_is_not_silently_attributed():
    for name in ['Apophis', '99942 Apophis', '', 'test;invalid', '../data']:
        with pytest.raises(ValueError): report(designation=name)
    assert module().normalize_designation('1998ky26') == '1998 KY26'
    assert module().normalize_designation('a99942') == '99942'


def test_real_numeric_zero_is_not_missing_or_inferred():
    data = payload(); data['summary']['ip'] = '0'
    for row in data['data']: row['ip'] = '0'
    assert report(data)['collision_probability'] == 0


def response(data):
    reply = Mock()
    reply.status_code = 200
    reply.headers = {}
    reply.content = encode(data)
    return reply


def test_actual_service_uses_guarded_transport_and_reuses_frozen_snapshot(monkeypatch):
    import requests
    request = Mock(return_value=response(payload()))
    monkeypatch.setattr(requests, 'get', request)
    service = module().SentryImpactService()
    first = service.assess('1998 KY26')
    second = service.assess('1998ky26')
    assert first == second and first is not second
    assert request.call_count == 1
    assert request.call_args.kwargs['params'] == {'des': '1998 KY26'}
    assert request.call_args.kwargs['allow_redirects'] is False
    request.return_value.close.assert_called_once()
    second['collision_probability'] = .99
    assert service.assess('1998 KY26')['collision_probability'] == 3e-7
    service.assess('1998 KY26', force_refresh=True)
    assert request.call_count == 2


def test_concurrent_same_object_callers_share_one_request(monkeypatch):
    import requests
    import time
    from concurrent.futures import ThreadPoolExecutor
    def send(*a, **k):
        time.sleep(.03)
        return response(payload())
    request = Mock(side_effect=send)
    monkeypatch.setattr(requests, 'get', request)
    service = module().SentryImpactService()
    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = list(pool.map(lambda _: service.assess('1998 KY26'), range(6)))
    assert request.call_count == 1 and all(module().verify_sentry_report(r) for r in reports)


def test_failed_refresh_cannot_return_old_probability_as_current(monkeypatch):
    import requests
    request = Mock(return_value=response(payload()))
    monkeypatch.setattr(requests, 'get', request)
    service = module().SentryImpactService()
    historical = service.assess('1998 KY26')
    request.side_effect = requests.Timeout('synthetic outage')
    with pytest.raises(module().provider_http.ProviderUnavailableError):
        service.assess('1998 KY26', force_refresh=True)
    assert module().verify_sentry_report(historical)
    assert not service._pending


def test_unknown_source_schema_is_not_cached_as_absent(monkeypatch):
    import requests
    data = payload(); data['signature']['version'] = 'unknown'
    monkeypatch.setattr(requests, 'get', Mock(return_value=response(data)))
    service = module().SentryImpactService()
    with pytest.raises(module().SentryResponseError): service.assess('1998 KY26')
    assert not service._cache and not service._pending


@pytest.fixture
def http_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aneos_api.endpoints import analysis
    app = FastAPI()
    app.include_router(analysis.router, prefix='/api/v1/analysis')
    app.dependency_overrides[analysis.get_current_user] = lambda: None
    service = Mock()
    service.assess.return_value = report()
    app.dependency_overrides[analysis.get_sentry_impact_service] = lambda: service
    with TestClient(app) as client:
        yield client, service


def test_existing_rest_route_has_positive_complete_replay_and_no_heuristic(http_client, monkeypatch):
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    monkeypatch.setattr(ImpactProbabilityCalculator, 'calculate_comprehensive_impact_probability',
                        Mock(side_effect=AssertionError('legacy calculator invoked')))
    client, service = http_client
    result = client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26', 'force_refresh': 'true'})
    assert result.status_code == 200, result.text
    assert result.json() == report()
    assert module().verify_sentry_report(result.json())
    service.assess.assert_called_once_with('1998 KY26', force_refresh=True)


@pytest.mark.parametrize('state', ['specified object not found', 'specified object removed'])
def test_rest_unlisted_or_removed_is_not_zero_risk(http_client, state):
    client, service = http_client
    service.assess.return_value = report({'signature': payload()['signature'], 'error': state,
        **({'removed': '2020-01-01 12:00:00'} if state.endswith('removed') else {})})
    result = client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26'})
    assert result.status_code == 200
    assert result.json()['collision_probability'] is None
    assert result.json()['calculation_confidence'] is None
    assert result.json()['risk_level'] == 'not_assessed'


@pytest.mark.parametrize('error,code', [(lambda: module().SentryResponseError('invalid'), 502),
    (lambda: module().provider_http.ProviderUnavailableError('outage'), 503),
    (lambda: RuntimeError('unexpected secret internal detail'), 500)])
def test_rest_failures_are_explicit_not_natural_or_zero(http_client, error, code):
    client, service = http_client
    service.assess.side_effect = error()
    result = client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26'})
    assert result.status_code == code
    assert 'collision_probability' not in result.json()
    assert 'secret' not in result.text


def test_rest_bad_selector_or_tampered_provider_report_cannot_pass(http_client):
    client, service = http_client
    assert client.get('/api/v1/analysis/impact', params={'designation': 'Apophis'}).status_code == 422
    service.assess.assert_not_called()
    service.assess.return_value['collision_probability'] = .99
    assert client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26'}).status_code == 502


def test_markdown_shows_scope_limits_and_missing_inference_without_fake_interval():
    from aneos_core.analysis.impact_evidence import render_sentry_markdown
    text = render_sentry_markdown(report())
    assert '3e-07' in text and '2050-01-01.25' in text
    assert 'factor of ten' in text and 'not assessed' in text
    assert '95% CI' not in text and 'Pacific Local' in text
    data = report(); data['collision_probability'] = 1
    with pytest.raises(ValueError): render_sentry_markdown(data)


def test_rest_rejects_valid_report_for_different_requested_object(http_client):
    client, service = http_client
    data = payload(); data['summary']['des'] = '2020 SO'
    service.assess.return_value = report(data, designation='2020 SO')
    result = client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26'})
    assert result.status_code == 502


def test_refresh_failure_is_shared_with_waiters_not_replaced_with_stale_success(monkeypatch):
    import requests
    import threading
    from concurrent.futures import ThreadPoolExecutor
    request = Mock(return_value=response(payload()))
    monkeypatch.setattr(requests, 'get', request)
    service = module().SentryImpactService()
    service.assess('1998 KY26')
    entered = threading.Event(); release = threading.Event()
    def failing(*a, **kw):
        entered.set()
        assert release.wait(5)
        raise requests.Timeout('synthetic refresh outage')
    request.side_effect = failing
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.assess, '1998 KY26', force_refresh=True)
        assert entered.wait(5)
        waiter_entered = threading.Event()
        shared = service._pending['1998 KY26']
        original_result = shared.result
        def observe_waiter(*args, **kwargs):
            waiter_entered.set()
            return original_result(*args, **kwargs)
        monkeypatch.setattr(shared, 'result', observe_waiter)
        second = pool.submit(service.assess, '1998 KY26', force_refresh=True)
        assert waiter_entered.wait(5)
        release.set()
        for future in (first, second):
            with pytest.raises(module().provider_http.ProviderUnavailableError): future.result(timeout=10)
    assert request.call_count == 2  # One initial success and one failed refresh.


def test_actual_application_route_uses_real_service_and_guarded_http(monkeypatch):
    import requests
    from fastapi.testclient import TestClient
    from aneos_api.app import create_app
    from aneos_api.endpoints import analysis
    request = Mock(return_value=response(payload()))
    monkeypatch.setattr(requests, 'get', request)
    app = create_app()
    app.dependency_overrides[analysis.get_current_user] = lambda: None
    app.dependency_overrides[analysis.get_sentry_impact_service] = lambda: module().SentryImpactService()
    client = TestClient(app)  # Routing/serialization only; no service lifespan boot.
    assert '/api/v1/analysis/impact' in app.openapi()['paths']
    result = client.get('/api/v1/analysis/impact', params={'designation': '1998 KY26'})
    assert result.status_code == 200, result.text
    assert module().verify_sentry_report(result.json())
    assert request.call_count == 1


def test_expired_snapshot_is_retrieved_again_without_overwriting_returned_history(monkeypatch):
    import requests
    request = Mock(return_value=response(payload()))
    monkeypatch.setattr(requests, 'get', request)
    service = module().SentryImpactService()
    first = service.assess('1998 KY26')
    expires, snapshot = service._cache['1998 KY26']
    service._cache['1998 KY26'] = (0, snapshot)
    changed = payload(); changed['summary']['ip'] = '4e-7'
    request.return_value = response(changed)
    second = service.assess('1998 KY26')
    assert request.call_count == 2
    assert first['collision_probability'] == 3e-7 and second['collision_probability'] == 4e-7
    assert first['source_snapshot']['sha256'] != second['source_snapshot']['sha256']
    assert module().verify_sentry_report(first) and module().verify_sentry_report(second)


def test_source_data_does_not_use_fetched_orbit_or_invent_covariance(monkeypatch):
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.analysis.impact_evidence import render_sentry_markdown
    monkeypatch.setattr(DataFetcher, 'fetch_neo_data', Mock(side_effect=AssertionError('unrelated catalogue query')))
    result = report()
    assert 'covariance' not in result['source']
    assert 'not a local aNEOS fit' in render_sentry_markdown(result)
    # The external result does not validate/reuse an incompatible local orbit.
    assert result['assessment_basis'] == 'authoritative_external_model_output'


def test_dated_ky26_actual_source_capture_is_not_the_historical_score_or_zero_risk():
    from pathlib import Path
    directory = Path(__file__).parents[1] / 'docs/case-studies/1998-KY26/sentry/20261112T211912Z'
    stored = json.loads((directory / 'report.json').read_text())
    execution = json.loads((directory / 'execution.json').read_text())
    replay = module().parse_sentry_response((directory / 'raw-response.json').read_bytes(),
        designation='1998 KY26', retrieved_at=stored['source']['retrieved_at'])
    assert replay == stored and module().verify_sentry_report(stored)
    assert stored['source_snapshot']['sha256'] == execution['raw_response_sha256']
    assert stored['status'] == 'not_listed' and stored['collision_probability'] is None
    assert stored['risk_level'] == 'not_assessed'
    assert 'screening_score' not in stored
    import hashlib
    for filename, digest in execution['source_code_sha256'].items():
        archived = directory / 'acquisition-code' / (Path(filename).name + '.txt')
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest


def test_published_openapi_snapshot_matches_current_registered_application():
    from pathlib import Path
    from aneos_api.app import create_app
    stored = json.loads((Path(__file__).parents[1] / 'docs/api/openapi.json').read_text())
    actual = create_app().openapi()
    assert stored == actual
    parameters = actual['paths']['/api/v1/analysis/impact']['get']['parameters']
    assert any(p['name'] == 'force_refresh' for p in parameters)
    assert 'source_snapshot' in actual['components']['schemas']['ImpactResponse']['properties']


def test_actual_mc_source_native_fractional_observation_dates_are_preserved():
    """Real captured response; the test timestamp is synthetic, not a capture claim."""
    from pathlib import Path
    raw = (Path(__file__).parent / 'fixtures/sentry_101955_20261112.json').read_bytes()
    import hashlib
    metadata = json.loads((Path(__file__).parent / 'fixtures/sentry_101955_20261112.metadata.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == metadata['raw_response_sha256']
    assert metadata['exact_response_retrieval_timestamp'] is None
    data = module().parse_sentry_response(raw, designation='101955', retrieved_at='2026-11-13T00:00:00Z')
    assert data['status'] == 'available' and data['source']['method'] == 'MC'
    assert len(data['virtual_impactors']) == 157
    assert data['source']['first_observation_utc'] == '1999-09-11.40624'
    assert data['source']['last_observation_utc'] == '2020-10-3.80160'
    assert data['source_torino_max'] is None  # Beyond the applicable source scale, not zero.
    assert data['collision_probability'] == .000571699999999996
    assert module().verify_sentry_report(data)


def test_actual_captured_mc_response_survives_guarded_service_and_http(http_client, monkeypatch):
    import requests
    from pathlib import Path
    client, service = http_client
    raw = (Path(__file__).parent / 'fixtures/sentry_101955_20261112.json').read_bytes()
    reply = response(payload()); reply.content = raw
    monkeypatch.setattr(requests, 'get', Mock(return_value=reply))
    actual = module().SentryImpactService()
    service.assess.side_effect = actual.assess
    result = client.get('/api/v1/analysis/impact', params={'designation': '101955'})
    assert result.status_code == 200, result.text
    assert module().verify_sentry_report(result.json())
    assert result.json()['source']['method'] == 'MC'
    assert result.json()['time_scope']['event_count'] == 157
    assert result.json()['source_torino_max'] is None


def test_first_concurrent_public_getters_construct_only_one_source_service(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from aneos_core.analysis import impact_evidence
    original = module().SentryImpactService
    def construct():
        time.sleep(.03)
        return original()
    factory = Mock(side_effect=construct)
    monkeypatch.setattr(module(), 'SentryImpactService', factory)
    monkeypatch.setattr(impact_evidence, '_sentry_service_instance', None, raising=False)
    if hasattr(impact_evidence.get_sentry_impact_service, 'cache_clear'):
        impact_evidence.get_sentry_impact_service.cache_clear()
    with ThreadPoolExecutor(max_workers=6) as pool:
        instances = list(pool.map(lambda _: impact_evidence.get_sentry_impact_service(), range(6)))
    assert factory.call_count == 1
    assert all(item is instances[0] for item in instances)
