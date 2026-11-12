"""Provider-shaped synthetic controls; no live JPL request during collection."""
import copy
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from aneos_core.config.settings import APIConfig
from aneos_core.data.sources.sbdb import SBDBSource
from aneos_core.data.models import NEOData, NonGravitationalParameters


@pytest.fixture
def payload():
    labels = ['e', 'q', 'tp', 'node', 'peri', 'i', 'A1', 'A2', 'A3']
    sigma = np.array([1e-7, 2e-7, .001, .001, .001, .001, 2e-11, 3e-14, 4e-12])
    corr = np.eye(9); corr[6, 8] = corr[8, 6] = .4
    return {'signature': {'version': '1.3', 'source': 'synthetic SBDB contract fixture'},
            'object': {'des': 'SYNTHETIC', 'spkid': 'synthetic-marker'},
            'orbit': {'orbit_id': 'synthetic-solution', 'epoch': '2461200.5',
                      'first_obs': '2000-01-01', 'last_obs': '2024-01-01',
                      'elements': [{'name': k, 'value': v} for k, v in [('a', '1.2'), ('e', '.2'), ('i', '1.5')]],
                      'model_pars': [
                          {'name': 'A1', 'value': '1e-11', 'sigma': '2e-11', 'units': 'au/d^2', 'kind': 'EST', 'n': 1},
                          {'name': 'A2', 'value': '-2e-13', 'sigma': '3e-14', 'units': 'au/d^2', 'kind': 'EST', 'n': 2},
                          {'name': 'A3', 'value': '1e-12', 'sigma': '4e-12', 'units': 'au/d^2', 'kind': 'CON', 'n': 3},
                          {'name': 'R0', 'value': '1', 'sigma': None, 'units': 'au', 'kind': 'SET', 'n': 0}],
                      'covariance': {'labels': labels, 'epoch': '2457098.5',
                                     'data': (corr * np.outer(sigma, sigma)).tolist(),
                                     'elements': [{'name': 'e', 'value': '.21'}]}}}


def parsed_ng(payload):
    parsed = SBDBSource(APIConfig())._parse_sbdb_response(payload, 'SYNTHETIC')
    return NonGravitationalParameters.from_dict(parsed['_nongrav']), parsed


def test_three_components_keep_sigma_units_kind_and_model_constants(payload):
    ng, parsed = parsed_ng(payload)
    assert ng.a1 == 1e-11 and ng.a2 == -2e-13 and ng.a3 == 1e-12
    assert ng.parameters['A3'].sigma == 4e-12
    assert ng.parameters['A3'].kind == 'CON'
    assert ng.parameters['A3'].units == 'au/d^2'
    assert ng.parameters['A3'].raw_value == '1e-12'
    assert ng.parameters['R0'].kind == 'SET'
    assert ng.parameters['R0'].sigma is None
    assert ng.solution['epoch_jd_tdb'] == '2461200.5'
    assert ng.covariance.epoch_jd_tdb == '2457098.5'
    assert ng.covariance.elements[0]['value'] == '.21'
    assert parsed['_evidence_metadata']['schema_status'] == 'supported'


def test_parameter_units_convert_value_and_sigma_not_kind(payload):
    ng, _ = parsed_ng(payload)
    record = ng.parameters['A3'].in_si()
    factor = 149597870700 / 86400**2
    assert record['value'] == pytest.approx(1e-12 * factor)
    assert record['sigma'] == pytest.approx(4e-12 * factor)
    assert record['units'] == 'm/s^2'
    assert record['kind'] == 'CON'
    assert ng.parameters['A3'].value == 1e-12


def test_covariance_keeps_order_correlations_and_cross_terms(payload):
    ng, _ = parsed_ng(payload)
    cov = ng.covariance
    assert cov.correlation('A1', 'A3') == pytest.approx(.4)
    block = cov.select(['A3', 'A1'])
    assert block['labels'] == ['A3', 'A1']
    assert block['epoch_jd_tdb'] == '2457098.5'
    assert block['data'][0][1] == pytest.approx(.4 * 4e-12 * 2e-11)
    si = ng.acceleration_covariance_si()
    factor = 149597870700 / 86400**2
    assert si['data'][0][2] == pytest.approx(cov.data[6][8] * factor**2)
    assert si['labels'] == ['A1', 'A2', 'A3']
    assert si['epoch_jd_tdb'] != ng.solution['epoch_jd_tdb']


def test_cache_disk_roundtrip_retains_complete_evidence(payload, tmp_path):
    from aneos_core.data.cache import CacheManager
    ng, _ = parsed_ng(payload)
    neo = NEOData('SYNTHETIC', nongrav=ng)
    before = json.loads(json.dumps(neo.to_dict(), allow_nan=False))
    cache = CacheManager(tmp_path)
    cache.set('nongrav-evidence', before)
    restored = NEOData.from_dict(CacheManager(tmp_path).get('nongrav-evidence'))
    assert restored.nongrav.to_dict() == ng.to_dict()
    assert restored.nongrav.covariance.correlation('A1', 'A3') == pytest.approx(.4)


def test_missing_components_are_not_zero(payload):
    payload['orbit']['model_pars'] = []
    payload['orbit']['covariance'] = None
    parsed = SBDBSource(APIConfig())._parse_sbdb_response(payload, 'SYNTHETIC')
    assert parsed['_nongrav'] is None
    assert parsed['_evidence_metadata']['nongrav_status'] == 'not_available'


def test_missing_covariance_retains_parameters_without_inventing_correlation(payload):
    payload['orbit']['covariance'] = None
    ng, _ = parsed_ng(payload)
    assert ng.a3 == 1e-12
    assert ng.covariance is None
    assert ng.acceleration_covariance_si() is None


@pytest.mark.parametrize('mutate', [
    lambda p: p['orbit']['covariance']['labels'].pop(),
    lambda p: p['orbit']['covariance']['labels'].__setitem__(0, 'A1'),
    lambda p: p['orbit']['covariance']['data'][0].pop(),
    lambda p: p['orbit']['covariance']['data'][0].__setitem__(0, -1e-20),
    lambda p: p['orbit']['covariance']['data'][6].__setitem__(8, 9e-23),
    lambda p: p['orbit']['model_pars'][0].__setitem__('value', 'nan'),
    lambda p: p['orbit']['model_pars'][0].__setitem__('sigma', '-1'),
    lambda p: p['orbit']['model_pars'][0].__setitem__('kind', 'UNKNOWN'),
    lambda p: p['signature'].__setitem__('version', '99.0'),
])
def test_invalid_provider_evidence_never_silently_passes(payload, mutate):
    mutate(payload)
    with pytest.raises(ValueError):
        parsed_ng(payload)


def test_non_psd_covariance_rejected_after_unit_normalization(payload):
    cov = payload['orbit']['covariance']['data']
    cov[6][8] = cov[8][6] = 1.1 * np.sqrt(cov[6][6] * cov[8][8])
    with pytest.raises(ValueError, match='positive semidefinite'):
        parsed_ng(payload)


def test_absent_sigma_is_unknown_not_zero(payload):
    payload['orbit']['model_pars'][2]['sigma'] = None
    ng, _ = parsed_ng(payload)
    assert ng.parameters['A3'].sigma is None
    assert ng.parameters['A3'].in_si()['sigma'] is None


def test_unknown_native_units_preserved_but_conversion_rejected(payload):
    payload['orbit']['model_pars'][2]['units'] = 'unknown-provider-unit'
    ng, _ = parsed_ng(payload)
    assert ng.parameters['A3'].units == 'unknown-provider-unit'
    with pytest.raises(ValueError, match='units'):
        ng.parameters['A3'].in_si()


@pytest.mark.asyncio
async def test_one_documented_request_and_no_nongrav_probe(payload):
    source = SBDBSource(APIConfig())
    source._http_get = AsyncMock(return_value=payload)
    result = await source.fetch_orbital_elements('SYNTHETIC')
    assert result.success
    assert source._http_get.await_count == 1
    params = source._http_get.call_args.args[1]
    assert params['cov'] == 'mat'
    assert params['full-prec'] == 'true'
    assert 'nongrav' not in params


@pytest.mark.asyncio
async def test_bad_covariance_returns_explicit_fetch_failure(payload):
    payload['orbit']['covariance']['labels'].pop()
    source = SBDBSource(APIConfig())
    source._http_get = AsyncMock(return_value=payload)
    result = await source.fetch_orbital_elements('SYNTHETIC')
    assert result.success is False
    assert result.error_message
    assert result.data is None


def test_actual_source_adapter_to_cache_to_menu_detector_and_api_report(payload, tmp_path):
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_core.data.sources.base import FetchResult
    from aneos_core.detection.detection_manager import DetectionManager, DetectorType
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_api.schemas.detection import DetectionResponse
    source = SBDBSource(APIConfig())
    parsed = source._parse_sbdb_response(payload, 'SYNTHETIC')
    source.health_check = AsyncMock(return_value=True)
    source.fetch_orbital_elements = AsyncMock(return_value=FetchResult(success=True, data=parsed))
    fetcher = DataFetcher(cache_manager=CacheManager(tmp_path))
    neo = fetcher._fetch_from_source('SBDB', source, 'SYNTHETIC')
    assert neo is not None
    assert neo.nongrav.a3 == 1e-12
    serialized = fetcher._serialize_neo_data(neo)
    restored = fetcher._deserialize_neo_data(serialized)
    assert restored.nongrav.to_dict() == neo.nongrav.to_dict()
    assert restored.evidence_metadata['solution']['orbit_id'] == 'synthetic-solution'
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    extra = menu._build_additional_dict(restored)
    report = extra['non_gravitational_evidence']
    assert report['components']['A3']['native']['sigma'] == 4e-12
    assert report['covariance_applies_to_display_epoch'] is False
    manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
    detected = manager.analyze_neo({'a': 1.2, 'e': .2, 'i': 1.5}, additional_data=extra)
    assert detected.metadata['non_gravitational_evidence'] == report
    # Merely attaching a fitted coefficient must not activate a thrust heuristic.
    baseline = manager.analyze_neo({'a': 1.2, 'e': .2, 'i': 1.5})
    assert detected.screening_score == baseline.screening_score
    assert detected.sigma_level is None
    assert detected.artificial_probability == baseline.artificial_probability
    response = DetectionResponse(designation='SYNTHETIC', is_artificial=False,
                                 artificial_probability=0, sigma_confidence=0,
                                 classification='TEST_ONLY', confidence=0,
                                 non_gravitational_evidence=report)
    assert json.loads(response.model_dump_json())['non_gravitational_evidence'] == report


@pytest.mark.asyncio
async def test_actual_api_get_and_batch_preserve_a3_without_scalar_thrust(payload, monkeypatch):
    from aneos_core.data.models import OrbitalElements
    from aneos_api.endpoints import analysis
    from aneos_core.detection.validated_sigma5_artificial_neo_detector import ValidatedSigma5ArtificialNEODetector
    ng, _ = parsed_ng(payload)
    neo = NEOData('SYNTHETIC', orbital_elements=OrbitalElements(
        semi_major_axis=1.2, eccentricity=.2, inclination=1.5), nongrav=ng)
    monkeypatch.setattr('aneos_core.data.fetcher.DataFetcher.fetch_neo_data', lambda *a, **kw: neo)
    monkeypatch.setattr('aneos_core.data.fetcher.DataFetcher.fetch_multiple', lambda *a, **kw: {'SYNTHETIC': neo})
    monkeypatch.setattr('aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history', lambda *a, **kw: [])
    monkeypatch.setattr(analysis, '_persist_detection_result', lambda *a: None)
    calls = []
    from aneos_core.analysis.screening import ScreeningAnalysisService
    original = ScreeningAnalysisService.analyze_neo
    def spy(self, evidence, *args, **kwargs):
        calls.append(evidence.nongrav.to_dict())
        return original(self, evidence, *args, **kwargs)
    monkeypatch.setattr(ScreeningAnalysisService, 'analyze_neo', spy)
    get = await analysis.detect_neo('SYNTHETIC', False, None)
    monkeypatch.setitem(analysis._batch_store, 'ng-test', {'results': [], 'completed': 0, 'failed': 0})
    await analysis._run_batch_detection('ng-test', ['SYNTHETIC'])
    batch = analysis._batch_store['ng-test']['results'][0]
    assert get.non_gravitational_evidence['components']['A3']['native']['value'] == 1e-12
    assert batch['non_gravitational_evidence'] == get.non_gravitational_evidence
    assert calls == [ng.to_dict(), ng.to_dict()]
    assert get.sigma_confidence is None
    assert get.evidence_snapshot["additional"]["non_gravitational_evidence"] == ng.evidence_report()


@pytest.mark.parametrize('fastest', ['SBDB', 'secondary'])
def test_source_timing_cannot_discard_fitted_acceleration(payload, tmp_path, fastest):
    import time
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_core.data.models import OrbitalElements
    ng, _ = parsed_ng(payload)
    fetcher = DataFetcher(cache_manager=CacheManager(tmp_path), source_priority=['SBDB', 'secondary'])
    fetcher.sources = {'SBDB': object(), 'secondary': object()}
    fetcher.fetch_stats['source_successes']['secondary'] = 0
    fetcher.fetch_stats['source_failures']['secondary'] = 0
    fetcher._fetch_close_approaches = lambda *a: []
    def fetch(name, *args):
        time.sleep(.001 if name == fastest else .02)
        return NEOData('SYNTHETIC', orbital_elements=OrbitalElements(
            semi_major_axis=1.2, eccentricity=.2, inclination=1.5),
            nongrav=ng if name == 'SBDB' else None)
    fetcher._fetch_from_source = fetch
    neo = fetcher._fetch_from_all_sources('SYNTHETIC')
    assert neo.nongrav.to_dict() == ng.to_dict()
    assert neo.evidence_metadata['field_sources']['nongrav'] == 'SBDB'


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_ky26_documented_contract_only():
    """Optional: no fetch or reachability probe occurs at module import."""
    source = SBDBSource(APIConfig())
    try:
        result = await source.fetch_orbital_elements('1998 KY26')
        assert result.success, result.error_message
        assert result.data['_evidence_metadata']['schema_status'] == 'supported'
        ng_data = result.data['_nongrav']
        # Provider availability may evolve; unavailable is explicit, never zero.
        if ng_data is not None:
            ng = NonGravitationalParameters.from_dict(ng_data)
            report = ng.evidence_report()
            assert report['origin_inference'] == 'not_assessed'
            assert 'A3' in report['components']
    finally:
        if source._async_session is not None:
            await source._async_session.close()


def test_cached_future_evidence_schema_is_not_silently_interpreted(payload):
    ng, _ = parsed_ng(payload)
    data = ng.to_dict()
    data['schema_version'] = 'unknown-future-version'
    with pytest.raises(ValueError, match='schema'):
        NonGravitationalParameters.from_dict(data)


def test_tiny_covariance_keeps_correlation_without_product_underflow():
    from aneos_core.data.orbit_evidence import OrbitalCovariance
    cov = OrbitalCovariance(['A1', 'A3'], [[1e-200, .5e-200], [.5e-200, 1e-200]], '2457098.5')
    assert cov.correlation('A1', 'A3') == pytest.approx(.5)


def test_cli_displays_all_axes_and_epoch_caution(payload):
    from aneos_menu_v2 import ANEOSMenuV2
    from unittest.mock import Mock
    ng, _ = parsed_ng(payload)
    neo = NEOData('SYNTHETIC', nongrav=ng)
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu.display_panel = Mock()
    menu._display_neo_data(neo.designation, neo)
    panel = menu.display_panel.call_args.args[0]
    assert 'A1:' in panel and 'A2:' in panel and 'A3:' in panel
    assert 'not thrust' in panel
    assert 'Covariance epoch differs' in panel


@pytest.mark.asyncio
async def test_provider_request_identifies_application_without_live_network(monkeypatch, payload):
    from unittest.mock import MagicMock
    import requests
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.content = b'synthetic buffered response'
    response.json.return_value = payload
    request = MagicMock(return_value=response)
    monkeypatch.setattr(requests, 'get', request)
    result = await SBDBSource(APIConfig()).fetch_orbital_elements('SYNTHETIC')
    assert result.success
    agent = request.call_args.kwargs['headers']['User-Agent']
    assert 'aNEOS/' in agent
    assert 'https://github.com/RobLe3/aneos-suite/issues' in agent
    assert response.close.called


def test_raw_parameter_and_covariance_literals_cannot_diverge_after_replay(payload):
    ng, _ = parsed_ng(payload)
    changed = ng.to_dict()
    changed['parameters']['A3']['value'] = 9e-12
    with pytest.raises(ValueError, match='raw literal'):
        NonGravitationalParameters.from_dict(changed)
    changed = ng.to_dict()
    changed['covariance']['raw_data'][0][0] *= 2
    with pytest.raises(ValueError, match='raw matrix'):
        NonGravitationalParameters.from_dict(changed)


def test_predicted_encounters_cannot_extend_observation_arc(payload):
    from datetime import datetime, timezone
    from aneos_core.data.models import CloseApproach
    ng, _ = parsed_ng(payload)
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    neo = NEOData('SYNTHETIC', nongrav=ng, first_observation=start, last_observation=end)
    neo.add_close_approach(CloseApproach('SYNTHETIC', close_approach_date=datetime(2031, 1, 1, tzinfo=timezone.utc)))
    assert neo.first_observation == start and neo.last_observation == end
    assert neo.nongrav.solution['last_obs'] == '2024-01-01'


@pytest.mark.parametrize('field,value', [('orbit', None), ('signature', []), ('model_pars', None),
                                        ('model_pars', [None]), ('model_pars', [{}]),
                                        ('covariance', {}), ('covariance', [])])
def test_incomplete_source_objects_fail_explicitly(payload, field, value):
    if field in ('orbit', 'signature'):
        payload[field] = value
    else:
        payload['orbit'][field] = value
    with pytest.raises(ValueError):
        SBDBSource(APIConfig())._parse_sbdb_response(payload, 'SYNTHETIC')


@pytest.mark.parametrize('fmt', ['1', '2'])
def test_real_menu_json_and_csv_exports_preserve_acceleration_provenance(payload, tmp_path, fmt):
    import csv
    from types import SimpleNamespace
    from unittest.mock import Mock
    from aneos_menu_v2 import ANEOSMenuV2
    ng, _ = parsed_ng(payload)
    report = ng.evidence_report()
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu._impact_results = {}
    menu._detection_results = {'SYNTHETIC': SimpleNamespace(
        sigma_level=0, artificial_probability=None, classification='TEST_ONLY',
        metadata={'non_gravitational_evidence': report})}
    path = tmp_path / ('export.json' if fmt == '1' else 'export.csv')
    menu._ask = Mock(side_effect=[fmt, str(path)])
    menu.show_success = Mock()
    menu.show_error = Mock(side_effect=AssertionError)
    menu._export_results()
    if fmt == '1':
        actual = json.loads(path.read_text())['neos'][0]['non_gravitational_evidence']
    else:
        lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
        actual = json.loads(next(csv.DictReader(lines))['non_gravitational_evidence'])
    assert actual == report
    assert actual['components']['A3']['native']['raw_sigma'] == '4e-12'


def test_finite_but_impossible_covariance_cannot_pass_via_normalization_overflow():
    from aneos_core.data.orbit_evidence import OrbitalCovariance
    with pytest.raises(ValueError, match='finite normalized'):
        OrbitalCovariance(['A1', 'A3'], [[1e-300, 1e300], [1e300, 1e-300]], '2457098.5')


def test_reference_equinox_fit_quality_and_physical_sources_are_retained(payload):
    payload['orbit'].update(equinox='J2000', rms='.6', n_obs_used=85, n_del_obs_used=2)
    payload['phys_par'] = [{'name': 'diameter', 'value': '.01', 'sigma': '.002', 'units': 'km', 'ref': 'synthetic-literature-marker'}]
    ng, parsed = parsed_ng(payload)
    assert ng.solution['equinox'] == 'J2000'
    assert ng.solution['fit_metadata']['n_obs_used'] == 85
    assert ng.solution['fit_metadata']['rms'] == '.6'
    assert parsed['_evidence_metadata']['physical_measurements'] == payload['phys_par']
    assert ng.solution['canonical_response_sha256']
