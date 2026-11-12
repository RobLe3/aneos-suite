"""Real shared-service controls: candidate sensitivity is not origin certainty."""
import asyncio
import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties, CloseApproach, NonGravitationalParameters


def snapshot(designation='synthetic-marker', unusual=True):
    return NEOData(designation, orbital_elements=OrbitalElements(
        semi_major_axis=1, eccentricity=.2, inclination=52.23568469321602 if unusual else 15),
        physical_properties=PhysicalProperties(diameter_km=.37, absolute_magnitude_h=19.7))


def assert_unavailable(payload):
    for key in ['sigma_confidence', 'combined_p_value', 'false_discovery_rate',
                'artificial_probability', 'is_artificial', 'confidence']:
        assert payload[key] is None
    assert payload['analysis_metadata']['five_sigma_claim_permitted'] is False
    assert payload['sigma_tier'] == 'UNAVAILABLE'
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize('unusual', [True, False])
def test_candidate_positive_and_negative_controls(unusual):
    result = ScreeningAnalysisService().analyze_neo(snapshot(unusual=unusual))
    assert result.candidate is unusual
    assert result.classification == ('FOLLOW_UP_CANDIDATE' if unusual else 'NOT_FLAGGED')
    assert_unavailable(result.to_payload())


@pytest.mark.parametrize('orbit', [{}, {'a': 1}, {'a': -2, 'e': 1.5, 'i': 10}, {'a': 5, 'e': .1, 'i': 20}])
def test_missing_or_out_of_domain_abstains(orbit):
    result = ScreeningAnalysisService().analyze_raw('synthetic-marker', orbit)
    assert result.classification == 'INCONCLUSIVE'
    assert result.screening_score is None
    assert not result.candidate
    assert_unavailable(result.to_payload())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True])
def test_invalid_input_never_becomes_a_score(bad):
    with pytest.raises(ValueError):
        ScreeningAnalysisService().analyze_raw('synthetic-marker', {'a': bad, 'e': .2, 'i': 20})


def test_zero_eccentricity_and_inclination_are_observed_not_missing():
    result = ScreeningAnalysisService().analyze_raw('synthetic-marker', {'a': 1, 'e': 0, 'i': 0})
    assert result.screening_score is not None
    assert result.evidence_sources[0]['analyzed']


def test_snapshot_replay_and_policy_fingerprints():
    service = ScreeningAnalysisService()
    neo = snapshot()
    original = service.analyze_neo(neo)
    replay = service.replay(original.evidence_snapshot, original.evidence_id)
    assert replay.to_payload() == original.to_payload()
    changed = ScreeningAnalysisService(ScreeningPolicy(candidate_threshold=10)).analyze_neo(neo)
    assert changed.evidence_id == original.evidence_id
    assert changed.policy_id != original.policy_id
    assert not changed.candidate
    tampered = copy.deepcopy(original.evidence_snapshot)
    tampered['orbital']['a'] += .1
    with pytest.raises(ValueError, match='fingerprint'):
        service.replay(tampered, original.evidence_id)


def test_snapshot_does_not_alias_caller_mutations():
    extra = {'orbital_history': [{'epoch': 1, 'a': 1}]}
    result = ScreeningAnalysisService().analyze_raw('synthetic-marker', {'a': 1, 'e': .2, 'i': 20}, additional_data=extra)
    extra['orbital_history'][0]['a'] = 100
    assert result.evidence_snapshot['additional']['orbital_history'][0]['a'] == 1


def test_prediction_history_and_a2_do_not_fabricate_propulsion_evidence():
    neo = snapshot()
    neo.close_approaches = [CloseApproach(neo.designation, distance_au=.05, relative_velocity_km_s=10)] * 3
    neo.nongrav = NonGravitationalParameters(a2=-5e-14)
    rich = ScreeningAnalysisService().analyze_neo(neo, {'orbital_history': [{'a': 1}, {'a': 10}]})
    plain = snapshot()
    baseline = ScreeningAnalysisService().analyze_neo(plain)
    assert rich.screening_score == baseline.screening_score
    extra = rich.evidence_snapshot['additional']
    assert extra['close_approach_history'][0]['velocity_km_s'] == 10
    assert extra['close_approach_history'][0]['epoch'] is None
    assert extra['non_gravitational_evidence']['components']['A2']['status'] == 'unverified_legacy_scalar'
    assert extra['non_gravitational_evidence']['components']['A2']['value'] == -5e-14
    assert all(not e['analyzed'] for e in rich.evidence_sources if e['type'] in
               {'course_corrections', 'propulsion_signatures', 'trajectory_patterns'})
    assert_unavailable(rich.to_payload())


def test_spacecraft_name_alone_cannot_create_five_sigma_or_confirm_origin():
    for name in ['2018 A1', '2020 SO', 'J002E3', 'Tesla Roadster']:
        assert_unavailable(ScreeningAnalysisService().analyze_neo(snapshot(name)).to_payload())


@pytest.mark.asyncio
async def test_actual_menu_get_post_batch_and_automatic_paths_match(monkeypatch):
    from aneos_api.endpoints import analysis
    from aneos_api.schemas.detection import OrbitalInput
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline
    from aneos_menu_v2 import ANEOSMenuV2
    neo = snapshot()
    monkeypatch.setattr('aneos_core.data.fetcher.DataFetcher.fetch_neo_data', lambda self, *a, **kw: neo)
    monkeypatch.setattr(analysis, '_persist_detection_result', lambda *a: None)
    direct = ScreeningAnalysisService().analyze_neo(neo).to_payload()
    fetched = (await analysis.detect_neo(neo.designation, False, None)).model_dump()
    raw = (await analysis.detect_neo_raw(OrbitalInput(
        designation=neo.designation, a=1, e=.2, i=neo.orbital_elements.inclination,
        diameter_km=.37, absolute_magnitude=19.7), None)).model_dump()
    monkeypatch.setitem(analysis._batch_store, 'parity', {'results': [], 'completed': 0, 'failed': 0})
    await analysis._run_batch_detection('parity', [neo.designation])
    batch = analysis._batch_store['parity']['results'][0]
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.show_error = Mock(side_effect=AssertionError)
    menu_result = menu._run_detection(neo.designation, neo).analysis['assessment']
    pipeline = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    automatic = (await pipeline._detect_artificial_neo({'designation': neo.designation, 'neo_data': neo})).analysis['assessment']
    keys = ['screening_score', 'candidate', 'classification', 'evidence_id', 'policy_id',
            'evidence_snapshot', 'sigma_confidence', 'artificial_probability', 'is_artificial',
            'combined_p_value', 'false_discovery_rate', 'evidence_sources', 'analysis_metadata']
    for adapter in [fetched, raw, batch, menu_result, automatic]:
        for key in keys:
            # Pydantic drops the duplicate evidence_type convenience key.
            if key == 'evidence_sources':
                assert [(e['type'], e['analyzed'], e['effect_size']) for e in adapter[key]] == [
                    (e['type'], e['analyzed'], e['effect_size']) for e in direct[key]]
            else:
                assert adapter[key] == direct[key], key
        assert_unavailable(adapter)


def test_menu_displays_unavailable_not_artificial_certainty():
    from aneos_menu_v2 import ANEOSMenuV2
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu.display_panel = Mock()
    result = ScreeningAnalysisService().analyze_neo(snapshot()).to_detection_result()
    menu._display_detection_result('synthetic-marker', result, verbose=True)
    panel = menu.display_panel.call_args.args[0]
    assert 'Screening score:' in panel
    assert 'unavailable' in panel
    assert 'σ=5' not in panel
    assert '99.999' not in panel


@pytest.mark.parametrize("field,value", [("diameter", True), ("diameter_km", "1"),
                                        ("albedo", False), ("absolute_magnitude", "25")])
def test_physical_units_require_numeric_measurements(field, value):
    with pytest.raises(ValueError, match="finite number"):
        ScreeningAnalysisService().analyze_raw("synthetic-marker", {"a": 1, "e": .2, "i": 20},
                                               {field: value})


def test_unavailable_canonical_detector_cannot_silently_fall_back():
    from aneos_core.detection.detection_manager import DetectionManager, DetectorType
    manager = DetectionManager()
    manager._detectors = {DetectorType.BASIC: Mock()}
    with pytest.raises(RuntimeError, match="fallback disabled"):
        manager.analyze_neo({"a": 1, "e": .2, "i": 20})
    manager._detectors[DetectorType.BASIC].analyze_neo.assert_not_called()


def test_reference_parameter_change_changes_policy_fingerprint():
    before = ScreeningAnalysisService()
    after = ScreeningAnalysisService()
    after.detector.neo_population_stats["orbital_elements"]["inclination"]["mean"] += 1
    assert before.analyze_neo(snapshot()).policy_id != after.analyze_neo(snapshot()).policy_id


@pytest.mark.asyncio
async def test_raw_api_merges_history_without_losing_other_evidence():
    from aneos_api.endpoints.analysis import detect_neo_raw
    from aneos_api.schemas.detection import OrbitalInput
    response = await detect_neo_raw(OrbitalInput(
        a=1, e=.2, i=20, orbital_history=[{"epoch": "synthetic"}],
        additional_data={"marker": "retained"}), None)
    extra = response.evidence_snapshot["additional"]
    assert extra["marker"] == "retained"
    assert extra["orbital_history"] == [{"epoch": "synthetic"}]
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await detect_neo_raw(OrbitalInput(
            a=1, e=.2, i=20, orbital_history=[{"epoch": "one"}],
            additional_data={"orbital_history": [{"epoch": "two"}]}), None)
    assert exc.value.status_code == 422


def test_component_adapter_cannot_leak_heuristic_p_values_as_significance():
    from aneos_core.detection.detection_manager import DetectionManager, DetectorType
    wrapper = DetectionManager()._detectors[DetectorType.VALIDATED]
    for component in [wrapper.analyze_orbital_dynamics({"a": 1, "e": .2, "i": 80}),
                      wrapper.analyze_physical_properties({"diameter": 11})]:
        assert component["anomaly_score"] > 0
        assert component["p_value"] is None
        assert component["confidence_interval"] is None
        assert component["score_semantics"] == "uncalibrated_ranking_heuristic"


@pytest.mark.parametrize('helper,args', [('_sigma_tier', (5,)), ('_build_interpretation', ()),
                                        ('_spacecraft_veto', ('Tesla Roadster',))])
def test_legacy_api_inference_helpers_have_controlled_deprecation(helper, args):
    from aneos_api.endpoints import analysis
    with pytest.raises(NotImplementedError, match='withdrawn'):
        getattr(analysis, helper)(*args)


def test_documented_openapi_matches_detection_inference_contract():
    from pathlib import Path
    from aneos_api.schemas.detection import DetectionResponse
    documented = json.loads((Path(__file__).parents[1] / 'docs/api/openapi.json').read_text())
    actual = DetectionResponse.model_json_schema(ref_template='#/components/schemas/{model}')
    actual.pop('$defs', None)
    assert documented['components']['schemas']['DetectionResponse'] == actual


def test_finite_extreme_input_cannot_produce_nonfinite_serialized_rank():
    with pytest.raises(ValueError, match='ranking contribution'):
        ScreeningAnalysisService().analyze_raw('synthetic-marker', {'a': 1, 'e': .2, 'i': 20},
                                               {'absolute_magnitude': 1e308})
