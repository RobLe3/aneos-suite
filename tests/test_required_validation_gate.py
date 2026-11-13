"""Required-gate controls, not a scientific accuracy evaluation."""
import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aneos_core.data.models import NEOData, OrbitalElements
from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline, PipelineConfig


def candidate(name='synthetic-control'):
    return {'designation': name, 'neo_data': NEOData(name, orbital_elements=OrbitalElements(
        semi_major_axis=1, eccentricity=.2, inclination=80)),
        'first_stage_score': {'overall_score': 1., 'artificial_neo_detected': True,
                              'sigma_level': 100}}


def pipeline(validator):
    p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    p.config = PipelineConfig()
    p.logger = logging.getLogger(__name__)
    p.multi_stage_validator = validator
    return p


def report(analysis, mode='passed', neo=None):
    from aneos_core.validation.review_evidence import freeze_review_inputs
    payload = analysis.analysis['assessment']
    stages = [SimpleNamespace(stage_number=n, passed=True, validation_status='passed',
                              details={'evidence_basis': 'synthetic_test_control'}) for n in range(1, 6)]
    result = SimpleNamespace(contract_version='required-review-v1',
        evidence_id=payload['evidence_id'], policy_id=payload['policy_id'],
        validation_evidence_id=freeze_review_inputs(neo)[1],
        overall_validation_passed=True, validation_status='passed',
        overall_confidence=None, overall_false_positive_probability=None,
        stage_results=stages)
    if mode == 'failed':
        stages[2].passed = False
        stages[2].validation_status = 'failed'
        result.overall_validation_passed = False
        result.validation_status = 'failed'
    elif mode == 'inconclusive':
        stages[3].passed = False
        stages[3].validation_status = 'inconclusive'
        result.overall_validation_passed = False
        result.validation_status = 'inconclusive'
    elif mode == 'missing_stage':
        stages.pop()
    elif mode == 'duplicate_stage':
        stages[4].stage_number = 4
    elif mode == 'truthy_not_boolean':
        stages[4].passed = 'false'
    elif mode == 'stale_evidence':
        result.evidence_id = 'wrong-evidence'
    elif mode == 'stale_policy':
        result.policy_id = 'wrong-policy'
    elif mode == 'legacy_contract':
        result.contract_version = 'legacy'
    elif mode == 'hidden_error':
        stages[0].details = {'error': 'synthetic failure'}
    elif mode == 'nonfinite':
        stages[0].details = {'infinite': float('inf')}
    return SimpleNamespace(validation_result=result)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['passed', 'failed', 'inconclusive', 'missing_stage',
    'duplicate_stage', 'truthy_not_boolean', 'stale_evidence', 'stale_policy',
    'legacy_contract', 'hidden_error', 'nonfinite', 'outage'])
async def test_every_candidate_runs_required_validator_and_cannot_bypass(mode):
    async def validate(neo, analysis):
        if mode == 'outage':
            raise RuntimeError('synthetic outage')
        return report(analysis, mode, neo)
    validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=validate))
    p = pipeline(validator)
    item = candidate()
    item['analysis_result'] = SimpleNamespace(is_artificial=True, sigma_level=100)
    result = await p._stage_multi_validation([item])
    validator.validate_analysis_result.assert_awaited_once()
    neo, analysis = validator.validate_analysis_result.call_args.args
    assert neo.designation == item['designation']
    assert analysis.analysis['assessment']['analysis_metadata']['five_sigma_claim_permitted'] is False
    assert result.output_count == (1 if mode == 'passed' else 0)
    state = item['multi_stage_validation']
    assert state['origin_status'] == 'inconclusive'
    assert state.get('fp_probability') is None
    assert state.get('validation_score') is None
    assert state['evidence_id'] == analysis.analysis['assessment']['evidence_id']
    assert state['policy_id'] == analysis.analysis['assessment']['policy_id']
    json.dumps(state, allow_nan=False)
    if mode == 'passed':
        assert state['recommendation'] == 'expert_review'
        assert state['validation_status'] == 'passed'
    else:
        assert state['validation_status'] != 'passed'
        assert state['recommendation'] != 'accept'


@pytest.mark.asyncio
async def test_unavailable_validator_has_explicit_outcome_for_every_object():
    items = [candidate('one'), candidate('two')]
    result = await pipeline(None)._stage_multi_validation(items)
    assert result.success is False
    assert result.output_count == 0
    assert [o['status'] for o in result.stage_metrics['outcomes']] == ['unavailable'] * 2
    assert all(c['multi_stage_validation']['validation_status'] == 'unavailable' for c in items)


@pytest.mark.asyncio
async def test_capacity_is_not_silent_rejection_or_success():
    validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n)))
    p = pipeline(validator)
    p.config.multi_stage.max_candidates = 1
    items = [candidate('one'), candidate('two')]
    result = await p._stage_multi_validation(items)
    assert result.output_count == 1
    assert validator.validate_analysis_result.await_count == 1
    assert items[1]['multi_stage_validation']['validation_status'] == 'not_processed_capacity'
    assert result.stage_metrics['outcomes'][1]['status'] == 'not_processed_capacity'


@pytest.mark.asyncio
async def test_stale_positive_state_is_replaced_after_outage():
    item = candidate()
    item['multi_stage_validation'] = {'validation_status': 'passed', 'recommendation': 'accept'}
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=RuntimeError('synthetic outage'))))
    result = await p._stage_multi_validation([item])
    assert result.output_count == 0
    assert item['multi_stage_validation']['validation_status'] == 'unavailable'
    assert item['multi_stage_validation']['recommendation'] == 'do_not_advance'


@pytest.mark.asyncio
async def test_real_validator_does_not_run_placeholder_physics_for_canonical_screening():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.validation.multi_stage_validator import MultiStageValidator
    neo = candidate()['neo_data']
    analysis = ScreeningAnalysisService().analyze_neo(neo).to_detection_result()
    validator = MultiStageValidator.__new__(MultiStageValidator)
    validator.logger = logging.getLogger(__name__)
    # Any use of the placeholder stages fails this test, rather than granting a pass.
    validator.stage1_data_quality_filter = AsyncMock(side_effect=AssertionError('legacy quality'))
    validator.stage2_known_object_crossmatch = AsyncMock(side_effect=AssertionError('legacy orbit'))
    result = (await validator.validate_analysis_result(neo, analysis)).validation_result
    assert result.contract_version == 'required-review-v1'
    assert result.validation_status == 'inconclusive'
    assert result.overall_validation_passed is False
    assert result.overall_false_positive_probability is None
    assert result.overall_confidence is None
    assert result.evidence_id == analysis.analysis['assessment']['evidence_id']
    assert {s.stage_number for s in result.stage_results} == set(range(1, 6))
    assert all(s.passed is False for s in result.stage_results)
    assert all(s.validation_status in ('inconclusive', 'unavailable') for s in result.stage_results)
    assert all(s.false_positive_reduction is None for s in result.stage_results)
    validator.stage1_data_quality_filter.assert_not_called()
    validator.stage2_known_object_crossmatch.assert_not_called()


@pytest.mark.asyncio
async def test_real_validator_path_is_inconclusive_not_a_successful_natural_detection():
    from aneos_core.validation.multi_stage_validator import MultiStageValidator
    validator = MultiStageValidator.__new__(MultiStageValidator)
    validator.logger = logging.getLogger(__name__)
    item = candidate()
    result = await pipeline(validator)._stage_multi_validation([item])
    assert result.output_count == 0
    assert item['multi_stage_validation']['validation_status'] == 'inconclusive'
    assert result.stage_metrics['outcomes'][0]['status'] == 'inconclusive'


@pytest.mark.asyncio
async def test_expert_queue_rechecks_gate_and_keeps_positive_control():
    validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n)))
    p = pipeline(validator)
    item = candidate()
    result = await p._stage_multi_validation([item])
    valid = result.candidates[0]
    bypass = candidate('bypass')
    queue = await p._stage_expert_review_prep([valid, bypass])
    assert queue.output_count == 1
    assert queue.candidates[0]['designation'] == valid['designation']
    assert queue.stage_metrics['excluded_unvalidated'] == ['bypass']


@pytest.mark.asyncio
async def test_mixed_failure_does_not_discard_gate_verified_positive_in_complete_pipeline():
    from aneos_core.pipeline.automatic_review_pipeline import StageResult, ProcessingStage
    async def validate(neo, analysis):
        if neo.designation == 'outage':
            raise RuntimeError('synthetic outage')
        return report(analysis, neo=neo)
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=validate)))
    p.console = None
    items = [candidate('positive'), candidate('outage')]
    historical = SimpleNamespace(total_chunks_processed=1)
    p._stage_historical_polling = AsyncMock(return_value=historical)
    p._extract_raw_objects = Mock(return_value=items)
    p._stage_first_review = AsyncMock(return_value=StageResult(
        stage=ProcessingStage.FIRST_STAGE_REVIEW, input_count=2, output_count=2,
        processing_time_seconds=0, success=True, candidates=items))
    p._calculate_pipeline_metrics = Mock(return_value={})
    p._save_pipeline_results = Mock()
    result = await p.run_complete_pipeline(1)
    assert result.final_candidates == 1
    assert result.validated_candidates[0]['designation'] == 'positive'
    stage = result.stage_results[ProcessingStage.MULTI_STAGE_VALIDATION]
    assert stage.success is False  # The outage is not hidden by partial advancement.
    assert stage.stage_metrics['validation_errors'] == 1
    assert items[1]['multi_stage_validation']['validation_status'] == 'unavailable'


@pytest.mark.asyncio
async def test_candidate_identity_mismatch_cannot_validate_another_object():
    validator = SimpleNamespace(validate_analysis_result=AsyncMock())
    item = candidate('one')
    item['neo_data'].designation = 'different-object'
    result = await pipeline(validator)._stage_multi_validation([item])
    assert result.output_count == 0
    validator.validate_analysis_result.assert_not_called()
    assert 'identity mismatch' in item['multi_stage_validation']['reason']


@pytest.mark.asyncio
async def test_timeout_is_an_explicit_unavailable_outcome():
    import asyncio
    async def hang(*args):
        await asyncio.sleep(10)
    validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=hang))
    p = pipeline(validator)
    p.config.multi_stage.processing_timeout_seconds = .001
    item = candidate()
    result = await p._stage_multi_validation([item])
    assert not result.success
    assert result.output_count == 0
    assert 'TimeoutError' in item['multi_stage_validation']['reason']


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [True, False])
async def test_enhanced_validation_fallback_never_assumes_passed(enabled):
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    wrapper = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
    wrapper.original_pipeline = SimpleNamespace(analyze_neo=AsyncMock(return_value=SimpleNamespace()))
    wrapper.enable_validation = enabled
    wrapper.validation_available = False
    wrapper.logger = logging.getLogger(__name__)
    result = await wrapper.analyze_neo_with_validation('synthetic-marker')
    assert result.validation_result.overall_validation_passed is False
    assert result.validation_result.validation_status == 'unavailable'
    assert result.validation_result.overall_false_positive_probability is None
    assert result.validation_result.overall_confidence is None


@pytest.mark.asyncio
async def test_enhanced_validator_exception_remains_explicit_and_preserves_original():
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    wrapper = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
    original = SimpleNamespace(marker='preserved')
    wrapper.original_pipeline = SimpleNamespace(analyze_neo=AsyncMock(return_value=original))
    wrapper.enable_validation = True
    wrapper.validation_available = True
    wrapper.validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=RuntimeError('synthetic outage')))
    wrapper.logger = logging.getLogger(__name__)
    result = await wrapper.analyze_neo('synthetic-marker', enhanced=True)
    assert result.original_result is original
    assert result.validation_result.validation_status == 'unavailable'
    assert result.validation_result.overall_validation_passed is False


@pytest.mark.asyncio
async def test_changed_evidence_or_validator_policy_is_not_covered_by_old_gate():
    for change in ('evidence', 'policy'):
        validator = SimpleNamespace(config={}, validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n)))
        p = pipeline(validator)
        item = candidate()
        await p._stage_multi_validation([item])
        assert item['multi_stage_validation']['validation_status'] == 'passed'
        if change == 'evidence':
            item['neo_data'].orbital_elements.inclination = 20
        else:
            validator.config['synthetic_policy_revision'] = 2
        queued = await p._stage_expert_review_prep([item])
        assert queued.output_count == 0
        assert queued.stage_metrics['excluded_unvalidated'] == [item['designation']]


@pytest.mark.asyncio
async def test_queue_capacity_is_explicit_and_rank_is_recomputed():
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n))))
    items = [candidate('one'), candidate('two')]
    await p._stage_multi_validation(items)
    p.config.expert_review.max_candidates = 1
    score = items[0]['required_validation_assessment']['screening_score']
    items[0]['required_validation_assessment']['screening_score'] = 9999
    queued = await p._stage_expert_review_prep(items)
    assert queued.output_count == 1
    assert queued.candidates[0]['expert_review']['final_score'] == score
    assert queued.stage_metrics['deferred_capacity'] == ['two']


@pytest.mark.asyncio
async def test_validator_policy_cannot_change_during_a_pass():
    validator = SimpleNamespace(config={})
    async def change_policy(neo, analysis):
        validator.config['synthetic_policy_revision'] = 2
        return report(analysis, neo=neo)
    validator.validate_analysis_result = AsyncMock(side_effect=change_policy)
    item = candidate()
    result = await pipeline(validator)._stage_multi_validation([item])
    assert result.output_count == 0
    assert 'policy changed during' in item['multi_stage_validation']['reason']


@pytest.mark.asyncio
async def test_saved_result_keeps_failed_inconclusive_and_capacity_records_without_replacing_history(tmp_path):
    from datetime import datetime
    from aneos_core.pipeline.automatic_review_pipeline import PipelineResult, ProcessingStage
    validator = SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, 'inconclusive', n)))
    p = pipeline(validator)
    p.results_dir = tmp_path
    p.config.multi_stage.max_candidates = 1
    stage = await p._stage_multi_validation([candidate('inconclusive'), candidate('deferred')])
    now = datetime.now()
    result = PipelineResult(total_input_objects=2, final_candidates=0,
        processing_start_time=now, processing_end_time=now,
        stage_results={ProcessingStage.MULTI_STAGE_VALIDATION: stage}, pipeline_metrics={}, validated_candidates=[])
    p._save_pipeline_results(result)
    first = next(tmp_path.glob('pipeline_result_*.json'))
    before = first.read_bytes()
    p._save_pipeline_results(result)
    assert len(list(tmp_path.glob('pipeline_result_*.json'))) == 2
    assert first.read_bytes() == before
    saved = json.loads(before)['stage_summary']['multi_stage_validation']['stage_metrics']
    assert [r['validation_status'] for r in saved['validation_records']] == ['inconclusive', 'not_processed_capacity']
    assert saved['validation_records'][0]['evidence_id']
    assert saved['validation_records'][0]['stage_results'][3]['validation_status'] == 'inconclusive'
    assert saved['outcomes'][1]['reason']


@pytest.mark.asyncio
async def test_pipeline_ui_and_summary_do_not_format_unavailable_as_probability():
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    from aneos_menu_v2 import ANEOSMenuV2
    wrapper = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
    wrapper.logger = logging.getLogger(__name__)
    wrapper.get_validation_status = Mock(return_value={})
    enhanced = wrapper._create_basic_enhanced_result(SimpleNamespace())
    summary = wrapper.get_validation_summary([enhanced])
    assert summary['false_positive_statistics']['mean_probability'] is None
    assert summary['false_positive_statistics']['unavailable_count'] == 1
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu.display_panel = Mock()
    proxy = SimpleNamespace(analysis={}, multi_stage_validation={'validation_score': None, 'validation_status': 'inconclusive'})
    menu._display_pipeline_candidate_detail('synthetic-marker', proxy)
    assert 'inconclusive' in menu.display_panel.call_args.args[0]
    assert 'unavailable' in menu.display_panel.call_args.args[0]


@pytest.mark.asyncio
async def test_actual_observation_span_is_preserved_without_fixed_quality_defaults():
    from datetime import datetime, timezone
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.validation.multi_stage_validator import MultiStageValidator
    neo = candidate()['neo_data']
    neo.first_observation = datetime(2024, 1, 1, tzinfo=timezone.utc)
    neo.last_observation = datetime(2024, 2, 1, tzinfo=timezone.utc)
    validator = MultiStageValidator.__new__(MultiStageValidator)
    analysis = ScreeningAnalysisService().analyze_neo(neo).to_detection_result()
    report_value = (await validator.validate_analysis_result(neo, analysis)).validation_result
    details = report_value.stage_results[0].details
    assert details['observation_span_days'] == 31
    assert details['source_reliability'] is None
    assert 'orbital_uncertainty' not in details
    assert report_value.overall_validation_passed is False


@pytest.mark.asyncio
async def test_initialized_validator_canonical_path_is_offline_and_inconclusive(tmp_path, monkeypatch):
    import socket
    from aneos_core.validation.multi_stage_validator import MultiStageValidator
    attempts = []
    def deny_network(*args, **kwargs):
        attempts.append('unexpected network')
        raise AssertionError('This offline validation test must not access providers')
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    validator = MultiStageValidator(config={'cache_dir': str(tmp_path)})
    item = candidate()
    result = await pipeline(validator)._stage_multi_validation([item])
    assert attempts == []
    assert result.output_count == 0
    assert item['multi_stage_validation']['validation_status'] == 'inconclusive'
    assert item['multi_stage_validation']['stage_results'][0]['details']['observation_span_days'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['observation_span', 'source', 'physical_uncertainty'])
async def test_validation_inputs_beyond_ranking_cannot_reuse_passed_gate(changed):
    from datetime import datetime, timezone
    from aneos_core.data.models import PhysicalProperties
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n))))
    item = candidate()
    item['neo_data'].physical_properties = PhysicalProperties(diameter_km=.01, diameter_uncertainty=.002)
    await p._stage_multi_validation([item])
    rank_id = item['required_validation_assessment']['evidence_id']
    if changed == 'observation_span':
        item['neo_data'].last_observation = datetime(2026, 1, 1, tzinfo=timezone.utc)
    elif changed == 'source':
        item['neo_data'].sources_used = ['different-synthetic-source']
    else:
        item['neo_data'].physical_properties.diameter_uncertainty = .005
    from aneos_core.analysis.screening import ScreeningAnalysisService
    # This specifically exercises evidence absent from the rank-only snapshot.
    assert ScreeningAnalysisService().analyze_neo(item['neo_data']).evidence_id == rank_id
    queued = await p._stage_expert_review_prep([item])
    assert queued.output_count == 0
    assert queued.stage_metrics['excluded_unvalidated'] == [item['designation']]


@pytest.mark.asyncio
async def test_full_validation_inputs_cannot_change_during_a_pass():
    async def validate(neo, analysis):
        value = report(analysis, neo=neo)
        neo.sources_used = ['mutated-synthetic-source']
        return value
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=validate)))
    item = candidate()
    result = await p._stage_multi_validation([item])
    assert result.output_count == 0
    assert 'evidence changed during' in item['multi_stage_validation']['reason']


def test_review_snapshot_preserves_full_physical_uncertainty_and_does_not_alias():
    from aneos_core.data.models import PhysicalProperties
    from aneos_core.validation.review_evidence import freeze_review_inputs
    neo = candidate()['neo_data']
    neo.physical_properties = PhysicalProperties(diameter_km=.01, diameter_uncertainty=.002)
    snap, identifier = freeze_review_inputs(neo)
    assert snap['physical_properties']['diameter_uncertainty'] == .002
    neo.physical_properties.diameter_uncertainty = .005
    assert snap['physical_properties']['diameter_uncertainty'] == .002
    assert freeze_review_inputs(neo)[1] != identifier


@pytest.mark.asyncio
async def test_gate_positive_control_reaches_existing_rich_store_and_detail_paths():
    from aneos_menu_v2 import ANEOSMenuV2
    p = pipeline(SimpleNamespace(validate_analysis_result=AsyncMock(side_effect=lambda n, a: report(a, neo=n))))
    stage = await p._stage_multi_validation([candidate()])
    queued = await p._stage_expert_review_prep(stage.candidates)
    result = {'status': 'success', 'pipeline_result': SimpleNamespace(
        pipeline_metrics={}, validated_candidates=queued.candidates)}
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu._detection_results = {}
    menu.display_table = Mock()
    menu.display_panel = Mock()
    menu._display_pipeline_result(result)
    menu._store_pipeline_candidates(result)
    proxy = menu._detection_results['SYNTHETIC-CONTROL']
    assert proxy.artificial_probability is None
    assert proxy.sigma_level is None
    assert proxy.screening_score > 5
    menu._display_detection_result('synthetic-control', proxy)
    assert 'passed' in menu.display_panel.call_args.args[0]
    assert 'unavailable' in menu.display_panel.call_args.args[0]
