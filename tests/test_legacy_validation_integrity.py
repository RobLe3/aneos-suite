"""Legacy validation corrections; fixtures do not establish scientific accuracy."""
import copy
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
from aneos_core.validation.multi_stage_validator import MultiStageValidator, ValidationStageResult


def neo():
    return NEOData('synthetic-control', orbital_elements=OrbitalElements(
        semi_major_axis=1, eccentricity=0, inclination=0, ascending_node=12,
        argument_of_perihelion=34, mean_anomaly=56), physical_properties=PhysicalProperties(
        diameter_km=.01, absolute_magnitude_h=25, albedo=.5),
        first_observation=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_observation=datetime(2024, 2, 1, tzinfo=timezone.utc))


def validator():
    v = MultiStageValidator.__new__(MultiStageValidator)
    v.logger = logging.getLogger(__name__)
    v.config = v._default_config()
    v._initialize_null_distributions()
    v.false_positive_prevention = Mock(side_effect=AssertionError('unapproved legacy crossmatch'))
    v.delta_bic_analyzer = Mock(side_effect=AssertionError('unfitted BIC'))
    v.statistical_testing = Mock(side_effect=AssertionError('uncalibrated null'))
    return v


def test_real_orbit_and_measured_span_are_not_replaced_with_fixed_defaults():
    v, data = validator(), neo()
    orbit = v._extract_orbital_elements(data)
    assert orbit['a'] == 1
    assert orbit['e'] == 0
    assert orbit['i'] == 0
    assert orbit['Omega'] == 12
    assert orbit['omega'] == 34
    assert orbit['M'] == 56
    assert v._assess_observation_span(data) == 31
    assert v._extract_orbital_elements(NEOData('missing')) == {}
    assert v._assess_observation_span(NEOData('missing')) is None


@pytest.mark.parametrize('name', ['_assess_source_reliability', '_assess_orbital_uncertainty'])
def test_unsupported_quality_is_unavailable_not_a_constant(name):
    assert getattr(validator(), name)(neo()) is None


def test_no_preinstalled_null_or_fake_indicator_evidence():
    v = validator()
    assert v.null_distributions == {}
    assert v._count_anomalous_categories(SimpleNamespace()) is None
    assert v._check_high_impact_indicators(SimpleNamespace()) is None
    assert v._run_statistical_tests(SimpleNamespace(indicator_results={})) == {}


@pytest.mark.asyncio
async def test_no_placeholder_uncertainty_or_confidence_interval():
    value = await validator()._calculate_uncertainties(neo(), SimpleNamespace())
    assert value['status'] == 'unavailable'
    assert value['overall_uncertainty'] is None
    assert value.get('confidence_bounds') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('number', [1, 2, 3, 4, 5])
async def test_direct_stage_methods_cannot_grant_pass_or_invent_probability(number):
    v = validator()
    methods = [v.stage1_data_quality_filter, v.stage2_known_object_crossmatch,
        v.stage3_physical_plausibility, v.stage4_statistical_significance, v.stage5_expert_review_threshold]
    args = [(neo(),), (neo(), SimpleNamespace()), (neo(), SimpleNamespace()),
            (SimpleNamespace(),), (SimpleNamespace(overall_score=1, confidence=1),)]
    result = await methods[number-1](*args[number-1])
    assert result.passed is False
    assert result.validation_status in ('unavailable', 'inconclusive')
    assert result.confidence is None
    assert result.false_positive_reduction is None
    assert result.score is None
    if number == 1:
        assert result.details['screening_field_coverage']['available_count'] == 6
        assert result.details['screening_field_coverage']['fraction'] == 1
        assert result.details['observation_span_days'] == 31
        assert result.details['source_reliability'] is None


@pytest.mark.asyncio
async def test_legacy_input_is_preserved_but_cannot_restore_scientific_claims():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    data = neo()
    legacy = SimpleNamespace(is_artificial=True, confidence=1, overall_score=100, classification='ARTIFICIAL')
    result = await validator().validate_analysis_result(data, legacy)
    assert result.original_result is legacy
    assert result.screening_assessment == ScreeningAnalysisService().analyze_neo(data).to_payload()
    assert result.validation_result.validation_status == 'inconclusive'
    assert result.validation_result.overall_false_positive_probability is None
    assert result.is_artificial is None
    assert result.confidence is None
    assert result.classification != 'ARTIFICIAL'


@pytest.mark.asyncio
async def test_invalid_data_is_explicit_unavailable_not_half_probability():
    result = await validator().validate_analysis_result(None, SimpleNamespace(overall_score=1))
    assert result.validation_result.validation_status == 'unavailable'
    assert result.validation_result.overall_validation_passed is False
    assert result.validation_result.overall_false_positive_probability is None
    assert result.validation_result.overall_confidence is None
    assert result.validation_result.synthetic_population_percentile is None
    assert result.uncertainty_analysis['status'] == 'unavailable'


def stages():
    return [ValidationStageResult(n, f'synthetic-control{n}', True, None, None, None,
        {'evidence_basis': 'synthetic_test_control'}, 0, validation_status='passed') for n in range(1,6)]


@pytest.mark.parametrize('mode', ['empty', 'missing', 'duplicate', 'error', 'truthy', 'legacy'])
def test_aggregate_cannot_turn_missing_or_failed_execution_into_a_pass(mode):
    values = stages()
    if mode == 'empty': values = []
    elif mode == 'missing': values.pop()
    elif mode == 'duplicate': values[-1].stage_number = 4
    elif mode == 'error': values[0].details['error'] = 'synthetic failure'
    elif mode == 'truthy': values[0].passed = 'false'
    elif mode == 'legacy': values[0].validation_status = None
    result = validator()._aggregate_validation_stages(values)
    assert result.overall_validation_passed is False
    assert result.validation_status == 'unavailable'
    assert result.overall_false_positive_probability is None
    assert result.overall_confidence is None


def test_complete_explicit_stage_positive_control_is_review_eligibility_not_origin():
    result = validator()._aggregate_validation_stages(stages())
    assert result.overall_validation_passed is True
    assert result.validation_status == 'passed'
    assert result.recommendation == 'expert_review'
    assert result.overall_false_positive_probability is None
    assert result.artificial_object_likelihood is None


@pytest.mark.asyncio
async def test_canonical_acquisition_adapter_is_real_and_matches_shared_policy():
    from aneos_core.analysis.screening import CanonicalAnalysisAdapter, ScreeningAnalysisService
    data = neo()
    fetcher = SimpleNamespace(fetch_neo_data=Mock(return_value=data))
    adapter = CanonicalAnalysisAdapter(fetcher=fetcher)
    result = await adapter.analyze_neo(data.designation)
    fetcher.fetch_neo_data.assert_called_once_with(data.designation)
    assert result.analysis['assessment'] == ScreeningAnalysisService().analyze_neo(data).to_payload()
    assert result.neo_data is data
    assert result.is_artificial is None


@pytest.mark.asyncio
async def test_advanced_wrapper_cannot_score_unavailable_stage_outputs_as_evidence():
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    from aneos_core.analysis.screening import CanonicalAnalysisAdapter
    wrapper = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
    wrapper.enable_validation = True
    wrapper.validation_available = True
    wrapper.logger = logging.getLogger(__name__)
    wrapper.original_pipeline = CanonicalAnalysisAdapter()
    wrapper.validator = validator()
    wrapper.advanced_scorer = Mock(side_effect=AssertionError('unavailable stages must not become clues'))
    result = await wrapper.analyze_neo_with_advanced_scoring('synthetic-control', neo())
    assert result['advanced_score'] is None
    assert result['advanced_scoring_status'] == 'unavailable_scientific_prerequisites'
    assert result['screening_assessment']['sigma_confidence'] is None
    assert result['enhanced_analysis'].validation_result.validation_status == 'inconclusive'


@pytest.mark.asyncio
async def test_missing_encounter_and_orbit_cannot_create_zero_inclination_triage():
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline, PipelineConfig
    p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    p.config = PipelineConfig()
    p.logger = logging.getLogger(__name__)
    p._designation_frequencies = {}
    p.xviii_swarm_scorer = None
    item = {'designation': 'synthetic-missing', 'orbital_elements': None}
    result = p._simple_first_stage_scoring(item)
    assert result['overall_score'] is None
    assert result['confidence'] is None
    assert result['processing_stage'] == 'unavailable_input'
    assert await p._process_first_stage_batch([item]) == []


@pytest.mark.asyncio
async def test_atlas_wrapper_does_not_fabricate_radar_thermal_or_delta_bic_from_missing_data():
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline
    p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    p.logger = logging.getLogger(__name__)
    p._designation_frequencies = {}
    p.xviii_swarm_scorer = Mock()
    result = await p._apply_xviii_swarm_scoring({'designation':'synthetic-missing', 'orbital_elements':None})
    assert result['overall_score'] is None
    p.xviii_swarm_scorer.calculate_score.assert_not_called()


def test_dashboard_metrics_keep_unknown_confidence_and_count_it_separately():
    from aneos_dashboard.monitoring.validation_metrics import ValidationMetricsCollector
    v = validator()
    result = v._aggregate_validation_stages([v._stage(n, str(n), 'inconclusive', {}) for n in range(1,6)])
    collector = ValidationMetricsCollector()
    assert collector.record_validation_session(result, 1.)
    assert collector.daily_stats['average_confidence'] is None
    assert collector.daily_stats['confidence_available_count'] == 0
    assert collector._get_confidence_distribution()['unavailable'] == 1
    dashboard = collector.get_dashboard_data()
    assert dashboard['system_overview']['false_positive_prevention_rate'] is None
    assert dashboard['validation_pipeline']['stage_performance']['stage_1']['avg_score'] is None
    assert collector.record_validation_session(result, 1.)  # Re-broadcast must not double-count the same report.
    assert len(collector.validation_sessions) == 1


@pytest.mark.asyncio
async def test_dashboard_uses_authoritative_validation_and_nullable_transport():
    from aneos_dashboard.api.validation_integration import ValidationPipelineIntegration
    from aneos_dashboard.monitoring.validation_metrics import ValidationMetricsCollector
    from aneos_dashboard.websockets.validation_websocket import ValidationWebSocketManager
    v = validator()
    original = v.validate_analysis_result
    v.validate_analysis_result = AsyncMock(side_effect=original)
    collector = ValidationMetricsCollector()
    manager = ValidationWebSocketManager(collector)
    manager.broadcast_message = AsyncMock()
    integration = ValidationPipelineIntegration.__new__(ValidationPipelineIntegration)
    integration.validator = v
    integration.metrics_collector = collector
    integration.websocket_manager = manager
    integration.module_status = {}
    integration.active_validations = {}
    integration.logger = logging.getLogger(__name__)
    value = await integration.validate_with_dashboard_integration(neo(), SimpleNamespace(overall_score=100), 'synthetic-session')
    v.validate_analysis_result.assert_awaited_once()
    assert value.validation_status == 'inconclusive'
    assert value.overall_false_positive_probability is None
    assert len(collector.validation_sessions) == 1
    assert not collector.artificial_object_alerts
    events = [call.args for call in manager.broadcast_message.call_args_list]
    result = next(payload for name, payload in events if name == 'validation_result')
    assert result['validation_status'] == 'inconclusive'
    assert result['false_positive_probability'] is None
    assert result['evidence_id']


@pytest.mark.asyncio
async def test_dashboard_visualization_does_not_convert_unknown_fp_to_origin_score():
    from aneos_dashboard.api.dashboard_endpoints import get_scatter_plot_data, get_trend_data
    history = [{'end_time': datetime.now().isoformat(), 'overall_confidence':None,
        'overall_false_positive_probability':None, 'recommendation':'additional_evidence_required',
        'validation_status':'inconclusive', 'total_processing_time_ms':1, 'stage_metrics':[]}]
    collector = SimpleNamespace(get_validation_history=lambda *a: history, get_dashboard_data=lambda: {})
    scatter = json.loads((await get_scatter_plot_data(24, collector)).body)
    assert scatter['scatter_data']['confidence_vs_score'][0]['x'] is None
    trend = json.loads((await get_trend_data(24, collector)).body)
    assert trend['time_series']['average_confidence'] == [None]


@pytest.mark.asyncio
async def test_legacy_claim_aliases_and_tampered_payload_cannot_restore_probabilities():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    data = neo()
    raw = ScreeningAnalysisService().analyze_neo(data).to_detection_result()
    raw.analysis['assessment']['artificial_probability'] = 1
    raw.analysis['assessment']['classification'] = 'ARTIFICIAL'
    raw.analysis['assessment']['analysis_metadata']['five_sigma_claim_permitted'] = True
    raw.bayesian_probability = 1
    result = await validator().validate_analysis_result(data, raw)
    assert result.original_result is raw
    assert result.artificial_probability is None
    assert result.bayesian_probability is None
    assert result.analysis['assessment']['artificial_probability'] is None
    assert result.metadata['five_sigma_claim_permitted'] is False
    assert result.classification != 'ARTIFICIAL'


@pytest.mark.asyncio
async def test_invalid_quality_does_not_become_complete_or_pass_on_error():
    data = neo()
    data.orbital_elements.inclination = float('nan')
    value = await validator().stage1_data_quality_filter(data)
    assert value.validation_status == 'unavailable'
    assert value.passed is False
    assert value.score is None
    assert value.confidence is None
    assert value.details['error']


def test_reversed_arc_is_not_defaulted_to_forty_five_days():
    data = neo()
    data.first_observation, data.last_observation = data.last_observation, data.first_observation
    with pytest.raises(ValueError, match='reversed'):
        validator()._assess_observation_span(data)


@pytest.mark.asyncio
async def test_input_only_rank_positive_and_encounter_proxy_are_kept_separate():
    from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline, PipelineConfig
    p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
    p.config = PipelineConfig()
    p.logger = logging.getLogger(__name__)
    p._designation_frequencies = {}
    p.xviii_swarm_scorer = Mock()
    item = {'designation': 'synthetic-control', 'orbital_elements': {
        'semi_major_axis': 1, 'eccentricity': .2, 'inclination': 80}}
    assert (await p._apply_xviii_swarm_scoring(item))['overall_score'] is None
    result = await p._process_first_stage_batch([item])
    assert len(result) == 1  # Real rank-positive control still advances without a proxy score.
    assert result[0]['first_stage_score']['screening_candidate'] is True
    assert result[0]['first_stage_score']['artificial_neo_detected'] is None
    p.xviii_swarm_scorer.calculate_score.assert_not_called()


def test_dashboard_javascript_does_not_plot_null_as_perfect_score():
    import subprocess
    import shutil
    from pathlib import Path
    node = shutil.which('node')
    assert node, 'Node.js is required for the existing dashboard regression control'
    script = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
let container={innerHTML:''};
const context={window:{}, document:{addEventListener(){},getElementById(){return container;}}, console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const panel=Object.create(context.window.ValidationDashboard.prototype);
const chart={data:{datasets:[{data:[]},{data:[]},{data:[]}]},update(){}};
panel.charts={validationScatter:chart};
panel.addValidationResult({false_positive_probability:null,confidence:null});
assert.equal(chart.data.datasets[2].data.length,0);
panel.addValidationResult({false_positive_probability:.01,confidence:.8});
assert.equal(chart.data.datasets[2].data.length,0);
panel.addValidationResult({screening_score:6,confidence:.8});
assert.equal(chart.data.datasets[2].data.length,1);
assert.equal(chart.data.datasets[2].data[0].x,6);
panel.addValidationResult({screening_score:0,confidence:0});
assert.equal(chart.data.datasets[2].data[1].x,0);
assert.equal(chart.data.datasets[2].data[1].y,0);
panel.updateSystemOverview({false_positive_prevention_rate:null,average_confidence:null});
assert.equal((container.innerHTML.match(/unavailable/g)||[]).length,2);
'''
    path = Path(__file__).parents[1] / 'aneos_dashboard/static/js/dashboard.js'
    result = subprocess.run([node, '-e', script, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_missing_validator_does_not_erase_a_valid_screening_positive():
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    from aneos_core.analysis.screening import CanonicalAnalysisAdapter
    data = neo()
    data.orbital_elements.inclination = 80
    wrapper = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
    wrapper.original_pipeline = CanonicalAnalysisAdapter()
    wrapper.enable_validation = True
    wrapper.validation_available = False
    wrapper.logger = logging.getLogger(__name__)
    result = await wrapper.analyze_neo_with_validation(data.designation, data)
    assert result.candidate is True
    assert result.screening_score > 5
    assert result.validation_result.validation_status == 'unavailable'
    assert result.is_artificial is None
