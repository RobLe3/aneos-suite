"""Actual legacy menu/launcher entrypoints; synthetic controls, never origin proof."""
import asyncio
import io
import json
from unittest.mock import Mock
import pytest
from rich.console import Console
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',Mock(side_effect=AssertionError('network forbidden')))


def neo(name='marker'):
    return NEOData(name,orbital_elements=OrbitalElements(semi_major_axis=1.,eccentricity=.2,inclination=52.23568469321602),
        physical_properties=PhysicalProperties(diameter_km=.37,absolute_magnitude_h=19.7))


@pytest.fixture
def pipeline():
    result=AnalysisPipeline()
    result.data_fetcher=Mock()
    result.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:neo(d)
    return result


def menu(pipeline,rich=False):
    from aneos_menu import ANEOSMenu
    result=ANEOSMenu.__new__(ANEOSMenu)
    result._screening_pipeline=pipeline
    result.console=Console(file=io.StringIO(),width=180,color_system=None) if rich else None
    result.get_input=Mock(return_value='marker')
    result.wait_for_input=Mock()
    result.show_error=Mock()
    return result


@pytest.mark.parametrize('rich',[False,True])
@pytest.mark.parametrize('method',['single_neo_analysis','interactive_analysis','enhanced_validation_pipeline',
    'spectral_analysis_suite','orbital_dynamics_modeling','cross_reference_database','statistical_analysis_tools',
    'xviii_swarm_advanced_scoring'])
def test_object_entrypoints_use_actual_shared_pipeline(pipeline,rich,method,capsys):
    ui=menu(pipeline,rich)
    ui._get_test_data=Mock(side_effect=AssertionError('alias fixture used as measurement'))
    result=getattr(ui,method)()
    assert result.screening_assessment==ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    pipeline.data_fetcher.fetch_neo_data.assert_called_once()
    assert result.screening_assessment['artificial_probability'] is None
    text=ui.console.file.getvalue() if rich else capsys.readouterr().out
    assert 'uncalibrated' in text.lower()
    assert 'confirmed' not in text.lower()
    assert not ui.show_error.called


@pytest.mark.parametrize('rich',[False,True])
def test_legacy_detector_display_replays_canonical_result_without_false_natural(pipeline,rich,capsys):
    ui=menu(pipeline,rich)
    result=ScreeningAnalysisService().analyze_neo(neo()).to_detection_result()
    ui.display_validated_result(result)
    text=ui.console.file.getvalue() if rich else capsys.readouterr().out
    assert 'FOLLOW_UP_CANDIDATE' in text
    assert 'unavailable' in text.lower()
    assert 'NATURAL OBJECT' not in text
    assert 'VALIDATED' not in text


def test_batch_retains_failed_and_duplicate_objects(pipeline,tmp_path,capsys):
    ui=menu(pipeline)
    path=tmp_path/'objects.txt';path.write_text('first\noffline\nfirst\n')
    ui.get_input.return_value=str(path)
    def fetch(d,**kw):
        if d=='offline': raise DataSourceUnavailableError('offline')
        return neo(d)
    pipeline.data_fetcher.fetch_neo_data.side_effect=fetch
    results=ui.batch_analysis()
    assert [r.designation for r in results]==['first','offline','first']
    assert results[1].anomaly_score.overall_score is None
    text=capsys.readouterr().out
    assert 'source_unavailable' in text
    assert 'Artificial Detections' not in text


def test_alias_names_cannot_bypass_acquisition_or_create_demo_history(pipeline):
    ui=menu(pipeline)
    ui.get_input.return_value='tesla'
    pipeline.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('offline')
    result=ui.single_neo_analysis()
    assert result.status=='source_unavailable'
    assert result.anomaly_score.overall_score is None
    pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('tesla',force_refresh=False)


def test_report_json_and_markdown_preserve_native_evidence_and_limits(pipeline):
    from aneos_core.analysis.reporting import build_screening_report,render_screening_markdown
    result=asyncio.run(pipeline.analyze_neo('marker'))
    report=build_screening_report(result)
    assert report['screening_assessment']==result.screening_assessment
    assert report['neo_data']==result.neo_data.to_dict()
    json.dumps(report,allow_nan=False)
    text=render_screening_markdown(report)
    assert result.screening_assessment['evidence_id'] in text
    assert result.screening_assessment['policy_id'] in text
    assert 'Artificial-origin probability: Not calibrated' in text


def test_launcher_no_python_code_interpolation(pipeline,monkeypatch,capsys):
    import aneos
    from aneos_core.analysis import cli
    monkeypatch.setattr(cli,'create_analysis_pipeline',lambda:pipeline)
    designation="marker';raise RuntimeError('injected');#"
    monkeypatch.setattr(aneos.sys,'argv',['aneos.py','analyze',designation,'--json'])
    actual_run=aneos.subprocess.run
    def only_readonly_git(args,**kwargs):
        assert args in (['git','rev-parse','HEAD'],['git','--no-optional-locks','status','--porcelain'])
        assert designation not in args and '-c' not in args
        return actual_run(args,**kwargs)
    monkeypatch.setattr(aneos.subprocess,'run',only_readonly_git)
    assert aneos.main()==0
    data=json.loads(capsys.readouterr().out)
    assert data['designation']==designation
    assert data['screening_assessment']['candidate']


@pytest.mark.parametrize('alias',['analyze','analysis','a','simple','detect'])
def test_existing_launcher_aliases_share_pipeline(alias,pipeline,monkeypatch,capsys):
    import aneos
    from aneos_core.analysis import cli
    monkeypatch.setattr(cli,'create_analysis_pipeline',lambda:pipeline)
    monkeypatch.setattr(aneos.sys,'argv',['aneos.py',alias,'marker','--json','--force-refresh'])
    assert aneos.main()==0
    data=json.loads(capsys.readouterr().out)
    assert data['screening_assessment']==ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    pipeline.data_fetcher.fetch_neo_data.assert_called_once_with('marker',force_refresh=True)


def test_entering_menu_cannot_cleanup_research_inputs_or_mutate_environment(monkeypatch,tmp_path):
    from aneos_menu import ANEOSMenu
    import os
    before=dict(os.environ)
    menu=ANEOSMenu.__new__(ANEOSMenu)
    menu.console=None
    menu._auto_cleanup_cache=Mock(side_effect=AssertionError('implicit deletion forbidden'))
    menu._auto_optimize_settings=Mock(side_effect=AssertionError('implicit environment changes forbidden'))
    menu._auto_system_management()
    assert dict(os.environ)==before
    menu._auto_cleanup_cache.assert_not_called()


@pytest.mark.parametrize('which',['_execute_comprehensive_workflow','_execute_rapid_screening_workflow',
    '_execute_deep_investigation_workflow','_execute_publication_ready_workflow'])
def test_workflow_aliases_cannot_fabricate_signatures_or_publication_readiness(pipeline,which,capsys):
    result=getattr(menu(pipeline),which)()
    assert result.screening_assessment['candidate']
    text=capsys.readouterr().out
    assert 'Not calibrated' in text
    assert 'READY_FOR_PUBLICATION' not in text


def test_comparative_workflow_preserves_all_actual_outcomes(pipeline):
    ui=menu(pipeline)
    ui.get_input.side_effect=['first','second','']
    results=ui._execute_comparative_analysis_workflow()
    assert [r.designation for r in results]==['first','second']
    assert all(r.screening_assessment['is_artificial'] is None for r in results)


@pytest.mark.parametrize('available',[True,False])
def test_history_is_actual_computed_evidence_never_observed_epoch_zero(pipeline,available,capsys):
    ui=menu(pipeline)
    ui._orbital_history_source=Mock()
    records=[{'epoch':'2451545.0 = native TDB label','a':1.1,'e':.2,'i':15.}]
    ui._orbital_history_source.fetch_orbital_history.return_value=records if available else []
    result=ui.orbital_history_analysis()
    assert result.screening_assessment==ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    text=capsys.readouterr().out
    assert 'computed' in text.lower()
    assert 'OBSERVED - current epoch' not in text
    if available: assert '2451545.0' in text
    else: assert 'unavailable' in text


@pytest.mark.parametrize('json_output',[True,False])
def test_cli_reports_are_exclusive_and_source_failure_is_nonzero(pipeline,tmp_path,capsys,json_output):
    from aneos_core.analysis import cli
    pipeline.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('offline')
    target=tmp_path/('result.json' if json_output else 'result.md')
    args=['marker','--output',str(target)]+(['--json'] if json_output else [])
    assert cli.main(args,pipeline)==1
    text=target.read_text()
    assert 'source_unavailable' in text
    assert cli.main(args,pipeline)==2
    assert target.read_text()==text
    assert 'unavailable' in capsys.readouterr().out


def test_report_replay_rejects_tampering_and_legacy_claims(pipeline):
    import copy
    from types import SimpleNamespace
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report
    report=build_screening_report(asyncio.run(pipeline.analyze_neo('marker')))
    assert verify_screening_report(report)
    changed=copy.deepcopy(report);changed['neo_data']['designation']='different'
    with pytest.raises(ValueError,match='fingerprint'):verify_screening_report(changed)
    legacy=build_screening_report(SimpleNamespace(designation='marker',sigma_level=12,is_artificial=True))
    assert legacy['screening_assessment'] is None
    assert legacy['errors']


def test_old_threshold_conversions_and_placeholder_workflows_are_controlled_deprecations(pipeline):
    ui=menu(pipeline)
    for name,args in [('_calculate_sigma_level',(.99,)),('_filter_by_validated_detector_sigma',([],5)),
        ('_run_temporal_analysis',('marker',)),('_run_threat_assessment',('marker',{}))]:
        with pytest.raises(NotImplementedError):getattr(ui,name)(*args)
    assert ui._assess_peer_review_readiness(None,{})['ready_for_publication'] is False
    assert ui._calculate_validation_metrics(None,{})['false_positive_risk'] is None


def test_raw_adapter_physical_units_match_shared_screening(pipeline):
    ui=menu(pipeline)
    n=neo('marker');n.physical_properties.rotation_period_hours=.1
    pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:n
    orbital,physical=ui._get_test_data('marker')
    assert physical['diameter']==370.
    assert physical['_units']=={'diameter':'m','rotation_period':'h'}
    assert physical['rotation_period']==.1
    assert orbital==ScreeningAnalysisService().analyze_neo(n).evidence_snapshot['orbital']


def test_history_report_hash_covers_additional_evidence(pipeline):
    from aneos_core.analysis.reporting import verify_screening_report
    ui=menu(pipeline)
    ui._orbital_history_source=Mock()
    ui._orbital_history_source.fetch_orbital_history.return_value=[{'epoch':'provider-native epoch','a':1.}]
    ui._display_screening_report=Mock()
    ui.orbital_history_analysis()
    report=ui._display_screening_report.call_args.args[0]
    assert verify_screening_report(report)
    assert report['additional_evidence']['evidence_basis']=='computed_ephemeris'
    report['additional_evidence']['records'][0]['a']=3.
    with pytest.raises(ValueError):verify_screening_report(report)


def test_full_raw_source_binding_cannot_be_rehashed_to_cover_stale_evidence(pipeline):
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report,_report_hash
    report=build_screening_report(asyncio.run(pipeline.analyze_neo('marker')))
    report['neo_data']['physical_properties']['diameter_km']=.5
    report['report_id']=_report_hash(report)
    with pytest.raises(ValueError,match='review-input|differs'):verify_screening_report(report)


def test_markdown_preserves_native_a3_sigma_basis_and_covariance(pipeline):
    from aneos_core.analysis.reporting import build_screening_report,render_screening_markdown
    from aneos_core.data.models import NonGravitationalParameters
    from aneos_core.data.orbit_evidence import OrbitModelParameter,OrbitalCovariance
    n=neo();n.nongrav=NonGravitationalParameters(parameters={
        'A3':OrbitModelParameter('A3',1.2e-12,2e-13,'au/d^2','EST',raw_value='1.2e-12',raw_sigma='2e-13')},
        covariance=OrbitalCovariance(['A3'],[[4e-26]],2451545.))
    pipeline.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:n
    report=build_screening_report(asyncio.run(pipeline.analyze_neo('marker')))
    assert report['neo_data']['nongrav']==n.nongrav.to_dict()
    text=render_screening_markdown(report)
    assert '1.2e-12' in text and '2e-13' in text and 'au/d^2' in text
    assert 'fitted_parameter' in text and 'EST' in text
    assert 'Covariance: available' in text


def test_diagnostics_are_synthetic_contract_controls_not_discovery_evidence(pipeline,capsys):
    result=menu(pipeline)._test_validated_detector()
    assert result['control_passed']
    assert not result['five_sigma_claim_permitted']
    assert result['scientific_validation_status']=='unavailable'
    assert 'Not independent scientific validation' in capsys.readouterr().out


def test_prototype_statistical_helpers_cannot_reintroduce_fixed_priors_or_fake_samples(pipeline):
    ui=menu(pipeline)
    with pytest.raises(NotImplementedError):ui._collect_statistical_data('tesla')
    with pytest.raises(NotImplementedError):ui._perform_statistical_validation({}, {})


def test_report_records_runtime_source_hashes_without_claiming_clean_or_locked_state(pipeline):
    from aneos_core.analysis.reporting import build_screening_report
    report=build_screening_report(asyncio.run(pipeline.analyze_neo('marker')))
    software=report['software']
    assert software['python_version']
    assert software['float_mantissa_bits']==53
    assert software['package_versions']['numpy']
    assert len(software['source_hashes_at_report_time']['aneos_core/analysis/screening.py'])==64
    assert 'not loaded bytecode' in software['limitations']
