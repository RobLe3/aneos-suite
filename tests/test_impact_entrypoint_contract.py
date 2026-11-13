"""Impact-source/scenario boundaries and persistence controls, not calibration."""
import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


def source_report(designation='1998 KY26'):
    from aneos_core.data.sources.sentry import parse_sentry_response
    payload = {'signature': {'source': 'NASA/JPL Sentry Data API', 'version': '2.0'},
               'error': 'specified object not found'}
    return parse_sentry_response(json.dumps(payload).encode(), designation=designation,
                                 retrieved_at='2026-11-12T21:00:00Z')


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('External socket forbidden')))
    monkeypatch.setenv('ANEOS_EVIDENCE_ARCHIVE', str(tmp_path / 'archive'))


def menu():
    from aneos_menu_v2 import ANEOSMenuV2
    result = ANEOSMenuV2.__new__(ANEOSMenuV2)
    result.console = None
    result._detection_results = {}; result._impact_results = {}
    result._ask = Mock(return_value='1998 KY26')
    result.show_error = Mock(); result.show_info = Mock(); result.show_success = Mock()
    result.display_panel = Mock(); result.display_table = Mock()
    result._fetch_neo_data = Mock(side_effect=AssertionError('Unrelated orbit query'))
    return result


def test_menu_uses_same_source_report_not_arc_heuristic_and_archives(monkeypatch):
    from aneos_core.analysis import impact_evidence
    from aneos_core.data.sources.sentry import verify_sentry_report
    service = Mock(); service.assess.return_value = source_report()
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    m = menu(); m._impact_assessment()
    service.assess.assert_called_once_with('1998 KY26')
    stored = m._impact_results['1998 KY26']
    assert stored == source_report() and verify_sentry_report(stored)
    assert 'not_listed' in m.display_panel.call_args.args[0]
    assert 'not available' in m.display_panel.call_args.args[0]
    m._fetch_neo_data.assert_not_called()
    assert m.show_success.called
    m.show_error.assert_not_called()


@pytest.mark.parametrize('bad', [None, {'collision_probability': 0.9}, source_report('2020 SO')])
def test_menu_rejects_unbound_or_legacy_reports_without_storage(monkeypatch, bad):
    from aneos_core.analysis import impact_evidence
    service = Mock(); service.assess.return_value = bad
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    m = menu(); m._impact_assessment()
    assert not m._impact_results and m.show_error.called
    assert not m.show_success.called


def test_menu_source_outage_does_not_generate_zero_or_persist_success(monkeypatch):
    from aneos_core.analysis import impact_evidence
    service = Mock(); service.assess.side_effect = RuntimeError('synthetic unavailable')
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    m = menu(); m._impact_assessment()
    assert m.show_error.called and not m._impact_results


def test_legacy_comprehensive_probability_name_is_controlled_not_silently_heuristic():
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    from aneos_core.data.models import OrbitalElements
    with pytest.raises(NotImplementedError, match='unvalidated|Sentry'):
        ImpactProbabilityCalculator().calculate_comprehensive_impact_probability(
            OrbitalElements(semi_major_axis=1, eccentricity=.2, inclination=5))


def test_explicit_kinetic_scenario_retains_real_units_and_no_collision_claim():
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    result = ImpactProbabilityCalculator().calculate_conditional_impact_energy(mass_kg=2, velocity_km_s=3)
    assert result['energy_joules'] == 9e6
    assert result['energy_megatons_tnt'] == pytest.approx(9e6/4.184e15)
    assert result['collision_probability'] is None
    assert result['input_basis'] == 'caller_supplied_conditional_scenario_not_a_measurement'
    assert 'crater' not in result


@pytest.mark.parametrize('mass,speed', [(None, 2), (True, 2), (-1, 2), (1, float('nan')), (1, float('inf'))])
def test_scenario_rejects_invalid_inputs(mass, speed):
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    with pytest.raises(ValueError):
        ImpactProbabilityCalculator().calculate_conditional_impact_energy(mass_kg=mass, velocity_km_s=speed)


def test_empty_impact_domain_record_is_not_negligible_zero():
    from aneos_core.data.models import ImpactAssessment
    record = ImpactAssessment('1998 KY26', 'not_fitted')
    data = record.to_dict()
    assert data['collision_probability'] is None and data['calculation_confidence'] is None
    assert data['probability_uncertainty'] is None and data['risk_level'] == 'not_assessed'
    assert ImpactAssessment.from_dict({'designation':'1998 KY26'}).to_dict()['collision_probability'] is None


@pytest.mark.asyncio
async def test_enhanced_pipeline_attaches_real_source_without_origin_or_orbit_proxy(monkeypatch):
    from aneos_core.analysis.impact_enhanced_pipeline import ImpactEnhancedPipeline
    from aneos_core.analysis import impact_evidence
    service = Mock(); service.assess.return_value = source_report()
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    original = Mock(); original.to_dict.return_value = {'screening':'preserved'}
    wrapped = Mock(analyze_neo_with_validation=AsyncMock(return_value=original))
    pipeline = ImpactEnhancedPipeline(wrapped)
    result = await pipeline.analyze_neo('1998 KY26')
    assert result.enhanced_result is original
    assert result.to_dict()['impact_assessment'] == source_report()
    assert result.get_impact_summary()['collision_probability'] is None
    service.assess.assert_called_once_with('1998 KY26')


@pytest.mark.asyncio
async def test_optional_impact_outage_and_disable_preserve_screening_without_safe_verdict(monkeypatch):
    from aneos_core.analysis.impact_enhanced_pipeline import ImpactEnhancedPipeline
    from aneos_core.analysis import impact_evidence
    from aneos_core.data.provider_http import ProviderUnavailableError
    service = Mock(); service.assess.side_effect = ProviderUnavailableError('outage')
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    original = Mock(); original.to_dict.return_value = {'screening':'preserved'}
    wrapped = Mock(analyze_neo_with_validation=AsyncMock(return_value=original))
    pipeline = ImpactEnhancedPipeline(wrapped)
    with pytest.raises(TypeError):
        pipeline.impact_config['artificial_probability_threshold'] = .1
    failed = await pipeline.analyze_neo('1998 KY26')
    assert failed.enhanced_result is original
    assert failed.get_impact_summary()['status'] == 'source_unavailable'
    assert failed.get_impact_summary()['collision_probability'] is None
    service.assess.reset_mock()
    disabled = await pipeline.analyze_neo_comprehensive('1998 KY26', include_impact_assessment=False)
    assert disabled.get_impact_summary()['status'] == 'not_requested'
    service.assess.assert_not_called()


def test_archive_idempotency_history_tamper_and_restart_process(tmp_path):
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    import os, subprocess, sys
    archive = EvidenceArchive(tmp_path/'history')
    first = source_report()
    path = archive.put(first)
    assert archive.put(first) == path and archive.get(first['report_sha256']) == first
    from aneos_core.data.sources.sentry import parse_sentry_response
    second = parse_sentry_response(first['source_snapshot']['raw_utf8'].encode(),
        designation='1998 KY26', retrieved_at='2026-11-13T21:00:00Z')
    assert archive.put(second) != path and len(list((tmp_path/'history').glob('*.json'))) == 2
    script = "from aneos_core.reporting.evidence_archive import EvidenceArchive;import sys,json; print(json.dumps(EvidenceArchive(sys.argv[1]).get(sys.argv[2])))"
    child = subprocess.run([sys.executable,'-c',script,str(tmp_path/'history'),first['report_sha256']],
        capture_output=True,text=True,timeout=15,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == first
    path.write_text('tampered')
    with pytest.raises((ValueError, json.JSONDecodeError)): archive.put(first)
    assert path.read_text() == 'tampered'  # No destructive repair/overwrite.


def test_archive_write_failure_is_visible_without_losing_session_report(monkeypatch):
    from aneos_core.analysis import impact_evidence
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    service = Mock(); service.assess.return_value = source_report()
    monkeypatch.setattr(impact_evidence, 'get_sentry_impact_service', lambda: service)
    monkeypatch.setattr(EvidenceArchive, 'put', Mock(side_effect=OSError('disk full')))
    m = menu(); m._impact_assessment()
    assert m._impact_results['1998 KY26'] == source_report()
    assert not m.show_success.called
    assert 'NOT confirmed saved' in m.show_error.call_args.args[0]


def test_archive_rejects_symlink_and_invalid_snapshot(tmp_path):
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    root=tmp_path/'actual';root.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(root)
    with pytest.raises(ValueError): EvidenceArchive(alias).put(source_report())
    changed=source_report(); changed['collision_probability']=.9
    with pytest.raises(ValueError): EvidenceArchive(root).put(changed)
    with pytest.raises(ValueError): EvidenceArchive(root).get('../anything')


def test_concurrent_archive_publish_never_overwrites_or_leaves_partial_final(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    archive=EvidenceArchive(tmp_path/'parallel');report=source_report()
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths=list(pool.map(lambda _:archive.put(report),range(6)))
    assert len(set(paths))==1
    assert archive.get(report['report_sha256']) == report
    assert len(list((tmp_path/'parallel').glob('*.json')))==1
    assert not list((tmp_path/'parallel').glob('.pending-*'))


def test_explicit_legacy_diagnostics_are_not_probabilities_and_require_scenario_arc():
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    from aneos_core.data.models import OrbitalElements
    calculator=ImpactProbabilityCalculator()
    result=calculator.calculate_legacy_heuristic_diagnostics(OrbitalElements(semi_major_axis=1.2,eccentricity=.2,inclination=5),observation_arc_days=100)
    assert result['status']=='unvalidated_heuristic_scenario'
    assert result['collision_probability'] is None and result['probability_uncertainty'] is None
    assert result['calculation_confidence'] is None and result['risk_level']=='not_assessed'
    assert result['legacy_diagnostic_scalars']['geometric_arc_score'] >= 0
    with pytest.raises(TypeError): calculator.calculate_legacy_heuristic_diagnostics(OrbitalElements())


def test_legacy_menu_origin_and_geometry_cannot_supply_impact_risk():
    from aneos_menu import ANEOSMenu
    m=ANEOSMenu.__new__(ANEOSMenu)
    obj=Mock(artificial_probability=.99999,orbital_elements={'q':.5,'Q':1.5},close_approaches=[{'distance':.0001}])
    assert m._assess_impact_risk(obj)=='NOT_ASSESSED'
    assert m._has_impact_risk_indicators(obj) is False


def available_mc_report():
    from aneos_core.data.sources.sentry import parse_sentry_response
    raw=(Path(__file__).parent/'fixtures/sentry_101955_20261112.json').read_bytes()
    return parse_sentry_response(raw,designation='101955',retrieved_at='2026-11-13T00:00:00Z')


def test_available_menu_positive_browser_and_json_csv_exports_keep_complete_replay(monkeypatch,tmp_path):
    from aneos_core.analysis import impact_evidence
    from aneos_core.data.sources.sentry import verify_sentry_report
    service=Mock();service.assess.return_value=available_mc_report()
    monkeypatch.setattr(impact_evidence,'get_sentry_impact_service',lambda:service)
    m=menu();m._ask.return_value='101955';m._impact_assessment()
    assert m._impact_results['101955']['collision_probability'] > 0
    assert 'MC' in m.display_panel.call_args.args[0]
    m._ask.return_value='';m._results_browser()
    assert m.display_table.call_args.kwargs['rows'][0][3] == 'available'
    for fmt,suffix in [('1','json'),('2','csv')]:
        output=tmp_path/f'output.{suffix}'
        m._ask.side_effect=[fmt,str(output)]
        m._export_results()
        if fmt=='1':
            report=json.loads(output.read_text())['neos'][0]['impact_evidence']
        else:
            import csv
            lines=[l for l in output.read_text().splitlines() if not l.startswith('#')]
            record=list(csv.DictReader(lines))[0]
            report=json.loads(record['impact_evidence_json'])
        assert verify_sentry_report(report) and report==available_mc_report()
        m._ask.side_effect=[fmt,str(output)]
        before=output.read_bytes();m.show_success.reset_mock();m._export_results()
        assert output.read_bytes()==before and not m.show_success.called


@pytest.mark.asyncio
async def test_real_core_enhanced_wrapper_preserves_screening_and_available_source(monkeypatch):
    from aneos_core.analysis.pipeline import AnalysisPipeline
    from aneos_core.analysis.impact_enhanced_pipeline import create_impact_enhanced_pipeline
    from aneos_core.analysis import impact_evidence
    from aneos_core.data.models import NEOData,OrbitalElements
    monkeypatch.setattr(AnalysisPipeline,'_create_data_source_manager',lambda self:Mock())
    core=AnalysisPipeline();core.data_fetcher=Mock()
    actual=NEOData('101955',orbital_elements=OrbitalElements(semi_major_axis=1.2,eccentricity=.2,inclination=5))
    core.data_fetcher.fetch_neo_data.return_value=actual
    service=Mock();service.assess.return_value=available_mc_report()
    monkeypatch.setattr(impact_evidence,'get_sentry_impact_service',lambda:service)
    result=await create_impact_enhanced_pipeline(core).analyze_neo('101955')
    data=result.to_dict()
    assert data['screening_assessment']['screening_score'] is not None
    assert data['impact_assessment']==available_mc_report()
    assert data['impact_persistence']['status']=='archived_verified'
    assert result.get_impact_summary()['calculation_confidence'] is None


@pytest.mark.asyncio
async def test_enhanced_invalid_evidence_is_not_source_absence_or_zero(monkeypatch):
    from aneos_core.analysis.impact_enhanced_pipeline import ImpactEnhancedPipeline
    from aneos_core.analysis import impact_evidence
    service=Mock();service.assess.return_value=source_report('2020 SO')
    monkeypatch.setattr(impact_evidence,'get_sentry_impact_service',lambda:service)
    wrapped=Mock(analyze_neo_with_validation=AsyncMock(return_value=Mock()))
    result=await ImpactEnhancedPipeline(wrapped).analyze_neo('1998 KY26')
    assert result.get_impact_summary()['status']=='invalid_evidence'
    assert result.get_impact_summary()['collision_probability'] is None


def test_killed_staging_writer_does_not_publish_partial_or_damage_existing_history(tmp_path):
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    import os,subprocess,sys,time
    archive=EvidenceArchive(tmp_path/'history');first=source_report();old=archive.put(first)
    from aneos_core.data.sources.sentry import parse_sentry_response
    second=parse_sentry_response(first['source_snapshot']['raw_utf8'].encode(),designation='1998 KY26',retrieved_at='2026-11-13T21:00:00Z')
    input_path=tmp_path/'input.json';input_path.write_text(json.dumps(second));marker=tmp_path/'staged'
    script='''
import json,sys,time
from pathlib import Path
import aneos_core.reporting.evidence_archive as a
original=a.os.link
def pause(*args,**kwargs):
 Path(sys.argv[3]).write_text('staged before publish')
 time.sleep(60)
 return original(*args,**kwargs)
a.os.link=pause
a.EvidenceArchive(sys.argv[1]).put(json.loads(Path(sys.argv[2]).read_text()))
'''
    child=subprocess.Popen([sys.executable,'-c',script,str(tmp_path/'history'),str(input_path),str(marker)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+10
        while not marker.exists() and time.monotonic()<end:time.sleep(.02)
        assert marker.exists()
    finally:
        child.kill();child.wait(timeout=5)
    assert archive.get(first['report_sha256'])==first
    assert not (tmp_path/'history'/(second['report_sha256']+'.json')).exists()
    assert archive.put(second).exists()
    assert old.exists()  # No cleanup deletes old reports or the killed process's staging bytes.


@pytest.mark.parametrize('mass,speed',[(1,299792.458),(1,1e200),(1e308,100),(1,'3')])
def test_classical_scenario_rejects_nonphysical_or_unrepresentable_energy(mass,speed):
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    with pytest.raises(ValueError):
        ImpactProbabilityCalculator().calculate_conditional_impact_energy(mass_kg=mass,velocity_km_s=speed)


def test_scenario_zero_is_explicit_given_input_not_missing_probability():
    from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
    result=ImpactProbabilityCalculator().calculate_conditional_impact_energy(mass_kg=0,velocity_km_s=3)
    assert result['energy_joules']==0 and result['collision_probability'] is None


def test_unsafe_or_tampered_results_browser_and_export_never_coerce_zero(tmp_path):
    m=menu();m._impact_results['1998 KY26']={'collision_probability':.99}
    m._ask.return_value='';m._results_browser()
    assert m.display_table.call_args.kwargs['rows'][0][2]=='not available'
    target=tmp_path/'invalid.json';m._ask.side_effect=['1',str(target)]
    m._export_results()
    assert not target.exists() and not m.show_success.called


@pytest.mark.asyncio
async def test_enhanced_archive_failure_is_separate_from_valid_source(monkeypatch):
    from aneos_core.analysis.impact_enhanced_pipeline import ImpactEnhancedPipeline
    from aneos_core.analysis import impact_evidence
    from aneos_core.reporting.evidence_archive import EvidenceArchive
    service=Mock();service.assess.return_value=source_report()
    monkeypatch.setattr(impact_evidence,'get_sentry_impact_service',lambda:service)
    monkeypatch.setattr(EvidenceArchive,'put',Mock(side_effect=OSError('full disk')))
    wrapped=Mock(analyze_neo_with_validation=AsyncMock(return_value=Mock()))
    result=await ImpactEnhancedPipeline(wrapped).analyze_neo('1998 KY26')
    assert result.impact_assessment==source_report()
    assert result.impact_persistence['status']=='not_confirmed_saved'
    assert result.get_impact_summary()['risk_level']=='not_assessed'


def test_offline_gate_fails_caught_network_attempt_and_allows_explicit_local_control(tmp_path):
    """Child pytest proves the gate itself, with no external provider involved."""
    import os,subprocess,sys
    (tmp_path/'conftest.py').write_text((Path(__file__).parents[1]/'conftest.py').read_text())
    (tmp_path/'pytest.ini').write_text('[pytest]\nmarkers =\n    network: explicitly authorized local network control\n')
    (tmp_path/'test_gate.py').write_text('''
import pytest,socket,threading

def test_caught_offline_attempt():
 try: socket.getaddrinfo('never-look-up.invalid',443)
 except AssertionError: pass

@pytest.mark.network
def test_authorized_local_positive():
 with socket.socket() as server:
  server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(3)
  def send():
   connection,_=server.accept()
   with connection: connection.sendall(b'synthetic-control')
  worker=threading.Thread(target=send);worker.start()
  with socket.create_connection(server.getsockname(),timeout=3) as client:
   assert client.recv(100)==b'synthetic-control'
  worker.join(timeout=3)
  assert not worker.is_alive()
''')
    child=subprocess.run([sys.executable,'-m','pytest','test_gate.py','-q','--tb=short'],cwd=tmp_path,
        capture_output=True,text=True,timeout=20,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert child.returncode==1,child.stdout+child.stderr
    assert 'Offline test attempted live network' in child.stdout
    assert '2 passed' in child.stdout and '1 error' in child.stdout
