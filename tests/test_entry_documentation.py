"""Entry-doc examples exercised offline; not scientific or deployment certification."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'docs/archive/entry-guides-2026-11-13'
ENTRIES = ('README.md', 'docs/README.md', 'docs/user-guide/installation.md',
           'docs/user-guide/quick-start.md', 'docs/user-guide/menu-system.md')


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('network forbidden')))
    monkeypatch.setattr(socket, 'getaddrinfo', Mock(side_effect=AssertionError('DNS forbidden')))
    monkeypatch.setenv('ANEOS_DATABASE_URL', f'sqlite:///{tmp_path}/evidence.sqlite')


def run_offline(tmp_path, script, args, cwd=ROOT):
    guard = tmp_path / 'guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text(
        'import socket\n'
        'def denied(*a, **kw): raise AssertionError("network/DNS forbidden in documentation controls")\n'
        'socket.socket.connect=denied\nsocket.getaddrinfo=denied\n')
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=cwd,
        capture_output=True, text=True, timeout=40,
        env={**os.environ, 'PYTHONPATH':str(guard), 'PYTHONDONTWRITEBYTECODE':'1',
             'ANEOS_DATABASE_URL':f'sqlite:///{tmp_path}/evidence.sqlite'})


@pytest.mark.parametrize('record', json.loads((ARCHIVE / 'manifest.json').read_text())['records'])
def test_archives_preserve_exact_original_bytes(record):
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    data = (ROOT / record['archive_path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['sha256']
    assert len(data) == record['bytes']
    assert len(data.splitlines()) == record['lines']
    original = subprocess.check_output(['git','show',f"{manifest['source_commit']}:{record['source_path']}"],cwd=ROOT)
    assert data == original


@pytest.mark.parametrize('name', ENTRIES)
def test_reviewed_entry_document_links_resolve(name):
    path = ROOT / name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
        if re.match(r'https?://|mailto:|#', target):
            continue
        target = target.split('#')[0]
        assert (path.parent / target).resolve().exists(), (name,target)


@pytest.mark.parametrize('script,args,expected', [
    ('aneos.py',['--version'],'aNEOS 2.2.0b2 (beta)'),
    ('aneos.py',['help'],'The 5.0 ranking threshold is not Gaussian sigma'),
    ('aneos.py',['analyze','--help'],'--output'),
])
def test_documented_offline_cli_commands(tmp_path,script,args,expected):
    done = run_offline(tmp_path,script,args)
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout
    assert not (tmp_path / 'evidence.sqlite').exists()


def test_guide_option_table_matches_real_catalogue():
    from aneos_menu_catalog import MENU_OPTIONS
    doc = (ROOT / 'docs/user-guide/menu-system.md').read_text()
    for option in MENU_OPTIONS:
        assert f'| {option.choice} | {option.label} |' in doc
    rows = re.findall(r'^\| (\d+) \|',doc,re.M)
    assert len(rows) == 16 and set(rows) == {item.choice for item in MENU_OPTIONS}


def test_documented_local_api_arguments_dispatch_without_starting_server(monkeypatch):
    import aneos
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(aneos.subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['aneos.py','api','--dev','--host','127.0.0.1','--port','8000'])
    assert aneos.main() == 0
    run.assert_called_once_with([sys.executable,'start_api.py','--dev','--host','127.0.0.1','--port','8000'])


def synthetic_pipeline():
    from aneos_core.analysis.pipeline import AnalysisPipeline
    from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
    p=AnalysisPipeline()
    p.data_fetcher=Mock()
    p.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:NEOData(d,
        orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49),
        physical_properties=PhysicalProperties(diameter_km=.011,albedo=.52,absolute_magnitude_h=25.9))
    return p


@pytest.mark.parametrize('json_mode',[True,False])
def test_documented_named_object_report_flags_and_no_overwrite(tmp_path,monkeypatch,capsys,json_mode):
    from aneos_core.analysis import cli
    p=synthetic_pipeline()
    target=tmp_path / ('ky26.json' if json_mode else 'ky26.md')
    flags=['--json'] if json_mode else []
    assert cli.main(['1998 KY26',*flags,'--output',str(target)],pipeline=p)==0
    data=target.read_bytes()
    if json_mode:
        report=json.loads(data)
        assert report['designation']=='1998 KY26'
        assert report['screening_assessment']['artificial_probability'] is None
        assert report['screening_assessment']['analysis_metadata']['score_basis']=='legacy_uncalibrated_dependent_features'
        assert report['screening_assessment']['sigma_confidence'] is None
    else:
        assert b'Artificial-origin probability: Not calibrated' in data
    count=p.data_fetcher.fetch_neo_data.call_count
    assert cli.main(['1998 KY26',*flags,'--output',str(target)],pipeline=p)==2
    assert target.read_bytes()==data
    assert p.data_fetcher.fetch_neo_data.call_count==count
    assert 'already exists' in capsys.readouterr().err


def test_actual_dashboard_file_command_keeps_input_and_does_not_poll(tmp_path):
    from aneos_core.analysis.reporting import build_screening_report
    p=synthetic_pipeline()
    report=build_screening_report(asyncio.run(p.analyze_neo('1998 KY26')))
    path=tmp_path/'ky26.json'
    data=json.dumps(report).encode();path.write_bytes(data)
    done=run_offline(tmp_path,'artificial_neo_dashboard_launcher.py',['--file',str(path)],cwd=tmp_path)
    assert done.returncode==0,done.stderr
    assert 'Error loading' not in done.stdout
    assert path.read_bytes()==data
    # Authorized positive control: the real loader completes a saved dashboard,
    # rather than trivially passing denial checks by doing nothing.
    assert list((tmp_path/'dashboard_results').glob('screening_dashboard_*/complete.json'))
    assert not (tmp_path/'evidence.sqlite').exists()


def test_documented_historical_replay_is_offline_and_not_current_inference(tmp_path):
    target=tmp_path/'ky26-legacy-replay.json'
    done=run_offline(tmp_path,'scripts/reproduce_ky26_baseline.py',['--output',str(target)])
    assert done.returncode==0,done.stderr
    data=target.read_bytes();report=json.loads(data)
    assert report['result']['sigma_confidence']==pytest.approx(4.141288008229254,rel=1e-12)
    assert report['exact_historical_reproduction'] is False
    again=run_offline(tmp_path,'scripts/reproduce_ky26_baseline.py',['--output',str(target)])
    assert again.returncode!=0
    assert target.read_bytes()==data


@pytest.mark.parametrize('failure',['recorded_source_error','fatal_pipeline_error'])
def test_documented_cli_failure_statuses_are_not_successful_origin_findings(tmp_path,capsys,failure):
    from aneos_core.analysis import cli
    from aneos_core.utils.errors import DataSourceUnavailableError
    p=synthetic_pipeline()
    target=tmp_path/'failure.json'
    if failure=='recorded_source_error':
        p.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('synthetic source failure')
        assert cli.main(['1998 KY26','--json','--output',str(target)],pipeline=p)==1
        report=json.loads(target.read_text())
        assert report['errors']
        assert report['screening_assessment'] is None
        assert report['status']=='source_unavailable'
    else:
        from unittest.mock import AsyncMock
        p.analyze_neo=AsyncMock(side_effect=RuntimeError('synthetic fatal failure'))
        assert cli.main(['1998 KY26','--json','--output',str(target)],pipeline=p)==2
        assert not target.exists()
        assert 'Analysis/report unavailable' in capsys.readouterr().err
