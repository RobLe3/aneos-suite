"""Source-shaped synthetic controls for complete request accounting, not astronomy validation."""
import asyncio
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from aneos_core.analysis.batch_reporting import read_input,parse_input,build_batch_report,verify_batch_report,render_batch_report
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
from aneos_core.utils.errors import DataSourceUnavailableError


@pytest.fixture(autouse=True)
def offline(monkeypatch,tmp_path):
    import socket
    monkeypatch.setattr(socket.socket,'connect',Mock(side_effect=AssertionError('network forbidden')))
    monkeypatch.setattr(socket,'getaddrinfo',Mock(side_effect=AssertionError('DNS forbidden')))
    monkeypatch.setenv('ANEOS_DATABASE_URL',f'sqlite:///{tmp_path}/evidence.sqlite')
    monkeypatch.chdir(tmp_path)


def neo(name):
    if name=='missing':return NEOData(name)
    return NEOData(name,orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,
        inclination=180. if name=='candidate' else 1.49),
        physical_properties=PhysicalProperties(diameter_km=.011,albedo=.52,absolute_magnitude_h=25.9))


def pipeline():
    p=AnalysisPipeline();p.data_fetcher=Mock()
    def fetch(name,**kw):
        if name=='offline':raise DataSourceUnavailableError('synthetic unavailable')
        if name=='wrong':return neo('other')
        return neo(name)
    p.data_fetcher.fetch_neo_data.side_effect=fetch
    return p


def menu_for_file(monkeypatch,path,p,width=None):
    from aneos_menu_v2 import ANEOSMenuV2
    menu=ANEOSMenuV2()
    menu.console=None
    if width:
        from rich.console import Console
        menu.console=Console(file=io.StringIO(),width=width,color_system=None,record=True)
    menu.browse_files=Mock(return_value=str(path))
    menu._persist_detection_result_async=Mock()
    monkeypatch.setattr('aneos_core.analysis.pipeline.AnalysisPipeline',lambda:p)
    return menu


def test_actual_batch_keeps_every_occurrence_and_inconclusive(monkeypatch,tmp_path,capsys):
    path=tmp_path/'objects.txt';path.write_text('candidate\nmissing\noffline\ncandidate\nwrong\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu._detect_batch()
    assert len(menu._batch_reports)==1
    report=verify_batch_report(menu._batch_reports[0])
    assert [r['designation'] for r in report['requests']]==['candidate','missing','offline','candidate','wrong']
    assert [r['report']['status'] for r in report['requests']]==['completed','inconclusive','source_unavailable','completed','invalid_evidence']
    assert report['summary']['request_count']==5
    assert report['summary']['distinct_exact_designations']==4
    assert report['requests'][3]['reused_exact_request'] is True
    assert report['requests'][0]['report']==report['requests'][3]['report']
    assert p.data_fetcher.fetch_neo_data.call_count==4
    assert 'missing' in menu._detection_results
    text=capsys.readouterr().out
    assert 'source_unavailable' in text and 'INCONCLUSIVE' in text
    assert 'skipped' not in text and '5 rows' in text


def test_unreadable_file_does_not_escape_or_fetch(monkeypatch,tmp_path):
    p=pipeline();menu=menu_for_file(monkeypatch,tmp_path/'missing.txt',p)
    menu.show_error=Mock()
    menu._detect_batch()
    menu.show_error.assert_called_once()
    p.data_fetcher.fetch_neo_data.assert_not_called()


@pytest.fixture
def bundle():
    from aneos_core.analysis.reporting import build_screening_report
    source=parse_input(b'# source-shaped synthetic controls\ncandidate\nmissing\noffline\ncandidate\nwrong\n','objects.txt')
    p=pipeline();names=list(dict.fromkeys(r['designation'] for r in source['requests']))
    results=asyncio.run(p.analyze_batch(names))
    return build_batch_report(source,{r.designation:build_screening_report(r) for r in results},
                              created_at='2026-11-13T16:00:00+00:00')


@pytest.mark.parametrize('raw,name,expected',[
    (b'\xef\xbb\xbf# comment\r\n  candidate  \r\n\nmissing\n','objects.txt',[(2,'candidate'),(4,'missing')]),
    (b'designation\r\n"1998 KY26"\r\n\r\n# comment\r\ncandidate\r\n','objects.csv',[(2,'1998 KY26'),(5,'candidate')]),
    (b'candidate\ncandidate\nCandidate\n','objects.txt',[(1,'candidate'),(2,'candidate'),(3,'Candidate')]),
])
def test_input_line_identity_and_original_bytes(raw,name,expected):
    import base64,hashlib
    parsed=parse_input(raw,name)
    assert [(r['source_line'],r['designation']) for r in parsed['requests']]==expected
    assert base64.b64decode(parsed['content_base64'])==raw
    assert parsed['sha256']==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('raw,name',[
    (b'','empty.txt'),(b'# comment\n  # indented\n','empty.txt'),
    (b'designation,a\nA,1\n','invalid.csv'),(b'"unterminated','invalid.csv'),
    (b'\xff','invalid.txt'),(b'A\x00B','invalid.txt'),(b'A\x1b[31m','invalid.txt'),
    (b'A','invalid.json'),(b'A\n'*1001,'large.txt'),(b'X'*(1024*1024+1),'large.txt'),
])
def test_bad_input_rejected_before_source_work(raw,name):
    with pytest.raises((ValueError,UnicodeError)):parse_input(raw,name)


@pytest.mark.parametrize('change',['hash','count','order','line','reuse','identity','input','policy','time','report'])
def test_resealed_inconsistent_batch_is_refused(bundle,change):
    from aneos_core.data.enrichment import fingerprint
    edited=copy.deepcopy(bundle)
    if change=='hash':edited['batch_report_id']='0'*64
    elif change=='count':edited['summary']['request_count']=999
    elif change=='order':edited['requests'].reverse()
    elif change=='line':edited['requests'][0]['source_line']=99
    elif change=='reuse':edited['requests'][3]['reused_exact_request']=False
    elif change=='identity':edited['requests'][0]['designation']='other'
    elif change=='input':edited['source_input']['sha256']='0'*64
    elif change=='policy':edited['input_policy']['max_requests']=9999
    elif change=='time':edited['created_at']='2026-11-13T16:00:00'
    elif change=='report':edited['requests'][0]['report']['status']='success'
    if change!='hash':edited['batch_report_id']=fingerprint({k:v for k,v in edited.items() if k!='batch_report_id'})
    with pytest.raises(ValueError):verify_batch_report(edited)
    assert verify_batch_report(bundle)==bundle


@pytest.mark.parametrize('width',[80,120,160])
def test_rich_and_plain_accounting_are_readable(bundle,width,capsys):
    from rich.console import Console
    out=io.StringIO();console=Console(file=out,width=width,color_system=None,record=True)
    render_batch_report(bundle,console)
    text=out.getvalue()
    assert max(map(len,text.splitlines()))<=width
    assert '[bold]' not in text and 'not independent evidence' in text
    assert 'INCONCLUSIVE' in text and 'candidate' in text
    render_batch_report(bundle)
    plain=capsys.readouterr().out
    assert 'source_unavailable' in plain and 'invalid_evidence' in plain
    assert bundle['batch_report_id'] in plain


@pytest.mark.parametrize('fmt',['json','csv'])
def test_existing_exporter_preserves_all_requests_and_refuses_replacement(bundle,tmp_path,fmt):
    import csv
    from aneos_core.reporting.exporters import Exporter
    exporter=Exporter(str(tmp_path));target=tmp_path/f'batch.{fmt}'
    exporter.export_batch_report(bundle,str(target),format=fmt)
    before=target.read_bytes()
    if fmt=='json':
        assert verify_batch_report(json.loads(before))==bundle
    else:
        with target.open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==5
        assert verify_batch_report(json.loads(rows[0]['batch_envelope_json']))==bundle
        assert all(not row['batch_envelope_json'] for row in rows[1:])
        assert rows[2]['status']=='source_unavailable' and rows[2]['screening_score']==''
        assert all(json.loads(row['report_json'])==bundle['requests'][i]['report'] for i,row in enumerate(rows))
    with pytest.raises(FileExistsError):exporter.export_batch_report(bundle,str(target),format=fmt)
    assert target.read_bytes()==before


@pytest.mark.parametrize('width',[None,80,120,160])
def test_actual_menu_browse_and_export_keep_failures(monkeypatch,tmp_path,capsys,width):
    path=tmp_path/'objects.txt';path.write_text('candidate\nmissing\noffline\ncandidate\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p,width)
    menu._detect_batch();count=p.data_fetcher.fetch_neo_data.call_count
    # Exact candidate/latest view differs from request occurrence history.
    assert len(menu._detection_results)==2 and len(menu._batch_reports[0]['requests'])==4
    menu._ask=Mock(side_effect=['','3',''])
    menu._results_browser()
    target=tmp_path/'history.json'
    menu._ask=Mock(side_effect=['2','','1',str(target)])
    menu._export_results()
    actual=verify_batch_report(json.loads(target.read_text()))
    assert actual['requests'][2]['report']['status']=='source_unavailable'
    assert p.data_fetcher.fetch_neo_data.call_count==count
    assert menu._batch_reports[0]==actual


def test_all_unavailable_batch_is_browsable_and_exportable(monkeypatch,tmp_path):
    path=tmp_path/'objects.txt';path.write_text('offline\noffline\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu._detect_batch()
    assert not menu._detection_results
    menu._ask=Mock(side_effect=['','1'])
    menu._results_browser()
    target=tmp_path/'history.json';menu._ask=Mock(side_effect=['2','','1',str(target)])
    menu._export_results()
    assert len(json.loads(target.read_text())['requests'])==2


def test_bad_file_encoding_stays_in_menu_without_poll(monkeypatch,tmp_path):
    path=tmp_path/'bad.txt';path.write_bytes(b'\xff')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu.show_error=Mock();menu._detect_batch()
    assert not menu._batch_reports
    menu.show_error.assert_called_once();p.data_fetcher.fetch_neo_data.assert_not_called()


@pytest.mark.parametrize('change',['classification','success_without_assessment'])
def test_resealed_failure_report_cannot_claim_success_or_candidate(bundle,change):
    from aneos_core.analysis.reporting import verify_screening_report,_report_hash
    report=copy.deepcopy(bundle['requests'][2]['report'])
    if change=='classification':report['classification']='FOLLOW_UP_CANDIDATE'
    else:report['status']='completed'
    report['report_id']=_report_hash(report)
    with pytest.raises(ValueError):verify_screening_report(report)


def test_saved_batch_dashboard_loader_preserves_rows_without_acquisition(bundle):
    from aneos_core.analysis.artificial_neo_dashboard import records_from_file,classify_record
    records=records_from_file(bundle)
    assert len(records)==5
    assert records[3]['batch_request_context']['reused_exact_request'] is True
    classified=[classify_record(r) for r in records]
    assert classified[0].category=='follow_up_candidate'
    assert classified[1].category=='inconclusive'
    assert classified[2].category=='inconclusive'
    assert classified[2].review is None
    assert classified[2].input_record['frozen_report']['status']=='source_unavailable'
    assert all(c.artificial_probability is None for c in classified)


def test_csv_designations_cannot_be_spreadsheet_formulas(tmp_path):
    import csv
    from aneos_core.analysis.reporting import build_screening_report
    from aneos_core.reporting.exporters import Exporter
    names=['=1+1','+123','-123','@SUM(A1)']
    source=parse_input(('\n'.join(names)+'\n').encode(),'formula.txt')
    results=asyncio.run(pipeline().analyze_batch(names))
    report=build_batch_report(source,{r.designation:build_screening_report(r) for r in results})
    path=Exporter(str(tmp_path)).export_batch_report(report,format='csv')
    with open(path,newline='') as stream:rows=list(csv.DictReader(stream))
    assert [r['designation'] for r in rows]==["'"+name for name in names]
    assert [json.loads(r['report_json'])['designation'] for r in rows]==names


def test_reproducible_actual_before_after_capture_script(tmp_path):
    import os,subprocess,sys,hashlib
    root=Path(__file__).resolve().parents[1];target=tmp_path/'captures'
    command=[sys.executable,str(root/'scripts/qa_batch_menu.py'),'--output',str(target)]
    done=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=40,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert done.returncode==0,done.stderr
    manifest=json.loads((target/'manifest.json').read_text())
    for name,digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
    for width in (80,120,160):
        after=(target/f'after-{width}.txt').read_text()
        assert 'SYNTHETIC QA CONTROLS' in after
        assert max(map(len,after.splitlines()))<=width
        assert 'source_unavailable' in after and 'INCONCLUSIVE' in after
        assert '@font-face' not in (target/f'after-{width}.svg').read_text()
    assert 'skipped' in (target/'before-plain.txt').read_text()
    assert 'skipped' not in (target/'after-plain.txt').read_text()
    before=(target/'manifest.json').read_bytes()
    again=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=40,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert again.returncode!=0
    assert (target/'manifest.json').read_bytes()==before


def test_actual_case_variants_are_not_silently_merged(monkeypatch,tmp_path):
    path=tmp_path/'objects.txt';path.write_text('candidate\nCandidate\ncandidate\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu._detect_batch()
    assert p.data_fetcher.fetch_neo_data.call_count==2
    assert set(menu._detection_results)=={'candidate','Candidate'}
    assert menu._batch_reports[0]['summary']['distinct_exact_designations']==2


def test_actual_saved_batch_file_command_does_not_poll(bundle,tmp_path):
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[1]
    path=tmp_path/'batch.json';data=json.dumps(bundle).encode();path.write_bytes(data)
    guard=tmp_path/'guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text('import socket\n'
        'def deny(*a,**k): raise AssertionError("offline batch replay must not poll")\n'
        'socket.socket.connect=deny\nsocket.getaddrinfo=deny\n')
    done=subprocess.run([sys.executable,str(root/'artificial_neo_dashboard_launcher.py'),'--file',str(path)],
        cwd=tmp_path,capture_output=True,text=True,timeout=40,
        env={**os.environ,'PYTHONPATH':str(guard),'PYTHONDONTWRITEBYTECODE':'1'})
    assert done.returncode==0,done.stderr
    assert 'Error loading' not in done.stdout
    assert '5 input records' in done.stdout
    assert path.read_bytes()==data
    completed=list((tmp_path/'dashboard_results').glob('screening_dashboard_*/report.json'))
    assert len(completed)==1
    report=json.loads(completed[0].read_text())
    assert len(report['records'])==5
    assert report['records'][2]['input_record']['frozen_report']['status']=='source_unavailable'


def test_session_analytics_counts_all_failed_requests(monkeypatch,tmp_path,capsys):
    path=tmp_path/'objects.txt';path.write_text('offline\noffline\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu._detect_batch();before=p.data_fetcher.fetch_neo_data.call_count
    menu._detection_analytics()
    text=capsys.readouterr().out
    assert '2 request occurrences' in text
    assert '1 distinct exact designation labels' in text
    assert 'No screening results this session' not in text
    assert p.data_fetcher.fetch_neo_data.call_count==before


def test_analytics_export_never_overwrites_existing_file(monkeypatch,tmp_path,capsys):
    path=tmp_path/'objects.txt';path.write_text('candidate\n')
    p=pipeline();menu=menu_for_file(monkeypatch,path,p)
    menu._detect_batch()
    from aneos_core.reporting import exporters
    real=exporters.Exporter(str(tmp_path));actual=real.export_to_json
    real.export_to_json=lambda data,**kw:actual(data,output_file='fixed.json',**kw)
    monkeypatch.setattr(exporters,'Exporter',lambda:real)
    menu._ask=Mock(return_value='y');menu._detection_analytics()
    before=(tmp_path/'fixed.json').read_bytes()
    menu._detection_analytics()
    assert (tmp_path/'fixed.json').read_bytes()==before
    assert 'Export refused' in capsys.readouterr().out
