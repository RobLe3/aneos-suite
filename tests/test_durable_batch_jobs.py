"""Real file-SQLite/worker controls; synthetic acquisition is not astronomical calibration."""
import asyncio
import copy
import threading
import time
from unittest.mock import Mock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore,LostLease,JobStorageUnavailable
from aneos_api.batch_jobs import (KIND,BatchWorker,BatchJobHandler,BatchRuntime,build_request,verify_result)
from aneos_core.analysis.batch_reporting import parse_input
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.data.models import NEOData,OrbitalElements
from aneos_core.utils.errors import DataSourceUnavailableError

PROFILE='1'*64


@pytest.fixture(autouse=True)
def offline(monkeypatch,tmp_path):
    import socket
    monkeypatch.setattr(socket.socket,'connect',Mock(side_effect=AssertionError('network forbidden')))
    monkeypatch.setattr(socket,'getaddrinfo',Mock(side_effect=AssertionError('DNS forbidden')))
    monkeypatch.setenv('ANEOS_DATABASE_URL',f'sqlite:///{tmp_path}/evidence.db')


@pytest.fixture
def store(tmp_path):
    engine=create_engine(f'sqlite:///{tmp_path}/jobs.db',connect_args={'check_same_thread':False,'timeout':5})
    s=JobStore(sessionmaker(bind=engine),capacity=8);s.initialize_schema()
    yield s,engine
    engine.dispose()


def request(names='A\nB\nA\n',tries=3):
    return build_request(parse_input(names.encode(),'synthetic.txt'),source_profile_id=PROFILE,max_attempts=tries,source_attempts=tries)


def pipeline():
    p=AnalysisPipeline();p.data_fetcher=Mock()
    p.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:NEOData(d,orbital_elements=OrbitalElements(
        semi_major_axis=1.23,eccentricity=.2,inclination=180. if d=='A' else 1.49))
    return p


def worker(store,p,**kwargs):
    return BatchWorker(store,handler=BatchJobHandler(pipeline_factory=lambda:p,source_profile=lambda:PROFILE),
                       lease_seconds=2,retry_seconds=.01,**kwargs)


def submit(store,req=None,key='recovery'):
    req=req or request()
    return store.submit(owner_scope='local-test',kind=KIND,request=req,idempotency_key=key,
                        max_attempts=req['max_attempts'])


def test_acknowledged_submission_and_all_occurrences_commit(store):
    s,_=store;p=pipeline();job=submit(s)
    assert worker(s,p).run_once(job_id=job['job_id'])=='completed'
    saved=s.get('local-test',job['job_id']);cps=s.read_checkpoints('local-test',job['job_id'])
    result=verify_result(saved['request'],cps,saved['result'])
    assert [r['designation'] for r in result['batch_report']['requests']]==['A','B','A']
    assert p.data_fetcher.fetch_neo_data.call_count==2
    assert saved['status']=='completed' and len(cps)==4
    assert s.read_checkpoints('other-owner',job['job_id']) is None


def test_transient_failure_retries_only_failed_object_and_preserves_attempt(store):
    s,_=store;p=pipeline();original=p.data_fetcher.fetch_neo_data.side_effect
    def failed(d,**kw):
        if d=='B':raise DataSourceUnavailableError('private-provider-key=SECRET_MARKER')
        return original(d,**kw)
    p.data_fetcher.fetch_neo_data.side_effect=failed;job=submit(s)
    w=worker(s,p);assert w.run_once()=='failure_recorded'
    assert s.get('local-test',job['job_id'])['status']=='retry_wait'
    p.data_fetcher.fetch_neo_data.reset_mock();p.data_fetcher.fetch_neo_data.side_effect=original
    time.sleep(.03);assert w.run_once()=='completed'
    assert [c.args[0] for c in p.data_fetcher.fetch_neo_data.call_args_list]==['B']
    cps=s.read_checkpoints('local-test',job['job_id'])
    assert 'object:1:attempt:1' in cps and 'object:1:attempt:2' in cps
    assert 'SECRET_MARKER' not in str(cps)


def test_exhausted_source_failure_is_explicit_report_not_zero(store):
    s,_=store;p=pipeline();p.data_fetcher.fetch_neo_data.side_effect=DataSourceUnavailableError('offline')
    job=submit(s,request('A\nA\n',tries=1))
    assert worker(s,p).run_once()=='completed'
    result=s.get('local-test',job['job_id'])['result']
    assert result['processing_status']=='completed_with_unavailable_or_invalid_requests'
    r=result['batch_report']['requests'][0]['report']
    assert r['classification']=='INCONCLUSIVE' and r['screening_assessment'] is None
    assert result['batch_report']['summary']['request_count']==2


def test_targeted_claim_does_not_execute_other_queued_job(store):
    s,_=store;p=pipeline();first=submit(s,request('A\n'),key='first');second=submit(s,request('B\n'),key='second')
    assert worker(s,p).run_once(job_id=second['job_id'])=='completed'
    assert s.get('local-test',first['job_id'])['status']=='queued'
    assert [c.args[0] for c in p.data_fetcher.fetch_neo_data.call_args_list]==['B']


def test_checkpoint_callback_failure_cannot_be_swallowed_as_progress(store):
    p=pipeline()
    def denied(result):raise JobStorageUnavailable('unconfirmed checkpoint')
    with pytest.raises(JobStorageUnavailable):asyncio.run(p.analyze_batch(['A'],result_callback=denied))


def test_owner_scope_and_targeted_runtime_cannot_claim_foreign_job(store):
    s,_=store;p=pipeline();job=submit(s)
    runtime=BatchRuntime(s,owner_scope='other-owner',handler=BatchJobHandler(
        pipeline_factory=lambda:p,source_profile=lambda:PROFILE))
    with pytest.raises(LookupError):runtime.run(job['job_id'])
    assert s.get('local-test',job['job_id'])['status']=='queued'
    p.data_fetcher.fetch_neo_data.assert_not_called()


def test_ready_checkpoint_recovery_does_not_initialize_drifted_provider(store):
    s,_=store;p=pipeline();job=submit(s)
    w=worker(s,p);original=s.finish
    def uncertain(*a):raise JobStorageUnavailable('finish unacknowledged')
    s.finish=uncertain
    with pytest.raises(JobStorageUnavailable):w.run_once()
    cps=s.read_checkpoints('local-test',job['job_id']);assert 'result' in cps
    s.finish=original;s.clock=lambda:time.time()+1000
    forbidden=Mock(side_effect=AssertionError('fully checkpointed recovery must not acquire'))
    h=BatchJobHandler(pipeline_factory=forbidden,source_profile=forbidden)
    assert BatchWorker(s,handler=h).run_once(job_id=job['job_id'])=='completed'
    forbidden.assert_not_called()


def test_checkpoint_commit_acknowledgement_loss_reuses_saved_report(store):
    s,_=store;p=pipeline();req=request('A\n')
    job=submit(s,req);original=s.checkpoint
    def uncertain(lease,key,payload):
        original(lease,key,payload)
        raise JobStorageUnavailable('checkpoint committed but acknowledgement lost')
    s.checkpoint=uncertain
    with pytest.raises(JobStorageUnavailable):worker(s,p).run_once()
    assert len(s.read_checkpoints('local-test',job['job_id']))==1
    s.checkpoint=original;s.clock=lambda:time.time()+1000
    forbidden=Mock(side_effect=AssertionError('do not repeat committed acquisition'))
    assert BatchWorker(s,handler=BatchJobHandler(pipeline_factory=forbidden,source_profile=forbidden)).run_once()=='completed'
    forbidden.assert_not_called()


def test_attempt_budget_is_fenced_and_cannot_be_forged(store):
    from dataclasses import replace
    s,_=store;job=submit(s);lease=s.claim('worker',kinds=[KIND])
    forged=replace(lease,max_attempts=5)
    with pytest.raises(LostLease):s.checkpoint(forged,'bad',{})
    with pytest.raises(LostLease):s.renew(forged)
    with pytest.raises(LostLease):s.checkpoints(forged)
    assert s.get('local-test',job['job_id'])['status']=='running'


def test_cancelled_provider_cannot_publish_report_after_return(store):
    s,_=store;p=pipeline();entered=threading.Event();release=threading.Event()
    original=p.data_fetcher.fetch_neo_data.side_effect
    def block(name,**kwargs):
        entered.set();release.wait(5)
        return original(name,**kwargs)
    p.data_fetcher.fetch_neo_data.side_effect=block;job=submit(s,request('A\n'))
    result=[];thread=threading.Thread(target=lambda:result.append(worker(s,p).run_once()))
    thread.start();assert entered.wait(3)
    assert s.cancel('local-test',job['job_id'])
    release.set();thread.join(5)
    assert not thread.is_alive() and result==['lease_lost']
    state=s.get('local-test',job['job_id'])
    assert state['status']=='cancelled' and state['result'] is None
    assert not s.read_checkpoints('local-test',job['job_id'])


def test_policy_drift_blocks_before_pipeline_source_work(store):
    s,_=store;p=pipeline();job=submit(s)
    factory=Mock(return_value=p)
    h=BatchJobHandler(pipeline_factory=factory,source_profile=lambda:'2'*64)
    assert BatchWorker(s,handler=h).run_once()=='failure_recorded'
    assert s.get('local-test',job['job_id'])['error_code']=='SOURCE_CONFIGURATION_CHANGED'
    factory.assert_not_called()


def test_before_item_failure_is_not_a_scientific_result():
    p=pipeline()
    def denied(name):raise LostLease('cancelled')
    with pytest.raises(LostLease):asyncio.run(p.analyze_batch(['A'],before_item_callback=denied))
    p.data_fetcher.fetch_neo_data.assert_not_called()


@pytest.mark.parametrize('change',['version','source','budget','force','execution','input'])
def test_invalid_frozen_request_cannot_poll(store,change):
    from aneos_api.batch_jobs import validate_request
    r=request()
    if change=='version':r['pipeline_version']='invented'
    elif change=='source':r['source_profile_id']='bad'
    elif change=='budget':r['max_attempts']=0
    elif change=='force':r['force_refresh']='yes'
    elif change=='execution':r['execution']['max_workers']=0
    elif change=='input':r['source_input']['sha256']='0'*64
    with pytest.raises(ValueError):validate_request(r)


def test_idempotent_submission_and_conflict_preserve_original(store):
    from aneos_api.job_store import IdempotencyConflict
    s,_=store;req=request()
    one=submit(s,req);two=submit(s,req)
    assert one['job_id']==two['job_id']
    with pytest.raises(IdempotencyConflict):submit(s,request('C\n'))
    assert s.get('local-test',one['job_id'])['request']==req


def test_terminal_result_cannot_disagree_with_committed_input(store):
    s,_=store;p=pipeline();job=submit(s);worker(s,p).run_once()
    state=s.get('local-test',job['job_id']);result=copy.deepcopy(state['result'])
    result['batch_report']['summary']['request_count']=123
    with pytest.raises(ValueError):verify_result(state['request'],s.read_checkpoints('local-test',job['job_id']),result)


def test_readonly_completed_inspection_does_not_enable_new_work(store,monkeypatch):
    from aneos_api.batch_jobs import default_runtime
    s,engine=store;p=pipeline();job=submit(s);worker(s,p).run_once()
    # Same trusted local scope injected explicitly; default URI host remains read-only.
    monkeypatch.setenv('ANEOS_DATABASE_URL','sqlite:///file:'+engine.url.database+'?mode=ro&uri=true')
    runtime=default_runtime();runtime.owner_scope='local-test'
    try:
        assert runtime.read_only
        assert runtime.run(job['job_id'])['status']=='completed'
        with pytest.raises(ValueError):runtime.submit(parse_input(b'A\n','new.txt'),'new-key')
        pending=submit(s,request('C\n'),key='pending')
        with pytest.raises(ValueError):runtime.run(pending['job_id'])
    finally:runtime.close()


def test_missing_or_memory_default_store_refuses_before_provider(monkeypatch):
    from aneos_api.batch_jobs import default_runtime
    monkeypatch.delenv('ANEOS_DATABASE_URL',raising=False)
    with pytest.raises(ValueError):default_runtime()
    monkeypatch.setenv('ANEOS_DATABASE_URL','sqlite:///:memory:')
    with pytest.raises(ValueError):default_runtime()


def test_primary_menu_survives_restart_without_polling_or_duplicate_db_write(monkeypatch,tmp_path):
    from aneos_menu_v2 import ANEOSMenuV2
    p=pipeline()
    monkeypatch.setattr('aneos_core.analysis.pipeline.AnalysisPipeline',lambda:p)
    path=tmp_path/'objects.txt';path.write_text('A\nB\nA\n')
    first=ANEOSMenuV2();first.console=None;first._ask=Mock(return_value='')
    first.browse_files=Mock(return_value=str(path));first._persist_detection_result_async=Mock()
    first._detect_batch()
    assert len(first._batch_reports)==1
    old=copy.deepcopy(first._batch_reports[0]);job_id=first._last_batch_job_id
    calls=p.data_fetcher.fetch_neo_data.call_count
    restarted=ANEOSMenuV2();restarted.console=None;restarted._ask=Mock(return_value=job_id)
    restarted.browse_files=Mock(side_effect=AssertionError('resume must not reread mutable file'))
    restarted._persist_detection_result_async=Mock()
    path.write_text('CHANGED\n')
    restarted._detect_batch()
    assert restarted._batch_reports==[old]
    assert p.data_fetcher.fetch_neo_data.call_count==calls
    restarted._persist_detection_result_async.assert_not_called()


def test_primary_menu_recovery_key_survives_ambiguous_submission_ack(monkeypatch,tmp_path):
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_api.batch_jobs import default_runtime
    p=pipeline();monkeypatch.setattr('aneos_core.analysis.pipeline.AnalysisPipeline',lambda:p)
    path=tmp_path/'objects.txt';path.write_text('A\n')
    menu=ANEOSMenuV2();menu.console=None;menu._ask=Mock(return_value='')
    menu.browse_files=Mock(return_value=str(path));menu._persist_detection_result_async=Mock()
    runtime=default_runtime();original=runtime.submit
    def uncertain(*a,**kw):
        original(*a,**kw)
        raise JobStorageUnavailable('submission committed but acknowledgement lost')
    runtime.submit=uncertain;menu._open_batch_runtime=Mock(return_value=runtime)
    menu._detect_batch()
    assert not menu._batch_reports
    p.data_fetcher.fetch_neo_data.assert_not_called()
    recovered=ANEOSMenuV2();recovered.console=None
    recovered._ask=Mock(return_value=menu._last_batch_recovery_key)
    recovered._persist_detection_result_async=Mock();recovered._detect_batch()
    assert recovered._batch_reports[0]['summary']['request_count']==1
    assert p.data_fetcher.fetch_neo_data.call_count==1


def test_job_worker_checkpoint_corruption_or_replacement_is_refused(store):
    from aneos_api.batch_jobs import _records
    s,_=store;p=pipeline();job=submit(s);worker(s,p).run_once()
    state=s.get('local-test',job['job_id']);cps=s.read_checkpoints('local-test',job['job_id'])
    edited=copy.deepcopy(cps);edited['object:00:attempt:01']=edited.pop('object:0:attempt:1')
    with pytest.raises(ValueError):_records(state['request'],edited)
    edited=copy.deepcopy(cps);duplicate=copy.deepcopy(edited['object:0:attempt:1']);duplicate['attempt']=2
    edited['object:0:attempt:2']=duplicate
    with pytest.raises(ValueError):_records(state['request'],edited)


def test_real_killed_process_recovers_checkpoints_then_fresh_reader(tmp_path,store):
    import json,os,subprocess,sys
    s,engine=store;job=submit(s,request('A\nB\n'))
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    marker=tmp_path/'blocked';log=tmp_path/'calls.txt'
    script=r'''
import os,sys,time,json,socket
def denied(*a,**k):raise AssertionError('No external network in worker controls')
socket.socket.connect=denied;socket.getaddrinfo=denied
from types import SimpleNamespace
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
from aneos_api.batch_jobs import BatchWorker,BatchJobHandler,verify_result
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.data.models import NEOData,OrbitalElements
engine=create_engine(sys.argv[1],connect_args={'check_same_thread':False})
s=JobStore(sessionmaker(bind=engine),capacity=8,clock=lambda:float(sys.argv[5]))
if sys.argv[2]=='read':
 state=s.get('local-test',sys.argv[3]);verify_result(state['request'],s.read_checkpoints('local-test',sys.argv[3]),state['result'])
 print(state['status']);sys.exit(0)
p=AnalysisPipeline()
def fetch(name,**kwargs):
 with open(sys.argv[6],'a') as out:out.write(sys.argv[2]+':'+name+'\n')
 if sys.argv[2]=='first' and name=='B':
  open(sys.argv[4],'w').write('blocked')
  while True:time.sleep(.05)
 if sys.argv[2]=='second' and name=='A':raise AssertionError('checkpointed A was reacquired')
 return NEOData(name,orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49))
p.data_fetcher=SimpleNamespace(fetch_neo_data=fetch)
w=BatchWorker(s,handler=BatchJobHandler(pipeline_factory=lambda:p,source_profile=lambda:'1'*64),lease_seconds=60)
print(w.run_once(job_id=sys.argv[3]))
'''
    # Controlled lease clocks avoid relying on scheduler-specific short leases.
    env={**os.environ,'PYTHONPATH':str(root),'PYTHONDONTWRITEBYTECODE':'1'}
    args=[sys.executable,'-c',script,str(engine.url),'first',job['job_id'],str(marker),str(s.clock()),str(log)]
    proc=subprocess.Popen(args,cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            cps=s.read_checkpoints('local-test',job['job_id'])
            if marker.exists() and 'object:0:attempt:1' in cps:break
            if proc.poll() is not None:raise AssertionError(proc.communicate())
            time.sleep(.03)
        else:raise AssertionError('child never reached checkpoint/blocked-provider boundary')
        proc.kill();proc.communicate(timeout=5)
        args[4]='second';args[7]=str(s.clock()+1000)
        resumed=subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=20)
        assert resumed.returncode==0,resumed.stderr
        assert 'completed' in resumed.stdout
        args[4]='read'
        read=subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
        assert read.returncode==0 and 'completed' in read.stdout,read.stderr
        assert 'second:A' not in log.read_text() and 'second:B' in log.read_text()
    finally:
        if proc.poll() is None:proc.kill();proc.communicate(timeout=5)


def test_runtime_repeated_submission_uses_original_creation_time_and_conflicts(store):
    from aneos_api.job_store import IdempotencyConflict
    s,_=store;p=pipeline();runtime=BatchRuntime(s,owner_scope='local-test',handler=BatchJobHandler(
        pipeline_factory=lambda:p,source_profile=lambda:PROFILE))
    source=parse_input(b'A\n','objects.txt')
    one=runtime.submit(source,'stable-key');time.sleep(.01)
    two=runtime.submit(source,'stable-key')
    assert one['job_id']==two['job_id'] and one['request']==two['request']
    with pytest.raises(IdempotencyConflict):runtime.submit(parse_input(b'B\n','objects.txt'),'stable-key')
    p.data_fetcher.fetch_neo_data.assert_not_called()


def test_runtime_concurrent_idempotency_does_not_duplicate_jobs(store):
    s,_=store;p=pipeline();runtime=BatchRuntime(s,owner_scope='local-test',handler=BatchJobHandler(
        pipeline_factory=lambda:p,source_profile=lambda:PROFILE))
    source=parse_input(b'A\n','objects.txt')
    barrier=threading.Barrier(2);answers=[];errors=[]
    def submit_one():
        try:
            barrier.wait(timeout=3);answers.append(runtime.submit(source,'concurrent'))
        except Exception as exc:errors.append(exc)
    threads=[threading.Thread(target=submit_one) for _ in range(2)]
    for t in threads:t.start()
    for t in threads:t.join(5)
    assert not any(t.is_alive() for t in threads) and not errors
    assert len(answers)==2 and answers[0]['job_id']==answers[1]['job_id']


@pytest.mark.parametrize('change',['identity','sigma_permission'])
def test_rehashed_report_cannot_change_object_or_enable_discovery_claim(change):
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report,_report_hash
    result=asyncio.run(pipeline().analyze_neo('A'))
    report=build_screening_report(result)
    if change=='identity':report['designation']='OTHER'
    else:report['five_sigma_claim_permitted']=True
    report['report_id']=_report_hash(report)
    with pytest.raises(ValueError):verify_screening_report(report)


def test_submission_report_assembly_and_source_dates_remain_separate(store):
    s,_=store;p=pipeline()
    req=request('A\n');req['created_at']='2020-01-01T00:00:00+00:00'
    job=submit(s,req);worker(s,p).run_once()
    state=s.get('local-test',job['job_id']);cps=s.read_checkpoints('local-test',job['job_id'])
    stamp=state['result']['batch_report']['created_at']
    assert stamp==cps['assembly']['created_at'] and stamp!=req['created_at']
    assert cps['assembly']['timestamp_role']=='local_report_assembly_not_observation'
    assert state['result']['batch_report']['requests'][0]['report']['neo_data']['fetched_at'] is None


@pytest.mark.parametrize('claim',['scientific_validation_status','validation_status','analysis_metadata'])
def test_rehashed_report_cannot_promote_unavailable_validation(claim):
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report,_report_hash
    report=build_screening_report(asyncio.run(pipeline().analyze_neo('A')))
    if claim=='analysis_metadata':report[claim]['five_sigma_claim_permitted']=True
    else:report[claim]='independently_validated'
    report['report_id']=_report_hash(report)
    with pytest.raises(ValueError):verify_screening_report(report)


@pytest.mark.parametrize('width',[None,80,120,160])
def test_partial_menu_progress_lists_all_rows_and_correct_execution_time_role(store,monkeypatch,tmp_path,capsys,width):
    import io
    from aneos_menu_v2 import ANEOSMenuV2
    from rich.console import Console
    from rich.cells import cell_len
    s,_=store;p=pipeline();original=p.data_fetcher.fetch_neo_data.side_effect
    p.data_fetcher.fetch_neo_data.side_effect=lambda d,**kw:(_ for _ in ()).throw(DataSourceUnavailableError('offline')) if d=='B' else original(d,**kw)
    runtime=BatchRuntime(s,owner_scope='local-test',handler=BatchJobHandler(
        pipeline_factory=lambda:p,source_profile=lambda:PROFILE),retry_seconds=20)
    job=runtime.submit(parse_input(b'A\nB\nA\n','objects.txt'),'partial',source_attempts=2)
    runtime.run(job['job_id'])
    before=p.data_fetcher.fetch_neo_data.call_count
    menu=ANEOSMenuV2();out=io.StringIO()
    menu.console=Console(file=out,width=width,color_system=None) if width else None
    menu._ask=Mock(side_effect=[job['job_id'],''])
    menu._open_batch_runtime=Mock(return_value=runtime)
    menu._detect_batch()
    text=out.getvalue() if width else capsys.readouterr().out
    normalized=' '.join(text.split())
    assert 'retry_wait' in text and 'not an observation date' in normalized
    assert 'No completed batch published' in normalized
    assert 'A' in text and 'B' in text and 'source_unavailable' in normalized
    assert not menu._batch_reports
    if width:assert max(map(cell_len,text.splitlines()))<=width
    assert p.data_fetcher.fetch_neo_data.call_count==before


def test_cancelled_menu_does_not_offer_impossible_retry(store,monkeypatch,capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    s,_=store;p=pipeline();job=submit(s);s.cancel('local-test',job['job_id'])
    runtime=BatchRuntime(s,owner_scope='local-test',handler=BatchJobHandler(
        pipeline_factory=lambda:p,source_profile=lambda:PROFILE))
    menu=ANEOSMenuV2();menu.console=None;menu._ask=Mock(side_effect=[job['job_id'],''])
    menu._open_batch_runtime=Mock(return_value=runtime);menu._detect_batch()
    text=capsys.readouterr().out
    assert 'Terminal execution state' in text
    assert 'not_assessed' in text and 'No completed batch published' in text
    assert 'Execution lease/queue eligibility' not in text
    p.data_fetcher.fetch_neo_data.assert_not_called()


def test_corrupt_configured_store_is_not_replaced_or_bypassed(monkeypatch,tmp_path,capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    db=tmp_path/'corrupt.sqlite';original=b'OWNED SYNTHETIC CORRUPT SQLITE CONTROL';db.write_bytes(original)
    monkeypatch.setenv('ANEOS_DATABASE_URL',f'sqlite:///{db}')
    p=pipeline();monkeypatch.setattr('aneos_core.analysis.pipeline.AnalysisPipeline',lambda:p)
    path=tmp_path/'objects.txt';path.write_text('A\n')
    menu=ANEOSMenuV2();menu.console=None;menu._ask=Mock(return_value='')
    menu.browse_files=Mock(return_value=str(path));menu._detect_batch()
    text=capsys.readouterr().out
    assert 'Recovery key' in text and 'Acknowledged SQL batch job' not in text
    assert not menu._batch_reports and db.read_bytes()==original
    p.data_fetcher.fetch_neo_data.assert_not_called()


def test_raw_literature_claim_does_not_authorize_discovery_flag():
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report
    p=pipeline();n=NEOData('A',orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49),
        evidence_metadata={'literature_claim':{'five_sigma_claim_permitted':True}})
    report=build_screening_report(asyncio.run(p.analyze_neo('A',n)))
    assert verify_screening_report(report)
    assert report['five_sigma_claim_permitted'] is False
    assert report['neo_data']['evidence_metadata']['literature_claim']['five_sigma_claim_permitted'] is True


@pytest.mark.parametrize('field,value',[('human_origin_status','excluded'),('origin_status','confirmed_artificial'),
                                        ('evidence_id','0'*64)])
def test_rehashed_owned_metadata_cannot_override_canonical_inference(field,value):
    from aneos_core.analysis.reporting import build_screening_report,verify_screening_report,_report_hash
    report=build_screening_report(asyncio.run(pipeline().analyze_neo('A')))
    report['analysis_metadata'][field]=value;report['report_id']=_report_hash(report)
    with pytest.raises(ValueError):verify_screening_report(report)


@pytest.mark.parametrize('raw',[None,[], 'not an object'])
def test_invalid_report_shape_is_controlled_not_attribute_error(raw):
    from aneos_core.analysis.reporting import verify_screening_report
    with pytest.raises(ValueError):verify_screening_report(raw)
