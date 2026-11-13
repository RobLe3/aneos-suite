"""Actual HTTP/SQL/lifecycle controls, synthetic non-sensitive objects/accounts."""
import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tests.test_nongrav_evidence import payload as nongrav_payload

URL='/api/v1/analysis/analyze/batch'
KEY_A='synthetic-batch-owner-a-'+'a'*40
KEY_B='synthetic-batch-owner-b-'+'b'*40
KEY_V='synthetic-batch-viewer-'+'v'*40
PROFILE='1'*64


def headers(key=KEY_A,**extra):
    return {'X-API-Key':key,**extra}


def test_actual_batch_route_no_longer_accepts_anonymous_background_work(tmp_path,monkeypatch):
    from aneos_api.endpoints import analysis
    from aneos_api.network_runtime import NetworkRuntime
    engine=create_engine(f'sqlite:///{tmp_path}/denial.db',connect_args={'check_same_thread':False})
    runtime=NetworkRuntime(sessionmaker(bind=engine),enabled=False)
    runtime.store.initialize_schema()
    application=FastAPI();application.include_router(analysis.router,prefix='/api/v1/analysis')
    application.state.network_runtime=runtime
    background=AsyncMock();monkeypatch.setattr(analysis,'_run_batch_detection',background)
    try:
        with TestClient(application) as client:
            response=client.post(URL,json={'designations':['synthetic-marker']})
            assert response.status_code==401,response.text
            background.assert_not_awaited()
    finally:runtime.close();engine.dispose()


class Source:
    calls=[]
    def fetch_neo_data(self,d,**kwargs):
        from aneos_core.data.models import NEOData,OrbitalElements
        from aneos_core.utils.errors import DataSourceUnavailableError
        self.calls.append((d,kwargs.get('force_refresh',False)))
        if d=='offline':raise DataSourceUnavailableError('synthetic-private-provider-detail')
        if d=='missing':return NEOData(d)
        return NEOData('other-object' if d=='wrong' else d,orbital_elements=OrbitalElements(
            semi_major_axis=1.23,eccentricity=.2,inclination=180. if d=='A' else 1.49))


def pipeline(source):
    from aneos_core.analysis.pipeline import AnalysisPipeline
    p=AnalysisPipeline();p.data_fetcher=source
    return p


@pytest.fixture
def api(tmp_path,monkeypatch):
    import functools,socket
    from aneos_api import app,database,batch_routes
    from aneos_api.network_runtime import NetworkRuntime
    from aneos_api.network_jobs import NetworkJobHandler
    from aneos_api.batch_jobs import BatchJobHandler,build_request
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Live transport forbidden')))
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Live DNS forbidden')))
    monkeypatch.setenv('ANEOS_DATABASE_URL',f'sqlite:///{tmp_path}/api.db')
    engine=create_engine(f'sqlite:///{tmp_path}/api.db',connect_args={'check_same_thread':False,'timeout':5})
    sessions=sessionmaker(bind=engine);database.Base.metadata.create_all(engine)
    with sessions() as s:
        for name,key,role in [('a',KEY_A,'analyst'),('b',KEY_B,'analyst'),('v',KEY_V,'viewer')]:
            s.add(database.User(user_id='batch-test-'+name,username='batch-test-'+name,role=role,api_keys=[key],is_active=True))
        s.commit()
    monkeypatch.setattr(database,'init_database',lambda:True)
    monkeypatch.setattr(database,'SessionLocal',sessions)
    services=Mock(initialize_services=AsyncMock(),shutdown_services=AsyncMock(),
        get_health_status=Mock(return_value={'status':'healthy','services':{},'version':'test'}))
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    # Freeze serial work in this HTTP fault-test host; production bounded parallel
    # execution remains covered by the registered-handler tests.
    monkeypatch.setattr(batch_routes,'build_request',functools.partial(build_request,
        execution={'max_workers':1,'timeout_seconds':30,'parallel':False,'queue_size':1}))
    source=Source();source.calls=[]
    runtime=NetworkRuntime(sessions,capacity=4,lease_seconds=2,shutdown_seconds=3,
        handler=NetworkJobHandler(source_factory=lambda:source,source_profile=lambda:PROFILE),
        batch_enabled=True,batch_handler=BatchJobHandler(pipeline_factory=lambda:pipeline(source),source_profile=lambda:PROFILE))
    application=app.create_app();application.state.network_runtime_factory=lambda:runtime
    with TestClient(application) as client:
        yield client,runtime,sessions,application,source
    engine.dispose()


def submit(client,names=('A',),**options):
    response=client.post(URL,json={'designations':list(names),**options},headers=headers())
    assert response.status_code==200,response.text
    return response.json()


def wait(client,job,key=KEY_A):
    until=time.monotonic()+15
    while time.monotonic()<until:
        response=client.get(job['status_url'],headers=headers(key))
        if response.status_code==429:
            time.sleep(float(response.headers.get('Retry-After',1)));continue
        assert response.status_code==200,response.text
        row=response.json()
        if row['execution_status'] in ('completed','cancelled','failed'):return row
        time.sleep(.1)
    raise AssertionError('Persistent batch did not terminate')


@pytest.mark.parametrize('credential',[{}, {'X-API-Key':'admin-key-not-configured'},
    {'Authorization':'Bearer mock_admin_token'},{'X-API-Key':'unknown-key'}])
def test_batch_admission_requires_real_fresh_credentials(api,credential):
    client,_,_,_,source=api
    assert client.post(URL,json={'designations':['A']},headers=credential).status_code==401
    assert source.calls==[]


def test_actual_owned_batch_has_every_occurrence_and_explicit_unavailable(api):
    client,runtime,_,_,source=api
    job=submit(client,['A','B','A','missing','offline','wrong'],force_refresh=True)
    assert job['execution_storage']=='sql_durable' and job['batch_id']==job['job_id']
    row=wait(client,job)
    assert row['total']==6 and row['completed']==4 and row['failed']==2 and row['pending']==0
    assert [r['designation'] for r in row['results']]==['A','B','A','missing','offline','wrong']
    assert row['request_progress']=={'ready_exact_labels':5,'reported_exact_labels':5,'required_exact_labels':5}
    assert source.calls==[(d,True) for d in ['A','B','missing','offline','wrong']]
    assert row['results'][0]['candidate'] is True and row['results'][1]['candidate'] is False
    assert row['results'][0]['report_id']==row['results'][2]['report_id']
    assert row['results'][3]['execution_status']=='inconclusive'
    for result in row['results'][3:]:
        assert result['screening_score'] is None and result['classification']=='INCONCLUSIVE'
    assert row['results'][4]['execution_status']=='source_unavailable'
    assert row['results'][5]['execution_status']=='invalid_evidence'
    assert 'synthetic-private-provider-detail' not in json.dumps(row)
    assert row['scientific_report'] is None and row['scientific_report_available'] is True
    assert all('raw_neo_data' not in result for result in row['results'])
    assert all(not r['five_sigma_claim_permitted'] for r in row['results'])
    assert row['started_at'].endswith('+00:00') and row['finished_at'].endswith('+00:00')
    assert all('observation' in role for role in row['timestamp_roles'].values())
    assert 'lease_token' not in row and 'owner_scope' not in row
    assert runtime.batch_ready and runtime.ready


def test_full_raw_report_replays_identical_canonical_assessment(api):
    from aneos_core.analysis.batch_reporting import verify_batch_report
    from aneos_core.analysis.screening import ScreeningAnalysisService
    client,*_=api
    row=wait(client,submit(client,['A'],include_raw_data=True))
    report=verify_batch_report(row['scientific_report'])
    result=row['results'][0]
    expected=ScreeningAnalysisService().analyze_neo(Source().fetch_neo_data('A')).to_payload()
    assert result['screening_assessment']==expected
    assert report['requests'][0]['report']['screening_assessment']==expected
    assert result['raw_neo_data']==report['requests'][0]['report']['neo_data']
    assert result['sigma_confidence'] is None and result['artificial_probability'] is None


def test_owned_reads_cancellation_and_global_cache_do_not_leak(api):
    client,*_=api
    job=submit(client,['unique-private-batch-marker']);wait(client,job)
    assert client.get(job['status_url'],headers=headers(KEY_B)).status_code==404
    assert client.delete('/api/v1/analysis/batch/'+job['batch_id'],headers=headers(KEY_B)).status_code==404
    assert client.get('/api/v1/analysis/results/unique-private-batch-marker',headers=headers(KEY_B)).status_code==404
    assert client.get(job['status_url'],headers=headers()).status_code==200


def test_viewer_cannot_admit_or_cancel_and_admin_not_implicit_owner(api,monkeypatch):
    client,*_=api
    assert client.post(URL,json={'designations':['A']},headers=headers(KEY_V)).status_code==403
    assert client.delete('/api/v1/analysis/batch/unknown',headers=headers(KEY_V)).status_code==403
    job=submit(client);wait(client,job)
    admin='synthetic-batch-admin-'+'z'*40;monkeypatch.setenv('ANEOS_ADMIN_API_KEY',admin)
    assert client.get(job['status_url'],headers=headers(admin)).status_code==404
    assert client.post(URL,json={'designations':['B']},headers=headers(admin)).status_code==200


@pytest.mark.parametrize('body',[{'designations':['A'],'owner_scope':'other'},
    {'designations':['A'],'force_refresh':'true'},{'designations':['A'],'include_raw_data':1},
    {'designations':['A'],'progress_webhook':'https://example.invalid/hook'},
    {'designations':['A\nB']},{'designations':['#A']},{'designations':['A\x7f']},
    {'designations':[]},{'designations':['A']*101}])
def test_invalid_or_unimplemented_batch_intent_rejected_before_provider(api,body):
    client,_,_,_,source=api
    assert client.post(URL,json=body,headers=headers()).status_code==422
    assert source.calls==[]


def test_idempotency_retains_frozen_config_and_conflicting_intent_is_rejected(api,monkeypatch):
    from aneos_api import batch_routes
    client,runtime,_,_,source=api
    h=headers(**{'Idempotency-Key':'retained-intent'})
    first=client.post(URL,json={'designations':['A']},headers=h)
    assert first.status_code==200
    saved=wait(client,first.json())
    monkeypatch.setattr(batch_routes,'build_request',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No new frozen clock/config on replay')))
    again=client.post(URL,json={'designations':['A']},headers=h)
    assert again.status_code==200 and again.json()['batch_id']==first.json()['batch_id']
    assert source.calls==[('A',False)]
    for change in ({'designations':['B']},{'designations':['A'],'force_refresh':True},
                   {'designations':['A'],'include_raw_data':True}):
        assert client.post(URL,json=change,headers=h).status_code==409
    assert saved['result_sha256']==wait(client,again.json())['result_sha256']


@pytest.mark.parametrize('key',['',' space','bad\tkey','x'*129])
def test_bad_idempotency_header_rejected(api,key):
    client,*_=api
    assert client.post(URL,json={'designations':['A']},headers=headers(**{'Idempotency-Key':key})).status_code==422


def test_duplicate_idempotency_or_credentials_not_silently_selected(api):
    client,*_=api
    base=[('X-API-Key',KEY_A)]
    assert client.post(URL,json={'designations':['A']},headers=base+[('Idempotency-Key','a'),('Idempotency-Key','b')]).status_code==422
    assert client.post(URL,json={'designations':['A']},headers=base+[('X-API-Key',KEY_B)]).status_code==401
    assert client.post(URL,json={'designations':['A']},headers=headers(Authorization='Bearer conflicting')).status_code==401


def test_batch_tokens_do_not_expand_network_token_scope_and_revocation_is_fresh(api,monkeypatch):
    from aneos_api import auth,database
    client,_,sessions,*_=api
    monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-batch-signing-key-'+'s'*40)
    batch_token=client.post('/api/v1/auth/token?audience=batch-jobs-v1',headers=headers())
    assert batch_token.status_code==200,batch_token.text
    h={'Authorization':'Bearer '+batch_token.json()['access_token']}
    job=client.post(URL,json={'designations':['A']},headers=h)
    assert job.status_code==200
    assert client.get(job.json()['status_url'],headers=h).status_code==200
    assert client.post('/api/v1/analysis/analyze/network',json={'designations':['A']},headers=h).status_code==401
    network_token=client.post('/api/v1/auth/token?audience=network-jobs-v1',headers=headers()).json()['access_token']
    assert client.get(job.json()['status_url'],headers={'Authorization':'Bearer '+network_token}).status_code==401
    legacy=auth.create_access_token('batch-test-a','admin')
    assert client.get(job.json()['status_url'],headers={'Authorization':'Bearer '+legacy}).status_code==401
    with sessions() as s:
        user=s.query(database.User).filter_by(user_id='batch-test-a').one();user.api_keys=[];s.commit()
    assert client.get(job.json()['status_url'],headers=h).status_code==401
    assert client.get(job.json()['status_url'],headers=headers()).status_code==401


def test_batch_bearer_permissions_follow_current_role(api,monkeypatch):
    from aneos_api import auth,database
    client,_,sessions,*_=api
    monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-batch-signing-key-'+'s'*40)
    token=client.post('/api/v1/auth/token?audience=batch-jobs-v1',headers=headers()).json()['access_token']
    job=submit(client);wait(client,job)
    with sessions() as s:
        user=s.query(database.User).filter_by(user_id='batch-test-a').one();user.role='viewer';s.commit()
    h={'Authorization':'Bearer '+token}
    assert client.get(job['status_url'],headers=h).status_code==200
    assert client.post(URL,json={'designations':['B']},headers=h).status_code==403
    assert client.delete('/api/v1/analysis/batch/'+job['batch_id'],headers=h).status_code==403


def test_partial_cancelled_progress_preserves_ready_object_and_fences_later_commit(api):
    client,runtime,_,_,source=api;entered=threading.Event();release=threading.Event()
    original=source.fetch_neo_data
    def fetch(d,**kwargs):
        if d=='B':entered.set();assert release.wait(5)
        return original(d,**kwargs)
    source.fetch_neo_data=fetch
    job=submit(client,['A','B','A'])
    try:
        assert entered.wait(3)
        partial=client.get(job['status_url'],headers=headers()).json()
        assert partial['completed']==2 and partial['pending']==1
        assert [r['execution_status'] for r in partial['results']]==['completed','not_assessed','completed']
        assert partial['results'][1]['screening_score'] is None
        cancel=client.delete('/api/v1/analysis/batch/'+job['batch_id'],headers=headers())
        assert cancel.status_code==200 and cancel.json()['execution_status']=='cancelled'
    finally:release.set()
    row=wait(client,job)
    assert row['checkpoint_count']==1 and row['result_sha256'] is None
    assert row['scientific_report'] is None and row['completed']==2 and row['pending']==1
    assert row['results'][0]['report_id']==partial['results'][0]['report_id']


def test_shared_capacity_across_registered_job_kinds_and_idempotent_reuse(api):
    client,runtime,_,_,source=api;entered=threading.Event();release=threading.Event()
    original=source.fetch_neo_data
    def fetch(d,**kwargs):entered.set();assert release.wait(5);return original(d,**kwargs)
    source.fetch_neo_data=fetch
    h=headers(**{'Idempotency-Key':'capacity-request'})
    first=client.post(URL,json={'designations':['A']},headers=h)
    assert first.status_code==200
    try:
        assert entered.wait(3)
        for d in ('B','C'):
            assert client.post(URL,json={'designations':[d]},headers=headers()).status_code==200
        network=client.post('/api/v1/analysis/analyze/network',json={'designations':['D'],'harmonics':False},headers=headers())
        assert network.status_code==200,network.text
        full=client.post(URL,json={'designations':['E']},headers=headers())
        assert full.status_code==429 and full.headers.get('Retry-After')=='5'
        assert client.post(URL,json={'designations':['A']},headers=h).json()['batch_id']==first.json()['batch_id']
    finally:release.set()


def test_shutdown_retains_read_and_idempotency_but_denies_new_admission(api):
    client,runtime,*_=api
    h=headers(**{'Idempotency-Key':'stopped-replay'})
    first=client.post(URL,json={'designations':['A']},headers=h);job=first.json();saved=wait(client,job)
    runtime.close()
    assert not runtime.batch_ready
    assert client.get(job['status_url'],headers=headers()).json()['result_sha256']==saved['result_sha256']
    assert client.post(URL,json={'designations':['B']},headers=headers()).status_code==503
    assert client.post(URL,json={'designations':['A']},headers=h).json()['batch_id']==job['batch_id']
    assert client.get('/health').json()['checks']['batch_worker']['status']=='degraded'


def test_acknowledgement_loss_after_commit_recovered_by_same_client_key(api,monkeypatch):
    from aneos_api.job_store import JobStorageUnavailable
    client,runtime,_,_,source=api
    original=runtime.store.get;failed=[False]
    def get(*a,**kwargs):
        if not failed[0]:failed[0]=True;raise JobStorageUnavailable('synthetic private DB URL')
        return original(*a,**kwargs)
    monkeypatch.setattr(runtime.store,'get',get)
    h=headers(**{'Idempotency-Key':'ack-loss'})
    first=client.post(URL,json={'designations':['A']},headers=h)
    assert first.status_code==503 and 'private DB URL' not in first.text
    recovered=client.post(URL,json={'designations':['A']},headers=h)
    assert recovered.status_code==200,recovered.text
    assert wait(client,recovered.json())['execution_status']=='completed'
    assert source.calls==[('A',False)]


def test_modified_checkpoint_is_not_a_successful_read(api):
    from aneos_api.job_store import JobCheckpoint
    client,_,sessions,*_=api
    job=submit(client);wait(client,job)
    with sessions() as s:
        row=s.query(JobCheckpoint).filter_by(job_id=job['job_id'],key='object:0:attempt:1').one()
        row.payload_json='{}';s.commit()
    response=client.get(job['status_url'],headers=headers())
    assert response.status_code==503 and response.json()['error']=='Persistent batch execution unavailable or integrity unconfirmed'


def test_unknown_wrong_kind_and_legacy_process_only_ids_not_recovered_as_success(api):
    client,*_=api
    for token in ('unknown','batch_old_process_id','job_'+'0'*32):
        assert client.get('/api/v1/analysis/batch/'+token+'/status',headers=headers()).status_code==404
    network=client.post('/api/v1/analysis/analyze/network',json={'designations':['A'],'harmonics':False},headers=headers()).json()
    assert client.get('/api/v1/analysis/batch/'+network['job_id']+'/status',headers=headers()).status_code==404


def test_disabled_host_still_reads_completed_owned_results(api):
    from aneos_api import app
    from aneos_api.network_runtime import NetworkRuntime
    client,runtime,sessions,*_=api
    job=submit(client);saved=wait(client,job);runtime.close()
    disabled=NetworkRuntime(sessions,enabled=False,batch_enabled=False,capacity=4)
    application=app.create_app();application.state.network_runtime_factory=lambda:disabled
    with TestClient(application) as reader:
        assert not disabled.batch_ready and disabled._thread is None
        assert reader.get(job['status_url'],headers=headers()).json()['result_sha256']==saved['result_sha256']
        assert reader.post(URL,json={'designations':['B']},headers=headers()).status_code==503


def test_network_disabled_batch_enabled_positive_host(api):
    from aneos_api import app
    from aneos_api.network_runtime import NetworkRuntime
    client,runtime,sessions,_,_=api;runtime.close()
    batch=NetworkRuntime(sessions,enabled=False,batch_enabled=True,capacity=4,
        batch_handler=runtime.batch_worker.handler)
    application=app.create_app();application.state.network_runtime_factory=lambda:batch
    with TestClient(application) as c:
        assert not batch.ready and batch.batch_ready
        assert wait(c,submit(c))['execution_status']=='completed'
        assert c.post('/api/v1/analysis/analyze/network',json={'designations':['B']},headers=headers()).status_code==503


def test_concurrent_same_key_admits_one_owned_request(api):
    from concurrent.futures import ThreadPoolExecutor
    from aneos_api.job_store import DurableJob
    from aneos_api.batch_jobs import KIND
    client,runtime,sessions,_,source=api
    def send(_):
        return client.post(URL,json={'designations':['A']},headers=headers(**{'Idempotency-Key':'concurrent-synthetic'}))
    with ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(send,range(4)))
    assert all(r.status_code==200 for r in responses),[r.text for r in responses]
    assert len({r.json()['job_id'] for r in responses})==1
    assert wait(client,responses[0].json())['execution_status']=='completed'
    with sessions() as s:assert s.query(DurableJob).filter_by(kind=KIND).count()==1
    assert runtime.store.get('db-user:batch-test-a',responses[0].json()['job_id']) is not None
    assert source.calls==[('A',False)]


def test_unacknowledged_storage_failures_and_cancellation_do_not_leak_details(api,monkeypatch):
    from aneos_api.job_store import JobStorageUnavailable
    client,runtime,*_=api
    job=submit(client);wait(client,job)
    def fail(*a,**k):raise JobStorageUnavailable('synthetic private storage connection')
    with monkeypatch.context() as patch:
        patch.setattr(runtime.store,'submit',fail)
        response=client.post(URL,json={'designations':['B']},headers=headers())
        assert response.status_code==503 and 'private storage' not in response.text
    with monkeypatch.context() as patch:
        patch.setattr(runtime.store,'cancel',fail)
        response=client.delete('/api/v1/analysis/batch/'+job['job_id'],headers=headers())
        assert response.status_code==503 and 'private storage' not in response.text


def test_actual_dead_shared_host_stops_both_admissions_but_preserves_verified_read(api,monkeypatch):
    from aneos_api.job_store import JobStorageUnavailable
    client,runtime,*_=api
    job=submit(client);saved=wait(client,job)
    def crash():raise JobStorageUnavailable('synthetic host fault')
    monkeypatch.setattr(runtime.batch_worker,'run_once',crash)
    runtime._thread.join(timeout=3)
    assert not runtime._thread.is_alive() and runtime.error=='EXECUTION_WORKER_STOPPED'
    assert not runtime.ready and not runtime.batch_ready
    assert client.get(job['status_url'],headers=headers()).json()['result_sha256']==saved['result_sha256']
    assert client.post(URL,json={'designations':['B']},headers=headers()).status_code==503
    assert client.post('/api/v1/analysis/analyze/network',json={'designations':['B']},headers=headers()).status_code==503


def test_scoped_token_issuance_rejects_duplicate_or_mixed_credentials(api,monkeypatch):
    from aneos_api import auth
    client,*_=api;monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-batch-signing-key-'+'s'*40)
    for url in ('/api/v1/auth/token?audience=batch-jobs-v1','/api/v1/auth/token?audience=network-jobs-v1'):
        assert client.post(url,headers=[('X-API-Key',KEY_A),('X-API-Key',KEY_B)]).status_code==401
        assert client.post(url,headers=headers(Authorization='Bearer other')).status_code==401


def test_same_client_key_is_scoped_to_owner_not_global(api):
    client,*_=api
    options={'Idempotency-Key':'same-key-independent-owners'}
    a=client.post(URL,json={'designations':['A']},headers=headers(**options))
    b=client.post(URL,json={'designations':['B']},headers=headers(KEY_B,**options))
    assert a.status_code==b.status_code==200 and a.json()['job_id']!=b.json()['job_id']
    assert wait(client,a.json())['results'][0]['designation']=='A'
    assert wait(client,b.json(),KEY_B)['results'][0]['designation']=='B'
    assert client.get(a.json()['status_url'],headers=headers(KEY_B)).status_code==404


def test_round_robin_host_executes_both_registered_handlers(api):
    client,runtime,_,_,source=api;entered=threading.Event();release=threading.Event()
    original=source.fetch_neo_data
    def fetch(d,**kwargs):
        if d=='A':entered.set();assert release.wait(5)
        return original(d,**kwargs)
    source.fetch_neo_data=fetch
    first=submit(client,['A'])
    try:
        assert entered.wait(3)
        second=submit(client,['C'])
        network=client.post('/api/v1/analysis/analyze/network',json={'designations':['B'],'harmonics':False},headers=headers())
        assert network.status_code==200,network.text
    finally:release.set()
    assert wait(client,first)['execution_status']=='completed'
    assert wait(client,second)['execution_status']=='completed'
    status=client.get(network.json()['status_url'],headers=headers())
    assert status.status_code==200 and status.json()['execution_status']=='completed'
    assert [d for d,_ in source.calls]==['A','B','C']


def test_actual_api_process_kill_recovers_only_missing_object_and_owner(api,tmp_path):
    import os,subprocess,sys
    from aneos_api import app
    from aneos_api.network_runtime import NetworkRuntime
    from aneos_api.batch_jobs import BatchJobHandler
    client,runtime,sessions,_,_=api;runtime.close()
    db_url=str(runtime.store.engine.url);submission=tmp_path/'http-ack.json';blocked=tmp_path/'second-entered'
    code='''
import sys,time,json,functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from aneos_api import app,database,batch_routes
from aneos_api.network_runtime import NetworkRuntime
from aneos_api.batch_jobs import BatchJobHandler,build_request
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.data.models import NEOData,OrbitalElements
class Source:
 def fetch_neo_data(self,d,**kwargs):
  if d=='B':
   Path(sys.argv[3]).write_text('A committed before B acquisition')
   time.sleep(60)
  return NEOData(d,orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=180.))
def pipeline():
 p=AnalysisPipeline();p.data_fetcher=Source();return p
engine=create_engine(sys.argv[1],connect_args={'check_same_thread':False,'timeout':5});sessions=sessionmaker(bind=engine)
database.init_database=lambda:True;database.SessionLocal=sessions
app.get_aneos_app=lambda:SimpleNamespace(initialize_services=AsyncMock(),shutdown_services=AsyncMock())
batch_routes.build_request=functools.partial(build_request,execution={'max_workers':1,'timeout_seconds':30,'parallel':False,'queue_size':1})
a=app.create_app();a.state.network_runtime_factory=lambda:NetworkRuntime(sessions,enabled=False,batch_enabled=True,
 capacity=4,lease_seconds=2,batch_handler=BatchJobHandler(pipeline_factory=pipeline,source_profile=lambda:'1'*64))
with TestClient(a) as client:
 r=client.post('/api/v1/analysis/analyze/batch',json={'designations':['A','B','A'],'include_raw_data':True},
  headers={'X-API-Key':'synthetic-batch-owner-a-'+'a'*40,'Idempotency-Key':'killed-http-job'})
 assert r.status_code==200,r.text
 Path(sys.argv[2]).write_text(json.dumps(r.json()))
 time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',code,db_url,str(submission),str(blocked)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+20
        while not (submission.exists() and blocked.exists()) and child.poll() is None and time.monotonic()<end:time.sleep(.03)
        assert submission.exists() and blocked.exists(),child.communicate(timeout=1) if child.poll() is not None else 'HTTP checkpoint/ack marker absent'
    finally:
        if child.poll() is None:child.kill()
        child.communicate(timeout=5)
    job=json.loads(submission.read_text())
    partial=client.get(job['status_url'],headers=headers())
    assert partial.status_code==200 and partial.json()['checkpoint_count']==1
    first_report=partial.json()['results'][0]['report_id']
    end=time.monotonic()+5
    while runtime.store.get('db-user:batch-test-a',job['job_id'])['lease_until']>time.time() and time.monotonic()<end:time.sleep(.03)
    calls=[]
    class Recovery(Source):
        def fetch_neo_data(self,d,**kwargs):
            assert d=='B','Committed A must not be acquired again'
            calls.append(d);return super().fetch_neo_data(d,**kwargs)
    replacement=NetworkRuntime(sessions,enabled=False,batch_enabled=True,capacity=4,lease_seconds=2,
        batch_handler=BatchJobHandler(pipeline_factory=lambda:pipeline(Recovery()),source_profile=lambda:PROFILE))
    application=app.create_app();application.state.network_runtime_factory=lambda:replacement
    with TestClient(application) as recovered:
        saved=wait(recovered,job)
        assert saved['execution_status']=='completed' and saved['attempts']==2 and calls==['B']
        assert saved['results'][0]['report_id']==first_report==saved['results'][2]['report_id']
        assert saved['total']==saved['completed']==3
        assert recovered.get(job['status_url'],headers=headers(KEY_B)).status_code==404
    # Third host: disabled execution still verifies the exact committed result.
    reader=NetworkRuntime(sessions,enabled=False,batch_enabled=False,capacity=4)
    fresh=app.create_app();fresh.state.network_runtime_factory=lambda:reader
    with TestClient(fresh) as third:
        assert third.get(job['status_url'],headers=headers()).json()['result_sha256']==saved['result_sha256']
    assert calls==['B']


@pytest.mark.parametrize('options',[{'include_raw_data':True}, {'schema_version':'unknown','include_raw_data':True},
    {'schema_version':'batch-http-response-v1','include_raw_data':'true'},
    {'schema_version':'batch-http-response-v1','include_raw_data':True,'owner':'other'}])
def test_http_options_invalid_inside_frozen_request(options):
    from aneos_api.batch_jobs import build_request
    from aneos_core.analysis.batch_reporting import parse_input
    with pytest.raises(ValueError):build_request(parse_input(b'A\n','input.txt'),response_options=options)


@pytest.mark.asyncio
async def test_legacy_process_callbacks_are_explicitly_withdrawn():
    from aneos_api.endpoints import analysis
    for function in (analysis.analyze_batch,analysis.get_batch_status,analysis._run_batch_detection):
        with pytest.raises(NotImplementedError):await function()
    assert not hasattr(analysis,'_batch_store')


def test_api_source_profile_drift_abstains_without_polling(api):
    from aneos_api import app
    from aneos_api.batch_jobs import build_request,BatchJobHandler,KIND
    from aneos_core.analysis.batch_reporting import parse_input
    from aneos_api.network_runtime import NetworkRuntime
    client,runtime,sessions,_,source=api;runtime.close()
    request=build_request(parse_input(b'A\n','older-source-profile.txt'),source_profile_id=PROFILE,
        response_options={'schema_version':'batch-http-response-v1','include_raw_data':False})
    job=runtime.store.submit(owner_scope='db-user:batch-test-a',kind=KIND,request=request,max_attempts=request['max_attempts'])
    ack={'status_url':f"/api/v1/analysis/batch/{job['job_id']}/status"}
    changed=NetworkRuntime(sessions,enabled=False,batch_enabled=True,capacity=4,
        batch_handler=BatchJobHandler(pipeline_factory=lambda:pipeline(source),source_profile=lambda:'2'*64))
    application=app.create_app();application.state.network_runtime_factory=lambda:changed
    with TestClient(application) as c:
        row=wait(c,ack)
        assert row['execution_status']=='failed' and row['error']=='SOURCE_CONFIGURATION_CHANGED'
        assert row['pending']==1 and row['results'][0]['screening_assessment'] is None
        assert row['result_sha256'] is None and row['results'][0]['screening_score'] is None
    assert source.calls==[]


def test_batch_request_openapi_declares_auth_and_unsupported_webhook(api):
    client,*_=api
    schema=client.get('/openapi.json').json()
    operation=schema['paths'][URL]['post']
    assert operation['security']
    assert any(p['name']=='Idempotency-Key' for p in operation['parameters'])
    model=schema['components']['schemas']['BatchAnalysisRequest']
    assert model['additionalProperties'] is False
    assert model['properties']['progress_webhook']['type']=='null'
    assert schema['paths']['/api/v1/analysis/batch/{batch_id}']['delete']['security']


def test_canonical_csv_serializer_refuses_lossy_unvalidated_intent():
    from types import SimpleNamespace
    from aneos_api.batch_routes import _source
    with pytest.raises(ValueError,match='round-trip'):_source(SimpleNamespace(designations=[' A ']))


def test_shared_host_default_flags_validate_before_startup(api,monkeypatch):
    from aneos_api.network_runtime import default_runtime,NetworkRuntime
    _,_,sessions,*_=api
    monkeypatch.setenv('ANEOS_NETWORK_WORKER_ENABLED','true')
    monkeypatch.setenv('ANEOS_BATCH_WORKER_ENABLED','true')
    runtime=default_runtime()
    assert runtime.enabled and runtime.batch_enabled and runtime._thread is None
    runtime.close()
    for name in ('ANEOS_NETWORK_WORKER_ENABLED','ANEOS_BATCH_WORKER_ENABLED'):
        with monkeypatch.context() as patch:
            patch.setenv(name,'not-a-boolean')
            with pytest.raises(ValueError):default_runtime()
    with pytest.raises(ValueError):NetworkRuntime(sessions,batch_enabled='true')


def test_source_shaped_accelerations_and_correlations_survive_actual_rest_batch(api,nongrav_payload):
    from tests.test_nongrav_evidence import parsed_ng
    from aneos_core.data.models import NEOData,OrbitalElements
    from aneos_core.analysis.batch_reporting import verify_batch_report
    from aneos_core.analysis.screening import ScreeningAnalysisService
    client,_,_,_,source=api
    ng,_=parsed_ng(nongrav_payload)
    neo=NEOData('SYNTHETIC',orbital_elements=OrbitalElements(semi_major_axis=1.2,eccentricity=.2,inclination=1.5),nongrav=ng)
    source.fetch_neo_data=Mock(return_value=neo)
    row=wait(client,submit(client,['SYNTHETIC'],include_raw_data=True))
    result=row['results'][0]
    expected=ScreeningAnalysisService().analyze_neo(neo).to_payload()
    assert result['screening_assessment']==expected
    assert result['non_gravitational_evidence']==ng.evidence_report()
    assert result['non_gravitational_evidence']['components']['A3']['native']['kind']=='CON'
    restored=NEOData.from_dict(result['raw_neo_data'])
    assert restored.nongrav.to_dict()==ng.to_dict()
    assert restored.nongrav.covariance.correlation('A1','A3')==pytest.approx(.4)
    assert result['non_gravitational_evidence']['components']['A3']['native']['value']==1e-12
    assert all(not e['analyzed'] for e in result['evidence_sources'] if e['type'] in ('propulsion_signatures','course_corrections'))
    verify_batch_report(row['scientific_report'])
    # Readback, verification and rendering do not poll the source again.
    assert client.get(f"/api/v1/analysis/batch/{row['job_id']}/status",headers=headers()).status_code==200
    source.fetch_neo_data.assert_called_once_with('SYNTHETIC',force_refresh=False)


@pytest.mark.parametrize('value',[float('inf'),1e308])
def test_corrupted_execution_timestamp_not_formatted_as_valid_chronology(api,value):
    from aneos_api.job_store import DurableJob
    client,_,sessions,*_=api
    job=submit(client);wait(client,job)
    with sessions() as s:
        row=s.query(DurableJob).filter_by(job_id=job['job_id']).one();row.updated_at=value;s.commit()
    response=client.get(job['status_url'],headers=headers())
    assert response.status_code==503 and 'Persistent batch execution' in response.text


def test_blocked_batch_shutdown_budget_is_explicit_not_false_health(api):
    client,runtime,_,_,source=api;entered=threading.Event();release=threading.Event()
    original=source.fetch_neo_data
    def fetch(d,**kwargs):entered.set();assert release.wait(5);return original(d,**kwargs)
    source.fetch_neo_data=fetch
    job=submit(client)
    try:
        assert entered.wait(3)
        runtime.shutdown_seconds=.01
        with pytest.raises(RuntimeError,match='shutdown unconfirmed'):runtime.close()
        assert runtime.error=='EXECUTION_WORKER_SHUTDOWN_UNCONFIRMED'
        assert not runtime.batch_ready and not runtime.ready
        assert client.post(URL,json={'designations':['B']},headers=headers()).status_code==503
    finally:
        release.set();runtime._thread.join(timeout=3);runtime.shutdown_seconds=3
    assert not runtime._thread.is_alive()
    state=client.get(job['status_url'],headers=headers()).json()
    assert state['execution_status']=='retry_wait' and state['pending']==1
    assert state['result_sha256'] is None and state['results'][0]['report_id'] is None
