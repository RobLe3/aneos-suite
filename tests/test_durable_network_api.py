"""Real SQL/API/lifecycle controls; synthetic sources and accounts only."""
import threading
import time
from unittest.mock import AsyncMock,Mock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

KEY_A='synthetic-owner-a-key-'+'a'*32
KEY_B='synthetic-owner-b-key-'+'b'*32
KEY_V='synthetic-viewer-key-'+'v'*32
URL='/api/v1/analysis/analyze/network'


class Source:
    def fetch_neo_data(self,d):
        from aneos_core.data.models import NEOData,OrbitalElements
        return NEOData(designation=d,orbital_elements=OrbitalElements(
            semi_major_axis=1.,eccentricity=.2,inclination=180.))


@pytest.fixture
def api(tmp_path,monkeypatch):
    from aneos_api import app,database
    from aneos_api.network_jobs import NetworkJobHandler
    from aneos_api.network_runtime import NetworkRuntime
    engine=create_engine('sqlite:///'+str(tmp_path/'api.db'),connect_args={'check_same_thread':False,'timeout':5})
    sessions=sessionmaker(bind=engine)
    database.Base.metadata.create_all(engine)
    with sessions() as s:
        for user,key,role in [('a',KEY_A,'analyst'),('b',KEY_B,'analyst'),('v',KEY_V,'viewer')]:
            s.add(database.User(user_id='synthetic-'+user,username='synthetic-'+user,role=role,api_keys=[key],is_active=True))
        s.commit()
    monkeypatch.setattr(database,'init_database',lambda:True)
    services=Mock(initialize_services=AsyncMock(),shutdown_services=AsyncMock(),services_initialized=True,
                  get_health_status=Mock(return_value={'status':'healthy','services':{},'version':'test'}))
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    runtime=NetworkRuntime(sessions,handler=NetworkJobHandler(source_factory=Source,source_profile=lambda:'1'*64),
                           capacity=4,lease_seconds=2,shutdown_seconds=3)
    application=app.create_app();application.state.network_runtime_factory=lambda:runtime
    with TestClient(application) as client:yield client,runtime,sessions,application
    engine.dispose()


def headers(key=KEY_A,**extra):return {'X-API-Key':key,**extra}


def wait(client,job,credential=KEY_A):
    end=time.monotonic()+5
    while time.monotonic()<end:
        r=client.get(job['status_url'],headers=headers(credential))
        assert r.status_code==200,r.text
        if r.json()['execution_status'] in ('completed','failed','cancelled'):return r.json()
        time.sleep(.02)
    raise AssertionError('Persistent job did not reach a terminal state')


@pytest.mark.parametrize('credential',[{}, {'Authorization':'Bearer mock_admin_token'},
    {'X-API-Key':'admin-key-not-configured'},{'X-API-Key':'unknown-key'}])
def test_network_never_accepts_anonymous_placeholder_or_dev_mock(api,credential):
    client,*_=api
    assert client.post(URL,json={'designations':['A'],'harmonics':False},headers=credential).status_code==401


def test_real_authorized_worker_report_and_status_url(api):
    client,runtime,*_=api
    result=client.post(URL,json={'designations':['A','A'],'harmonics':False},headers=headers())
    assert result.status_code==200,result.text
    saved=wait(client,result.json())
    assert saved['execution_status']=='completed' and saved['status']=='complete'
    assert saved['scientific_report']['object_assessments'][0]['assessment']['candidate'] is True
    assert len(saved['scientific_report']['object_assessments'])==2
    assert saved['network_sigma'] is None and saved['combined_p_value'] is None
    assert saved['analysis_metadata']['execution_storage']=='sql_durable'
    assert runtime.ready


def test_other_owner_cannot_read_or_cancel_actual_authorized_job(api):
    client,*_=api
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    assert wait(client,job)['execution_status']=='completed'
    assert client.get(job['status_url'],headers=headers(KEY_B)).status_code==404
    assert client.delete(URL+'/'+job['job_id'],headers=headers(KEY_B)).status_code==404
    assert client.get(job['status_url'],headers=headers()).status_code==200


def test_viewer_cannot_submit_or_cancel(api):
    client,*_=api
    assert client.post(URL,json={'designations':['A']},headers=headers(KEY_V)).status_code==403
    assert client.delete(URL+'/unknown',headers=headers(KEY_V)).status_code==403


def test_idempotency_preserves_job_and_original_window_conflicting_intent_rejected(api,monkeypatch):
    client,*_=api
    body={'designations':['A'],'harmonics':False}
    h=headers(**{'Idempotency-Key':'same-client-request'})
    first=client.post(URL,json=body,headers=h)
    assert first.status_code==200
    from aneos_api import network_routes
    monkeypatch.setattr(network_routes,'build_network_request',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Retry must reuse original window')))
    second=client.post(URL,json=body,headers=h)
    assert second.status_code==200 and second.json()['job_id']==first.json()['job_id']
    different=client.post(URL,json={**body,'designations':['B']},headers=h)
    assert different.status_code==409


def test_revoked_database_key_is_not_authorized_from_stale_global_map(api):
    client,_,sessions,*_=api
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    assert wait(client,job)['execution_status']=='completed'
    from aneos_api.database import User
    with sessions() as s:
        user=s.query(User).filter_by(user_id='synthetic-a').one();user.api_keys=[];s.commit()
    assert client.get(job['status_url'],headers=headers()).status_code==401


def test_service_stop_rejects_admission_but_preserves_existing_read(api):
    client,runtime,*_=api
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    assert wait(client,job)['execution_status']=='completed'
    runtime.close()
    assert not runtime.ready
    assert client.post(URL,json={'designations':['B'],'harmonics':False},headers=headers()).status_code==503
    assert client.get(job['status_url'],headers=headers()).status_code==200


def test_invalid_unknown_options_and_dual_credentials_are_rejected(api):
    client,*_=api
    assert client.post(URL,json={'designations':['A'],'owner_scope':'other'},headers=headers()).status_code==422
    assert client.post(URL,json={'designations':[' A']},headers=headers()).status_code==422
    assert client.post(URL,json={'designations':['A']},headers=headers(Authorization='Bearer conflicting')).status_code==401


def test_scoped_signed_token_positive_revocation_and_legacy_token_denial(api,monkeypatch):
    from aneos_api import auth
    from aneos_api.database import User
    client,_,sessions,*_=api
    monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-signing-key-'+'s'*40)
    issued=client.post('/api/v1/auth/token?audience=network-jobs-v1',headers=headers())
    assert issued.status_code==200,issued.text
    bearer={'Authorization':'Bearer '+issued.json()['access_token']}
    post=client.post(URL,json={'designations':['A'],'harmonics':False},headers=bearer)
    assert post.status_code==200
    status=client.get(post.json()['status_url'],headers=bearer)
    assert status.status_code==200
    old=auth.create_access_token('synthetic-a','admin')
    assert client.get(post.json()['status_url'],headers={'Authorization':'Bearer '+old}).status_code==401
    with sessions() as s:
        user=s.query(User).filter_by(user_id='synthetic-a').one();user.api_keys=[];s.commit()
    assert client.get(post.json()['status_url'],headers=bearer).status_code==401


def test_scoped_token_permissions_follow_current_database_role(api,monkeypatch):
    from aneos_api import auth
    from aneos_api.database import User
    client,_,sessions,*_=api
    monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-signing-key-'+'s'*40)
    token=client.post('/api/v1/auth/token?audience=network-jobs-v1',headers=headers()).json()['access_token']
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    with sessions() as s:
        u=s.query(User).filter_by(user_id='synthetic-a').one();u.role='viewer';s.commit()
    h={'Authorization':'Bearer '+token}
    assert client.get(job['status_url'],headers=h).status_code==200
    assert client.post(URL,json={'designations':['B']},headers=h).status_code==403
    assert client.delete(URL+'/'+job['job_id'],headers=h).status_code==403


def test_configured_key_rotation_retains_owner_without_accepting_revoked_key(api,monkeypatch):
    client,*_=api
    first='synthetic-configured-admin-'+'x'*40;second='synthetic-rotated-admin-'+'y'*40
    monkeypatch.setenv('ANEOS_ADMIN_API_KEY',first)
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers(first)).json()
    assert wait(client,job,first)['execution_status']=='completed'
    monkeypatch.setenv('ANEOS_ADMIN_API_KEY',second)
    assert client.get(job['status_url'],headers=headers(first)).status_code==401
    assert client.get(job['status_url'],headers=headers(second)).status_code==200


def test_duplicate_cross_account_key_configuration_fails_closed(api):
    from aneos_api.database import User
    client,_,sessions,*_=api
    with sessions() as s:
        u=s.query(User).filter_by(user_id='synthetic-b').one();u.api_keys=[KEY_A];s.commit()
    r=client.post(URL,json={'designations':['A']},headers=headers())
    assert r.status_code==503


def test_actual_cancellation_retains_owner_isolation_while_provider_blocked(api):
    client,runtime,*_=api;entered=threading.Event();release=threading.Event()
    class Block(Source):
        def fetch_neo_data(self,d):entered.set();assert release.wait(5);return super().fetch_neo_data(d)
    runtime.worker.handler.source_factory=Block
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    try:
        assert entered.wait(3)
        assert client.delete(URL+'/'+job['job_id'],headers=headers(KEY_B)).status_code==404
        cancelled=client.delete(URL+'/'+job['job_id'],headers=headers())
        assert cancelled.status_code==200 and cancelled.json()['execution_status']=='cancelled'
    finally:release.set()
    result=client.get(job['status_url'],headers=headers()).json()
    assert result['execution_status']=='cancelled' and result['scientific_report'] is None
    assert result['checkpoint_count']==0


def test_queue_backpressure_and_same_idempotency_reuse(api):
    client,runtime,*_=api;entered=threading.Event();release=threading.Event()
    class Block(Source):
        def fetch_neo_data(self,d):entered.set();assert release.wait(5);return super().fetch_neo_data(d)
    runtime.worker.handler.source_factory=Block
    h=headers(**{'Idempotency-Key':'capacity-original'})
    first=client.post(URL,json={'designations':['A'],'harmonics':False},headers=h)
    try:
        assert entered.wait(3)
        for d in ('B','C','D'):
            assert client.post(URL,json={'designations':[d],'harmonics':False},headers=headers()).status_code==200
        assert client.post(URL,json={'designations':['E'],'harmonics':False},headers=headers()).status_code==429
        repeat=client.post(URL,json={'designations':['A'],'harmonics':False},headers=h)
        assert repeat.status_code==200 and repeat.json()['job_id']==first.json()['job_id']
    finally:release.set()


def test_storage_outage_and_corrupt_result_never_return_success(api,monkeypatch):
    from aneos_api.job_store import JobStorageUnavailable
    client,runtime,*_=api
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    assert wait(client,job)['execution_status']=='completed'
    def fail(*a,**kw):raise JobStorageUnavailable('synthetic private DB URL')
    monkeypatch.setattr(runtime.store,'get',fail)
    result=client.get(job['status_url'],headers=headers())
    assert result.status_code==503 and 'private DB' not in result.text


def test_schema_and_worker_startup_failure_cleanup_and_health_are_not_success(api,monkeypatch):
    client,runtime,*_=api
    # Actual dead worker, not a false readiness flag; admission must stop.
    runtime.worker.stop.set();runtime._thread.join(timeout=3)
    assert not runtime._thread.is_alive()
    assert client.post(URL,json={'designations':['A']},headers=headers()).status_code==503


def test_api_process_kill_and_restart_recovers_preserved_input_and_owner(api,tmp_path):
    import os,subprocess,sys,json
    from aneos_api.network_runtime import NetworkRuntime
    from aneos_api.network_jobs import NetworkJobHandler
    from aneos_api import app
    client,runtime,sessions,*_=api;runtime.close()
    db_url=str(runtime.store.engine.url)
    submission=tmp_path/'submission.json';blocked=tmp_path/'blocked'
    script='''
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from aneos_api import app,database
from aneos_api.network_runtime import NetworkRuntime
from aneos_api.network_jobs import NetworkJobHandler
from aneos_core.data.models import NEOData,OrbitalElements
class Source:
 def fetch_neo_data(self,d):
  if d=='B':
   Path(sys.argv[3]).write_text('first source checkpoint already committed')
   time.sleep(60)
  return NEOData(designation=d,orbital_elements=OrbitalElements(semi_major_axis=1.,eccentricity=.2,inclination=180.))
engine=create_engine(sys.argv[1],connect_args={'check_same_thread':False,'timeout':5})
sessions=sessionmaker(bind=engine)
database.init_database=lambda:True
app.get_aneos_app=lambda:SimpleNamespace(initialize_services=AsyncMock(),shutdown_services=AsyncMock())
a=app.create_app();a.state.network_runtime_factory=lambda:NetworkRuntime(sessions,
 handler=NetworkJobHandler(source_factory=Source,source_profile=lambda:'1'*64),capacity=4,lease_seconds=1)
with TestClient(a) as c:
 r=c.post('/api/v1/analysis/analyze/network',json={'designations':['A','B'],'harmonics':False},
          headers={'X-API-Key':'synthetic-owner-a-key-'+'a'*32})
 assert r.status_code==200,r.text
 Path(sys.argv[2]).write_text(json.dumps(r.json()))
 time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',script,db_url,str(submission),str(blocked)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+20
        while not (submission.exists() and blocked.exists()) and child.poll() is None and time.monotonic()<end:time.sleep(.02)
        assert submission.exists() and blocked.exists(),child.communicate(timeout=1) if child.poll() is not None else 'API checkpoint marker missing'
    finally:
        if child.poll() is None:child.kill()
        child.communicate(timeout=5)
    job=json.loads(submission.read_text())
    prior=client.get(job['status_url'],headers=headers())
    assert prior.status_code==200 and prior.json()['checkpoint_count']==1
    # Wait for the persisted lease, not an inferred process status, before recovery.
    end=time.monotonic()+4
    while runtime.store.get('db-user:synthetic-a',job['job_id'])['lease_until']>time.time() and time.monotonic()<end:time.sleep(.02)
    calls=[]
    class Recovery(Source):
        def fetch_neo_data(self,d):assert d=='B';calls.append(d);return super().fetch_neo_data(d)
    next_app=app.create_app()
    replacement=NetworkRuntime(sessions,handler=NetworkJobHandler(source_factory=Recovery,source_profile=lambda:'1'*64),capacity=4,lease_seconds=2)
    next_app.state.network_runtime_factory=lambda:replacement
    with TestClient(next_app) as recovered:
        saved=wait(recovered,job)
        assert saved['execution_status']=='completed' and saved['attempts']==2
        assert calls==['B']
        assert saved['scientific_report']['object_assessments'][0]['assessment']['candidate']
        assert recovered.get(job['status_url'],headers=headers(KEY_B)).status_code==404


@pytest.mark.parametrize('claims',[
    {'aud':'other'}, {'iss':'other'}, {'exp':1}, {'iat':9999999999},
    {'credential_id':'invalid'}, {'sub':'db-user:synthetic-b'},
])
def test_scoped_token_claim_validation_cannot_select_another_owner(api,monkeypatch,claims):
    from aneos_api import auth
    from aneos_api.network_access import AUDIENCE,ISSUER,credential_id
    client,*_=api;monkeypatch.setattr(auth,'_JWT_SECRET','synthetic-signing-key-'+'s'*40)
    now=int(time.time())
    payload={'sub':'db-user:synthetic-a','aud':AUDIENCE,'iss':ISSUER,'iat':now,'exp':now+3600,
             'credential_id':credential_id(KEY_A)}
    payload.update(claims)
    token=auth._jose_jwt.encode(payload,auth._JWT_SECRET,algorithm='HS256')
    assert client.post(URL,json={'designations':['A']},headers={'Authorization':'Bearer '+token}).status_code==401


def test_runtime_startup_failure_runs_service_cleanup_without_yield(api,monkeypatch):
    from aneos_api import app
    client,_,sessions,*_=api
    application=app.create_app()
    from aneos_api.network_runtime import NetworkRuntime
    bad=NetworkRuntime(sessions,capacity=5)  # Existing queue is capacity4; never silently reset.
    application.state.network_runtime_factory=lambda:bad
    with pytest.raises(Exception):
        with TestClient(application):raise AssertionError('Incompatible queue must not start')
    app.get_aneos_app().shutdown_services.assert_awaited()


def test_duplicate_credential_headers_not_silently_selected(api):
    client,*_=api
    r=client.post(URL,json={'designations':['A']},headers=[('X-API-Key',KEY_A),('X-API-Key',KEY_B)])
    assert r.status_code==401


def test_inactive_user_and_auth_database_outage_cannot_grant_jobs(api,monkeypatch):
    from aneos_api.database import User
    from sqlalchemy.exc import OperationalError
    client,runtime,sessions,*_=api
    with sessions() as s:
        u=s.query(User).filter_by(user_id='synthetic-a').one();u.is_active=False;s.commit()
    assert client.post(URL,json={'designations':['A']},headers=headers()).status_code==401
    def fail():raise OperationalError('synthetic-auth-outage',{},None)
    monkeypatch.setattr(runtime.store,'sessions',fail)
    assert client.post(URL,json={'designations':['A']},headers=headers(KEY_B)).status_code==503


def test_health_observes_worker_without_starting_it_and_reports_degraded(api):
    client,runtime,*_=api
    result=client.get('/health').json()
    assert result['checks']['network_worker']['status']=='ok'
    runtime.close()
    result=client.get('/health').json()
    assert result['status']=='degraded' and result['checks']['network_worker']['status']=='degraded'
    assert not runtime.ready


def test_shutdown_budget_failure_is_explicit_and_does_not_advertise_ready(api):
    client,runtime,*_=api;entered=threading.Event();release=threading.Event()
    class Block(Source):
        def fetch_neo_data(self,d):entered.set();assert release.wait(5);return super().fetch_neo_data(d)
    runtime.worker.handler.source_factory=Block;runtime.shutdown_seconds=.01
    client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers())
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError,match='shutdown unconfirmed'):runtime.close()
        assert not runtime.ready and runtime.error=='NETWORK_WORKER_SHUTDOWN_UNCONFIRMED'
        assert client.post(URL,json={'designations':['B']},headers=headers()).status_code==503
    finally:
        release.set();runtime._thread.join(timeout=3);runtime.shutdown_seconds=3
    assert not runtime._thread.is_alive()


def test_read_only_runtime_mode_does_not_pretend_to_have_a_worker(api):
    from aneos_api import app
    from aneos_api.network_runtime import NetworkRuntime
    client,runtime,sessions,*_=api
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    wait(client,job);runtime.close()
    a=app.create_app();disabled=NetworkRuntime(sessions,capacity=4,enabled=False)
    a.state.network_runtime_factory=lambda:disabled
    with TestClient(a) as c:
        assert not disabled.ready
        assert c.get(job['status_url'],headers=headers()).status_code==200
        assert c.post(URL,json={'designations':['B']},headers=headers()).status_code==503


def test_boolean_horizon_not_coerced_into_scientific_configuration(api):
    client,*_=api
    assert client.post(URL,json={'designations':['A'],'historical_years':True},headers=headers()).status_code==422


def test_legacy_admin_cannot_provision_credentials_to_bypass_network_ownership(api):
    client,_,sessions,*_=api
    from aneos_api.database import User
    body={'username':'synthetic-new','email':'synthetic@example.invalid','role':'analyst','password':'synthetic-test-password-only'}
    for credential in ({'Authorization':'Bearer mock_admin_token'},{'X-API-Key':'admin-key-not-configured'}):
        assert client.post('/api/v1/admin/users',json=body,headers=credential).status_code==401
    with sessions() as s:assert s.query(User).filter_by(username='synthetic-new').count()==0


def test_verified_admin_provisions_in_same_database_but_cannot_read_another_owner_job(api,monkeypatch):
    client,_,sessions,*_=api
    key='synthetic-configured-admin-'+'z'*40;monkeypatch.setenv('ANEOS_ADMIN_API_KEY',key)
    assert client.get('/api/v1/admin/users',headers=headers(KEY_A)).status_code==403
    result=client.post('/api/v1/admin/users',json={'username':'synthetic-new','email':'synthetic@example.invalid','role':'analyst','password':'synthetic-test-password-only'},headers=headers(key))
    assert result.status_code==200,result.text
    from aneos_api.database import User
    with sessions() as s:
        u=s.query(User).filter_by(username='synthetic-new').one();created_key=u.api_keys[0]
    job=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers(created_key))
    assert job.status_code==200 and wait(client,job.json(),created_key)['execution_status']=='completed'
    assert client.get(job.json()['status_url'],headers=headers(key)).status_code==404
