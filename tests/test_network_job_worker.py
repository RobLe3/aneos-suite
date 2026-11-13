"""Actual SQL/handler controls with synthetic sources, not scientific calibration."""
import copy
import json
import threading
import time
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
from aneos_core.data.models import NEOData,OrbitalElements,CloseApproach


@pytest.fixture
def store(tmp_path):
    engine=create_engine('sqlite:///'+str(tmp_path/'jobs.db'),connect_args={'check_same_thread':False,'timeout':5})
    s=JobStore(sessionmaker(bind=engine),capacity=4)
    s.initialize_schema()
    yield s,engine
    engine.dispose()


def worker_module():
    from aneos_api import network_jobs
    return network_jobs


def neo(designation):
    return NEOData(designation=designation,orbital_elements=OrbitalElements(
        semi_major_axis=1.,eccentricity=.2,inclination=180.))


PROFILE='1'*64


def request(designations=None,**kwargs):
    return worker_module().build_network_request(designations or ['A'],source_profile_id=PROFILE,
                                                as_of_day='2026-11-12',**kwargs)


def submit(s,req=None,**kwargs):
    return s.submit(owner_scope='test-owner',kind=worker_module().NETWORK_KIND,request=req or request(),**kwargs)


class Source:
    def __init__(self):self.calls=[];self.history_calls=[]
    def fetch_neo_data(self,designation):
        self.calls.append(designation);return neo(designation)
    def fetch_approaches_between(self,designation,date_min,date_max):
        from aneos_core.data.sources.cad import CloseApproachQuery
        self.history_calls.append((designation,date_min,date_max))
        return CloseApproachQuery([CloseApproach(designation=designation,epoch_jd_tdb=2450000.,time_scale='TDB',body='Earth')],
            {'query':{'des':designation,'date-min':date_min,'date-max':date_max,'body':'Earth','limit':10},
             'completeness_status':'partial','count':1,'total':2,'canonical_response_sha256':'2'*64})


def worker(s,source,**kwargs):
    m=worker_module()
    h=m.NetworkJobHandler(source_factory=lambda:source,source_profile=lambda:PROFILE)
    return m.NetworkWorker(s,handler=h,lease_seconds=2,**kwargs)


def test_actual_execution_candidate_report_and_frozen_history(store):
    s,_=store;src=Source();job=submit(s,request(['A','B','A']))
    assert worker(s,src).run_once()=='completed'
    saved=s.get('test-owner',job['job_id'])
    assert saved['status']=='completed'
    result=worker_module().verify_network_job_result(saved['request'],saved['result'])
    report=result['scientific_report']
    assert [o['assessment']['designation'] for o in report['object_assessments']]==['A','B','A']
    assert report['object_assessments'][0]['assessment']['candidate']
    assert report['network_sigma'] is None
    assert src.calls==['A','B'] and len(src.history_calls)==2
    assert src.history_calls[0][1:] == ('1996-11-19','2026-11-12')
    assert result['acquisitions'][0]['history']['metadata']['completeness_status']=='partial'
    assert report['modules']['harmonics']['event_inventory'][0]['epochs_jd_tdb']==[2450000.]
    assert saved['checkpoint_count']==5  # two neo + two history + final result


def test_disabled_history_never_requests_provider(store):
    from aneos_core.pattern_analysis.session import PatternAnalysisConfig
    s,_=store;src=Source();job=submit(s,request(config=PatternAnalysisConfig(harmonics=False)))
    worker(s,src).run_once()
    assert not src.history_calls
    assert s.get('test-owner',job['job_id'])['status']=='completed'


def test_retry_keeps_first_committed_object_and_does_not_refetch(store):
    s,_=store;src=Source();original=src.fetch_neo_data
    def fail_second(d):
        if d=='B':raise RuntimeError('private provider detail')
        return original(d)
    src.fetch_neo_data=fail_second
    job=submit(s,request(['A','B']),max_attempts=2)
    w=worker(s,src,retry_seconds=.01);w.run_once()
    state=s.get('test-owner',job['job_id']);assert state['status']=='retry_wait'
    assert state['checkpoint_count']==2 and state['result'] is None
    src.fetch_neo_data=original;time.sleep(.02);assert w.run_once()=='completed'
    assert src.calls==['A','B']
    assert 'private provider detail' not in json.dumps(s.get('test-owner',job['job_id']))


def test_provider_outage_exhausts_bounded_attempts_not_success(store):
    s,_=store
    class Missing(Source):
        def fetch_neo_data(self,d):return None
    job=submit(s,max_attempts=1);worker(s,Missing()).run_once()
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='failed' and state['result'] is None
    assert state['error_code']=='ACQUISITION_UNAVAILABLE'


def test_wrong_identity_terminal_and_no_evidence_checkpoint(store):
    s,_=store
    class Wrong(Source):
        def fetch_neo_data(self,d):return neo('other')
    job=submit(s);worker(s,Wrong()).run_once()
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='failed' and state['error_code']=='INVALID_EVIDENCE'
    assert state['checkpoint_count']==0


def test_request_validation_before_any_source_access(store):
    s,_=store;src=Source();req=request();req['configuration']['combination_method']='fisher'
    job=submit(s,req);worker(s,src).run_once()
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='failed' and state['error_code']=='INVALID_REQUEST'
    assert not src.calls


def test_cancel_during_source_prevents_checkpoint_or_terminal_write(store):
    s,_=store;job=submit(s)
    class Cancel(Source):
        def fetch_neo_data(self,d):
            s.cancel('test-owner',job['job_id']);return neo(d)
    assert worker(s,Cancel()).run_once()=='lease_lost'
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='cancelled' and state['checkpoint_count']==0 and state['result'] is None


def test_heartbeat_retains_lease_during_slow_provider(store):
    s,_=store;job=submit(s)
    class Slow(Source):
        def fetch_neo_data(self,d):time.sleep(.9);return neo(d)
    m=worker_module();w=m.NetworkWorker(s,handler=m.NetworkJobHandler(source_factory=Slow,source_profile=lambda:PROFILE),lease_seconds=.5)
    assert w.run_once()=='completed'
    history=s.history('test-owner',job['job_id'])
    assert any(e['event']=='renewed' for e in history)


def test_result_validator_refuses_rehashed_report_wrong_request_or_inputs(store):
    s,_=store;job=submit(s);worker(s,Source()).run_once()
    saved=s.get('test-owner',job['job_id']);m=worker_module()
    bad=copy.deepcopy(saved['result']);bad['request_id']='0'*64
    with pytest.raises(ValueError):m.verify_network_job_result(saved['request'],bad)
    bad=copy.deepcopy(saved['result']);bad['scientific_report']['network_sigma']=5
    with pytest.raises(ValueError):m.verify_network_job_result(saved['request'],bad)
    other=request(['other'])
    with pytest.raises(ValueError):m.verify_network_job_result(other,saved['result'])


def test_unknown_kind_not_claimed(store):
    s,_=store;job=s.submit(owner_scope='test-owner',kind='unregistered-v1',request={'unknown':True})
    assert worker(s,Source()).run_once()=='idle'
    assert s.get('test-owner',job['job_id'])['attempts']==0


def test_source_profile_drift_blocks_new_acquisition(store):
    s,_=store;job=submit(s);src=Source();m=worker_module()
    h=m.NetworkJobHandler(source_factory=lambda:src,source_profile=lambda:'9'*64)
    m.NetworkWorker(s,handler=h).run_once()
    assert s.get('test-owner',job['job_id'])['error_code']=='SOURCE_CONFIGURATION_CHANGED'
    assert not src.calls


def test_real_process_termination_resumes_checkpoint_with_changed_source(store,tmp_path):
    import os,subprocess,sys
    from aneos_core.pattern_analysis.session import PatternAnalysisConfig
    s,engine=store;s.clock=lambda:1000.
    job=submit(s,request(['A','B'],config=PatternAnalysisConfig(harmonics=False)))
    marker=tmp_path/'blocked-after-first-checkpoint'
    script='''
import sys,time
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
from aneos_api.network_jobs import NetworkWorker,NetworkJobHandler
from aneos_core.data.models import NEOData,OrbitalElements
class Source:
 def fetch_neo_data(self,d):
  if d=='B':
   Path(sys.argv[2]).write_text('A is committed; B acquisition blocked')
   time.sleep(60)
  return NEOData(designation=d,orbital_elements=OrbitalElements(semi_major_axis=1.,eccentricity=.2,inclination=180.))
s=JobStore(sessionmaker(bind=create_engine(sys.argv[1])),capacity=4,clock=lambda:1000.)
w=NetworkWorker(s,handler=NetworkJobHandler(source_factory=Source,source_profile=lambda:'1'*64),lease_seconds=.5)
w.run_once()
'''
    child=subprocess.Popen([sys.executable,'-c',script,str(engine.url),str(marker)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+20
        while not marker.exists() and child.poll() is None and time.monotonic()<end:time.sleep(.02)
        assert marker.exists(), child.communicate(timeout=1) if child.poll() is not None else 'checkpoint marker missing'
    finally:
        if child.poll() is None:child.kill()
        child.communicate(timeout=5)
    interrupted=s.get('test-owner',job['job_id'])
    assert interrupted['status']=='running' and interrupted['checkpoint_count']==1
    s.clock=lambda:1001.
    src=Source()
    def changed_provider(d):
        assert d=='B','Committed A must not be reacquired or refreshed'
        src.calls.append(d);return neo(d)
    src.fetch_neo_data=changed_provider
    assert worker(s,src).run_once()=='completed'
    assert src.calls==['B']
    saved=s.get('test-owner',job['job_id']);assert saved['attempts']==2
    worker_module().verify_network_job_result(saved['request'],saved['result'])
    engine.dispose()
    # A third database/process instance reads the terminal, replayable report.
    read='''
import sys,json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
from aneos_api.network_jobs import verify_network_job_result
s=JobStore(sessionmaker(bind=create_engine(sys.argv[1])),capacity=4)
r=s.get('test-owner',sys.argv[2]);verify_network_job_result(r['request'],r['result'])
print(json.dumps({'status':r['status'],'candidate':r['result']['scientific_report']['object_assessments'][0]['assessment']['candidate']}))
'''
    output=subprocess.run([sys.executable,'-c',read,str(engine.url),job['job_id']],capture_output=True,text=True,timeout=20)
    assert output.returncode==0,output.stderr
    assert json.loads(output.stdout)=={'status':'completed','candidate':True}


def test_unconfirmed_terminal_storage_recovers_final_checkpoint_without_provider(store,monkeypatch):
    from aneos_api.job_store import JobStorageUnavailable
    s,_=store;s.clock=lambda:1000.;job=submit(s);src=Source()
    original=s.finish
    def outage(*a):raise JobStorageUnavailable('unconfirmed write')
    monkeypatch.setattr(s,'finish',outage)
    with pytest.raises(JobStorageUnavailable):worker(s,src).run_once()
    assert s.get('test-owner',job['job_id'])['status']=='running'
    assert s.get('test-owner',job['job_id'])['checkpoint_count']==3
    monkeypatch.setattr(s,'finish',original);s.clock=lambda:1003.
    def forbidden():raise AssertionError('Frozen recovery must not initialize/acquire a provider')
    h=worker_module().NetworkJobHandler(source_factory=forbidden,source_profile=forbidden)
    assert worker_module().NetworkWorker(s,handler=h).run_once()=='completed'


def test_committed_checkpoint_binding_cannot_be_bypassed_by_handler(store):
    s,_=store;job=submit(s);m=worker_module()
    class Forged(m.NetworkJobHandler):
        def run(self,context):
            result=super().run(context)
            result['acquisitions'][0]['input']['neo_snapshot']['orbital_elements']['inclination']=0.
            return m._result_from_records(context.lease.request,result['acquisitions'])
    h=Forged(source_factory=Source,source_profile=lambda:PROFILE)
    m.NetworkWorker(s,handler=h).run_once()
    saved=s.get('test-owner',job['job_id'])
    assert saved['status']=='failed' and saved['result'] is None
    assert saved['error_code']=='INVALID_EVIDENCE'


def test_foreign_checkpoint_refused_without_source_access(store):
    s,_=store;job=submit(s);lease=s.claim('first',lease_seconds=.2,kinds=[worker_module().NETWORK_KIND])
    s.checkpoint(lease,'neo:0',{'schema_version':'network-input-checkpoint-v1','request_id':'0'*64,
                            'designation':'A','neo_snapshot':neo('A').to_dict()})
    time.sleep(.22);src=Source();worker(s,src).run_once()
    assert not src.calls
    assert s.get('test-owner',job['job_id'])['status']=='failed'


def test_cooperative_worker_stop_releases_to_retry_without_completing(store):
    s,_=store;job=submit(s);stop=threading.Event()
    class Stop(Source):
        def fetch_neo_data(self,d):stop.set();return neo(d)
    m=worker_module();w=m.NetworkWorker(s,handler=m.NetworkJobHandler(source_factory=Stop,source_profile=lambda:PROFILE),stop_event=stop)
    assert w.run_once()=='failure_recorded'
    assert s.get('test-owner',job['job_id'])['status']=='retry_wait'
    assert s.get('test-owner',job['job_id'])['error_code']=='WORKER_STOPPED'
    assert w.run_once()=='stopped'


def test_attempt_deadline_is_not_scientific_abstention(store):
    s,_=store;job=submit(s)
    class Slow(Source):
        def fetch_neo_data(self,d):time.sleep(.04);return neo(d)
    worker(s,Slow(),attempt_seconds=.02).run_once()
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='retry_wait' and state['error_code']=='ATTEMPT_DEADLINE'
    assert state['checkpoint_count']==0


def test_provider_cooldown_hint_not_retried_early(store):
    from aneos_core.data.provider_http import ProviderUnavailableError
    s,_=store;job=submit(s);later=time.time()+3600
    class Cooldown(Source):
        def fetch_neo_data(self,d):raise ProviderUnavailableError('synthetic cooldown',retry_at=later)
    worker(s,Cooldown()).run_once()
    assert s.get('test-owner',job['job_id'])['available_at']>=later


def test_invalid_history_metadata_cannot_be_frozen_as_coverage(store):
    s,_=store;job=submit(s)
    class InvalidHistory(Source):
        def fetch_approaches_between(self,*args):
            events=super().fetch_approaches_between(*args)
            events.query_metadata['completeness_status']='complete'
            return events
    worker(s,InvalidHistory()).run_once()
    state=s.get('test-owner',job['job_id'])
    assert state['status']=='failed' and state['checkpoint_count']==1


def test_fixed_window_fetcher_uses_documented_bounds_without_relative_cache(monkeypatch):
    from aneos_core.data.fetcher import DataFetcher
    from unittest.mock import Mock
    f=DataFetcher.__new__(DataFetcher);bound=Mock(return_value='synthetic')
    monkeypatch.setattr(f,'_fetch_close_approaches',bound)
    assert f.fetch_approaches_between('A','2000-01-01','2026-11-12')=='synthetic'
    bound.assert_called_once_with('A',date_min='2000-01-01',date_max='2026-11-12')
    with pytest.raises(ValueError):f.fetch_approaches_between('A','2026-11-12','2000-01-01')


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(schema_version='unsupported'),
    lambda r:r.update(unregistered_option=True),
    lambda r:r.update(designations=['']),
    lambda r:r.update(designations=[' A']),
    lambda r:r['historical_query'].update(date_min='1900-01-01'),
    lambda r:r.update(screening_policy_id='0'*64),
    lambda r:r.update(report_engine='invented-calibrated-model'),
    lambda r:r.update(source_profile_id='invalid'),
])
def test_versioned_request_rejects_unknown_or_changed_contract(mutate):
    m=worker_module();r=request();mutate(r)
    with pytest.raises(ValueError):m.validate_network_request(r)


@pytest.mark.parametrize('kwargs',[{'lease_seconds':0},{'lease_seconds':True},{'retry_seconds':float('nan')},
                                    {'attempt_seconds':0},{'retry_seconds':301}])
def test_worker_policy_bounds(store,kwargs):
    with pytest.raises(ValueError):worker_module().NetworkWorker(store[0],**kwargs)


def test_default_factory_honors_source_priority_and_profiles_settings(monkeypatch):
    from aneos_core.config import settings
    from aneos_core.data import fetcher
    from types import SimpleNamespace
    from unittest.mock import Mock
    api=settings.APIConfig(data_sources_priority=['SBDB','Horizons'])
    monkeypatch.setattr(settings,'get_config',lambda:SimpleNamespace(api=api))
    factory=Mock(return_value='explicit-source');monkeypatch.setattr(fetcher,'DataFetcher',factory)
    m=worker_module();before=m.current_source_profile()
    assert m._default_source()=='explicit-source'
    factory.assert_called_once_with(source_priority=['SBDB','Horizons'])
    api.request_timeout+=1
    assert m.current_source_profile()!=before


def test_actual_fixed_window_transport_parses_empty_coverage(monkeypatch):
    from unittest.mock import Mock
    from aneos_core.data import fetcher
    f=fetcher.DataFetcher.__new__(fetcher.DataFetcher)
    response=Mock();response.json.return_value={'signature':{'source':'NASA/JPL SBDB Close Approach Data API','version':'1.5'},'count':0,'total':0}
    call=Mock(return_value=response);monkeypatch.setattr(fetcher,'provider_get',call)
    events=f.fetch_approaches_between('A','2000-01-01','2026-11-12')
    assert list(events)==[] and events.query_metadata['completeness_status']=='complete'
    assert call.call_args.kwargs['params']['date-min']=='2000-01-01'
    assert call.call_args.kwargs['params']['date-max']=='2026-11-12'
    assert call.call_args.kwargs['params']['body']=='Earth'
    assert call.call_args.kwargs['params']['limit']==10


def test_internal_handler_failure_is_not_invalid_science_or_success(store):
    s,_=store;job=submit(s);m=worker_module()
    class Bug(m.NetworkJobHandler):
        def run(self,ctx):raise RuntimeError('private internal failure')
    m.NetworkWorker(s,handler=Bug()).run_once()
    saved=s.get('test-owner',job['job_id'])
    assert saved['error_code']=='WORKER_INTERNAL_ERROR' and saved['result'] is None


def test_source_initialization_outage_is_retryable(store):
    s,_=store;job=submit(s);m=worker_module()
    def fail():raise RuntimeError('private source initialization')
    m.NetworkWorker(s,handler=m.NetworkJobHandler(source_factory=fail,source_profile=lambda:PROFILE)).run_once()
    assert s.get('test-owner',job['job_id'])['status']=='retry_wait'


def test_empty_persistent_queue_and_cooperative_serving(store):
    s,_=store;w=worker(s,Source())
    assert w.run_once()=='idle'
    t=threading.Thread(target=lambda:w.serve(poll_seconds=.01));t.start()
    job=submit(s);end=time.monotonic()+5
    try:
        while s.get('test-owner',job['job_id'])['status']!='completed' and time.monotonic()<end:time.sleep(.02)
        assert s.get('test-owner',job['job_id'])['status']=='completed'
    finally:w.stop.set();t.join(timeout=5)
    assert not t.is_alive()
    with pytest.raises(ValueError):w.serve(poll_seconds=0)


@pytest.mark.parametrize('signal',['stop','deadline'])
def test_blocked_provider_stops_renewing_and_replacement_worker_fences_old_execution(store,signal):
    s,_=store;job=submit(s);m=worker_module();entered=threading.Event();release=threading.Event()
    class Blocked(Source):
        def fetch_neo_data(self,d):
            entered.set();assert release.wait(5);return neo(d)
    stop=threading.Event();outcomes=[]
    class Capture(m.NetworkJobHandler):
        def run(self,context):
            self.context=context
            return super().run(context)
    handler=Capture(source_factory=Blocked,source_profile=lambda:PROFILE)
    first=m.NetworkWorker(s,handler=handler,lease_seconds=1,attempt_seconds=10,stop_event=stop)
    thread=threading.Thread(target=lambda:outcomes.append(first.run_once()));thread.start()
    try:
        assert entered.wait(3)
        if signal=='stop':stop.set()
        else:handler.context.deadline=time.monotonic()-1
        # Heartbeat must stop extending a blocked attempt. Wait for actual SQL
        # lease expiry, not an assumed status message or just a thread marker.
        end=time.monotonic()+3
        while s.get('test-owner',job['job_id'])['lease_until']>time.time() and time.monotonic()<end:time.sleep(.02)
        assert s.get('test-owner',job['job_id'])['lease_until']<=time.time()
        assert worker(s,Source()).run_once()=='completed'
        assert s.get('test-owner',job['job_id'])['attempts']==2
    finally:release.set();thread.join(timeout=5)
    assert not thread.is_alive() and outcomes==['lease_lost']
    saved=s.get('test-owner',job['job_id'])
    assert saved['status']=='completed' and saved['result'] is not None
