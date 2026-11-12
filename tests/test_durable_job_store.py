"""Real SQL transactions/processes; synthetic payloads, no scientific validation."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def module():
    from aneos_api import job_store
    return job_store


class Clock:
    def __init__(self): self.now=1000.
    def __call__(self): return self.now


@pytest.fixture
def store(tmp_path):
    engine=create_engine('sqlite:///'+str(tmp_path/'jobs.db'),connect_args={'check_same_thread':False,'timeout':10})
    factory=sessionmaker(bind=engine)
    clock=Clock()
    result=module().JobStore(factory,clock=clock,capacity=3)
    result.initialize_schema()
    yield result,clock,engine
    engine.dispose()


def submit(store, owner='scope-A', **kwargs):
    return store.submit(owner_scope=owner,kind='network-v1',request={'designations':['SYNTHETIC'],'policy':'test-only'},**kwargs)


def test_submit_owner_scope_no_fabricated_science_and_idempotency(store):
    s,clock,_=store
    job=submit(s,idempotency_key='same-request')
    assert job['status']=='queued' and job['result'] is None
    assert 'network_sigma' not in job and 'combined_p_value' not in job
    assert submit(s,idempotency_key='same-request')['job_id']==job['job_id']
    assert s.get('scope-B',job['job_id']) is None
    assert s.get('scope-A',job['job_id'])['request']==job['request']
    assert 'lease_token' not in job
    with pytest.raises(module().IdempotencyConflict):
        s.submit(owner_scope='scope-A',kind='network-v1',request={'changed':True},idempotency_key='same-request')


def test_capacity_is_transactional_and_released_only_once(store):
    s,_,_=store
    jobs=[submit(s) for _ in range(3)]
    with pytest.raises(module().QueueFull): submit(s)
    lease=s.claim('worker-1')
    s.finish(lease,{'status':'synthetic_transport_complete_not_validated'})
    assert submit(s)['status']=='queued'
    with pytest.raises(module().LostLease): s.finish(lease,{'again':True})
    with pytest.raises(module().QueueFull): submit(s)


def test_two_owners_cannot_cancel_or_read_each_others_jobs(store):
    s,_,_=store
    job=submit(s)
    assert not s.cancel('scope-B',job['job_id'])
    assert s.get('scope-A',job['job_id'])['status']=='queued'
    assert s.cancel('scope-A',job['job_id'])
    assert s.get('scope-A',job['job_id'])['status']=='cancelled'
    assert not s.cancel('scope-A',job['job_id'])


def test_expired_worker_cannot_renew_checkpoint_or_complete_after_reclaim(store):
    s,clock,_=store
    job=submit(s)
    old=s.claim('worker-A',lease_seconds=10)
    s.checkpoint(old,'input:0',{'native':'synthetic frozen evidence'})
    clock.now+=11
    new=s.claim('worker-B',lease_seconds=10)
    assert new.job_id==old.job_id and new.token!=old.token and new.attempt==2
    assert s.checkpoints(new)['input:0']=={'native':'synthetic frozen evidence'}
    for operation in [lambda:s.renew(old),lambda:s.checkpoint(old,'later',{}),lambda:s.finish(old,{}),lambda:s.fail(old,'STALE'),lambda:s.checkpoints(old)]:
        with pytest.raises(module().LostLease): operation()
    s.finish(new,{'status':'synthetic_completed'})
    assert s.get('scope-A',job['job_id'])['result']=={'status':'synthetic_completed'}


def test_checkpoint_is_immutable_and_copies_do_not_mutate_history(store):
    s,_,_=store
    submit(s);lease=s.claim('worker')
    payload={'raw':['not-measured-test']}
    s.checkpoint(lease,'source:0',payload);payload['raw'].append('mutated')
    assert s.checkpoints(lease)['source:0']=={'raw':['not-measured-test']}
    s.checkpoint(lease,'source:0',{'raw':['not-measured-test']})
    with pytest.raises(module().ImmutableCheckpoint):s.checkpoint(lease,'source:0',{'different':True})
    assert s.checkpoints(lease)['source:0']=={'raw':['not-measured-test']}


def test_cancel_running_job_fences_worker_but_preserves_evidence(store):
    s,_,_=store
    job=submit(s);lease=s.claim('worker')
    s.checkpoint(lease,'frozen',{'input':42})
    assert s.cancel('scope-A',job['job_id'])
    with pytest.raises(module().LostLease):s.finish(lease,{'probability':.9})
    saved=s.get('scope-A',job['job_id'])
    assert saved['result'] is None and saved['checkpoint_count']==1
    assert s.history('scope-B',job['job_id']) is None
    assert [e['event'] for e in s.history('scope-A',job['job_id'])]==['queued','claimed','checkpoint','cancelled']


def test_retry_not_eligible_before_deadline_and_attempt_limit_terminal(store):
    s,clock,_=store
    job=submit(s,max_attempts=2)
    lease=s.claim('worker');s.fail(lease,'SOURCE_UNAVAILABLE',retry_at=clock.now+20)
    assert s.get('scope-A',job['job_id'])['status']=='retry_wait'
    assert s.claim('other') is None
    clock.now+=20
    lease=s.claim('other');assert lease.attempt==2
    s.fail(lease,'SOURCE_UNAVAILABLE',retry_at=clock.now+20)
    assert s.get('scope-A',job['job_id'])['status']=='failed'
    assert s.claim('other') is None
    assert submit(s)['status']=='queued'


def test_last_killed_attempt_becomes_failed_on_recovery_without_reexecution(store):
    s,clock,_=store
    job=submit(s,max_attempts=1)
    s.claim('worker',lease_seconds=5);clock.now+=6
    assert s.claim('recovery') is None
    assert s.get('scope-A',job['job_id'])['status']=='failed'
    assert s.get('scope-A',job['job_id'])['result'] is None


def test_parallel_claimers_obtain_only_one_lease(store):
    s,_,_=store
    submit(s)
    with ThreadPoolExecutor(max_workers=6) as pool:
        leases=list(pool.map(lambda i:s.claim(f'worker-{i}'),range(6)))
    assert sum(x is not None for x in leases)==1


def test_capacity_parallel_submission_is_not_advisory_only(store):
    s,_,_=store
    def attempt(i):
        try:return submit(s)['job_id']
        except module().QueueFull:return None
    with ThreadPoolExecutor(max_workers=8) as pool:ids=list(pool.map(attempt,range(8)))
    assert len([i for i in ids if i])==3


@pytest.mark.parametrize('payload',[{'bad':float('nan')},{'bad':float('inf')},['not a request object']])
def test_invalid_json_does_not_consume_admission_or_install_fake_request(store,payload):
    s,_,_=store
    with pytest.raises(ValueError):s.submit(owner_scope='scope-A',kind='network-v1',request=payload)
    assert len([submit(s) for _ in range(3)])==3


def test_storage_failure_is_not_an_empty_success(store):
    s,_,engine=store
    with engine.begin() as connection:connection.execute(text('DROP TABLE durable_jobs'))
    with pytest.raises(module().JobStorageUnavailable):submit(s)


def test_file_sqlite_uses_independent_connections_not_shared_static_pool():
    from aneos_api.database import engine
    from sqlalchemy.pool import StaticPool
    assert not isinstance(engine.pool,StaticPool)


def test_first_claim_is_not_recorded_as_recovery_and_event_revisions_are_contiguous(store):
    s,_,_=store
    job=submit(s);lease=s.claim('worker')
    assert s.history('scope-A',job['job_id'])[1]['details']['recovered_expired_lease'] is False
    s.renew(lease);s.checkpoint(lease,'point',{})
    events=s.history('scope-A',job['job_id'])
    assert [e['revision'] for e in events]==list(range(1,len(events)+1))


def test_mutated_request_cannot_be_used_as_the_original_lease(store):
    s,_,_=store
    job=submit(s);lease=s.claim('worker')
    lease.request['policy']='changed locally'
    with pytest.raises(module().LostLease):s.finish(lease,{'value':'from altered request'})
    assert s.get('scope-A',job['job_id'])['request']['policy']=='test-only'


def test_nonstring_keys_and_non_json_python_types_are_not_silently_coerced(store):
    s,_,_=store
    for request in [{1:'lost-type'},{'tuple':(1,2)},{'nested':[{None:'not a JSON key'}]}]:
        with pytest.raises(ValueError):s.submit(owner_scope='scope-A',kind='network-v1',request=request)


def test_unsafe_sqlite_durability_settings_cannot_confirm_saved_jobs(store):
    from sqlalchemy import event
    s,_,engine=store
    engine.dispose()
    def unsafe(connection, _):connection.execute('PRAGMA synchronous=OFF')
    event.listen(engine,'connect',unsafe)
    with pytest.raises(module().JobStorageUnavailable):submit(s)


def test_parallel_idempotency_reuses_one_admission_slot(store):
    s,_,_=store
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(lambda _:submit(s,idempotency_key='same'),range(8)))
    assert len({r['job_id'] for r in rows})==1
    assert len([submit(s) for _ in range(2)])==2
    with pytest.raises(module().QueueFull):submit(s)


def test_db_lock_wait_cannot_extend_an_expired_workers_permission_to_commit(store):
    s,clock,engine=store
    job=submit(s);lease=s.claim('worker',lease_seconds=10)
    before_wait=threading.Event()
    def observed_clock():
        value=clock.now
        before_wait.set()
        return value
    s.clock=observed_clock
    connection=engine.connect();transaction=connection.begin()
    connection.execute(text('UPDATE durable_jobs SET revision=revision WHERE job_id=:id'),{'id':job['job_id']})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(s.finish,lease,{'late':'must not commit'})
        try:
            assert before_wait.wait(5)
            clock.now+=11
        finally:
            transaction.commit();connection.close()
        with pytest.raises(module().LostLease):future.result(timeout=10)
    assert s.get('scope-A',job['job_id'])['result'] is None


@pytest.mark.parametrize('url',['sqlite:///:memory:','sqlite:///file::memory:?cache=shared&uri=true',
                               'sqlite:///file:jobmem?mode=memory&cache=shared&uri=true'])
def test_ephemeral_database_is_never_accepted_as_durable_storage(url):
    from sqlalchemy.pool import QueuePool
    engine=create_engine(url,poolclass=QueuePool)
    try:
        with pytest.raises(ValueError,match='file SQLite'):module().JobStore(sessionmaker(bind=engine))
    finally:engine.dispose()


def test_configuration_mismatch_and_corruption_are_not_reset_by_initialization(store):
    s,_,engine=store
    job=submit(s)
    other=module().JobStore(sessionmaker(bind=engine),capacity=10)
    with pytest.raises(module().JobIntegrityError):other.initialize_schema()
    assert s.get('scope-A',job['job_id'])['status']=='queued'
    with engine.begin() as connection:
        connection.execute(text('UPDATE durable_jobs SET request_json=:raw WHERE job_id=:id'),{'raw':'{"changed":true}','id':job['job_id']})
    with pytest.raises(module().JobIntegrityError):s.get('scope-A',job['job_id'])
    with pytest.raises(module().JobIntegrityError):s.claim('worker')


def test_checkpoint_error_rolls_back_event_revision_and_admission_state(store):
    s,_,_=store
    job=submit(s);lease=s.claim('worker')
    s.checkpoint(lease,'frozen',{'value':1})
    before=s.history('scope-A',job['job_id'])
    with pytest.raises(module().ImmutableCheckpoint):s.checkpoint(lease,'frozen',{'value':2})
    assert s.history('scope-A',job['job_id'])==before
    assert s.get('scope-A',job['job_id'])['revision']==before[-1]['revision']


def test_real_process_kill_recovers_frozen_input_and_new_process_reads_completion(store,tmp_path):
    import os,subprocess,sys,time
    s,_,engine=store;s.clock=time.time
    job=submit(s);marker=tmp_path/'checkpointed'
    script='''
import sys,time
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
s=JobStore(sessionmaker(bind=create_engine(sys.argv[1])),capacity=3)
lease=s.claim('killed-worker',lease_seconds=.3)
s.checkpoint(lease,'input:0',{'native':'frozen synthetic data','missing_measurement':None})
Path(sys.argv[2]).write_text('committed')
time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',script,str(engine.url),str(marker)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+15
        while not marker.exists() and time.monotonic()<end:time.sleep(.02)
        assert marker.exists()
    finally:
        child.kill();child.wait(timeout=5)
    state=s.get('scope-A',job['job_id'])
    time.sleep(max(0,state['lease_until']-time.time())+.02)
    lease=s.claim('recovery-worker')
    assert lease.attempt==2
    assert s.checkpoints(lease)['input:0']['missing_measurement'] is None
    s.finish(lease,{'execution':'completed_test','scientific_inference':None})
    read='''
import sys,json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
s=JobStore(sessionmaker(bind=create_engine(sys.argv[1])),capacity=3)
print(json.dumps(s.get('scope-A',sys.argv[2])))
'''
    other=subprocess.run([sys.executable,'-c',read,str(engine.url),job['job_id']],capture_output=True,text=True,timeout=15)
    assert other.returncode==0,other.stderr
    saved=json.loads(other.stdout)
    assert saved['status']=='completed' and saved['result']=={'execution':'completed_test','scientific_inference':None}
    assert saved['checkpoint_count']==1
    assert any(e['event']=='claimed' and e['details'].get('recovered_expired_lease') for e in s.history('scope-A',job['job_id']))


def test_kill_inside_transaction_rolls_back_checkpoint_event_and_revision(store,tmp_path):
    import os,subprocess,sys,time
    s,_,engine=store;s.clock=time.time
    job=submit(s);marker=tmp_path/'uncommitted'
    script='''
import sys,time
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.job_store import JobStore
s=JobStore(sessionmaker(bind=create_engine(sys.argv[1])),capacity=3)
lease=s.claim('crashing-worker',lease_seconds=.3)
original=s._event
def pause(session,*args,**kwargs):
 original(session,*args,**kwargs)
 session.flush()
 Path(sys.argv[2]).write_text('flushed but not committed')
 time.sleep(60)
s._event=pause
s.checkpoint(lease,'must-rollback',{'uncommitted':True})
'''
    child=subprocess.Popen([sys.executable,'-c',script,str(engine.url),str(marker)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        end=time.monotonic()+15
        while not marker.exists() and time.monotonic()<end:time.sleep(.02)
        assert marker.exists()
    finally:
        child.kill();child.wait(timeout=5)
    state=s.get('scope-A',job['job_id'])
    assert state['revision']==2 and state['checkpoint_count']==0
    assert [e['event'] for e in s.history('scope-A',job['job_id'])]==['queued','claimed']
    time.sleep(max(0,state['lease_until']-time.time())+.02)
    lease=s.claim('recovery')
    assert s.checkpoints(lease)=={}
    s.finish(lease,{'execution':'recovered'})


def test_result_integrity_and_kind_scope_are_not_silently_relaxed(store):
    s,_,engine=store
    job=submit(s)
    assert s.claim('other-kind',kinds=['unregistered-v2']) is None
    with pytest.raises(ValueError):s.claim('empty-kind-scope',kinds=[])
    lease=s.claim('allowed',kinds=['network-v1'])
    s.finish(lease,{'payload':'synthetic result'})
    with engine.begin() as connection:
        connection.execute(text('UPDATE durable_jobs SET result_json=:payload WHERE job_id=:id'),
            {'payload':'{"tampered":true}','id':job['job_id']})
    with pytest.raises(module().JobIntegrityError):s.get('scope-A',job['job_id'])


def test_unsupported_sqlite_version_fails_before_admission(store,monkeypatch):
    s,_,engine=store
    monkeypatch.setattr(engine.dialect,'server_version_info',(3,34,0))
    with pytest.raises(module().JobStorageUnavailable,match='3.35'):submit(s)


def test_same_base_registration_and_new_store_do_not_reset_existing_results(store):
    from aneos_api.database import Base
    s,_,engine=store
    assert module().DurableJob.__table__.metadata is Base.metadata
    job=submit(s);lease=s.claim('worker');s.finish(lease,{'execution':'synthetic completed'})
    other=module().JobStore(sessionmaker(bind=engine),capacity=3)
    other.initialize_schema()
    assert other.get('scope-A',job['job_id'])['result']=={'execution':'synthetic completed'}
    assert len(other.history('scope-A',job['job_id']))==3


def test_terminal_write_outage_rolls_back_result_event_and_capacity(store,monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    s,_,_=store
    jobs=[submit(s) for _ in range(3)];lease=s.claim('worker')
    original=s._event
    def fail_after_flush(session,*args,**kwargs):
        original(session,*args,**kwargs);session.flush()
        raise SQLAlchemyError('synthetic storage outage before commit')
    monkeypatch.setattr(s,'_event',fail_after_flush)
    with pytest.raises(module().JobStorageUnavailable):s.finish(lease,{'execution':'not committed'})
    state=s.get('scope-A',lease.job_id)
    assert state['status']=='running' and state['result'] is None and state['revision']==2
    with pytest.raises(module().QueueFull):submit(s)
    assert [e['event'] for e in s.history('scope-A',lease.job_id)]==['queued','claimed']
    monkeypatch.setattr(s,'_event',original)
    s.finish(lease,{'execution':'confirmed'})
    assert submit(s)['status']=='queued'


def test_event_and_checkpoint_identity_binding_corruption_is_detected(store):
    s,_,engine=store
    job=submit(s);lease=s.claim('worker');s.checkpoint(lease,'source-A',{'raw':'synthetic'})
    with engine.begin() as connection:
        connection.execute(text('UPDATE durable_job_events SET event=:event WHERE job_id=:id AND revision=1'),
                           {'event':'completed','id':job['job_id']})
        connection.execute(text('UPDATE durable_job_checkpoints SET key=:key WHERE job_id=:id'),
                           {'key':'source-B','id':job['job_id']})
    with pytest.raises(module().JobIntegrityError):s.history('scope-A',job['job_id'])
    with pytest.raises(module().JobIntegrityError):s.checkpoints(lease)


def test_caller_owned_connection_or_session_is_not_borrowed_or_rolled_back(store):
    s,_,engine=store
    with engine.begin() as connection:
        with pytest.raises(ValueError,match='caller-owned'):module().JobStore(sessionmaker(bind=connection))
        assert connection.in_transaction()
    with pytest.raises(ValueError,match='sessionmaker'):module().JobStore(lambda:None)


def test_storage_import_does_not_assemble_app_import_optional_ml_or_create_database(tmp_path):
    import os,subprocess,sys
    target=tmp_path/'must-not-create-on-import.db'
    script='''
import sys
from pathlib import Path
class NoComposition:
 def find_spec(self,fullname,*args):
  if fullname=='aneos_api.app' or fullname=='torch' or fullname.startswith('torch.'):
   raise AssertionError('storage import must not assemble optional services')
sys.meta_path.insert(0,NoComposition())
import aneos_api
import aneos_api.database
import aneos_api.job_store
assert 'aneos_api.app' not in sys.modules and 'torch' not in sys.modules, repr({k:repr(sys.modules.get(k)) for k in ['aneos_api.app','torch']})
assert not Path(sys.argv[1]).exists()
assert 'create_app' in aneos_api.__all__ and 'create_app' in dir(aneos_api)
'''
    result=subprocess.run([sys.executable,'-c',script,str(target)],capture_output=True,text=True,timeout=15,
        env={**os.environ,'ANEOS_DATABASE_URL':'sqlite:///'+str(target),'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stdout+result.stderr
    assert not target.exists()


def test_lazy_api_public_exports_still_resolve_to_original_implementations():
    import aneos_api
    from aneos_api.models import APIResponse,ErrorResponse
    assert aneos_api.APIResponse is APIResponse and aneos_api.ErrorResponse is ErrorResponse
    assert aneos_api._EXPORTS['create_app']==('.app','create_app')
    assert aneos_api._EXPORTS['ANEOSApp']==('.app','ANEOSApp')
    with pytest.raises(AttributeError):getattr(aneos_api,'not_a_real_export')


def test_read_only_reopen_can_inspect_history_but_cannot_submit(store):
    s,_,engine=store
    job=submit(s);lease=s.claim('worker');s.finish(lease,{'execution':'completed'})
    readonly=create_engine('sqlite:///file:'+str(engine.url.database)+'?mode=ro&uri=true')
    try:
        observer=module().JobStore(sessionmaker(bind=readonly),capacity=3)
        assert observer.get('scope-A',job['job_id'])['result']=={'execution':'completed'}
        assert len(observer.history('scope-A',job['job_id']))==3
        with pytest.raises(module().JobStorageUnavailable):submit(observer)
    finally:readonly.dispose()


def test_incompatible_existing_schema_is_refused_not_destructively_repaired(store):
    s,_,engine=store
    job=submit(s)
    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE durable_job_checkpoints DROP COLUMN receipt_sha256'))
    with pytest.raises(module().JobIntegrityError,match='migration'):s.initialize_schema()
    with engine.connect() as connection:
        assert connection.execute(text('SELECT status FROM durable_jobs WHERE job_id=:id'),{'id':job['job_id']}).scalar_one()=='queued'


def test_missing_or_inconsistent_admission_state_is_not_reset_for_existing_work(store):
    s,_,engine=store
    job=submit(s)
    with engine.begin() as connection:connection.execute(text('UPDATE durable_job_queue SET active=0'))
    with pytest.raises(module().JobIntegrityError,match='accounting'):s.initialize_schema()
    with engine.begin() as connection:connection.execute(text('DELETE FROM durable_job_queue'))
    with pytest.raises(module().JobIntegrityError,match='Admission state missing'):s.initialize_schema()
    with engine.connect() as connection:
        assert connection.execute(text('SELECT count(*) FROM durable_job_queue')).scalar_one()==0
        assert connection.execute(text('SELECT status FROM durable_jobs WHERE job_id=:id'),{'id':job['job_id']}).scalar_one()=='queued'
