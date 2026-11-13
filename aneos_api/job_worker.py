"""Shared fenced execution driver; storage and scientific contracts remain separate."""
import math
import threading
import time
from .job_store import JobStore, LostLease, JobIntegrityError, JobStorageUnavailable

class AcquisitionUnavailable(RuntimeError):
    def __init__(self, retry_at=None):
        super().__init__('Acquisition unavailable')
        self.retry_at = retry_at


class SourceConfigurationChanged(ValueError): pass
class WorkerStopping(RuntimeError): pass
class AttemptDeadline(RuntimeError): pass


class _Execution:
    def __init__(self, store, lease, stop, deadline):
        self.store=store;self.lease=lease;self.stop=stop;self.deadline=deadline
        self.heartbeat_error=None

    def check(self):
        if self.stop.is_set():raise WorkerStopping('Worker is stopping')
        if self.heartbeat_error is not None:raise self.heartbeat_error
        if time.monotonic() >= self.deadline:raise AttemptDeadline('Attempt deadline passed')
        self.store.checkpoints(self.lease)  # Fence against cancellation/reclaim at each boundary.

    def checkpoints(self):
        self.check()
        return self.store.checkpoints(self.lease)

    def checkpoint(self,key,payload):
        self.check()
        self.store.checkpoint(self.lease,key,payload)


class DurableWorker:
    """Explicit registered worker. Does not start on import or bind REST routes.

    run_once executes real work; serve is a blocking, cooperatively stopped loop
    for a future managed host. A provider call cannot be forcibly interrupted;
    stale/cancelled work is fenced at the next boundary and at every commit.
    """
    job_kind = None

    def validate_request(self, request):
        raise NotImplementedError('A registered request contract is required')

    def verify_result(self, request, checkpoints, result):
        raise NotImplementedError('A registered result contract is required')

    def __init__(self, store, *, handler, worker_id='job-worker',
                 lease_seconds=60, retry_seconds=5, attempt_seconds=1800, stop_event=None):
        if not isinstance(store,JobStore):raise TypeError('Expected JobStore')
        if not isinstance(self.job_kind,str) or not self.job_kind:raise ValueError('Registered job kind required')
        for name,value,low,high in (('lease',lease_seconds,.2,3600),('retry',retry_seconds,.001,300),
                                    ('attempt',attempt_seconds,.01,86400)):
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'Invalid {name} policy')
        self.store=store;self.handler=handler;self.worker_id=worker_id
        self.lease_seconds=lease_seconds;self.retry_seconds=retry_seconds;self.attempt_seconds=attempt_seconds
        self.stop=stop_event if stop_event is not None else threading.Event()
        self.last_outcome='not_started'

    def run_once(self, *, job_id=None):
        if self.stop.is_set():return 'stopped'
        lease=self.store.claim(self.worker_id,lease_seconds=self.lease_seconds,kinds=[self.job_kind],job_id=job_id)
        if lease is None:self.last_outcome='idle';return 'idle'
        context=_Execution(self.store,lease,self.stop,time.monotonic()+self.attempt_seconds)
        ended=threading.Event()
        def heartbeat():
            while not ended.wait(self.lease_seconds/4):
                if self.stop.is_set():
                    context.heartbeat_error=WorkerStopping('Worker is stopping');return
                if time.monotonic()>=context.deadline:
                    context.heartbeat_error=AttemptDeadline('Attempt deadline passed');return
                try:self.store.renew(lease,lease_seconds=self.lease_seconds)
                except Exception as exc:
                    context.heartbeat_error=exc;return
        thread=threading.Thread(target=heartbeat,name='aneos-job-lease',daemon=False)
        thread.start()
        error=None;result=None
        try:
            self.validate_request(lease.request)
            result=self.handler.run(context)
            self.verify_result(lease.request,context.checkpoints(),result)
            context.check()
        except Exception as exc:error=exc
        finally:
            ended.set();thread.join()
        if error is None:
            try:
                context.check()
                self.store.finish(lease,result)
                self.last_outcome='completed';return 'completed'
            except (LostLease,WorkerStopping,AttemptDeadline) as exc:error=exc
        if isinstance(error,LostLease):self.last_outcome='lease_lost';return 'lease_lost'
        if isinstance(error,JobStorageUnavailable):raise error
        retry=isinstance(error,(AcquisitionUnavailable,WorkerStopping,AttemptDeadline))
        code=('SOURCE_CONFIGURATION_CHANGED' if isinstance(error,SourceConfigurationChanged) else
              'ACQUISITION_UNAVAILABLE' if isinstance(error,AcquisitionUnavailable) else
              'WORKER_STOPPED' if isinstance(error,WorkerStopping) else
              'ATTEMPT_DEADLINE' if isinstance(error,AttemptDeadline) else
              'CHECKPOINT_INVALID' if isinstance(error,JobIntegrityError) else
              'INVALID_REQUEST' if result is None and self._invalid_request(lease.request) else
              'INVALID_EVIDENCE' if isinstance(error,(ValueError,TypeError,KeyError)) else 'WORKER_INTERNAL_ERROR')
        retry_at=None
        if retry:
            now=self.store.clock();retry_at=now+min(300,self.retry_seconds*2**min(lease.attempt-1,10))
            hint=getattr(error,'retry_at',None)
            if type(hint) in (int,float) and math.isfinite(hint):retry_at=max(retry_at,hint)
        try:self.store.fail(lease,code,retry_at=retry_at)
        except LostLease:self.last_outcome='lease_lost';return 'lease_lost'
        self.last_outcome='failure_recorded';return 'failure_recorded'

    def _invalid_request(self,request):
        try:self.validate_request(request);return False
        except (ValueError,TypeError):return True

    def serve(self, *, poll_seconds=.5):
        if type(poll_seconds) not in (int,float) or not math.isfinite(poll_seconds) or not .01<=poll_seconds<=60:
            raise ValueError('Invalid idle poll interval')
        while not self.stop.is_set():
            if self.run_once()=='idle':self.stop.wait(poll_seconds)
