"""App-owned registered SQL execution host (ADR-079/097), never import-started."""
import logging
import math
import threading
from .job_store import JobStore
from .network_jobs import NetworkWorker, NetworkJobHandler

logger=logging.getLogger(__name__)


class NetworkRuntime:
    def __init__(self,sessions,*,handler=None,capacity=128,lease_seconds=60,shutdown_seconds=30,enabled=True,
                 batch_enabled=False,batch_handler=None):
        if type(enabled) is not bool or type(batch_enabled) is not bool:raise ValueError('enabled flags must be boolean')
        if type(shutdown_seconds) not in (int,float) or not math.isfinite(shutdown_seconds) or not 0<shutdown_seconds<=300:
            raise ValueError('Invalid worker shutdown budget')
        self.store=JobStore(sessions,capacity=capacity)
        self.worker=NetworkWorker(self.store,handler=handler or NetworkJobHandler(),lease_seconds=lease_seconds)
        from .batch_jobs import BatchWorker, BatchJobHandler
        self.batch_worker=BatchWorker(self.store,handler=batch_handler or BatchJobHandler(),
            lease_seconds=lease_seconds,stop_event=self.worker.stop)
        self.enabled=enabled;self.batch_enabled=batch_enabled;self.shutdown_seconds=shutdown_seconds
        self._thread=None;self._accepting=False;self.error=None

    @property
    def ready(self):
        return bool(self.enabled and self._host_ready)

    @property
    def _host_ready(self):
        return bool(self._accepting and self._thread and self._thread.is_alive() and self.error is None)

    @property
    def batch_ready(self):
        return bool(self.batch_enabled and self._host_ready)

    def start(self):
        if self._thread is not None:raise RuntimeError('Network runtime already started')
        self.store.initialize_schema()
        if not self.enabled and not self.batch_enabled:return
        started=threading.Event()
        def run():
            started.set()
            try:
                if not self.batch_enabled:
                    self.worker.serve(poll_seconds=.05)
                else:
                    # One app-owned host, shared capacity and fences. This is not
                    # concurrent job throughput; a blocked call can delay both kinds.
                    while not self.worker.stop.is_set():
                        busy=self.enabled and self.worker.run_once()!='idle'
                        if not self.worker.stop.is_set():
                            busy=(self.batch_worker.run_once()!='idle') or busy
                        if not busy:self.worker.stop.wait(.05)
            except BaseException:
                self.error='EXECUTION_WORKER_STOPPED' if self.batch_enabled else 'NETWORK_WORKER_STOPPED'
                logger.exception('Persistent execution host stopped')
            finally:self._accepting=False
        self._thread=threading.Thread(target=run,
            name='aneos-execution-host' if self.batch_enabled else 'aneos-network-worker',daemon=False)
        self._accepting=True;self._thread.start()
        if not started.wait(5) or not self._host_ready:
            self.close();raise RuntimeError('Execution host startup unconfirmed')

    def close(self):
        self._accepting=False;self.worker.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.shutdown_seconds)
            if self._thread.is_alive():
                self.error='EXECUTION_WORKER_SHUTDOWN_UNCONFIRMED' if self.batch_enabled else 'NETWORK_WORKER_SHUTDOWN_UNCONFIRMED'
                raise RuntimeError('Persistent worker shutdown unconfirmed; process supervision required')

    def health(self):
        return {'ready':self.ready,'enabled':self.enabled,'error':self.error,
                'last_outcome':self.worker.last_outcome,'execution_storage':'sql_durable',
                'batch_ready':self.batch_ready,'batch_enabled':self.batch_enabled,
                'batch_last_outcome':self.batch_worker.last_outcome}


def default_runtime():
    import os
    from .database import SessionLocal
    flag=os.getenv('ANEOS_NETWORK_WORKER_ENABLED','true').lower()
    if flag not in ('true','false'):raise ValueError('Use true/false for ANEOS_NETWORK_WORKER_ENABLED')
    batch_flag=os.getenv('ANEOS_BATCH_WORKER_ENABLED','true').lower()
    if batch_flag not in ('true','false'):raise ValueError('Use true/false for ANEOS_BATCH_WORKER_ENABLED')
    return NetworkRuntime(SessionLocal,enabled=flag=='true',batch_enabled=batch_flag=='true')
