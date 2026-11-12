"""App-owned, explicitly started SQL network execution runtime (ADR-079)."""
import logging
import math
import threading
from .job_store import JobStore
from .network_jobs import NetworkWorker, NetworkJobHandler

logger=logging.getLogger(__name__)


class NetworkRuntime:
    def __init__(self,sessions,*,handler=None,capacity=128,lease_seconds=60,shutdown_seconds=30,enabled=True):
        if type(enabled) is not bool:raise ValueError('enabled must be boolean')
        if type(shutdown_seconds) not in (int,float) or not math.isfinite(shutdown_seconds) or not 0<shutdown_seconds<=300:
            raise ValueError('Invalid worker shutdown budget')
        self.store=JobStore(sessions,capacity=capacity)
        self.worker=NetworkWorker(self.store,handler=handler or NetworkJobHandler(),lease_seconds=lease_seconds)
        self.enabled=enabled;self.shutdown_seconds=shutdown_seconds
        self._thread=None;self._accepting=False;self.error=None

    @property
    def ready(self):
        return bool(self._accepting and self._thread and self._thread.is_alive() and self.error is None)

    def start(self):
        if self._thread is not None:raise RuntimeError('Network runtime already started')
        self.store.initialize_schema()
        if not self.enabled:return
        started=threading.Event()
        def run():
            started.set()
            try:self.worker.serve(poll_seconds=.05)
            except BaseException:
                self.error='NETWORK_WORKER_STOPPED';logger.exception('Network worker stopped')
            finally:self._accepting=False
        self._thread=threading.Thread(target=run,name='aneos-network-worker',daemon=False)
        self._accepting=True;self._thread.start()
        if not started.wait(5) or not self.ready:
            self.close();raise RuntimeError('Network worker startup unconfirmed')

    def close(self):
        self._accepting=False;self.worker.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.shutdown_seconds)
            if self._thread.is_alive():
                self.error='NETWORK_WORKER_SHUTDOWN_UNCONFIRMED'
                raise RuntimeError('Network worker shutdown unconfirmed; process supervision required')

    def health(self):
        return {'ready':self.ready,'enabled':self.enabled,'error':self.error,
                'last_outcome':self.worker.last_outcome,'execution_storage':'sql_durable'}


def default_runtime():
    import os
    from .database import SessionLocal
    flag=os.getenv('ANEOS_NETWORK_WORKER_ENABLED','true').lower()
    if flag not in ('true','false'):raise ValueError('Use true/false for ANEOS_NETWORK_WORKER_ENABLED')
    return NetworkRuntime(SessionLocal,enabled=flag=='true')
