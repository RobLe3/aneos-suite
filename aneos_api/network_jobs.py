"""Registered resumable population worker; hosting/authenticated REST are separate.

Provider calls and numerical work run outside SQL transactions. Stored completion
means a request-bound report was committed, never calibrated science or origin.
"""
from dataclasses import asdict
from datetime import date, timedelta, datetime, timezone
from hashlib import sha256
import json
import math
import threading
import time

from .job_store import JobStore, LostLease, JobIntegrityError, JobStorageUnavailable
from aneos_core.data.models import NEOData, CloseApproach
from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy
from aneos_core.pattern_analysis.session import (
    NetworkAnalysisSession, PatternAnalysisConfig, verify_network_report)

NETWORK_KIND = 'network-evidence-job-v1'
REQUEST_SCHEMA = 'network-execution-request-v1'
RESULT_SCHEMA = 'network-execution-result-v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return sha256(_json(value).encode()).hexdigest()


def _copy(value):
    return json.loads(_json(value))


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a SHA-256 profile identifier')
    return value


def current_source_profile():
    """Fingerprint configuration, not plaintext credentials or source authentication."""
    from aneos_core.config.settings import get_config
    return _hash({'adapter_profile':'data-fetcher-window-v1', 'api':asdict(get_config().api)})


def _default_source():
    from aneos_core.config.settings import get_config
    from aneos_core.data.fetcher import DataFetcher
    return DataFetcher(source_priority=list(get_config().api.data_sources_priority))


def _policy_id(policy):
    return ScreeningAnalysisService(policy=policy).analyze_raw('contract-check', {}).policy_id


def build_network_request(designations, *, config=None, policy=None, source_profile_id=None, as_of_day=None):
    config = config or PatternAnalysisConfig()
    policy = policy or ScreeningPolicy()
    if type(config) is not PatternAnalysisConfig or type(policy) is not ScreeningPolicy:
        raise ValueError('Expected versioned configuration and policy')
    if (not isinstance(designations, list) or not 1 <= len(designations) <= 500 or
        any(not isinstance(d, str) or not d or d != d.strip() or len(d) > 80 or
            any(ord(c) < 32 for c in d) for d in designations)):
        raise ValueError('Expected 1–500 bounded, nonempty designations')
    if len(designations) > config.max_objects:
        raise ValueError('Requested batch exceeds its configured cap; split it explicitly')
    anchor = as_of_day if as_of_day is not None else datetime.now(timezone.utc).date().isoformat()
    # These are explicit calendar query labels, not UTC-to-TDB observation epochs.
    end = date.fromisoformat(anchor)
    if end.isoformat() != anchor:
        raise ValueError('Use an ISO calendar date')
    request = {
        'schema_version':REQUEST_SCHEMA, 'designations':list(designations),
        'configuration':asdict(config), 'screening_policy':asdict(policy),
        'screening_policy_id':_policy_id(policy),
        'source_profile_id':_digest(source_profile_id if source_profile_id is not None else current_source_profile()),
        'historical_query':{'date_min':(end-timedelta(days=365*config.historical_years)).isoformat(),
            'date_max':anchor, 'time_scale':'TDB_calendar_query_labels',
            'horizon_semantics':'historical_years_times_365_days', 'body':'Earth', 'limit':10},
        'report_engine':'network-descriptors-v1',
    }
    return _copy(request)


def validate_network_request(request):
    if not isinstance(request, dict) or request.get('schema_version') != REQUEST_SCHEMA:
        raise ValueError('Unsupported network execution request')
    try:
        expected = build_network_request(request['designations'],
            config=PatternAnalysisConfig(**request['configuration']),
            policy=ScreeningPolicy(**request['screening_policy']),
            source_profile_id=request['source_profile_id'],
            as_of_day=request['historical_query']['date_max'])
        if _json(expected) != _json(request):
            raise ValueError('Request version, configuration or scientific policy differs')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid network execution request') from exc
    return expected


class AcquisitionUnavailable(RuntimeError):
    def __init__(self, retry_at=None):
        super().__init__('Acquisition unavailable')
        self.retry_at = retry_at


class SourceConfigurationChanged(ValueError): pass
class WorkerStopping(RuntimeError): pass
class AttemptDeadline(RuntimeError): pass


def _input_record(record, request, designation, *, history=False):
    schema = 'network-history-checkpoint-v1' if history else 'network-input-checkpoint-v1'
    if (not isinstance(record, dict) or record.get('schema_version') != schema or
        record.get('request_id') != _hash(request) or record.get('designation') != designation):
        raise ValueError('Frozen acquisition does not match request')
    if history:
        if set(record) != {'schema_version','request_id','designation','approaches','metadata'}:
            raise ValueError('Invalid historical checkpoint fields')
        meta = record['metadata']
        if not isinstance(meta,dict) or not isinstance(meta.get('query'),dict):
            raise ValueError('Historical query metadata unavailable')
        query = meta['query']; window = request['historical_query']
        if (query.get('des') != designation or query.get('date-min') != window['date_min'] or
            query.get('date-max') != window['date_max'] or query.get('body') != 'Earth' or
            query.get('limit') != 10 or meta.get('completeness_status') not in ('complete','partial')):
            raise ValueError('Historical query binding or coverage unavailable')
        if not isinstance(record['approaches'], list):
            raise ValueError('Historical records must be a list')
        count,total=meta.get('count'),meta.get('total')
        if (type(count) is not int or type(total) is not int or count!=len(record['approaches']) or
            not 0<=count<=10 or total<count or
            meta['completeness_status']!=('complete' if count==total else 'partial')):
            raise ValueError('Historical count/coverage differs')
        _digest(meta.get('canonical_response_sha256'))
        for raw in record['approaches']:
            ca = CloseApproach.from_dict(raw)
            if ca.designation != designation:
                raise ValueError('Historical object identity differs')
    else:
        if set(record) != {'schema_version','request_id','designation','neo_snapshot'}:
            raise ValueError('Invalid input checkpoint fields')
        snapshot=record['neo_snapshot']
        if not isinstance(snapshot,dict):raise ValueError('Invalid input snapshot')
        n = NEOData.from_dict(snapshot)
        if set(snapshot)!=set(n.to_dict()):raise ValueError('Incomplete or unknown input snapshot fields')
        for field in ('created_at','updated_at'):
            if not isinstance(snapshot[field],str):raise ValueError('Snapshot timestamps unavailable')
            datetime.fromisoformat(snapshot[field].replace('Z','+00:00'))
        if n.designation != designation:
            raise ValueError('Input object identity differs')
        ScreeningAnalysisService(policy=ScreeningPolicy(**request['screening_policy'])).analyze_neo(n)
    _json(record)
    return record


def _compose(record, history):
    neo = NEOData.from_dict(record['neo_snapshot'])
    if history is not None:
        # Preserve raw original input separately; reporting receives an owned copy.
        neo.close_approaches = [CloseApproach.from_dict(raw) for raw in history['approaches']]
        neo.evidence_metadata['historical_query_evidence'] = _copy(history['metadata'])
    return neo


def _result_from_records(request, records):
    unique = list(dict.fromkeys(request['designations']))
    if not isinstance(records, list) or len(records) != len(unique):
        raise ValueError('Incomplete unique acquisition register')
    objects = {}
    for designation, entry in zip(unique, records):
        if set(entry) != {'designation','input','history'} or entry['designation'] != designation:
            raise ValueError('Acquisition identity/order differs')
        raw = _input_record(entry['input'], request, designation)
        history = entry['history']
        if request['configuration']['harmonics']:
            _input_record(history, request, designation, history=True)
        elif history is not None:
            raise ValueError('Unrequested history was attached')
        objects[designation] = _compose(raw, history)
    report = NetworkAnalysisSession(PatternAnalysisConfig(**request['configuration']),
        policy=ScreeningPolicy(**request['screening_policy'])).run(
            [objects[d] for d in request['designations']])
    return {'schema_version':RESULT_SCHEMA, 'request_id':_hash(request),
            'acquisitions':_copy(records), 'scientific_report':report,
            'execution_meaning':'request_bound_report_not_scientific_validation'}


def verify_network_job_result(request, result):
    request = validate_network_request(request)
    if not isinstance(result, dict) or result.get('schema_version') != RESULT_SCHEMA:
        raise ValueError('Unsupported network execution result')
    try:
        verify_network_report(result['scientific_report'])
        expected = _result_from_records(request, result['acquisitions'])
        if _json(expected) != _json(result):
            raise ValueError('Execution result does not match frozen request/acquisitions')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid network execution result') from exc
    return expected


def _verify_committable(request, checkpoints, result):
    """Bind reported acquisitions to SQL checkpoints, not just their own hashes."""
    unique=list(dict.fromkeys(request['designations']))
    keys={'result'} | {f'neo:{i}' for i in range(len(unique))}
    if request['configuration']['harmonics']:keys|={f'history:{i}' for i in range(len(unique))}
    if set(checkpoints)!=keys:raise ValueError('Missing or unexpected execution checkpoints')
    records=[{'designation':d,'input':checkpoints[f'neo:{i}'],
              'history':checkpoints[f'history:{i}'] if request['configuration']['harmonics'] else None}
             for i,d in enumerate(unique)]
    expected=_result_from_records(request,records)
    if _json(result)!=_json(expected) or _json(checkpoints['result'])!=_json(expected):
        raise ValueError('Result is not bound to committed input checkpoints')
    return verify_network_job_result(request,result)


class NetworkJobHandler:
    """Only successful validated acquisitions are immutable input checkpoints.

    A transient failure leaves missing work for bounded retry; already committed
    measurements are never refreshed or replaced. No failure is frozen as zero.
    """
    def __init__(self, *, source_factory=_default_source, source_profile=current_source_profile):
        self.source_factory = source_factory
        self.source_profile = source_profile

    def run(self, context):
        request = validate_network_request(context.lease.request)
        checkpoints = context.checkpoints()
        if 'result' in checkpoints:
            return _verify_committable(request,checkpoints,checkpoints['result'])
        source, acquisitions = None, []
        unique = list(dict.fromkeys(request['designations']))
        allowed = {'result'} | {f'{kind}:{i}' for kind in ('neo','history') for i in range(len(unique))}
        if set(checkpoints) - allowed:
            raise ValueError('Unexpected frozen checkpoint')
        def acquire(call):
            nonlocal source
            context.check()
            try:
                if source is None:
                    if self.source_profile() != request['source_profile_id']:
                        raise SourceConfigurationChanged('Acquisition profile changed')
                    source = self.source_factory()
                result = call(source)
            except (LostLease, WorkerStopping, AttemptDeadline, SourceConfigurationChanged):
                raise
            except Exception as exc:
                raise AcquisitionUnavailable(getattr(exc,'retry_at',None)) from exc
            context.check()
            return result
        for index, designation in enumerate(unique):
            context.check()
            key = f'neo:{index}'
            raw = checkpoints.get(key)
            if raw is None:
                n = acquire(lambda src: src.fetch_neo_data(designation))
                if n is None:
                    raise AcquisitionUnavailable()
                if not isinstance(n, NEOData) or n.designation != designation:
                    raise ValueError('Acquired object identity/type differs')
                raw = {'schema_version':'network-input-checkpoint-v1', 'request_id':_hash(request),
                       'designation':designation, 'neo_snapshot':_copy(n.to_dict())}
                _input_record(raw,request,designation)
                context.checkpoint(key,raw)
            else:
                _input_record(raw,request,designation)
            history = None
            if request['configuration']['harmonics']:
                key = f'history:{index}';history = checkpoints.get(key)
                if history is None:
                    window = request['historical_query']
                    events = acquire(lambda src: src.fetch_approaches_between(designation,window['date_min'],window['date_max']))
                    if events is None:
                        raise AcquisitionUnavailable()
                    from aneos_core.data.sources.cad import CloseApproachQuery
                    if not isinstance(events,CloseApproachQuery) or any(not isinstance(ca,CloseApproach) for ca in events):
                        raise ValueError('Historical provider result is not typed/query-bound')
                    history = {'schema_version':'network-history-checkpoint-v1','request_id':_hash(request),
                               'designation':designation,'approaches':[ca.to_dict() for ca in events],
                               'metadata':_copy(events.query_metadata)}
                    _input_record(history,request,designation,history=True)
                    context.checkpoint(key,history)
                else:
                    _input_record(history,request,designation,history=True)
            acquisitions.append({'designation':designation,'input':raw,'history':history})
        context.check()
        result = _result_from_records(request,acquisitions)
        verify_network_job_result(request,result)
        context.checkpoint('result',result)
        return result


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


class NetworkWorker:
    """Explicit registered worker. Does not start on import or bind REST routes.

    run_once executes real work; serve is a blocking, cooperatively stopped loop
    for a future managed host. A provider call cannot be forcibly interrupted;
    stale/cancelled work is fenced at the next boundary and at every commit.
    """
    def __init__(self, store, *, handler=None, worker_id='network-worker',
                 lease_seconds=60, retry_seconds=5, attempt_seconds=1800, stop_event=None):
        if not isinstance(store,JobStore):raise TypeError('Expected JobStore')
        for name,value,low,high in (('lease',lease_seconds,.2,3600),('retry',retry_seconds,.001,300),
                                    ('attempt',attempt_seconds,.01,86400)):
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'Invalid {name} policy')
        self.store=store;self.handler=handler or NetworkJobHandler();self.worker_id=worker_id
        self.lease_seconds=lease_seconds;self.retry_seconds=retry_seconds;self.attempt_seconds=attempt_seconds
        self.stop=stop_event if stop_event is not None else threading.Event()
        self.last_outcome='not_started'

    def run_once(self):
        if self.stop.is_set():return 'stopped'
        lease=self.store.claim(self.worker_id,lease_seconds=self.lease_seconds,kinds=[NETWORK_KIND])
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
        thread=threading.Thread(target=heartbeat,name='aneos-network-lease',daemon=False)
        thread.start()
        error=None;result=None
        try:
            validate_network_request(lease.request)
            result=self.handler.run(context)
            _verify_committable(lease.request,context.checkpoints(),result)
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

    @staticmethod
    def _invalid_request(request):
        try:validate_network_request(request);return False
        except (ValueError,TypeError):return True

    def serve(self, *, poll_seconds=.5):
        if type(poll_seconds) not in (int,float) or not math.isfinite(poll_seconds) or not .01<=poll_seconds<=60:
            raise ValueError('Invalid idle poll interval')
        while not self.stop.is_set():
            if self.run_once()=='idle':self.stop.wait(poll_seconds)
