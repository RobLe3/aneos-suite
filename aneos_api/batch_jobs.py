"""Registered canonical batch jobs on the existing queue; no import-time execution."""
import asyncio
import base64
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import os
import re

from aneos_core.analysis.batch_reporting import parse_input,build_batch_report,verify_batch_report
from aneos_core.analysis.reporting import build_screening_report,verify_screening_report
from aneos_core.analysis.screening import ScreeningAnalysisService,ScreeningPolicy
from aneos_core.data.enrichment import clone,fingerprint
from .job_store import JobStore,JobIntegrityError,IdempotencyConflict
from .job_worker import DurableWorker,AcquisitionUnavailable,SourceConfigurationChanged

KIND='canonical-batch-job-v1'
REQUEST_SCHEMA='canonical-batch-request-v1'
RESULT_SCHEMA='canonical-batch-result-v1'
LOCAL_OWNER='aneos-primary-menu-batch-v1'  # Trusted local application scope, not authentication.
RETRYABLE={'source_unavailable','service_unavailable','evidence_persistence_unavailable','failed','unavailable'}


def current_source_profile():
    from .network_jobs import current_source_profile as profile
    return profile()


def default_pipeline():
    from aneos_core.analysis.pipeline import AnalysisPipeline
    return AnalysisPipeline()


def build_request(source_input,*,source_profile_id=None,max_attempts=3,force_refresh=False,created_at=None,execution=None,source_attempts=1,response_options=None):
    source=clone(source_input)
    if parse_input(base64.b64decode(source['content_base64'],validate=True),source['name'])!=source:
        raise ValueError('Original batch input does not reproduce')
    if type(max_attempts) is not int or not 1<=max_attempts<=5:
        raise ValueError('Batch attempts must be 1..5')
    if type(source_attempts) is not int or not 1<=source_attempts<=max_attempts:
        raise ValueError('Source attempts must be within the execution budget')
    if type(force_refresh) is not bool:raise ValueError('Refresh policy must be boolean')
    profile=current_source_profile() if source_profile_id is None else source_profile_id
    if not isinstance(profile,str) or not re.fullmatch('[0-9a-f]{64}',profile):
        raise ValueError('Source configuration fingerprint required')
    stamp=created_at or datetime.now(timezone.utc).isoformat()
    parsed=datetime.fromisoformat(stamp)
    if parsed.utcoffset() is None:raise ValueError('Report timestamp needs an explicit offset')
    from aneos_core.analysis.pipeline import PIPELINE_VERSION
    if execution is None:
        from aneos_core.config.settings import get_config
        config=get_config()
        execution={'max_workers':config.max_workers,'timeout_seconds':config.analysis_timeout,
                   'parallel':config.analysis_parallel,'queue_size':config.analysis_queue_size}
    if not isinstance(execution,dict) or set(execution)!={'max_workers','timeout_seconds','parallel','queue_size'}:
        raise ValueError('Expected frozen execution configuration')
    for key,high in [('max_workers',64),('timeout_seconds',86400),('queue_size',10000)]:
        if type(execution[key]) is not int or not 1<=execution[key]<=high:raise ValueError('Invalid execution budget')
    if type(execution['parallel']) is not bool:raise ValueError('Invalid parallel policy')
    if response_options is not None:
        if (not isinstance(response_options,dict) or set(response_options)!={'schema_version','include_raw_data'} or
            response_options['schema_version']!='batch-http-response-v1' or type(response_options['include_raw_data']) is not bool):
            raise ValueError('Invalid batch HTTP response options')
    policy=ScreeningPolicy()
    result = {'schema_version':REQUEST_SCHEMA,'source_input':source,'source_profile_id':profile,
        'max_attempts':max_attempts,'source_attempts':source_attempts,'force_refresh':force_refresh,'created_at':stamp,'execution':clone(execution),
        'timestamp_role':'local_submission_not_observation','pipeline_version':PIPELINE_VERSION,
        'screening_policy':asdict(policy),
        'policy_id':ScreeningAnalysisService(policy).analyze_raw('job-contract',{}).policy_id}
    if response_options is not None:result['response_options']=clone(response_options)
    return clone(result)


def validate_request(request):
    try:
        expected=build_request(request['source_input'],source_profile_id=request['source_profile_id'],
            max_attempts=request['max_attempts'],force_refresh=request['force_refresh'],created_at=request['created_at'],
            execution=request['execution'],source_attempts=request['source_attempts'],response_options=request.get('response_options'))
        if expected!=request:raise ValueError('Batch request policy/version differs')
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise ValueError('Invalid canonical batch request') from exc
    return expected


def _names(request):
    return list(dict.fromkeys(r['designation'] for r in request['source_input']['requests']))


def _records(request,checkpoints):
    """Read all immutable attempts; never let a later row replace ready evidence."""
    names=_names(request);attempts={};selected={};ready=set();failures={}
    for key,record in checkpoints.items():
        if key in ('result','assembly'):continue
        match=re.fullmatch(r'object:(\d+):attempt:(\d+)',key)
        if not match:raise ValueError('Unknown batch checkpoint')
        index,attempt=map(int,match.groups())
        if key!=f'object:{index}:attempt:{attempt}':raise ValueError('Noncanonical checkpoint key')
        if not 0<=index<len(names) or not 1<=attempt<=request['max_attempts']:
            raise ValueError('Checkpoint index/attempt outside request')
        expected_keys={'schema_version','request_id','designation','attempt','report'}
        if (set(record)!=expected_keys or record['schema_version']!='batch-object-checkpoint-v1' or
            record['request_id']!=fingerprint(request) or record['designation']!=names[index] or
            type(record['attempt']) is not int or record['attempt']!=attempt):
            raise ValueError('Batch checkpoint binding differs')
        verify_screening_report(record['report'])
        if record['report']['designation']!=names[index]:raise ValueError('Report/object identity differs')
        attempts[(index,attempt)]=record['report']
    for (index,attempt),report in sorted(attempts.items()):
        name=names[index]
        if name in ready:raise ValueError('Ready object was reacquired/replaced')
        selected[name]=report
        if report['status'] in RETRYABLE:failures[name]=failures.get(name,0)+1
        if (report['status'] not in RETRYABLE or attempt==request['max_attempts'] or
            failures.get(name,0)>=request['source_attempts']):
            ready.add(name)
    if 'assembly' in checkpoints and ready!=set(names):
        raise ValueError('Assembly timestamp exists before all requests are ready')
    return selected,ready


def _assembly_time(request,checkpoints):
    clock=checkpoints.get('assembly')
    if (not isinstance(clock,dict) or set(clock)!={'schema_version','request_id','created_at','timestamp_role'} or
        clock['schema_version']!='batch-report-clock-v1' or clock['request_id']!=fingerprint(request) or
        clock['timestamp_role']!='local_report_assembly_not_observation'):
        raise ValueError('Batch assembly clock is missing or unbound')
    stamp=clock['created_at']
    if not isinstance(stamp,str) or datetime.fromisoformat(stamp).utcoffset() is None:
        raise ValueError('Batch assembly clock needs an explicit offset')
    return stamp


def compose_result(request,checkpoints):
    request=validate_request(request);reports,ready=_records(request,checkpoints)
    if ready!=set(_names(request)):raise ValueError('Batch has unprocessed or retryable requests')
    batch=build_batch_report(request['source_input'],reports,created_at=_assembly_time(request,checkpoints))
    failed=sum(bool(r['report']['errors']) for r in batch['requests'])
    return {'schema_version':RESULT_SCHEMA,'request_id':fingerprint(request),'batch_report':batch,
        'processing_status':'completed_with_unavailable_or_invalid_requests' if failed else 'completed',
        'scientific_status':'not_independently_validated'}


def verify_result(request,checkpoints,result):
    expected=compose_result(request,checkpoints)
    if result!=expected:raise ValueError('Final batch does not reproduce committed report checkpoints')
    if 'result' in checkpoints and checkpoints['result']!=result:
        raise ValueError('Stored final batch differs')
    return clone(expected)


def _report(result):
    # Public execution reports retain codes/receipts, not arbitrary exception/credential strings.
    if result.errors:
        result=copy.copy(result)
        result.errors=[f'{result.status.upper()}: no successful source assessment; inspect acquisition history.']
        result.data_quality={'reliability_score':None,'execution_status':result.status}
        result.analysis_metadata={k:v for k,v in result.analysis_metadata.items()
                                  if k in ('pipeline_version','five_sigma_claim_permitted','acquisition_receipt')}
    return build_screening_report(result)


class BatchJobHandler:
    def __init__(self,*,pipeline_factory=default_pipeline,source_profile=current_source_profile):
        self.pipeline_factory=pipeline_factory;self.source_profile=source_profile

    def run(self,context):
        request=validate_request(context.lease.request)
        if request['max_attempts']!=context.lease.max_attempts:
            raise JobIntegrityError('Queue attempt limit differs from frozen request')
        checkpoints=context.checkpoints()
        if 'result' in checkpoints:return verify_result(request,checkpoints,checkpoints['result'])
        reports,ready=_records(request,checkpoints)
        unresolved=[n for n in _names(request) if n not in ready]
        if unresolved:
            if self.source_profile()!=request['source_profile_id']:
                raise SourceConfigurationChanged('Source configuration changed')
            pipeline=self.pipeline_factory()
            if (not isinstance(pipeline.screening_service,ScreeningAnalysisService) or
                pipeline.screening_service.analyze_raw('job-contract',{}).policy_id!=request['policy_id']):
                raise ValueError('Pipeline scientific policy differs')
            execution=request['execution']
            pipeline.pipeline_config.max_workers=execution['max_workers']
            pipeline.pipeline_config.timeout_seconds=execution['timeout_seconds']
            pipeline.analysis_parallel=execution['parallel'];pipeline.analysis_queue_size=execution['queue_size']
            indexes={n:i for i,n in enumerate(_names(request))}
            def before_item(name):
                context.check()
                if self.source_profile()!=request['source_profile_id']:
                    raise SourceConfigurationChanged('Source configuration changed')
            def checkpoint(result):
                context.check()
                if result.designation not in unresolved:raise ValueError('Unexpected batch result identity')
                key=f'object:{indexes[result.designation]}:attempt:{context.lease.attempt}'
                record={'schema_version':'batch-object-checkpoint-v1','request_id':fingerprint(request),
                        'designation':result.designation,'attempt':context.lease.attempt,'report':_report(result)}
                try:context.checkpoint(key,record)
                except BaseException as exc:
                    context.heartbeat_error=exc
                    raise
            asyncio.run(pipeline.analyze_batch(unresolved,force_refresh=request['force_refresh'],
                        result_callback=checkpoint,before_item_callback=before_item))
        context.check()
        checkpoints=context.checkpoints();reports,ready=_records(request,checkpoints)
        if ready!=set(_names(request)):raise AcquisitionUnavailable()
        if 'assembly' not in checkpoints:
            context.checkpoint('assembly',{'schema_version':'batch-report-clock-v1',
                'request_id':fingerprint(request),'created_at':datetime.now(timezone.utc).isoformat(),
                'timestamp_role':'local_report_assembly_not_observation'})
            checkpoints=context.checkpoints()
        result=compose_result(request,checkpoints)
        context.checkpoint('result',result)
        return result


class BatchWorker(DurableWorker):
    job_kind=KIND
    def __init__(self,store,*,handler=None,worker_id='batch-worker',**kwargs):
        super().__init__(store,handler=handler or BatchJobHandler(),worker_id=worker_id,**kwargs)
    def validate_request(self,request):return validate_request(request)
    def verify_result(self,request,checkpoints,result):return verify_result(request,checkpoints,result)


class BatchRuntime:
    """Explicit synchronous local host; no fallback to unacknowledged session work."""
    def __init__(self,store,*,handler=None,owner_scope=LOCAL_OWNER,**worker_options):
        self.store=store;self.owner_scope=owner_scope
        self.worker=BatchWorker(store,handler=handler,**worker_options)
        self.engine=None;self.read_only=False

    def submit(self,source_input,key,*,max_attempts=3,source_attempts=1):
        if self.read_only:raise ValueError('Read-only job inspection cannot submit new work')
        existing=self.store.find_submission(self.owner_scope,KIND,key)
        request=build_request(source_input,max_attempts=max_attempts,source_attempts=source_attempts,
            source_profile_id=self.worker.handler.source_profile(),
            created_at=existing['request']['created_at'] if existing else None)
        try:
            job=self.store.submit(owner_scope=self.owner_scope,kind=KIND,request=request,
                                  idempotency_key=key,max_attempts=max_attempts)
        except IdempotencyConflict:
            # Concurrent first submissions may sample different local creation times.
            # Compare every other frozen field; do not mistake a timestamp for new evidence.
            old=self.store.find_submission(self.owner_scope,KIND,key)
            if old is None:raise
            candidate=clone(request);candidate['created_at']=old['request']['created_at']
            if candidate!=old['request'] or old['max_attempts']!=max_attempts:raise
            job=old
        return self.read('job:'+job['job_id'])

    def read(self,token):
        if not isinstance(token,str) or not token or len(token)>132 or any(ord(c)<32 for c in token):
            raise ValueError('Invalid batch recovery token')
        mode='auto'
        for prefix in ('job:', 'key:'):
            if token.startswith(prefix):
                mode=prefix[:-1];token=token[len(prefix):];break
        if not token or len(token)>128:raise ValueError('Invalid batch recovery token')
        by_id=self.store.get(self.owner_scope,token) if mode!='key' else None
        if by_id is not None and by_id['kind']!=KIND:by_id=None
        by_key=self.store.find_submission(self.owner_scope,KIND,token) if mode!='job' else None
        if by_id is not None and by_key is not None and by_id['job_id']!=by_key['job_id']:
            raise ValueError('Ambiguous recovery token; use job:ID or key:KEY explicitly')
        job=by_id or by_key
        if job is None:raise LookupError('Batch job/recovery key not found in this local scope')
        validate_request(job['request'])
        if job['request']['max_attempts']!=job['max_attempts']:raise JobIntegrityError('Stored retry budget differs')
        checkpoints=self.store.read_checkpoints(self.owner_scope,job['job_id'])
        reports,ready=_records(job['request'],checkpoints)
        if job['status']=='completed':verify_result(job['request'],checkpoints,job['result'])
        job['request_progress']={'ready_exact_labels':len(ready),'reported_exact_labels':len(reports),
                                 'required_exact_labels':len(_names(job['request']))}
        job['request_outcomes']=[{**row,'execution_status':reports[row['designation']]['status']
            if row['designation'] in reports else 'not_assessed',
            'checkpoint_ready':row['designation'] in ready,
            'report':clone(reports[row['designation']]) if row['designation'] in reports else None}
            for row in job['request']['source_input']['requests']]
        return job

    def run(self,token):
        job=self.read(token)  # Enforce owner lookup before the privileged targeted claim.
        if job['status'] not in ('completed','cancelled','failed'):
            if self.read_only:raise ValueError('Read-only job inspection cannot resume provider work')
            self.worker.run_once(job_id=job['job_id'])
        return self.read('job:'+job['job_id'])

    def close(self):
        self.worker.stop.set()
        if self.engine is not None:self.engine.dispose()


def default_runtime():
    """Fresh explicit configured store; never initialize the checked-in default DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from .enrichment_store import require_writable_acquisition_uri
    url=os.environ.get('ANEOS_DATABASE_URL')
    if not url:raise ValueError('Set ANEOS_DATABASE_URL to an explicit file-SQLite test/research database first')
    engine=create_engine(url,connect_args={'check_same_thread':False} if url.startswith('sqlite') else {})
    try:
        if engine.dialect.name!='sqlite':raise ValueError('Local batch execution is qualified only for file SQLite')
        immutable=engine.url.query.get('immutable','0')
        read_only=(engine.url.query.get('mode')=='ro' or
                   isinstance(immutable,str) and immutable.lower() in ('1','true','on','yes'))
        if not read_only:require_writable_acquisition_uri(engine)
        store=JobStore(sessionmaker(bind=engine))
        if not read_only:store.initialize_schema()
        runtime=BatchRuntime(store);runtime.engine=engine;runtime.read_only=read_only
        return runtime
    except BaseException:
        engine.dispose();raise
