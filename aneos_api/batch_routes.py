"""Owner-scoped projections of the existing registered canonical batch jobs."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import math
import re

from fastapi import Request, Depends, Header, HTTPException
from .models import BatchAnalysisRequest
from .network_access import require_batch_principal, require_writer, get_runtime
from .batch_jobs import KIND, BatchRuntime, build_request, validate_request
from .job_store import QueueFull, IdempotencyConflict, JobStorageUnavailable, JobIntegrityError
from aneos_core.analysis.batch_reporting import parse_input
from aneos_core.data.enrichment import clone


ERRORS=(QueueFull,IdempotencyConflict,JobStorageUnavailable,JobIntegrityError,ValueError,TypeError,KeyError)


def _error(exc):
    if isinstance(exc,QueueFull):
        raise HTTPException(429,'Persistent execution queue is full',headers={'Retry-After':'5'}) from exc
    if isinstance(exc,IdempotencyConflict):
        raise HTTPException(409,'Idempotency key is bound to another request') from exc
    raise HTTPException(503,'Persistent batch execution unavailable or integrity unconfirmed') from exc


def _source(body):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n');writer.writerow(['designation'])
    writer.writerows([d] for d in body.designations)
    source=parse_input(stream.getvalue().encode('utf-8'),'http-batch-designations.csv')
    if [r['designation'] for r in source['requests']]!=body.designations:
        raise ValueError('API input does not round-trip through the canonical batch parser')
    return source


def _options(body):
    return {'schema_version':'batch-http-response-v1','include_raw_data':body.include_raw_data}


def _same_intent(row,body,source):
    saved=validate_request(row['request'])
    return (saved['source_input']==source and saved['force_refresh']==body.force_refresh
            and saved.get('response_options')==_options(body))


def _reader(runtime,subject):
    return BatchRuntime(runtime.store,handler=runtime.batch_worker.handler,owner_scope=subject)


def _owned(runtime,subject,job_id):
    if not isinstance(job_id,str) or not re.fullmatch('job_[0-9a-f]{32}',job_id):
        raise HTTPException(404,'Batch job not found')
    try:
        return _reader(runtime,subject).read('job:'+job_id)
    except LookupError as exc:
        raise HTTPException(404,'Batch job not found') from exc


def _ack(row):
    return {'batch_id':row['job_id'],'job_id':row['job_id'],'status':_status(row['status']),
            'execution_status':row['status'],'total_neos':len(row['request']['source_input']['requests']),
            'status_url':f"/api/v1/analysis/batch/{row['job_id']}/status",
            'execution_storage':'sql_durable','scientific_status':'not_independently_validated'}


def _status(status):
    return 'processing' if status=='running' else status


def submit_batch(request:Request,body:BatchAnalysisRequest,principal=Depends(require_batch_principal),
                 idempotency_key:str|None=Header(None,alias='Idempotency-Key')):
    require_writer(principal);runtime=get_runtime(request)
    if len(request.headers.getlist('idempotency-key'))>1:
        raise HTTPException(422,'Use one idempotency key')
    key=None
    if idempotency_key is not None:
        if (not idempotency_key or len(idempotency_key)>128 or idempotency_key!=idempotency_key.strip()
            or any(ord(c)<33 or ord(c)==127 for c in idempotency_key)):
            raise HTTPException(422,'Invalid idempotency key')
        key='http-batch-v1:'+sha256(idempotency_key.encode()).hexdigest()
    try:source=_source(body)
    except ValueError as exc:raise HTTPException(422,'Invalid canonical batch input') from exc
    try:
        if key:
            old=runtime.store.find_submission(principal.subject,KIND,key)
            if old:
                if not _same_intent(old,body,source):raise IdempotencyConflict('Different intent')
                return _ack(_owned(runtime,principal.subject,old['job_id']))
        if not runtime.batch_ready:raise HTTPException(503,'Batch worker is not ready for admission')
        payload=build_request(source,force_refresh=body.force_refresh,response_options=_options(body),
                              source_profile_id=runtime.batch_worker.handler.source_profile())
        try:
            row=runtime.store.submit(owner_scope=principal.subject,kind=KIND,request=payload,
                                     idempotency_key=key,max_attempts=payload['max_attempts'])
        except IdempotencyConflict:
            old=runtime.store.find_submission(principal.subject,KIND,key)
            if not old or not _same_intent(old,body,source):raise
            row=old
        # A commit call alone is not the acknowledgement. Verify an owned readback.
        return _ack(_owned(runtime,principal.subject,row['job_id']))
    except ERRORS as exc:_error(exc)


def _utc(value):
    if value is None:return None
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Invalid stored execution time')
    try:return datetime.fromtimestamp(value,timezone.utc).isoformat()
    except (OverflowError,OSError,ValueError) as exc:
        raise ValueError('Stored execution time is outside supported range') from exc


def _projection(row,worker_ready):
    include_raw=row['request'].get('response_options',{}).get('include_raw_data',False)
    results=[];completed=failed=0
    for outcome in row['request_outcomes']:
        report=outcome['report']
        assessment=report['screening_assessment'] if report else None
        entry=clone(assessment) if assessment is not None else {
            'classification':'INCONCLUSIVE','screening_score':None,'candidate':None,
            'artificial_probability':None,'sigma_confidence':None,'false_discovery_rate':None,
            'is_artificial':None,'confidence':None,'combined_p_value':None,
            'significance_status':'unavailable_uncalibrated_model','origin_status':'inconclusive',
            'human_origin_status':'not_assessed'}
        entry.update({k:outcome[k] for k in ('request_index','source_line','designation','execution_status','checkpoint_ready')})
        entry['screening_assessment']=clone(assessment)
        entry['report_id']=report['report_id'] if report else None
        entry['status']='not_assessed' if report is None else 'failed' if report['errors'] else 'success'
        entry['errors']=clone(report['errors']) if report else []
        entry['five_sigma_claim_permitted']=False
        if report:
            if report['errors']:failed+=1
            else:completed+=1
            if include_raw:entry['raw_neo_data']=clone(report['neo_data'])
        results.append(entry)
    terminal=row['status'] in ('completed','cancelled','failed')
    result=row['result']
    return {'batch_id':row['job_id'],'job_id':row['job_id'],'status':_status(row['status']),
        'execution_status':row['status'],'total':len(results),'completed':completed,'failed':failed,
        'pending':len(results)-completed-failed,'results':results,'attempts':row['attempts'],
        'checkpoint_count':row['checkpoint_count'],'revision':row['revision'],'error':row['error_code'],
        'started_at':row['request']['created_at'],'finished_at':_utc(row['updated_at']) if terminal else None,
        'timestamp_roles':{'started_at':'local_submission_not_observation','finished_at':'local_terminal_execution_not_observation'},
        'queue_available_at':_utc(row['available_at']),'lease_until':_utc(row['lease_until']),
        'execution_storage':'sql_durable','worker_ready':worker_ready,
        'request_progress':clone(row['request_progress']),
        'scientific_report':clone(result['batch_report']) if include_raw and result is not None else None,
        'scientific_report_available':result is not None,
        'request_sha256':row['request_sha256'],'result_sha256':row['result_sha256'],
        'scientific_status':'not_independently_validated','five_sigma_claim_permitted':False,
        'count_semantics':'Request occurrences with retained reports; successful processing is not independent scientific validation.'}


def batch_status(request:Request,batch_id:str,principal=Depends(require_batch_principal)):
    runtime=get_runtime(request)
    try:return _projection(_owned(runtime,principal.subject,batch_id),runtime.batch_ready)
    except ERRORS as exc:_error(exc)


def cancel_batch(request:Request,batch_id:str,principal=Depends(require_batch_principal)):
    require_writer(principal);runtime=get_runtime(request)
    try:
        _owned(runtime,principal.subject,batch_id)
        runtime.store.cancel(principal.subject,batch_id)
        return batch_status(request,batch_id,principal)
    except ERRORS as exc:_error(exc)
