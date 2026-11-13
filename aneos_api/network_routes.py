"""Existing network routes backed by authenticated SQL execution, not daemon dicts."""
from dataclasses import asdict
from hashlib import sha256
from fastapi import Request,Depends,Header,HTTPException
from .network_access import require_network_principal,require_writer,get_runtime
from .schemas.network import NetworkRequest,NetworkStatusResponse
from .job_store import QueueFull,IdempotencyConflict,JobStorageUnavailable,JobIntegrityError
from .network_jobs import NETWORK_KIND,build_network_request,verify_network_job_result
from aneos_core.pattern_analysis.session import PatternAnalysisConfig


def _config(body):
    return PatternAnalysisConfig(clustering=body.clustering,harmonics=body.harmonics,
        correlation=body.correlation,rendezvous=body.rendezvous,historical_years=body.historical_years)


def _same_intent(existing,body):
    request=existing['request']
    return request.get('designations')==body.designations and request.get('configuration')==asdict(_config(body))


def _ack(row):
    return {'job_id':row['job_id'],'status':_status(row['status']),
            'status_url':f"/api/v1/analysis/analyze/network/{row['job_id']}/status"}


def _status(value):
    return {'running':'processing','completed':'complete','failed':'error'}.get(value,value)


def _error(exc):
    if isinstance(exc,QueueFull):raise HTTPException(429,'Persistent execution queue is full',headers={'Retry-After':'5'}) from exc
    if isinstance(exc,IdempotencyConflict):raise HTTPException(409,'Idempotency key is bound to another request') from exc
    raise HTTPException(503,'Persistent execution unavailable or integrity unconfirmed') from exc


def submit_network(request:Request,body:NetworkRequest,principal=Depends(require_network_principal),
                   idempotency_key:str|None=Header(None,alias='Idempotency-Key')):
    require_writer(principal);runtime=get_runtime(request)
    if not runtime.ready:raise HTTPException(503,'Network worker is not ready for admission')
    key=None
    if idempotency_key is not None:
        if not idempotency_key or len(idempotency_key)>128 or idempotency_key!=idempotency_key.strip() or any(ord(c)<33 for c in idempotency_key):
            raise HTTPException(422,'Invalid idempotency key')
        key='http-network-v1:'+sha256(idempotency_key.encode()).hexdigest()
    try:
        if key:
            old=runtime.store.find_submission(principal.subject,NETWORK_KIND,key)
            if old:
                if not _same_intent(old,body):raise IdempotencyConflict('Different intent')
                return _ack(old)
        payload=build_network_request(body.designations,config=_config(body),
                                      source_profile_id=runtime.worker.handler.source_profile())
        try:
            row=runtime.store.submit(owner_scope=principal.subject,kind=NETWORK_KIND,request=payload,idempotency_key=key)
        except IdempotencyConflict:
            old=runtime.store.find_submission(principal.subject,NETWORK_KIND,key)
            if not old or not _same_intent(old,body):raise
            row=old
        return _ack(row)
    except (QueueFull,IdempotencyConflict,JobStorageUnavailable,JobIntegrityError,ValueError,TypeError,KeyError) as exc:_error(exc)


def network_status(request:Request,job_id:str,principal=Depends(require_network_principal)):
    runtime=get_runtime(request)
    try:
        row=runtime.store.get(principal.subject,job_id)
        if not row or row['kind']!=NETWORK_KIND:raise HTTPException(404,'Job not found')
        result=row['result'];report=None
        if result is not None:
            result=verify_network_job_result(row['request'],result);report=result['scientific_report']
        values={'job_id':job_id,'status':_status(row['status']),'execution_status':row['status'],
            'error':row['error_code'],'attempts':row['attempts'],'revision':row['revision'],
            'checkpoint_count':row['checkpoint_count'],'execution_result':result,'scientific_report':report,
            'analysis_metadata':{'execution_storage':'sql_durable','worker_ready':runtime.ready,
                                 'result_sha256':row['result_sha256'],'request_sha256':row['request_sha256']}}
        if report:
            for key in ('designations_analyzed','clusters','harmonic_signals','correlation_matrix',
                        'network_sigma','network_tier','combined_p_value','sub_module_p_values'):
                values[key]=report[key]
            values['analysis_metadata'].update(report['analysis_metadata'])
            values['acquisition_outcomes']=[{'input_index':i,'designation':d,'status':'available'}
                                           for i,d in enumerate(row['request']['designations'])]
        return NetworkStatusResponse(**values)
    except (JobStorageUnavailable,JobIntegrityError,ValueError,TypeError,KeyError) as exc:_error(exc)


def cancel_network(request:Request,job_id:str,principal=Depends(require_network_principal)):
    require_writer(principal);runtime=get_runtime(request)
    try:
        row=runtime.store.get(principal.subject,job_id)
        if not row or row['kind']!=NETWORK_KIND:raise HTTPException(404,'Job not found')
        runtime.store.cancel(principal.subject,job_id)
        return network_status(request,job_id,principal)
    except (JobStorageUnavailable,JobIntegrityError) as exc:_error(exc)
