"""Existing enhanced REST routes for actual screening and review availability.

These adapters install no calibrated significance, uncertainty or origin model.
"""
from typing import List, Optional, Dict, Any, Annotated
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline, create_enhanced_pipeline
from aneos_core.analysis.pipeline import AnalysisPipeline, PipelineResult
from aneos_core.analysis.screening import ScreeningAnalysisService
from ..enhanced_models import (EnhancedAnalysisRequest, EnhancedAnalysisResponse,
    EnhancedBatchAnalysisResponse, ValidationSummaryResponse, ValidationResultModel)
from ..auth import get_current_user

logger = logging.getLogger(__name__)
HAS_FASTAPI = HAS_ENHANCED_ANALYSIS = True
router = enhanced_router = APIRouter()
_enhanced_pipeline: Optional[EnhancedAnalysisPipeline] = None


async def get_analysis_pipeline() -> AnalysisPipeline:
    """Use the same actual application service as /analyze, never a mock."""
    from .analysis import get_analysis_pipeline as actual_pipeline
    return await actual_pipeline()


async def get_neo_data(designation: str, force_refresh: bool = False):
    """Compatibility helper for typed acquisition; propagate source errors."""
    pipeline = await get_analysis_pipeline()
    return await pipeline._fetch_neo_data(designation, force_refresh=force_refresh)


async def get_enhanced_analysis_pipeline() -> EnhancedAnalysisPipeline:
    """Bind review wrapper to the current application pipeline, not a stale service."""
    global _enhanced_pipeline
    original = await get_analysis_pipeline()
    if _enhanced_pipeline is None or _enhanced_pipeline.original_pipeline is not original:
        _enhanced_pipeline = create_enhanced_pipeline(original_pipeline=original, enable_validation=True)
    return _enhanced_pipeline


def _execution_options(request, result):
    stages = result.stage_results if result is not None else []
    return {
        'requested': request.model_dump(exclude={'designation'}),
        'effective_validation_method': 'required_prerequisite_inventory' if stages else None,
        'required_review_pass_granted': bool(result is not None and result.overall_validation_passed),
        'uncertainty_computation_performed': False,
        'uncertainty_status': 'unavailable_no_calibrated_observation_model',
        'method_note': 'Requested sampling/method preferences are recorded, not performed; no fast/statistical-only bypass of required prerequisites is installed.',
    }


def _error_response(designation, status, errors, elapsed, timestamp):
    return EnhancedAnalysisResponse(designation=designation, classification='INCONCLUSIVE',
        status=status, errors=errors, processing_time_ms=elapsed, timestamp=timestamp,
        analysis_metadata={'five_sigma_claim_permitted': False, 'scientific_validation_status': 'unavailable'})


def _validation_transport(report, original):
    """Check review transport; no installed artifact supports numerical inference.

    This guard is not a physical/statistical validator. It refuses malformed,
    incomplete or stale workflow passes while retaining the core assessment.
    """
    from aneos_core.validation.review_evidence import freeze_review_inputs
    try:
        model = ValidationResultModel.model_validate(report)
    except (ValueError, TypeError):
        model = ValidationResultModel(overall_validation_passed=False,
            recommendation='do_not_advance', expert_review_priority='medium',
            validation_status='unavailable', validation_timestamp=datetime.now(timezone.utc),
            transport_limitations=['Malformed review report; no pass is available'])
    if model.overall_validation_passed:
        try:
            _, review_id = freeze_review_inputs(original.neo_data)
            assessment = original.screening_assessment
            current = ScreeningAnalysisService().analyze_neo(original.neo_data)
            stages = report.stage_results
            valid = (type(report.overall_validation_passed) is bool and
                report.contract_version == 'required-review-v1' and report.validation_status == 'passed' and
                len(stages) == 5 and {s.stage_number for s in stages} == set(range(1, 6)) and
                all(type(s.stage_number) is int and type(s.passed) is bool and s.passed and
                    s.validation_status == 'passed' and not s.details.get('error') and
                    not s.details.get('error_message') for s in stages) and
                report.evidence_id == assessment['evidence_id'] == current.evidence_id and
                report.policy_id == assessment['policy_id'] and
                report.validation_evidence_id == review_id)
        except (ValueError, TypeError, AttributeError, KeyError):
            valid = False
        if not valid:
            model.overall_validation_passed = False
            model.validation_status = 'unavailable'
            model.recommendation = 'do_not_advance'
            model.transport_limitations.append('Incomplete, malformed or stale review binding; no pass is available')
    # Legacy default timing/numerical fields are not measured or calibrated.
    model.total_processing_time_ms = None
    model.overall_false_positive_probability = model.overall_confidence = None
    model.synthetic_population_percentile = model.statistical_significance_summary = None
    for stage in model.stage_results:
        stage.processing_time_ms = stage.score = stage.confidence = stage.false_positive_reduction = None
    model.transport_limitations.append('No calibrated numerical inference or instrumented stage timing is installed')
    return model


async def _analyze_request(request, pipeline):
    start, timestamp = time.monotonic(), datetime.now(timezone.utc)
    result = await pipeline.analyze_neo_with_validation(request.designation,
        force_refresh=request.force_refresh, enable_validation=request.enable_validation)
    original = result.original_result
    if not isinstance(original, PipelineResult):
        raise HTTPException(500, detail='Enhanced adapter requires the canonical core analysis contract')
    if original.errors:
        code = {'source_unavailable':503, 'invalid_evidence':422}.get(original.status,500)
        raise HTTPException(code, detail={'status':original.status, 'errors':original.errors,
            'classification':'INCONCLUSIVE', 'screening_score':None})
    # Core owns the decision; optional review cannot erase a legitimate candidate.
    assessment = original.screening_assessment
    if not isinstance(assessment, dict):
        raise HTTPException(500, detail='Canonical screening assessment unavailable')
    validation = _validation_transport(result.validation_result, original)
    return EnhancedAnalysisResponse(
        designation=original.designation, overall_score=assessment['screening_score'],
        confidence=None, classification=assessment['classification'], status=original.status,
        errors=original.errors, warnings=original.warnings,
        processing_time_ms=(time.monotonic()-start)*1000, timestamp=timestamp,
        screening_assessment=assessment, analysis_metadata=original.analysis_metadata,
        raw_neo_data=original.neo_data.to_dict() if request.include_raw_data and original.neo_data is not None else None,
        indicator_scores=original.anomaly_score.indicator_scores if request.include_indicators else {},
        validation_result=validation, statistical_tests=result.statistical_tests,
        uncertainty_analysis=result.uncertainty_analysis,
        execution_options=_execution_options(request,validation),
        enhancement_timestamp=result.enhancement_timestamp,
        enhancement_version='enhanced-rest-v1')


@router.post('/enhanced', response_model=EnhancedAnalysisResponse)
async def analyze_neo_enhanced(request: EnhancedAnalysisRequest, background_tasks: BackgroundTasks,
    pipeline: EnhancedAnalysisPipeline = Depends(get_enhanced_analysis_pipeline),
    current_user: Optional[Dict] = Depends(get_current_user)):
    """Shared screening plus explicit required-review status, not calibrated science."""
    try:
        response = await _analyze_request(request,pipeline)
        background_tasks.add_task(_log_enhanced_analysis_result, request.designation,
            response, response.processing_time_ms)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Enhanced analysis failed for %s',request.designation)
        raise HTTPException(500, detail='Enhanced analysis failed') from exc


def _batch_summary(results):
    return {
        'requested':len(results),
        'successful_rankings':sum(r.status=='completed' for r in results),
        'inconclusive':sum(r.status=='inconclusive' for r in results),
        'failed':sum(r.status in {'source_unavailable','invalid_evidence','failed'} for r in results),
        'candidates':sum(bool(r.screening_assessment and r.screening_assessment['candidate']) for r in results),
        'required_review_passed':sum(bool(r.validation_result and r.validation_result.overall_validation_passed) for r in results),
        'false_positive_probability':None,
        'summary_semantics':'execution_and_screening_counts_not_scientific_accuracy',
    }


@router.post('/enhanced/batch', response_model=EnhancedBatchAnalysisResponse)
async def batch_analyze_enhanced(designations: Annotated[List[str],Body(min_length=1,max_length=100)],
    background_tasks: BackgroundTasks, force_refresh: bool = False, enable_validation: bool = True,
    pipeline: EnhancedAnalysisPipeline = Depends(get_enhanced_analysis_pipeline),
    current_user: Optional[Dict] = Depends(get_current_user)):
    """Ordered outcomes for every object, including source failures and abstentions."""
    # Validate all identities before acquisition; preserve duplicates rather than zip/drop.
    try:
        requests = [EnhancedAnalysisRequest(designation=d, force_refresh=force_refresh,
            enable_validation=enable_validation) for d in designations]
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if not 1 <= len(requests) <= 100:
        raise HTTPException(422, detail='Batch must contain 1–100 designations')
    start, timestamp = time.monotonic(), datetime.now(timezone.utc)
    config = pipeline.original_pipeline.pipeline_config
    semaphore = asyncio.Semaphore(max(1,min(config.max_workers,pipeline.original_pipeline.analysis_queue_size))
        if pipeline.original_pipeline.analysis_parallel else 1)

    async def run(request):
        async with semaphore:
            item_start = time.monotonic()
            try:
                return await asyncio.wait_for(_analyze_request(request,pipeline),config.timeout_seconds)
            except HTTPException as exc:
                status = {503:'source_unavailable',422:'invalid_evidence'}.get(exc.status_code,'failed')
                details = exc.detail if isinstance(exc.detail,dict) else {'errors':[str(exc.detail)]}
                return _error_response(request.designation,status,details.get('errors',['Analysis failed']),
                    (time.monotonic()-item_start)*1000,datetime.now(timezone.utc))
            except Exception as exc:
                logger.exception('Enhanced batch item failed: %s',request.designation)
                reason = 'Analysis timeout' if isinstance(exc,asyncio.TimeoutError) else 'Analysis failed'
                return _error_response(request.designation,'failed',[reason],
                    (time.monotonic()-item_start)*1000,datetime.now(timezone.utc))

    results = list(await asyncio.gather(*(run(r) for r in requests)))
    response = EnhancedBatchAnalysisResponse(batch_id=f'enhanced_batch_{uuid4().hex}',
        results=results, summary_statistics=_batch_summary(results),
        total_processing_time_ms=(time.monotonic()-start)*1000,timestamp=timestamp)
    background_tasks.add_task(_log_batch_analysis_result,response.batch_id,len(requests),
        sum(not r.errors for r in results),response.total_processing_time_ms)
    return response


def _load_summary_rows(days):
    from ..database import SessionLocal, AnalysisResult, HAS_SQLALCHEMY
    if not HAS_SQLALCHEMY:
        raise HTTPException(503,detail='Database unavailable')
    db = SessionLocal()
    try:
        rows = db.query(AnalysisResult).filter(AnalysisResult.analysis_date >=
            datetime.now(timezone.utc)-timedelta(days=days)).all()
        return [{'anomaly_score_data':r.anomaly_score_data} for r in rows]
    finally:
        db.close()


def _rank_statistics(scores):
    return {'mean':sum(scores)/len(scores) if scores else None,
        'min':min(scores) if scores else None,'max':max(scores) if scores else None,
        'available_count':len(scores),'score_semantics':'uncalibrated_screening_rank'}


def _summarize_rows(rows, days):
    service = ScreeningAnalysisService()
    included, excluded, groups = [], {'legacy_or_missing_contract':0,'invalid_or_unsupported_contract':0}, {}
    for row in rows:
        data = row.get('anomaly_score_data')
        if not isinstance(data,dict) or data.get('schema_version')!='screening-assessment-v1':
            excluded['legacy_or_missing_contract'] += 1
            continue
        try:
            replay = service.replay(data['evidence_snapshot'],data['evidence_id'])
            canonical = replay.to_payload()
            fields = ['schema_version','engine_version','policy_id','screening_score','candidate',
                'classification','confidence','sigma_confidence','combined_p_value',
                'artificial_probability','false_discovery_rate','is_artificial']
            if any(k not in data or data[k]!=canonical[k] for k in fields):
                raise ValueError('Unsupported policy or modified assessment claims')
            included.append(canonical)
            groups.setdefault(canonical['policy_id'],[]).append(canonical)
        except (ValueError,KeyError,TypeError,AttributeError):
            excluded['invalid_or_unsupported_contract'] += 1
    scores = [r['screening_score'] for r in included if r['screening_score'] is not None]
    return ValidationSummaryResponse(total_analyses=len(rows),included_screening_assessments=len(included),
        excluded_rows=excluded,
        screening_counts={name:sum(r['classification']==name for r in included)
            for name in ['FOLLOW_UP_CANDIDATE','NOT_FLAGGED','INCONCLUSIVE']},
        policy_groups={key:{'assessments':len(values),**_rank_statistics([
            v['screening_score'] for v in values if v['screening_score'] is not None])}
            for key,values in groups.items()},
        false_positive_statistics={'status':'not_reliably_estimated','mean_probability':None,
            'median_probability':None,'high_fp_risk_count':None,'low_fp_risk_count':None},
        recommendations={'unknown':len(rows)},overall_score_statistics=_rank_statistics(scores),
        validation_system_status={'database_available':True,'validation_available':False,
            'scientific_validation_status':'unavailable_uncalibrated','period_days':days,
            'measurement_unit':'persisted_assessment_rows_not_independent_objects',
            'summary_scope':'replayable_current_policy_screening_only_not_enhanced_validation_history',
            'unique_included_objects':len({r['designation'] for r in included}),
            'five_sigma_claim_permitted':False})


@router.get('/enhanced/summary', response_model=ValidationSummaryResponse)
async def get_validation_summary(days: int = Query(7,ge=1,le=365),
    current_user: Optional[Dict] = Depends(get_current_user)):
    """Descriptive stored screening counts, never inferred FDR or acceptance."""
    try:
        return _summarize_rows(await asyncio.to_thread(_load_summary_rows,days),days)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Enhanced summary unavailable')
        raise HTTPException(500,detail='Enhanced summary unavailable') from exc


async def _log_enhanced_analysis_result(designation,enhanced_result,processing_time):
    validation = enhanced_result.validation_result
    probability = validation.overall_false_positive_probability if validation is not None else None
    logger.info('Enhanced assessment: %s; FP probability=%s; time=%.1fms',designation,
        probability if probability is not None else 'unavailable',processing_time)


async def _log_batch_analysis_result(batch_id,requested_count,completed_count,processing_time):
    logger.info('Enhanced batch %s: completed=%d requested=%d; time=%.1fms (execution, not accuracy)',
        batch_id,completed_count,requested_count,processing_time)
