"""
Analysis endpoints for aNEOS API.

Provides NEO analysis services including single and batch analysis,
result retrieval, and data export capabilities.
"""

from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
    from fastapi.responses import StreamingResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, analysis endpoints disabled")

try:
    from aneos_core.analysis.pipeline import AnalysisPipeline
    from aneos_core.config.settings import get_config
except ImportError:
    AnalysisPipeline = None
    get_config = lambda: {'api': {}, 'analysis': {}, 'ml': {}, 'monitoring': {}}
from ..models import (
    AnalysisRequest, BatchAnalysisRequest, AnalysisResponse,
    SearchRequest, PaginatedResponse, ExportRequest, ExportResponse, ErrorResponse
)
# Import moved to avoid circular imports
# from ..app import get_aneos_app
from ..auth import get_current_user
from ..schemas.detection import DetectionResponse, EvidenceSummary, OrbitalInput
from ..schemas.history import OrbitalHistoryResponse, OrbitalElementPoint
from ..schemas.impact import ImpactResponse
from aneos_core.analysis.impact_evidence import get_sentry_impact_service
from ..schemas.network import NetworkRequest, NetworkStatusResponse

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    router = APIRouter()
else:
    # Fallback router for when FastAPI is not available
    class MockRouter:
        def __init__(self):
            self.routes = []
            self.on_startup = []
            self.on_shutdown = []
            self.dependencies = []
            
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def put(self, *args, **kwargs): return lambda f: f
        def delete(self, *args, **kwargs): return lambda f: f
        def include_router(self, *args, **kwargs): pass
        def add_api_route(self, *args, **kwargs): pass
        def mount(self, *args, **kwargs): pass
    router = MockRouter()

# Analysis cache for recent results
_analysis_cache: Dict[str, AnalysisResponse] = {}
_export_jobs: Dict[str, Dict[str, Any]] = {}
_batch_store: Dict[str, Dict[str, Any]] = {}
_network_store: Dict[str, Any] = {}
# Detection result cache: designation → serializable dict
_detection_cache: Dict[str, Dict[str, Any]] = {}


def _persist_detection_result(designation: str, result: dict) -> None:
    """Write detection result to AnalysisResult table (background task)."""
    try:
        from aneos_api.database import SessionLocal, AnalysisService, HAS_SQLALCHEMY
        if not HAS_SQLALCHEMY:
            return
        db = SessionLocal()
        try:
            service = AnalysisService(db)
            service.save_analysis_result({
                'designation': designation,
                'overall_score': result.get('screening_score'),
                'classification': result.get('classification', 'UNKNOWN'),
                'confidence': result.get('confidence', 0.0),
                'processing_time': 0.0,
                'anomaly_score_data': result,
                'analyzed_by': 'detect_neo',
            })
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not persist detection result for %s: %s", designation, exc)

def _sigma_tier(sigma):
    """Deprecated: a heuristic rank does not define a significance tier."""
    raise NotImplementedError("Uncalibrated sigma tiers withdrawn; use screening classification")


def _build_interpretation(*args, **kwargs):
    """Deprecated: assumed Bayesian posteriors are not calibrated probabilities."""
    raise NotImplementedError("Uncalibrated interpretation withdrawn; use assessment.to_payload()")


def _spacecraft_veto(designation):
    """Deprecated: designation lookup cannot establish sigma or source identity."""
    raise NotImplementedError("Name-only spacecraft veto withdrawn; provenance-backed identity required")


def _screening_payload(neo, service=None):
    """Shared snapshot adapter; no per-endpoint scientific decision logic."""
    from aneos_core.analysis.screening import ScreeningAnalysisService
    assessment = (service or ScreeningAnalysisService()).analyze_neo(neo)
    payload = assessment.to_payload()
    payload['data_source'] = neo.sources_used[0] if neo.sources_used else None
    payload['data_freshness'] = neo.fetched_at.isoformat() if neo.fetched_at else None
    return payload


async def get_analysis_pipeline() -> AnalysisPipeline:
    """Get the analysis pipeline from the application."""
    from ..app import get_aneos_app  # Import here to avoid circular imports
    aneos_app = get_aneos_app()
    if not aneos_app.analysis_pipeline:
        raise HTTPException(status_code=503, detail="Analysis pipeline not available")
    return aneos_app.analysis_pipeline

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_neo(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    pipeline: AnalysisPipeline = Depends(get_analysis_pipeline),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Acquire current evidence and return the shared uncalibrated screening result."""
    try:
        result = await pipeline.analyze_neo(request.designation, force_refresh=request.force_refresh)
        if result.errors:
            code = {'source_unavailable': 503, 'invalid_evidence': 422}.get(result.status, 500)
            raise HTTPException(status_code=code, detail={
                'status': result.status, 'errors': result.errors,
                'classification': 'INCONCLUSIVE', 'screening_score': None})
        neo = result.neo_data
        raw = neo.to_dict() if neo is not None else {}
        score = result.anomaly_score.to_dict()
        if not request.include_indicators:
            score['indicator_scores'] = {}
        response = AnalysisResponse(
            designation=result.designation, anomaly_score=score,
            processing_time=result.processing_time, data_quality=result.data_quality,
            orbital_elements=raw.get('orbital_elements'),
            close_approaches=raw.get('close_approaches', []),
            raw_neo_data=raw if request.include_raw_data else None,
            screening_assessment=result.screening_assessment,
            analysis_metadata=result.analysis_metadata, status=result.status,
            errors=result.errors, warnings=result.warnings)
        # Snapshot history for retrieval/export, never a designation-only read cache.
        evidence_id = result.screening_assessment['evidence_id']
        policy_id = result.screening_assessment['policy_id']
        cache_key = f'{request.designation}:{evidence_id}:{policy_id}:{request.include_raw_data}:{request.include_indicators}'
        _analysis_cache[cache_key] = response
        _detection_cache[request.designation] = result.screening_assessment
        background_tasks.add_task(_log_analysis_completion, request.designation,
            result.processing_time, result.anomaly_score.overall_score)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Analysis failed for %s', request.designation)
        raise HTTPException(status_code=500, detail='Analysis failed') from exc

@router.get(
    "/detect",
    response_model=DetectionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid source measurements"},
        503: {"model": ErrorResponse, "description": "Source data unavailable"},
        500: {"model": ErrorResponse, "description": "Detection pipeline failure"},
    },
)
async def detect_neo(
    designation: str = Query(..., description="NEO designation, e.g. '2020 SO'"),
    force_refresh: bool = Query(False, description="Bypass cache and re-fetch from source"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Screen a named object; unavailable significance/origin remain null."""
    import asyncio
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.utils.errors import DataSourceUnavailableError
    try:
        neo = await asyncio.to_thread(DataFetcher().fetch_neo_data, designation, force_refresh=force_refresh)
        payload = _screening_payload(neo)
        _detection_cache[designation] = payload
        await asyncio.to_thread(_persist_detection_result, designation, payload)
        return DetectionResponse(**payload)
    except DataSourceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Source data unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Screening failed for %s", designation)
        raise HTTPException(status_code=500, detail="Screening failed") from exc

@router.get(
    "/impact", response_model=ImpactResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Use a primary number or provisional designation"},
        502: {"model": ErrorResponse, "description": "Invalid or unsupported Sentry response"},
        503: {"model": ErrorResponse, "description": "Authoritative impact source unavailable"},
    },
)
async def impact_neo(
    designation: str = Query(..., description="Primary asteroid number or unpacked provisional designation; names/aliases are not resolved"),
    force_refresh: bool = Query(False, description="Explicitly bypass the per-process Sentry TTL cache"),
    current_user: Optional[Dict] = Depends(get_current_user),
    impact_service=Depends(get_sentry_impact_service),
):
    """Return replayable JPL Sentry evidence for its tabulated Earth-impact events.

    Not listed/removed is not zero risk. No lunar probability, local fitted
    probability, calibrated confidence, damage prediction or origin inference.
    Source method, dates, uncertainty limits and complete input snapshot accompany
    every result. This endpoint does not use the legacy heuristic calculator.
    """
    from aneos_core.data.sources.sentry import normalize_designation, SentryResponseError, verify_sentry_report
    from aneos_core.data.provider_http import ProviderUnavailableError
    try:
        canonical = normalize_designation(designation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        report = await asyncio.to_thread(impact_service.assess, canonical, force_refresh=force_refresh)
        if not verify_sentry_report(report) or report.get('designation') != canonical:
            raise SentryResponseError('Invalid or modified Sentry report')
        return ImpactResponse(**report)
    except SentryResponseError as exc:
        raise HTTPException(status_code=502, detail="Sentry evidence schema or identity could not be verified") from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Sentry impact source unavailable; no probability assessment") from exc
    except Exception as exc:
        logger.exception("Authoritative impact retrieval failed")
        raise HTTPException(status_code=500, detail="Impact evidence retrieval failed; no probability assessment") from exc


@router.post("/detect", response_model=DetectionResponse)
async def detect_neo_raw(
    request: OrbitalInput,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Screen supplied measurements; a name alone is not origin evidence."""
    from aneos_core.analysis.screening import ScreeningAnalysisService
    try:
        designation = request.designation or 'USER_PROVIDED'
        physical = {'diameter_km': request.diameter_km, 'albedo': request.albedo,
                    'absolute_magnitude': request.absolute_magnitude,
                    'spectral_type': request.spectral_type}
        extra = dict(request.additional_data or {})
        if request.orbital_history is not None:
            if 'orbital_history' in extra and extra['orbital_history'] != request.orbital_history:
                raise ValueError('Conflicting orbital history inputs')
            extra['orbital_history'] = request.orbital_history
        assessment = ScreeningAnalysisService().analyze_raw(
            designation, {'a': request.a, 'e': request.e, 'i': request.i}, physical,
            extra,
            request.evidence_metadata,
        )
        return DetectionResponse(**assessment.to_payload(), data_source='user_provided')
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Raw screening failed")
        raise HTTPException(status_code=500, detail="Screening failed") from exc


@router.get("/history", response_model=OrbitalHistoryResponse)
async def orbital_history(
    designation: str = Query(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[^\x00-\x1f\x7f]+$",
        description="NEO designation, e.g. 'Apophis'",
    ),
    years: int = Query(10, ge=1, le=50, description="Number of years to span"),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Fetch time-series Keplerian orbital elements from JPL Horizons.
    Returns bounded computed samples with native epochs, never observation coverage.
    """
    try:
        from aneos_core.data.sources.horizons import HorizonsSource

        if not designation.strip():
            raise HTTPException(status_code=422, detail="Nonempty designation required")
        source = HorizonsSource()
        raw_points = await asyncio.to_thread(
            source.fetch_orbital_history, designation, years=years
        )
        query_evidence = getattr(raw_points, "query_metadata", None)
        status = (
            query_evidence.get("status") if isinstance(query_evidence, dict) else None
        )
        if status == "unavailable":
            raise HTTPException(
                status_code=503,
                detail="Orbital history source unavailable; no observational assessment",
            )
        if status == "invalid_evidence":
            raise HTTPException(
                status_code=502,
                detail="Orbital history source evidence could not be verified",
            )
        if query_evidence is not None and status not in ("available", "empty"):
            raise HTTPException(
                status_code=502,
                detail="Orbital history query outcome could not be verified",
            )
        if status in ("available", "empty") and bool(raw_points) != (
            status == "available"
        ):
            raise HTTPException(
                status_code=502,
                detail="Orbital history query outcome contradicts its points",
            )
        if status in ("available", "empty") and (
            query_evidence.get("schema_version") != "orbital-history-query-v1"
            or query_evidence.get("designation_requested") != designation
            or type(query_evidence.get("returned_points")) is not int
            or query_evidence["returned_points"] != len(raw_points)
        ):
            raise HTTPException(
                status_code=502,
                detail="Orbital history query binding could not be verified",
            )
        if not raw_points:
            raise HTTPException(
                status_code=404,
                detail=f"No orbital history available for {designation}",
            )
        points = [OrbitalElementPoint(**p) for p in raw_points]
        return OrbitalHistoryResponse(
            designation=designation,
            years=years,
            points=points,
            query_evidence=query_evidence,
        )
    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Orbital history contains unsupported evidence for %s", designation
        )
        raise HTTPException(
            status_code=502,
            detail="Orbital history source evidence could not be verified",
        ) from exc
    except Exception as exc:
        logger.exception("Orbital history failed for %s", designation)
        raise HTTPException(
            status_code=503,
            detail="Orbital history source unavailable; no observational assessment",
        ) from exc


@router.post("/analyze/batch", response_model=Dict[str, Any])
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Screen multiple NEOs using the same DataFetcher + shared service as GET /detect.
    Results remain in-memory; restart durability is tracked separately.
    """
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    _batch_store[batch_id] = {
        'batch_id': batch_id,
        'status': 'processing',
        'total': len(request.designations),
        'completed': 0,
        'failed': 0,
        'results': [],
        'started_at': datetime.now().isoformat(),
    }
    background_tasks.add_task(_run_batch_detection, batch_id, request.designations,
        force_refresh=request.force_refresh, include_raw_data=request.include_raw_data)
    return {
        'batch_id': batch_id,
        'status': 'processing',
        'total_neos': len(request.designations),
        'status_url': f'/api/v1/analysis/batch/{batch_id}/status',
    }


@router.get("/batch/{batch_id}/status", response_model=Dict[str, Any])
async def get_batch_status(
    batch_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get status and results of a batch detection job."""
    if batch_id not in _batch_store:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return _batch_store[batch_id]

@router.get("/results/{designation}")
async def get_analysis_result(
    designation: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get analysis result for a NEO (memory cache then DB fallback)."""
    # Check in-memory detection cache first
    if designation in _detection_cache:
        return _detection_cache[designation]
    # Check analysis cache
    for result in reversed(list(_analysis_cache.values())):
        if result.designation.upper() == designation.upper():
            return result
    # DB fallback
    try:
        from aneos_api.database import SessionLocal, AnalysisService, HAS_SQLALCHEMY
        if HAS_SQLALCHEMY:
            db = SessionLocal()
            try:
                service = AnalysisService(db)
                rows = service.get_analysis_results(designation=designation, limit=1)
                if rows:
                    return rows[0]
            finally:
                db.close()
    except Exception as exc:
        logger.debug("DB fallback for %s failed: %s", designation, exc)
    raise HTTPException(
        status_code=404,
        detail=f"No results found for {designation}"
    )

@router.get("/search", response_model=PaginatedResponse)
async def search_analyses(
    request: SearchRequest = Depends(),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Search and filter analysis results.
    
    Supports filtering by classification, score range, date range,
    and specific NEO designations.
    """
    try:
        # For now, return cached results (in production would query database)
        all_results = list(_analysis_cache.values())
        
        # Apply filters
        filtered_results = all_results
        if request.filters:
            if request.filters.classification:
                filtered_results = [
                    r for r in filtered_results 
                    if r.anomaly_score.classification in request.filters.classification
                ]
            
            if request.filters.min_score is not None:
                filtered_results = [
                    r for r in filtered_results 
                    if r.anomaly_score.overall_score is not None and r.anomaly_score.overall_score >= request.filters.min_score
                ]
            
            if request.filters.max_score is not None:
                filtered_results = [
                    r for r in filtered_results 
                    if r.anomaly_score.overall_score is not None and r.anomaly_score.overall_score <= request.filters.max_score
                ]
        
        # Apply pagination
        total = len(filtered_results)
        start = (request.page - 1) * request.page_size
        end = start + request.page_size
        page_results = filtered_results[start:end]
        
        return PaginatedResponse(
            items=page_results,
            total=total,
            page=request.page,
            page_size=request.page_size,
            total_pages=max(1, (total + request.page_size - 1) // request.page_size)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/export", response_model=ExportResponse)
async def export_analyses(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Export analysis results in various formats.
    
    Supports JSON, CSV, XLSX, and PDF export formats with filtering options.
    """
    try:
        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create export job
        export_job = {
            'export_id': export_id,
            'format': request.format,
            'status': 'processing',
            'created_at': datetime.now(),
            'filters': request.filters,
            'include_raw_data': request.include_raw_data
        }
        
        _export_jobs[export_id] = export_job
        
        # Start export processing in background
        background_tasks.add_task(_process_export, export_id, request)
        
        return ExportResponse(
            export_id=export_id,
            format=request.format,
            status='processing'
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/export/{export_id}/status", response_model=ExportResponse)
async def get_export_status(
    export_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get status of an export job."""
    if export_id not in _export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")
    
    job = _export_jobs[export_id]
    return ExportResponse(
        export_id=export_id,
        format=job['format'],
        status=job['status'],
        download_url=job.get('download_url'),
        file_size_bytes=job.get('file_size_bytes'),
        expires_at=job.get('expires_at')
    )

@router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Download completed export file (JSON or CSV)."""
    if export_id not in _export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")
    job = _export_jobs[export_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Export not yet completed")
    content = job.get('content', b'')
    fmt = str(job.get('format', 'json')).lower().replace('exportformat.', '')
    media_type = 'text/csv' if fmt == 'csv' else 'application/json'
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=aneos_export_{export_id}.{fmt}"}
    )

# =============================================================================
# ENHANCED ANALYSIS ENDPOINTS - SCIENTIFIC RIGOR VALIDATION
# =============================================================================
# These endpoints provide enhanced analysis with comprehensive validation
# while preserving all existing endpoints unchanged for backward compatibility.

# Enhanced analysis availability check
try:
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline, create_enhanced_pipeline
    HAS_ENHANCED_ANALYSIS = True
except ImportError:
    HAS_ENHANCED_ANALYSIS = False
    logger.warning("Enhanced analysis features not available")

if HAS_ENHANCED_ANALYSIS:
    # Enhanced endpoint will be added in a separate enhancement module
    # This preserves the existing analysis.py from any breaking changes
    logger.info("Enhanced analysis endpoints available - see enhanced_analysis.py")
else:
    logger.info("Enhanced analysis endpoints not available - using standard analysis only")

@router.get("/stats", response_model=Dict[str, Any])
async def get_analysis_stats(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get analysis statistics and system metrics."""
    return {
        'total_analyses': len(_analysis_cache),
        'cache_size': len(_analysis_cache),
        'export_jobs': len(_export_jobs),
        'system_status': 'operational'
    }

# Background task functions
async def _log_analysis_completion(
    designation: str,
    processing_time: float,
    overall_score: Optional[float]
):
    """Log analysis completion for monitoring."""
    logger.info(
        f"Analysis completed: {designation}, "
        f"Uncalibrated rank: {overall_score if overall_score is not None else 'unavailable'}, "
        f"Time: {processing_time:.2f}s"
    )

async def _run_batch_detection(batch_id: str, designations: List[str],
                               force_refresh: bool = False, include_raw_data: bool = False):
    """Use the same snapshot service as single-object screening."""
    import asyncio
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.analysis.screening import ScreeningAnalysisService
    store = _batch_store[batch_id]
    fetcher = DataFetcher()
    service = ScreeningAnalysisService()
    for designation in designations:
        try:
            neo = await asyncio.to_thread(fetcher.fetch_neo_data, designation, force_refresh=force_refresh)
            payload = _screening_payload(neo, service)
            if include_raw_data:
                payload['raw_neo_data'] = neo.to_dict()
            store['results'].append({**payload, 'status': 'success'})
            _detection_cache[designation] = payload
            await asyncio.to_thread(_persist_detection_result, designation, payload)
            store['completed'] += 1
        except Exception as exc:
            store['failed'] += 1
            store['results'].append({'designation': designation, 'status': 'failed', 'error': str(exc)})
    store['status'] = 'completed'
    store['finished_at'] = datetime.now().isoformat()

def _run_network_analysis(*args, **kwargs):
    """Controlled deprecation: daemon/dictionary execution is not durable."""
    raise NotImplementedError('Use the registered SQL NetworkWorker; process-memory execution withdrawn')


async def start_network_analysis(*args, **kwargs):
    raise NotImplementedError('Use the authenticated durable network HTTP route')


async def get_network_status(*args, **kwargs):
    raise NotImplementedError('Use the owner-filtered persistent network HTTP route')


from ..network_routes import submit_network, network_status, cancel_network
router.post('/analyze/network', summary='Submit authenticated durable population analysis')(submit_network)
router.get('/analyze/network/{job_id}/status', response_model=NetworkStatusResponse,
           summary='Read an owned persistent population job')(network_status)
router.delete('/analyze/network/{job_id}', response_model=NetworkStatusResponse,
              summary='Cancel an owned persistent population job')(cancel_network)


async def _process_export(export_id: str, request: ExportRequest):
    """Serialize _analysis_cache to JSON or CSV content."""
    import io
    import csv
    import json as _json
    job = _export_jobs[export_id]
    fmt = str(getattr(request, 'format', 'json')).lower().replace('exportformat.', '')
    items = list(_analysis_cache.values())
    try:
        if fmt == 'csv':
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['designation', 'screening_score', 'score_semantics', 'classification',
                             'significance_status', 'evidence_id', 'policy_id'])
            for item in items:
                assessment = item.screening_assessment
                writer.writerow([item.designation, assessment['screening_score'],
                    item.anomaly_score.score_semantics, assessment['classification'],
                    assessment['significance_status'], assessment['evidence_id'], assessment['policy_id']])
            content = buf.getvalue().encode('utf-8')
        elif fmt == 'json':
            content = _json.dumps([i.model_dump(mode='json', exclude={'raw_neo_data'})
                if not request.include_raw_data else i.model_dump(mode='json') for i in items],
                allow_nan=False).encode('utf-8')
        else:
            raise ValueError('Only JSON and CSV analysis exports are implemented')
        job['content'] = content
        job['status'] = 'completed'
        job['file_size_bytes'] = len(content)
        job['download_url'] = f'/api/v1/analysis/export/{export_id}/download'
        job['expires_at'] = datetime.now().isoformat()
        logger.info(f"Export {export_id} completed")
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        logger.error(f"Export {export_id} failed: {e}")
