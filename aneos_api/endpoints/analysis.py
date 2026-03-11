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
                'overall_score': result.get('artificial_probability', 0.0),
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

# Known spacecraft catalog for API-level pre-check
_KNOWN_SPACECRAFT = {
    "2018 a1": "Tesla Roadster (SpaceX Falcon Heavy, 2018-017A)",
    "2020 so": "Centaur upper stage (Surveyor 2, 1966-084A)",
    "j002e3":  "S-IVB upper stage (Apollo 12)",
}


def _sigma_tier(sigma: float) -> str:
    if sigma >= 5.0: return "EXCEPTIONAL"
    if sigma >= 4.0: return "ANOMALOUS"
    if sigma >= 3.0: return "SIGNIFICANT"
    if sigma >= 2.0: return "INTERESTING"
    if sigma >= 1.0: return "NOTABLE"
    return "ROUTINE"


def _build_interpretation(designation, sigma_confidence, tier, top_type, bayesian_prob):
    pct = bayesian_prob * 100
    return (
        f"{designation} shows {tier} orbital/physical characteristics "
        f"(sigma={sigma_confidence:.2f}, dominant evidence: {top_type}). "
        f"artificial_probability={pct:.2f}% — incorporates 0.1% base-rate prior. "
        f"Propulsion or course-correction evidence required to exceed 10%."
    )


def _spacecraft_veto(designation: str):
    """Return (is_vetoed, reason) tuple for known spacecraft catalog."""
    key = designation.strip().lower()
    label = _KNOWN_SPACECRAFT.get(key)
    if label:
        return True, f"Known spacecraft: {label}"
    return False, None

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
    """
    Analyze a single NEO for anomaly indicators.
    
    Performs comprehensive analysis including orbital mechanics,
    approach patterns, and anomaly detection.
    """
    try:
        logger.info(f"Starting analysis for NEO: {request.designation}")
        start_time = datetime.now()
        
        # Check cache first (unless force refresh requested)
        cache_key = f"{request.designation}_{request.include_raw_data}_{request.include_indicators}"
        if not request.force_refresh and cache_key in _analysis_cache:
            cached_result = _analysis_cache[cache_key]
            logger.info(f"Returning cached analysis for {request.designation}")
            return cached_result
        
        # Perform analysis
        analysis_result = await pipeline.analyze_neo(request.designation)
        
        if not analysis_result:
            raise HTTPException(
                status_code=404, 
                detail=f"NEO {request.designation} not found or analysis failed"
            )
        
        # Build response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = AnalysisResponse(
            designation=request.designation,
            anomaly_score=analysis_result.anomaly_score.__dict__,
            processing_time=processing_time,
            data_quality=analysis_result.data_quality.__dict__,
            orbital_elements=analysis_result.neo_data.orbital_elements.__dict__ if analysis_result.neo_data.orbital_elements else None,
            close_approaches=[ca.__dict__ for ca in analysis_result.neo_data.close_approaches] if analysis_result.neo_data.close_approaches else [],
            raw_neo_data=analysis_result.neo_data.__dict__ if request.include_raw_data else None
        )
        
        # Cache result
        _analysis_cache[cache_key] = response
        
        # Log completion
        background_tasks.add_task(
            _log_analysis_completion,
            request.designation, 
            processing_time,
            analysis_result.anomaly_score.overall_score
        )
        
        logger.info(f"Analysis completed for {request.designation} in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.designation}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get(
    "/detect",
    response_model=DetectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No orbital data for designation"},
        500: {"model": ErrorResponse, "description": "Detection pipeline failure"},
    },
)
async def detect_neo(
    designation: str = Query(..., description="NEO designation, e.g. '2020 SO'"),
    force_refresh: bool = Query(False, description="Bypass cache and re-fetch from source"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Run the canonical ValidatedSigma5 artificial-NEO detector on a named object.

    Fetches orbital elements from the data pipeline and returns a typed
    DetectionResponse with sigma_confidence and artificial_probability.
    """
    try:
        vetoed, veto_reason = _spacecraft_veto(designation)
        if vetoed:
            return DetectionResponse(
                designation=designation,
                is_artificial=True,
                artificial_probability=1.0,
                sigma_confidence=5.0,
                sigma_tier="EXCEPTIONAL",
                classification="ARTIFICIAL",
                confidence=1.0,
                evidence_count=0,
                spacecraft_veto=True,
                veto_reason=veto_reason,
                interpretation=(
                    f"{designation} is a confirmed spacecraft ({veto_reason}). "
                    "Statistical analysis skipped."
                ),
            )
        from aneos_core.data.fetcher import DataFetcher
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector,
        )
        fetcher = DataFetcher()
        neo = fetcher.fetch_neo_data(designation, force_refresh=force_refresh)
        data_source = neo.sources_used[0] if neo.sources_used else None
        data_freshness = neo.fetched_at.isoformat() if neo.fetched_at else datetime.now().isoformat()
        oe = neo.orbital_elements
        if oe is None:
            raise HTTPException(status_code=404, detail=f"No orbital data for {designation}")

        orbital_dict = {
            "a": oe.semi_major_axis,
            "e": oe.eccentricity,
            "i": oe.inclination,
        }
        # Enrich detector with available physical + approach data
        physical_data = None
        if neo.physical_properties:
            pp = neo.physical_properties
            physical_data = {
                "diameter": (pp.diameter_km * 1000) if pp.diameter_km else None,  # km → m
                "albedo": pp.albedo,
                "absolute_magnitude": pp.absolute_magnitude_h,
            }
        close_approach_history = None
        if neo.close_approaches:
            close_approach_history = [
                {
                    "date": ca.close_approach_date.isoformat() if ca.close_approach_date else None,
                    "distance_au": ca.distance_au,
                    "velocity_kms": ca.relative_velocity_km_s,
                }
                for ca in neo.close_approaches
            ]
        # Fetch orbital history from Horizons for course correction analysis
        orbital_history = None
        try:
            from aneos_core.data.sources.horizons import HorizonsSource
            _h = HorizonsSource()
            _raw_hist = _h.fetch_orbital_history(designation, years=10)
            if _raw_hist:
                orbital_history = _raw_hist
        except Exception as _hist_exc:
            logger.debug(f"Horizons orbital history unavailable for {designation}: {_hist_exc}")

        # Build observation_data for propulsion signature analysis
        # Includes non-gravitational acceleration (A2) if fetched from SBDB
        observation_data: Dict[str, Any] = {}
        if neo.nongrav is not None:
            if neo.nongrav.a2 is not None:
                observation_data["non_gravitational_accel"] = abs(neo.nongrav.a2)
            if neo.nongrav.a1 is not None:
                observation_data["nongrav_a1"] = neo.nongrav.a1
            if neo.nongrav.a3 is not None:
                observation_data["nongrav_a3"] = neo.nongrav.a3

        detector = ValidatedSigma5ArtificialNEODetector()
        result = detector.analyze_neo_validated(
            orbital_dict,
            physical_data=physical_data,
            close_approach_history=close_approach_history,
            orbital_history=orbital_history,
            observation_data=observation_data or None,
        )

        classification = "ARTIFICIAL" if result.is_artificial else "NATURAL"
        evidence_summaries = [
            EvidenceSummary(
                type=e.evidence_type.value,
                anomaly_score=e.anomaly_score,
                p_value=e.p_value,
                quality_score=e.quality_score,
                effect_size=e.effect_size,
                confidence_interval=list(e.confidence_interval),
                sample_size=e.sample_size,
                analyzed=getattr(e, 'analyzed', True),
                data_available=getattr(e, 'data_available', True),
            )
            for e in result.evidence_sources
        ]
        tier = _sigma_tier(result.sigma_confidence)
        top_type = evidence_summaries[0].type if evidence_summaries else "orbital_dynamics"
        detection_dict = {
            'designation': designation,
            'is_artificial': result.is_artificial,
            'artificial_probability': result.bayesian_probability,
            'sigma_confidence': result.sigma_confidence,
            'sigma_tier': tier,
            'classification': classification,
            'confidence': min(result.sigma_confidence / 5.0, 1.0),
            'evidence_count': len(result.evidence_sources),
            'data_source': data_source,
            'data_freshness': data_freshness,
        }
        _detection_cache[designation] = detection_dict
        import threading
        threading.Thread(
            target=_persist_detection_result, args=(designation, detection_dict), daemon=True
        ).start()
        return DetectionResponse(
            designation=designation,
            is_artificial=result.is_artificial,
            artificial_probability=result.bayesian_probability,
            sigma_confidence=result.sigma_confidence,
            sigma_tier=tier,
            classification=classification,
            confidence=min(result.sigma_confidence / 5.0, 1.0),
            evidence_count=len(result.evidence_sources),
            evidence_sources=evidence_summaries,
            data_source=data_source,
            data_freshness=data_freshness,
            interpretation=_build_interpretation(
                designation, result.sigma_confidence, tier, top_type, result.bayesian_probability
            ),
            combined_p_value=result.combined_p_value,
            false_discovery_rate=result.false_discovery_rate,
            analysis_metadata=result.analysis_metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed for {designation}: {e}")
        from aneos_core.utils.errors import DataSourceUnavailableError
        if isinstance(e, DataSourceUnavailableError):
            raise HTTPException(status_code=404, detail=f"NEO not found: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get(
    "/impact",
    response_model=ImpactResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No orbital data for designation"},
        500: {"model": ErrorResponse, "description": "Impact calculation failure"},
    },
)
async def impact_neo(
    designation: str = Query(..., description="NEO designation, e.g. '99942'"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Calculate Earth/Moon impact probability for a named NEO.

    Returns collision_probability, moon_collision_probability, risk_level,
    impact_energy_mt, and crater_diameter_km.
    """
    try:
        from aneos_core.data.fetcher import DataFetcher
        from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
        fetcher = DataFetcher()
        neo = fetcher.fetch_neo_data(designation)
        oe = neo.orbital_elements
        if oe is None:
            raise HTTPException(status_code=404, detail=f"No orbital data for {designation}")
        oe.designation = designation
        calc = ImpactProbabilityCalculator()
        result = calc.calculate_comprehensive_impact_probability(
            orbital_elements=oe,
            close_approaches=neo.close_approaches or [],
            physical_properties=neo.physical_properties,
        )
        return ImpactResponse(
            designation=designation,
            collision_probability=result.collision_probability,
            probability_uncertainty=list(result.probability_uncertainty),
            calculation_confidence=result.calculation_confidence,
            moon_collision_probability=result.moon_collision_probability,
            moon_earth_ratio=result.earth_vs_moon_impact_ratio,
            impact_energy_mt=result.impact_energy_mt,
            crater_diameter_km=result.crater_diameter_km,
            damage_radius_km=result.damage_radius_km,
            risk_level=result.risk_level,
            comparative_risk=result.comparative_risk,
            time_to_impact_years=result.time_to_impact_years,
            peak_risk_period=list(result.peak_risk_period) if result.peak_risk_period else None,
            keyhole_passages=result.keyhole_passages,
            primary_risk_factors=result.primary_risk_factors,
            impact_probability_by_decade=result.impact_probability_by_decade,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Impact calculation failed for {designation}: {e}")
        raise HTTPException(status_code=500, detail=f"Impact calculation failed: {str(e)}")


@router.post("/detect", response_model=DetectionResponse)
async def detect_neo_raw(
    request: OrbitalInput,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Run ValidatedSigma5 detector on caller-supplied orbital elements.
    No data source lookup — use when you have elements from your own reduction.
    """
    try:
        designation = request.designation or f"user_{request.a:.3f}_{request.e:.3f}"
        if request.designation:
            vetoed, veto_reason = _spacecraft_veto(request.designation)
            if vetoed:
                return DetectionResponse(
                    designation=designation, is_artificial=True, artificial_probability=1.0,
                    sigma_confidence=5.0, sigma_tier="EXCEPTIONAL", classification="ARTIFICIAL",
                    confidence=1.0, spacecraft_veto=True, veto_reason=veto_reason,
                    interpretation=f"{designation} is a confirmed spacecraft ({veto_reason}).",
                )
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector,
        )
        orbital_dict = {"a": request.a, "e": request.e, "i": request.i}
        physical_data = None
        if request.diameter_km is not None:
            physical_data = {
                "diameter": request.diameter_km * 1000,
                "albedo": request.albedo,
            }
        detector = ValidatedSigma5ArtificialNEODetector()
        result = detector.analyze_neo_validated(
            orbital_dict,
            physical_data=physical_data,
            orbital_history=request.orbital_history,
        )
        classification = "ARTIFICIAL" if result.is_artificial else "NATURAL"
        evidence_summaries = [
            EvidenceSummary(
                type=e.evidence_type.value, anomaly_score=e.anomaly_score,
                p_value=e.p_value, quality_score=e.quality_score, effect_size=e.effect_size,
                confidence_interval=list(e.confidence_interval), sample_size=e.sample_size,
                analyzed=getattr(e, 'analyzed', True),
                data_available=getattr(e, 'data_available', True),
            )
            for e in result.evidence_sources
        ]
        tier = _sigma_tier(result.sigma_confidence)
        top_type = evidence_summaries[0].type if evidence_summaries else "orbital_dynamics"
        return DetectionResponse(
            designation=designation, is_artificial=result.is_artificial,
            artificial_probability=result.bayesian_probability,
            sigma_confidence=result.sigma_confidence, sigma_tier=tier,
            classification=classification, confidence=min(result.sigma_confidence / 5.0, 1.0),
            evidence_count=len(result.evidence_sources), evidence_sources=evidence_summaries,
            data_source="user_provided", data_freshness=datetime.now().isoformat(),
            interpretation=_build_interpretation(
                designation, result.sigma_confidence, tier, top_type, result.bayesian_probability
            ),
            combined_p_value=result.combined_p_value,
            false_discovery_rate=result.false_discovery_rate,
            analysis_metadata=result.analysis_metadata,
        )
    except Exception as e:
        logger.error(f"Raw detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/history", response_model=OrbitalHistoryResponse)
async def orbital_history(
    designation: str = Query(..., description="NEO designation, e.g. 'Apophis'"),
    years: int = Query(10, ge=1, le=50, description="Number of years to span"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Fetch time-series Keplerian orbital elements from JPL Horizons.
    Returns one data point per year over the requested span.
    """
    try:
        from aneos_core.data.sources.horizons import HorizonsSource
        source = HorizonsSource()
        raw_points = source.fetch_orbital_history(designation, years=years)
        if not raw_points:
            raise HTTPException(
                status_code=404,
                detail=f"No orbital history available for {designation}"
            )
        points = [OrbitalElementPoint(**p) for p in raw_points]
        return OrbitalHistoryResponse(
            designation=designation,
            years=years,
            points=points,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orbital history failed for {designation}: {e}")
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")


@router.post("/analyze/batch", response_model=Dict[str, Any])
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Analyze multiple NEOs for artificial signatures (batch mode).
    Uses the same DataFetcher + ValidatedSigma5 path as GET /detect.
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
    background_tasks.add_task(_run_batch_detection, batch_id, request.designations)
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
    for key, result in _analysis_cache.items():
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
                    if r.anomaly_score.get('classification') in request.filters.classification
                ]
            
            if request.filters.min_score is not None:
                filtered_results = [
                    r for r in filtered_results 
                    if r.anomaly_score.get('overall_score', 0) >= request.filters.min_score
                ]
            
            if request.filters.max_score is not None:
                filtered_results = [
                    r for r in filtered_results 
                    if r.anomaly_score.get('overall_score', 1) <= request.filters.max_score
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
    overall_score: float
):
    """Log analysis completion for monitoring."""
    logger.info(
        f"Analysis completed: {designation}, "
        f"Score: {overall_score:.3f}, "
        f"Time: {processing_time:.2f}s"
    )

async def _run_batch_detection(batch_id: str, designations: List[str]):
    """Run ValidatedSigma5 detection concurrently; populate _batch_store."""
    import asyncio
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
        ValidatedSigma5ArtificialNEODetector,
    )
    store = _batch_store[batch_id]
    fetcher = DataFetcher()
    detector = ValidatedSigma5ArtificialNEODetector()

    # fetch_multiple is synchronous blocking I/O — run in a thread to avoid blocking
    # the asyncio event loop (BackgroundTasks run in the same loop as request handlers)
    loop = asyncio.get_event_loop()
    neo_map = await loop.run_in_executor(None, fetcher.fetch_multiple, designations)

    for designation in designations:
        neo = neo_map.get(designation)
        try:
            if neo is None or neo.orbital_elements is None:
                raise ValueError("No orbital elements returned")
            oe = neo.orbital_elements
            orbital_dict = {"a": oe.semi_major_axis, "e": oe.eccentricity, "i": oe.inclination}
            physical_data = None
            if neo.physical_properties:
                pp = neo.physical_properties
                physical_data = {
                    "diameter": (pp.diameter_km * 1000) if pp.diameter_km else None,
                    "albedo": pp.albedo,
                    "absolute_magnitude": pp.absolute_magnitude_h,
                }
            close_approach_history = None
            if neo.close_approaches:
                close_approach_history = [
                    {
                        "date": ca.close_approach_date.isoformat() if ca.close_approach_date else None,
                        "distance_au": ca.distance_au,
                        "velocity_kms": ca.relative_velocity_km_s,
                    }
                    for ca in neo.close_approaches
                ]
            observation_data: Dict[str, Any] = {}
            if neo.nongrav is not None:
                if neo.nongrav.a2 is not None:
                    observation_data["non_gravitational_accel"] = abs(neo.nongrav.a2)
                if neo.nongrav.a1 is not None:
                    observation_data["nongrav_a1"] = neo.nongrav.a1
                if neo.nongrav.a3 is not None:
                    observation_data["nongrav_a3"] = neo.nongrav.a3
            result = detector.analyze_neo_validated(
                orbital_dict,
                physical_data=physical_data,
                close_approach_history=close_approach_history,
                observation_data=observation_data or None,
            )
            _tier = _sigma_tier(result.sigma_confidence)
            _top = result.evidence_sources[0].evidence_type.value if result.evidence_sources else "orbital_dynamics"
            store['results'].append({
                'designation': designation,
                'status': 'success',
                'is_artificial': result.is_artificial,
                'sigma_confidence': result.sigma_confidence,
                'sigma_tier': _tier,
                'artificial_probability': result.bayesian_probability,
                'combined_p_value': result.combined_p_value,
                'false_discovery_rate': result.false_discovery_rate,
                'classification': 'ARTIFICIAL' if result.is_artificial else 'NATURAL',
                'evidence_count': len(result.evidence_sources),
                'evidence_sources': [
                    {
                        'type': e.evidence_type.value,
                        'anomaly_score': e.anomaly_score,
                        'p_value': e.p_value,
                        'quality_score': e.quality_score,
                        'effect_size': e.effect_size,
                    }
                    for e in result.evidence_sources
                ],
                'interpretation': _build_interpretation(
                    designation, result.sigma_confidence, _tier, _top, result.bayesian_probability
                ),
            })
            store['completed'] += 1
        except Exception as e:
            store['failed'] += 1
            store['results'].append({'designation': designation, 'status': 'failed', 'error': str(e)})
    store['status'] = 'completed'
    store['finished_at'] = datetime.now().isoformat()
    logger.info(f"Batch detection {batch_id} completed")

def _run_network_analysis(job_id: str, request: NetworkRequest) -> None:
    """Background worker for population pattern analysis."""
    try:
        _network_store[job_id]["status"] = "processing"
        from aneos_core.data.fetcher import DataFetcher
        from aneos_core.pattern_analysis.session import (
            NetworkAnalysisSession, PatternAnalysisConfig
        )
        fetcher = DataFetcher()
        designations = request.designations
        neo_objects = []
        for des in designations:
            try:
                neo_objects.append(fetcher.fetch_neo_data(des))
            except Exception as e:
                logger.debug(f"Network analysis: could not fetch {des}: {e}")

        cfg = PatternAnalysisConfig(
            clustering=request.clustering,
            harmonics=request.harmonics,
            correlation=request.correlation,
            rendezvous=False,   # deferred ADR-045
            historical_years=request.historical_years,
        )
        session = NetworkAnalysisSession(config=cfg, fetcher=fetcher)
        result = session.run(neo_objects)
        _network_store[job_id].update(result)
        _network_store[job_id]["status"] = "complete"
    except Exception as exc:
        logger.exception(f"Network analysis job {job_id} failed: {exc}")
        _network_store[job_id]["status"] = "error"
        _network_store[job_id]["error"] = str(exc)


@router.post("/analyze/network", summary="Start population pattern analysis")
async def start_network_analysis(request: NetworkRequest):
    """Submit a batch for population-level pattern analysis. Returns a job_id immediately."""
    import threading
    import uuid
    from datetime import datetime as _dt, timezone as _tz
    job_id = f"net_{_dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    _network_store[job_id] = {
        "status": "queued",
        "designations_analyzed": 0,
        "clusters": [],
        "harmonic_signals": [],
        "correlation_matrix": None,
        "network_sigma": 0.0,
        "network_tier": "NETWORK_ROUTINE",
        "combined_p_value": 1.0,
        "sub_module_p_values": {},
        "analysis_metadata": {},
        "error": None,
    }
    thread = threading.Thread(
        target=_run_network_analysis, args=(job_id, request), daemon=True
    )
    thread.start()
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/v1/analysis/network/{job_id}/status",
    }


@router.get("/analyze/network/{job_id}/status",
            response_model=NetworkStatusResponse,
            summary="Poll population pattern analysis status")
async def get_network_status(job_id: str):
    """Return current status / results for a network analysis job."""
    store = _network_store.get(job_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return NetworkStatusResponse(job_id=job_id, **store)


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
            writer.writerow(['designation', 'sigma_confidence', 'artificial_probability', 'classification'])
            for item in items:
                score = item.anomaly_score if isinstance(item.anomaly_score, dict) else {}
                writer.writerow([
                    item.designation,
                    score.get('sigma_confidence', ''),
                    score.get('artificial_probability', ''),
                    score.get('classification', ''),
                ])
            content = buf.getvalue().encode('utf-8')
        else:
            data = [
                {'designation': i.designation, 'anomaly_score': i.anomaly_score}
                for i in items
            ]
            content = _json.dumps(data, default=str).encode('utf-8')
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
