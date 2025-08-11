"""
Enhanced Analysis Endpoints for aNEOS Scientific Rigor Validation.

This module provides enhanced analysis endpoints with comprehensive validation
while preserving all existing endpoints unchanged for backward compatibility.

Enhanced endpoints include:
- /api/v1/analysis/enhanced - Enhanced single analysis with validation
- /api/v1/analysis/enhanced/batch - Enhanced batch analysis
- /api/v1/analysis/enhanced/summary - Validation summary statistics
"""

from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, enhanced analysis endpoints disabled")

# Import enhanced components
try:
    from aneos_core.analysis.enhanced_pipeline import EnhancedAnalysisPipeline, create_enhanced_pipeline
    from aneos_core.analysis.pipeline import AnalysisPipeline
    from ..enhanced_models import (
        EnhancedAnalysisRequest, 
        EnhancedAnalysisResponse,
        EnhancedBatchAnalysisResponse,
        ValidationSummaryResponse
    )
    from ..auth import get_current_user
    HAS_ENHANCED_ANALYSIS = True
except ImportError:
    HAS_ENHANCED_ANALYSIS = False

logger = logging.getLogger(__name__)

# Create router for enhanced endpoints
enhanced_router = APIRouter() if HAS_FASTAPI else None

# Global enhanced pipeline instance (lazy initialization)
_enhanced_pipeline: Optional[EnhancedAnalysisPipeline] = None

async def get_neo_data(designation: str):
    """Get NEO data (placeholder - would implement actual data retrieval)."""
    # Placeholder implementation
    class MockNEOData:
        def __init__(self, designation):
            self.designation = designation
    return MockNEOData(designation)

async def get_analysis_pipeline() -> AnalysisPipeline:
    """Get the analysis pipeline (placeholder)."""
    # This would be imported from the main analysis endpoints
    # For now, return a mock
    class MockPipeline:
        async def analyze_neo(self, designation: str, neo_data=None):
            class MockResult:
                def __init__(self):
                    self.overall_score = 0.75
                    self.confidence = 0.85
                    self.classification = "potentially_artificial"
            return MockResult()
    return MockPipeline()

async def get_enhanced_analysis_pipeline() -> EnhancedAnalysisPipeline:
    """Get enhanced analysis pipeline with lazy initialization."""
    global _enhanced_pipeline
    
    if _enhanced_pipeline is None:
        # Get original pipeline
        original_pipeline = await get_analysis_pipeline()
        
        # Create enhanced wrapper
        _enhanced_pipeline = create_enhanced_pipeline(
            original_pipeline=original_pipeline,
            enable_validation=True
        )
        
        logger.info("Enhanced analysis pipeline initialized")
    
    return _enhanced_pipeline

if HAS_ENHANCED_ANALYSIS and HAS_FASTAPI:
    
    @enhanced_router.post("/enhanced", response_model=EnhancedAnalysisResponse)
    async def analyze_neo_enhanced(
        request: EnhancedAnalysisRequest,
        background_tasks: BackgroundTasks,
        pipeline: EnhancedAnalysisPipeline = Depends(get_enhanced_analysis_pipeline),
        current_user: Optional[Dict] = Depends(get_current_user)
    ):
        """
        Enhanced NEO analysis with comprehensive scientific rigor validation.
        
        This endpoint provides the same analysis as /analyze but with additional:
        - 5-stage validation pipeline for false positive prevention
        - Statistical significance testing with multiple testing corrections  
        - Monte Carlo uncertainty quantification
        - Space debris catalog cross-matching
        - Physical plausibility assessment
        
        The original /analyze endpoint remains completely unchanged.
        """
        try:
            start_time = datetime.now()
            
            # Validate enhanced pipeline status
            validation_status = pipeline.get_validation_status()
            if not validation_status['validation_available']:
                logger.warning("Enhanced validation unavailable, falling back to basic analysis")
            
            # Get NEO data
            neo_data = await get_neo_data(request.designation)
            if not neo_data and not request.force_refresh:
                raise HTTPException(
                    status_code=404, 
                    detail=f"NEO {request.designation} not found"
                )
            
            # Run enhanced analysis
            logger.info(f"Starting enhanced analysis for {request.designation}")
            enhanced_result = await pipeline.analyze_neo_with_validation(
                request.designation, 
                neo_data
            )
            
            if not enhanced_result:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Enhanced analysis failed for {request.designation}"
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Convert to API response model
            original_result = enhanced_result.original_result
            
            response = EnhancedAnalysisResponse(
                # Original analysis fields
                designation=request.designation,
                overall_score=getattr(original_result, 'overall_score', 0.0),
                confidence=getattr(original_result, 'confidence', 0.0),
                classification=getattr(original_result, 'classification', 'unknown'),
                processing_time_ms=processing_time,
                timestamp=start_time,
                
                # Enhanced validation fields
                validation_result=enhanced_result.validation_result,
                statistical_tests=enhanced_result.statistical_tests,
                uncertainty_analysis=enhanced_result.uncertainty_analysis,
                
                # Enhancement metadata
                enhancement_timestamp=enhanced_result.enhancement_timestamp,
                enhancement_version=enhanced_result.enhancement_version
            )
            
            # Add background tasks for result logging
            background_tasks.add_task(
                _log_enhanced_analysis_result, 
                request.designation, 
                enhanced_result,
                processing_time
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Enhanced analysis failed for {request.designation}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Enhanced analysis failed: {str(e)}"
            )
    
    @enhanced_router.post("/enhanced/batch", response_model=EnhancedBatchAnalysisResponse)
    async def batch_analyze_enhanced(
        designations: List[str],
        background_tasks: BackgroundTasks,
        force_refresh: bool = False,
        enable_validation: bool = True,
        pipeline: EnhancedAnalysisPipeline = Depends(get_enhanced_analysis_pipeline),
        current_user: Optional[Dict] = Depends(get_current_user)
    ):
        """
        Enhanced batch analysis with scientific rigor validation.
        
        Processes multiple NEOs with enhanced validation in parallel.
        """
        try:
            start_time = datetime.now()
            batch_id = f"enhanced_batch_{int(start_time.timestamp())}"
            
            logger.info(f"Starting enhanced batch analysis for {len(designations)} NEOs")
            
            # Process in parallel using the enhanced pipeline
            enhanced_results = await pipeline.bulk_analyze_with_validation(
                designations=designations
            )
            
            # Convert results to API responses
            api_responses = []
            for designation, enhanced_result in zip(designations, enhanced_results):
                if enhanced_result and hasattr(enhanced_result, 'original_result'):
                    original_result = enhanced_result.original_result
                    
                    api_response = EnhancedAnalysisResponse(
                        designation=designation,
                        overall_score=getattr(original_result, 'overall_score', 0.0),
                        confidence=getattr(original_result, 'confidence', 0.0),
                        classification=getattr(original_result, 'classification', 'unknown'),
                        processing_time_ms=0.0,  # Individual times not tracked in batch
                        timestamp=start_time,
                        validation_result=enhanced_result.validation_result,
                        statistical_tests=enhanced_result.statistical_tests,
                        uncertainty_analysis=enhanced_result.uncertainty_analysis,
                        enhancement_timestamp=enhanced_result.enhancement_timestamp,
                        enhancement_version=enhanced_result.enhancement_version
                    )
                    api_responses.append(api_response)
            
            # Generate summary statistics
            summary_stats = pipeline.get_validation_summary(enhanced_results)
            
            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = EnhancedBatchAnalysisResponse(
                batch_id=batch_id,
                results=api_responses,
                summary_statistics=summary_stats,
                total_processing_time_ms=total_processing_time,
                timestamp=start_time
            )
            
            # Background logging
            background_tasks.add_task(
                _log_batch_analysis_result,
                batch_id,
                len(designations),
                len(api_responses),
                total_processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced batch analysis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Enhanced batch analysis failed: {str(e)}"
            )
    
    @enhanced_router.get("/enhanced/summary", response_model=ValidationSummaryResponse)
    async def get_validation_summary(
        days: int = Query(default=7, ge=1, le=365, description="Number of days to include in summary"),
        pipeline: EnhancedAnalysisPipeline = Depends(get_enhanced_analysis_pipeline),
        current_user: Optional[Dict] = Depends(get_current_user)
    ):
        """
        Get validation summary statistics over specified time period.
        
        Provides aggregate statistics on validation results, false positive rates,
        and system performance for monitoring and calibration purposes.
        """
        try:
            validation_status = pipeline.get_validation_status()
            
            # In production, this would query stored results from database
            summary = ValidationSummaryResponse(
                total_analyses=0,  # Would be queried from database
                false_positive_statistics={
                    "mean_probability": 0.0,
                    "median_probability": 0.0,
                    "high_fp_risk_count": 0,
                    "low_fp_risk_count": 0
                },
                recommendations={
                    "accept": 0,
                    "reject": 0,
                    "expert_review": 0,
                    "unknown": 0
                },
                overall_score_statistics={
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0
                },
                validation_system_status=validation_status
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Validation summary failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Validation summary failed: {str(e)}"
            )

# Background task functions
async def _log_enhanced_analysis_result(
    designation: str,
    enhanced_result: Any,
    processing_time: float
):
    """Log enhanced analysis result for monitoring."""
    try:
        fp_probability = enhanced_result.validation_result.overall_false_positive_probability
        recommendation = enhanced_result.validation_result.recommendation
        
        logger.info(
            f"Enhanced analysis logged: {designation}, "
            f"FP={fp_probability:.3f}, "
            f"Recommendation={recommendation}, "
            f"Time={processing_time:.1f}ms"
        )
        
    except Exception as e:
        logger.error(f"Failed to log enhanced analysis result: {e}")

async def _log_batch_analysis_result(
    batch_id: str,
    requested_count: int,
    completed_count: int,
    processing_time: float
):
    """Log batch analysis result for monitoring."""
    try:
        success_rate = completed_count / requested_count if requested_count > 0 else 0.0
        
        logger.info(
            f"Enhanced batch analysis logged: {batch_id}, "
            f"Success rate={success_rate:.1%}, "
            f"Time={processing_time:.1f}ms"
        )
        
    except Exception as e:
        logger.error(f"Failed to log batch analysis result: {e}")

# Export the router if available
router = enhanced_router if HAS_ENHANCED_ANALYSIS and HAS_FASTAPI else None