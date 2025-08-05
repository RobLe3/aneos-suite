"""
ML Prediction endpoints for aNEOS API.

Provides machine learning prediction services including real-time anomaly
prediction, model management, and feature analysis.
"""

from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, prediction endpoints disabled")

try:
    from aneos_core.ml.prediction import RealTimePredictor, PredictionResult
    from aneos_core.ml.models import ModelEnsemble
except ImportError:
    RealTimePredictor = None
    PredictionResult = None
    ModelEnsemble = None
from ..models import (
    PredictionRequest, BatchPredictionRequest, PredictionResponse,
    FeatureContributionResponse
)
# Import moved to avoid circular imports
# from ..app import get_aneos_app
from ..auth import get_current_user

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

# Prediction cache
_prediction_cache: Dict[str, PredictionResponse] = {}
_batch_predictions: Dict[str, Dict[str, Any]] = {}

async def get_ml_predictor() -> RealTimePredictor:
    """Get the ML predictor from the application."""
    from ..app import get_aneos_app  # Import here to avoid circular imports
    aneos_app = get_aneos_app()
    if not aneos_app.ml_predictor:
        raise HTTPException(status_code=503, detail="ML predictor not available")
    return aneos_app.ml_predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Predict anomaly score for a single NEO using ML models.
    
    Uses trained ensemble models to provide real-time anomaly predictions
    with confidence scores and feature contributions.
    """
    try:
        logger.info(f"Starting ML prediction for NEO: {request.designation}")
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"{request.designation}_{request.model_id}"
        if request.use_cache and cache_key in _prediction_cache:
            cached_result = _prediction_cache[cache_key]
            logger.info(f"Returning cached prediction for {request.designation}")
            return cached_result
        
        # Perform ML prediction
        prediction_result = await predictor.predict_anomaly(request.designation)
        
        if not prediction_result:
            raise HTTPException(
                status_code=404,
                detail=f"NEO {request.designation} not found or prediction failed"
            )
        
        # Build feature contributions
        feature_contributions = []
        if hasattr(prediction_result, 'feature_contributions'):
            for feature_name, contribution in prediction_result.feature_contributions.items():
                feature_contributions.append(FeatureContributionResponse(
                    feature_name=feature_name,
                    contribution=contribution['contribution'],
                    feature_value=contribution['value']
                ))
        
        # Build response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = PredictionResponse(
            designation=request.designation,
            anomaly_score=prediction_result.anomaly_score,
            anomaly_probability=prediction_result.anomaly_probability,
            is_anomaly=prediction_result.is_anomaly,
            confidence=prediction_result.confidence,
            model_id=prediction_result.model_id,
            feature_contributions=feature_contributions,
            model_predictions=prediction_result.model_predictions,
            execution_time_ms=processing_time * 1000
        )
        
        # Cache result
        if request.use_cache:
            _prediction_cache[cache_key] = response
        
        # Log prediction
        background_tasks.add_task(
            _log_prediction_completion,
            request.designation,
            processing_time,
            prediction_result.anomaly_score
        )
        
        logger.info(f"ML prediction completed for {request.designation} in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"ML prediction failed for {request.designation}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=Dict[str, Any])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Predict anomaly scores for multiple NEOs in batch mode.
    
    Processes multiple NEO predictions concurrently with progress tracking.
    """
    try:
        logger.info(f"Starting batch prediction for {len(request.designations)} NEOs")
        
        # Create batch job
        batch_id = f"predict_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_status = {
            'batch_id': batch_id,
            'status': 'processing',
            'total': len(request.designations),
            'completed': 0,
            'failed': 0,
            'results': [],
            'started_at': datetime.now(),
            'model_id': request.model_id
        }
        
        _batch_predictions[batch_id] = batch_status
        
        # Start batch processing in background
        background_tasks.add_task(
            _process_batch_predictions,
            batch_id,
            request.designations,
            request.use_cache,
            request.model_id,
            predictor
        )
        
        return {
            'batch_id': batch_id,
            'status': 'processing',
            'total_neos': len(request.designations),
            'estimated_completion': '2-10 minutes',
            'progress_url': f'/api/v1/prediction/batch/{batch_id}/status'
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/batch/{batch_id}/status", response_model=Dict[str, Any])
async def get_batch_prediction_status(
    batch_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get status of a batch prediction job."""
    if batch_id not in _batch_predictions:
        raise HTTPException(status_code=404, detail="Batch prediction job not found")
    
    return _batch_predictions[batch_id]

@router.get("/batch/{batch_id}/results", response_model=List[PredictionResponse])
async def get_batch_prediction_results(
    batch_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get results of a completed batch prediction job."""
    if batch_id not in _batch_predictions:
        raise HTTPException(status_code=404, detail="Batch prediction job not found")
    
    batch_status = _batch_predictions[batch_id]
    if batch_status['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Batch prediction not completed")
    
    return batch_status['results']

@router.get("/models", response_model=Dict[str, Any])
async def list_available_models(
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """List available ML models and their status."""
    try:
        models_info = await predictor.get_models_info()
        
        return {
            'available_models': models_info,
            'active_model': predictor.config.default_model_id,
            'ensemble_enabled': predictor.config.use_ensemble,
            'model_versions': predictor.get_model_versions()
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_id}/info", response_model=Dict[str, Any])
async def get_model_info(
    model_id: str,
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get detailed information about a specific model."""
    try:
        model_info = await predictor.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/models/{model_id}/activate", response_model=Dict[str, Any])
async def activate_model(
    model_id: str,
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Activate a specific model for predictions."""
    try:
        success = await predictor.activate_model(model_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to activate model {model_id}")
        
        return {
            'status': 'success',
            'active_model': model_id,
            'activated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to activate model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")

@router.get("/features/{designation}", response_model=Dict[str, Any])
async def get_neo_features(
    designation: str,
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get extracted features for a NEO without making predictions."""
    try:
        features = await predictor.extract_features(designation)
        
        if not features:
            raise HTTPException(
                status_code=404,
                detail=f"Features not available for NEO {designation}"
            )
        
        return {
            'designation': designation,
            'features': features.features,
            'feature_names': features.feature_names,
            'feature_quality': features.quality_score,
            'extraction_time': features.extraction_time,
            'total_features': len(features.features)
        }
        
    except Exception as e:
        logger.error(f"Feature extraction failed for {designation}: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_prediction_performance(
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get ML prediction performance metrics."""
    try:
        performance_metrics = predictor.get_performance_metrics()
        
        return {
            'cache_stats': {
                'cached_predictions': len(_prediction_cache),
                'cache_hit_rate': performance_metrics.get('cache_hit_rate', 0),
                'cache_size_mb': performance_metrics.get('cache_size_mb', 0)
            },
            'prediction_stats': {
                'total_predictions': performance_metrics.get('total_predictions', 0),
                'average_latency_ms': performance_metrics.get('average_latency_ms', 0),
                'predictions_per_hour': performance_metrics.get('predictions_per_hour', 0),
                'model_agreement_rate': performance_metrics.get('model_agreement_rate', 0)
            },
            'model_stats': {
                'active_models': performance_metrics.get('active_models', 0),
                'ensemble_size': performance_metrics.get('ensemble_size', 0),
                'feature_count': performance_metrics.get('feature_count', 0),
                'last_training': performance_metrics.get('last_training')
            },
            'batch_jobs': {
                'active_batches': len([b for b in _batch_predictions.values() if b['status'] == 'processing']),
                'completed_batches': len([b for b in _batch_predictions.values() if b['status'] == 'completed']),
                'total_batch_jobs': len(_batch_predictions)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.delete("/cache", response_model=Dict[str, Any])
async def clear_prediction_cache(
    predictor: RealTimePredictor = Depends(get_ml_predictor),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Clear the prediction cache."""
    try:
        # Clear local cache
        cache_size = len(_prediction_cache)
        _prediction_cache.clear()
        
        # Clear predictor cache
        await predictor.clear_cache()
        
        return {
            'status': 'success',
            'cleared_predictions': cache_size,
            'cleared_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# Background task functions
async def _log_prediction_completion(designation: str, processing_time: float, anomaly_score: float):
    """Log prediction completion for metrics."""
    logger.info(f"Prediction logged: {designation} - {processing_time:.2f}s - Score: {anomaly_score:.3f}")

async def _process_batch_predictions(
    batch_id: str,
    designations: List[str],
    use_cache: bool,
    model_id: Optional[str],
    predictor: RealTimePredictor
):
    """Process batch predictions in background."""
    try:
        batch_status = _batch_predictions[batch_id]
        results = []
        
        for designation in designations:
            try:
                # Perform prediction
                prediction_result = await predictor.predict_anomaly(designation)
                
                if prediction_result:
                    # Convert to response format
                    response = PredictionResponse(
                        designation=designation,
                        anomaly_score=prediction_result.anomaly_score,
                        anomaly_probability=prediction_result.anomaly_probability,
                        is_anomaly=prediction_result.is_anomaly,
                        confidence=prediction_result.confidence,
                        model_id=prediction_result.model_id,
                        feature_contributions=[],
                        model_predictions=prediction_result.model_predictions
                    )
                    results.append(response)
                    batch_status['completed'] += 1
                else:
                    batch_status['failed'] += 1
                
                # Update progress
                progress = ((batch_status['completed'] + batch_status['failed']) / batch_status['total']) * 100
                batch_status['progress'] = progress
                
                logger.info(f"Batch prediction {batch_id}: {designation} processed ({progress:.1f}%)")
                
            except Exception as e:
                batch_status['failed'] += 1
                logger.error(f"Batch prediction {batch_id}: {designation} failed - {e}")
        
        batch_status['status'] = 'completed'
        batch_status['results'] = results
        batch_status['completed_at'] = datetime.now()
        
        logger.info(f"Batch prediction {batch_id} completed: {len(results)} successful, {batch_status['failed']} failed")
        
    except Exception as e:
        batch_status['status'] = 'failed'
        batch_status['error'] = str(e)
        logger.error(f"Batch prediction {batch_id} failed: {e}")