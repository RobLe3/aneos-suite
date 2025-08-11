"""
Administration endpoints for aNEOS API.

Provides administrative functions including user management, system configuration,
model training, and system maintenance operations.
"""

from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, admin endpoints disabled")

try:
    from aneos_core.ml.training import TrainingPipeline, TrainingConfig
    from aneos_core.config.settings import get_config
except ImportError:
    TrainingPipeline = None
    TrainingConfig = None
    get_config = lambda: {'api': {}, 'analysis': {}, 'ml': {}, 'monitoring': {}}
from ..models import (
    TrainingRequest, TrainingResponse, CreateUserRequest, UserResponse,
    ConfigResponse, SystemStatusResponse
)
# Import moved to avoid circular imports
# from ..app import get_aneos_app
from ..auth import get_current_user, require_admin

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

# Training session tracking
_training_sessions: Dict[str, Dict[str, Any]] = {}

async def get_training_pipeline():
    """Get the training pipeline from the application."""
    from ..app import get_aneos_app  # Import here to avoid circular imports
    aneos_app = get_aneos_app()
    return aneos_app.training_pipeline

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: Dict = Depends(require_admin)
):
    """Get comprehensive system status for administrators."""
    try:
        from ..app import get_aneos_app  # Import here to avoid circular imports
        aneos_app = get_aneos_app()
        health_status = aneos_app.get_health_status()
        
        # Calculate uptime
        uptime_seconds = 0
        if aneos_app.startup_time:
            uptime_seconds = (datetime.now() - aneos_app.startup_time).total_seconds()
        
        return SystemStatusResponse(
            status=health_status['status'],
            uptime_seconds=uptime_seconds,
            services=health_status['services'],
            version=health_status['version'],
            deployment_info={
                'environment': 'development',  # Would be configurable
                'deployment_date': aneos_app.startup_time.isoformat() if aneos_app.startup_time else None,
                'git_commit': 'unknown',  # Would be injected during build
                'build_number': 'unknown'
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/config", response_model=ConfigResponse)
async def get_system_config(
    current_user: Dict = Depends(require_admin)
):
    """Get current system configuration."""
    try:
        config = get_config()
        
        return ConfigResponse(
            api_config={
                'max_concurrent_requests': config.get('api', {}).get('max_concurrent_requests', 100),
                'request_timeout': config.get('api', {}).get('request_timeout', 300),
                'rate_limit': config.get('api', {}).get('rate_limit', 1000),
                'cors_origins': config.get('api', {}).get('cors_origins', ['*'])
            },
            analysis_config={
                'max_batch_size': config.get('analysis', {}).get('max_batch_size', 100),
                'cache_ttl': config.get('analysis', {}).get('cache_ttl', 3600),
                'timeout': config.get('analysis', {}).get('timeout', 60),
                'retry_attempts': config.get('analysis', {}).get('retry_attempts', 3)
            },
            ml_config={
                'model_cache_size': config.get('ml', {}).get('model_cache_size', 10),
                'feature_cache_ttl': config.get('ml', {}).get('feature_cache_ttl', 1800),
                'ensemble_threshold': config.get('ml', {}).get('ensemble_threshold', 0.7),
                'training_data_size': config.get('ml', {}).get('training_data_size', 1000)
            },
            monitoring_config={
                'metrics_interval': config.get('monitoring', {}).get('metrics_interval', 60),
                'alert_cooldown': config.get('monitoring', {}).get('alert_cooldown', 300),
                'log_level': config.get('monitoring', {}).get('log_level', 'INFO'),
                'retention_days': config.get('monitoring', {}).get('retention_days', 30)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system config: {str(e)}")

@router.post("/config", response_model=Dict[str, Any])
async def update_system_config(
    config_updates: Dict[str, Any],
    current_user: Dict = Depends(require_admin)
):
    """Update system configuration (requires restart for some changes)."""
    try:
        # Validate configuration updates
        valid_sections = ['api', 'analysis', 'ml', 'monitoring']
        
        for section in config_updates:
            if section not in valid_sections:
                raise HTTPException(status_code=400, detail=f"Invalid config section: {section}")
        
        # Apply configuration updates (mock implementation)
        # In production, this would update the actual configuration
        logger.info(f"Configuration updated by {current_user['username']}: {config_updates}")
        
        return {
            'status': 'success',
            'updated_sections': list(config_updates.keys()),
            'updated_by': current_user['username'],
            'updated_at': datetime.now().isoformat(),
            'restart_required': True  # Some config changes require restart
        }
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.post("/training/start", response_model=TrainingResponse)
async def start_model_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    training_pipeline = Depends(get_training_pipeline),
    current_user: Dict = Depends(require_admin)
):
    """Start ML model training with specified parameters."""
    try:
        if not training_pipeline:
            raise HTTPException(status_code=503, detail="Training pipeline not available")
        logger.info(f"Starting model training requested by {current_user['username']}")
        
        # Create training session
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_session = {
            'session_id': session_id,
            'status': 'starting',
            'models_trained': [],
            'started_by': current_user['username'],
            'started_at': datetime.now(),
            'designations': request.designations,
            'model_types': request.model_types,
            'use_ensemble': request.use_ensemble,
            'hyperparameter_optimization': request.hyperparameter_optimization
        }
        
        _training_sessions[session_id] = training_session
        
        # Start training in background
        background_tasks.add_task(
            _execute_model_training,
            session_id,
            request,
            training_pipeline
        )
        
        return TrainingResponse(
            session_id=session_id,
            status='starting',
            models_trained=[],
            training_score=None,
            validation_score=None,
            training_time=None,
            model_paths=[]
        )
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/training/{session_id}/status", response_model=TrainingResponse)
async def get_training_status(
    session_id: str,
    current_user: Dict = Depends(require_admin)
):
    """Get status of a training session."""
    if session_id not in _training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = _training_sessions[session_id]
    
    return TrainingResponse(
        session_id=session_id,
        status=session['status'],
        models_trained=session['models_trained'],
        training_score=session.get('training_score'),
        validation_score=session.get('validation_score'),
        training_time=session.get('training_time'),
        model_paths=session.get('model_paths', [])
    )

@router.get("/training/sessions", response_model=List[TrainingResponse])
async def list_training_sessions(
    current_user: Dict = Depends(require_admin)
):
    """List all training sessions."""
    return [
        TrainingResponse(
            session_id=session_id,
            status=session['status'],
            models_trained=session['models_trained'],
            training_score=session.get('training_score'),
            validation_score=session.get('validation_score'),
            training_time=session.get('training_time'),
            model_paths=session.get('model_paths', [])
        )
        for session_id, session in _training_sessions.items()
    ]

@router.delete("/training/{session_id}", response_model=Dict[str, Any])
async def cancel_training_session(
    session_id: str,
    current_user: Dict = Depends(require_admin)
):
    """Cancel a running training session."""
    if session_id not in _training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = _training_sessions[session_id]
    
    if session['status'] not in ['starting', 'training']:
        raise HTTPException(status_code=400, detail="Training session cannot be cancelled")
    
    # Cancel training (mock implementation)
    session['status'] = 'cancelled'
    session['cancelled_by'] = current_user['username']
    session['cancelled_at'] = datetime.now()
    
    return {
        'status': 'success',
        'session_id': session_id,
        'cancelled_by': current_user['username'],
        'cancelled_at': datetime.now().isoformat()
    }

@router.post("/users", response_model=UserResponse)
async def create_user(
    request: CreateUserRequest,
    current_user: Dict = Depends(require_admin)
):
    """Create a new user account."""
    try:
        # Mock user creation (would integrate with actual auth system)
        user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        new_user = UserResponse(
            user_id=user_id,
            username=request.username,
            email=request.email,
            role=request.role,
            created_at=datetime.now(),
            last_login=None,
            is_active=True
        )
        
        logger.info(f"User {request.username} created by {current_user['username']}")
        
        return new_user
        
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: Dict = Depends(require_admin)
):
    """List all user accounts."""
    # Mock user list (would query actual user database)
    return [
        UserResponse(
            user_id="admin_001",
            username="admin",
            email="admin@aneos.local",
            role="admin",
            created_at=datetime.now(),
            last_login=datetime.now(),
            is_active=True
        ),
        UserResponse(
            user_id="analyst_001",
            username="analyst",
            email="analyst@aneos.local",
            role="analyst",
            created_at=datetime.now(),
            last_login=datetime.now(),
            is_active=True
        )
    ]

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    updates: Dict[str, Any],
    current_user: Dict = Depends(require_admin)
):
    """Update user account details."""
    try:
        # Mock user update (would update actual user database)
        logger.info(f"User {user_id} updated by {current_user['username']}: {updates}")
        
        return UserResponse(
            user_id=user_id,
            username=updates.get('username', 'updated_user'),
            email=updates.get('email', 'updated@aneos.local'),
            role=updates.get('role', 'viewer'),
            created_at=datetime.now(),
            last_login=datetime.now(),
            is_active=updates.get('is_active', True)
        )
        
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

@router.delete("/users/{user_id}", response_model=Dict[str, Any])
async def delete_user(
    user_id: str,
    current_user: Dict = Depends(require_admin)
):
    """Delete user account."""
    try:
        # Mock user deletion (would delete from actual user database)
        logger.info(f"User {user_id} deleted by {current_user['username']}")
        
        return {
            'status': 'success',
            'user_id': user_id,
            'deleted_by': current_user['username'],
            'deleted_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")

@router.post("/maintenance/cache/clear", response_model=Dict[str, Any])
async def clear_system_cache(
    current_user: Dict = Depends(require_admin)
):
    """Clear all system caches."""
    try:
        from ..app import get_aneos_app  # Import here to avoid circular imports
        aneos_app = get_aneos_app()
        
        # Clear analysis cache
        if aneos_app.analysis_pipeline:
            await aneos_app.analysis_pipeline.clear_cache()
        
        # Clear ML prediction cache
        if aneos_app.ml_predictor:
            await aneos_app.ml_predictor.clear_cache()
        
        logger.info(f"System cache cleared by {current_user['username']}")
        
        return {
            'status': 'success',
            'cleared_by': current_user['username'],
            'cleared_at': datetime.now().isoformat(),
            'cache_types': ['analysis', 'prediction', 'features']
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.post("/maintenance/gc", response_model=Dict[str, Any])
async def force_garbage_collection(
    current_user: Dict = Depends(require_admin)
):
    """Force Python garbage collection."""
    try:
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        logger.info(f"Garbage collection forced by {current_user['username']}: {collected} objects collected")
        
        return {
            'status': 'success',
            'objects_collected': collected,
            'triggered_by': current_user['username'],
            'triggered_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to force garbage collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force garbage collection: {str(e)}")

@router.get("/logs", response_model=Dict[str, Any])
async def get_system_logs(
    lines: int = 100,
    level: str = 'INFO',
    current_user: Dict = Depends(require_admin)
):
    """Get recent system logs."""
    try:
        # Mock log retrieval (would read from actual log files)
        logs = [
            f"2025-08-04 10:00:00 INFO: System started successfully",
            f"2025-08-04 10:01:00 INFO: Analysis pipeline initialized",
            f"2025-08-04 10:02:00 INFO: ML predictor ready",
            f"2025-08-04 10:03:00 INFO: API server listening on port 8000",
            f"2025-08-04 10:04:00 DEBUG: Cache hit rate: 85%"
        ]
        
        return {
            'logs': logs[-lines:],
            'total_lines': len(logs),
            'level_filter': level,
            'retrieved_by': current_user['username'],
            'retrieved_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system logs: {str(e)}")

# Background task functions
async def _execute_model_training(
    session_id: str,
    request: TrainingRequest,
    training_pipeline: TrainingPipeline
):
    """Execute model training in background."""
    try:
        session = _training_sessions[session_id]
        session['status'] = 'training'
        
        # Mock training process (would use actual training pipeline)
        logger.info(f"Starting training session {session_id}")
        
        # Simulate training time
        await asyncio.sleep(30)  # Mock 30 second training
        
        # Update session with results
        session['status'] = 'completed'
        session['models_trained'] = request.model_types
        session['training_score'] = 0.92  # Mock score
        session['validation_score'] = 0.89  # Mock score
        session['training_time'] = 30.0
        session['model_paths'] = [f'/models/{model_type}_model.pkl' for model_type in request.model_types]
        session['completed_at'] = datetime.now()
        
        logger.info(f"Training session {session_id} completed successfully")
        
    except Exception as e:
        session = _training_sessions[session_id]
        session['status'] = 'failed'
        session['error'] = str(e)
        session['failed_at'] = datetime.now()
        logger.error(f"Training session {session_id} failed: {e}")