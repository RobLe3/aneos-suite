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
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

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
import json
from pathlib import Path
from ..models import (
    TrainingRequest, TrainingResponse, CreateUserRequest, UserResponse,
    ConfigResponse, SystemStatusResponse
)
# Import moved to avoid circular imports
# from ..app import get_aneos_app
from ..auth import get_current_user, require_admin
from ..database import User as DBUser, get_database, HAS_SQLALCHEMY

CONFIG_FILE = Path("aneos_config_override.json")
LOG_PATH = Path("logs/aneos.log")

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
            uptime_seconds = (datetime.now(UTC) - aneos_app.startup_time).total_seconds()
        
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

@router.get("/config")
async def get_system_config(
    current_user: Dict = Depends(require_admin)
):
    """Get current system configuration (includes persisted overrides)."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
        config = get_config()
        return {
            'api': config.get('api', {}),
            'analysis': config.get('analysis', {}),
            'ml': config.get('ml', {}),
            'monitoring': config.get('monitoring', {}),
        }
    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system config: {str(e)}")

@router.post("/config", response_model=Dict[str, Any])
async def update_system_config(
    config_updates: Dict[str, Any],
    current_user: Dict = Depends(require_admin)
):
    """Update and persist system configuration."""
    try:
        CONFIG_FILE.write_text(json.dumps(config_updates, indent=2))
        logger.info(f"Configuration persisted by {current_user['username']}: {list(config_updates.keys())}")
        return {
            'status': 'saved',
            'updated_sections': list(config_updates.keys()),
            'updated_by': current_user['username'],
            'updated_at': datetime.now(UTC).isoformat(),
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
        user_id = f"user_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        if HAS_SQLALCHEMY:
            from ..database import SessionLocal
            db = SessionLocal()
            try:
                db_user = DBUser(
                    user_id=user_id,
                    username=request.username,
                    email=request.email,
                    role=request.role,
                )
                db.add(db_user)
                db.commit()
                db.refresh(db_user)
                user_id = db_user.user_id or user_id
                created_at = db_user.created_at or datetime.now(UTC)
            finally:
                db.close()
        else:
            created_at = datetime.now(UTC)

        logger.info(f"User {request.username} created by {current_user['username']}")
        return UserResponse(
            user_id=user_id,
            username=request.username,
            email=request.email,
            role=request.role,
            created_at=created_at,
            last_login=None,
            is_active=True,
        )
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: Dict = Depends(require_admin)
):
    """List all user accounts."""
    if HAS_SQLALCHEMY:
        from ..database import SessionLocal
        db = SessionLocal()
        try:
            users = db.query(DBUser).all()
            return [
                UserResponse(
                    user_id=u.user_id or str(u.id),
                    username=u.username,
                    email=u.email or "",
                    role=u.role or "viewer",
                    created_at=u.created_at or datetime.now(UTC),
                    last_login=u.last_login,
                    is_active=u.is_active if u.is_active is not None else True,
                )
                for u in users
            ]
        finally:
            db.close()
    return []

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
            if hasattr(aneos_app.analysis_pipeline, 'cache_manager'):
                aneos_app.analysis_pipeline.cache_manager.clear()

        # Clear ML prediction cache
        if hasattr(aneos_app, 'ml_predictor') and aneos_app.ml_predictor:
            if hasattr(aneos_app.ml_predictor, 'clear_cache'):
                aneos_app.ml_predictor.clear_cache()
        
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
        if LOG_PATH.exists():
            content = LOG_PATH.read_text(errors='replace').splitlines()
            log_lines = content[-lines:]
            total = len(content)
        else:
            log_lines = []
            total = 0
        return {
            'lines': log_lines,
            'total_lines': total,
            'level_filter': level,
            'retrieved_by': current_user['username'],
            'retrieved_at': datetime.now(UTC).isoformat(),
            'note': None if LOG_PATH.exists() else f"No log file at {LOG_PATH}",
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