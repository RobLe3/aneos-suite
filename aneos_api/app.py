"""
Main aNEOS API application with FastAPI framework.

This module provides the core API application with all endpoints,
middleware, and service integrations.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.openapi.utils import get_openapi
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, API services will be disabled")

try:
    from aneos_core.analysis.pipeline import AnalysisPipeline, create_analysis_pipeline
    HAS_ANALYSIS = True
except ImportError:
    AnalysisPipeline = None
    create_analysis_pipeline = None
    HAS_ANALYSIS = False

# Lazy import ML components to avoid PyTorch dependency issues
HAS_ML = False
RealTimePredictor = None
PredictionConfig = None
TrainingPipeline = None
TrainingConfig = None

def _load_ml_components():
    """Lazy load ML components only when needed."""
    global HAS_ML, RealTimePredictor, PredictionConfig, TrainingPipeline, TrainingConfig
    if not HAS_ML:
        try:
            from aneos_core.ml.prediction import RealTimePredictor, PredictionConfig
            from aneos_core.ml.training import TrainingPipeline, TrainingConfig
            HAS_ML = True
        except ImportError as e:
            logging.warning(f"ML components not available: {e}")
            HAS_ML = False
    return HAS_ML

try:
    from aneos_core.monitoring.metrics import MetricsCollector
    from aneos_core.monitoring.alerts import AlertManager
    HAS_MONITORING = True
except ImportError:
    MetricsCollector = None
    AlertManager = None
    HAS_MONITORING = False

try:
    from aneos_core.config.settings import get_config
    HAS_CONFIG = True
except ImportError:
    get_config = None
    HAS_CONFIG = False
    create_analysis_pipeline = None
    RealTimePredictor = None
    PredictionConfig = None
    TrainingPipeline = None
    TrainingConfig = None
    MetricsCollector = None
    AlertManager = None
    get_config = lambda: {'api': {}, 'analysis': {}, 'ml': {}, 'monitoring': {}}

from .models import APIResponse, ErrorResponse
from .auth import AuthManager, get_current_user
from .middleware import setup_middleware
from .endpoints import analysis, prediction, monitoring, admin, streaming
from . import dashboard

logger = logging.getLogger(__name__)

class ANEOSApp:
    """Main aNEOS application class managing all services."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize aNEOS application."""
        self.config = get_config()
        
        # Core services
        self.analysis_pipeline = None
        self.ml_predictor = None
        self.training_pipeline = None
        self.metrics_collector = None
        self.alert_manager = None
        self.auth_manager: Optional[AuthManager] = None
        
        # Service status
        self.services_initialized = False
        self.startup_time: Optional[datetime] = None
        
        logger.info("aNEOS application initialized")
    
    async def initialize_services(self) -> None:
        """Initialize all aNEOS services."""
        if self.services_initialized:
            return
        
        logger.info("Initializing aNEOS services...")
        
        try:
            # Initialize analysis pipeline (if available)
            if create_analysis_pipeline is not None:
                self.analysis_pipeline = create_analysis_pipeline()
                logger.info("✓ Analysis pipeline initialized")
            else:
                logger.warning("⚠️  Analysis pipeline not available - core modules not found")
            
            # Initialize ML predictor (if components available)
            if _load_ml_components() and self.analysis_pipeline:
                prediction_config = PredictionConfig()
                self.ml_predictor = RealTimePredictor(self.analysis_pipeline, prediction_config)
                logger.info("✓ ML predictor initialized")
            else:
                logger.warning("⚠️  ML predictor not available - dependencies missing")
            
            # Initialize training pipeline (if available)
            if HAS_ML and self.analysis_pipeline:
                training_config = TrainingConfig()
                self.training_pipeline = TrainingPipeline(self.analysis_pipeline, training_config)
                logger.info("✓ Training pipeline initialized")
            else:
                logger.warning("⚠️  Training pipeline not available - dependencies missing")
            
            # Initialize monitoring (if available)
            if MetricsCollector is not None:
                self.metrics_collector = MetricsCollector(collection_interval=60)
                if self.analysis_pipeline:
                    self.metrics_collector.analysis_pipeline = self.analysis_pipeline
                if self.ml_predictor:
                    self.metrics_collector.ml_predictor = self.ml_predictor
                self.metrics_collector.start_collection()
                logger.info("✓ Metrics collector started")
            else:
                logger.warning("⚠️  Metrics collector not available - core modules missing")
            
            # Initialize alert manager (if available)
            if AlertManager is not None:
                self.alert_manager = AlertManager()
                if self.metrics_collector:
                    self.metrics_collector.alert_manager = self.alert_manager
                logger.info("✓ Alert manager initialized")
            else:
                logger.warning("⚠️  Alert manager not available - core modules missing")
            
            # Initialize authentication (basic version always available)
            from .auth import AuthManager as APIAuthManager
            self.auth_manager = APIAuthManager(self.config)
            logger.info("✓ Authentication manager initialized")
            
            self.services_initialized = True
            self.startup_time = datetime.now()
            
            logger.info("🚀 Basic API services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # Don't raise - allow basic API to work even with limited functionality
            self.services_initialized = True
            self.startup_time = datetime.now()
            logger.warning("⚠️  API started with limited functionality")
    
    async def shutdown_services(self) -> None:
        """Shutdown all aNEOS services."""
        logger.info("Shutting down aNEOS services...")
        
        if self.metrics_collector:
            self.metrics_collector.stop_collection()
            logger.info("✓ Metrics collector stopped")
        
        # Additional cleanup can be added here
        
        logger.info("✓ aNEOS services shutdown complete")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get application health status."""
        return {
            'status': 'healthy' if self.services_initialized else 'initializing',
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'services': {
                'analysis_pipeline': self.analysis_pipeline is not None,
                'ml_predictor': self.ml_predictor is not None,
                'training_pipeline': self.training_pipeline is not None,
                'metrics_collector': self.metrics_collector is not None and self.metrics_collector.running,
                'alert_manager': self.alert_manager is not None,
                'auth_manager': self.auth_manager is not None
            },
            'version': '2.0.0'
        }

# Global application instance
_aneos_app: Optional[ANEOSApp] = None

def get_aneos_app() -> ANEOSApp:
    """Get global aNEOS application instance."""
    global _aneos_app
    if _aneos_app is None:
        _aneos_app = ANEOSApp()
    return _aneos_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    aneos_app = get_aneos_app()
    await aneos_app.initialize_services()
    
    yield
    
    # Shutdown
    await aneos_app.shutdown_services()

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for API services")
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="aNEOS API",
        description="Advanced Near Earth Object detection System - RESTful API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers (only if they are actual FastAPI routers)
    from fastapi import APIRouter
    if isinstance(analysis.router, APIRouter):
        app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
    if isinstance(prediction.router, APIRouter):
        app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["ML Prediction"])
    if isinstance(monitoring.router, APIRouter):
        app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    if isinstance(admin.router, APIRouter):
        app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])
    if isinstance(streaming.router, APIRouter):
        app.include_router(streaming.router, prefix="/api/v1/stream", tags=["Streaming"])
    if isinstance(dashboard.router, APIRouter):
        app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
    
    # Health check endpoint
    @app.get("/health", response_model=Dict[str, Any])
    async def health_check():
        """Health check endpoint."""
        aneos_app = get_aneos_app()
        return aneos_app.get_health_status()
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "aNEOS API",
            "version": "2.0.0",
            "description": "Advanced Near Earth Object detection System",
            "documentation": "/docs",
            "dashboard": "/dashboard",
            "health": "/health",
            "endpoints": {
                "analysis": "/api/v1/analysis",
                "prediction": "/api/v1/prediction",
                "monitoring": "/api/v1/monitoring",
                "admin": "/api/v1/admin",
                "streaming": "/api/v1/stream"
            }
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="aNEOS API",
            version="2.0.0",
            description="Advanced Near Earth Object detection System - RESTful API for artificial NEO detection",
            routes=app.routes,
        )
        
        # Add custom schema elements
        openapi_schema["info"]["x-logo"] = {
            "url": "https://example.com/aneos-logo.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                status_code=exc.status_code,
                timestamp=datetime.now()
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                status_code=500,
                timestamp=datetime.now()
            ).dict()
        )
    
    logger.info("FastAPI application created successfully")
    return app

# Development server
if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        
        app = create_app()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=True
        )
    else:
        print("FastAPI not available. Please install: pip install fastapi uvicorn")