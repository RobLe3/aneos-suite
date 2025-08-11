"""
aNEOS Real-Time Validation Dashboard Application

Main application entry point for the comprehensive real-time validation
dashboard system. Integrates with the aNEOS API and validation pipeline
to provide operational monitoring and visualization.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, dashboard application disabled")

from .monitoring.validation_metrics import ValidationMetricsCollector
from .websockets.validation_websocket import ValidationWebSocketManager
from .api.dashboard_endpoints import router as dashboard_router, initialize_dashboard
from .api.validation_integration import ValidationPipelineIntegration, initialize_integration

logger = logging.getLogger(__name__)

class DashboardApp:
    """
    Main dashboard application class.
    
    Orchestrates all dashboard components including metrics collection,
    WebSocket management, API endpoints, and validation integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard application.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.app: Optional[FastAPI] = None
        self.metrics_collector: Optional[ValidationMetricsCollector] = None
        self.websocket_manager: Optional[ValidationWebSocketManager] = None
        self.validation_integration: Optional[ValidationPipelineIntegration] = None
        
        # Get static and template paths
        self.dashboard_path = Path(__file__).parent
        self.static_path = self.dashboard_path / "static"
        self.templates_path = self.dashboard_path / "templates"
        
        if not HAS_FASTAPI:
            logger.error("FastAPI not available - dashboard cannot be initialized")
            return
        
        # Initialize components
        self._initialize_components()
        self._create_app()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for dashboard application."""
        return {
            'title': 'aNEOS Real-Time Validation Dashboard',
            'description': 'Comprehensive monitoring and visualization for the aNEOS validation pipeline',
            'version': '1.0.0',
            'max_history_hours': 24,
            'max_sessions_memory': 1000,
            'websocket_update_interval': 1.0,
            'allow_origins': ['*'],
            'debug': False,
            'validation_config': {
                'enable_dashboard_integration': True,
                'real_time_metrics': True,
                'alert_system': True
            }
        }
    
    def _initialize_components(self):
        """Initialize all dashboard components."""
        try:
            # Initialize metrics collector
            self.metrics_collector = ValidationMetricsCollector(
                max_history_hours=self.config['max_history_hours'],
                max_sessions_memory=self.config['max_sessions_memory']
            )
            
            # Initialize WebSocket manager
            self.websocket_manager = ValidationWebSocketManager(self.metrics_collector)
            self.websocket_manager.update_interval = self.config['websocket_update_interval']
            
            # Initialize validation integration
            self.validation_integration = initialize_integration(
                self.metrics_collector,
                self.websocket_manager,
                self.config.get('validation_config')
            )
            
            # Initialize dashboard endpoints
            initialize_dashboard(self.metrics_collector, self.websocket_manager)
            
            logger.info("Dashboard components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard components: {e}")
            raise
    
    def _create_app(self):
        """Create FastAPI application with all routes and middleware."""
        try:
            self.app = FastAPI(
                title=self.config['title'],
                description=self.config['description'],
                version=self.config['version'],
                docs_url='/dashboard/docs',
                redoc_url='/dashboard/redoc',
                openapi_url='/dashboard/openapi.json'
            )
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config['allow_origins'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
            
            # Mount static files
            if self.static_path.exists():
                self.app.mount("/dashboard/static", StaticFiles(directory=str(self.static_path)), name="static")
            
            # Include dashboard API routes
            self.app.include_router(dashboard_router, prefix="/dashboard")
            
            # Add main dashboard route
            @self.app.get("/dashboard/", response_class=HTMLResponse)
            @self.app.get("/dashboard", response_class=HTMLResponse)
            async def dashboard_home(request: Request):
                """Serve main dashboard page."""
                try:
                    dashboard_html_path = self.templates_path / "validation_dashboard.html"
                    
                    if dashboard_html_path.exists():
                        with open(dashboard_html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        return HTMLResponse(content=html_content)
                    else:
                        return HTMLResponse(
                            content=self._create_fallback_dashboard(),
                            status_code=200
                        )
                        
                except Exception as e:
                    logger.error(f"Error serving dashboard: {e}")
                    return HTMLResponse(
                        content=f"<h1>Dashboard Error</h1><p>{str(e)}</p>",
                        status_code=500
                    )
            
            # Add redirect from root to dashboard
            @self.app.get("/")
            async def root():
                """Redirect root to dashboard."""
                return RedirectResponse(url="/dashboard/")
            
            # Add health check endpoint
            @self.app.get("/dashboard/health")
            async def dashboard_health():
                """Dashboard health check endpoint."""
                try:
                    if self.validation_integration:
                        health_status = await self.validation_integration.perform_system_health_check()
                    else:
                        health_status = {
                            'status': 'degraded',
                            'message': 'Validation integration not available'
                        }
                    
                    return health_status
                    
                except Exception as e:
                    return {
                        'status': 'unhealthy',
                        'error': str(e),
                        'timestamp': None
                    }
            
            # Startup event
            @self.app.on_event("startup")
            async def startup_event():
                """Application startup tasks."""
                logger.info("Starting aNEOS Validation Dashboard...")
                
                # Start monitoring loop
                if self.validation_integration:
                    self.validation_integration.start_monitoring_loop()
                
                logger.info("Dashboard startup complete")
            
            # Shutdown event
            @self.app.on_event("shutdown")
            async def shutdown_event():
                """Application shutdown tasks."""
                logger.info("Shutting down aNEOS Validation Dashboard...")
                
                # Close WebSocket connections
                if self.websocket_manager:
                    for websocket in list(self.websocket_manager.active_connections):
                        try:
                            await websocket.close()
                        except:
                            pass
                
                logger.info("Dashboard shutdown complete")
            
            logger.info("FastAPI dashboard application created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create FastAPI application: {e}")
            raise
    
    def _create_fallback_dashboard(self) -> str:
        """Create fallback dashboard HTML when template is not available."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>aNEOS Dashboard - Fallback</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 2rem;
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 600px;
            text-align: center;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        p {
            font-size: 1.1rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        .api-links {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2rem;
        }
        .api-link {
            background: rgba(255,255,255,0.1);
            color: white;
            text-decoration: none;
            padding: 0.75rem 1.5rem;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
        }
        .api-link:hover {
            background: rgba(255,255,255,0.2);
        }
        .status {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>aNEOS Validation Dashboard</h1>
        <p>The dashboard template is not available, but the API endpoints are active.</p>
        
        <div class="api-links">
            <a href="/dashboard/api/dashboard/data" class="api-link">Dashboard Data API</a>
            <a href="/dashboard/api/system/health" class="api-link">System Health API</a>
            <a href="/dashboard/docs" class="api-link">API Documentation</a>
        </div>
        
        <div class="status">
            <strong>System Status:</strong> Dashboard API Active<br>
            <strong>WebSocket Endpoint:</strong> ws://[host]/dashboard/ws/validation<br>
            <strong>Health Check:</strong> <a href="/dashboard/health" style="color: #3498db;">Available</a>
        </div>
    </div>
    
    <script>
        // Auto-refresh status every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
        """
    
    async def validate_with_dashboard(self, neo_data: Any, analysis_result: Any, session_id: Optional[str] = None):
        """
        Public method to run validation with dashboard integration.
        
        Args:
            neo_data: NEO data object
            analysis_result: Analysis result from aNEOS pipeline
            session_id: Optional session identifier
            
        Returns:
            Validation result with dashboard integration
        """
        if not self.validation_integration:
            raise RuntimeError("Dashboard validation integration not available")
        
        return await self.validation_integration.validate_with_dashboard_integration(
            neo_data, analysis_result, session_id
        )
    
    def get_app(self) -> Optional[FastAPI]:
        """Get the FastAPI application instance."""
        return self.app
    
    def get_metrics_collector(self) -> Optional[ValidationMetricsCollector]:
        """Get the metrics collector instance."""
        return self.metrics_collector
    
    def get_websocket_manager(self) -> Optional[ValidationWebSocketManager]:
        """Get the WebSocket manager instance."""
        return self.websocket_manager
    
    def get_validation_integration(self) -> Optional[ValidationPipelineIntegration]:
        """Get the validation integration instance."""
        return self.validation_integration
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get current dashboard status and statistics."""
        try:
            status = {
                'dashboard_initialized': self.app is not None,
                'components': {
                    'metrics_collector': self.metrics_collector is not None,
                    'websocket_manager': self.websocket_manager is not None,
                    'validation_integration': self.validation_integration is not None
                },
                'configuration': {
                    'max_history_hours': self.config['max_history_hours'],
                    'max_sessions_memory': self.config['max_sessions_memory'],
                    'websocket_update_interval': self.config['websocket_update_interval']
                }
            }
            
            # Add WebSocket statistics if available
            if self.websocket_manager:
                status['websocket_stats'] = self.websocket_manager.get_connection_stats()
            
            # Add validation status if available
            if self.validation_integration:
                status['active_validations'] = self.validation_integration.get_active_validations()
                status['module_performance'] = self.validation_integration.get_module_performance_metrics()
            
            return status
            
        except Exception as e:
            return {'error': str(e)}

# Global dashboard application instance
_dashboard_app: Optional[DashboardApp] = None

def create_dashboard_app(config: Optional[Dict[str, Any]] = None) -> DashboardApp:
    """
    Create and initialize the dashboard application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized DashboardApp instance
    """
    global _dashboard_app
    
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not available - cannot create dashboard application")
    
    _dashboard_app = DashboardApp(config)
    return _dashboard_app

def get_dashboard_app() -> Optional[DashboardApp]:
    """Get the global dashboard application instance."""
    return _dashboard_app

# For direct FastAPI application access
def get_fastapi_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Get FastAPI application for ASGI deployment.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        FastAPI application instance
    """
    if not _dashboard_app:
        create_dashboard_app(config)
    
    if not _dashboard_app or not _dashboard_app.get_app():
        raise RuntimeError("Failed to create dashboard application")
    
    return _dashboard_app.get_app()

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    # Create dashboard app
    dashboard = create_dashboard_app({
        'debug': True,
        'max_history_hours': 48,
        'websocket_update_interval': 0.5
    })
    
    if dashboard and dashboard.get_app():
        print("Starting aNEOS Validation Dashboard...")
        print("Dashboard available at: http://localhost:8000/dashboard/")
        print("API Documentation: http://localhost:8000/dashboard/docs")
        print("WebSocket endpoint: ws://localhost:8000/dashboard/ws/validation")
        
        # Run the application
        uvicorn.run(
            dashboard.get_app(),
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    else:
        print("Failed to create dashboard application")