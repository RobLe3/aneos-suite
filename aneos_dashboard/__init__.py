"""
aNEOS Real-Time Validation Dashboard

Comprehensive real-time monitoring and visualization system for the aNEOS 
multi-stage validation pipeline. Provides operational monitoring, performance 
tracking, and interactive visualization of validation results.

Features:
- Real-time validation pipeline monitoring
- Interactive scatter plots and statistical visualizations
- Alert system for artificial object detections
- Multi-stage validation performance tracking
- Expert review queue management
- Integration with all Phase 1-3 validation modules

Architecture:
- FastAPI backend with WebSocket support for real-time updates
- HTML/JavaScript frontend with Chart.js and D3.js visualizations
- Real-time metrics collection and streaming
- Responsive design for multiple screen sizes
- Authentication integration for secure access

Author: NU SWARM - Real-Time Validation Dashboard Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "NU SWARM - Real-Time Validation Dashboard Team"

from .api.dashboard_endpoints import router as dashboard_router
from .websockets.validation_websocket import ValidationWebSocketManager
from .monitoring.validation_metrics import ValidationMetricsCollector

__all__ = [
    'dashboard_router',
    'ValidationWebSocketManager', 
    'ValidationMetricsCollector'
]