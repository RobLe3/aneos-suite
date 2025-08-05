"""
aNEOS API Module - RESTful web services for NEO analysis platform.

This module provides comprehensive API services including:
- NEO analysis endpoints
- ML prediction services
- Monitoring and metrics APIs
- Real-time streaming interfaces
"""

from .app import create_app, ANEOSApp
from .endpoints import analysis, prediction, monitoring, admin
from .models import APIResponse, ErrorResponse, PaginatedResponse
from .auth import AuthManager, APIKeyAuth
from .middleware import RateLimiter, RequestLogger, ErrorHandler

__version__ = "2.0.0"
__author__ = "aNEOS Project"

__all__ = [
    'create_app', 'ANEOSApp',
    'APIResponse', 'ErrorResponse', 'PaginatedResponse',
    'AuthManager', 'APIKeyAuth',
    'RateLimiter', 'RequestLogger', 'ErrorHandler'
]