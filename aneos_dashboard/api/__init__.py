"""
aNEOS Dashboard API Module

FastAPI endpoints for real-time validation dashboard.
"""

from .dashboard_endpoints import router, initialize_dashboard

__all__ = ['router', 'initialize_dashboard']