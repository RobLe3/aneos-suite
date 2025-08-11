"""
Monitoring endpoints for aNEOS API.

Provides system health monitoring, metrics collection, alerting,
and performance tracking capabilities.
"""

from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

try:
    from fastapi import APIRouter, HTTPException, Depends, Query
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, monitoring endpoints disabled")

try:
    from aneos_core.monitoring.metrics import MetricsCollector
    from aneos_core.monitoring.alerts import AlertManager, Alert
except ImportError:
    MetricsCollector = None
    AlertManager = None
    Alert = None
from ..models import (
    MetricsResponse, SystemMetricsResponse, AnalysisMetricsResponse,
    MLMetricsResponse, AlertResponse
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

async def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the metrics collector from the application."""
    from ..app import get_aneos_app  # Import here to avoid circular imports
    aneos_app = get_aneos_app()
    return aneos_app.metrics_collector

async def get_alert_manager() -> Optional[AlertManager]:
    """Get the alert manager from the application."""
    from ..app import get_aneos_app  # Import here to avoid circular imports
    aneos_app = get_aneos_app()
    return aneos_app.alert_manager

# Health endpoint removed to prevent conflicts with main health endpoint in app.py
# All health check functionality is now centralized in the main app health endpoint

@router.get("/metrics", response_model=MetricsResponse)
async def get_current_metrics(
    include_history: bool = Query(False, description="Include historical metrics"),
    metrics_collector: Optional[MetricsCollector] = Depends(get_metrics_collector),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get current system metrics including system, analysis, and ML metrics."""
    try:
        if not metrics_collector:
            # Return empty/mock response instead of 503 error
            return MetricsResponse(
                system_metrics=None,
                analysis_metrics=None,
                ml_metrics=None,
                recent_alerts=[],
                performance_summary={
                    'system_load': 0,
                    'memory_usage': 0,
                    'analysis_success_rate': 0,
                    'ml_prediction_rate': 0,
                    'active_alerts': 0,
                    'status': 'monitoring_unavailable'
                }
            )
        # Get latest metrics
        system_metrics = metrics_collector.get_system_metrics()
        analysis_metrics = metrics_collector.get_analysis_metrics()
        ml_metrics = metrics_collector.get_ml_metrics()
        
        # Convert to response models
        system_response = None
        if system_metrics:
            system_response = SystemMetricsResponse(
                timestamp=system_metrics.timestamp,
                cpu_percent=system_metrics.cpu_percent,
                memory_percent=system_metrics.memory_percent,
                memory_used_mb=system_metrics.memory_used_mb,
                disk_usage_percent=system_metrics.disk_usage_percent,
                network_bytes_sent=system_metrics.network_bytes_sent,
                network_bytes_recv=system_metrics.network_bytes_recv,
                process_count=system_metrics.process_count
            )
        
        analysis_response = None
        if analysis_metrics:
            analysis_response = AnalysisMetricsResponse(
                timestamp=analysis_metrics.timestamp,
                total_analyses=analysis_metrics.total_analyses,
                successful_analyses=analysis_metrics.successful_analyses,
                failed_analyses=analysis_metrics.failed_analyses,
                average_processing_time=analysis_metrics.average_processing_time,
                cache_hit_rate=analysis_metrics.cache_hit_rate,
                anomaly_detection_rate=analysis_metrics.anomaly_detection_rate
            )
        
        ml_response = None
        if ml_metrics:
            ml_response = MLMetricsResponse(
                timestamp=ml_metrics.timestamp,
                model_predictions=ml_metrics.model_predictions,
                prediction_latency=ml_metrics.prediction_latency,
                feature_quality=ml_metrics.feature_quality,
                ensemble_agreement=ml_metrics.ensemble_agreement,
                alert_count=ml_metrics.alert_count
            )
        
        # Get recent alerts
        alert_manager = await get_alert_manager()
        recent_alerts = alert_manager.get_recent_alerts(limit=10) if alert_manager else []
        alert_responses = [
            AlertResponse(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type,
                alert_level=alert.alert_level,
                title=alert.title,
                message=alert.message,
                timestamp=alert.timestamp,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
                data=alert.data
            )
            for alert in recent_alerts
        ]
        
        # Build performance summary
        performance_summary = {
            'system_load': system_metrics.cpu_percent if system_metrics else 0,
            'memory_usage': system_metrics.memory_percent if system_metrics else 0,
            'analysis_success_rate': (analysis_metrics.successful_analyses / max(analysis_metrics.total_analyses, 1) * 100) if analysis_metrics else 0,
            'ml_prediction_rate': ml_metrics.model_predictions if ml_metrics else 0,
            'active_alerts': len([a for a in recent_alerts if not a.resolved])
        }
        
        return MetricsResponse(
            system_metrics=system_response,
            analysis_metrics=analysis_response,
            ml_metrics=ml_response,
            recent_alerts=alert_responses,
            performance_summary=performance_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/metrics/history", response_model=Dict[str, Any])
async def get_metrics_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    metrics_collector: Optional[MetricsCollector] = Depends(get_metrics_collector),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get historical metrics data for trending analysis."""
    try:
        if not metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get historical data
        history = metrics_collector.get_metrics_history(start_time, end_time)
        
        return {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'data_points': len(history),
            'metrics_history': history,
            'trends': {
                'cpu_trend': _calculate_trend([m.get('cpu_percent', 0) for m in history]),
                'memory_trend': _calculate_trend([m.get('memory_percent', 0) for m in history]),
                'analysis_rate_trend': _calculate_trend([m.get('analysis_rate', 0) for m in history])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics history: {str(e)}")

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    alert_manager: Optional[AlertManager] = Depends(get_alert_manager),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get system alerts with optional filtering."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert manager not available")
        alerts = alert_manager.get_alerts(
            level=level,
            resolved=resolved,
            limit=limit
        )
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type,
                alert_level=alert.alert_level,
                title=alert.title,
                message=alert.message,
                timestamp=alert.timestamp,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
                data=alert.data
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(
    alert_id: str,
    alert_manager: Optional[AlertManager] = Depends(get_alert_manager),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Acknowledge an alert."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert manager not available")
        success = alert_manager.acknowledge_alert(alert_id, current_user.get('username', 'unknown') if current_user else 'api')
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return {
            'status': 'success',
            'alert_id': alert_id,
            'acknowledged_at': datetime.now().isoformat(),
            'acknowledged_by': current_user.get('username', 'api') if current_user else 'api'
        }
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: str,
    alert_manager: Optional[AlertManager] = Depends(get_alert_manager),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Resolve an alert."""
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert manager not available")
        success = alert_manager.resolve_alert(alert_id, current_user.get('username', 'unknown') if current_user else 'api')
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return {
            'status': 'success',
            'alert_id': alert_id,
            'resolved_at': datetime.now().isoformat(),
            'resolved_by': current_user.get('username', 'api') if current_user else 'api'
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    metrics_collector: Optional[MetricsCollector] = Depends(get_metrics_collector),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get detailed performance metrics and benchmarks."""
    try:
        if not metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        performance_data = metrics_collector.get_performance_summary()
        
        # Calculate additional performance indicators
        system_metrics = metrics_collector.get_system_metrics()
        analysis_metrics = metrics_collector.get_analysis_metrics()
        ml_metrics = metrics_collector.get_ml_metrics()
        
        return {
            'system_performance': {
                'cpu_utilization': system_metrics.cpu_percent if system_metrics else 0,
                'memory_utilization': system_metrics.memory_percent if system_metrics else 0,
                'disk_utilization': system_metrics.disk_usage_percent if system_metrics else 0,
                'load_average': performance_data.get('load_average', 0),
                'performance_score': _calculate_performance_score(system_metrics)
            },
            'analysis_performance': {
                'throughput_per_hour': analysis_metrics.total_analyses if analysis_metrics else 0,
                'average_latency_ms': analysis_metrics.average_processing_time * 1000 if analysis_metrics else 0,
                'success_rate': (analysis_metrics.successful_analyses / max(analysis_metrics.total_analyses, 1) * 100) if analysis_metrics else 0,
                'cache_efficiency': analysis_metrics.cache_hit_rate if analysis_metrics else 0
            },
            'ml_performance': {
                'predictions_per_hour': ml_metrics.model_predictions if ml_metrics else 0,
                'prediction_latency_ms': ml_metrics.prediction_latency * 1000 if ml_metrics else 0,
                'feature_quality_score': ml_metrics.feature_quality if ml_metrics else 0,
                'ensemble_agreement': ml_metrics.ensemble_agreement if ml_metrics else 0
            },
            'bottlenecks': _identify_bottlenecks(system_metrics, analysis_metrics, ml_metrics),
            'recommendations': _generate_performance_recommendations(system_metrics, analysis_metrics, ml_metrics)
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard_data(
    metrics_collector: Optional[MetricsCollector] = Depends(get_metrics_collector),
    alert_manager: Optional[AlertManager] = Depends(get_alert_manager),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get comprehensive dashboard data for monitoring interface."""
    try:
        if not metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        # Get current metrics
        system_metrics = metrics_collector.get_system_metrics()
        analysis_metrics = metrics_collector.get_analysis_metrics()
        ml_metrics = metrics_collector.get_ml_metrics()
        
        # Get recent alerts
        recent_alerts = alert_manager.get_recent_alerts(limit=5) if alert_manager else []
        active_alerts = [a for a in recent_alerts if not a.resolved]
        
        # Build dashboard data
        return {
            'status_overview': {
                'overall_health': 'healthy' if len(active_alerts) == 0 else 'warning' if len(active_alerts) < 3 else 'critical',
                'services_online': 6,  # Mock - would be dynamic
                'active_alerts': len(active_alerts),
                'uptime_hours': 24.5  # Mock - would be calculated
            },
            'key_metrics': {
                'cpu_usage': system_metrics.cpu_percent if system_metrics else 0,
                'memory_usage': system_metrics.memory_percent if system_metrics else 0,
                'analyses_today': analysis_metrics.total_analyses if analysis_metrics else 0,
                'predictions_today': ml_metrics.model_predictions if ml_metrics else 0,
                'anomalies_detected': int((analysis_metrics.anomaly_detection_rate / 100) * analysis_metrics.total_analyses) if analysis_metrics else 0
            },
            'recent_activity': {
                'latest_analysis': datetime.now() - timedelta(minutes=5),
                'latest_prediction': datetime.now() - timedelta(minutes=2),
                'latest_alert': recent_alerts[0].timestamp if recent_alerts else None
            },
            'trends': {
                'analysis_trend': 'increasing',  # Mock - would be calculated
                'performance_trend': 'stable',
                'alert_trend': 'decreasing'
            },
            'alerts_summary': [
                {
                    'level': alert.alert_level,
                    'title': alert.title,
                    'timestamp': alert.timestamp,
                    'resolved': alert.resolved
                }
                for alert in recent_alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# Helper functions
def _calculate_trend(values: List[float]) -> str:
    """Calculate trend direction from a list of values."""
    if len(values) < 2:
        return 'stable'
    
    recent_avg = sum(values[-5:]) / min(len(values), 5)
    older_avg = sum(values[:-5]) / max(len(values) - 5, 1)
    
    if recent_avg > older_avg * 1.1:
        return 'increasing'
    elif recent_avg < older_avg * 0.9:
        return 'decreasing'
    else:
        return 'stable'

def _calculate_performance_score(system_metrics) -> float:
    """Calculate overall performance score."""
    if not system_metrics:
        return 0.0
    
    # Simple scoring based on resource utilization
    cpu_score = max(0, 100 - system_metrics.cpu_percent) / 100
    memory_score = max(0, 100 - system_metrics.memory_percent) / 100
    
    return (cpu_score + memory_score) / 2

def _identify_bottlenecks(system_metrics, analysis_metrics, ml_metrics) -> List[str]:
    """Identify system bottlenecks."""
    bottlenecks = []
    
    if system_metrics:
        if system_metrics.cpu_percent > 80:
            bottlenecks.append('High CPU utilization')
        if system_metrics.memory_percent > 85:
            bottlenecks.append('High memory usage')
    
    if analysis_metrics:
        if analysis_metrics.average_processing_time > 10:
            bottlenecks.append('Slow analysis processing')
        if analysis_metrics.cache_hit_rate < 0.5:
            bottlenecks.append('Low cache efficiency')
    
    if ml_metrics:
        if ml_metrics.prediction_latency > 2:
            bottlenecks.append('Slow ML predictions')
        if ml_metrics.feature_quality < 0.7:
            bottlenecks.append('Low feature quality')
    
    return bottlenecks

def _generate_performance_recommendations(system_metrics, analysis_metrics, ml_metrics) -> List[str]:
    """Generate performance improvement recommendations."""
    recommendations = []
    
    if system_metrics:
        if system_metrics.cpu_percent > 70:
            recommendations.append('Consider scaling CPU resources or optimizing algorithms')
        if system_metrics.memory_percent > 80:
            recommendations.append('Increase memory allocation or optimize memory usage')
    
    if analysis_metrics:
        if analysis_metrics.cache_hit_rate < 0.6:
            recommendations.append('Optimize caching strategy for better performance')
        if analysis_metrics.average_processing_time > 5:
            recommendations.append('Profile and optimize analysis algorithms')
    
    if ml_metrics:
        if ml_metrics.prediction_latency > 1:
            recommendations.append('Optimize ML model inference or use model quantization')
        if ml_metrics.ensemble_agreement < 0.8:
            recommendations.append('Retrain models or adjust ensemble weights')
    
    return recommendations