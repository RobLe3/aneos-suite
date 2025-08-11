"""
Real-Time Validation Dashboard API Endpoints

FastAPI endpoints for the comprehensive real-time validation dashboard,
providing metrics, alerts, visualization data, and WebSocket connections.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, dashboard endpoints disabled")

from ..monitoring.validation_metrics import ValidationMetricsCollector
from ..websockets.validation_websocket import ValidationWebSocketManager

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    router = APIRouter()
else:
    # Fallback router for when FastAPI is not available
    class MockRouter:
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def websocket(self, *args, **kwargs): return lambda f: f
    router = MockRouter()

# Global instances (will be initialized by the main application)
metrics_collector: Optional[ValidationMetricsCollector] = None
websocket_manager: Optional[ValidationWebSocketManager] = None

def initialize_dashboard(collector: ValidationMetricsCollector, ws_manager: ValidationWebSocketManager):
    """Initialize dashboard with metrics collector and WebSocket manager."""
    global metrics_collector, websocket_manager
    metrics_collector = collector
    websocket_manager = ws_manager

async def get_metrics_collector():
    """Dependency to get metrics collector."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return metrics_collector

async def get_websocket_manager():
    """Dependency to get WebSocket manager."""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return websocket_manager

@router.websocket("/ws/validation")
async def validation_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time validation dashboard updates.
    
    Provides live streaming of validation metrics, alerts, and system status.
    """
    if not HAS_FASTAPI:
        return
    
    client_id = f"client_{datetime.now().timestamp()}"
    ws_manager = await get_websocket_manager()
    
    connection_success = await ws_manager.connect(websocket, client_id)
    if not connection_success:
        return
    
    try:
        while True:
            # Wait for incoming client messages
            data = await websocket.receive_text()
            await ws_manager.handle_client_message(websocket, data)
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await ws_manager.disconnect(websocket, client_id)

@router.get("/api/dashboard/data", response_model=Dict[str, Any])
async def get_dashboard_data(
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get comprehensive real-time dashboard data.
    
    Returns:
        Complete dashboard data including system overview, validation pipeline
        performance, detection statistics, alerts, and real-time metrics.
    """
    try:
        dashboard_data = collector.get_dashboard_data()
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.get("/api/validation/history", response_model=List[Dict[str, Any]])
async def get_validation_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get validation session history for specified time period.
    
    Args:
        hours: Number of hours of history to retrieve (1-168 hours)
        
    Returns:
        List of validation session records with detailed metrics
    """
    try:
        history = collector.get_validation_history(hours)
        return JSONResponse(content=history)
        
    except Exception as e:
        logger.error(f"Failed to get validation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation history: {str(e)}")

@router.get("/api/validation/stages/performance", response_model=Dict[str, Any])
async def get_stage_performance_metrics(
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get detailed performance metrics for each validation stage.
    
    Returns:
        Comprehensive performance data for validation pipeline stages
        including processing times, success rates, and bottleneck analysis.
    """
    try:
        dashboard_data = collector.get_dashboard_data()
        stage_performance = dashboard_data.get('validation_pipeline', {})
        
        return JSONResponse(content={
            'stage_performance': stage_performance.get('stage_performance', {}),
            'bottlenecks': stage_performance.get('bottlenecks', []),
            'module_availability': stage_performance.get('module_availability', {}),
            'processing_trends': stage_performance.get('processing_trends', {}),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get stage performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stage performance: {str(e)}")

@router.get("/api/detection/statistics", response_model=Dict[str, Any])
async def get_detection_statistics(
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get comprehensive detection and classification statistics.
    
    Returns:
        Detection statistics including acceptance/rejection rates,
        confidence distributions, and artificial object detection metrics.
    """
    try:
        dashboard_data = collector.get_dashboard_data()
        detection_stats = dashboard_data.get('detection_statistics', {})
        
        return JSONResponse(content=detection_stats)
        
    except Exception as e:
        logger.error(f"Failed to get detection statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection statistics: {str(e)}")

@router.get("/api/alerts/artificial-objects", response_model=Dict[str, Any])
async def get_artificial_object_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get artificial object detection alerts with filtering options.
    
    Args:
        level: Filter by alert level ('info', 'warning', 'critical', 'urgent')
        resolved: Filter by resolved status (True/False)
        limit: Maximum number of alerts to return
        
    Returns:
        Filtered list of artificial object alerts with details
    """
    try:
        dashboard_data = collector.get_dashboard_data()
        alerts_data = dashboard_data.get('alerts_and_notifications', {})
        all_alerts = alerts_data.get('recent_alerts', [])
        
        # Apply filters
        filtered_alerts = all_alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.get('alert_level') == level]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.get('resolved') == resolved]
        
        # Apply limit
        filtered_alerts = filtered_alerts[:limit]
        
        return JSONResponse(content={
            'alerts': filtered_alerts,
            'total_count': len(all_alerts),
            'filtered_count': len(filtered_alerts),
            'alert_rate_per_hour': alerts_data.get('alert_rate_per_hour', 0),
            'critical_alerts': alerts_data.get('critical_alerts', 0),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get artificial object alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/api/alerts/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(
    alert_id: str,
    user: str = Query("dashboard_user", description="User acknowledging the alert"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Acknowledge an artificial object alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        user: Username of the person acknowledging the alert
        
    Returns:
        Acknowledgment status and details
    """
    try:
        success = collector.acknowledge_alert(alert_id, user)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return JSONResponse(content={
            'status': 'success',
            'alert_id': alert_id,
            'acknowledged_at': datetime.now().isoformat(),
            'acknowledged_by': user,
            'message': f'Alert {alert_id} acknowledged successfully'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/api/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: str,
    user: str = Query("dashboard_user", description="User resolving the alert"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Resolve an artificial object alert.
    
    Args:
        alert_id: ID of the alert to resolve
        user: Username of the person resolving the alert
        
    Returns:
        Resolution status and details
    """
    try:
        success = collector.resolve_alert(alert_id, user)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        return JSONResponse(content={
            'status': 'success',
            'alert_id': alert_id,
            'resolved_at': datetime.now().isoformat(),
            'resolved_by': user,
            'message': f'Alert {alert_id} resolved successfully'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/api/system/health", response_model=Dict[str, Any])
async def get_system_health(
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get current system health and performance metrics.
    
    Returns:
        System health metrics including resource utilization,
        processing performance, and operational status.
    """
    try:
        dashboard_data = collector.get_dashboard_data()
        system_health = dashboard_data.get('system_health', {})
        system_overview = dashboard_data.get('system_overview', {})
        
        return JSONResponse(content={
            'system_health': system_health,
            'overview': system_overview,
            'real_time_metrics': dashboard_data.get('real_time_metrics', {}),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/api/visualization/scatter-data", response_model=Dict[str, Any])
async def get_scatter_plot_data(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to include"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get data formatted for validation result scatter plots.
    
    Args:
        hours: Hours of validation data to include
        
    Returns:
        Scatter plot data with confidence vs score, colored by stage results
        and recommendation outcomes.
    """
    try:
        history = collector.get_validation_history(hours)
        
        # Format data for scatter plot visualization
        scatter_data = {
            'confidence_vs_score': [],
            'stage_performance': [],
            'recommendation_distribution': {},
            'processing_time_distribution': [],
            'artificial_probability_distribution': []
        }
        
        recommendation_counts = {'accept': 0, 'reject': 0, 'expert_review': 0}
        
        for session in history:
            confidence = session.get('overall_confidence', 0)
            fp_prob = session.get('overall_false_positive_probability', 0)
            score = 1 - fp_prob  # Convert FP probability to score
            recommendation = session.get('recommendation', 'unknown')
            processing_time = session.get('total_processing_time_ms', 0)
            
            scatter_data['confidence_vs_score'].append({
                'x': score,
                'y': confidence,
                'recommendation': recommendation,
                'object_designation': session.get('object_designation', 'unknown'),
                'processing_time_ms': processing_time,
                'artificial_likelihood': session.get('artificial_object_likelihood', 0)
            })
            
            # Stage performance data
            for stage in session.get('stage_metrics', []):
                scatter_data['stage_performance'].append({
                    'stage_number': stage.get('stage_number', 0),
                    'stage_name': stage.get('stage_name', 'unknown'),
                    'score': stage.get('score', 0),
                    'confidence': stage.get('confidence', 0),
                    'processing_time_ms': stage.get('processing_time_ms', 0),
                    'passed': stage.get('passed', False)
                })
            
            # Count recommendations
            if recommendation in recommendation_counts:
                recommendation_counts[recommendation] += 1
            
            # Processing time distribution
            scatter_data['processing_time_distribution'].append({
                'processing_time_ms': processing_time,
                'recommendation': recommendation
            })
            
            # Artificial probability distribution
            if session.get('artificial_object_likelihood'):
                scatter_data['artificial_probability_distribution'].append({
                    'artificial_probability': session.get('artificial_object_likelihood', 0),
                    'confidence': confidence,
                    'recommendation': recommendation
                })
        
        scatter_data['recommendation_distribution'] = recommendation_counts
        
        return JSONResponse(content={
            'scatter_data': scatter_data,
            'data_points': len(history),
            'time_range_hours': hours,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get scatter plot data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scatter plot data: {str(e)}")

@router.get("/api/visualization/trends", response_model=Dict[str, Any])
async def get_trend_data(
    hours: int = Query(24, ge=1, le=168, description="Hours of trend data"),
    collector: ValidationMetricsCollector = Depends(get_metrics_collector)
):
    """
    Get time series data for trend analysis and visualization.
    
    Args:
        hours: Hours of trend data to retrieve
        
    Returns:
        Time series data for validation rates, processing times,
        confidence trends, and detection statistics over time.
    """
    try:
        history = collector.get_validation_history(hours)
        dashboard_data = collector.get_dashboard_data()
        
        # Group data by time intervals (hourly)
        hourly_data = {}
        
        for session in history:
            end_time = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))
            hour_key = end_time.strftime('%Y-%m-%d %H:00:00')
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {
                    'validations_count': 0,
                    'total_processing_time': 0,
                    'confidence_sum': 0,
                    'recommendations': {'accept': 0, 'reject': 0, 'expert_review': 0},
                    'artificial_detections': 0
                }
            
            hourly_data[hour_key]['validations_count'] += 1
            hourly_data[hour_key]['total_processing_time'] += session.get('total_processing_time_ms', 0)
            hourly_data[hour_key]['confidence_sum'] += session.get('overall_confidence', 0)
            
            recommendation = session.get('recommendation', 'expert_review')
            if recommendation in hourly_data[hour_key]['recommendations']:
                hourly_data[hour_key]['recommendations'][recommendation] += 1
            
            if session.get('artificial_object_likelihood', 0) > 0.7:
                hourly_data[hour_key]['artificial_detections'] += 1
        
        # Format for time series visualization
        time_series = {
            'timestamps': [],
            'validation_rates': [],
            'average_processing_times': [],
            'average_confidence': [],
            'acceptance_rates': [],
            'rejection_rates': [],
            'artificial_detection_rates': []
        }
        
        for hour_key in sorted(hourly_data.keys()):
            data = hourly_data[hour_key]
            count = data['validations_count']
            
            time_series['timestamps'].append(hour_key)
            time_series['validation_rates'].append(count)
            time_series['average_processing_times'].append(
                data['total_processing_time'] / count if count > 0 else 0
            )
            time_series['average_confidence'].append(
                data['confidence_sum'] / count if count > 0 else 0
            )
            time_series['acceptance_rates'].append(
                data['recommendations']['accept'] / count * 100 if count > 0 else 0
            )
            time_series['rejection_rates'].append(
                data['recommendations']['reject'] / count * 100 if count > 0 else 0
            )
            time_series['artificial_detection_rates'].append(
                data['artificial_detections'] / count * 100 if count > 0 else 0
            )
        
        return JSONResponse(content={
            'time_series': time_series,
            'processing_trends': dashboard_data.get('validation_pipeline', {}).get('processing_trends', {}),
            'hours_analyzed': hours,
            'total_data_points': len(hourly_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get trend data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trend data: {str(e)}")

@router.get("/api/websocket/stats", response_model=Dict[str, Any])
async def get_websocket_statistics(
    ws_manager: ValidationWebSocketManager = Depends(get_websocket_manager)
):
    """
    Get WebSocket connection statistics and status.
    
    Returns:
        WebSocket connection statistics including active connections,
        subscription status, and performance metrics.
    """
    try:
        stats = ws_manager.get_connection_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}")

# Health check endpoint for dashboard services
@router.get("/api/health", response_model=Dict[str, Any])
async def dashboard_health_check():
    """Health check endpoint for dashboard services."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'metrics_collector': metrics_collector is not None,
                'websocket_manager': websocket_manager is not None,
                'fastapi': HAS_FASTAPI
            },
            'version': '1.0.0'
        }
        
        # Check service health
        if not metrics_collector or not websocket_manager:
            health_status['status'] = 'degraded'
            health_status['message'] = 'Some services not initialized'
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Dashboard health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )