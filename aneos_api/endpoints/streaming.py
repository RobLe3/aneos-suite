"""
Streaming endpoints for aNEOS API.

Provides real-time data streaming capabilities including WebSocket connections,
Server-Sent Events, and live data feeds for monitoring and analysis updates.
"""

from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
    from fastapi.responses import StreamingResponse
    from sse_starlette.sse import EventSourceResponse
    HAS_FASTAPI = True
    HAS_SSE = True
except ImportError:
    HAS_FASTAPI = False
    HAS_SSE = False
    logging.warning("FastAPI or SSE not available, streaming endpoints disabled")

from ..models import StreamingEvent, StreamingEventType
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
            
        def websocket(self, *args, **kwargs): return lambda f: f
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def include_router(self, *args, **kwargs): pass
        def add_api_route(self, *args, **kwargs): pass
        def mount(self, *args, **kwargs): pass
    router = MockRouter()

# Connection management
class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, set] = {}  # session_id -> set of event types
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and store a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.subscriptions[session_id] = set()
        logger.info(f"WebSocket connection established: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.subscriptions:
            del self.subscriptions[session_id]
        logger.info(f"WebSocket connection closed: {session_id}")
    
    def subscribe(self, session_id: str, event_types: list):
        """Subscribe to specific event types."""
        if session_id in self.subscriptions:
            self.subscriptions[session_id].update(event_types)
            logger.info(f"Session {session_id} subscribed to: {event_types}")
    
    async def send_personal_message(self, message: str, session_id: str):
        """Send a message to a specific connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast_event(self, event: StreamingEvent):
        """Broadcast an event to all subscribed connections."""
        if not self.active_connections:
            return
        
        message = json.dumps(event.dict())
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            # Check if session is subscribed to this event type
            if event.event_type in self.subscriptions.get(session_id, set()):
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {session_id}: {e}")
                    disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)

# Global connection manager
connection_manager = ConnectionManager()

# SSE connection tracking
sse_connections: Dict[str, bool] = {}

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time data streaming.
    
    Provides bidirectional communication for real-time updates including
    analysis completions, predictions, alerts, and system status.
    """
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await _handle_websocket_message(websocket, session_id, message)
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'error': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        connection_manager.disconnect(session_id)

@router.get("/events")
async def stream_events(
    event_types: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Server-Sent Events endpoint for real-time updates.
    
    Streams events as they occur using SSE protocol.
    Compatible with EventSource JavaScript API.
    """
    if not HAS_SSE:
        return {"error": "Server-Sent Events not available"}
    
    # Parse event types filter
    subscribed_events = set()
    if event_types:
        subscribed_events = set(event_types.split(','))
    else:
        subscribed_events = {e.value for e in StreamingEventType}
    
    session_id = f"sse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sse_connections[session_id] = True
    
    async def event_generator():
        try:
            logger.info(f"SSE connection started: {session_id}")
            
            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({
                    "session_id": session_id,
                    "subscribed_events": list(subscribed_events),
                    "timestamp": datetime.now().isoformat()
                })
            }
            
            # Stream events
            while sse_connections.get(session_id, False):
                # Get next event (mock implementation)
                event = await _get_next_sse_event(subscribed_events)
                
                if event:
                    yield {
                        "event": event.event_type,
                        "data": json.dumps(event.dict())
                    }
                
                # Wait before next event
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"SSE stream error for {session_id}: {e}")
        finally:
            sse_connections.pop(session_id, None)
            logger.info(f"SSE connection closed: {session_id}")
    
    return EventSourceResponse(event_generator())

@router.get("/metrics/live")
async def stream_live_metrics(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Stream live system metrics via Server-Sent Events."""
    if not HAS_SSE:
        return {"error": "Server-Sent Events not available"}
    
    session_id = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sse_connections[session_id] = True
    
    async def metrics_generator():
        try:
            logger.info(f"Live metrics stream started: {session_id}")
            
            while sse_connections.get(session_id, False):
                # Get current metrics
                from ..app import get_aneos_app  # Import here to avoid circular imports
                aneos_app = get_aneos_app()
                metrics_data = {}
                
                if aneos_app.metrics_collector:
                    system_metrics = aneos_app.metrics_collector.get_system_metrics()
                    analysis_metrics = aneos_app.metrics_collector.get_analysis_metrics()
                    ml_metrics = aneos_app.metrics_collector.get_ml_metrics()
                    
                    metrics_data = {
                        'timestamp': datetime.now().isoformat(),
                        'system': {
                            'cpu_percent': system_metrics.cpu_percent if system_metrics else 0,
                            'memory_percent': system_metrics.memory_percent if system_metrics else 0,
                            'disk_usage_percent': system_metrics.disk_usage_percent if system_metrics else 0
                        },
                        'analysis': {
                            'total_analyses': analysis_metrics.total_analyses if analysis_metrics else 0,
                            'cache_hit_rate': analysis_metrics.cache_hit_rate if analysis_metrics else 0,
                            'average_processing_time': analysis_metrics.average_processing_time if analysis_metrics else 0
                        },
                        'ml': {
                            'model_predictions': ml_metrics.model_predictions if ml_metrics else 0,
                            'prediction_latency': ml_metrics.prediction_latency if ml_metrics else 0,
                            'ensemble_agreement': ml_metrics.ensemble_agreement if ml_metrics else 0
                        }
                    }
                
                yield {
                    "event": "metrics_update",
                    "data": json.dumps(metrics_data)
                }
                
                # Update every 5 seconds
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Metrics stream error for {session_id}: {e}")
        finally:
            sse_connections.pop(session_id, None)
            logger.info(f"Metrics stream closed: {session_id}")
    
    return EventSourceResponse(metrics_generator())

@router.get("/alerts/live")
async def stream_live_alerts(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Stream live alerts via Server-Sent Events."""
    if not HAS_SSE:
        return {"error": "Server-Sent Events not available"}
    
    session_id = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sse_connections[session_id] = True
    
    async def alerts_generator():
        try:
            logger.info(f"Live alerts stream started: {session_id}")
            
            # Track last alert timestamp to avoid duplicates
            last_alert_time = datetime.now()
            
            while sse_connections.get(session_id, False):
                # Get new alerts
                from ..app import get_aneos_app  # Import here to avoid circular imports
                aneos_app = get_aneos_app()
                
                if aneos_app.alert_manager:
                    new_alerts = aneos_app.alert_manager.get_alerts_since(last_alert_time)
                    
                    for alert in new_alerts:
                        alert_data = {
                            'alert_id': alert.alert_id,
                            'alert_type': alert.alert_type,
                            'alert_level': alert.alert_level,
                            'title': alert.title,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat(),
                            'data': alert.data
                        }
                        
                        yield {
                            "event": "alert_generated",
                            "data": json.dumps(alert_data)
                        }
                        
                        last_alert_time = max(last_alert_time, alert.timestamp)
                
                # Check every 2 seconds
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Alerts stream error for {session_id}: {e}")
        finally:
            sse_connections.pop(session_id, None)
            logger.info(f"Alerts stream closed: {session_id}")
    
    return EventSourceResponse(alerts_generator())

@router.post("/broadcast", response_model=Dict[str, Any])
async def broadcast_event(
    event_data: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Broadcast a custom event to all connected WebSocket clients."""
    try:
        # Create streaming event
        event = StreamingEvent(
            event_type=event_data.get('event_type', 'custom'),
            data=event_data.get('data', {}),
            session_id=event_data.get('session_id')
        )
        
        # Broadcast to WebSocket connections
        await connection_manager.broadcast_event(event)
        
        return {
            'status': 'success',
            'event_type': event.event_type,
            'recipients': len(connection_manager.active_connections),
            'broadcast_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to broadcast event: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'broadcast_at': datetime.now().isoformat()
        }

@router.get("/connections", response_model=Dict[str, Any])
async def get_active_connections(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get information about active streaming connections."""
    return {
        'websocket_connections': len(connection_manager.active_connections),
        'sse_connections': len(sse_connections),
        'total_connections': len(connection_manager.active_connections) + len(sse_connections),
        'connection_details': {
            'websocket_sessions': list(connection_manager.active_connections.keys()),
            'sse_sessions': list(sse_connections.keys()),
            'subscriptions': {
                session_id: list(subs)
                for session_id, subs in connection_manager.subscriptions.items()
            }
        },
        'timestamp': datetime.now().isoformat()
    }

# Helper functions
async def _handle_websocket_message(websocket: WebSocket, session_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    message_type = message.get('type')
    
    if message_type == 'subscribe':
        # Subscribe to event types
        event_types = message.get('event_types', [])
        connection_manager.subscribe(session_id, event_types)
        
        await websocket.send_text(json.dumps({
            'type': 'subscription_confirmed',
            'event_types': event_types,
            'timestamp': datetime.now().isoformat()
        }))
        
    elif message_type == 'ping':
        # Respond to ping with pong
        await websocket.send_text(json.dumps({
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        }))
        
    elif message_type == 'get_status':
        # Send current system status
        from ..app import get_aneos_app  # Import here to avoid circular imports
        aneos_app = get_aneos_app()
        status = aneos_app.get_health_status()
        
        await websocket.send_text(json.dumps({
            'type': 'status_response',
            'data': status,
            'timestamp': datetime.now().isoformat()
        }))
        
    else:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': f'Unknown message type: {message_type}',
            'timestamp': datetime.now().isoformat()
        }))

async def _get_next_sse_event(subscribed_events: set) -> Optional[StreamingEvent]:
    """Get the next SSE event (mock implementation)."""
    # Mock event generation for demonstration
    import random
    
    if random.random() < 0.1:  # 10% chance of event
        event_types = list(subscribed_events)
        if event_types:
            event_type = random.choice(event_types)
            
            # Generate mock event data based on type
            if event_type == StreamingEventType.SYSTEM_STATUS:
                data = {
                    'status': 'healthy',
                    'uptime': 3600,
                    'services_online': 6
                }
            elif event_type == StreamingEventType.METRICS_UPDATE:
                data = {
                    'cpu_percent': random.uniform(10, 80),
                    'memory_percent': random.uniform(20, 70),
                    'analyses_count': random.randint(0, 5)
                }
            elif event_type == StreamingEventType.ALERT_GENERATED:
                data = {
                    'alert_level': random.choice(['low', 'medium', 'high']),
                    'title': 'System Alert',
                    'message': 'Mock alert for demonstration'
                }
            else:
                data = {'mock': True, 'event_type': event_type}
            
            return StreamingEvent(
                event_type=event_type,
                data=data
            )
    
    return None

# Utility function to trigger events (for testing)
async def trigger_test_event(event_type: str, data: Dict[str, Any]):
    """Trigger a test event for all connections."""
    event = StreamingEvent(
        event_type=event_type,
        data=data
    )
    
    await connection_manager.broadcast_event(event)