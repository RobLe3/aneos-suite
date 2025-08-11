"""
WebSocket Manager for Real-Time Validation Dashboard Updates

Provides WebSocket connections for live streaming of validation pipeline
metrics, alerts, and performance data to dashboard clients.
"""

import asyncio
import json
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import logging

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    WebSocket = None
    WebSocketDisconnect = Exception
    ConnectionClosed = Exception
    HAS_WEBSOCKETS = False

from ..monitoring.validation_metrics import ValidationMetricsCollector

logger = logging.getLogger(__name__)

class ValidationWebSocketManager:
    """
    Manages WebSocket connections for real-time dashboard updates.
    
    Handles multiple concurrent connections, broadcasts validation metrics,
    alerts, and system status updates to connected clients.
    """
    
    def __init__(self, metrics_collector: ValidationMetricsCollector):
        """
        Initialize WebSocket manager.
        
        Args:
            metrics_collector: ValidationMetricsCollector instance for metrics data
        """
        self.metrics_collector = metrics_collector
        self.active_connections: Set[WebSocket] = set()
        self.client_subscriptions: Dict[WebSocket, Dict[str, bool]] = {}
        self.update_task: Optional[asyncio.Task] = None
        self.update_interval = 1.0  # Update every 1 second
        self.logger = logging.getLogger(__name__)
        
        # Message types for subscription filtering
        self.message_types = {
            'validation_metrics': True,
            'system_health': True,
            'alerts': True,
            'stage_performance': True,
            'detection_statistics': True,
            'processing_trends': True
        }
    
    async def connect(self, websocket: WebSocket, client_id: str = None) -> bool:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to accept
            client_id: Optional client identifier
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not HAS_WEBSOCKETS:
            self.logger.error("WebSocket support not available")
            return False
        
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            
            # Default subscription to all message types
            self.client_subscriptions[websocket] = self.message_types.copy()
            
            self.logger.info(f"WebSocket client connected: {client_id or 'anonymous'} "
                           f"(total connections: {len(self.active_connections)})")
            
            # Send initial dashboard data
            await self.send_initial_data(websocket)
            
            # Start update task if this is the first connection
            if len(self.active_connections) == 1 and not self.update_task:
                self.update_task = asyncio.create_task(self._broadcast_updates())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket client: {e}")
            return False
    
    async def disconnect(self, websocket: WebSocket, client_id: str = None):
        """
        Remove a WebSocket connection and clean up.
        
        Args:
            websocket: WebSocket connection to disconnect
            client_id: Optional client identifier
        """
        try:
            self.active_connections.discard(websocket)
            self.client_subscriptions.pop(websocket, None)
            
            self.logger.info(f"WebSocket client disconnected: {client_id or 'anonymous'} "
                           f"(remaining connections: {len(self.active_connections)})")
            
            # Stop update task if no more connections
            if len(self.active_connections) == 0 and self.update_task:
                self.update_task.cancel()
                self.update_task = None
                
        except Exception as e:
            self.logger.error(f"Error during WebSocket disconnect: {e}")
    
    async def send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data to newly connected client."""
        try:
            dashboard_data = self.metrics_collector.get_dashboard_data()
            initial_message = {
                'type': 'initial_data',
                'timestamp': datetime.now().isoformat(),
                'data': dashboard_data
            }
            
            await websocket.send_text(json.dumps(initial_message))
            
        except Exception as e:
            self.logger.error(f"Failed to send initial data: {e}")
    
    async def broadcast_message(self, message_type: str, data: Any):
        """
        Broadcast a message to all connected clients with subscription filtering.
        
        Args:
            message_type: Type of message (for subscription filtering)
            data: Message data to broadcast
        """
        if not self.active_connections:
            return
        
        message = {
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        message_json = json.dumps(message)
        
        # Broadcast to subscribed clients
        disconnected_clients = set()
        
        for websocket in list(self.active_connections):
            try:
                # Check if client is subscribed to this message type
                subscriptions = self.client_subscriptions.get(websocket, {})
                if subscriptions.get(message_type, False):
                    await websocket.send_text(message_json)
                    
            except (WebSocketDisconnect, ConnectionClosed, ConnectionResetError):
                # Client disconnected
                disconnected_clients.add(websocket)
                self.logger.info(f"Client disconnected during broadcast")
                
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected_clients:
            await self.disconnect(websocket)
    
    async def _broadcast_updates(self):
        """
        Background task to periodically broadcast dashboard updates.
        Runs continuously while there are active connections.
        """
        self.logger.info("Starting WebSocket broadcast updates task")
        
        try:
            while self.active_connections:
                # Get fresh dashboard data
                dashboard_data = self.metrics_collector.get_dashboard_data()
                
                # Broadcast different message types based on data changes
                await self.broadcast_message('validation_metrics', {
                    'system_overview': dashboard_data.get('system_overview', {}),
                    'validation_pipeline': dashboard_data.get('validation_pipeline', {}),
                    'real_time_metrics': dashboard_data.get('real_time_metrics', {})
                })
                
                await self.broadcast_message('system_health', {
                    'system_health': dashboard_data.get('system_health', {}),
                    'processing_trends': dashboard_data.get('validation_pipeline', {}).get('processing_trends', {})
                })
                
                await self.broadcast_message('alerts', {
                    'alerts_and_notifications': dashboard_data.get('alerts_and_notifications', {})
                })
                
                await self.broadcast_message('detection_statistics', {
                    'detection_statistics': dashboard_data.get('detection_statistics', {})
                })
                
                # Wait for next update interval
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("WebSocket broadcast updates task cancelled")
        except Exception as e:
            self.logger.error(f"Error in WebSocket broadcast task: {e}")
        finally:
            self.logger.info("WebSocket broadcast updates task ended")
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """
        Handle incoming messages from WebSocket clients.
        
        Args:
            websocket: WebSocket connection that sent the message
            message: JSON message string from client
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Update client subscriptions
                subscriptions = data.get('subscriptions', {})
                self.client_subscriptions[websocket].update(subscriptions)
                
                # Send acknowledgment
                response = {
                    'type': 'subscription_updated',
                    'timestamp': datetime.now().isoformat(),
                    'subscriptions': self.client_subscriptions[websocket]
                }
                await websocket.send_text(json.dumps(response))
                
            elif message_type == 'acknowledge_alert':
                # Acknowledge an alert
                alert_id = data.get('alert_id')
                user = data.get('user', 'dashboard_client')
                
                success = self.metrics_collector.acknowledge_alert(alert_id, user)
                
                response = {
                    'type': 'alert_acknowledged',
                    'timestamp': datetime.now().isoformat(),
                    'alert_id': alert_id,
                    'success': success
                }
                await websocket.send_text(json.dumps(response))
                
                # Broadcast alert update to all clients
                if success:
                    await self.broadcast_message('alerts', {
                        'alerts_and_notifications': self.metrics_collector.get_dashboard_data().get('alerts_and_notifications', {})
                    })
                
            elif message_type == 'resolve_alert':
                # Resolve an alert
                alert_id = data.get('alert_id')
                user = data.get('user', 'dashboard_client')
                
                success = self.metrics_collector.resolve_alert(alert_id, user)
                
                response = {
                    'type': 'alert_resolved',
                    'timestamp': datetime.now().isoformat(),
                    'alert_id': alert_id,
                    'success': success
                }
                await websocket.send_text(json.dumps(response))
                
                # Broadcast alert update to all clients
                if success:
                    await self.broadcast_message('alerts', {
                        'alerts_and_notifications': self.metrics_collector.get_dashboard_data().get('alerts_and_notifications', {})
                    })
            
            elif message_type == 'request_history':
                # Send historical data
                hours = data.get('hours', 24)
                history = self.metrics_collector.get_validation_history(hours)
                
                response = {
                    'type': 'validation_history',
                    'timestamp': datetime.now().isoformat(),
                    'hours': hours,
                    'data': history
                }
                await websocket.send_text(json.dumps(response))
                
            else:
                # Unknown message type
                response = {
                    'type': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'message': f"Unknown message type: {message_type}"
                }
                await websocket.send_text(json.dumps(response))
                
        except json.JSONDecodeError:
            response = {
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'message': 'Invalid JSON message'
            }
            await websocket.send_text(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
            response = {
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'message': f"Error processing message: {str(e)}"
            }
            await websocket.send_text(json.dumps(response))
    
    async def broadcast_validation_result(self, validation_result: Any, processing_time_ms: float):
        """
        Broadcast a new validation result to all connected clients.
        
        Args:
            validation_result: ValidationResult from MultiStageValidator
            processing_time_ms: Processing time in milliseconds
        """
        try:
            # Record the validation session first
            session_id = self.metrics_collector.record_validation_session(
                validation_result, processing_time_ms
            )
            
            # Create broadcast message
            validation_data = {
                'session_id': session_id,
                'object_designation': getattr(validation_result, 'object_designation', 'unknown'),
                'recommendation': validation_result.recommendation,
                'confidence': validation_result.overall_confidence,
                'false_positive_probability': validation_result.overall_false_positive_probability,
                'processing_time_ms': processing_time_ms,
                'stage_results': [
                    {
                        'stage_number': stage.stage_number,
                        'stage_name': stage.stage_name,
                        'passed': stage.passed,
                        'score': stage.score,
                        'confidence': stage.confidence,
                        'processing_time_ms': stage.processing_time_ms
                    }
                    for stage in validation_result.stage_results
                ] if hasattr(validation_result, 'stage_results') else []
            }
            
            await self.broadcast_message('validation_result', validation_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting validation result: {e}")
    
    async def broadcast_artificial_object_alert(self, alert_data: Dict[str, Any]):
        """
        Broadcast an artificial object alert to all connected clients.
        
        Args:
            alert_data: Alert data dictionary
        """
        try:
            await self.broadcast_message('artificial_object_alert', alert_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting artificial object alert: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            'active_connections': len(self.active_connections),
            'update_task_running': self.update_task is not None and not self.update_task.done(),
            'update_interval_seconds': self.update_interval,
            'client_subscriptions': {
                f"client_{i}": subscriptions
                for i, subscriptions in enumerate(self.client_subscriptions.values())
            }
        }