"""
Advanced Alert System for aNEOS Validation Dashboard

Comprehensive alert management system for high-confidence artificial object
detections, anomalous validation results, and system monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"

class AlertCategory(Enum):
    """Alert categories."""
    ARTIFICIAL_OBJECT = "artificial_object"
    VALIDATION_ANOMALY = "validation_anomaly"
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class AlertRule:
    """Configuration for alert generation rules."""
    rule_id: str
    name: str
    category: AlertCategory
    level: AlertLevel
    condition: Callable[[Any], bool]
    threshold: float
    cooldown_minutes: int = 5
    description: str = ""
    enabled: bool = True

@dataclass 
class Alert:
    """Alert data structure."""
    alert_id: str
    rule_id: str
    category: AlertCategory
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    auto_resolve: bool = False
    expiry: Optional[datetime] = None

class AlertManager:
    """
    Advanced alert management system for validation dashboard.
    
    Manages alert rules, generates alerts based on conditions,
    handles acknowledgments and resolutions, and provides
    notification capabilities.
    """
    
    def __init__(self, max_alerts: int = 1000):
        """
        Initialize alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to retain in memory
        """
        self.max_alerts = max_alerts
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.rule_cooldowns: Dict[str, datetime] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules for validation monitoring."""
        
        # High-confidence artificial object detection
        self.add_rule(AlertRule(
            rule_id="high_artificial_probability",
            name="High Artificial Object Probability",
            category=AlertCategory.ARTIFICIAL_OBJECT,
            level=AlertLevel.CRITICAL,
            condition=lambda data: data.get('artificial_probability', 0) > 0.9,
            threshold=0.9,
            cooldown_minutes=1,
            description="Object detected with >90% artificial probability"
        ))
        
        # Multi-module artificial detection consensus
        self.add_rule(AlertRule(
            rule_id="multi_module_artificial",
            name="Multi-Module Artificial Detection",
            category=AlertCategory.ARTIFICIAL_OBJECT,
            level=AlertLevel.URGENT,
            condition=lambda data: len(data.get('detection_modules', [])) >= 3 and data.get('artificial_probability', 0) > 0.8,
            threshold=0.8,
            cooldown_minutes=1,
            description="Multiple validation modules detected artificial object"
        ))
        
        # Validation pipeline failure
        self.add_rule(AlertRule(
            rule_id="validation_pipeline_failure",
            name="Validation Pipeline Failure",
            category=AlertCategory.VALIDATION_ANOMALY,
            level=AlertLevel.CRITICAL,
            condition=lambda data: data.get('stage_failures', 0) > 2,
            threshold=2,
            cooldown_minutes=5,
            description="Multiple validation stages failing"
        ))
        
        # Processing time anomaly
        self.add_rule(AlertRule(
            rule_id="processing_time_anomaly",
            name="Processing Time Anomaly",
            category=AlertCategory.PERFORMANCE,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('processing_time_ms', 0) > 5000,
            threshold=5000,
            cooldown_minutes=10,
            description="Validation processing time exceeding 5 seconds"
        ))
        
        # High system resource usage
        self.add_rule(AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            category=AlertCategory.SYSTEM_HEALTH,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('cpu_percent', 0) > 85,
            threshold=85,
            cooldown_minutes=5,
            description="System CPU usage above 85%"
        ))
        
        self.add_rule(AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            category=AlertCategory.SYSTEM_HEALTH,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('memory_percent', 0) > 90,
            threshold=90,
            cooldown_minutes=5,
            description="System memory usage above 90%"
        ))
        
        # Validation module unavailability
        self.add_rule(AlertRule(
            rule_id="module_unavailable",
            name="Validation Module Unavailable",
            category=AlertCategory.SYSTEM_HEALTH,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('module_availability', 1.0) < 0.8,
            threshold=0.8,
            cooldown_minutes=15,
            description="Validation module availability below 80%"
        ))
        
        # Expert review queue overflow
        self.add_rule(AlertRule(
            rule_id="expert_review_overflow",
            name="Expert Review Queue Overflow",
            category=AlertCategory.PERFORMANCE,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('expert_review_queue_size', 0) > 100,
            threshold=100,
            cooldown_minutes=30,
            description="Expert review queue size exceeding 100 objects"
        ))
        
        # Anomalous false positive rate
        self.add_rule(AlertRule(
            rule_id="anomalous_fp_rate",
            name="Anomalous False Positive Rate",
            category=AlertCategory.VALIDATION_ANOMALY,
            level=AlertLevel.WARNING,
            condition=lambda data: data.get('false_positive_rate', 0) > 0.3,
            threshold=0.3,
            cooldown_minutes=20,
            description="False positive rate exceeding 30%"
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule to the manager."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.rule_cooldowns.pop(rule_id, None)
            self.logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = enabled
            self.logger.info(f"{'Enabled' if enabled else 'Disabled'} alert rule: {rule_id}")
    
    async def evaluate_data(self, data: Dict[str, Any], context: Optional[str] = None):
        """
        Evaluate data against all alert rules and generate alerts as needed.
        
        Args:
            data: Data to evaluate against alert rules
            context: Optional context identifier for the data
        """
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule_id in self.rule_cooldowns:
                    cooldown_end = self.rule_cooldowns[rule_id] + timedelta(minutes=rule.cooldown_minutes)
                    if datetime.now() < cooldown_end:
                        continue
                
                # Evaluate condition
                try:
                    if rule.condition(data):
                        await self._generate_alert(rule, data, context)
                        self.rule_cooldowns[rule_id] = datetime.now()
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert data: {e}")
    
    async def _generate_alert(self, rule: AlertRule, data: Dict[str, Any], context: Optional[str] = None):
        """Generate an alert based on rule and data."""
        try:
            alert_id = f"{rule.rule_id}_{int(datetime.now().timestamp())}"
            
            # Create alert title and message based on rule and data
            title = self._format_alert_title(rule, data)
            message = self._format_alert_message(rule, data, context)
            
            # Create alert
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                category=rule.category,
                level=rule.level,
                title=title,
                message=message,
                timestamp=datetime.now(),
                data=data.copy(),
                auto_resolve=rule.category == AlertCategory.SYSTEM_HEALTH,
                expiry=datetime.now() + timedelta(hours=24) if rule.level == AlertLevel.INFO else None
            )
            
            # Add alert to collection
            self.alerts.append(alert)
            
            # Maintain maximum alerts limit
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            self.logger.warning(f"Alert generated: {alert.title} (Level: {alert.level.value})")
            
            # Notify handlers
            await self._notify_handlers(alert)
            
        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")
    
    def _format_alert_title(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Format alert title based on rule and data."""
        if rule.category == AlertCategory.ARTIFICIAL_OBJECT:
            designation = data.get('object_designation', 'Unknown')
            probability = data.get('artificial_probability', 0) * 100
            return f"Artificial Object Detected: {designation} ({probability:.1f}%)"
            
        elif rule.category == AlertCategory.VALIDATION_ANOMALY:
            if 'stage_failures' in data:
                return f"Validation Pipeline Issues ({data['stage_failures']} stage failures)"
            else:
                return f"Validation Anomaly: {rule.name}"
                
        elif rule.category == AlertCategory.SYSTEM_HEALTH:
            if 'cpu_percent' in data:
                return f"High CPU Usage: {data['cpu_percent']:.1f}%"
            elif 'memory_percent' in data:
                return f"High Memory Usage: {data['memory_percent']:.1f}%"
            elif 'module_availability' in data:
                return f"Module Availability: {data['module_availability']*100:.1f}%"
            else:
                return f"System Health: {rule.name}"
                
        elif rule.category == AlertCategory.PERFORMANCE:
            if 'processing_time_ms' in data:
                return f"Slow Processing: {data['processing_time_ms']:.0f}ms"
            elif 'expert_review_queue_size' in data:
                return f"Expert Review Queue: {data['expert_review_queue_size']} objects"
            else:
                return f"Performance: {rule.name}"
        
        return rule.name
    
    def _format_alert_message(self, rule: AlertRule, data: Dict[str, Any], context: Optional[str] = None) -> str:
        """Format alert message with detailed information."""
        message_parts = [rule.description]
        
        if rule.category == AlertCategory.ARTIFICIAL_OBJECT:
            modules = data.get('detection_modules', [])
            confidence = data.get('confidence', 0) * 100
            message_parts.extend([
                f"Detection confidence: {confidence:.1f}%",
                f"Detecting modules: {', '.join(modules)}" if modules else "No module details"
            ])
            
        elif rule.category == AlertCategory.VALIDATION_ANOMALY:
            if 'stage_results' in data:
                failed_stages = [s for s in data['stage_results'] if not s.get('passed', True)]
                if failed_stages:
                    message_parts.append(f"Failed stages: {', '.join(s.get('stage_name', 'Unknown') for s in failed_stages)}")
            
        elif rule.category == AlertCategory.SYSTEM_HEALTH:
            if 'timestamp' in data:
                message_parts.append(f"Detected at: {data['timestamp']}")
        
        if context:
            message_parts.append(f"Context: {context}")
        
        return " | ".join(message_parts)
    
    async def _notify_handlers(self, alert: Alert):
        """Notify all registered handlers about the new alert."""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert notification handler: {e}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def get_alerts(self, 
                   category: Optional[AlertCategory] = None,
                   level: Optional[AlertLevel] = None,
                   resolved: Optional[bool] = None,
                   limit: Optional[int] = None) -> List[Alert]:
        """
        Get alerts with optional filtering.
        
        Args:
            category: Filter by alert category
            level: Filter by alert level
            resolved: Filter by resolved status
            limit: Limit number of results
            
        Returns:
            List of matching alerts, sorted by timestamp (newest first)
        """
        alerts = list(self.alerts)
        
        # Apply filters
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # Remove expired alerts
        now = datetime.now()
        alerts = [a for a in alerts if not a.expiry or a.expiry > now]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24, limit: int = 50) -> List[Alert]:
        """Get recent alerts within specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alerts if a.timestamp >= cutoff]
        recent.sort(key=lambda x: x.timestamp, reverse=True)
        return recent[:limit]
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_by = user
                alert.resolved_at = datetime.now()
                alert.acknowledged = True  # Auto-acknowledge when resolving
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                self.logger.info(f"Alert {alert_id} resolved by {user}")
                return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(weeks=1)
        
        all_alerts = list(self.alerts)
        recent_alerts = [a for a in all_alerts if a.timestamp >= day_ago]
        weekly_alerts = [a for a in all_alerts if a.timestamp >= week_ago]
        
        return {
            'total_alerts': len(all_alerts),
            'recent_24h': len(recent_alerts),
            'recent_7d': len(weekly_alerts),
            'unresolved': len([a for a in all_alerts if not a.resolved]),
            'unacknowledged': len([a for a in all_alerts if not a.acknowledged]),
            'by_level': {
                level.value: len([a for a in all_alerts if a.level == level])
                for level in AlertLevel
            },
            'by_category': {
                category.value: len([a for a in all_alerts if a.category == category])
                for category in AlertCategory
            },
            'alert_rate_24h': len(recent_alerts) / 24.0,  # Per hour
            'resolution_rate': (
                len([a for a in all_alerts if a.resolved]) / max(len(all_alerts), 1) * 100
            ),
            'active_rules': len([r for r in self.alert_rules.values() if r.enabled])
        }
    
    async def cleanup_expired_alerts(self):
        """Remove expired alerts from memory."""
        now = datetime.now()
        initial_count = len(self.alerts)
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.expiry or alert.expiry > now
        ]
        
        removed = initial_count - len(self.alerts)
        if removed > 0:
            self.logger.info(f"Cleaned up {removed} expired alerts")
    
    def export_alerts(self, format: str = 'json') -> str:
        """Export alerts in specified format."""
        if format.lower() == 'json':
            alert_data = []
            for alert in self.alerts:
                alert_dict = {
                    'alert_id': alert.alert_id,
                    'rule_id': alert.rule_id,
                    'category': alert.category.value,
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved,
                    'data': alert.data
                }
                alert_data.append(alert_dict)
            
            return json.dumps(alert_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global alert manager instance
_alert_manager: Optional[AlertManager] = None

def get_alert_manager() -> Optional[AlertManager]:
    """Get the global alert manager instance."""
    global _alert_manager
    if not _alert_manager:
        _alert_manager = AlertManager()
    return _alert_manager

def initialize_alert_manager(max_alerts: int = 1000) -> AlertManager:
    """Initialize the global alert manager."""
    global _alert_manager
    _alert_manager = AlertManager(max_alerts)
    return _alert_manager