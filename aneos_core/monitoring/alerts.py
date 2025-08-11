"""
Alert management system for aNEOS monitoring.

This module provides comprehensive alerting capabilities including
rule-based alerts, notification management, and alert correlation.
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import asyncio
import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import threading

from ..ml.prediction import Alert as MLAlert, PredictionResult
from ..analysis.scoring import AnomalyScore

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts."""
    ANOMALOUS_NEO = "anomalous_neo"
    SYSTEM_PERFORMANCE = "system_performance"
    MODEL_DRIFT = "model_drift"
    DATA_QUALITY = "data_quality"
    SYSTEM_ERROR = "system_error"

@dataclass
class AlertRule:
    """Rule for generating alerts."""
    rule_id: str
    name: str
    alert_type: AlertType
    alert_level: AlertLevel
    condition: str  # Human-readable condition description
    threshold_function: Callable[[Dict[str, Any]], bool]
    enabled: bool = True
    cooldown_minutes: int = 60  # Minimum time between identical alerts
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met."""
        if not self.enabled:
            return False
        
        try:
            return self.threshold_function(data)
        except Exception as e:
            logger.error(f"Alert rule evaluation failed for {self.rule_id}: {e}")
            return False

@dataclass
class Alert:
    """System alert with full metadata."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'alert_type': self.alert_type.value,
            'alert_level': self.alert_level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'tags': list(self.tags)
        }

class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, channel_id: str, enabled: bool = True):
        self.channel_id = channel_id
        self.enabled = enabled
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, channel_id: str, smtp_server: str, smtp_port: int,
                 username: str, password: str, recipients: List[str],
                 use_tls: bool = True, enabled: bool = True):
        super().__init__(channel_id, enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.enabled or not self.recipients:
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"aNEOS Alert: {alert.alert_level.value.upper()} - {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        level_colors = {
            AlertLevel.LOW: "#FFA500",
            AlertLevel.MEDIUM: "#FF8C00",
            AlertLevel.HIGH: "#FF4500",
            AlertLevel.CRITICAL: "#FF0000"
        }
        
        color = level_colors.get(alert.alert_level, "#808080")
        
        html = f"""
        <html>
        <body>
            <h2 style="color: {color};">aNEOS Alert: {alert.alert_level.value.upper()}</h2>
            <h3>{alert.title}</h3>
            <p><strong>Alert ID:</strong> {alert.alert_id}</p>
            <p><strong>Type:</strong> {alert.alert_type.value}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <hr>
            <h4>Details:</h4>
            <p>{alert.message}</p>
        """
        
        if alert.data:
            html += "<h4>Additional Data:</h4><ul>"
            for key, value in alert.data.items():
                if isinstance(value, (int, float, str, bool)):
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
            <hr>
            <p><em>This is an automated alert from the aNEOS system.</em></p>
        </body>
        </html>
        """
        
        return html

class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, channel_id: str, webhook_url: str,
                 headers: Optional[Dict[str, str]] = None, enabled: bool = True):
        super().__init__(channel_id, enabled)
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.enabled:
            return False
        
        try:
            import aiohttp
            
            payload = {
                'alert': alert.to_dict(),
                'system': 'aNEOS',
                'timestamp': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent for {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {e}")
            return False

class AlertManager:
    """Manages alerts, rules, and notifications."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alert tracking
        self.alert_history: Dict[str, datetime] = {}  # Track cooldowns
        self.alert_counts = {level: 0 for level in AlertLevel}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Create default rules
        self._create_default_rules()
        
        logger.info("AlertManager initialized")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.channel_id}")
    
    def check_anomaly_alert(self, prediction_result: PredictionResult, anomaly_score: AnomalyScore) -> None:
        """Check if NEO analysis results warrant an alert."""
        data = {
            'designation': prediction_result.designation,
            'ml_probability': prediction_result.anomaly_probability,
            'anomaly_score': prediction_result.anomaly_score,
            'classification': anomaly_score.classification,
            'confidence': prediction_result.confidence,
            'risk_factors': anomaly_score.risk_factors
        }
        
        self.evaluate_rules(AlertType.ANOMALOUS_NEO, data)
    
    def check_system_performance(self, metrics: Dict[str, Any]) -> None:
        """Check system performance metrics for alerts."""
        self.evaluate_rules(AlertType.SYSTEM_PERFORMANCE, metrics)
    
    def check_data_quality(self, quality_metrics: Dict[str, Any]) -> None:
        """Check data quality metrics for alerts."""
        self.evaluate_rules(AlertType.DATA_QUALITY, quality_metrics)
    
    def evaluate_rules(self, alert_type: AlertType, data: Dict[str, Any]) -> None:
        """Evaluate all rules of a specific type."""
        with self._lock:
            for rule in self.rules.values():
                if rule.alert_type == alert_type and rule.enabled:
                    try:
                        if rule.evaluate(data):
                            self._create_alert(rule, data)
                    except Exception as e:
                        logger.error(f"Rule evaluation failed for {rule.rule_id}: {e}")
    
    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create and process a new alert."""
        # Check cooldown
        cooldown_key = f"{rule.rule_id}_{hash(str(sorted(data.items())))}"
        
        if cooldown_key in self.alert_history:
            last_alert_time = self.alert_history[cooldown_key]
            if datetime.now() - last_alert_time < timedelta(minutes=rule.cooldown_minutes):
                return None  # Still in cooldown
        
        # Create alert
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.rule_id}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            title=self._generate_alert_title(rule, data),
            message=self._generate_alert_message(rule, data),
            timestamp=datetime.now(),
            data=data.copy()
        )
        
        # Add to alerts list
        self.alerts.append(alert)
        self.alert_counts[rule.alert_level] += 1
        self.alert_history[cooldown_key] = alert.timestamp
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        logger.info(f"Alert created: {alert.alert_id} ({alert.alert_level.value})")
        return alert
    
    def _generate_alert_title(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert title."""
        if rule.alert_type == AlertType.ANOMALOUS_NEO:
            designation = data.get('designation', 'Unknown')
            return f"High Anomaly Score Detected: {designation}"
        elif rule.alert_type == AlertType.SYSTEM_PERFORMANCE:
            return "System Performance Issue Detected"
        elif rule.alert_type == AlertType.DATA_QUALITY:
            return "Data Quality Issue Detected"
        else:
            return f"{rule.name} Alert"
    
    def _generate_alert_message(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert message."""
        if rule.alert_type == AlertType.ANOMALOUS_NEO:
            designation = data.get('designation', 'Unknown')
            ml_prob = data.get('ml_probability', 0)
            classification = data.get('classification', 'unknown')
            risk_factors = data.get('risk_factors', [])
            
            message = f"NEO {designation} has been classified as '{classification}' with ML probability {ml_prob:.3f}."
            
            if risk_factors:
                message += f" Risk factors: {', '.join(risk_factors[:3])}"
                if len(risk_factors) > 3:
                    message += f" and {len(risk_factors) - 3} more."
            
            return message
        
        elif rule.alert_type == AlertType.SYSTEM_PERFORMANCE:
            return f"System performance metrics have exceeded thresholds: {rule.condition}"
        
        elif rule.alert_type == AlertType.DATA_QUALITY:
            return f"Data quality issues detected: {rule.condition}"
        
        else:
            return f"Alert condition met: {rule.condition}"
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for alert."""
        for channel in self.notification_channels.values():
            try:
                await channel.send_notification(alert)
            except Exception as e:
                logger.error(f"Notification failed for channel {channel.channel_id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    
                    logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
            
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
            
            return False
    
    def get_active_alerts(self, alert_level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if alert_level:
                active_alerts = [alert for alert in active_alerts if alert.alert_level == alert_level]
            
            return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from recent time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
            return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        with self._lock:
            total_alerts = len(self.alerts)
            active_alerts = len([alert for alert in self.alerts if not alert.resolved])
            acknowledged_alerts = len([alert for alert in self.alerts if alert.acknowledged])
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'resolved_alerts': total_alerts - active_alerts,
                'alerts_by_level': dict(self.alert_counts),
                'active_rules': len([rule for rule in self.rules.values() if rule.enabled]),
                'notification_channels': len(self.notification_channels),
                'recent_alerts_24h': len(self.get_recent_alerts(24))
            }
    
    def _create_default_rules(self) -> None:
        """Create default alert rules."""
        
        # High anomaly NEO alert
        high_anomaly_rule = AlertRule(
            rule_id="high_anomaly_neo",
            name="High Anomaly NEO Detection",
            alert_type=AlertType.ANOMALOUS_NEO,
            alert_level=AlertLevel.HIGH,
            condition="ML probability > 0.8 AND classification is highly_suspicious or artificial",
            threshold_function=lambda data: (
                data.get('ml_probability', 0) > 0.8 and 
                data.get('classification', '') in ['highly_suspicious', 'artificial']
            ),
            cooldown_minutes=30
        )
        self.add_rule(high_anomaly_rule)
        
        # Critical anomaly NEO alert
        critical_anomaly_rule = AlertRule(
            rule_id="critical_anomaly_neo",
            name="Critical Anomaly NEO Detection",
            alert_type=AlertType.ANOMALOUS_NEO,
            alert_level=AlertLevel.CRITICAL,
            condition="ML probability > 0.9 AND classification is artificial",
            threshold_function=lambda data: (
                data.get('ml_probability', 0) > 0.9 and 
                data.get('classification', '') == 'artificial'
            ),
            cooldown_minutes=15
        )
        self.add_rule(critical_anomaly_rule)
        
        # System performance alert
        performance_rule = AlertRule(
            rule_id="system_performance",
            name="System Performance Degradation",
            alert_type=AlertType.SYSTEM_PERFORMANCE,
            alert_level=AlertLevel.MEDIUM,
            condition="Processing time > 60s OR error rate > 10%",
            threshold_function=lambda data: (
                data.get('average_processing_time', 0) > 60 or 
                data.get('error_rate', 0) > 0.1
            ),
            cooldown_minutes=60
        )
        self.add_rule(performance_rule)
        
        # Data quality alert
        data_quality_rule = AlertRule(
            rule_id="data_quality",
            name="Data Quality Issues",
            alert_type=AlertType.DATA_QUALITY,
            alert_level=AlertLevel.MEDIUM,
            condition="Data completeness < 70% OR validity < 80%",
            threshold_function=lambda data: (
                data.get('completeness', 1.0) < 0.7 or 
                data.get('validity', 1.0) < 0.8
            ),
            cooldown_minutes=120
        )
        self.add_rule(data_quality_rule)
    
    def load_config(self, config_path: str) -> None:
        """Load alert configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load notification channels
            for channel_config in config.get('notification_channels', []):
                if channel_config['type'] == 'email':
                    channel = EmailNotificationChannel(
                        channel_id=channel_config['id'],
                        smtp_server=channel_config['smtp_server'],
                        smtp_port=channel_config['smtp_port'],
                        username=channel_config['username'],
                        password=channel_config['password'],
                        recipients=channel_config['recipients'],
                        use_tls=channel_config.get('use_tls', True),
                        enabled=channel_config.get('enabled', True)
                    )
                    self.add_notification_channel(channel)
                
                elif channel_config['type'] == 'webhook':
                    channel = WebhookNotificationChannel(
                        channel_id=channel_config['id'],
                        webhook_url=channel_config['webhook_url'],
                        headers=channel_config.get('headers'),
                        enabled=channel_config.get('enabled', True)
                    )
                    self.add_notification_channel(channel)
            
            logger.info(f"Alert configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}")
    
    def save_config(self, config_path: str) -> None:
        """Save current alert configuration to file."""
        try:
            config = {
                'rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'alert_type': rule.alert_type.value,
                        'alert_level': rule.alert_level.value,
                        'condition': rule.condition,
                        'enabled': rule.enabled,
                        'cooldown_minutes': rule.cooldown_minutes,
                        'metadata': rule.metadata
                    }
                    for rule in self.rules.values()
                ],
                'notification_channels': [
                    {
                        'id': channel.channel_id,
                        'type': type(channel).__name__.replace('NotificationChannel', '').lower(),
                        'enabled': channel.enabled
                    }
                    for channel in self.notification_channels.values()
                ]
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Alert configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save alert configuration: {e}")
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """Clean up alerts older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with self._lock:
            initial_count = len(self.alerts)
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
            removed_count = initial_count - len(self.alerts)
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old alerts")
            
            return removed_count
    
    def get_alerts(self, level: Optional[str] = None, resolved: Optional[bool] = None, limit: int = 50) -> List[Alert]:
        """Get alerts with optional filtering."""
        with self._lock:
            filtered_alerts = self.alerts.copy()
            
            # Filter by level
            if level:
                try:
                    alert_level = AlertLevel(level.lower())
                    filtered_alerts = [a for a in filtered_alerts if a.alert_level == alert_level]
                except ValueError:
                    logger.warning(f"Invalid alert level filter: {level}")
            
            # Filter by resolved status
            if resolved is not None:
                filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
            
            # Sort by timestamp (newest first) and limit
            sorted_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
            return sorted_alerts[:limit]
    
    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    # Add resolved_by metadata if provided
                    if resolved_by:
                        alert.data['resolved_by'] = resolved_by
                    
                    logger.info(f"Alert resolved: {alert_id}" + (f" by {resolved_by}" if resolved_by else ""))
                    return True
            
            return False