"""
Real-Time Validation Pipeline Metrics Collection

Collects and manages metrics from the multi-stage validation pipeline
for real-time dashboard monitoring and performance analysis.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationStageMetrics:
    """Metrics for a single validation stage."""
    stage_number: int
    stage_name: str
    processing_time_ms: float
    passed: bool
    score: float
    confidence: float
    false_positive_reduction: float
    timestamp: datetime
    object_designation: str

@dataclass
class ValidationSessionMetrics:
    """Complete validation session metrics."""
    session_id: str
    object_designation: str
    start_time: datetime
    end_time: datetime
    total_processing_time_ms: float
    overall_validation_passed: bool
    overall_false_positive_probability: float
    overall_confidence: float
    stage_metrics: List[ValidationStageMetrics]
    recommendation: str
    expert_review_priority: str
    
    # Enhanced module results
    delta_bic_available: bool = False
    spectral_analysis_available: bool = False  
    radar_analysis_available: bool = False
    thermal_ir_available: bool = False
    gaia_analysis_available: bool = False
    
    # Performance indicators
    throughput_objects_per_hour: float = 0.0
    false_positive_prevention_effectiveness: float = 0.0

@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    active_validation_sessions: int
    completed_validations_today: int
    average_processing_time_ms: float
    alert_count: int
    expert_review_queue_size: int

@dataclass
class ArtificialObjectAlert:
    """Alert for high-confidence artificial object detection."""
    alert_id: str
    object_designation: str
    confidence: float
    artificial_probability: float
    detection_modules: List[str]
    timestamp: datetime
    alert_level: str  # 'info', 'warning', 'critical', 'urgent'
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

class ValidationMetricsCollector:
    """
    Collects, processes, and provides real-time validation pipeline metrics
    for dashboard monitoring and operational oversight.
    """
    
    def __init__(self, max_history_hours: int = 24, max_sessions_memory: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history_hours: Maximum hours of metrics history to retain
            max_sessions_memory: Maximum validation sessions to keep in memory
        """
        self.max_history_hours = max_history_hours
        self.max_sessions_memory = max_sessions_memory
        
        # Real-time metrics storage
        self.validation_sessions: deque = deque(maxlen=max_sessions_memory)
        self.system_health_history: deque = deque(maxlen=max_history_hours * 60)  # Per minute
        self.artificial_object_alerts: deque = deque(maxlen=500)
        
        # Current session tracking
        self.active_sessions: Dict[str, datetime] = {}
        self.stage_performance: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.module_availability: Dict[str, bool] = {
            'delta_bic': True,
            'spectral_analysis': True,
            'radar_analysis': True,
            'thermal_ir': True,
            'gaia_analysis': True
        }
        
        # Performance counters
        self.daily_stats = {
            'total_validations': 0,
            'accepted_objects': 0,
            'rejected_objects': 0,
            'expert_review_objects': 0,
            'artificial_detections': 0,
            'false_positive_prevention_rate': 0.0,
            'average_confidence': 0.0
        }
        
        # Real-time aggregation
        self.real_time_metrics = {
            'validations_per_minute': 0.0,
            'current_throughput': 0.0,
            'stage_bottlenecks': [],
            'module_performance': {},
            'alert_rate': 0.0
        }
        
        self.last_update = datetime.now()
        self.logger = logging.getLogger(__name__)
        
    def record_validation_session(self, validation_result: Any, processing_time_ms: float) -> str:
        """
        Record a completed validation session.
        
        Args:
            validation_result: ValidationResult from MultiStageValidator
            processing_time_ms: Total processing time in milliseconds
            
        Returns:
            session_id: Unique identifier for this validation session
        """
        try:
            session_id = f"validation_{int(time.time())}_{len(self.validation_sessions)}"
            timestamp = datetime.now()
            
            # Extract stage metrics from validation result
            stage_metrics = []
            if hasattr(validation_result, 'stage_results'):
                for stage_result in validation_result.stage_results:
                    stage_metric = ValidationStageMetrics(
                        stage_number=stage_result.stage_number,
                        stage_name=stage_result.stage_name,
                        processing_time_ms=stage_result.processing_time_ms,
                        passed=stage_result.passed,
                        score=stage_result.score,
                        confidence=stage_result.confidence,
                        false_positive_reduction=stage_result.false_positive_reduction,
                        timestamp=timestamp,
                        object_designation=getattr(validation_result, 'object_designation', 'unknown')
                    )
                    stage_metrics.append(stage_metric)
                    
                    # Update stage performance tracking
                    self.stage_performance[stage_result.stage_number]['total_time'] += stage_result.processing_time_ms
                    self.stage_performance[stage_result.stage_number]['count'] += 1
                    self.stage_performance[stage_result.stage_number]['success_rate'] = (
                        (self.stage_performance[stage_result.stage_number].get('successes', 0) + (1 if stage_result.passed else 0)) /
                        self.stage_performance[stage_result.stage_number]['count']
                    )
                    
                    if stage_result.passed:
                        self.stage_performance[stage_result.stage_number]['successes'] = (
                            self.stage_performance[stage_result.stage_number].get('successes', 0) + 1
                        )
            
            # Create session metrics
            session_metrics = ValidationSessionMetrics(
                session_id=session_id,
                object_designation=getattr(validation_result, 'object_designation', 'unknown'),
                start_time=timestamp - timedelta(milliseconds=processing_time_ms),
                end_time=timestamp,
                total_processing_time_ms=processing_time_ms,
                overall_validation_passed=validation_result.overall_validation_passed,
                overall_false_positive_probability=validation_result.overall_false_positive_probability,
                overall_confidence=validation_result.overall_confidence,
                stage_metrics=stage_metrics,
                recommendation=validation_result.recommendation,
                expert_review_priority=validation_result.expert_review_priority,
                delta_bic_available=validation_result.delta_bic_analysis is not None,
                spectral_analysis_available=validation_result.spectral_analysis_result is not None,
                radar_analysis_available=validation_result.radar_polarization_result is not None,
                thermal_ir_available=validation_result.thermal_ir_result is not None,
                gaia_analysis_available=validation_result.gaia_astrometric_result is not None
            )
            
            # Store session
            self.validation_sessions.append(session_metrics)
            
            # Update daily statistics
            self.daily_stats['total_validations'] += 1
            if validation_result.recommendation == 'accept':
                self.daily_stats['accepted_objects'] += 1
            elif validation_result.recommendation == 'reject':
                self.daily_stats['rejected_objects'] += 1
            else:
                self.daily_stats['expert_review_objects'] += 1
            
            # Update confidence running average
            self.daily_stats['average_confidence'] = (
                (self.daily_stats['average_confidence'] * (self.daily_stats['total_validations'] - 1) +
                 validation_result.overall_confidence) / self.daily_stats['total_validations']
            )
            
            # Check for artificial object alerts
            self._check_artificial_object_alert(validation_result)
            
            # Update real-time metrics
            self._update_real_time_metrics()
            
            self.logger.info(f"Recorded validation session {session_id} for {session_metrics.object_designation}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to record validation session: {e}")
            return ""
    
    def _check_artificial_object_alert(self, validation_result: Any):
        """Check if validation result warrants an artificial object alert."""
        try:
            # High-confidence artificial object detection criteria
            artificial_indicators = []
            artificial_probability = 0.0
            
            # Check various artificial object indicators
            if hasattr(validation_result, 'artificial_object_likelihood') and validation_result.artificial_object_likelihood:
                if validation_result.artificial_object_likelihood > 0.7:
                    artificial_indicators.append('delta_bic_analysis')
                    artificial_probability = max(artificial_probability, validation_result.artificial_object_likelihood)
            
            if hasattr(validation_result, 'artificial_material_probability') and validation_result.artificial_material_probability:
                if validation_result.artificial_material_probability > 0.7:
                    artificial_indicators.append('spectral_analysis')
                    artificial_probability = max(artificial_probability, validation_result.artificial_material_probability)
            
            if hasattr(validation_result, 'radar_artificial_probability') and validation_result.radar_artificial_probability:
                if validation_result.radar_artificial_probability > 0.7:
                    artificial_indicators.append('radar_polarization')
                    artificial_probability = max(artificial_probability, validation_result.radar_artificial_probability)
            
            if hasattr(validation_result, 'thermal_artificial_probability') and validation_result.thermal_artificial_probability:
                if validation_result.thermal_artificial_probability > 0.7:
                    artificial_indicators.append('thermal_ir_analysis')
                    artificial_probability = max(artificial_probability, validation_result.thermal_artificial_probability)
            
            if hasattr(validation_result, 'gaia_artificial_probability') and validation_result.gaia_artificial_probability:
                if validation_result.gaia_artificial_probability > 0.7:
                    artificial_indicators.append('gaia_astrometry')
                    artificial_probability = max(artificial_probability, validation_result.gaia_artificial_probability)
            
            # Generate alert if criteria met
            if artificial_indicators and artificial_probability > 0.7:
                # Determine alert level
                if artificial_probability > 0.9 and len(artificial_indicators) >= 3:
                    alert_level = 'urgent'
                elif artificial_probability > 0.85 and len(artificial_indicators) >= 2:
                    alert_level = 'critical'
                elif artificial_probability > 0.8:
                    alert_level = 'warning'
                else:
                    alert_level = 'info'
                
                alert = ArtificialObjectAlert(
                    alert_id=f"artificial_{int(time.time())}_{len(self.artificial_object_alerts)}",
                    object_designation=getattr(validation_result, 'object_designation', 'unknown'),
                    confidence=validation_result.overall_confidence,
                    artificial_probability=artificial_probability,
                    detection_modules=artificial_indicators,
                    timestamp=datetime.now(),
                    alert_level=alert_level,
                    details={
                        'validation_recommendation': validation_result.recommendation,
                        'expert_review_priority': validation_result.expert_review_priority,
                        'false_positive_probability': validation_result.overall_false_positive_probability,
                        'detection_details': {
                            module: getattr(validation_result, f"{module}_result", None) 
                            for module in artificial_indicators
                        }
                    }
                )
                
                self.artificial_object_alerts.append(alert)
                self.daily_stats['artificial_detections'] += 1
                
                self.logger.warning(
                    f"Artificial object alert generated: {alert.object_designation} "
                    f"(probability: {artificial_probability:.3f}, modules: {artificial_indicators})"
                )
        
        except Exception as e:
            self.logger.error(f"Failed to check artificial object alert: {e}")
    
    def _update_real_time_metrics(self):
        """Update real-time aggregated metrics."""
        try:
            now = datetime.now()
            
            # Calculate validations per minute over last 5 minutes
            five_minutes_ago = now - timedelta(minutes=5)
            recent_sessions = [
                s for s in self.validation_sessions 
                if s.end_time >= five_minutes_ago
            ]
            
            self.real_time_metrics['validations_per_minute'] = len(recent_sessions) / 5.0
            self.real_time_metrics['current_throughput'] = len(recent_sessions) * 12.0  # Per hour
            
            # Identify stage bottlenecks
            bottlenecks = []
            for stage_num, metrics in self.stage_performance.items():
                if metrics['count'] > 0:
                    avg_time = metrics['total_time'] / metrics['count']
                    if avg_time > 1000:  # > 1 second
                        bottlenecks.append({
                            'stage_number': stage_num,
                            'average_time_ms': avg_time,
                            'success_rate': metrics.get('success_rate', 0.0)
                        })
            
            self.real_time_metrics['stage_bottlenecks'] = sorted(
                bottlenecks, key=lambda x: x['average_time_ms'], reverse=True
            )[:3]  # Top 3 bottlenecks
            
            # Calculate module performance
            if recent_sessions:
                module_performance = {
                    'delta_bic': sum(1 for s in recent_sessions if s.delta_bic_available) / len(recent_sessions),
                    'spectral_analysis': sum(1 for s in recent_sessions if s.spectral_analysis_available) / len(recent_sessions),
                    'radar_analysis': sum(1 for s in recent_sessions if s.radar_analysis_available) / len(recent_sessions),
                    'thermal_ir': sum(1 for s in recent_sessions if s.thermal_ir_available) / len(recent_sessions),
                    'gaia_analysis': sum(1 for s in recent_sessions if s.gaia_analysis_available) / len(recent_sessions)
                }
                self.real_time_metrics['module_performance'] = module_performance
            
            # Calculate alert rate (alerts per hour over last hour)
            one_hour_ago = now - timedelta(hours=1)
            recent_alerts = [
                a for a in self.artificial_object_alerts 
                if a.timestamp >= one_hour_ago
            ]
            self.real_time_metrics['alert_rate'] = len(recent_alerts)
            
            # Update false positive prevention rate
            if self.daily_stats['total_validations'] > 0:
                self.daily_stats['false_positive_prevention_rate'] = (
                    (self.daily_stats['rejected_objects'] + 
                     self.daily_stats['artificial_detections']) /
                    self.daily_stats['total_validations']
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update real-time metrics: {e}")
    
    def record_system_health(self, cpu_percent: float, memory_percent: float, 
                           active_sessions: int, alert_count: int, review_queue_size: int):
        """Record current system health metrics."""
        try:
            health_metrics = SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                active_validation_sessions=active_sessions,
                completed_validations_today=self.daily_stats['total_validations'],
                average_processing_time_ms=self._calculate_average_processing_time(),
                alert_count=alert_count,
                expert_review_queue_size=review_queue_size
            )
            
            self.system_health_history.append(health_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to record system health: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for real-time monitoring."""
        try:
            now = datetime.now()
            
            # Get recent validation sessions (last hour)
            one_hour_ago = now - timedelta(hours=1)
            recent_sessions = [
                s for s in self.validation_sessions 
                if s.end_time >= one_hour_ago
            ]
            
            # Get unresolved alerts
            unresolved_alerts = [a for a in self.artificial_object_alerts if not a.resolved]
            
            # Calculate validation statistics
            stage_stats = {}
            for stage_num in range(1, 6):
                stage_sessions = [s for s in recent_sessions if any(sm.stage_number == stage_num for sm in s.stage_metrics)]
                if stage_sessions:
                    stage_metrics = [sm for s in stage_sessions for sm in s.stage_metrics if sm.stage_number == stage_num]
                    stage_stats[f"stage_{stage_num}"] = {
                        'avg_processing_time_ms': np.mean([sm.processing_time_ms for sm in stage_metrics]),
                        'pass_rate': np.mean([sm.passed for sm in stage_metrics]),
                        'avg_score': np.mean([sm.score for sm in stage_metrics]),
                        'avg_confidence': np.mean([sm.confidence for sm in stage_metrics])
                    }
            
            return {
                'timestamp': now.isoformat(),
                'system_overview': {
                    'total_validations_today': self.daily_stats['total_validations'],
                    'validations_last_hour': len(recent_sessions),
                    'current_throughput_per_hour': self.real_time_metrics['current_throughput'],
                    'active_alerts': len(unresolved_alerts),
                    'expert_review_queue': sum(1 for s in recent_sessions if s.recommendation == 'expert_review'),
                    'false_positive_prevention_rate': self.daily_stats['false_positive_prevention_rate'],
                    'average_confidence': self.daily_stats['average_confidence']
                },
                'validation_pipeline': {
                    'stage_performance': stage_stats,
                    'bottlenecks': self.real_time_metrics['stage_bottlenecks'],
                    'module_availability': self.real_time_metrics['module_performance'],
                    'processing_trends': self._get_processing_trends()
                },
                'detection_statistics': {
                    'accepted_objects': self.daily_stats['accepted_objects'],
                    'rejected_objects': self.daily_stats['rejected_objects'],
                    'expert_review_objects': self.daily_stats['expert_review_objects'],
                    'artificial_detections': self.daily_stats['artificial_detections'],
                    'confidence_distribution': self._get_confidence_distribution()
                },
                'alerts_and_notifications': {
                    'recent_alerts': [asdict(a) for a in list(self.artificial_object_alerts)[-10:]],
                    'alert_rate_per_hour': self.real_time_metrics['alert_rate'],
                    'critical_alerts': len([a for a in unresolved_alerts if a.alert_level in ['critical', 'urgent']])
                },
                'real_time_metrics': self.real_time_metrics,
                'system_health': self._get_latest_system_health()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_validation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get validation session history for specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_sessions = [
                asdict(s) for s in self.validation_sessions 
                if s.end_time >= cutoff_time
            ]
            return recent_sessions
            
        except Exception as e:
            self.logger.error(f"Failed to get validation history: {e}")
            return []
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time from recent sessions."""
        if not self.validation_sessions:
            return 0.0
        
        recent_sessions = list(self.validation_sessions)[-100:]  # Last 100 sessions
        if not recent_sessions:
            return 0.0
        
        return sum(s.total_processing_time_ms for s in recent_sessions) / len(recent_sessions)
    
    def _get_processing_trends(self) -> Dict[str, Any]:
        """Calculate processing time trends."""
        if len(self.validation_sessions) < 10:
            return {'trend': 'insufficient_data'}
        
        # Get recent processing times
        recent_times = [s.total_processing_time_ms for s in list(self.validation_sessions)[-50:]]
        older_times = [s.total_processing_time_ms for s in list(self.validation_sessions)[-100:-50]] if len(self.validation_sessions) >= 100 else []
        
        if older_times:
            recent_avg = np.mean(recent_times)
            older_avg = np.mean(older_times)
            
            if recent_avg > older_avg * 1.1:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_average_ms': np.mean(recent_times),
            'median_ms': np.median(recent_times),
            'std_deviation_ms': np.std(recent_times)
        }
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        if not self.validation_sessions:
            return {}
        
        confidences = [s.overall_confidence for s in self.validation_sessions]
        
        # Bin confidences
        bins = {
            'very_low': sum(1 for c in confidences if c < 0.2),
            'low': sum(1 for c in confidences if 0.2 <= c < 0.4),
            'medium': sum(1 for c in confidences if 0.4 <= c < 0.6),
            'high': sum(1 for c in confidences if 0.6 <= c < 0.8),
            'very_high': sum(1 for c in confidences if c >= 0.8)
        }
        
        return bins
    
    def _get_latest_system_health(self) -> Optional[Dict[str, Any]]:
        """Get latest system health metrics."""
        if not self.system_health_history:
            return None
        
        latest = self.system_health_history[-1]
        return asdict(latest)
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an artificial object alert."""
        try:
            for alert in self.artificial_object_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an artificial object alert."""
        try:
            for alert in self.artificial_object_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.acknowledged = True
                    self.logger.info(f"Alert {alert_id} resolved by {user}")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False