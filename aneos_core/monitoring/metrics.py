"""
Metrics collection and monitoring for aNEOS system.

This module provides comprehensive metrics collection including
system performance, analysis quality, and operational statistics.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import psutil
import logging
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'load_average': self.load_average
        }

@dataclass
class AnalysisMetrics:
    """Analysis pipeline performance metrics."""
    timestamp: datetime
    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    average_processing_time: float
    cache_hit_rate: float
    data_quality_score: float
    anomaly_detection_rate: float
    indicator_performance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_analyses': self.total_analyses,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'average_processing_time': self.average_processing_time,
            'cache_hit_rate': self.cache_hit_rate,
            'data_quality_score': self.data_quality_score,
            'anomaly_detection_rate': self.anomaly_detection_rate,
            'indicator_performance': self.indicator_performance
        }

@dataclass
class MLMetrics:
    """Machine learning model performance metrics."""
    timestamp: datetime
    model_predictions: int
    prediction_latency: float
    model_accuracy: Optional[float]
    feature_quality: float
    ensemble_agreement: float
    alert_count: int
    model_drift_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_predictions': self.model_predictions,
            'prediction_latency': self.prediction_latency,
            'model_accuracy': self.model_accuracy,
            'feature_quality': self.feature_quality,
            'ensemble_agreement': self.ensemble_agreement,
            'alert_count': self.alert_count,
            'model_drift_score': self.model_drift_score
        }

class MetricsBuffer:
    """Thread-safe buffer for storing metrics with retention policy."""
    
    def __init__(self, max_size: int = 1000, retention_hours: int = 24):
        self.max_size = max_size
        self.retention_hours = retention_hours
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
    
    def add(self, metric: Any) -> None:
        """Add metric to buffer."""
        with self._lock:
            self.buffer.append(metric)
            self._cleanup_old_metrics()
    
    def get_recent(self, hours: int = 1) -> List[Any]:
        """Get metrics from recent time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = []
            for metric in reversed(self.buffer):
                if hasattr(metric, 'timestamp') and metric.timestamp >= cutoff_time:
                    recent_metrics.append(metric)
                elif hasattr(metric, 'timestamp') and metric.timestamp < cutoff_time:
                    break  # Metrics are in chronological order
            
            return list(reversed(recent_metrics))
    
    def get_all(self) -> List[Any]:
        """Get all metrics in buffer."""
        with self._lock:
            return list(self.buffer)
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        if not self.buffer:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Remove old metrics from the left
        while (self.buffer and 
               hasattr(self.buffer[0], 'timestamp') and 
               self.buffer[0].timestamp < cutoff_time):
            self.buffer.popleft()
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self.buffer.clear()

class MetricsCollector:
    """Main metrics collection coordinator."""
    
    def __init__(self, collection_interval: int = 60):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval in seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.running = False
        
        # Metric buffers
        self.system_metrics = MetricsBuffer(max_size=1440, retention_hours=24)  # 24 hours of minute-level data
        self.analysis_metrics = MetricsBuffer(max_size=1440, retention_hours=24)
        self.ml_metrics = MetricsBuffer(max_size=1440, retention_hours=24)
        
        # Collection thread
        self.collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Custom metric counters
        self.custom_counters: Dict[str, int] = defaultdict(int)
        self.custom_gauges: Dict[str, float] = {}
        self.custom_timers: Dict[str, List[float]] = defaultdict(list)
        
        # Analysis pipeline references (set externally)
        self.analysis_pipeline = None
        self.ml_predictor = None
        self.alert_manager = None
        
        logger.info(f"MetricsCollector initialized with {collection_interval}s interval")
    
    def start_collection(self) -> None:
        """Start automated metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self._stop_event.clear()
        
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop automated metrics collection."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    self.system_metrics.add(system_metrics)
                
                # Collect analysis metrics
                if self.analysis_pipeline:
                    analysis_metrics = self._collect_analysis_metrics()
                    if analysis_metrics:
                        self.analysis_metrics.add(analysis_metrics)
                
                # Collect ML metrics
                if self.ml_predictor:
                    ml_metrics = self._collect_ml_metrics()
                    if ml_metrics:
                        self.ml_metrics.add(ml_metrics)
                
                # Clean up old custom timer data
                self._cleanup_custom_timers()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage (for root partition)
            disk = psutil.disk_usage('/')
            
            # Network statistics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix/Linux only)
            load_average = None
            try:
                load_average = psutil.getloadavg()
            except (AttributeError, OSError):
                pass  # Not available on Windows
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return None
    
    def _collect_analysis_metrics(self) -> Optional[AnalysisMetrics]:
        """Collect analysis pipeline metrics."""
        try:
            if not hasattr(self.analysis_pipeline, 'get_performance_metrics'):
                return None
            
            perf_metrics = self.analysis_pipeline.get_performance_metrics()
            pipeline_metrics = perf_metrics.get('pipeline_metrics', {})
            
            # Extract indicator performance
            indicator_performance = {}
            indicator_metrics = perf_metrics.get('indicator_metrics', {})
            
            for indicator_name, metrics in indicator_metrics.items():
                if 'metrics' in metrics and 'average_score' in metrics['metrics']:
                    indicator_performance[indicator_name] = metrics['metrics']['average_score']
            
            # Calculate data quality score (simplified)
            cache_metrics = perf_metrics.get('cache_metrics', {})
            cache_hit_rate = cache_metrics.get('hit_rate', 0.0)
            
            # Anomaly detection rate
            total_analyses = pipeline_metrics.get('total_analyses', 0)
            successful_analyses = pipeline_metrics.get('successful_analyses', 0)
            
            anomaly_detection_rate = 0.0
            if hasattr(self.analysis_pipeline, 'statistical_analyzer'):
                stats = self.analysis_pipeline.statistical_analyzer.get_summary_statistics()
                if 'overall_analysis' in stats and 'anomaly_rate' in stats['overall_analysis']:
                    anomaly_detection_rate = stats['overall_analysis']['anomaly_rate'] / 100.0
            
            return AnalysisMetrics(
                timestamp=datetime.now(),
                total_analyses=total_analyses,
                successful_analyses=successful_analyses,
                failed_analyses=pipeline_metrics.get('failed_analyses', 0),
                average_processing_time=pipeline_metrics.get('average_processing_time', 0.0),
                cache_hit_rate=cache_hit_rate,
                data_quality_score=0.8,  # Placeholder - would calculate from actual data quality
                anomaly_detection_rate=anomaly_detection_rate,
                indicator_performance=indicator_performance
            )
            
        except Exception as e:
            logger.error(f"Analysis metrics collection failed: {e}")
            return None
    
    def _collect_ml_metrics(self) -> Optional[MLMetrics]:
        """Collect ML model performance metrics."""
        try:
            if not hasattr(self.ml_predictor, 'get_prediction_stats'):
                return None
            
            ml_stats = self.ml_predictor.get_prediction_stats()
            
            # Alert count
            alert_count = 0
            if self.alert_manager:
                recent_alerts = self.alert_manager.get_recent_alerts(1)  # Last hour
                alert_count = len(recent_alerts)
            
            return MLMetrics(
                timestamp=datetime.now(),
                model_predictions=ml_stats.get('total_predictions', 0),
                prediction_latency=ml_stats.get('average_prediction_time', 0.0),
                model_accuracy=None,  # Would need validation data
                feature_quality=0.8,  # Placeholder - would calculate from feature quality metrics
                ensemble_agreement=0.75,  # Placeholder - would calculate from ensemble predictions
                alert_count=alert_count
            )
            
        except Exception as e:
            logger.error(f"ML metrics collection failed: {e}")
            return None
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a custom counter."""
        self.custom_counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a custom gauge value."""
        self.custom_gauges[name] = value
    
    def record_timer(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        self.custom_timers[name].append(duration)
        
        # Keep only recent measurements (last 1000)
        if len(self.custom_timers[name]) > 1000:
            self.custom_timers[name] = self.custom_timers[name][-1000:]
    
    def _cleanup_custom_timers(self) -> None:
        """Clean up old timer data."""
        max_timer_history = 1000
        
        for name, timings in self.custom_timers.items():
            if len(timings) > max_timer_history:
                self.custom_timers[name] = timings[-max_timer_history:]
    
    def get_system_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get system metrics summary."""
        recent_metrics = self.system_metrics.get_recent(hours)
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'current_disk_usage': recent_metrics[-1].disk_usage_percent,
            'current_process_count': recent_metrics[-1].process_count,
            'sample_count': len(recent_metrics)
        }
    
    def get_analysis_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get analysis pipeline summary."""
        recent_metrics = self.analysis_metrics.get_recent(hours)
        
        if not recent_metrics:
            return {}
        
        # Get latest metrics
        latest = recent_metrics[-1]
        
        # Calculate rates
        processing_times = [m.average_processing_time for m in recent_metrics if m.average_processing_time > 0]
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate >= 0]
        
        return {
            'total_analyses': latest.total_analyses,
            'success_rate': latest.successful_analyses / max(latest.total_analyses, 1),
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'avg_cache_hit_rate': sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
            'anomaly_detection_rate': latest.anomaly_detection_rate,
            'data_quality_score': latest.data_quality_score,
            'sample_count': len(recent_metrics)
        }
    
    def get_ml_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get ML performance summary."""
        recent_metrics = self.ml_metrics.get_recent(hours)
        
        if not recent_metrics:
            return {}
        
        # Get latest metrics
        latest = recent_metrics[-1]
        
        # Calculate averages
        latency_values = [m.prediction_latency for m in recent_metrics if m.prediction_latency > 0]
        feature_quality_values = [m.feature_quality for m in recent_metrics]
        
        return {
            'total_predictions': latest.model_predictions,
            'avg_prediction_latency': sum(latency_values) / len(latency_values) if latency_values else 0,
            'avg_feature_quality': sum(feature_quality_values) / len(feature_quality_values) if feature_quality_values else 0,
            'ensemble_agreement': latest.ensemble_agreement,
            'recent_alerts': latest.alert_count,
            'sample_count': len(recent_metrics)
        }
    
    def get_custom_metrics_summary(self) -> Dict[str, Any]:
        """Get custom metrics summary."""
        timer_summaries = {}
        
        for name, timings in self.custom_timers.items():
            if timings:
                timer_summaries[name] = {
                    'count': len(timings),
                    'avg': sum(timings) / len(timings),
                    'min': min(timings),
                    'max': max(timings)
                }
        
        return {
            'counters': dict(self.custom_counters),
            'gauges': dict(self.custom_gauges),
            'timers': timer_summaries
        }
    
    def get_comprehensive_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'collection_interval': self.collection_interval,
            'is_collecting': self.running,
            'system_metrics': self.get_system_summary(hours),
            'analysis_metrics': self.get_analysis_summary(hours),
            'ml_metrics': self.get_ml_summary(hours),
            'custom_metrics': self.get_custom_metrics_summary()
        }
    
    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        recent_metrics = self.system_metrics.get_recent(hours=1)
        return recent_metrics[-1] if recent_metrics else None
    
    def get_analysis_metrics(self) -> Optional[AnalysisMetrics]:
        """Get latest analysis metrics.""" 
        recent_metrics = self.analysis_metrics.get_recent(hours=1)
        return recent_metrics[-1] if recent_metrics else None
    
    def get_ml_metrics(self) -> Optional[MLMetrics]:
        """Get latest ML metrics."""
        recent_metrics = self.ml_metrics.get_recent(hours=1)
        return recent_metrics[-1] if recent_metrics else None
    
    def get_metrics_history(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get historical metrics data for trending analysis."""
        history = []
        
        # Get all metrics within time range
        system_history = []
        analysis_history = []
        ml_history = []
        
        for metric in self.system_metrics.get_all():
            if start_time <= metric.timestamp <= end_time:
                system_history.append(metric.to_dict())
        
        for metric in self.analysis_metrics.get_all():
            if start_time <= metric.timestamp <= end_time:
                analysis_history.append(metric.to_dict())
        
        for metric in self.ml_metrics.get_all():
            if start_time <= metric.timestamp <= end_time:
                ml_history.append(metric.to_dict())
        
        # Combine into time-series format
        all_timestamps = set()
        for metrics_list in [system_history, analysis_history, ml_history]:
            for metric in metrics_list:
                all_timestamps.add(metric['timestamp'])
        
        for timestamp in sorted(all_timestamps):
            entry = {'timestamp': timestamp}
            
            # Find matching metrics for this timestamp
            for metric in system_history:
                if metric['timestamp'] == timestamp:
                    entry.update({f'system_{k}': v for k, v in metric.items() if k != 'timestamp'})
                    break
            
            for metric in analysis_history:
                if metric['timestamp'] == timestamp:
                    entry.update({f'analysis_{k}': v for k, v in metric.items() if k != 'timestamp'})
                    break
                    
            for metric in ml_history:
                if metric['timestamp'] == timestamp:
                    entry.update({f'ml_{k}': v for k, v in metric.items() if k != 'timestamp'})
                    break
            
            history.append(entry)
        
        return history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        try:
            system_metrics = self.get_system_metrics()
            analysis_metrics = self.get_analysis_metrics()
            ml_metrics = self.get_ml_metrics()
            
            summary = {
                'load_average': getattr(system_metrics, 'load_average', None) if system_metrics else None,
                'uptime_hours': 24.0,  # Mock - would calculate from startup time
                'service_health': 'healthy',
                'last_updated': datetime.now().isoformat()
            }
            
            if system_metrics:
                summary.update({
                    'cpu_utilization': system_metrics.cpu_percent,
                    'memory_utilization': system_metrics.memory_percent,
                    'disk_utilization': system_metrics.disk_usage_percent
                })
            
            if analysis_metrics:
                summary.update({
                    'analysis_throughput': analysis_metrics.total_analyses,
                    'analysis_success_rate': analysis_metrics.successful_analyses / max(analysis_metrics.total_analyses, 1)
                })
            
            if ml_metrics:
                summary.update({
                    'prediction_throughput': ml_metrics.model_predictions,
                    'prediction_latency': ml_metrics.prediction_latency
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'error': 'Failed to get performance summary',
                'last_updated': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics and metrics collector stats."""
        return {
            'metrics_collected': {
                'system': len(self.system_metrics.get_all()),
                'analysis': len(self.analysis_metrics.get_all()),
                'ml': len(self.ml_metrics.get_all())
            },
            'collection_running': self.running,
            'collection_interval': self.collection_interval,
            'custom_counters': len(self.custom_counters),
            'custom_gauges': len(self.custom_gauges),
            'custom_timers': len(self.custom_timers)
        }
    
    def export_metrics_to_file(self, filepath: str, hours: int = 24) -> None:
        """Export metrics to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'hours_included': hours,
                'system_metrics': [m.to_dict() for m in self.system_metrics.get_recent(hours)],
                'analysis_metrics': [m.to_dict() for m in self.analysis_metrics.get_recent(hours)],
                'ml_metrics': [m.to_dict() for m in self.ml_metrics.get_recent(hours)],
                'custom_metrics': self.get_custom_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def clear_all_metrics(self) -> None:
        """Clear all collected metrics."""
        self.system_metrics.clear()
        self.analysis_metrics.clear()
        self.ml_metrics.clear()
        self.custom_counters.clear()
        self.custom_gauges.clear()
        self.custom_timers.clear()
        
        logger.info("All metrics cleared")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_collection()