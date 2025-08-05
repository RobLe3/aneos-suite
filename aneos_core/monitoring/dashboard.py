"""
Monitoring dashboard for aNEOS system.

This module provides a simple text-based dashboard for monitoring
system health, performance metrics, and alert status.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import threading
import os

from .metrics import MetricsCollector
from .alerts import AlertManager, AlertLevel

class MonitoringDashboard:
    """Text-based monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager,
                 refresh_interval: int = 30):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.refresh_interval = refresh_interval
        
        self.running = False
        self.display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start_dashboard(self) -> None:
        """Start the dashboard display."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name="MonitoringDashboard",
            daemon=True
        )
        self.display_thread.start()
    
    def stop_dashboard(self) -> None:
        """Stop the dashboard display."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
    
    def _display_loop(self) -> None:
        """Main display loop."""
        while not self._stop_event.wait(self.refresh_interval):
            try:
                self._clear_screen()
                self._display_dashboard()
            except Exception as e:
                print(f"Dashboard error: {e}")
    
    def _clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _display_dashboard(self) -> None:
        """Display the complete dashboard."""
        print("=" * 80)
        print(" " * 25 + "aNEOS MONITORING DASHBOARD")
        print("=" * 80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System health overview
        self._display_system_health()
        print()
        
        # Alert status
        self._display_alert_status()
        print()
        
        # Performance metrics
        self._display_performance_metrics()
        print()
        
        # Analysis pipeline status
        self._display_analysis_status()
        print()
        
        # ML model status
        self._display_ml_status()
        print()
        
        print("=" * 80)
        print(f"Refresh Interval: {self.refresh_interval}s | Press Ctrl+C to stop")
    
    def _display_system_health(self) -> None:
        """Display system health overview."""
        print("🖥️  SYSTEM HEALTH")
        print("-" * 40)
        
        try:
            system_summary = self.metrics_collector.get_system_summary(1)
            
            if system_summary:
                cpu_percent = system_summary.get('avg_cpu_percent', 0)
                memory_percent = system_summary.get('avg_memory_percent', 0)
                disk_usage = system_summary.get('current_disk_usage', 0)
                
                # Color coding for health status
                cpu_status = self._get_health_indicator(cpu_percent, 80, 90)
                memory_status = self._get_health_indicator(memory_percent, 80, 90)
                disk_status = self._get_health_indicator(disk_usage, 80, 90)
                
                print(f"CPU Usage:    {cpu_percent:5.1f}% {cpu_status}")
                print(f"Memory Usage: {memory_percent:5.1f}% {memory_status}")
                print(f"Disk Usage:   {disk_usage:5.1f}% {disk_status}")
                print(f"Processes:    {system_summary.get('current_process_count', 0)}")
            else:
                print("❌ System metrics not available")
                
        except Exception as e:
            print(f"❌ Error retrieving system health: {e}")
    
    def _display_alert_status(self) -> None:
        """Display alert status."""
        print("🚨 ALERT STATUS")
        print("-" * 40)
        
        try:
            alert_stats = self.alert_manager.get_alert_statistics()
            recent_alerts = self.alert_manager.get_recent_alerts(1)  # Last hour
            
            # Alert counts by level
            critical_alerts = len([a for a in recent_alerts if a.alert_level == AlertLevel.CRITICAL])
            high_alerts = len([a for a in recent_alerts if a.alert_level == AlertLevel.HIGH])
            medium_alerts = len([a for a in recent_alerts if a.alert_level == AlertLevel.MEDIUM])
            low_alerts = len([a for a in recent_alerts if a.alert_level == AlertLevel.LOW])
            
            total_active = alert_stats.get('active_alerts', 0)
            
            print(f"Active Alerts: {total_active}")
            print(f"Recent (1h):   Critical: {critical_alerts} | High: {high_alerts} | Medium: {medium_alerts} | Low: {low_alerts}")
            
            # Show most recent critical/high alerts
            critical_high_alerts = [a for a in recent_alerts 
                                  if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]][:3]
            
            if critical_high_alerts:
                print("\nRecent High-Priority Alerts:")
                for alert in critical_high_alerts:
                    timestamp = alert.timestamp.strftime('%H:%M:%S')
                    level_icon = "🔴" if alert.alert_level == AlertLevel.CRITICAL else "🟠"
                    print(f"  {level_icon} {timestamp} - {alert.title}")
            else:
                print("✅ No recent high-priority alerts")
                
        except Exception as e:
            print(f"❌ Error retrieving alert status: {e}")
    
    def _display_performance_metrics(self) -> None:
        """Display performance metrics."""
        print("📊 PERFORMANCE METRICS")
        print("-" * 40)
        
        try:
            # Get comprehensive summary
            summary = self.metrics_collector.get_comprehensive_summary(1)
            
            # Collection status
            collection_status = "🟢 Active" if self.metrics_collector.running else "🔴 Stopped"
            print(f"Collection:   {collection_status}")
            print(f"Interval:     {self.metrics_collector.collection_interval}s")
            
            # Custom metrics
            custom_metrics = summary.get('custom_metrics', {})
            counters = custom_metrics.get('counters', {})
            gauges = custom_metrics.get('gauges', {})
            
            if counters:
                print("\nCounters:")
                for name, value in list(counters.items())[:5]:  # Show top 5
                    print(f"  {name}: {value}")
            
            if gauges:
                print("\nGauges:")
                for name, value in list(gauges.items())[:5]:  # Show top 5
                    print(f"  {name}: {value:.3f}")
                    
        except Exception as e:
            print(f"❌ Error retrieving performance metrics: {e}")
    
    def _display_analysis_status(self) -> None:
        """Display analysis pipeline status."""
        print("🔬 ANALYSIS PIPELINE")
        print("-" * 40)
        
        try:
            analysis_summary = self.metrics_collector.get_analysis_summary(1)
            
            if analysis_summary:
                total_analyses = analysis_summary.get('total_analyses', 0)
                success_rate = analysis_summary.get('success_rate', 0) * 100
                avg_processing_time = analysis_summary.get('avg_processing_time', 0)
                cache_hit_rate = analysis_summary.get('avg_cache_hit_rate', 0) * 100
                anomaly_rate = analysis_summary.get('anomaly_detection_rate', 0) * 100
                
                print(f"Total Analyses: {total_analyses}")
                print(f"Success Rate:   {success_rate:5.1f}%")
                print(f"Avg Process:    {avg_processing_time:5.2f}s")
                print(f"Cache Hit Rate: {cache_hit_rate:5.1f}%")
                print(f"Anomaly Rate:   {anomaly_rate:5.1f}%")
                
                # Health indicator
                health_score = (success_rate + cache_hit_rate) / 2
                health_indicator = self._get_health_indicator(health_score, 80, 90)
                print(f"Pipeline Health: {health_indicator}")
            else:
                print("❌ Analysis metrics not available")
                
        except Exception as e:
            print(f"❌ Error retrieving analysis status: {e}")
    
    def _display_ml_status(self) -> None:
        """Display ML model status."""
        print("🤖 MACHINE LEARNING")
        print("-" * 40)
        
        try:
            ml_summary = self.metrics_collector.get_ml_summary(1)
            
            if ml_summary:
                total_predictions = ml_summary.get('total_predictions', 0)
                avg_latency = ml_summary.get('avg_prediction_latency', 0) * 1000  # Convert to ms
                feature_quality = ml_summary.get('avg_feature_quality', 0) * 100
                ensemble_agreement = ml_summary.get('ensemble_agreement', 0) * 100
                recent_alerts = ml_summary.get('recent_alerts', 0)
                
                print(f"Predictions:    {total_predictions}")
                print(f"Avg Latency:    {avg_latency:6.1f}ms")
                print(f"Feature Quality:{feature_quality:6.1f}%")
                print(f"Ensemble Agree: {ensemble_agreement:6.1f}%")
                print(f"ML Alerts (1h): {recent_alerts}")
                
                # Overall ML health
                ml_health = (feature_quality + ensemble_agreement) / 2
                health_indicator = self._get_health_indicator(ml_health, 70, 85)
                print(f"ML Health:      {health_indicator}")
            else:
                print("❌ ML metrics not available")
                
        except Exception as e:
            print(f"❌ Error retrieving ML status: {e}")
    
    def _get_health_indicator(self, value: float, warning_threshold: float, 
                            critical_threshold: float) -> str:
        """Get health indicator emoji based on value and thresholds."""
        if value >= critical_threshold:
            return "🔴 CRITICAL"
        elif value >= warning_threshold:
            return "🟡 WARNING"
        else:
            return "🟢 HEALTHY"
    
    def generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        report_lines = []
        
        report_lines.append("aNEOS SYSTEM STATUS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # System summary
        try:
            system_summary = self.metrics_collector.get_system_summary(24)  # 24 hour summary
            
            report_lines.append("SYSTEM PERFORMANCE (24h)")
            report_lines.append("-" * 30)
            report_lines.append(f"Average CPU Usage:    {system_summary.get('avg_cpu_percent', 0):5.1f}%")
            report_lines.append(f"Peak CPU Usage:       {system_summary.get('max_cpu_percent', 0):5.1f}%")
            report_lines.append(f"Average Memory Usage: {system_summary.get('avg_memory_percent', 0):5.1f}%")
            report_lines.append(f"Peak Memory Usage:    {system_summary.get('max_memory_percent', 0):5.1f}%")
            report_lines.append(f"Current Disk Usage:   {system_summary.get('current_disk_usage', 0):5.1f}%")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"System metrics error: {e}")
            report_lines.append("")
        
        # Analysis summary
        try:
            analysis_summary = self.metrics_collector.get_analysis_summary(24)
            
            report_lines.append("ANALYSIS PIPELINE (24h)")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Analyses:       {analysis_summary.get('total_analyses', 0)}")
            report_lines.append(f"Success Rate:         {analysis_summary.get('success_rate', 0)*100:5.1f}%")
            report_lines.append(f"Average Process Time: {analysis_summary.get('avg_processing_time', 0):5.2f}s")
            report_lines.append(f"Cache Hit Rate:       {analysis_summary.get('avg_cache_hit_rate', 0)*100:5.1f}%")
            report_lines.append(f"Anomaly Detection:    {analysis_summary.get('anomaly_detection_rate', 0)*100:5.1f}%")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"Analysis metrics error: {e}")
            report_lines.append("")
        
        # Alert summary
        try:
            alert_stats = self.alert_manager.get_alert_statistics()
            recent_alerts = self.alert_manager.get_recent_alerts(24)
            
            report_lines.append("ALERT SUMMARY (24h)")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Active Alerts:  {alert_stats.get('active_alerts', 0)}")
            report_lines.append(f"Recent Alerts:        {len(recent_alerts)}")
            
            # Alert breakdown
            for level in AlertLevel:
                count = len([a for a in recent_alerts if a.alert_level == level])
                report_lines.append(f"  {level.value.capitalize():>10}: {count}")
            
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"Alert metrics error: {e}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, filepath: str) -> None:
        """Save a comprehensive report to file."""
        try:
            report = self.generate_text_report()
            
            with open(filepath, 'w') as f:
                f.write(report)
            
            print(f"Report saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    def run_interactive(self) -> None:
        """Run interactive dashboard mode."""
        print("Starting aNEOS Monitoring Dashboard...")
        print("Press Ctrl+C to stop")
        
        try:
            self.start_dashboard()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            self.stop_dashboard()
            print("Dashboard stopped.")
        except Exception as e:
            print(f"Dashboard error: {e}")
            self.stop_dashboard()