"""
Validation Pipeline Integration for Real-Time Dashboard

Integrates the real-time dashboard with the MultiStageValidator and all
Phase 1-3 validation modules for comprehensive monitoring and reporting.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

try:
    from aneos_core.validation.multi_stage_validator import MultiStageValidator, ValidationResult
    from aneos_core.validation.delta_bic_analysis import DeltaBICAnalyzer
    from aneos_core.validation.spectral_outlier_analysis import SpectralOutlierAnalyzer
    from aneos_core.validation.radar_polarization_analysis import RadarPolarizationAnalyzer
    from aneos_core.validation.thermal_ir_analysis import ThermalIRAnalyzer
    from aneos_core.validation.gaia_astrometric_calibration import GaiaAstrometricCalibrator
    HAS_VALIDATION_MODULES = True
except ImportError:
    HAS_VALIDATION_MODULES = False
    logging.warning("Validation modules not available")

from ..monitoring.validation_metrics import ValidationMetricsCollector
from ..websockets.validation_websocket import ValidationWebSocketManager

logger = logging.getLogger(__name__)

class ValidationPipelineIntegration:
    """
    Integration layer between the validation pipeline and real-time dashboard.
    
    Monitors validation processes, collects metrics, and provides real-time
    updates to dashboard clients through WebSocket connections.
    """
    
    def __init__(self, 
                 metrics_collector: ValidationMetricsCollector,
                 websocket_manager: ValidationWebSocketManager,
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize validation pipeline integration.
        
        Args:
            metrics_collector: Metrics collector for validation data
            websocket_manager: WebSocket manager for real-time updates
            validation_config: Optional configuration for validation modules
        """
        self.metrics_collector = metrics_collector
        self.websocket_manager = websocket_manager
        self.validation_config = validation_config or {}
        
        # Initialize validation modules if available
        self.validator = None
        self.module_status = {
            'delta_bic': False,
            'spectral_analysis': False,
            'radar_analysis': False,
            'thermal_ir': False,
            'gaia_analysis': False
        }
        
        if HAS_VALIDATION_MODULES:
            self._initialize_validation_modules()
        
        self.logger = logging.getLogger(__name__)
        self.active_validations = {}
        
    def _initialize_validation_modules(self):
        """Initialize all validation modules with dashboard integration."""
        try:
            # Initialize MultiStageValidator with enhanced configuration
            enhanced_config = self.validation_config.copy()
            enhanced_config.update({
                'enable_dashboard_integration': True,
                'real_time_metrics': True,
                'alert_system': True
            })
            
            self.validator = MultiStageValidator(enhanced_config)
            
            # Check module availability
            if hasattr(self.validator, 'delta_bic_analyzer') and self.validator.delta_bic_analyzer:
                self.module_status['delta_bic'] = True
                
            if hasattr(self.validator, 'spectral_analyzer') and self.validator.spectral_analyzer:
                self.module_status['spectral_analysis'] = True
                
            if hasattr(self.validator, 'radar_analyzer') and self.validator.radar_analyzer:
                self.module_status['radar_analysis'] = True
                
            if hasattr(self.validator, 'thermal_ir_analyzer') and self.validator.thermal_ir_analyzer:
                self.module_status['thermal_ir'] = True
                
            if hasattr(self.validator, 'gaia_calibrator') and self.validator.gaia_calibrator:
                self.module_status['gaia_analysis'] = True
            
            self.logger.info(f"Validation modules initialized: {self.module_status}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize validation modules: {e}")
    
    async def validate_with_dashboard_integration(self, 
                                                neo_data: Any, 
                                                analysis_result: Any,
                                                session_id: Optional[str] = None) -> ValidationResult:
        """
        Run validation with integrated dashboard monitoring and real-time updates.
        
        Args:
            neo_data: NEO data object
            analysis_result: Analysis result from aNEOS pipeline
            session_id: Optional session identifier for tracking
            
        Returns:
            Enhanced validation result with dashboard integration
        """
        if not self.validator:
            raise RuntimeError("Validation modules not available")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"validation_{int(datetime.now().timestamp())}_{len(self.active_validations)}"
        
        start_time = datetime.now()
        object_designation = getattr(neo_data, 'designation', 'unknown')
        
        try:
            self.logger.info(f"Starting validation for {object_designation} (session: {session_id})")
            
            # Track active validation
            self.active_validations[session_id] = {
                'start_time': start_time,
                'object_designation': object_designation,
                'current_stage': 0
            }
            
            # Broadcast validation start
            await self.websocket_manager.broadcast_message('validation_started', {
                'session_id': session_id,
                'object_designation': object_designation,
                'timestamp': start_time.isoformat(),
                'module_status': self.module_status
            })
            
            # Run validation with stage-by-stage monitoring
            validation_result = await self._run_monitored_validation(
                neo_data, analysis_result, session_id
            )
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Add dashboard-specific metadata
            validation_result.object_designation = object_designation
            validation_result.session_id = session_id
            validation_result.dashboard_metadata = {
                'processing_start': start_time.isoformat(),
                'processing_end': end_time.isoformat(),
                'module_status': self.module_status,
                'real_time_updates': True
            }
            
            # Record metrics
            metrics_session_id = self.metrics_collector.record_validation_session(
                validation_result, processing_time_ms
            )
            
            # Broadcast validation result
            await self.websocket_manager.broadcast_validation_result(
                validation_result, processing_time_ms
            )
            
            # Check for artificial object alerts
            await self._check_and_broadcast_alerts(validation_result, session_id)
            
            # Remove from active validations
            self.active_validations.pop(session_id, None)
            
            self.logger.info(
                f"Validation completed for {object_designation}: "
                f"{validation_result.recommendation} (confidence: {validation_result.overall_confidence if validation_result.overall_confidence is not None else 'unavailable'})"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed for {object_designation}: {e}")
            
            # Broadcast error
            await self.websocket_manager.broadcast_message('validation_error', {
                'session_id': session_id,
                'object_designation': object_designation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Remove from active validations
            self.active_validations.pop(session_id, None)
            
            raise
    
    async def _run_monitored_validation(self, 
                                      neo_data: Any, 
                                      analysis_result: Any,
                                      session_id: str) -> ValidationResult:
        """Run validation with real-time stage monitoring."""
        
        enhanced = await self.validator.validate_analysis_result(neo_data, analysis_result)
        result = enhanced.validation_result
        # Publish actual outcomes after the authoritative validator returns.
        # Do not run a second orchestration or fabricate stage-start timing.
        for stage in result.stage_results:
            self.active_validations[session_id]['current_stage'] = stage.stage_number
            await self.websocket_manager.broadcast_message('stage_completed', {
                'session_id': session_id, 'stage_number': stage.stage_number,
                'stage_name': stage.stage_name, 'passed': stage.passed,
                'validation_status': stage.validation_status, 'score': stage.score,
                'confidence': stage.confidence, 'processing_time_ms': stage.processing_time_ms,
                'timestamp': datetime.now().isoformat(), 'timing_semantics': 'outcome_publication_not_stage_execution'})
        return result

    async def _check_and_broadcast_alerts(self, validation_result, session_id):
        """No installed origin model: thresholds cannot create artificial alerts."""
        return {'status': 'unavailable_no_origin_model'}

    def get_active_validations(self) -> Dict[str, Any]:
        """Get currently active validation sessions."""
        return {
            'active_count': len(self.active_validations),
            'sessions': list(self.active_validations.values()),
            'module_status': self.module_status,
            'validator_available': self.validator is not None
        }
    
    def get_module_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for each validation module."""
        try:
            dashboard_data = self.metrics_collector.get_dashboard_data()
            module_performance = dashboard_data.get('validation_pipeline', {}).get('module_availability', {})
            
            return {
                'module_availability': module_performance,
                'module_status': self.module_status,
                'performance_summary': {
                    'delta_bic_analyzer': {
                        'available': self.module_status['delta_bic'],
                        'performance': module_performance.get('delta_bic', 0),
                        'description': 'Orbital dynamics analysis using Bayesian Information Criterion'
                    },
                    'spectral_analyzer': {
                        'available': self.module_status['spectral_analysis'],
                        'performance': module_performance.get('spectral_analysis', 0),
                        'description': 'Spectral classification and artificial material detection'
                    },
                    'radar_analyzer': {
                        'available': self.module_status['radar_analysis'],
                        'performance': module_performance.get('radar_analysis', 0),
                        'description': 'Radar polarization analysis for surface characterization'
                    },
                    'thermal_ir_analyzer': {
                        'available': self.module_status['thermal_ir'],
                        'performance': module_performance.get('thermal_ir', 0),
                        'description': 'Thermal-IR analysis and Yarkovsky effect detection'
                    },
                    'gaia_calibrator': {
                        'available': self.module_status['gaia_analysis'],
                        'performance': module_performance.get('gaia_analysis', 0),
                        'description': 'Gaia astrometric precision calibration and validation'
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get module performance metrics: {e}")
            return {'error': str(e)}
    
    async def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'validation_system': {
                    'validator_available': self.validator is not None,
                    'modules_available': sum(self.module_status.values()),
                    'total_modules': len(self.module_status),
                    'module_details': self.module_status
                },
                'dashboard_services': {
                    'metrics_collector': self.metrics_collector is not None,
                    'websocket_manager': self.websocket_manager is not None,
                    'active_connections': self.websocket_manager.get_connection_stats()['active_connections'] if self.websocket_manager else 0
                },
                'active_validations': {
                    'count': len(self.active_validations),
                    'details': list(self.active_validations.values())
                }
            }
            
            # Determine overall health status
            if not self.validator:
                health_status['overall_status'] = 'degraded'
                health_status['issues'] = ['Validation modules not available']
            elif sum(self.module_status.values()) < len(self.module_status) * 0.8:
                health_status['overall_status'] = 'warning'
                health_status['issues'] = ['Some validation modules unavailable']
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unhealthy',
                'error': str(e)
            }
    
    async def update_system_metrics(self):
        """Update system metrics for dashboard monitoring."""
        try:
            import psutil
            
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Count active sessions and alerts
            active_sessions = len(self.active_validations)
            dashboard_data = self.metrics_collector.get_dashboard_data()
            alert_count = len([
                alert for alert in dashboard_data.get('alerts_and_notifications', {}).get('recent_alerts', [])
                if not alert.get('resolved', True)
            ])
            expert_review_queue = dashboard_data.get('system_overview', {}).get('expert_review_queue', 0)
            
            # Record system health
            self.metrics_collector.record_system_health(
                cpu_percent, memory_percent, active_sessions, alert_count, expert_review_queue
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    def start_monitoring_loop(self, interval_seconds: int = 30):
        """Start background monitoring loop for system metrics."""
        async def monitoring_loop():
            while True:
                try:
                    await self.update_system_metrics()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(interval_seconds)
        
        # Start the monitoring loop as a background task
        asyncio.create_task(monitoring_loop())
        self.logger.info(f"Started system monitoring loop (interval: {interval_seconds}s)")

# Global integration instance (will be initialized by the main application)
validation_integration: Optional[ValidationPipelineIntegration] = None

def initialize_integration(metrics_collector: ValidationMetricsCollector,
                          websocket_manager: ValidationWebSocketManager,
                          validation_config: Optional[Dict[str, Any]] = None) -> ValidationPipelineIntegration:
    """Initialize the validation pipeline integration."""
    global validation_integration
    validation_integration = ValidationPipelineIntegration(
        metrics_collector, websocket_manager, validation_config
    )
    return validation_integration

def get_integration() -> Optional[ValidationPipelineIntegration]:
    """Get the global integration instance."""
    return validation_integration