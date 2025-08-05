"""
Enhanced logging utilities for aNEOS Core.

Provides structured logging with file rotation, performance metrics,
and integration with the existing logging infrastructure.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    structured_format: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for aNEOS Core.
    
    Args:
        log_file: Path to log file (None for console only)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        structured_format: Use structured logging format
        
    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger("aneos_core")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if structured_format:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add performance logging
    perf_logger = _setup_performance_logger(log_file)
    
    logger.info("Logging initialized for aNEOS Core")
    return logger


def _setup_performance_logger(base_log_file: Optional[str] = None) -> logging.Logger:
    """Set up performance metrics logging."""
    perf_logger = logging.getLogger("aneos_core.performance")
    perf_logger.setLevel(logging.INFO)
    
    if base_log_file:
        # Create performance log file path
        log_path = Path(base_log_file)
        perf_log_file = log_path.parent / f"{log_path.stem}_performance.log"
        
        perf_handler = logging.handlers.RotatingFileHandler(
            str(perf_log_file),
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        
        perf_formatter = logging.Formatter(
            '%(asctime)s | PERF | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
    
    return perf_logger


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("aneos_core.performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Started: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            status = "FAILED" if exc_type else "SUCCESS"
            self.logger.info(f"Completed: {self.operation_name} | Duration: {duration:.3f}s | Status: {status}")


def log_performance(operation_name: str):
    """Decorator for automatic performance logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(f"{func.__module__}.{func.__name__}({operation_name})"):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class StructuredLogger:
    """
    Structured logging helper for consistent log formatting.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_api_request(self, source: str, endpoint: str, params: Dict[str, Any], duration: float, success: bool):
        """Log API request details."""
        self.logger.info(
            f"API_REQUEST | Source: {source} | Endpoint: {endpoint} | "
            f"Duration: {duration:.3f}s | Success: {success} | Params: {params}"
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, size_bytes: Optional[int] = None):
        """Log cache operation details."""
        size_str = f" | Size: {size_bytes} bytes" if size_bytes else ""
        self.logger.info(
            f"CACHE_{operation.upper()} | Key: {key} | Hit: {hit}{size_str}"
        )
    
    def log_data_quality(self, designation: str, completeness: float, sources: list, anomaly_score: Optional[float] = None):
        """Log data quality metrics."""
        anomaly_str = f" | Anomaly: {anomaly_score:.3f}" if anomaly_score else ""
        self.logger.info(
            f"DATA_QUALITY | NEO: {designation} | Completeness: {completeness:.2f} | "
            f"Sources: {','.join(sources)}{anomaly_str}"
        )
    
    def log_analysis_result(self, designation: str, category: str, score: float, indicators: Dict[str, Any]):
        """Log analysis results."""
        self.logger.info(
            f"ANALYSIS | NEO: {designation} | Category: {category} | Score: {score:.3f} | "
            f"Indicators: {indicators}"
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        self.logger.error(
            f"ERROR | Type: {type(error).__name__} | Message: {str(error)} | "
            f"Context: {context}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent naming.
    
    Args:
        name: Logger name (will be prefixed with 'aneos_core.')
        
    Returns:
        Logger instance
    """
    if not name.startswith("aneos_core."):
        name = f"aneos_core.{name}"
    
    return logging.getLogger(name)


def configure_external_loggers():
    """Configure external library loggers to reduce noise."""
    # Reduce urllib3 logging
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    
    # Reduce requests logging
    logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
    
    # Reduce astroquery logging
    logging.getLogger("astroquery").setLevel(logging.WARNING)
    
    # Reduce matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)