"""
Middleware components for aNEOS API.

Provides request/response processing, rate limiting, logging,
CORS handling, and error processing middleware.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque

try:
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, middleware disabled")

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens_per_second = requests_per_minute / 60.0
        
        # Store token buckets per client IP
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tokens': burst_size,
            'last_refill': time.time()
        })
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed based on rate limits."""
        now = time.time()
        bucket = self.buckets[client_ip]
        
        # Refill tokens based on time elapsed
        time_elapsed = now - bucket['last_refill']
        tokens_to_add = time_elapsed * self.tokens_per_second
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
        
        # Check if we have tokens available
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def get_retry_after(self, client_ip: str) -> int:
        """Get retry-after time in seconds."""
        bucket = self.buckets[client_ip]
        tokens_needed = 1 - bucket['tokens']
        return max(1, int(tokens_needed / self.tokens_per_second))

class RequestLogger:
    """Request/response logging middleware."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(f"{__name__}.requests")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Track request metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
    
    def log_request(self, request: Request, start_time: float, status_code: int, response_size: int = 0):
        """Log request details."""
        end_time = time.time()
        duration = end_time - start_time
        
        # Update metrics
        self.request_count += 1
        self.response_times.append(duration)
        
        if status_code >= 400:
            self.error_count += 1
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log the request
        self.logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {status_code} - "
            f"Duration: {duration:.3f}s - "
            f"Size: {response_size}B - "
            f"IP: {client_ip} - "
            f"UA: {user_agent[:50]}..."
        )
        
        # Log slow requests as warnings
        if duration > 5.0:
            self.logger.warning(f"Slow request: {request.method} {request.url.path} took {duration:.3f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get request metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
            'average_response_time': avg_response_time,
            'recent_response_times': list(self.response_times)[-10:]  # Last 10 response times
        }

class ErrorHandler:
    """Centralized error handling and reporting."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.errors")
        self.error_counts = defaultdict(int)
        self.last_errors = deque(maxlen=100)  # Keep last 100 errors
    
    def handle_error(self, request: Request, error: Exception) -> JSONResponse:
        """Handle and log errors, return appropriate response."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error tracking
        self.error_counts[error_type] += 1
        self.last_errors.append({
            'type': error_type,
            'message': error_message,
            'path': str(request.url.path),
            'method': request.method,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log the error
        self.logger.error(
            f"Error in {request.method} {request.url.path}: "
            f"{error_type}: {error_message}",
            exc_info=True
        )
        
        # Return appropriate error response
        if isinstance(error, HTTPException):
            return JSONResponse(
                status_code=error.status_code,
                content={
                    'error': error.detail,
                    'status_code': error.status_code,
                    'timestamp': datetime.now().isoformat(),
                    'path': str(request.url.path)
                }
            )
        else:
            # Generic server error
            return JSONResponse(
                status_code=500,
                content={
                    'error': 'Internal server error',
                    'status_code': 500,
                    'timestamp': datetime.now().isoformat(),
                    'path': str(request.url.path)
                }
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': list(self.last_errors)[-10:]  # Last 10 errors
        }

# Middleware classes for FastAPI
if HAS_FASTAPI:
    
    class RateLimitingMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for rate limiting."""
        
        def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 10):
            super().__init__(app)
            self.rate_limiter = RateLimiter(requests_per_minute, burst_size)
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            client_ip = request.client.host if request.client else "unknown"
            
            if not self.rate_limiter.is_allowed(client_ip):
                retry_after = self.rate_limiter.get_retry_after(client_ip)
                
                return JSONResponse(
                    status_code=429,
                    content={
                        'error': 'Rate limit exceeded',
                        'retry_after': retry_after,
                        'timestamp': datetime.now().isoformat()
                    },
                    headers={'Retry-After': str(retry_after)}
                )
            
            response = await call_next(request)
            return response
    
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for request logging."""
        
        def __init__(self, app, log_level: str = "INFO"):
            super().__init__(app)
            self.request_logger = RequestLogger(log_level)
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Get response size if available
                response_size = 0
                if hasattr(response, 'body'):
                    response_size = len(response.body) if response.body else 0
                
                # Log the request
                self.request_logger.log_request(
                    request, start_time, response.status_code, response_size
                )
                
                return response
                
            except Exception as e:
                # Log failed requests
                self.request_logger.log_request(request, start_time, 500)
                raise e
    
    class ErrorHandlingMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for error handling."""
        
        def __init__(self, app):
            super().__init__(app)
            self.error_handler = ErrorHandler()
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                return self.error_handler.handle_error(request, e)
    
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for security headers."""
        
        def __init__(self, app):
            super().__init__(app)
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            response = await call_next(request)
            
            # Add security headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            response.headers['X-API-Version'] = '2.0.0'
            
            return response
    
    class MetricsMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for collecting request metrics."""
        
        def __init__(self, app):
            super().__init__(app)
            self.metrics = {
                'requests_total': 0,
                'requests_by_method': defaultdict(int),
                'requests_by_status': defaultdict(int),
                'response_times': deque(maxlen=1000),
                'active_requests': 0
            }
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            start_time = time.time()
            self.metrics['active_requests'] += 1
            
            try:
                response = await call_next(request)
                
                # Update metrics
                duration = time.time() - start_time
                self.metrics['requests_total'] += 1
                self.metrics['requests_by_method'][request.method] += 1
                self.metrics['requests_by_status'][response.status_code] += 1
                self.metrics['response_times'].append(duration)
                
                # Add metrics headers
                response.headers['X-Response-Time'] = f"{duration:.3f}"
                response.headers['X-Request-ID'] = str(id(request))
                
                return response
                
            finally:
                self.metrics['active_requests'] -= 1
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get current metrics."""
            avg_response_time = (
                sum(self.metrics['response_times']) / len(self.metrics['response_times'])
                if self.metrics['response_times'] else 0
            )
            
            return {
                'requests_total': self.metrics['requests_total'],
                'requests_by_method': dict(self.metrics['requests_by_method']),
                'requests_by_status': dict(self.metrics['requests_by_status']),
                'average_response_time': avg_response_time,
                'active_requests': self.metrics['active_requests']
            }

# Global middleware instances
_rate_limiter: Optional[RateLimiter] = None
_request_logger: Optional[RequestLogger] = None
_error_handler: Optional[ErrorHandler] = None
_metrics_middleware: Optional[MetricsMiddleware] = None

def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the FastAPI application."""
    global _rate_limiter, _request_logger, _error_handler, _metrics_middleware
    
    if not HAS_FASTAPI:
        logger.warning("FastAPI not available, skipping middleware setup")
        return
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Metrics middleware
    app.add_middleware(MetricsMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware, log_level="INFO")
    
    # Rate limiting middleware
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=100, burst_size=20)
    
    # Error handling middleware (add last so it catches everything)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Trusted host middleware (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    logger.info("All middleware configured successfully")

def get_middleware_metrics() -> Dict[str, Any]:
    """Get metrics from all middleware components."""
    metrics = {}
    
    if _request_logger:
        metrics['request_logging'] = _request_logger.get_metrics()
    
    if _error_handler:
        metrics['error_handling'] = _error_handler.get_error_summary()
    
    if _metrics_middleware:
        metrics['request_metrics'] = _metrics_middleware.get_metrics()
    
    return metrics

# Utility functions
def create_rate_limiter(requests_per_minute: int = 60, burst_size: int = 10) -> RateLimiter:
    """Create a rate limiter instance."""
    return RateLimiter(requests_per_minute, burst_size)

def create_request_logger(log_level: str = "INFO") -> RequestLogger:
    """Create a request logger instance."""
    return RequestLogger(log_level)

def create_error_handler() -> ErrorHandler:
    """Create an error handler instance."""
    return ErrorHandler()

# Health check for middleware
def middleware_health_check() -> Dict[str, Any]:
    """Perform health check on middleware components."""
    return {
        'status': 'healthy',
        'components': {
            'rate_limiter': _rate_limiter is not None,
            'request_logger': _request_logger is not None,
            'error_handler': _error_handler is not None,
            'metrics_middleware': _metrics_middleware is not None
        },
        'timestamp': datetime.now().isoformat()
    }