"""
Utility patterns for aNEOS - Circuit Breaker, Retry Logic, etc.

This module provides common patterns used throughout the aNEOS system
for robust error handling and resilience.
"""

import time
import asyncio
import random
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening
    timeout_seconds: int = 60        # Time before trying half-open
    success_threshold: int = 3       # Successes needed to close from half-open
    monitoring_window: int = 300     # Window for tracking failures (seconds)

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_timestamps: List[float] = []
        
        logger.info(f"Circuit breaker '{name}' initialized: {self.config}")
    
    def _is_failure_threshold_exceeded(self) -> bool:
        """Check if we've exceeded the failure threshold in the monitoring window."""
        current_time = time.time()
        window_start = current_time - self.config.monitoring_window
        
        # Remove old failures outside the window
        self.failure_timestamps = [
            ts for ts in self.failure_timestamps if ts > window_start
        ]
        
        return len(self.failure_timestamps) >= self.config.failure_threshold
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if current_time - self.last_failure_time > self.config.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                return True
            return False
            
        else:  # HALF_OPEN
            return True
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.failure_timestamps.clear()
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure tracking on success
            if self.failure_timestamps:
                self.failure_timestamps.clear()
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        current_time = time.time()
        self.failure_count += 1
        self.failure_timestamps.append(current_time)
        self.last_failure_time = current_time
        
        if self.state == CircuitState.CLOSED:
            if self._is_failure_threshold_exceeded():
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN after {len(self.failure_timestamps)} failures")
                
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' transitioning back to OPEN")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        current_time = time.time()
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': len(self.failure_timestamps),
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_since_last_failure': current_time - self.last_failure_time if self.last_failure_time else None,
            'can_execute': self.can_execute(),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'success_threshold': self.config.success_threshold,
                'monitoring_window': self.config.monitoring_window
            }
        }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, message: str = "Circuit breaker is open"):
        self.circuit_name = circuit_name
        super().__init__(f"{message}: {circuit_name}")

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker pattern to functions."""
    breaker = CircuitBreaker(name, config)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitBreakerOpenException(name)
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        # Attach breaker to function for status checking
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator

def async_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Async decorator to apply circuit breaker pattern to async functions."""
    breaker = CircuitBreaker(name, config)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitBreakerOpenException(name)
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        # Attach breaker to function for status checking
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)

class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt."""
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        # Add random jitter (±25%)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)

def retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions."""
    retry_config = config or RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts:
                        logger.error(f"Function {func.__name__} failed after {attempt} attempts")
                        raise RetryExhaustedException(attempt, e)
                    
                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt}/{retry_config.max_attempts}), retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise RetryExhaustedException(retry_config.max_attempts, last_exception)
        
        return wrapper
    return decorator

def async_retry(config: Optional[RetryConfig] = None):
    """Async decorator to add retry logic to async functions."""
    retry_config = config or RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts:
                        logger.error(f"Async function {func.__name__} failed after {attempt} attempts")
                        raise RetryExhaustedException(attempt, e)
                    
                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(f"Async function {func.__name__} failed (attempt {attempt}/{retry_config.max_attempts}), retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            raise RetryExhaustedException(retry_config.max_attempts, last_exception)
        
        return wrapper
    return decorator

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """
        Initialize rate limiter.
        
        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self._lock = asyncio.Lock() if asyncio.iscoroutinefunction(self) else None
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def can_proceed(self, tokens_needed: int = 1) -> bool:
        """Check if we can proceed with the operation."""
        self._refill_tokens()
        
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        
        return False
    
    async def acquire(self, tokens_needed: int = 1) -> None:
        """Acquire tokens, waiting if necessary (async version)."""
        while not self.can_proceed(tokens_needed):
            wait_time = (tokens_needed - self.tokens) / self.refill_rate
            await asyncio.sleep(min(wait_time, 1.0))  # Max 1 second wait
    
    def acquire_sync(self, tokens_needed: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens, waiting if necessary (sync version)."""
        start_time = time.time()
        
        while not self.can_proceed(tokens_needed):
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            wait_time = (tokens_needed - self.tokens) / self.refill_rate
            time.sleep(min(wait_time, 1.0))  # Max 1 second wait
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        self._refill_tokens()
        return {
            'current_tokens': self.tokens,
            'max_tokens': self.max_tokens,
            'refill_rate': self.refill_rate,
            'utilization': 1.0 - (self.tokens / self.max_tokens)
        }

def rate_limited(max_tokens: int, refill_rate: float):
    """Decorator to add rate limiting to functions."""
    limiter = RateLimiter(max_tokens, refill_rate)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire_sync(timeout=30):  # 30 second timeout
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            return func(*args, **kwargs)
        
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator

def async_rate_limited(max_tokens: int, refill_rate: float):
    """Async decorator to add rate limiting to async functions."""
    limiter = RateLimiter(max_tokens, refill_rate)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator

# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name."""
    return _circuit_breakers.get(name)

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all circuit breakers."""
    return _circuit_breakers.copy()

def register_circuit_breaker(breaker: CircuitBreaker) -> None:
    """Register a circuit breaker."""
    _circuit_breakers[breaker.name] = breaker