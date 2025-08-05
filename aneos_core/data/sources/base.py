"""
Abstract base classes for data sources in aNEOS.

This module provides the foundation for all data source implementations,
including circuit breaker patterns, retry logic, and health monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from ...utils.patterns import (
    CircuitBreaker, CircuitBreakerConfig, RetryConfig, 
    async_retry, RateLimiter
)
from ...config.settings import APIConfig
from ..models import OrbitalElements, NEOData
from ..cache import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class DataSourceStatus:
    """Status information for a data source."""
    name: str
    available: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0

@dataclass 
class FetchResult:
    """Result from data fetching operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    source: Optional[str] = None
    response_time_ms: Optional[float] = None
    cached: bool = False

class DataSourceException(Exception):
    """Base exception for data source operations."""
    
    def __init__(self, source: str, message: str, retryable: bool = True):
        self.source = source
        self.retryable = retryable
        super().__init__(f"[{source}] {message}")

class DataSourceBase(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, name: str, config: APIConfig, cache_manager: Optional[CacheManager] = None):
        self.name = name
        self.config = config
        self.cache_manager = cache_manager
        
        # Circuit breaker for reliability
        self._circuit_breaker = CircuitBreaker(
            name=f"data_source_{name}",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60,
                success_threshold=3
            )
        )
        
        # Rate limiter to respect API limits
        self._rate_limiter = RateLimiter(
            max_tokens=10,  # 10 requests per second by default
            refill_rate=1.0
        )
        
        # Status tracking
        self._status = DataSourceStatus(
            name=name,
            available=True,
            last_check=datetime.utcnow()
        )
        
        # Session for HTTP requests
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Data source '{name}' initialized")
    
    def _get_session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retries
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
            
            # Set timeout
            self._session.timeout = self.config.request_timeout
            
        return self._session
    
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session."""
        if self._async_session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._async_session = aiohttp.ClientSession(timeout=timeout)
        
        return self._async_session
    
    @abstractmethod
    def get_base_url(self) -> str:
        """Get the base URL for this data source."""
        pass
    
    @abstractmethod
    async def fetch_orbital_elements(self, designation: str) -> FetchResult:
        """Fetch orbital elements for a designation."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check on the data source."""
        pass
    
    def _get_cache_key(self, designation: str, data_type: str = "orbital") -> str:
        """Generate cache key for a designation."""
        return f"{self.name}_{data_type}_{designation}"
    
    async def _fetch_from_cache(self, designation: str, data_type: str = "orbital") -> Optional[Dict[str, Any]]:
        """Fetch data from cache if available."""
        if not self.cache_manager:
            return None
        
        cache_key = self._get_cache_key(designation, data_type)
        return self.cache_manager.get(cache_key)
    
    async def _store_in_cache(self, designation: str, data: Dict[str, Any], 
                            data_type: str = "orbital", ttl: Optional[int] = None) -> None:
        """Store data in cache."""
        if not self.cache_manager:
            return
        
        cache_key = self._get_cache_key(designation, data_type)
        self.cache_manager.set(cache_key, data, ttl)
    
    async def fetch_with_resilience(self, designation: str) -> FetchResult:
        """Fetch data with circuit breaker and rate limiting."""
        
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            return FetchResult(
                success=False,
                error_message=f"Circuit breaker open for {self.name}",
                source=self.name
            )
        
        # Apply rate limiting
        await self._rate_limiter.acquire()
        
        # Check cache first
        cached_data = await self._fetch_from_cache(designation)
        if cached_data:
            return FetchResult(
                success=True,
                data=cached_data,
                source=self.name,
                cached=True
            )
        
        # Fetch from source
        start_time = datetime.utcnow()
        
        try:
            result = await self.fetch_orbital_elements(designation)
            
            # Record success
            self._circuit_breaker.record_success()
            self._status.total_requests += 1
            
            # Cache the result if successful
            if result.success and result.data:
                await self._store_in_cache(designation, result.data)
            
            # Update response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.response_time_ms = response_time
            self._status.response_time_ms = response_time
            
            return result
            
        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            self._status.total_requests += 1
            self._status.failed_requests += 1
            self._status.error_message = str(e)
            
            logger.error(f"Data source {self.name} failed for {designation}: {e}")
            
            return FetchResult(
                success=False,
                error_message=str(e),
                source=self.name
            )
    
    def get_status(self) -> DataSourceStatus:
        """Get current status of the data source."""
        # Update success rate
        if self._status.total_requests > 0:
            self._status.success_rate = 1.0 - (self._status.failed_requests / self._status.total_requests)
        
        return self._status
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()
    
    def get_rate_limiter_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        return self._rate_limiter.get_status()
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        self._circuit_breaker.state = self._circuit_breaker.state.CLOSED
        self._circuit_breaker.failure_count = 0
        self._circuit_breaker.failure_timestamps.clear()
        logger.info(f"Circuit breaker reset for {self.name}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            self._session.close()
        
        if self._async_session:
            await self._async_session.close()
        
        logger.info(f"Data source '{self.name}' cleaned up")

class HTTPDataSource(DataSourceBase):
    """Base class for HTTP-based data sources."""
    
    def __init__(self, name: str, base_url: str, config: APIConfig, 
                 cache_manager: Optional[CacheManager] = None):
        super().__init__(name, config, cache_manager)
        self.base_url = base_url.rstrip('/')
    
    def get_base_url(self) -> str:
        """Get the base URL for this data source."""
        return self.base_url
    
    async def health_check(self) -> bool:
        """Perform HTTP health check."""
        try:
            session = await self._get_async_session()
            async with session.get(self.base_url) as response:
                is_healthy = response.status in [200, 400]  # 400 might be expected for some APIs
                
                self._status.available = is_healthy
                self._status.last_check = datetime.utcnow()
                
                if not is_healthy:
                    self._status.error_message = f"HTTP {response.status}"
                else:
                    self._status.error_message = None
                
                return is_healthy
                
        except Exception as e:
            self._status.available = False
            self._status.last_check = datetime.utcnow()
            self._status.error_message = str(e)
            
            logger.error(f"Health check failed for {self.name}: {e}")
            return False
    
    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build full URL with parameters."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items() if v is not None])
            if param_str:
                url += f"?{param_str}"
        
        return url
    
    async def _http_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform HTTP GET request."""
        url = self._build_url(endpoint, params)
        
        session = await self._get_async_session()
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()

class DataSourceManager:
    """Manages multiple data sources with fallback capabilities."""
    
    def __init__(self, sources: List[DataSourceBase], cache_manager: Optional[CacheManager] = None):
        self.sources = {source.name: source for source in sources}
        self.cache_manager = cache_manager
        self.source_priority = [source.name for source in sources]
        
        logger.info(f"DataSourceManager initialized with sources: {list(self.sources.keys())}")
    
    def add_source(self, source: DataSourceBase) -> None:
        """Add a data source."""
        self.sources[source.name] = source
        if source.name not in self.source_priority:
            self.source_priority.append(source.name)
        
        logger.info(f"Added data source: {source.name}")
    
    def remove_source(self, name: str) -> None:
        """Remove a data source."""
        if name in self.sources:
            del self.sources[name]
            if name in self.source_priority:
                self.source_priority.remove(name)
            
            logger.info(f"Removed data source: {name}")
    
    def set_priority(self, priority_order: List[str]) -> None:
        """Set the priority order for data sources."""
        # Validate that all sources exist
        for name in priority_order:
            if name not in self.sources:
                raise ValueError(f"Unknown data source: {name}")
        
        self.source_priority = priority_order
        logger.info(f"Updated source priority: {priority_order}")
    
    async def fetch_orbital_elements(self, designation: str, 
                                   preferred_sources: Optional[List[str]] = None) -> Dict[str, FetchResult]:
        """Fetch orbital elements from multiple sources."""
        sources_to_try = preferred_sources or self.source_priority
        results = {}
        
        # Fetch from all available sources concurrently
        tasks = []
        for source_name in sources_to_try:
            if source_name in self.sources:
                source = self.sources[source_name]
                task = asyncio.create_task(
                    source.fetch_with_resilience(designation),
                    name=f"fetch_{source_name}_{designation}"
                )
                tasks.append((source_name, task))
        
        # Wait for all tasks to complete
        for source_name, task in tasks:
            try:
                result = await task
                results[source_name] = result
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                results[source_name] = FetchResult(
                    success=False,
                    error_message=str(e),
                    source=source_name
                )
        
        return results
    
    def get_best_result(self, results: Dict[str, FetchResult]) -> Optional[FetchResult]:
        """Get the best result based on source priority and success."""
        for source_name in self.source_priority:
            if source_name in results:
                result = results[source_name]
                if result.success and result.data:
                    return result
        
        return None
    
    def merge_results(self, results: Dict[str, FetchResult]) -> Optional[Dict[str, Any]]:
        """Merge results from multiple sources, preferring more complete data."""
        successful_results = {
            name: result for name, result in results.items() 
            if result.success and result.data
        }
        
        if not successful_results:
            return None
        
        # Calculate completeness scores
        completeness_scores = {}
        for name, result in successful_results.items():
            if result.data:
                try:
                    elements = OrbitalElements.from_dict(result.data)
                    completeness_scores[name] = elements.completeness_score()
                except Exception:
                    completeness_scores[name] = 0.0
        
        # Sort by completeness, then by priority
        sorted_sources = sorted(
            successful_results.keys(),
            key=lambda x: (-completeness_scores[x], self.source_priority.index(x))
        )
        
        # Start with the most complete result
        if not sorted_sources:
            return None
        
        best_source = sorted_sources[0]
        merged_data = successful_results[best_source].data.copy()
        
        # Fill in missing fields from other sources
        for source_name in sorted_sources[1:]:
            source_data = successful_results[source_name].data
            
            for key, value in source_data.items():
                if key not in merged_data or merged_data[key] is None:
                    merged_data[key] = value
        
        # Add metadata about sources used
        merged_data['_sources_used'] = list(successful_results.keys())
        merged_data['_primary_source'] = best_source
        merged_data['_completeness_scores'] = completeness_scores
        
        return merged_data
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all sources."""
        results = {}
        tasks = []
        
        for name, source in self.sources.items():
            task = asyncio.create_task(source.health_check(), name=f"health_{name}")
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
        
        return results
    
    def get_all_statuses(self) -> Dict[str, DataSourceStatus]:
        """Get status of all data sources."""
        return {name: source.get_status() for name, source in self.sources.items()}
    
    async def cleanup_all(self) -> None:
        """Clean up all data sources."""
        cleanup_tasks = [source.cleanup() for source in self.sources.values()]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("All data sources cleaned up")