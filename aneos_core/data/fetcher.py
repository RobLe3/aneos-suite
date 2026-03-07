"""
Data fetcher orchestrator for aNEOS Core.

Coordinates data fetching from multiple sources with caching,
error handling, and data quality assessment.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone

from .cache import CacheManager
from .models import NEOData, OrbitalElements
from .sources.base import DataSourceBase
from .sources.sbdb import SBDBSource
from .sources.neodys import NEODySSource
from .sources.mpc import MPCSource
from .sources.horizons import HorizonsSource
from ..utils.errors import DataSourceUnavailableError


class DataFetcher:
    """
    Orchestrates data fetching from multiple NEO data sources.
    
    Features:
    - Multi-source data fetching with prioritization
    - Automatic caching and cache management
    - Data quality assessment and merging
    - Concurrent fetching for performance
    - Health monitoring of data sources
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        source_priority: Optional[List[str]] = None,
        max_workers: int = 4,
        cache_ttl: int = 3600
    ):
        """
        Initialize the data fetcher.
        
        Args:
            cache_manager: Cache manager instance (creates default if None)
            source_priority: Priority order for data sources
            max_workers: Maximum worker threads for concurrent fetching
            cache_ttl: Cache TTL in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        from pathlib import Path
        cache_dir = Path("neo_data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = cache_manager or CacheManager(cache_dir)
        self.cache_ttl = cache_ttl
        
        # Thread pool for concurrent operations
        self.max_workers = max_workers
        
        # Initialize data sources — each source is guarded so a single
        # broken source does not prevent the others from loading.
        from ..config.settings import get_config
        _cfg = get_config()

        self.sources: Dict[str, DataSourceBase] = {}
        _source_defs = [
            ("SBDB",    lambda: SBDBSource(_cfg.api, self.cache_manager)),
            ("NEODyS",  lambda: NEODySSource(_cfg.api, self.cache_manager)),
            ("MPC",     lambda: MPCSource(_cfg.api, self.cache_manager)),
            ("Horizons", lambda: HorizonsSource(_cfg.api, self.cache_manager)),
        ]
        for _name, _factory in _source_defs:
            try:
                self.sources[_name] = _factory()
            except Exception as _exc:
                self.logger.warning(f"Data source {_name} failed to initialize: {_exc}")

        # Source priority (default order, restricted to those that loaded)
        default_priority = [n for n, _ in _source_defs if n in self.sources]
        self.source_priority = source_priority or default_priority
        
        # Statistics tracking
        self.fetch_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "source_successes": {source: 0 for source in self.sources},
            "source_failures": {source: 0 for source in self.sources}
        }
    
    def fetch_neo_data(self, designation: str, force_refresh: bool = False) -> NEOData:
        """
        Fetch comprehensive NEO data from all available sources.
        
        Args:
            designation: NEO designation or name
            force_refresh: Force refresh from sources (bypass cache)
            
        Returns:
            NEOData instance with merged data from all sources
        """
        self.fetch_stats["total_requests"] += 1
        
        # Check cache first
        cache_key = f"neo_data:{designation}"
        if not force_refresh:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                self.fetch_stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for {designation}")
                return self._deserialize_neo_data(cached_data)
        
        self.fetch_stats["cache_misses"] += 1
        
        # Fetch from all sources concurrently — raises DataSourceUnavailableError if all fail
        neo_data = self._fetch_from_all_sources(designation)

        # Cache the result
        serialized_data = self._serialize_neo_data(neo_data)
        self.cache_manager.set(cache_key, serialized_data, ttl=self.cache_ttl)

        self.logger.info(f"Successfully fetched data for {designation} from {len(neo_data.sources_used)} sources")
        return neo_data
    
    def _fetch_from_all_sources(self, designation: str) -> Optional[NEOData]:
        """Fetch data from all available sources concurrently."""
        neo_data = None
        
        # Submit tasks to thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {
                executor.submit(self._fetch_from_source, source_name, source, designation): source_name
                for source_name, source in self.sources.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    source_data = future.result()
                    if source_data:
                        self.fetch_stats["source_successes"][source_name] += 1
                        
                        if neo_data is None:
                            neo_data = source_data
                        else:
                            # Merge: accumulate sources; first successful result wins for orbital data
                            for s in source_data.sources_used:
                                if s not in neo_data.sources_used:
                                    neo_data.sources_used.append(s)
                    else:
                        self.fetch_stats["source_failures"][source_name] += 1
                        
                except Exception as e:
                    self.fetch_stats["source_failures"][source_name] += 1
                    self.logger.error(f"Error fetching from {source_name}: {e}")
        
        if neo_data is None:
            raise DataSourceUnavailableError(
                f"All data sources failed for '{designation}'. "
                f"Tried: {', '.join(self.source_priority)}. "
                f"Check network access and API availability."
            )

        # Augment with close approach data from CAD API (supplemental — never blocks)
        try:
            approaches = self._fetch_close_approaches(designation)
            for approach in approaches:
                neo_data.add_close_approach(approach)
        except Exception:
            pass

        return neo_data

    def _fetch_close_approaches(self, designation: str) -> List:
        """Fetch upcoming close approaches from SBDB CAD API for a single designation."""
        import requests
        from datetime import datetime as _dt
        from .models import CloseApproach
        try:
            resp = requests.get(
                "https://ssd-api.jpl.nasa.gov/cad.api",
                params={
                    "des": designation,
                    "date-min": "now",
                    "dist-max": "0.2",
                    "limit": "10",
                    "fullname": "true",
                },
                timeout=15,
            )
            if not resp.ok:
                return []
            data = resp.json()
            approaches = []
            for row in data.get("data", []):
                # CAD API columns: des,orbit_id,jd,cd,dist,dist_min,dist_max,v_rel,v_inf,t_sigma_f,body
                try:
                    ca = CloseApproach(
                        designation=designation,
                        close_approach_date=_dt.strptime(row[3].strip(), "%Y-%b-%d %H:%M"),
                        distance_au=float(row[4]),
                        relative_velocity_km_s=float(row[7]) if row[7] else None,
                    )
                    approaches.append(ca)
                except Exception:
                    continue
            return approaches
        except Exception as e:
            self.logger.debug(f"CAD API fetch failed for {designation}: {e}")
            return []

    def _fetch_from_source(self, source_name: str, source: DataSourceBase, designation: str) -> Optional[NEOData]:
        """Fetch data from a single source, handling both async and sync source implementations."""
        import asyncio
        import inspect

        def _run(value):
            """Execute a value that may be a coroutine."""
            if inspect.isawaitable(value):
                return asyncio.run(value)
            return value

        try:
            # Health check — sources may be async; run coroutines if needed
            healthy = _run(source.health_check())
            if not healthy:
                self.logger.warning(f"Source {source_name} health check failed")
                return None

            # Fetch orbital elements — result may be a FetchResult or a plain dict
            raw = _run(source.fetch_orbital_elements(designation))
            orbital_elements_data = None
            if raw is not None:
                if hasattr(raw, 'success'):
                    # FetchResult (returned by HTTPDataSource subclasses like SBDB)
                    if raw.success and raw.data:
                        orbital_elements_data = raw.data
                else:
                    # Plain dict (returned by sync-only sources)
                    orbital_elements_data = raw

            orbital_elements = None
            physical_properties = None
            if orbital_elements_data:
                # Keep only keys that OrbitalElements accepts; drop internal metadata (_source, etc.)
                valid_fields = OrbitalElements.__dataclass_fields__
                oe_kwargs = {k: v for k, v in orbital_elements_data.items() if k in valid_fields}
                try:
                    orbital_elements = OrbitalElements(**oe_kwargs)
                except Exception as exc:
                    self.logger.debug(f"OrbitalElements construction failed for {source_name}/{designation}: {exc}")

                # Build PhysicalProperties from the _physical sub-dict if present
                phys_raw = orbital_elements_data.get("_physical", {})
                if phys_raw:
                    from .models import PhysicalProperties
                    try:
                        physical_properties = PhysicalProperties(
                            diameter_km=phys_raw.get("diameter"),
                            albedo=phys_raw.get("albedo"),
                            rotation_period_hours=phys_raw.get("rot_per"),
                            spectral_type=phys_raw.get("spectral_type"),
                            absolute_magnitude_h=phys_raw.get("absolute_magnitude_h"),
                        )
                    except Exception:
                        pass

            if orbital_elements:
                neo_data = NEOData(
                    designation=designation,
                    orbital_elements=orbital_elements,
                    physical_properties=physical_properties,
                    sources_used=[source_name],
                )
                neo_data.fetched_at = datetime.now(timezone.utc)
                return neo_data

        except Exception as e:
            self.logger.error(f"Error fetching from {source_name} for {designation}: {e}")

        return None
    
    def fetch_multiple(self, designations: List[str], force_refresh: bool = False) -> Dict[str, Optional[NEOData]]:
        """
        Fetch data for multiple NEOs concurrently.
        
        Args:
            designations: List of NEO designations
            force_refresh: Force refresh from sources
            
        Returns:
            Dictionary mapping designations to NEOData instances
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_designation = {
                executor.submit(self.fetch_neo_data, designation, force_refresh): designation
                for designation in designations
            }
            
            for future in as_completed(future_to_designation):
                designation = future_to_designation[future]
                try:
                    results[designation] = future.result()
                except Exception as e:
                    self.logger.error(f"Error fetching {designation}: {e}")
                    results[designation] = None
        
        return results
    
    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status of all data sources.
        
        Returns:
            Dictionary containing health status for each source
        """
        health_status = {}
        
        for source_name, source in self.sources.items():
            try:
                status = source.get_status()
                health_status[source_name] = {
                    "name": status.name,
                    "status": "healthy" if status.available else "unhealthy",
                    "last_check": status.last_check.isoformat() if status.last_check else None,
                    "response_time_ms": status.response_time_ms,
                    "success_rate": status.success_rate,
                    "total_requests": status.total_requests,
                    "failed_requests": status.failed_requests,
                    "error": status.error_message,
                }
            except Exception as e:
                health_status[source_name] = {
                    "name": source_name,
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    def get_fetch_statistics(self) -> Dict[str, Any]:
        """
        Get data fetching statistics.
        
        Returns:
            Dictionary containing fetch statistics
        """
        cache_stats = self.cache_manager.get_stats()
        
        total_source_requests = sum(
            self.fetch_stats["source_successes"][source] + self.fetch_stats["source_failures"][source]
            for source in self.sources
        )
        
        return {
            "total_requests": self.fetch_stats["total_requests"],
            "cache_hits": self.fetch_stats["cache_hits"],
            "cache_misses": self.fetch_stats["cache_misses"],
            "cache_hit_rate": (
                self.fetch_stats["cache_hits"] / self.fetch_stats["total_requests"] * 100
                if self.fetch_stats["total_requests"] > 0 else 0
            ),
            "source_statistics": {
                source: {
                    "successes": self.fetch_stats["source_successes"][source],
                    "failures": self.fetch_stats["source_failures"][source],
                    "success_rate": (
                        self.fetch_stats["source_successes"][source] / 
                        (self.fetch_stats["source_successes"][source] + self.fetch_stats["source_failures"][source]) * 100
                        if (self.fetch_stats["source_successes"][source] + self.fetch_stats["source_failures"][source]) > 0
                        else 0
                    )
                }
                for source in self.sources
            },
            "cache_statistics": cache_stats
        }
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (if None, clears all)
        """
        if pattern:
            # Clear specific pattern (would need implementation in cache manager)
            self.logger.info(f"Clearing cache entries matching pattern: {pattern}")
        else:
            self.cache_manager.clear()
            self.logger.info("Cleared all cache entries")
    
    def reset_statistics(self):
        """Reset fetch statistics."""
        self.fetch_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "source_successes": {source: 0 for source in self.sources},
            "source_failures": {source: 0 for source in self.sources}
        }
        
        # Reset source statistics (method is optional on each source)
        for source in self.sources.values():
            reset = getattr(source, 'reset_metrics', None)
            if reset is not None:
                try:
                    reset()
                except Exception:
                    pass
        
        self.logger.info("Reset all fetch statistics")
    
    def _serialize_neo_data(self, neo_data: NEOData) -> Dict[str, Any]:
        """Serialize NEOData for caching."""
        return neo_data.to_dict()
    
    def _deserialize_neo_data(self, data: Dict[str, Any]) -> NEOData:
        """Deserialize NEOData from cache using the model's own from_dict method."""
        return NEOData.from_dict(data)
    
    def add_custom_source(self, name: str, source: DataSourceBase):
        """
        Add a custom data source.
        
        Args:
            name: Source name
            source: DataSourceBase implementation
        """
        self.sources[name] = source
        self.fetch_stats["source_successes"][name] = 0
        self.fetch_stats["source_failures"][name] = 0
        self.logger.info(f"Added custom data source: {name}")
    
    def remove_source(self, name: str):
        """
        Remove a data source.
        
        Args:
            name: Source name to remove
        """
        if name in self.sources:
            del self.sources[name]
            del self.fetch_stats["source_successes"][name]
            del self.fetch_stats["source_failures"][name]
            self.logger.info(f"Removed data source: {name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cache_manager.close()
    
    def close(self):
        """Close data fetcher and cleanup resources."""
        self.cache_manager.close()
        self.logger.info("Data fetcher closed")