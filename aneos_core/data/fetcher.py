"""
Data fetcher orchestrator for aNEOS Core.

Coordinates data fetching from multiple sources with caching,
error handling, and data quality assessment.
"""

from aneos_core.data.provider_http import get as provider_get
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
        cache_key = f"neo_data:atomic-physical-v1-native-epoch-v1-source-projection-v2:{designation}"
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
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending = {
                executor.submit(self._fetch_from_source, name, source, designation): name
                for name, source in self.sources.items()
            }
            for future in as_completed(pending):
                name = pending[future]
                try:
                    result = future.result()
                    if result is not None and result.orbital_elements is not None:
                        results[name] = result
                        self.fetch_stats["source_successes"][name] += 1
                    else:
                        self.fetch_stats["source_failures"][name] += 1
                except Exception as exc:
                    self.fetch_stats["source_failures"][name] += 1
                    self.logger.error("Error fetching from %s: %s", name, exc)
        if not results:
            raise DataSourceUnavailableError(
                f"All data sources failed for '{designation}'. "
                f"Tried: {', '.join(self.source_priority)}."
            )
        neo_data = self._reconcile_sources(results)

        # Augment with close approach data from CAD API (supplemental — never blocks)
        try:
            approaches = self._fetch_close_approaches(designation)
            for approach in approaches:
                neo_data.add_close_approach(approach)
            neo_data.evidence_metadata['close_approach_query'] = getattr(approaches, 'query_metadata', {'status': 'unverified_legacy_adapter'})
        except Exception as exc:
            neo_data.evidence_metadata['close_approach_query'] = {'status': 'unavailable', 'reason': str(exc)}

        return neo_data

    def _reconcile_sources(self, results: Dict[str, NEOData]) -> NEOData:
        """Use the shared atomic source projection, including physical-only inputs."""
        from .reconciliation import reconcile_sources
        return reconcile_sources(results,self.source_priority)

    def _fetch_close_approaches(self, designation: str, date_min: str = "now", date_max: Optional[str] = None) -> List:
        """A bounded CAD query, with native TDB evidence and explicit source failure."""
        import requests
        from .sources.cad import parse_cad_payload, close_approach_from_record, CloseApproachQuery
        from .sources.base import _APPLICATION_HEADERS
        query = {'des': designation, 'date-min': date_min, 'dist-max': '0.2',
                 'body': 'Earth', 'limit': 10, 'fullname': 'true'}
        if date_max is not None:
            query['date-max'] = date_max
        try:
            response = provider_get('https://ssd-api.jpl.nasa.gov/cad.api', params=query,
                                    headers=_APPLICATION_HEADERS, timeout=15)
            response.raise_for_status()
            parsed = parse_cad_payload(response.json(), query=query, require_complete=False,
                retrieved_at=datetime.now(timezone.utc).isoformat())
            return CloseApproachQuery([close_approach_from_record(r) for r in parsed['records']], parsed['metadata'])
        except Exception as exc:
            raise DataSourceUnavailableError(f'CAD query unavailable for {designation}: {exc}') from exc

    def fetch_approaches_between(self, designation: str, date_min: str, date_max: str) -> List:
        """Explicit frozen CAD calendar window, bounded rows and visible coverage.

        Dates are TDB query calendar labels, not converted observation timestamps.
        Does not reuse a relative-horizon cache whose window can drift on retry.
        """
        from datetime import date
        if (not isinstance(date_min,str) or not isinstance(date_max,str) or
            date.fromisoformat(date_min).isoformat()!=date_min or
            date.fromisoformat(date_max).isoformat()!=date_max or date_min>date_max):
            raise ValueError('Expected ordered ISO calendar query bounds')
        return self._fetch_close_approaches(designation,date_min=date_min,date_max=date_max)

    def fetch_historical_approaches(self, designation: str, years_back: int = 30) -> List:
        """Bounded historical CAD query, not a complete observation time series."""
        from datetime import timedelta
        from .models import CloseApproach
        from .sources.cad import CloseApproachQuery
        key = f'cad_v1_hist:{designation}:{years_back}'
        cached = self.cache_manager.get(key)
        if isinstance(cached, dict) and cached.get('schema_version') == 'cad-approaches-v1':
            return CloseApproachQuery([CloseApproach.from_dict(r) for r in cached['approaches']], cached['query_metadata'])
        date_min = (datetime.now(timezone.utc) - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        approaches = self._fetch_close_approaches(designation, date_min=date_min)
        self.cache_manager.set(key, {'schema_version': 'cad-approaches-v1',
            'approaches': [ca.to_dict() for ca in approaches],
            'query_metadata': approaches.query_metadata}, ttl=604800)
        return approaches

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
            # The actual request is authoritative. A separate per-object health
            # probe is redundant traffic and can veto an otherwise valid query.
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
                if orbital_elements_data.get('_evidence_metadata',{}).get('physical_evidence') is not None:
                    from .physical_evidence import physical_projection
                    projected=physical_projection(orbital_elements_data['_evidence_metadata']['physical_evidence'])
                    compatibility={'diameter_km':'diameter','rotation_period_hours':'rot_per'}
                    if {compatibility.get(k,k):v for k,v in projected.items()}!=phys_raw:
                        raise ValueError('Physical adapter projection does not match its evidence')
                if phys_raw:
                    from .models import PhysicalProperties
                    try:
                        physical_properties = PhysicalProperties(
                            diameter_km=phys_raw.get("diameter"),
                            diameter_uncertainty=phys_raw.get("diameter_uncertainty"),
                            albedo=phys_raw.get("albedo"),
                            albedo_uncertainty=phys_raw.get("albedo_uncertainty"),
                            rotation_period_hours=phys_raw.get("rot_per"),
                            rotation_period_uncertainty=phys_raw.get("rotation_period_uncertainty"),
                            spectral_type=phys_raw.get("spectral_type"),
                            spectral_class=phys_raw.get("spectral_class"),
                            absolute_magnitude_h=phys_raw.get("absolute_magnitude_h"),
                        )
                    except Exception as exc:
                        raise ValueError(f'Invalid physical projection: {exc}') from exc

                # Build NonGravitationalParameters from _nongrav sub-dict if present
                nongrav = None
                ng_raw = orbital_elements_data.get("_nongrav") if orbital_elements_data else None
                if ng_raw:
                    from .models import NonGravitationalParameters
                    try:
                        nongrav = NonGravitationalParameters.from_dict(ng_raw)
                    except Exception as exc:
                        # A malformed coefficient/covariance is a source error,
                        # not a successful fetch with acceleration silently lost.
                        raise ValueError(f'Invalid non-gravitational evidence: {exc}') from exc

            if orbital_elements:
                neo_data = NEOData(
                    designation=designation,
                    orbital_elements=orbital_elements,
                    physical_properties=physical_properties,
                    nongrav=nongrav,
                    sources_used=[source_name],
                    evidence_metadata=orbital_elements_data.get("_evidence_metadata", {}),
                )
                neo_data.fetched_at = datetime.now(timezone.utc)

                # Wire observation window from SBDB first_obs / last_obs fields
                for date_key, attr in (
                    ("first_observation_date", "first_observation"),
                    ("last_observation_date",  "last_observation"),
                ):
                    raw = orbital_elements_data.get(date_key)
                    if raw:
                        from datetime import timezone as _tz
                        for fmt in ("%Y-%b-%d", "%Y-%m-%d"):
                            try:
                                parsed = datetime.strptime(raw, fmt).replace(tzinfo=_tz.utc)
                                setattr(neo_data, attr, parsed)
                                break
                            except ValueError:
                                pass

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