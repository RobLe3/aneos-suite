"""
Data fetcher orchestrator for aNEOS Core.

Coordinates data fetching from multiple sources with caching,
error handling, and data quality assessment.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from .cache import CacheManager
from .models import NEOData, OrbitalElements, PhysicalProperties
from .sources.base import DataSourceBase
from .sources.sbdb import SBDBSource
from .sources.neodys import NEODySSource
from .sources.mpc import MPCSource
from .sources.horizons import HorizonsSource


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
        
        # Initialize data sources
        self.sources: Dict[str, DataSourceBase] = {
            "SBDB": SBDBSource(),
            "NEODyS": NEODySSource(),
            "MPC": MPCSource(),
            "Horizons": HorizonsSource()
        }
        
        # Source priority (default order)
        self.source_priority = source_priority or ["SBDB", "NEODyS", "MPC", "Horizons"]
        
        # Statistics tracking
        self.fetch_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "source_successes": {source: 0 for source in self.sources},
            "source_failures": {source: 0 for source in self.sources}
        }
    
    def fetch_neo_data(self, designation: str, force_refresh: bool = False) -> Optional[NEOData]:
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
        
        # Fetch from all sources concurrently
        neo_data = self._fetch_from_all_sources(designation)
        
        if neo_data:
            # Cache the result
            serialized_data = self._serialize_neo_data(neo_data)
            self.cache_manager.set(cache_key, serialized_data, ttl=self.cache_ttl)
            
            self.logger.info(f"Successfully fetched data for {designation} from {len(neo_data.data_sources)} sources")
        else:
            self.logger.warning(f"No data found for {designation}")
        
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
                            # Merge data based on source priority
                            neo_data.merge_from_source(source_data, self.source_priority)
                    else:
                        self.fetch_stats["source_failures"][source_name] += 1
                        
                except Exception as e:
                    self.fetch_stats["source_failures"][source_name] += 1
                    self.logger.error(f"Error fetching from {source_name}: {e}")
        
        return neo_data
    
    def _fetch_from_source(self, source_name: str, source: DataSourceBase, designation: str) -> Optional[NEOData]:
        """Fetch data from a single source."""
        try:
            # Check source health
            if not source.health_check():
                self.logger.warning(f"Source {source_name} health check failed")
                return None
            
            # Fetch orbital elements
            orbital_elements_data = source.fetch_orbital_elements(designation)
            orbital_elements = None
            if orbital_elements_data:
                orbital_elements = OrbitalElements(**orbital_elements_data)
            
            # Fetch physical properties
            physical_properties_data = source.fetch_physical_properties(designation)
            physical_properties = None
            if physical_properties_data:
                # Convert string spectral type to enum if needed
                if "spectral_type" in physical_properties_data:
                    from .models import SpectralType
                    spectral_str = physical_properties_data.get("spectral_type")
                    if spectral_str:
                        try:
                            physical_properties_data["spectral_type"] = SpectralType(spectral_str)
                        except ValueError:
                            physical_properties_data["spectral_type"] = SpectralType.UNKNOWN
                
                physical_properties = PhysicalProperties(**physical_properties_data)
            
            # Create NEOData instance
            if orbital_elements or physical_properties:
                neo_data = NEOData(
                    designation=designation,
                    orbital_elements=orbital_elements,
                    physical_properties=physical_properties,
                    data_sources=[source_name]
                )
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
                health_status[source_name] = source.get_health_status()
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
        
        # Reset source statistics
        for source in self.sources.values():
            source.reset_metrics()
        
        self.logger.info("Reset all fetch statistics")
    
    def _serialize_neo_data(self, neo_data: NEOData) -> Dict[str, Any]:
        """Serialize NEOData for caching."""
        return neo_data.to_dict()
    
    def _deserialize_neo_data(self, data: Dict[str, Any]) -> NEOData:
        """Deserialize NEOData from cache."""
        # Create NEOData from dictionary
        neo_data = NEOData(designation=data["designation"])
        
        # Restore data from serialized form
        if data.get("name"):
            neo_data.name = data["name"]
        
        if data.get("orbital_elements"):
            oe_data = data["orbital_elements"]
            # Convert epoch string back to datetime if present
            if oe_data.get("epoch"):
                oe_data["epoch"] = datetime.fromisoformat(oe_data["epoch"])
            neo_data.orbital_elements = OrbitalElements(**oe_data)
        
        if data.get("physical_properties"):
            pp_data = data["physical_properties"]
            # Convert spectral type back to enum if present
            if pp_data.get("spectral_type"):
                from .models import SpectralType
                try:
                    pp_data["spectral_type"] = SpectralType(pp_data["spectral_type"])
                except ValueError:
                    pp_data["spectral_type"] = SpectralType.UNKNOWN
            neo_data.physical_properties = PhysicalProperties(**pp_data)
        
        # Restore other attributes
        neo_data.data_sources = data.get("data_sources", [])
        neo_data.anomaly_scores = data.get("anomaly_scores", {})
        neo_data.total_anomaly_score = data.get("total_anomaly_score")
        neo_data.is_artificial_candidate = data.get("is_artificial_candidate", False)
        
        if data.get("last_updated"):
            neo_data.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return neo_data
    
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