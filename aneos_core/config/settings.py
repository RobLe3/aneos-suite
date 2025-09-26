"""
Configuration management for aNEOS - replaces global CONFIG dictionary.

This module provides structured, type-safe configuration management with support
for loading from files, environment variables, and programmatic configuration.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Optional yaml support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


def _get_bool_env(var_name: str, default: bool) -> bool:
    """Parse boolean environment variables using common truthy strings."""
    value = os.getenv(var_name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass
class ThresholdConfig:
    """Thresholds for anomaly detection."""
    eccentricity: float = 0.8
    inclination: float = 45.0
    velocity_shift: float = 5.0
    temporal_inertia: float = 100.0
    geo_eps: int = 5
    geo_min_samples: int = 2
    geo_min_clusters: int = 2
    diameter_min: float = 0.1
    diameter_max: float = 10.0
    albedo_min: float = 0.05
    albedo_max: float = 0.5
    min_subpoints: int = 2
    acceleration_threshold: float = 0.0005
    observation_gap_multiplier: int = 3
    albedo_artificial: float = 0.6

@dataclass
class WeightConfig:
    """Weights for different anomaly indicators."""
    orbital_mechanics: float = 1.5
    velocity_shifts: float = 2.0
    close_approach_regularity: float = 2.0
    purpose_driven: float = 2.0
    physical_anomalies: float = 1.0
    temporal_anomalies: float = 1.0
    geographic_clustering: float = 1.0
    acceleration_anomalies: float = 2.0
    spectral_anomalies: float = 1.5
    observation_history: float = 1.0
    detection_history: float = 1.0

@dataclass
class APIConfig:
    """API endpoints and connection settings."""
    neodys_url: str = "https://newton.spacedys.com/neodys/api/"
    mpc_url: str = "https://www.minorplanetcenter.net/"
    horizons_url: str = "https://ssd.jpl.nasa.gov/api/horizons.api"
    sbdb_url: str = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    cad_url: str = "https://ssd-api.jpl.nasa.gov/cad.api"
    request_timeout: int = 10
    max_retries: int = 3
    initial_retry_delay: int = 3
    
    # Data source priority order
    data_sources_priority: List[str] = field(default_factory=lambda: ["SBDB", "NEODyS", "MPC", "Horizons"])

@dataclass
class DirectoryConfig:
    """Directory layout used by the analysis pipeline."""

    base_dir: str = "dataneos"
    data_dir: Optional[str] = None
    orbital_dir: Optional[str] = None
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    log_file: Optional[str] = None
    cache_file: Optional[str] = None

    def __post_init__(self) -> None:
        self._apply_defaults()

    def _apply_defaults(self, force: bool = False) -> None:
        """Populate derived paths while respecting explicit overrides."""

        if force or self.data_dir is None:
            self.data_dir = os.path.join(self.base_dir, "data")
        if force or self.orbital_dir is None:
            self.orbital_dir = os.path.join(self.base_dir, "orbital_elements")
        if force or self.output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "daily_outputs")
        if force or self.cache_dir is None:
            self.cache_dir = os.path.join(self.base_dir, "cache")
        if force or self.log_file is None:
            self.log_file = os.path.join(self.base_dir, "neos_analyzer.log")
        if force or self.cache_file is None:
            self.cache_file = os.path.join(self.base_dir, "orbital_elements_cache")

    def apply_base_dir(self, base_dir: str) -> None:
        """Update the base directory and recompute derived paths."""

        self.base_dir = base_dir
        self._apply_defaults(force=True)

    def to_dict(self) -> Dict[str, str]:
        return {
            "base_dir": self.base_dir,
            "data_dir": self.data_dir,
            "orbital_dir": self.orbital_dir,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "log_file": self.log_file,
            "cache_file": self.cache_file,
        }

@dataclass
class ANEOSConfig:
    """Complete aNEOS configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    
    # Processing settings
    max_workers: int = 10
    max_subpoint_workers: int = 20
    analysis_parallel: bool = True
    analysis_queue_size: int = 1000
    analysis_timeout: int = 300
    batch_processing_enabled: bool = True
    batch_size: int = 100
    batch_max_size: int = 1000
    batch_timeout: int = 3600
    analysis_memory_limit: int = 4096
    analysis_temp_dir: str = "/tmp/aneos"
    analysis_cleanup_temp_files: bool = True
    cache_ttl: int = 3600  # 1 hour default TTL
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ANEOSConfig':
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    if not HAS_YAML:
                        raise ImportError("PyYAML not installed, cannot load YAML config")
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return cls._from_dict(data)
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> 'ANEOSConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # API configuration from environment
        config.api.neodys_url = os.getenv('ANEOS_NEODYS_URL', config.api.neodys_url)
        config.api.mpc_url = os.getenv('ANEOS_MPC_URL', config.api.mpc_url)
        config.api.horizons_url = os.getenv('ANEOS_HORIZONS_URL', config.api.horizons_url)
        config.api.sbdb_url = os.getenv('ANEOS_SBDB_URL', config.api.sbdb_url)
        config.api.request_timeout = int(os.getenv('ANEOS_REQUEST_TIMEOUT', config.api.request_timeout))
        config.api.max_retries = int(os.getenv('ANEOS_MAX_RETRIES', config.api.max_retries))

        data_source_primary = os.getenv('ANEOS_DATA_SOURCES_PRIMARY')
        data_source_fallback = os.getenv('ANEOS_DATA_SOURCES_FALLBACK')
        if data_source_primary or data_source_fallback:
            priority: List[str] = []
            if data_source_primary:
                priority.append(data_source_primary.strip())
            if data_source_fallback:
                priority.extend([
                    source.strip() for source in data_source_fallback.split(',') if source.strip()
                ])
            if priority:
                config.api.data_sources_priority = priority
        data_sources_timeout = os.getenv('ANEOS_DATA_SOURCES_TIMEOUT')
        if data_sources_timeout is not None:
            config.api.request_timeout = int(data_sources_timeout)
        data_sources_retries = os.getenv('ANEOS_DATA_SOURCES_RETRY_ATTEMPTS')
        if data_sources_retries is not None:
            config.api.max_retries = int(data_sources_retries)

        # Path configuration from environment
        base_dir_override = os.getenv('ANEOS_BASE_DIR')
        if base_dir_override:
            config.directories.apply_base_dir(base_dir_override)

        data_dir_override = os.getenv('ANEOS_DATA_DIR')
        if data_dir_override:
            config.directories.data_dir = data_dir_override
        orbital_dir_override = os.getenv('ANEOS_ORBITAL_DIR')
        if orbital_dir_override:
            config.directories.orbital_dir = orbital_dir_override
        output_dir_override = os.getenv('ANEOS_OUTPUT_DIR')
        if output_dir_override:
            config.directories.output_dir = output_dir_override
        cache_file_override = os.getenv('ANEOS_CACHE_FILE')
        if cache_file_override:
            config.directories.cache_file = cache_file_override
        log_file_override = os.getenv('ANEOS_LOG_FILE')
        if log_file_override:
            config.directories.log_file = log_file_override

        # Processing settings
        analysis_max_workers = os.getenv('ANEOS_ANALYSIS_MAX_WORKERS')
        if analysis_max_workers is not None:
            config.max_workers = int(analysis_max_workers)
        else:
            config.max_workers = int(os.getenv('ANEOS_MAX_WORKERS', config.max_workers))

        config.analysis_parallel = _get_bool_env('ANEOS_ANALYSIS_PARALLEL', config.analysis_parallel)
        config.analysis_queue_size = int(os.getenv('ANEOS_ANALYSIS_QUEUE_SIZE', config.analysis_queue_size))
        config.analysis_timeout = int(os.getenv('ANEOS_ANALYSIS_TIMEOUT', config.analysis_timeout))

        config.batch_processing_enabled = _get_bool_env('ANEOS_BATCH_PROCESSING_ENABLED', config.batch_processing_enabled)
        config.batch_size = int(os.getenv('ANEOS_BATCH_SIZE', config.batch_size))
        config.batch_max_size = int(os.getenv('ANEOS_BATCH_MAX_SIZE', config.batch_max_size))
        config.batch_timeout = int(os.getenv('ANEOS_BATCH_TIMEOUT', config.batch_timeout))

        config.analysis_memory_limit = int(os.getenv('ANEOS_ANALYSIS_MEMORY_LIMIT', config.analysis_memory_limit))
        config.analysis_temp_dir = os.getenv('ANEOS_ANALYSIS_TEMP_DIR', config.analysis_temp_dir)
        config.analysis_cleanup_temp_files = _get_bool_env(
            'ANEOS_ANALYSIS_CLEANUP_TEMP_FILES', config.analysis_cleanup_temp_files
        )

        config.cache_ttl = int(os.getenv('ANEOS_CACHE_TTL', config.cache_ttl))

        eccentricity_override = os.getenv('ANEOS_ECCENTRICITY_THRESHOLD')
        if eccentricity_override is not None:
            config.thresholds.eccentricity = float(eccentricity_override)

        return config

    @property
    def paths(self) -> DirectoryConfig:
        """Backward compatible alias for the renamed directory settings."""

        return self.directories

    @paths.setter
    def paths(self, value: DirectoryConfig) -> None:
        self.directories = value
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'ANEOSConfig':
        """Create config from dictionary."""
        # Extract nested configurations
        api_data = data.get('api', {})
        paths_data = data.get('paths', {})
        directories_data = data.get('directories')
        thresholds_data = data.get('thresholds', {})
        weights_data = data.get('weights', {})

        if not directories_data and paths_data:
            directories_data = {
                'base_dir': paths_data.get('data_neos_dir') or paths_data.get('base_dir', 'dataneos'),
                'data_dir': paths_data.get('data_dir'),
                'orbital_dir': paths_data.get('orbital_dir'),
                'output_dir': paths_data.get('output_dir'),
                'cache_dir': paths_data.get('cache_dir'),
                'log_file': paths_data.get('log_file'),
                'cache_file': paths_data.get('cache_file'),
            }

        return cls(
            api=APIConfig(**api_data),
            directories=DirectoryConfig(**(directories_data or {})),
            thresholds=ThresholdConfig(**thresholds_data),
            weights=WeightConfig(**weights_data),
            max_workers=data.get('max_workers', 10),
            max_subpoint_workers=data.get('max_subpoint_workers', 20),
            analysis_parallel=data.get('analysis_parallel', True),
            analysis_queue_size=data.get('analysis_queue_size', 1000),
            analysis_timeout=data.get('analysis_timeout', 300),
            batch_processing_enabled=data.get('batch_processing_enabled', True),
            batch_size=data.get('batch_size', 100),
            batch_max_size=data.get('batch_max_size', 1000),
            batch_timeout=data.get('batch_timeout', 3600),
            analysis_memory_limit=data.get('analysis_memory_limit', 4096),
            analysis_temp_dir=data.get('analysis_temp_dir', "/tmp/aneos"),
            analysis_cleanup_temp_files=data.get('analysis_cleanup_temp_files', True),
            cache_ttl=data.get('cache_ttl', 3600)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'api': self.api.__dict__,
            'directories': self.directories.to_dict(),
            'thresholds': self.thresholds.__dict__,
            'weights': self.weights.__dict__,
            'max_workers': self.max_workers,
            'max_subpoint_workers': self.max_subpoint_workers,
            'analysis_parallel': self.analysis_parallel,
            'analysis_queue_size': self.analysis_queue_size,
            'analysis_timeout': self.analysis_timeout,
            'batch_processing_enabled': self.batch_processing_enabled,
            'batch_size': self.batch_size,
            'batch_max_size': self.batch_max_size,
            'batch_timeout': self.batch_timeout,
            'analysis_memory_limit': self.analysis_memory_limit,
            'analysis_temp_dir': self.analysis_temp_dir,
            'analysis_cleanup_temp_files': self.analysis_cleanup_temp_files,
            'cache_ttl': self.cache_ttl
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    if not HAS_YAML:
                        raise ImportError("PyYAML not installed, cannot save YAML config")
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
                else:
                    json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise

class ConfigManager:
    """Manages aNEOS configuration with multiple loading strategies."""

    def __init__(self, config_path: Optional[str] = None, *, config_file: Optional[str] = None):
        self._config = None
        self._config_path = config_file or config_path
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration using priority order: file -> env -> defaults."""
        if self._config_path and Path(self._config_path).exists():
            logger.info(f"Loading configuration from file: {self._config_path}")
            self._config = ANEOSConfig.from_file(self._config_path)
        else:
            logger.info("Loading configuration from environment variables")
            self._config = ANEOSConfig.from_env()

        # Validate configuration
        self._validate_config()

    def _collect_validation_errors(self) -> List[str]:
        """Return a list of validation issues for the current configuration."""
        errors = []

        # Validate thresholds
        if not (0 < self._config.thresholds.eccentricity <= 1):
            errors.append("eccentricity threshold must be between 0 and 1")

        if not (0 <= self._config.thresholds.inclination <= 180):
            errors.append("inclination threshold must be between 0 and 180 degrees")

        if self._config.api.request_timeout <= 0:
            errors.append("request_timeout must be positive")

        if self._config.api.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self._config.max_workers <= 0:
            errors.append("max_workers must be positive")

        return errors

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        errors = self._collect_validation_errors()

        if errors:
            error_msg = "Configuration validation errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")

    @property
    def config(self) -> ANEOSConfig:
        """Get the current configuration."""
        return self._config

    @property
    def thresholds(self) -> ThresholdConfig:
        return self._config.thresholds

    @property
    def weights(self) -> WeightConfig:
        return self._config.weights

    @property
    def api(self) -> APIConfig:
        return self._config.api

    @property
    def directories(self) -> DirectoryConfig:
        return self._config.directories

    @property
    def data_sources_priority(self) -> List[str]:
        return list(self._config.api.data_sources_priority)

    def reload(self) -> None:
        """Reload configuration from source."""
        logger.info("Reloading configuration")
        self._load_config()

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters programmatically."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated config parameter: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

        self._validate_config()

    def validate_configuration(self) -> bool:
        """Public validation helper used by the regression tests."""

        errors = self._collect_validation_errors()
        if errors:
            logger.error("Configuration validation errors: %s", "; ".join(errors))
            return False
        return True

    def get_legacy_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for backwards compatibility."""
        return {
            "DATA_NEOS_DIR": self._config.directories.base_dir,
            "DATA_DIR": self._config.directories.data_dir,
            "ORBITAL_DIR": self._config.directories.orbital_dir,
            "LOG_FILE": self._config.directories.log_file,
            "OUTPUT_DIR": self._config.directories.output_dir,
            "CACHE_FILE": self._config.directories.cache_file,
            "NEODYS_API_URL": self._config.api.neodys_url,
            "MPC_API_URL": self._config.api.mpc_url,
            "HORIZONS_API_URL": self._config.api.horizons_url,
            "DATA_SOURCES_PRIORITY": self._config.api.data_sources_priority,
            "WEIGHTS": self._config.weights.__dict__,
            "THRESHOLDS": self._config.thresholds.__dict__,
            "REQUEST_TIMEOUT": self._config.api.request_timeout,
            "MAX_RETRIES": self._config.api.max_retries,
            "INITIAL_RETRY_DELAY": self._config.api.initial_retry_delay
        }

    def save_to_file(self, config_path: str) -> None:
        """Persist the active configuration to a JSON/YAML file."""

        self._config.save(config_path)

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return (
            "ConfigManager(base_dir={base}, data_sources={sources})"
        ).format(
            base=self._config.directories.base_dir,
            sources=",".join(self._config.api.data_sources_priority),
        )

# Global instance for backwards compatibility
_global_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager

def get_config() -> ANEOSConfig:
    """Get the current configuration."""
    return get_config_manager().config
