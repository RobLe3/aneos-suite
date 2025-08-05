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
class PathConfig:
    """File system paths for data storage."""
    data_neos_dir: str = "dataneos"
    data_dir: str = "dataneos/data"
    orbital_dir: str = "dataneos/orbital_elements"
    output_dir: str = "dataneos/daily_outputs"
    cache_dir: str = "dataneos/cache"
    log_file: str = "dataneos/neos_analyzer.log"
    cache_file: str = "dataneos/orbital_elements_cache"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in ['data_neos_dir', 'data_dir', 'orbital_dir', 'output_dir', 'cache_dir']:
            path = getattr(self, attr_name)
            Path(path).mkdir(parents=True, exist_ok=True)

@dataclass
class ANEOSConfig:
    """Complete aNEOS configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    
    # Processing settings
    max_workers: int = 10
    max_subpoint_workers: int = 20
    batch_size: int = 100
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
        
        # Path configuration from environment
        config.paths.data_neos_dir = os.getenv('ANEOS_DATA_DIR', config.paths.data_neos_dir)
        config.paths.log_file = os.getenv('ANEOS_LOG_FILE', config.paths.log_file)
        
        # Processing settings
        config.max_workers = int(os.getenv('ANEOS_MAX_WORKERS', config.max_workers))
        config.batch_size = int(os.getenv('ANEOS_BATCH_SIZE', config.batch_size))
        config.cache_ttl = int(os.getenv('ANEOS_CACHE_TTL', config.cache_ttl))
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'ANEOSConfig':
        """Create config from dictionary."""
        # Extract nested configurations
        api_data = data.get('api', {})
        paths_data = data.get('paths', {})
        thresholds_data = data.get('thresholds', {})
        weights_data = data.get('weights', {})
        
        return cls(
            api=APIConfig(**api_data),
            paths=PathConfig(**paths_data),
            thresholds=ThresholdConfig(**thresholds_data),
            weights=WeightConfig(**weights_data),
            max_workers=data.get('max_workers', 10),
            max_subpoint_workers=data.get('max_subpoint_workers', 20),
            batch_size=data.get('batch_size', 100),
            cache_ttl=data.get('cache_ttl', 3600)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'api': self.api.__dict__,
            'paths': self.paths.__dict__,
            'thresholds': self.thresholds.__dict__,
            'weights': self.weights.__dict__,
            'max_workers': self.max_workers,
            'max_subpoint_workers': self.max_subpoint_workers,
            'batch_size': self.batch_size,
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
    
    def __init__(self, config_path: Optional[str] = None):
        self._config = None
        self._config_path = config_path
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
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
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
        
        if errors:
            error_msg = "Configuration validation errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    @property
    def config(self) -> ANEOSConfig:
        """Get the current configuration."""
        return self._config
    
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
    
    def get_legacy_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for backwards compatibility."""
        return {
            "DATA_NEOS_DIR": self._config.paths.data_neos_dir,
            "DATA_DIR": self._config.paths.data_dir,
            "ORBITAL_DIR": self._config.paths.orbital_dir,
            "LOG_FILE": self._config.paths.log_file,
            "OUTPUT_DIR": self._config.paths.output_dir,
            "CACHE_FILE": self._config.paths.cache_file,
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