"""
Unit tests for configuration management.
"""

import os
import tempfile
import json
import pytest
from unittest.mock import patch

from aneos_core.config.settings import ConfigManager, ThresholdConfig, WeightConfig, APIConfig, DirectoryConfig


class TestThresholdConfig:
    """Test ThresholdConfig dataclass."""
    
    def test_default_values(self):
        """Test default threshold values."""
        config = ThresholdConfig()
        
        assert config.eccentricity == 0.8
        assert config.inclination == 45.0
        assert config.velocity_shift == 5.0
        assert config.temporal_inertia == 100.0
        assert config.geo_eps == 5
        assert config.geo_min_samples == 2
        assert config.geo_min_clusters == 2
        assert config.diameter_min == 0.1
        assert config.diameter_max == 10.0
        assert config.albedo_min == 0.05
        assert config.albedo_max == 0.5
        assert config.min_subpoints == 2
        assert config.acceleration_threshold == 0.0005
        assert config.observation_gap_multiplier == 3
        assert config.albedo_artificial == 0.6
    
    def test_custom_values(self):
        """Test custom threshold values."""
        config = ThresholdConfig(
            eccentricity=0.9,
            inclination=60.0,
            velocity_shift=10.0
        )
        
        assert config.eccentricity == 0.9
        assert config.inclination == 60.0
        assert config.velocity_shift == 10.0
        # Other values should remain default
        assert config.temporal_inertia == 100.0


class TestWeightConfig:
    """Test WeightConfig dataclass."""
    
    def test_default_values(self):
        """Test default weight values."""
        config = WeightConfig()
        
        assert config.orbital_mechanics == 1.5
        assert config.velocity_shifts == 2.0
        assert config.close_approach_regularity == 2.0
        assert config.purpose_driven == 2.0
        assert config.physical_anomalies == 1.0
        assert config.temporal_anomalies == 1.0
        assert config.geographic_clustering == 1.0
        assert config.acceleration_anomalies == 2.0
        assert config.spectral_anomalies == 1.5
        assert config.observation_history == 1.0
        assert config.detection_history == 1.0


class TestAPIConfig:
    """Test APIConfig dataclass."""
    
    def test_default_values(self):
        """Test default API configuration values."""
        config = APIConfig()
        
        assert config.neodys_url == "https://newton.spacedys.com/neodys/api/"
        assert config.mpc_url == "https://www.minorplanetcenter.net/"
        assert config.horizons_url == "https://ssd.jpl.nasa.gov/api/horizons.api"
        assert config.sbdb_url == "https://ssd-api.jpl.nasa.gov/sbdb.api"
        assert config.request_timeout == 10
        assert config.max_retries == 3
        assert config.initial_retry_delay == 3


class TestDirectoryConfig:
    """Test DirectoryConfig dataclass."""
    
    def test_default_paths(self):
        """Test default directory paths."""
        config = DirectoryConfig()
        
        assert config.base_dir == "dataneos"
        assert config.data_dir == os.path.join("dataneos", "data")
        assert config.orbital_dir == os.path.join("dataneos", "orbital_elements")
        assert config.output_dir == os.path.join("dataneos", "daily_outputs")
        assert config.log_file == os.path.join("dataneos", "neos_analyzer.log")
        assert config.cache_file == os.path.join("dataneos", "orbital_elements_cache")
    
    def test_custom_base_dir(self):
        """Test custom base directory."""
        config = DirectoryConfig(base_dir="/custom/path")
        
        assert config.base_dir == "/custom/path"
        assert config.data_dir == os.path.join("/custom/path", "data")
        assert config.orbital_dir == os.path.join("/custom/path", "orbital_elements")


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        
        assert isinstance(config_manager.thresholds, ThresholdConfig)
        assert isinstance(config_manager.weights, WeightConfig)
        assert isinstance(config_manager.api, APIConfig)
        assert isinstance(config_manager.directories, DirectoryConfig)
        assert config_manager.data_sources_priority == ["SBDB", "NEODyS", "MPC", "Horizons"]
    
    def test_get_legacy_config(self):
        """Test legacy configuration format generation."""
        config_manager = ConfigManager()
        legacy_config = config_manager.get_legacy_config()
        
        # Check structure
        assert "DATA_NEOS_DIR" in legacy_config
        assert "WEIGHTS" in legacy_config
        assert "THRESHOLDS" in legacy_config
        assert "DATA_SOURCES_PRIORITY" in legacy_config
        
        # Check values
        assert legacy_config["DATA_NEOS_DIR"] == "dataneos"
        assert legacy_config["WEIGHTS"]["orbital_mechanics"] == 1.5
        assert legacy_config["THRESHOLDS"]["eccentricity"] == 0.8
        assert legacy_config["DATA_SOURCES_PRIORITY"] == ["SBDB", "NEODyS", "MPC", "Horizons"]
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            "thresholds": {
                "eccentricity": 0.9,
                "inclination": 50.0
            },
            "weights": {
                "orbital_mechanics": 2.0
            },
            "api": {
                "request_timeout": 15
            },
            "directories": {
                "base_dir": "/tmp/test"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            
            # Check that values were loaded
            assert config_manager.thresholds.eccentricity == 0.9
            assert config_manager.thresholds.inclination == 50.0
            # Default values should be preserved
            assert config_manager.thresholds.velocity_shift == 5.0
            
            assert config_manager.weights.orbital_mechanics == 2.0
            assert config_manager.api.request_timeout == 15
            assert config_manager.directories.base_dir == "/tmp/test"
            
        finally:
            os.unlink(config_file)
    
    @patch.dict(os.environ, {
        'ANEOS_BASE_DIR': '/env/test',
        'ANEOS_REQUEST_TIMEOUT': '20',
        'ANEOS_ECCENTRICITY_THRESHOLD': '0.85'
    })
    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        config_manager = ConfigManager()
        
        assert config_manager.directories.base_dir == "/env/test"
        assert config_manager.api.request_timeout == 20
        assert config_manager.thresholds.eccentricity == 0.85
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config_manager.save_to_file(config_file)
            
            # Load and verify
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            
            assert "thresholds" in saved_data
            assert "weights" in saved_data
            assert "api" in saved_data
            assert "directories" in saved_data
            assert saved_data["thresholds"]["eccentricity"] == 0.8
            
        finally:
            os.unlink(config_file)
    
    def test_validation_success(self):
        """Test successful configuration validation."""
        config_manager = ConfigManager()
        assert config_manager.validate_configuration() is True
    
    def test_validation_failure(self):
        """Test configuration validation with invalid values."""
        config_manager = ConfigManager()
        
        # Set invalid values
        config_manager.thresholds.eccentricity = 1.5  # > 1
        config_manager.api.request_timeout = -1  # negative
        
        assert config_manager.validate_configuration() is False
    
    def test_string_representation(self):
        """Test string representation of ConfigManager."""
        config_manager = ConfigManager()
        str_repr = str(config_manager)
        
        assert "ConfigManager" in str_repr
        assert "dataneos" in str_repr
        assert "SBDB" in str_repr