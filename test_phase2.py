#!/usr/bin/env python3
"""
Test script for Phase 2 modular refactoring components.

This script tests the core functionality of the new modular architecture
including ConfigManager, CacheManager, and DataSourceBase implementations.
"""

import asyncio
import sys
import logging
from pathlib import Path
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from aneos_core.config.settings import ANEOSConfig, ConfigManager
from aneos_core.data.cache import CacheManager, OrbitElementsCache
from aneos_core.data.models import OrbitalElements, NEOData, CloseApproach
from aneos_core.data.sources.sbdb import SBDBSource
from aneos_core.utils.patterns import CircuitBreaker, RateLimiter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_manager():
    """Test ConfigManager functionality."""
    logger.info("🧪 Testing ConfigManager...")
    
    try:
        # Test default configuration
        config_manager = ConfigManager()
        config = config_manager.config
        
        assert config.api.sbdb_url == "https://ssd-api.jpl.nasa.gov/sbdb.api"
        assert config.thresholds.eccentricity == 0.8
        assert config.weights.orbital_mechanics == 1.5
        
        # Test configuration update
        config_manager.update_config(max_workers=20)
        assert config_manager.config.max_workers == 20
        
        # Test legacy compatibility
        legacy_config = config_manager.get_legacy_config()
        assert "DATA_NEOS_DIR" in legacy_config
        assert "WEIGHTS" in legacy_config
        
        logger.info("✅ ConfigManager tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ConfigManager test failed: {e}")
        return False

def test_cache_manager():
    """Test CacheManager functionality."""
    logger.info("🧪 Testing CacheManager...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache manager
            cache = CacheManager(temp_dir, default_ttl=10)
            
            # Test basic operations
            cache.set("test_key", {"value": 123, "data": "test"})
            result = cache.get("test_key")
            assert result is not None
            assert result["value"] == 123
            
            # Test TTL
            import time
            cache.set("ttl_test", "expires_soon", ttl=1)
            time.sleep(2)
            expired_result = cache.get("ttl_test")
            assert expired_result is None
            
            # Test orbital elements cache
            orbital_cache = OrbitElementsCache(temp_dir)
            test_elements = {
                "eccentricity": 0.5,
                "inclination": 15.0,
                "semi_major_axis": 1.2
            }
            
            orbital_cache.set_orbital_elements("2023 TEST", "SBDB", test_elements)
            retrieved = orbital_cache.get_orbital_elements("2023 TEST", "SBDB")
            assert retrieved is not None
            assert retrieved["eccentricity"] == 0.5
            
            # Test statistics
            stats = cache.get_stats()
            assert "hits" in stats
            assert "misses" in stats
            
            cache.shutdown()
            
        logger.info("✅ CacheManager tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ CacheManager test failed: {e}")
        return False

def test_data_models():
    """Test data model functionality."""
    logger.info("🧪 Testing Data Models...")
    
    try:
        # Test OrbitalElements
        elements_data = {
            "eccentricity": 0.7,
            "inclination": 25.0,
            "semi_major_axis": 1.5,
            "ra_of_ascending_node": 45.0,
            "arg_of_periapsis": 90.0,
            "mean_anomaly": 180.0,
            "diameter": 2.5,
            "albedo": 0.15
        }
        
        elements = OrbitalElements.from_dict(elements_data)
        assert elements.eccentricity == 0.7
        assert elements.completeness_score() > 0.8  # Should be quite complete
        
        # Test validation
        try:
            invalid_elements = OrbitalElements(eccentricity=1.5)  # Invalid: > 1
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected
        
        # Test NEOData
        neo = NEOData(designation="2023 TEST")
        neo.set_orbital_elements(elements)
        
        # Add close approach
        approach = CloseApproach(
            designation="2023 TEST",
            distance_au=0.05,
            relative_velocity_km_s=15.5
        )
        neo.add_close_approach(approach)
        
        assert len(neo.close_approaches) == 1
        assert neo.completeness > 0.8
        
        # Test serialization
        neo_dict = neo.to_dict()
        reconstructed = NEOData.from_dict(neo_dict)
        assert reconstructed.designation == "2023 TEST"
        assert reconstructed.orbital_elements.eccentricity == 0.7
        
        logger.info("✅ Data Models tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Models test failed: {e}")
        return False

def test_circuit_breaker():
    """Test Circuit Breaker pattern."""
    logger.info("🧪 Testing Circuit Breaker...")
    
    try:
        breaker = CircuitBreaker("test_breaker")
        
        # Test normal operation
        assert breaker.can_execute() == True
        breaker.record_success()
        
        # Test failure handling
        for _ in range(5):  # Exceed failure threshold
            breaker.record_failure()
        
        # Should be open now
        assert breaker.can_execute() == False
        
        status = breaker.get_status()
        assert status["state"] == "open"
        assert status["name"] == "test_breaker"
        
        logger.info("✅ Circuit Breaker tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Circuit Breaker test failed: {e}")
        return False

async def test_data_source():
    """Test DataSource implementation."""
    logger.info("🧪 Testing DataSource...")
    
    try:
        # Create configuration
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Create cache manager
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(temp_dir)
            
            # Create SBDB source
            sbdb_source = SBDBSource(config.api, cache)
            
            # Test health check (this is a real API call, might fail in some environments)
            try:
                health_status = await sbdb_source.health_check()
                logger.info(f"SBDB health check result: {health_status}")
            except Exception as e:
                logger.warning(f"SBDB health check failed (expected in some environments): {e}")
            
            # Test status retrieval
            status = sbdb_source.get_status()
            assert status.name == "SBDB"
            assert hasattr(status, 'available')
            
            # Test circuit breaker status
            cb_status = sbdb_source.get_circuit_breaker_status()
            assert "state" in cb_status
            assert "name" in cb_status
            
            # Cleanup
            await sbdb_source.cleanup()
            cache.shutdown()
        
        logger.info("✅ DataSource tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DataSource test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Phase 2 Component Tests...")
    
    tests = [
        ("ConfigManager", test_config_manager),
        ("CacheManager", test_cache_manager),
        ("Data Models", test_data_models),
        ("Circuit Breaker", test_circuit_breaker),
        ("DataSource", test_data_source),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} tests...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            logger.error(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("🎯 TEST SUMMARY:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 2 component tests completed successfully!")
        logger.info("🏗️  Phase 2 modular architecture is ready!")
    else:
        logger.error("⚠️  Some tests failed. Review the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)