"""
Unit tests for cache management.
"""

import time
import tempfile
import threading
from datetime import datetime, timedelta
import pytest

from aneos_core.data.cache import CacheManager, CacheEntry, CacheStats


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_creation(self):
        """Test cache entry creation."""
        now = datetime.now()
        expires_at = now + timedelta(hours=1)
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now,
            expires_at=expires_at
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at == now
        assert entry.expires_at == expires_at
        assert entry.access_count == 0
        assert entry.last_accessed is None
    
    def test_is_expired(self):
        """Test expiration check."""
        now = datetime.now()
        
        # Not expired
        future_expire = CacheEntry(
            key="test",
            value="value",
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )
        assert not future_expire.is_expired()
        
        # Expired
        past_expire = CacheEntry(
            key="test",
            value="value",
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1)
        )
        assert past_expire.is_expired()
        
        # No expiration
        no_expire = CacheEntry(
            key="test",
            value="value",
            created_at=now,
            expires_at=None
        )
        assert not no_expire.is_expired()
    
    def test_touch(self):
        """Test access tracking."""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=datetime.now(),
            expires_at=None
        )
        
        assert entry.access_count == 0
        assert entry.last_accessed is None
        
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.last_accessed is not None
        
        entry.touch()
        assert entry.access_count == 2


class TestCacheStats:
    """Test CacheStats dataclass."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 80.0
        assert stats.miss_rate == 20.0
        
        # Zero case
        empty_stats = CacheStats()
        assert empty_stats.hit_rate == 0.0
        assert empty_stats.miss_rate == 100.0


class TestCacheManager:
    """Test CacheManager class."""
    
    def test_initialization(self):
        """Test cache manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                memory_limit=50,
                default_ttl=3600
            )
            
            assert cache_manager.memory_limit == 50
            assert cache_manager.default_ttl == 3600
            assert cache_manager.size() == 0
            
            cache_manager.close()
    
    def test_basic_operations(self):
        """Test basic get/set operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Test set and get
            cache_manager.set("key1", "value1")
            assert cache_manager.get("key1") == "value1"
            
            # Test default value
            assert cache_manager.get("nonexistent", "default") == "default"
            
            # Test exists
            assert cache_manager.exists("key1")
            assert not cache_manager.exists("nonexistent")
            
            cache_manager.close()
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Set with short TTL
            cache_manager.set("expire_key", "expire_value", ttl=1)
            assert cache_manager.get("expire_key") == "expire_value"
            
            # Wait for expiration
            time.sleep(1.1)
            assert cache_manager.get("expire_key") is None
            
            cache_manager.close()
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                memory_limit=3  # Small limit to force eviction
            )
            
            # Fill cache to limit
            cache_manager.set("key1", "value1")
            cache_manager.set("key2", "value2")
            cache_manager.set("key3", "value3")
            
            # Access key1 to make it most recently used
            cache_manager.get("key1")
            
            # Add another item to trigger eviction
            cache_manager.set("key4", "value4")
            
            # key2 should be evicted (least recently used)
            assert cache_manager.get("key1") == "value1"  # Still there
            assert cache_manager.get("key2") is None  # Evicted
            assert cache_manager.get("key3") == "value3"  # Still there
            assert cache_manager.get("key4") == "value4"  # New item
            
            cache_manager.close()
    
    def test_delete_operation(self):
        """Test delete operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            cache_manager.set("delete_key", "delete_value")
            assert cache_manager.exists("delete_key")
            
            # Delete existing key
            assert cache_manager.delete("delete_key") is True
            assert not cache_manager.exists("delete_key")
            
            # Delete non-existent key
            assert cache_manager.delete("nonexistent") is False
            
            cache_manager.close()
    
    def test_clear_operation(self):
        """Test clear operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Add some items
            cache_manager.set("key1", "value1")
            cache_manager.set("key2", "value2")
            assert cache_manager.size() == 2
            
            # Clear cache
            cache_manager.clear()
            assert cache_manager.size() == 0
            assert cache_manager.get("key1") is None
            assert cache_manager.get("key2") is None
            
            cache_manager.close()
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Add items with different TTLs
            cache_manager.set("keep", "value", ttl=10)  # Long TTL
            cache_manager.set("expire", "value", ttl=1)  # Short TTL
            
            # Wait for short TTL to expire
            time.sleep(1.1)
            
            # Manual cleanup
            removed_count = cache_manager.cleanup_expired()
            
            assert removed_count >= 1  # At least the expired item
            assert cache_manager.get("keep") == "value"
            assert cache_manager.get("expire") is None
            
            cache_manager.close()
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            def worker(thread_id):
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    cache_manager.set(key, value)
                    retrieved = cache_manager.get(key)
                    assert retrieved == value
            
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            cache_manager.close()
    
    def test_statistics(self):
        """Test statistics collection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Generate some cache activity
            cache_manager.set("key1", "value1")
            cache_manager.get("key1")  # Hit
            cache_manager.get("nonexistent")  # Miss
            
            stats = cache_manager.get_stats()
            
            assert "memory_cache" in stats
            assert "performance" in stats
            assert "disk_cache" in stats
            
            # Check performance stats
            perf_stats = stats["performance"]
            assert perf_stats["hits"] >= 1
            assert perf_stats["misses"] >= 1
            assert 0 <= perf_stats["hit_rate"] <= 100
            
            cache_manager.close()
    
    def test_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with CacheManager(cache_dir=temp_dir) as cache_manager:
                cache_manager.set("context_key", "context_value")
                assert cache_manager.get("context_key") == "context_value"
            
            # Cache should be closed after context
    
    def test_bracket_notation(self):
        """Test dictionary-style bracket notation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Set using bracket notation
            cache_manager["bracket_key"] = "bracket_value"
            
            # Get using bracket notation
            assert cache_manager["bracket_key"] == "bracket_value"
            
            # Check contains
            assert "bracket_key" in cache_manager
            assert "nonexistent" not in cache_manager
            
            # Delete using bracket notation
            del cache_manager["bracket_key"]
            
            # Should raise KeyError for non-existent key
            with pytest.raises(KeyError):
                _ = cache_manager["nonexistent"]
            
            cache_manager.close()
    
    def test_serialization(self):
        """Test serialization of different data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            
            # Test different data types
            test_data = {
                "string": "hello world",
                "integer": 42,
                "float": 3.14159,
                "list": [1, 2, 3, "four"],
                "dict": {"nested": "dictionary", "number": 123},
                "boolean": True,
                "none": None
            }
            
            for key, value in test_data.items():
                cache_manager.set(key, value)
                retrieved = cache_manager.get(key)
                assert retrieved == value
            
            cache_manager.close()
    
    def test_memory_only_mode(self):
        """Test memory-only caching mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                disk_cache_enabled=False
            )
            
            cache_manager.set("memory_key", "memory_value")
            assert cache_manager.get("memory_key") == "memory_value"
            
            # Test memory-only flag
            cache_manager.set("memory_only_key", "memory_only_value", memory_only=True)
            assert cache_manager.get("memory_only_key") == "memory_only_value"
            
            cache_manager.close()