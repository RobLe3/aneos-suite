"""
Thread-safe caching system for aNEOS - replaces simple JSON file caching.

This module provides a robust caching layer with memory + disk persistence,
TTL support, automatic cleanup, and thread-safe operations.
"""

import threading
import json
import time
import hashlib
from typing import Any, Optional, Dict, Set
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    data: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if the entry is valid (not expired and has data)."""
        return not self.is_expired() and self.data is not None
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheManager:
    """Thread-safe cache manager with memory + disk persistence."""
    
    def __init__(self, cache_dir: str, default_ttl: int = 3600, 
                 max_memory_entries: int = 1000, cleanup_interval: int = 300):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache storage
            default_ttl: Default time-to-live in seconds
            max_memory_entries: Maximum entries in memory cache
            cleanup_interval: Automatic cleanup interval in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries
        self.cleanup_interval = cleanup_interval
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0,
            'cleanup_runs': 0
        }
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
        logger.info(f"CacheManager initialized: dir={cache_dir}, ttl={default_ttl}s, max_entries={max_memory_entries}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_expired()
                self._stats['cleanup_runs'] += 1
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate safe cache key for filesystem."""
        # Hash long keys to avoid filesystem limitations
        if len(key) > 100 or any(c in key for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            return hashlib.sha256(key.encode()).hexdigest()
        return key.replace('/', '_').replace('\\', '_')
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = self._generate_cache_key(key)
        return self.cache_dir / f"{safe_key}.cache"
    
    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry for disk storage."""
        try:
            # Use pickle for complex objects, JSON for simple ones
            if isinstance(entry.data, (dict, list, str, int, float, bool, type(None))):
                serialized_data = json.dumps(entry.data).encode('utf-8')
                data_type = 'json'
            else:
                serialized_data = pickle.dumps(entry.data)
                data_type = 'pickle'
            
            metadata = {
                'data_type': data_type,
                'created_at': entry.created_at,
                'expires_at': entry.expires_at,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'size_bytes': len(serialized_data)
            }
            
            return json.dumps(metadata).encode('utf-8') + b'\n' + serialized_data
            
        except Exception as e:
            logger.error(f"Error serializing cache entry: {e}")
            raise
    
    def _deserialize_entry(self, data: bytes) -> Optional[CacheEntry]:
        """Deserialize cache entry from disk storage."""
        try:
            # Split metadata and data
            lines = data.split(b'\n', 1)
            if len(lines) != 2:
                return None
            
            metadata = json.loads(lines[0].decode('utf-8'))
            serialized_data = lines[1]
            
            # Deserialize data based on type
            if metadata['data_type'] == 'json':
                entry_data = json.loads(serialized_data.decode('utf-8'))
            else:
                entry_data = pickle.loads(serialized_data)
            
            return CacheEntry(
                data=entry_data,
                created_at=metadata['created_at'],
                expires_at=metadata['expires_at'],
                access_count=metadata['access_count'],
                last_accessed=metadata['last_accessed'],
                size_bytes=metadata['size_bytes']
            )
            
        except Exception as e:
            logger.error(f"Error deserializing cache entry: {e}")
            return None
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk."""
        cache_file = self._get_cache_file_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = f.read()
            
            entry = self._deserialize_entry(data)
            if entry and entry.is_valid():
                self._stats['disk_reads'] += 1
                return entry
            else:
                # Remove invalid/expired entry
                cache_file.unlink(missing_ok=True)
                return None
                
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            # Remove corrupted file
            cache_file.unlink(missing_ok=True)
            return None
    
    def _save_to_disk(self, key: str, entry: CacheEntry) -> bool:
        """Save cache entry to disk."""
        cache_file = self._get_cache_file_path(key)
        
        try:
            # Use temporary file for atomic write
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                           dir=self.cache_dir, prefix='tmp_') as temp_file:
                serialized = self._serialize_entry(entry)
                temp_file.write(serialized)
                temp_file.flush()
                temp_path = temp_file.name
            
            # Atomic move
            shutil.move(temp_path, cache_file)
            self._stats['disk_writes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error saving cache file {cache_file}: {e}")
            # Clean up temporary file if it exists
            try:
                Path(temp_path).unlink(missing_ok=True)
            except:
                pass
            return False
    
    def _evict_lru_memory(self) -> None:
        """Evict least recently used entries from memory cache."""
        if len(self._memory_cache) <= self.max_memory_entries:
            return
        
        # Sort by last accessed time
        entries = [(key, entry.last_accessed) for key, entry in self._memory_cache.items()]
        entries.sort(key=lambda x: x[1])
        
        # Remove oldest entries
        to_remove = len(entries) - self.max_memory_entries + 1
        for key, _ in entries[:to_remove]:
            # Save to disk before evicting from memory
            entry = self._memory_cache[key]
            self._save_to_disk(key, entry)
            del self._memory_cache[key]
            self._stats['evictions'] += 1
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries from memory and disk."""
        current_time = time.time()
        
        with self._lock:
            # Clean memory cache
            expired_keys = [key for key, entry in self._memory_cache.items() 
                          if entry.is_expired()]
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired memory entries")
        
        # Clean disk cache
        expired_files = []
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                # Check file modification time as quick filter
                if cache_file.stat().st_mtime < current_time - (2 * self.default_ttl):
                    # Load and check actual expiration
                    with open(cache_file, 'rb') as f:
                        data = f.read()
                    entry = self._deserialize_entry(data)
                    if entry and entry.is_expired():
                        expired_files.append(cache_file)
            except Exception:
                # Remove corrupted files
                expired_files.append(cache_file)
        
        for cache_file in expired_files:
            cache_file.unlink(missing_ok=True)
        
        if expired_files:
            logger.debug(f"Cleaned up {len(expired_files)} expired disk entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry.is_valid():
                    entry.touch()
                    self._stats['hits'] += 1
                    return entry.data
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
            
            # Check disk cache
            entry = self._load_from_disk(key)
            if entry and entry.is_valid():
                entry.touch()
                # Load back to memory cache
                self._memory_cache[key] = entry
                self._evict_lru_memory()
                self._stats['hits'] += 1
                return entry.data
            
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if value is None:
            return self.delete(key)
        
        ttl = ttl or self.default_ttl
        current_time = time.time()
        
        # Calculate size estimate
        try:
            size_estimate = len(json.dumps(value, default=str))
        except:
            size_estimate = len(str(value))
        
        entry = CacheEntry(
            data=value,
            created_at=current_time,
            expires_at=current_time + ttl,
            size_bytes=size_estimate
        )
        
        with self._lock:
            # Store in memory
            self._memory_cache[key] = entry
            self._evict_lru_memory()
            
            # Async save to disk (best effort)
            success = self._save_to_disk(key, entry)
            
            if success:
                logger.debug(f"Cached key '{key}' with TTL {ttl}s")
            else:
                logger.warning(f"Failed to save key '{key}' to disk")
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            deleted = False
            
            # Remove from memory
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
            
            # Remove from disk
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
            
            if deleted:
                logger.debug(f"Deleted cache key '{key}'")
            
            return deleted
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Clear memory
            self._memory_cache.clear()
            
            # Clear disk
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink(missing_ok=True)
            
            logger.info("Cache cleared")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None
    
    def keys(self) -> Set[str]:
        """Get all cache keys (memory + disk)."""
        with self._lock:
            keys = set(self._memory_cache.keys())
            
            # Add disk keys
            for cache_file in self.cache_dir.glob("*.cache"):
                # Try to reverse the key generation
                file_key = cache_file.stem
                keys.add(file_key)
            
            return keys
    
    def size(self) -> int:
        """Get total number of cache entries."""
        return len(self.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'memory_entries': len(self._memory_cache),
                'total_entries': self.size(),
                'cache_dir': str(self.cache_dir),
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._memory_cache.values())
            
            return {
                'memory_entries': len(self._memory_cache),
                'estimated_memory_bytes': total_size,
                'estimated_memory_mb': total_size / (1024 * 1024),
                'average_entry_size': total_size / len(self._memory_cache) if self._memory_cache else 0
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown cache manager and cleanup resources."""
        logger.info("Shutting down cache manager")
        
        # Stop cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
        
        # Final cleanup
        try:
            self._cleanup_expired()
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")

class OrbitElementsCache(CacheManager):
    """Specialized cache for orbital elements data."""
    
    def __init__(self, cache_dir: str, default_ttl: int = 7200):  # 2 hours default
        super().__init__(cache_dir, default_ttl)
    
    def get_orbital_elements(self, designation: str, source: str) -> Optional[Dict[str, Any]]:
        """Get orbital elements for a specific designation and source."""
        key = f"orbital_{source}_{designation}"
        return self.get(key)
    
    def set_orbital_elements(self, designation: str, source: str, 
                           elements: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set orbital elements for a specific designation and source."""
        key = f"orbital_{source}_{designation}"
        return self.set(key, elements, ttl)
    
    def get_sources_for_designation(self, designation: str) -> Set[str]:
        """Get all available sources for a designation."""
        prefix = f"orbital_"
        suffix = f"_{designation}"
        
        sources = set()
        for key in self.keys():
            if key.startswith(prefix) and key.endswith(suffix):
                # Extract source name
                source = key[len(prefix):-len(suffix)]
                sources.add(source)
        
        return sources

# Global cache instances for backwards compatibility
_global_cache_manager = None
_global_orbital_cache = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        from ..config.settings import get_config
        config = get_config()
        _global_cache_manager = CacheManager(
            cache_dir=config.paths.cache_dir,
            default_ttl=config.cache_ttl
        )
    return _global_cache_manager

def get_orbital_cache() -> OrbitElementsCache:
    """Get global orbital elements cache instance."""
    global _global_orbital_cache
    if _global_orbital_cache is None:
        from ..config.settings import get_config
        config = get_config()
        cache_dir = Path(config.paths.cache_dir) / "orbital_elements"
        _global_orbital_cache = OrbitElementsCache(
            cache_dir=str(cache_dir),
            default_ttl=config.cache_ttl
        )
    return _global_orbital_cache