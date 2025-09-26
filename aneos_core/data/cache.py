"""Thread-safe cache abstraction used by the regression test-suite.

The original project recently grew a much more feature-rich cache layer that
tracked byte sizes, background cleaners, and hybrid JSON/pickle persistence.
While those additions were useful in the production branch, the unit tests in
this repository still target the earlier, simpler API that exposed
``CacheEntry`` (with ``key``/``value`` fields) and ``CacheStats`` dataclasses
as well as a ``CacheManager`` that behaved like a dictionary.

To restore test compatibility—and therefore the ability to execute the suite
as part of the Phase 5 “production readiness” verification—we reintroduce the
legacy surface area while retaining conveniences such as optional disk
persistence, TTL handling, and best-effort LRU eviction.  The implementation is
carefully synchronised via an ``RLock`` so that the thread-safety regression
tests can hammer the cache without tripping race conditions.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, MutableMapping, Optional


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalise naive datetimes to UTC-aware values."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _safe_key(key: str) -> str:
    """Return a filesystem safe cache key."""

    # Limit key length and strip path separators so we can persist on disk when
    # requested.  The simple heuristic mirrors the historical implementation.
    sanitized = key.replace("/", "_").replace("\\", "_")
    if len(sanitized) > 128:
        return sanitized[:128]
    return sanitized


@dataclass
class CacheEntry:
    """Container for cached values with optional expiry metadata."""

    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self, reference_time: Optional[datetime] = None) -> bool:
        """Return ``True`` when the entry has passed its expiry time."""

        if self.expires_at is None:
            return False
        reference = reference_time or _utcnow()
        expires_at = self.expires_at
        return _ensure_utc(reference) >= _ensure_utc(expires_at)

    def touch(self) -> None:
        """Update the access metadata after a successful retrieval."""

        self.access_count += 1
        self.last_accessed = _utcnow()


@dataclass
class CacheStats:
    """Lightweight statistics snapshot for the cache manager."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    disk_reads: int = 0
    disk_writes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 100.0
        return (self.misses / total) * 100


class CacheManager(MutableMapping[str, Any]):
    """Thread-safe cache with optional disk persistence and TTL support."""

    def __init__(
        self,
        cache_dir: str,
        *,
        memory_limit: int = 1000,
        default_ttl: int = 3600,
        disk_cache_enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory_limit = max(1, memory_limit)
        self.default_ttl = default_ttl
        self.disk_cache_enabled = disk_cache_enabled

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()

    # -- MutableMapping interface -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            # Iterate over a snapshot of valid, non-expired keys.
            now = _utcnow()
            return iter(k for k, v in list(self._cache.items()) if not v.is_expired(now))

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return self.size()

    # -- Core operations ----------------------------------------------------------
    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        memory_only: bool = False,
    ) -> None:
        """Store a value in the cache."""

        expires_at = None
        if ttl is None:
            ttl = self.default_ttl
        if ttl > 0:
            expires_at = _utcnow() + timedelta(seconds=ttl)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=_utcnow(),
            expires_at=expires_at,
            last_accessed=_utcnow(),
        )

        with self._lock:
            self._cache[key] = entry
            self._enforce_capacity()
            if self.disk_cache_enabled and not memory_only:
                self._persist_entry(entry)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._cache.get(key)
            now = _utcnow()
            if entry and entry.is_expired(now):
                self._evict(key)
                entry = None

            if entry is None and self.disk_cache_enabled:
                entry = self._load_entry(key)
                if entry:
                    self._cache[key] = entry

            if entry is None:
                self._stats.misses += 1
                return default

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._evict(key, remove_disk=True)

    def size(self) -> int:
        with self._lock:
            now = _utcnow()
            return sum(1 for entry in self._cache.values() if not entry.is_expired(now))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            if self.disk_cache_enabled:
                for file in self.cache_dir.glob("*.cache"):
                    file.unlink(missing_ok=True)

    def cleanup_expired(self) -> int:
        with self._lock:
            now = _utcnow()
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired(now)]
            for key in expired_keys:
                self._evict(key, remove_disk=True)
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = {
                "memory_cache": {
                    "size": self.size(),
                    "limit": self.memory_limit,
                },
                "performance": {
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "hit_rate": self._stats.hit_rate,
                    "miss_rate": self._stats.miss_rate,
                    "evictions": self._stats.evictions,
                    "disk_reads": self._stats.disk_reads,
                    "disk_writes": self._stats.disk_writes,
                },
                "disk_cache": {
                    "enabled": self.disk_cache_enabled,
                    "cache_dir": str(self.cache_dir),
                },
            }
        return stats

    # -- Context manager helpers --------------------------------------------------
    def close(self) -> None:
        """No-op placeholder for API compatibility."""

        # The legacy implementation flushed data to disk when closing.  Our
        # writes are already persisted eagerly, so there is nothing else to do.
        return None

    def __enter__(self) -> "CacheManager":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()

    # -- Internal helpers ---------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{_safe_key(key)}.cache"

    def _persist_entry(self, entry: CacheEntry) -> None:
        cache_path = self._cache_path(entry.key)
        payload = {
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "value": entry.value,
        }

        try:
            # Prefer JSON for simple values and fall back to pickle for complex
            # objects that are not JSON serialisable.
            try:
                serialised = json.dumps(payload)
                cache_path.write_text(serialised)
            except TypeError:
                cache_path.write_bytes(pickle.dumps(payload))
            self._stats.disk_writes += 1
        except OSError:
            # Disk persistence is best effort; ignore write failures but keep
            # the in-memory value.
            pass

    def _load_entry(self, key: str) -> Optional[CacheEntry]:
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return None

        try:
            try:
                payload = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = pickle.loads(cache_path.read_bytes())

            created_at = datetime.fromisoformat(payload["created_at"])
            expires_at_raw = payload.get("expires_at")
            expires_at = datetime.fromisoformat(expires_at_raw) if expires_at_raw else None
            entry = CacheEntry(
                key=key,
                value=payload.get("value"),
                created_at=created_at,
                expires_at=expires_at,
                last_accessed=_utcnow(),
            )
            self._stats.disk_reads += 1
            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                return None
            return entry
        except (OSError, pickle.PickleError, ValueError):
            cache_path.unlink(missing_ok=True)
            return None

    def _enforce_capacity(self) -> None:
        if len(self._cache) <= self.memory_limit:
            return

        # Evict the least recently accessed entry.
        def _sort_key(item: CacheEntry) -> float:
            timestamp = item.last_accessed or item.created_at
            return _ensure_utc(timestamp).timestamp()

        while len(self._cache) > self.memory_limit:
            lru_key = min(self._cache, key=lambda k: _sort_key(self._cache[k]))
            self._evict(lru_key, remove_disk=True)

    def _evict(self, key: str, *, remove_disk: bool = False) -> bool:
        entry = self._cache.pop(key, None)
        if entry is None:
            return False
        if remove_disk and self.disk_cache_enabled:
            cache_path = self._cache_path(key)
            cache_path.unlink(missing_ok=True)
        self._stats.evictions += 1
        return True


_global_cache_manager: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager(cache_dir: Optional[str] = None) -> CacheManager:
    """Return a process-wide cache manager instance."""

    global _global_cache_manager
    with _cache_lock:
        if _global_cache_manager is None:
            resolved_dir = cache_dir or os.path.join(tempfile.gettempdir(), "aneos_cache")
            _global_cache_manager = CacheManager(cache_dir=resolved_dir)
        return _global_cache_manager


def reset_cache_manager() -> None:
    """Dispose of the global cache manager (primarily for tests)."""

    global _global_cache_manager
    with _cache_lock:
        if _global_cache_manager is not None:
            _global_cache_manager.close()
            _global_cache_manager = None


__all__ = [
    "CacheEntry",
    "CacheManager",
    "CacheStats",
    "get_cache_manager",
    "reset_cache_manager",
]

