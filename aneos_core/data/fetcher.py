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
        cache_ttl: int = 3600,
        evidence_repository=None,
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
        self.evidence_repository = evidence_repository
        from ..config.settings import get_config

        _cfg = get_config()

        # Initialize cache manager
        from pathlib import Path

        cache_dir = Path(_cfg.directories.cache_dir)
        self.cache_manager = (
            cache_manager if cache_manager is not None else CacheManager(cache_dir)
        )
        self.cache_ttl = cache_ttl

        # Thread pool for concurrent operations
        self.max_workers = max_workers

        # Initialize data sources — each source is guarded so a single
        # broken source does not prevent the others from loading.
        self.sources: Dict[str, DataSourceBase] = {}
        _source_defs = [
            ("SBDB", lambda: SBDBSource(_cfg.api, self.cache_manager)),
            ("NEODyS", lambda: NEODySSource(_cfg.api, self.cache_manager)),
            ("MPC", lambda: MPCSource(_cfg.api, self.cache_manager)),
            ("Horizons", lambda: HorizonsSource(_cfg.api, self.cache_manager)),
        ]
        from .acquisition import plan

        configured_priority = (
            source_priority
            if source_priority is not None
            else list(_cfg.api.data_sources_priority)
        )
        # Stable default public-source cache scope, isolated by working directory.
        # Custom endpoint settings get an instance nonce: no URL secrets are hashed
        # into/exported through the query record or reused across configurations.
        from ..config.settings import APIConfig
        from .enrichment import fingerprint
        from uuid import uuid4

        defaults = APIConfig()
        endpoints = (
            "sbdb_url",
            "neodys_url",
            "neodys_rest_url",
            "mpc_url",
            "horizons_url",
            "cad_url",
        )
        default_endpoints = all(
            getattr(_cfg.api, key) == getattr(defaults, key) for key in endpoints
        )
        scope = {
            "workspace": str(Path.cwd().resolve()),
            "adapter_contract_revision": "object-acquisition-v2-neodys-oef2-photometry",
            "endpoint_scope": (
                "default-public-v1" if default_endpoints else str(uuid4())
            ),
        }
        self.source_plan = plan(configured_priority, context_id=fingerprint(scope))
        self.source_priority = self.source_plan["source_priority"]
        self.source_initialization = {"CAD": "deferred"}
        for _name, _factory in _source_defs:
            try:
                self.sources[_name] = _factory()
                self.source_initialization[_name] = "available"
            except Exception as _exc:
                self.source_initialization[_name] = "unavailable"
                self.logger.warning("Data source %s initialization unavailable", _name)

        # Statistics tracking
        self.fetch_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "source_successes": {source: 0 for source in self.sources},
            "source_failures": {source: 0 for source in self.sources},
        }

    def fetch_neo_data(self, designation: str, force_refresh: bool = False) -> NEOData:
        """Acquire/cache a frozen query and commit it locally before returning evidence.

        Cache hits reuse the original query/receipt, not fictitious new observations.
        A required persistence failure is distinct from provider unavailability.
        """
        from .acquisition import validate_bundle, result_from_bundle
        from .enrichment import identifier
        from ..utils.errors import EvidencePersistenceError

        designation = identifier(designation, "designation")
        source_plan = self._query_plan()
        try:
            repository = getattr(self, "evidence_repository", None)
            if repository is None:
                # Compatibility adapter uses the existing project SQLAlchemy Base;
                # importing the core/constructing a fetcher performs no database I/O.
                from aneos_api.acquisition_repository import SQLEvidenceRepository

                repository = SQLEvidenceRepository()
        except Exception as exc:
            raise EvidencePersistenceError(
                "Local evidence storage prerequisite unavailable; sources not queried"
            ) from exc
        self.fetch_stats["total_requests"] += 1
        cache_key = (
            f"neo-query:object-acquisition-v1:{source_plan['plan_id']}:{designation}"
        )
        bundle = None
        if not force_refresh:
            try:
                cached = self.cache_manager.get(cache_key)
            except Exception:
                self.fetch_stats["cache_errors"] = (
                    self.fetch_stats.get("cache_errors", 0) + 1
                )
                self.logger.warning("Acquisition cache read unavailable; using sources")
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == "object-query-cache-v1"
            ):
                try:
                    candidate = validate_bundle(cached["bundle"])
                    if (
                        candidate["designation"] != designation
                        or candidate["source_plan"] != source_plan
                    ):
                        raise ValueError("Cache query binding mismatch")
                    bundle = candidate
                except (ValueError, TypeError, KeyError):
                    self.logger.warning(
                        "Ignoring invalid acquisition cache for requested designation"
                    )
        cache_hit = bundle is not None
        if cache_hit:
            self.fetch_stats["cache_hits"] += 1
        else:
            self.fetch_stats["cache_misses"] += 1
            try:
                admission = repository.prepare_acquisition()
                if admission.get("status") != "ready":
                    raise ValueError("Local evidence schema not admitted")
            except Exception as exc:
                raise EvidencePersistenceError(
                    "Local evidence storage prerequisite unavailable; sources not queried"
                ) from exc
            try:
                acquired = self._fetch_from_all_sources(
                    designation, source_plan=source_plan
                )
                bundle = acquired._acquisition_bundle
            except DataSourceUnavailableError as exc:
                bundle = getattr(exc, "acquisition_bundle", None)
                if bundle is None:
                    raise
        bundle = validate_bundle(bundle)
        try:
            receipt = repository.commit_acquisition(bundle)
            if (
                receipt.get("commit_status") != "committed"
                or receipt.get("query_id") != bundle["query_id"]
                or receipt.get("bundle_id") != bundle["bundle_id"]
            ):
                raise ValueError("Acquisition persistence receipt mismatch")
        except Exception as exc:
            failure = EvidencePersistenceError(
                "Acquired evidence could not be committed locally; no scientific result returned"
            )
            failure.acquisition_bundle = bundle
            raise failure from exc
        if bundle["neo_data"] is None:
            failure = DataSourceUnavailableError(
                f"No usable source data; acquisition query {bundle['query_id']} recorded locally"
            )
            failure.acquisition_receipt = receipt
            failure.acquisition_bundle = bundle
            raise failure
        # Only committed, replay-verified query envelopes may enter the cache.
        if not cache_hit:
            try:
                self.cache_manager.set(
                    cache_key,
                    {"schema_version": "object-query-cache-v1", "bundle": bundle},
                    ttl=self.cache_ttl,
                )
            except Exception:
                self.fetch_stats["cache_errors"] = (
                    self.fetch_stats.get("cache_errors", 0) + 1
                )
                self.logger.warning(
                    "Acquisition cache write unavailable; committed evidence remains available"
                )
        return result_from_bundle(bundle, receipt)

    def _query_plan(self):
        from .acquisition import plan
        from .enrichment import clone

        record = clone(
            getattr(self, "source_plan", None) or plan(list(self.source_priority))
        )
        if record != plan(
            record["source_priority"], context_id=record.get("context_id")
        ):
            raise ValueError("Configured source plan is invalid")
        return record

    def _fetch_from_all_sources(self, designation: str, *, source_plan=None) -> NEOData:
        """Collect every fixed planned outcome, retaining partial/failed sources."""
        from .acquisition import (
            SUPPORTED,
            now,
            outcome,
            build_bundle,
            validate_outcome,
            accepted_designations,
            designation_key,
        )
        from .sources.cad import CAD_VERSION
        from ..utils.errors import SourcePayloadError

        source_plan = source_plan if source_plan is not None else self._query_plan()
        started = now()
        outcomes = {}
        pending_sources = [s for s in source_plan["sources"] if s != "CAD"]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for name in pending_sources:
                if name not in SUPPORTED:
                    outcomes[name] = outcome(
                        name,
                        designation,
                        status="not_supported",
                        reason_code="UNSUPPORTED_CONFIGURED_SOURCE",
                        basis="source_plan",
                    )
                elif name not in self.sources:
                    outcomes[name] = outcome(
                        name,
                        designation,
                        status="unavailable",
                        reason_code="SOURCE_INITIALIZATION_FAILED",
                        basis="source_initialization",
                    )
                else:
                    futures[
                        executor.submit(
                            self._read_source, name, self.sources[name], designation
                        )
                    ] = name
            for future in as_completed(futures):
                name = futures[future]
                value = None
                try:
                    value = future.result()
                    # Binding and finite/typed payload checks occur before any commit.
                    if value["source"] != name or value["designation"] != designation:
                        raise ValueError("Source outcome mismatch")
                    outcomes[name] = validate_outcome(value, name, designation)
                except Exception:
                    from .enrichment import clone

                    try:
                        packet = (
                            clone(value.get("adapter_payload"))
                            if isinstance(value, dict)
                            else None
                        )
                    except Exception:
                        packet = None
                    outcomes[name] = outcome(
                        name,
                        designation,
                        status="invalid",
                        adapter_payload=packet,
                        reason_code="SOURCE_OUTCOME_INVALID",
                    )
        cad_started = now()
        try:
            approaches = self._fetch_close_approaches(designation)
            metadata = getattr(approaches, "query_metadata", None)
            if (
                not isinstance(metadata, dict)
                or metadata.get("api_version") != CAD_VERSION
            ):
                outcomes["CAD"] = outcome(
                    "CAD",
                    designation,
                    status="unavailable",
                    started_at=cad_started,
                    reason_code="CAD_SCHEMA_UNVERIFIED",
                    basis="legacy_adapter_outcome_unknown",
                )
            elif metadata.get("count") != len(approaches):
                outcomes["CAD"] = outcome(
                    "CAD",
                    designation,
                    status="invalid",
                    started_at=cad_started,
                    adapter_payload={"query_metadata": metadata},
                    reason_code="CAD_RECORD_COUNT_MISMATCH",
                )
            elif approaches and any(
                designation_key(event.designation)
                not in accepted_designations(designation, list(outcomes.values()))
                for event in approaches
            ):
                outcomes["CAD"] = outcome(
                    "CAD",
                    designation,
                    status="ambiguous",
                    started_at=cad_started,
                    adapter_payload={
                        "query_metadata": metadata,
                        "close_approaches": [event.to_dict() for event in approaches],
                    },
                    reason_code="CAD_IDENTITY_ALIAS_UNRESOLVED",
                )
            elif approaches:
                stamp = now()
                neo = NEOData(
                    designation,
                    close_approaches=list(approaches),
                    sources_used=["CAD"],
                    evidence_metadata={"close_approach_query": metadata},
                    fetched_at=datetime.fromisoformat(stamp),
                    created_at=datetime.fromisoformat(stamp),
                    updated_at=datetime.fromisoformat(stamp),
                )
                outcomes["CAD"] = outcome(
                    "CAD",
                    designation,
                    status="available",
                    neo=neo,
                    started_at=cad_started,
                    completed_at=stamp,
                    basis="computed_encounters_not_observations",
                )
            else:
                outcomes["CAD"] = outcome(
                    "CAD",
                    designation,
                    status="missing",
                    adapter_payload={"query_metadata": metadata},
                    started_at=cad_started,
                    reason_code="VALIDATED_EMPTY_ENCOUNTER_QUERY",
                    basis="computed_encounters_not_observations",
                )
        except SourcePayloadError as exc:
            from .enrichment import clone

            try:
                packet = clone(
                    {"_native_response": exc.native_response, "query": exc.query}
                )
            except (ValueError, TypeError):
                packet = None
            outcomes["CAD"] = outcome(
                "CAD",
                designation,
                status="invalid",
                adapter_payload=packet,
                started_at=cad_started,
                reason_code="CAD_RESPONSE_NOT_ADMITTED",
            )
        except Exception:
            outcomes["CAD"] = outcome(
                "CAD",
                designation,
                status="unavailable",
                started_at=cad_started,
                reason_code="CAD_QUERY_UNAVAILABLE",
            )
        bundle = build_bundle(
            designation, source_plan, list(outcomes.values()), started_at=started
        )
        for name, value in outcomes.items():
            self.fetch_stats.setdefault("source_successes", {}).setdefault(name, 0)
            self.fetch_stats.setdefault("source_failures", {}).setdefault(name, 0)
            counter = (
                "source_successes"
                if value["status"] in ("available", "missing")
                else "source_failures"
            )
            self.fetch_stats[counter][name] += 1
        if bundle["neo_data"] is None:
            failure = DataSourceUnavailableError(
                "No usable source data in the fixed acquisition plan"
            )
            failure.acquisition_bundle = bundle
            raise failure
        neo = NEOData.from_dict(bundle["neo_data"])
        neo._acquisition_bundle = (
            bundle  # Internal owned envelope; not recursively serialized into evidence.
        )
        return neo

    def _reconcile_sources(self, results: Dict[str, NEOData]) -> NEOData:
        """Use the shared atomic source projection, including physical-only inputs."""
        from .reconciliation import reconcile_sources

        return reconcile_sources(results, self.source_priority)

    def _fetch_close_approaches(
        self, designation: str, date_min: str = "now", date_max: Optional[str] = None
    ) -> List:
        """A bounded CAD query, with native TDB evidence and explicit source failure."""
        import requests
        from .sources.cad import (
            parse_cad_payload,
            close_approach_from_record,
            CloseApproachQuery,
        )
        from .sources.base import _APPLICATION_HEADERS

        query = {
            "des": designation,
            "date-min": date_min,
            "dist-max": "0.2",
            "body": "Earth",
            "limit": 10,
            "fullname": "true",
        }
        if date_max is not None:
            query["date-max"] = date_max
        from ..utils.errors import SourcePayloadError

        response = None
        native = None
        try:
            response = provider_get(
                "https://ssd-api.jpl.nasa.gov/cad.api",
                params=query,
                headers=_APPLICATION_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            try:
                native = response.json()
                parsed = parse_cad_payload(
                    native,
                    query=query,
                    require_complete=False,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise SourcePayloadError(
                    "CAD response schema/content not admitted",
                    native_response=native,
                    query=query,
                ) from exc
            return CloseApproachQuery(
                [close_approach_from_record(r) for r in parsed["records"]],
                parsed["metadata"],
            )
        except SourcePayloadError:
            raise
        except Exception as exc:
            raise DataSourceUnavailableError("CAD query unavailable") from exc
        finally:
            if response is not None and callable(getattr(response, "close", None)):
                response.close()

    def fetch_approaches_between(
        self, designation: str, date_min: str, date_max: str
    ) -> List:
        """Explicit frozen CAD calendar window, bounded rows and visible coverage.

        Dates are TDB query calendar labels, not converted observation timestamps.
        Does not reuse a relative-horizon cache whose window can drift on retry.
        """
        from datetime import date

        if (
            not isinstance(date_min, str)
            or not isinstance(date_max, str)
            or date.fromisoformat(date_min).isoformat() != date_min
            or date.fromisoformat(date_max).isoformat() != date_max
            or date_min > date_max
        ):
            raise ValueError("Expected ordered ISO calendar query bounds")
        return self._fetch_close_approaches(
            designation, date_min=date_min, date_max=date_max
        )

    def fetch_historical_approaches(
        self, designation: str, years_back: int = 30
    ) -> List:
        """Bounded historical CAD query, not a complete observation time series."""
        from datetime import timedelta
        from .models import CloseApproach
        from .sources.cad import CloseApproachQuery

        key = f"cad_v1_hist:{designation}:{years_back}"
        cached = self.cache_manager.get(key)
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == "cad-approaches-v1"
        ):
            return CloseApproachQuery(
                [CloseApproach.from_dict(r) for r in cached["approaches"]],
                cached["query_metadata"],
            )
        date_min = (
            datetime.now(timezone.utc) - timedelta(days=years_back * 365)
        ).strftime("%Y-%m-%d")
        approaches = self._fetch_close_approaches(designation, date_min=date_min)
        self.cache_manager.set(
            key,
            {
                "schema_version": "cad-approaches-v1",
                "approaches": [ca.to_dict() for ca in approaches],
                "query_metadata": approaches.query_metadata,
            },
            ttl=604800,
        )
        return approaches

    def _read_source(self, source_name, source, designation):
        """One adapter call; preserve explicit or unclassified outcome without guessing."""
        import asyncio, inspect, json
        from .acquisition import now, outcome, adapter_to_neo, has_material, STATUSES

        started = now()
        packet = None

        def freeze(value):
            def default(item):
                if isinstance(item, datetime):
                    return item.isoformat()
                raise TypeError("Unsupported native adapter value")

            return json.loads(json.dumps(value, allow_nan=False, default=default))

        try:
            read = (
                source.fetch_acquisition
                if callable(getattr(type(source), "fetch_acquisition", None))
                else source.fetch_orbital_elements
            )
            raw = read(designation)
            if inspect.isawaitable(raw):
                raw = asyncio.run(raw)
            if hasattr(raw, "success"):
                data = raw.data
                native = getattr(raw, "raw_response", None)
                if not raw.success:
                    status = getattr(raw, "outcome", None)
                    status = (
                        status
                        if status in STATUSES and status != "available"
                        else "unavailable"
                    )
                    packet = (
                        freeze({"_native_response": native})
                        if native is not None
                        else None
                    )
                    code = getattr(raw, "error_code", None)
                    import re

                    code = (
                        code
                        if isinstance(code, str)
                        and re.fullmatch("[A-Z0-9_]{1,80}", code)
                        else "ADAPTER_UNSUCCESSFUL"
                    )
                    return outcome(
                        source_name,
                        designation,
                        status=status,
                        adapter_payload=packet,
                        started_at=started,
                        reason_code=code,
                        basis=(
                            "explicit_adapter_outcome"
                            if getattr(raw, "outcome", None)
                            else "legacy_adapter_outcome_unknown"
                        ),
                    )
                packet = freeze(data)
                if native is not None:
                    packet["_native_response"] = freeze(native)
            else:
                if raw is None:
                    return outcome(
                        source_name,
                        designation,
                        status="unavailable",
                        started_at=started,
                        reason_code="ADAPTER_NO_RESULT_UNCLASSIFIED",
                        basis="legacy_adapter_outcome_unknown",
                    )
                packet = freeze(raw)
            stamp = now()
            neo = adapter_to_neo(source_name, designation, packet, stamp)
            if not has_material(neo):
                return outcome(
                    source_name,
                    designation,
                    status="invalid",
                    adapter_payload=packet,
                    started_at=started,
                    completed_at=stamp,
                    reason_code="NO_INTERPRETABLE_SOURCE_FIELDS",
                )
            return outcome(
                source_name,
                designation,
                status="available",
                neo=neo,
                adapter_payload=packet,
                started_at=started,
                completed_at=stamp,
                basis="typed_adapter_projection_not_independently_verified",
            )
        except (ValueError, TypeError, KeyError):
            return outcome(
                source_name,
                designation,
                status="invalid",
                adapter_payload=packet,
                started_at=started,
                reason_code="INVALID_SOURCE_PAYLOAD",
            )
        except Exception:
            return outcome(
                source_name,
                designation,
                status="unavailable",
                adapter_payload=packet,
                started_at=started,
                reason_code="SOURCE_QUERY_UNAVAILABLE",
            )

    def _fetch_from_source(
        self, source_name: str, source: DataSourceBase, designation: str
    ) -> Optional[NEOData]:
        """Compatibility typed projection; detailed outcomes are retained by public queries."""
        record = self._read_source(source_name, source, designation)
        return (
            NEOData.from_dict(record["neo_data"])
            if record["status"] == "available"
            else None
        )

    def fetch_multiple(
        self, designations: List[str], force_refresh: bool = False
    ) -> Dict[str, Optional[NEOData]]:
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
                executor.submit(
                    self.fetch_neo_data, designation, force_refresh
                ): designation
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
                    "last_check": (
                        status.last_check.isoformat() if status.last_check else None
                    ),
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
                    "error": str(e),
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
            self.fetch_stats["source_successes"][source]
            + self.fetch_stats["source_failures"][source]
            for source in self.sources
        )

        return {
            "total_requests": self.fetch_stats["total_requests"],
            "cache_hits": self.fetch_stats["cache_hits"],
            "cache_misses": self.fetch_stats["cache_misses"],
            "cache_hit_rate": (
                self.fetch_stats["cache_hits"]
                / self.fetch_stats["total_requests"]
                * 100
                if self.fetch_stats["total_requests"] > 0
                else 0
            ),
            "source_statistics": {
                source: {
                    "successes": self.fetch_stats["source_successes"][source],
                    "failures": self.fetch_stats["source_failures"][source],
                    "success_rate": (
                        self.fetch_stats["source_successes"][source]
                        / (
                            self.fetch_stats["source_successes"][source]
                            + self.fetch_stats["source_failures"][source]
                        )
                        * 100
                        if (
                            self.fetch_stats["source_successes"][source]
                            + self.fetch_stats["source_failures"][source]
                        )
                        > 0
                        else 0
                    ),
                }
                for source in self.sources
            },
            "cache_statistics": cache_stats,
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
            "source_failures": {source: 0 for source in self.sources},
        }

        # Reset source statistics (method is optional on each source)
        for source in self.sources.values():
            reset = getattr(source, "reset_metrics", None)
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
