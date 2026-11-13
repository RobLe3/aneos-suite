"""Read-only SQL freshness inspection and explicit shared-pipeline refresh.

No guessed ORM column names, schema writes during inspection or legacy poller.
The existing acquisition repository owns all refresh writes and historical evidence.
"""

from datetime import datetime, timezone
from sqlalchemy import inspect, select, union
from aneos_core.data.acquisition import plan, SUPPORTED
from aneos_core.data.enrichment import canonical_source, identifier, clone
from aneos_core.data.freshness import source_freshness, POLICY, check_clock
from .database import EnrichedNEO
from .enrichment_store import SourceAssertion, ObjectAcquisition, records, schema_ready
from .acquisition_repository import query_history

LEGACY_COLUMNS = {
    "SBDB": "nasa_sbdb_last_update",
    "NEODyS": "neodys_last_update",
    "MPC": "mpc_last_update",
    "CAD": "nasa_cad_last_update",
}


def _events(history, assertions, source):
    result = []
    # Query receipts bind both positive and explicit-empty/failed attempts.
    for record in history:
        for outcome in record["bundle"]["outcomes"]:
            if outcome["source"] != source:
                continue
            result.append(
                {
                    "status": outcome["status"],
                    "read_at": (
                        outcome["neo_data"].get("fetched_at")
                        if outcome["neo_data"] is not None
                        else (
                            outcome["completed_at"]
                            if outcome["status"] == "missing"
                            else None
                        )
                    ),
                    "attempt_at": outcome["completed_at"],
                    "ledger_recorded_at": record["receipt"]["committed_at"],
                    "reference": {
                        "query_id": record["bundle"]["query_id"],
                        "source": source,
                    },
                }
            )
    # Separate typed non-query intake, never infer freshness for legacy projections.
    # Order cross-stream events by immutable source assertion sequence, not wall clocks.
    by_query = {e["reference"]["query_id"]: e for e in result}
    ordered = []
    seen = set()
    for record in assertions:
        if record["source"] != source:
            continue
        payload = record["payload"]
        query_id = payload.get("acquisition_query_id")
        if query_id in by_query:
            if query_id not in seen:
                ordered.append(by_query[query_id])
                seen.add(query_id)
        elif (
            record["kind"] == "source_assertion"
            and payload.get("snapshot_schema") == "neo-data-v1"
            and "legacy_import_context" not in payload
        ):
            ordered.append(
                {
                    "status": "available",
                    "read_at": payload["neo_data"].get("fetched_at"),
                    "attempt_at": payload["neo_data"].get("fetched_at"),
                    "ledger_recorded_at": record["recorded_at"],
                    "reference": {
                        "assertion_id": record["assertion_id"],
                        "source": source,
                    },
                }
            )
    return ordered


def inspect_freshness(db, priority, *, checked_at=None, limit=50, designations=None):
    """Bound objects, not retained history; incomplete/malformed schemas fail closed."""
    clock = check_clock(
        checked_at if checked_at is not None else datetime.now(timezone.utc)
    )
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Freshness object limit must be 1..50")
    source_plan = plan(priority)
    tables = set(inspect(db.get_bind()).get_table_names())
    has_assertions = SourceAssertion.__tablename__ in tables
    if has_assertions and not schema_ready(db.get_bind()):
        raise ValueError("Evidence schema unavailable or invalid; no freshness claim")
    selects = []
    if EnrichedNEO.__tablename__ in tables:
        selects.append(select(EnrichedNEO.designation))
    if has_assertions:
        selects.append(select(SourceAssertion.designation))
    if ObjectAcquisition.__tablename__ in tables:
        selects.append(select(ObjectAcquisition.designation))
    requested = None
    if designations is not None:
        if not isinstance(designations, list) or len(designations) > 50:
            raise ValueError("At most 50 explicit designations required")
        requested = list(
            dict.fromkeys(identifier(d, "designation") for d in designations)
        )
    names = []
    present = []
    truncated = False
    if selects:
        combined = union(*selects).subquery()
        query = select(combined.c.designation).order_by(combined.c.designation)
        if requested is not None:
            query = query.where(combined.c.designation.in_(requested))
        # Explicit selections already have a hard 50-name bound. Check their
        # presence separately from the smaller review limit: not reviewed != absent.
        present = list(
            db.scalars(query.limit(50 if requested is not None else limit + 1))
        )
        truncated = len(present) > limit
        names = present[:limit]
    objects = []
    for name in names:
        history = query_history(db, name)
        assertions = records(db, name) if has_assertions else []
        row = (
            db.scalar(select(EnrichedNEO).where(EnrichedNEO.designation == name))
            if EnrichedNEO.__tablename__ in tables
            else None
        )
        sources = []
        for source in source_plan["sources"]:
            raw = (
                getattr(row, LEGACY_COLUMNS.get(source, ""), None)
                if row is not None
                else None
            )
            sources.append(
                source_freshness(
                    source,
                    _events(history, assertions, source),
                    checked_at=clock,
                    legacy_timestamp=raw.isoformat() if raw is not None else None,
                )
            )
        due = [
            s["source"]
            for s in sources
            if s["needs_review"] and s["source"] in SUPPORTED
        ]
        # A refresh uses the entire fixed plan. A blocked source blocks that object;
        # do not bypass cooldown or silently shrink its source denominator.
        blocked = [
            s["source"]
            for s in sources
            if s["source"] in SUPPORTED
            and s["needs_review"]
            and not s["retry_eligible"]
        ]
        objects.append(
            {
                "designation": name,
                "sources": sources,
                "missing_sources": due,
                "needs_review": bool(due),
                "retry_eligible": bool(due) and not blocked,
                "blocked_sources": blocked,
                "refresh_scope": "whole_fixed_source_plan_not_selective_backfill",
            }
        )
    outdated = [o for o in objects if o["needs_review"]]
    return clone(
        {
            "schema_version": "object-source-freshness-v1",
            "checked_at": clock.isoformat(),
            "policy": POLICY,
            "source_plan": source_plan,
            "objects": objects,
            "checked_count": len(objects),
            "outdated_count": len(outdated),
            "outdated_objects": outdated,
            "eligible_count": sum(o["retry_eligible"] for o in outdated),
            "truncated": truncated,
            "object_limit": limit,
            "requested_designations_not_found": [
                d for d in requested or [] if d not in present
            ],
            "requested_designations_not_reviewed": [
                d for d in requested or [] if d in present and d not in names
            ],
            "limits": "Read-only bounded object review of replay-checked stored adapter receipts. Unknown legacy clocks are not UTC. No live health probe, scientific completeness, alias equivalence or automatic selective backfill.",
        }
    )


def refresh_objects(engine, designations, priority, *, checked_at=None):
    """Explicit bounded whole-plan refresh; recheck SQL, commit proof before success.

    No selection supplied by a menu can bypass cooldown. The shared fetcher/transport
    owns live admission, initialization failures, pacing and source outcome capture.
    """
    from sqlalchemy.orm import sessionmaker
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.utils.errors import DataSourceUnavailableError
    from .acquisition_repository import SQLEvidenceRepository

    if not isinstance(designations, list) or not 1 <= len(designations) <= 10:
        raise ValueError("Explicit refresh requires 1..10 designations")
    sessions = sessionmaker(bind=engine)
    with sessions() as db:
        report = inspect_freshness(
            db, priority, checked_at=checked_at, designations=designations
        )
    repository = SQLEvidenceRepository(engine)
    results = []
    fetcher = None
    try:
        for obj in report["objects"]:
            name = obj["designation"]
            # Re-evaluate immediately before each object's query, not merely at
            # the beginning of a multi-object menu batch. This is not a network lock.
            with sessions() as db:
                current = inspect_freshness(
                    db, priority, checked_at=checked_at, designations=[name]
                )
                prior_ids = {r["bundle"]["query_id"] for r in query_history(db, name)}
            obj = current["objects"][0]
            if not obj["retry_eligible"]:
                results.append(
                    {
                        "designation": name,
                        "status": "deferred",
                        "blocked_sources": obj["blocked_sources"],
                    }
                )
                continue
            try:
                if fetcher is None:
                    fetcher = DataFetcher(
                        evidence_repository=repository, source_priority=priority
                    )
                try:
                    neo = fetcher.fetch_neo_data(name, force_refresh=True)
                    receipt = neo.evidence_metadata["acquisition"]["storage_receipt"]
                except DataSourceUnavailableError as exc:
                    receipt = getattr(exc, "acquisition_receipt", None)
                    if not isinstance(receipt, dict):
                        raise ValueError("No committed acquisition receipt") from exc
                if receipt.get("commit_status") != "committed":
                    raise ValueError("Uncommitted refresh")
                saved = repository.load_acquisition(receipt["query_id"])
                if (
                    saved is None
                    or receipt["query_id"] in prior_ids
                    or saved["receipt"] != receipt
                    or saved["bundle"]["designation"] != name
                ):
                    raise ValueError("Refresh receipt does not replay")
                coverage = saved["bundle"]["coverage"]
                count = coverage["contributing_sources"]
                total = coverage["planned_sources"]
                results.append(
                    {
                        "designation": name,
                        "status": (
                            "recorded_full_plan"
                            if count == total
                            else "recorded_partial" if count else "recorded_no_data"
                        ),
                        "query_id": receipt["query_id"],
                        "coverage": coverage,
                        "limits": "Committed adapter query, not proof of a new wire observation or scientific completeness.",
                    }
                )
            except Exception:
                # Never display arbitrary exception/provider text (may contain secrets).
                results.append(
                    {
                        "designation": name,
                        "status": "failed",
                        "reason_code": "REFRESH_NOT_VERIFIED",
                    }
                )
    finally:
        cleanup_status = "not_required"
        if fetcher is not None:
            try:
                fetcher.close()
                cleanup_status = "closed"
            except Exception:
                cleanup_status = "failed"
    return {
        "cleanup_status": cleanup_status,
        "schema_version": "explicit-object-refresh-v1",
        "inspection": report,
        "results": results,
        "recorded_count": sum(r["status"].startswith("recorded_") for r in results),
        "limits": "Explicit whole-plan refresh of at most ten existing objects, not an automatic selective backfill scheduler. Provider transport cooldowns still apply.",
    }
