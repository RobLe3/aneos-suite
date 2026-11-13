"""Receipt-age review, not astronomical recency or proof of a new network read.

Pure deterministic policy. Caller supplies replay-checked records in append order.
The original adapter read is separate from local ingestion/cache rehydration.
"""

from datetime import datetime, timedelta, timezone
import re
from .acquisition import SUPPORTED
from .enrichment import canonical_source, clone

POLICY = {
    "schema_version": "source-freshness-policy-v1",
    "adapter_read_max_age_seconds": 30 * 86400,
    "failed_attempt_cooldown_seconds": 3600,
    "scope": "installed_object_adapters_not_all_observations",
}


def instant(value):
    """Only explicit offset ISO instants, <= microsecond precision; no UTC guessing."""
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value
    ):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except (ValueError, OverflowError):
        return None


def check_clock(value):
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("Freshness review requires an explicit offset-aware clock")
    return value.astimezone(timezone.utc)


def source_freshness(source, events, *, checked_at, legacy_timestamp=None):
    """Latest append determines latest attempt; never sort by a possibly bad clock."""
    clock = check_clock(checked_at)
    source = canonical_source(source)
    events = clone(events)
    latest = events[-1] if events else {}
    good = [e for e in events if e.get("status") in ("available", "missing")]
    last_good = good[-1] if good else {}
    read_at = last_good.get("read_at")
    read = instant(read_at)
    age = (clock - read).total_seconds() if read is not None else None
    freshness = (
        "unknown"
        if age is None
        else (
            "clock_ahead"
            if age < 0
            else (
                "recent_adapter_read"
                if age <= POLICY["adapter_read_max_age_seconds"]
                else "stale_adapter_read"
            )
        )
    )
    bad_clock = any(instant(e.get("attempt_at")) is None for e in events)
    valid_clocks = [instant(e.get("attempt_at")) for e in events]
    regression = any(
        b < a
        for a, b in zip(valid_clocks, valid_clocks[1:])
        if a is not None and b is not None
    )
    latest_status = latest.get("status", "not_recorded")
    needs_review = freshness != "recent_adapter_read" or latest_status not in (
        "available",
        "missing",
    )
    attempt = instant(latest.get("attempt_at"))
    retry_after = None
    eligible = source in SUPPORTED and needs_review
    reason = "review_due" if eligible else "recent_adapter_read"
    if source not in SUPPORTED:
        reason = "unsupported_source"
    elif eligible and events and latest_status not in ("available", "missing"):
        if attempt is None:
            eligible = False
            reason = "unverified_attempt_clock"
        else:
            # Near year 9999 the derived retry label can exceed datetime's range.
            try:
                retry = attempt + timedelta(
                    seconds=POLICY["failed_attempt_cooldown_seconds"]
                )
            except OverflowError:
                retry = None
            if retry is None or clock < retry:
                eligible = False
                reason = (
                    "attempt_clock_ahead"
                    if attempt > clock
                    else "failed_attempt_cooldown"
                )
                retry_after = retry.isoformat() if retry is not None else None
    return {
        "source": source,
        "latest_outcome": latest_status,
        "freshness": freshness,
        "needs_review": needs_review,
        "last_adapter_read_at": read_at,
        "adapter_read_age_seconds": age,
        "last_attempt_at": latest.get("attempt_at"),
        "last_attempt_reference": latest.get("reference"),
        "last_usable_reference": last_good.get("reference"),
        "last_ledger_recorded_at": latest.get("ledger_recorded_at"),
        "retry_eligible": eligible,
        "retry_reason": reason,
        "retry_after": retry_after,
        "clock_regression": regression,
        "unverified_attempt_clock": bad_clock,
        "legacy_timestamp": legacy_timestamp,
        "limits": "Adapter read/empty response age, not wire freshness, observation epoch, provider health, source accuracy or scientific completeness. Runtime transport admission/cooldown still applies.",
    }
