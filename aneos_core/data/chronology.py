"""Replayable date roles; calendar labels and epochs are never invented UTC instants."""

from datetime import date, datetime
from hashlib import sha256
import json
import math
import re

SCHEMA = "object-chronology-v1"
DISCOVERY_SCHEMA = "sbdb-discovery-evidence-v1"
MONTHS = {
    name: n
    for n, name in enumerate(
        (
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ),
        1,
    )
}


def clone(value):
    return json.loads(json.dumps(value, allow_nan=False))


def fingerprint(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _text(value, label, limit=1000):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("Invalid " + label)
    return value


def build_sbdb_discovery(native, context):
    if not isinstance(context, dict):
        raise ValueError("Invalid discovery context")
    _text(context.get("requested_designation"), "discovery designation", 100)
    if not re.fullmatch("[0-9a-f]{64}", str(context.get("source_response_sha256", ""))):
        raise ValueError("Missing discovery response fingerprint")
    status = "not_available"
    normalized = None
    issues = []
    if native is not None:
        if not isinstance(native, dict):
            status = "invalid"
            issues.append("unsupported_discovery_section")
        elif native.get("date") not in (None, ""):
            raw = native["date"]
            try:
                if not isinstance(raw, str) or not re.fullmatch(
                    r"[0-9]{4}-[A-Za-z]{3}-[0-9]{2}", raw
                ):
                    raise ValueError("Unknown date layout")
                year, month, day = raw.split("-")
                value = date(int(year), MONTHS[month], int(day)).isoformat()
                normalized = {
                    "calendar_date": value,
                    "precision": "day",
                    "time_scale": "not_declared",
                    "role": "discovery",
                }
                status = "available"
            except (KeyError, ValueError, TypeError):
                status = "invalid"
                issues.append("invalid_or_unsupported_discovery_date")
    record = {
        "schema_version": DISCOVERY_SCHEMA,
        "source": "SBDB",
        "adapter": "sbdb-discovery-1.3",
        "native": clone(native),
        "context": clone(context),
        "normalized": normalized,
        "status": status,
        "issues": issues,
        "evidence_basis": "catalogued_discovery_circumstances_not_independently_verified",
        "limitations": [
            "SBDB documents discovery circumstances only for numbered asteroids and numbered comets.",
            "A day label is not a UTC instant, an observation arc endpoint or a local storage receipt.",
        ],
    }
    record["assertion_id"] = fingerprint(record)
    return record


def verify_discovery(record):
    if not isinstance(record, dict) or record.get("schema_version") != DISCOVERY_SCHEMA:
        raise ValueError("Unsupported discovery evidence")
    try:
        actual = build_sbdb_discovery(record["native"], record["context"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid discovery evidence") from exc
    if actual != record:
        raise ValueError("Discovery assertion does not reproduce")
    return actual


def verify_neo_discovery(neo):
    meta = neo.evidence_metadata
    candidates = []
    if meta.get("discovery_evidence") is not None:
        candidates.append(meta["discovery_evidence"])
    for source, metadata in (meta.get("source_solutions") or {}).items():
        if (
            isinstance(metadata, dict)
            and metadata.get("discovery_evidence") is not None
        ):
            candidates.append(metadata["discovery_evidence"])
    for item in candidates:
        e = verify_discovery(item)
        if e["context"]["requested_designation"] != neo.designation:
            raise ValueError("Discovery assertion/object mismatch")
    return True


def source_input(snapshot, source, reference, *, metadata=None):
    """Compact display inputs, bound by references to the original frozen snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("Expected source snapshot")
    from .enrichment import canonical_source

    source = canonical_source(source)
    meta = snapshot.get("evidence_metadata") or {} if metadata is None else metadata
    oe = snapshot.get("orbital_elements") or {}
    ng = snapshot.get("nongrav") or {}
    solution = meta.get("solution") or {}
    cov = ng.get("covariance") or meta.get("covariance") or {}
    covariance_reference = reference + (
        "/nongrav/covariance/epoch_jd_tdb"
        if ng.get("covariance")
        else "/evidence_metadata/covariance/epoch_jd_tdb"
    )
    return clone(
        {
            "source": source,
            "reference": reference,
            "designation": snapshot["designation"],
            "first_observation": snapshot.get("first_observation"),
            "last_observation": snapshot.get("last_observation"),
            "fetched_at": snapshot.get("fetched_at"),
            "native_observation_window": meta.get("native_observation_window") or {},
            "native_orbit_epoch": oe.get("native_epoch"),
            "legacy_orbit_epoch": oe.get("epoch"),
            "covariance_epoch_jd_tdb": cov.get("epoch_jd_tdb"),
            "covariance_reference": covariance_reference,
            "orbit_fit_label": solution.get("solution_date"),
            "orbit_fit_schema": solution.get("schema_version"),
            "discovery_evidence": meta.get("discovery_evidence"),
        }
    )


def event(
    role, value, source, reference, representation, scale, precision, status, meaning
):
    return {
        "role": role,
        "value": clone(value),
        "source": source,
        "reference": reference,
        "representation": representation,
        "time_scale": scale,
        "precision": precision,
        "status": status,
        "meaning": meaning,
    }


def timestamp_event(role, value, source, reference, *, ledger=False):
    status = "not_available"
    scale = "not_declared"
    if value is not None:
        status = "invalid"
        try:
            _text(value, "timestamp", 80)
            if not re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T", value):
                raise ValueError("Not an ISO timestamp")
            if ledger and not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})",
                value,
            ):
                raise ValueError("Unsupported ledger timestamp precision/layout")
            d = datetime.fromisoformat(value.replace("Z", "+00:00"))
            offset = d.utcoffset()
            if offset is None:
                status = "time_scale_unverified"
            elif offset.total_seconds() == 0:
                status = "explicit_utc_offset"
                scale = "UTC"
            elif not ledger:
                status = "explicit_offset"
                scale = "explicit_offset"
        except (ValueError, TypeError, OverflowError):
            pass
    return event(
        role,
        value,
        source,
        reference,
        "ISO timestamp",
        scale,
        "literal timestamp precision",
        status,
        (
            "Local append receipt; not discovery or observation time. Ledger order is sequence order."
            if ledger
            else "Adapter receipt; not an observation, discovery or orbit epoch. An offset does not certify clock accuracy."
        ),
    )


def observation_event(role, value, source, reference):
    status = "not_available"
    precision = "not_established"
    representation = "native calendar label"
    if value is not None:
        status = "invalid"
        if isinstance(value, str) and re.fullmatch(
            r"[0-9]{4}-(?:[0-9]{2}|\?\?)-(?:[0-9]{2}|\?\?)", value
        ):
            year, month, day = value.split("-")
            try:
                if int(year) == 0:
                    raise ValueError("Unsupported year")
                if month != "??" and not 1 <= int(month) <= 12:
                    raise ValueError("Invalid month")
                if day != "??" and not 1 <= int(day) <= 31:
                    raise ValueError("Invalid day")
                if "??" in value:
                    status = "partial_calendar_label"
                    precision = "partial month/day"
                else:
                    date.fromisoformat(value)
                    status = "catalogue_calendar_label"
                    precision = "day"
            except ValueError:
                pass
        elif isinstance(value, str) and "T" in value:
            e = timestamp_event(role, value, source, reference)
            e["meaning"] = (
                "Legacy observation timestamp; native precision/time basis may be unavailable. Not a discovery date."
            )
            return e
    return event(
        role,
        value,
        source,
        reference,
        representation,
        "not_declared",
        precision,
        status,
        "Observation used in the orbit fit; may include precovery. Not the discovery date or an invented midnight UTC instant.",
    )


def source_events(s):
    source = s["source"]
    ref = s["reference"]
    out = []
    for role in ("first_observation", "last_observation"):
        raw = (s["native_observation_window"].get(role) or {}).get("value")
        out.append(
            observation_event(
                role, raw if raw is not None else s[role], source, ref + "/" + role
            )
        )
    native = s["native_orbit_epoch"]
    if native is not None:
        from .epoch_evidence import NativeEpoch

        try:
            if (
                not isinstance(native, dict)
                or len(str(native.get("native_value", ""))) > 500
            ):
                raise ValueError("Unsupported native epoch")
            if native.get("format") in ("JD", "MJD") and not math.isfinite(
                float(native["native_value"])
            ):
                raise ValueError("Unsupported native epoch")
            n = NativeEpoch.from_dict(native)
            out.append(
                event(
                    "orbit_epoch",
                    n.native_value,
                    source,
                    ref + "/orbital_elements/native_epoch",
                    n.format,
                    n.time_scale,
                    "native literal precision",
                    "native_epoch",
                    "Osculating-element epoch; not observation/retrieval time. No conversion or propagation performed.",
                )
            )
        except (ValueError, KeyError, TypeError, OverflowError):
            out.append(
                event(
                    "orbit_epoch",
                    native,
                    source,
                    ref + "/orbital_elements/native_epoch",
                    "native epoch",
                    "not_declared",
                    "not_established",
                    "invalid",
                    "Unsupported native epoch retained; not converted.",
                )
            )
    elif s["legacy_orbit_epoch"] is not None:
        e = timestamp_event(
            "orbit_epoch",
            s["legacy_orbit_epoch"],
            source,
            ref + "/orbital_elements/epoch",
        )
        e["meaning"] = (
            "Legacy epoch timestamp, not a source-native JD/MJD scale assertion; not propagated."
        )
        out.append(e)
    covariance = s["covariance_epoch_jd_tdb"]
    if covariance is not None:
        out.append(
            event(
                "covariance_epoch",
                str(covariance),
                source,
                s["covariance_reference"],
                "JD",
                "TDB",
                "native literal precision",
                "native_covariance_epoch",
                "Labelled covariance epoch from the existing typed TDB contract; not propagated to the displayed orbit epoch.",
            )
        )
    if s["orbit_fit_label"] is not None:
        scale = (
            "Pacific local; offset not supplied"
            if source == "SBDB" and s["orbit_fit_schema"] == "1.3"
            else "not_declared"
        )
        out.append(
            event(
                "orbit_fit_date",
                s["orbit_fit_label"],
                source,
                ref + "/evidence_metadata/solution/solution_date",
                "native fit-date label",
                scale,
                "native literal precision",
                "native_label",
                "When the catalogue fit was produced, not its dynamical epoch. No UTC/DST offset inferred.",
            )
        )
    out.append(
        timestamp_event("adapter_receipt", s["fetched_at"], source, ref + "/fetched_at")
    )
    discovery = s["discovery_evidence"]
    if discovery is not None:
        d = verify_discovery(discovery)
        if d["context"]["requested_designation"] != s["designation"]:
            raise ValueError("Discovery/source binding mismatch")
        out.append(
            event(
                "discovery",
                d["normalized"]["calendar_date"] if d["normalized"] else None,
                d["source"],
                ref + "/evidence_metadata/discovery_evidence",
                "catalogue calendar date",
                "not_declared",
                "day",
                d["status"],
                "Catalogued discovery day, not a UTC instant or a storage receipt. SBDB circumstances are documented only for numbered objects.",
            )
        )
    return out


def build_chronology(
    designation,
    sources=(),
    ledger_records=(),
    legacy_dates=None,
    reference_scope="compact_inputs_v1",
):
    # A frozen legacy review may contain controls in its label. Preserve the input;
    # presentation escapes it rather than executing or silently renaming it.
    if not isinstance(designation, str) or not designation or len(designation) > 100:
        raise ValueError("Invalid designation")
    if reference_scope not in (None, "compact_inputs_v1"):
        raise ValueError("Unsupported chronology reference scope")
    sources = clone(list(sources))
    ledger_records = clone(list(ledger_records))
    legacy_dates = clone(legacy_dates or {})
    prototype = set(source_input({"designation": designation}, "source", "/snapshot"))
    for s in sources:
        if (
            not isinstance(s, dict)
            or set(s) != prototype
            or s["designation"] != designation
        ):
            raise ValueError("Invalid chronology source input")
        _text(s["source"], "source", 100)
        _text(s["reference"], "reference")
        if not isinstance(s["native_observation_window"], dict):
            raise ValueError("Invalid observation window")
    events = []
    for index, s in enumerate(sources):
        items = source_events(s)
        if reference_scope == "compact_inputs_v1":
            base = f"/inputs/sources/{index}"
            fields = {
                "orbit_epoch": (
                    "native_orbit_epoch"
                    if s["native_orbit_epoch"] is not None
                    else "legacy_orbit_epoch"
                ),
                "covariance_epoch": "covariance_epoch_jd_tdb",
                "orbit_fit_date": "orbit_fit_label",
                "adapter_receipt": "fetched_at",
                "discovery": "discovery_evidence",
            }
            for e in items:
                role = e["role"]
                if role in ("first_observation", "last_observation"):
                    native = (s["native_observation_window"].get(role) or {}).get(
                        "value"
                    )
                    suffix = (
                        "native_observation_window/" + role + "/value"
                        if native is not None
                        else role
                    )
                else:
                    suffix = fields[role]
                e["reference"] = base + "/" + suffix
        events.extend(items)
    discovery = [
        e for e in events if e["role"] == "discovery" and e["status"] == "available"
    ]
    unique = {e["value"] for e in discovery}
    selected = discovery[0] if len(unique) == 1 else None
    receipts = []
    last_sequence = 0
    for record_index, r in enumerate(ledger_records):
        if not isinstance(r, dict) or set(r) != {
            "sequence",
            "recorded_at",
            "reference",
            "assertion_id",
            "kind",
        }:
            raise ValueError("Invalid ledger date input")
        if type(r["sequence"]) is not int or r["sequence"] <= last_sequence:
            raise ValueError("Ledger sequence is not strictly ordered")
        last_sequence = r["sequence"]
        _text(r["kind"], "ledger record kind", 32)
        _text(r["reference"], "ledger reference")
        _text(r["assertion_id"], "assertion id", 100)
        receipts.append(
            timestamp_event(
                "ledger_receipt",
                r["recorded_at"],
                "local evidence ledger",
                (
                    f"/inputs/ledger_records/{record_index}/recorded_at"
                    if reference_scope == "compact_inputs_v1"
                    else r["reference"] + "/recorded_at"
                ),
                ledger=True,
            )
        )
    first = receipts[0] if receipts else None
    last = receipts[-1] if receipts else None
    good = lambda e: e is not None and e["status"] == "explicit_utc_offset"
    first_value = first["value"] if good(first) else None
    last_value = last["value"] if good(last) else None
    clock_order = "not_established"
    if first_value and last_value:
        known = [
            datetime.fromisoformat(e["value"].replace("Z", "+00:00"))
            for e in receipts
            if good(e)
        ]
        clock_order = (
            "clock_regression_sequence_order_retained"
            if any(a > b for a, b in zip(known, known[1:]))
            else (
                "unverified_clock_values_present"
                if any(not good(e) for e in receipts)
                else "nondecreasing_verified_clock_values"
            )
        )
    legacy = {}
    for key, value in legacy_dates.items():
        _text(key, "legacy date key", 100)
        legacy[key] = {
            "value": value,
            "status": "semantics_and_time_scale_unverified",
            "time_scale": "not_declared",
            "meaning": "Preserved compatibility value; not accepted as discovery or a UTC ledger receipt.",
        }
    result = {
        "schema_version": SCHEMA,
        "designation": designation,
        "events": events,
        "discovery": {
            "status": (
                "available"
                if selected
                else "ambiguous" if len(unique) > 1 else "not_available"
            ),
            "selected": selected,
            "claims": discovery,
        },
        "local_receipts": {
            "first_ledger_recorded_at": first_value,
            "last_ledger_recorded_at": last_value,
            "scope": "verified ledger sequence only; earlier local intake may predate this ledger",
            "clock_order": clock_order,
            "unverified_count": sum(not good(e) for e in receipts),
            "records": receipts,
        },
        "legacy_dates": legacy,
        "inputs": {
            "designation": designation,
            "sources": sources,
            "ledger_records": ledger_records,
            "legacy_dates": legacy_dates,
        },
        "limitations": [
            "Source labels, local receipts and dynamical epochs are different roles; no common UTC time axis is inferred.",
            "Consistency/replay does not authenticate sources, verify clock synchronization or establish scientific readiness.",
        ],
    }
    if reference_scope is not None:
        result["reference_scope"] = "chronology_input_json_pointers"
        result["inputs"]["reference_scope"] = reference_scope
    result["chronology_id"] = fingerprint(result)
    return result


def verify_chronology(document):
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported chronology")
    try:
        inputs = clone(document["inputs"])
        if "reference_scope" not in inputs:
            inputs["reference_scope"] = (
                None  # Explicit compatibility replay, not the new-builder default.
            )
        actual = build_chronology(**inputs)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid chronology") from exc
    if actual != document:
        raise ValueError("Chronology does not reproduce")
    return actual


def build_object_chronology(snapshot):
    meta = snapshot.get("evidence_metadata") or {}
    selected = (meta.get("field_sources") or {}).get("orbital_elements")
    source = selected or (snapshot.get("sources_used") or ["selected snapshot"])[0]
    from .enrichment import canonical_source

    source = canonical_source(source)
    solutions = meta.get("source_solutions") or {}
    sources = [
        source_input(
            snapshot, source, "/source_snapshot", metadata=solutions.get(source, meta)
        )
    ]
    # Other discovery claims survive reconciliation even when their orbit was not selected.
    for name, metadata in solutions.items():
        if name != source and metadata.get("discovery_evidence") is not None:
            sources.append(
                source_input(
                    {
                        "designation": snapshot["designation"],
                        "evidence_metadata": {
                            "discovery_evidence": metadata["discovery_evidence"]
                        },
                    },
                    name,
                    "/source_snapshot/evidence_metadata/source_solutions/" + name,
                )
            )
    return build_chronology(
        snapshot["designation"], sources, reference_scope="compact_inputs_v1"
    )


def resolve_chronology_reference(document, reference):
    """Resolve a checked compact-input pointer, never a filesystem or network path."""
    verified = verify_chronology(document)
    if (
        verified.get("reference_scope") != "chronology_input_json_pointers"
        or not isinstance(reference, str)
        or not reference.startswith("/inputs/")
    ):
        raise ValueError("Not a compact chronology input pointer")
    value = verified
    try:
        for token in reference.split("/")[1:]:
            if re.search(r"~(?![01])", token):
                raise ValueError("Invalid JSON pointer escape")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise ValueError("Invalid JSON pointer array index")
                value = value[int(token)]
            else:
                value = value[token]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError("Unresolvable chronology input pointer") from exc
    return clone(value)
