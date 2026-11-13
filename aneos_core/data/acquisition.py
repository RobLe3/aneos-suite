"""Frozen per-query source plan/outcomes, independent of SQL and network clients.

Source response coverage, contributing records and scientific completeness differ.
Native adapter payloads are retained, not authenticated. No origin inference here.
"""

from datetime import datetime, timezone
from dataclasses import fields
import math
from uuid import uuid4

from .enrichment import clone, fingerprint, identifier, canonical_source
from .models import (
    NEOData,
    OrbitalElements,
    PhysicalProperties,
    NonGravitationalParameters,
)
from .reconciliation import reconcile_sources

SCHEMA = "object-acquisition-v1"
SUPPORTED = ("SBDB", "NEODyS", "MPC", "Horizons", "CAD")
STATUSES = (
    "available",
    "missing",
    "unavailable",
    "invalid",
    "ambiguous",
    "not_supported",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(priority, *, context_id=None):
    if not isinstance(priority, (list, tuple)) or any(
        not isinstance(s, str) for s in priority
    ):
        raise ValueError("Source priority requires a list of source names")
    sources = list(
        dict.fromkeys([canonical_source(s) for s in priority] + list(SUPPORTED))
    )
    sources = [s for s in sources if s != "CAD"] + ["CAD"]
    record = {
        "schema_version": "object-source-plan-v1",
        "context_id": context_id,
        "profile": "installed-object-catalogue-and-bounded-earth-encounters-v1",
        "sources": sources,
        "source_priority": [s for s in sources if s != "CAD"],
        "roles": {
            s: (
                "computed_earth_encounters_bounded_10_rows"
                if s == "CAD"
                else (
                    "catalogue_orbit_and_available_physical_fields"
                    if s in SUPPORTED
                    else "unsupported_configured_source"
                )
            )
            for s in sources
        },
        "identity_policy": "exact_request_designation_native_alias_resolution_not_established",
        "limits": "Installed catalogue adapters plus bounded CAD, not every scientific provider, observation dataset or physical-fit requirement.",
    }
    record["plan_id"] = fingerprint(record)
    return record


def has_material(neo):
    orbital = neo.orbital_elements
    return bool(
        (
            orbital is not None
            and any(
                getattr(orbital, k) is not None
                for k in (
                    "semi_major_axis",
                    "eccentricity",
                    "inclination",
                    "mean_anomaly",
                    "native_epoch",
                    "epoch",
                )
            )
        )
        or (
            neo.physical_properties is not None
            and any(
                v is not None and v != {}
                for k, v in neo.physical_properties.to_dict().items()
                if k
                not in {"reference_source", "measurement_method", "data_quality_flag"}
            )
        )
        or neo.nongrav is not None
        or neo.close_approaches
    )


def _model_numbers(obj, excluded):
    if obj is None:
        return
    for item in fields(obj):
        value = getattr(obj, item.name)
        if item.name not in excluded and value is not None:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Nonfinite or nonnumeric source measurement")


def adapter_to_neo(source, designation, payload, stamp):
    """Normalize only named unit-bearing adapter fields; retain the whole packet separately."""
    if not isinstance(payload, dict):
        raise ValueError("Source adapter payload must be an object")
    if payload.get("_designation") not in (None, designation):
        raise ValueError("Source adapter designation mismatch")
    metadata = clone(payload.get("_evidence_metadata", {}))
    oe_dict = {
        k: v for k, v in payload.items() if k in OrbitalElements.__dataclass_fields__
    }
    orbital = OrbitalElements.from_dict(oe_dict) if oe_dict else None
    if orbital is not None and orbital.designation not in (None, designation):
        raise ValueError("Orbital/object designation mismatch")
    _model_numbers(orbital, {"designation", "epoch", "native_epoch"})
    physical_raw = payload.get("_physical")
    if physical_raw is None:
        physical_raw = {}
    if not isinstance(physical_raw, dict):
        raise ValueError("Invalid physical adapter payload")
    if metadata.get("physical_evidence") is not None:
        from .physical_evidence import physical_projection

        compatibility = {"diameter_km": "diameter", "rotation_period_hours": "rot_per"}
        expected = {
            compatibility.get(k, k): v
            for k, v in physical_projection(metadata["physical_evidence"]).items()
        }
        if expected != physical_raw:
            raise ValueError("Physical adapter projection does not match its evidence")
    aliases = {"diameter": "diameter_km", "rot_per": "rotation_period_hours"}
    pp_dict = {}
    for key, value in physical_raw.items():
        key = aliases.get(key, key)
        if key in PhysicalProperties.__dataclass_fields__:
            if key in pp_dict and pp_dict[key] != value:
                raise ValueError("Conflicting physical aliases")
            pp_dict[key] = value
    physical = PhysicalProperties.from_dict(pp_dict) if pp_dict else None
    _model_numbers(
        physical,
        {
            "spectral_type",
            "spectral_class",
            "color_indices",
            "reference_source",
            "measurement_method",
            "data_quality_flag",
        },
    )
    if physical is not None:
        for key in (
            "spectral_type",
            "spectral_class",
            "reference_source",
            "measurement_method",
            "data_quality_flag",
        ):
            if getattr(physical, key) is not None and not isinstance(
                getattr(physical, key), str
            ):
                raise ValueError("Invalid physical source label")
        for key in (
            "diameter_uncertainty",
            "albedo_uncertainty",
            "rotation_period_uncertainty",
        ):
            if getattr(physical, key) is not None and getattr(physical, key) < 0:
                raise ValueError("Invalid physical uncertainty")
        if not isinstance(physical.color_indices, dict) or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in physical.color_indices.values()
        ):
            raise ValueError("Invalid colour index measurements")
    ng = (
        NonGravitationalParameters.from_dict(payload["_nongrav"])
        if payload.get("_nongrav")
        else None
    )
    date = datetime.fromisoformat(stamp)
    neo = NEOData(
        designation,
        orbital_elements=orbital,
        physical_properties=physical,
        nongrav=ng,
        sources_used=[source],
        evidence_metadata=metadata,
        fetched_at=date,
        created_at=date,
        updated_at=date,
    )
    native_window = {}
    for key, attr in (
        ("first_observation_date", "first_observation"),
        ("last_observation_date", "last_observation"),
    ):
        literal = payload.get(key)
        if literal is not None:
            native_window[attr] = {
                "value": str(literal),
                "time_scale": "not_declared",
                "basis": "catalogue_calendar_label_not_instant",
            }
            for fmt in ("%Y-%b-%d", "%Y-%m-%d"):
                try:
                    # Preserve date-only catalogue compatibility WITHOUT inventing UTC.
                    setattr(neo, attr, datetime.strptime(str(literal), fmt))
                    break
                except ValueError:
                    pass
    if native_window:
        neo.evidence_metadata["native_observation_window"] = native_window
    return neo


def outcome(
    source,
    designation,
    *,
    status,
    neo=None,
    adapter_payload=None,
    started_at=None,
    completed_at=None,
    reason_code=None,
    basis="adapter_result"
):
    if status not in STATUSES:
        raise ValueError("Unsupported source outcome")
    stamp = completed_at or now()
    return clone(
        {
            "source": canonical_source(source),
            "designation": identifier(designation, "designation"),
            "status": status,
            "started_at": started_at or stamp,
            "completed_at": stamp,
            "reason_code": reason_code,
            "basis": basis,
            "native_response_sha256": (
                fingerprint(adapter_payload["_native_response"])
                if isinstance(adapter_payload, dict)
                and "_native_response" in adapter_payload
                else None
            ),
            "adapter_payload": adapter_payload,
            "neo_data": neo.to_dict() if neo is not None else None,
        }
    )


def coverage(source_plan, outcomes):
    total = len(source_plan["sources"])
    counts = {s: sum(o["status"] == s for o in outcomes) for s in STATUSES}
    contributing = counts["available"]
    answered = contributing + counts["missing"]
    return {
        "profile": source_plan["profile"],
        "planned_sources": total,
        "status_counts": counts,
        "contributing_sources": contributing,
        "validated_response_sources": answered,
        "contributing_fraction": contributing / total if total else None,
        "validated_response_fraction": answered / total if total else None,
        "network_request_count": None,
        "scientific_completeness": None,
        "semantics": "fixed planned adapters; empty validated responses differ from data contributions; not scientific readiness or independent evidence",
    }


def build_bundle(
    designation, source_plan, outcomes, *, query_id=None, started_at=None, software=None
):
    if software is None:
        from aneos_core.analysis.reporting import _software_metadata

        software = _software_metadata()
    lookup = {o["source"]: o for o in outcomes}
    if len(lookup) != len(outcomes) or set(lookup) != set(source_plan["sources"]):
        raise ValueError("Every frozen planned source requires exactly one outcome")
    outcomes = [lookup[s] for s in source_plan["sources"]]
    snapshots = {
        o["source"]: NEOData.from_dict(o["neo_data"])
        for o in outcomes
        if o["status"] == "available"
    }
    merged = (
        reconcile_sources(snapshots, source_plan["source_priority"] + ["CAD"])
        if snapshots
        else None
    )
    document = {
        "schema_version": SCHEMA,
        "query_id": query_id or str(uuid4()),
        "designation": identifier(designation, "designation"),
        "started_at": started_at or now(),
        "source_plan": clone(source_plan),
        "outcomes": clone(outcomes),
        "software": clone(software),
        "coverage": coverage(source_plan, outcomes),
        "neo_data": merged.to_dict() if merged is not None else None,
    }
    document["bundle_id"] = fingerprint(document)
    validate_bundle(document)
    return document


def validate_outcome(o, source, designation):
    o = clone(o)
    if not isinstance(o, dict) or set(o) != {
        "source",
        "designation",
        "status",
        "started_at",
        "completed_at",
        "reason_code",
        "basis",
        "native_response_sha256",
        "adapter_payload",
        "neo_data",
    }:
        raise ValueError("Unsupported source outcome schema")
    if (
        o["source"] != source
        or o["status"] not in STATUSES
        or o["designation"] != designation
    ):
        raise ValueError("Invalid source outcome binding")
    for stamp in (o["started_at"], o["completed_at"]):
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Acquisition timestamps require explicit UTC offset")
    native_hash = (
        fingerprint(o["adapter_payload"]["_native_response"])
        if isinstance(o["adapter_payload"], dict)
        and "_native_response" in o["adapter_payload"]
        else None
    )
    if native_hash != o["native_response_sha256"]:
        raise ValueError("Native response fingerprint mismatch")
    if o["status"] == "available":
        n = NEOData.from_dict(o["neo_data"])
        if n.designation != designation or not has_material(n):
            raise ValueError("Source outcome has no valid material evidence")
        if o["adapter_payload"] is not None:
            replay = adapter_to_neo(
                source, designation, o["adapter_payload"], o["completed_at"]
            )
            if replay.to_dict() != n.to_dict():
                raise ValueError("Native adapter projection does not reproduce")
    elif o["neo_data"] is not None:
        raise ValueError("Unsuccessful outcome cannot publish source evidence")
    return o


def validate_bundle(document):
    document = clone(document)
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "query_id",
            "designation",
            "started_at",
            "source_plan",
            "outcomes",
            "coverage",
            "neo_data",
            "bundle_id",
            "software",
        }
        or document["schema_version"] != SCHEMA
    ):
        raise ValueError("Unsupported acquisition envelope")
    if not isinstance(document["software"], dict) or not isinstance(
        document["software"].get("application_version"), str
    ):
        raise ValueError("Acquisition software identity unavailable")
    identifier(document["query_id"], "query id", 100)
    identifier(document["designation"], "designation")
    p = document["source_plan"]
    if not isinstance(p, dict) or p != plan(
        p.get("source_priority", []), context_id=p.get("context_id")
    ):
        raise ValueError("Acquisition source plan does not reproduce")
    outcomes = document["outcomes"]
    if (
        not isinstance(outcomes, list)
        or [o.get("source") for o in outcomes] != p["sources"]
    ):
        raise ValueError("Frozen source denominator/outcomes mismatch")
    snapshots = {}
    for o in outcomes:
        o = validate_outcome(o, o["source"], document["designation"])
        if o["status"] == "available":
            snapshots[o["source"]] = NEOData.from_dict(o["neo_data"])
    parsed = datetime.fromisoformat(document["started_at"])
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Acquisition timestamps require explicit UTC offset")
    if "CAD" in snapshots:
        accepted = accepted_designations(document["designation"], outcomes)
        if any(
            designation_key(event.designation) not in accepted
            for event in snapshots["CAD"].close_approaches
        ):
            raise ValueError("Computed encounter identity is not bound to the query")
    expected = (
        reconcile_sources(snapshots, p["source_priority"] + ["CAD"]).to_dict()
        if snapshots
        else None
    )
    if (
        expected != document["neo_data"]
        or coverage(p, outcomes) != document["coverage"]
    ):
        raise ValueError("Acquisition result/coverage does not reproduce")
    if (
        fingerprint({k: v for k, v in document.items() if k != "bundle_id"})
        != document["bundle_id"]
    ):
        raise ValueError("Acquisition envelope fingerprint mismatch")
    return document


def result_from_bundle(bundle, receipt):
    bundle = validate_bundle(bundle)
    if bundle["neo_data"] is None:
        return None
    neo = NEOData.from_dict(bundle["neo_data"])
    neo.evidence_metadata["acquisition"] = summarize_bundle(bundle, receipt)
    return neo


def designation_key(value):
    """Whitespace/case/leading-zero normalization only; not an alias resolver."""
    value = identifier(value, "designation")
    compact = "".join(value.split()).upper()
    return str(int(compact)) if compact.isdigit() else compact


def accepted_designations(designation, outcomes):
    """Within-query provider-declared IDs; no global merge or origin hypothesis join."""
    accepted = {designation_key(designation)}
    for o in outcomes:
        if o["status"] != "available" or o["neo_data"] is None:
            continue
        metadata = o["neo_data"].get("evidence_metadata") or {}
        if o["source"] == "SBDB":
            solution = metadata.get("solution") or {}
            native = (solution.get("object") or {}).get("des")
            if solution.get("source") == "JPL SBDB" and isinstance(native, str):
                accepted.add(designation_key(native))
        if o["source"] == "MPC":
            identity = metadata.get("identity_assessment") or {}
            native = (metadata.get("native_catalogue_row") or {}).get("number")
            if (
                identity.get("status") == "matched_native_identifier_not_authenticated"
                and identity.get("requested_designation") == designation
                and type(native) is int
            ):
                accepted.add(designation_key(str(native)))
    return sorted(accepted)


def summarize_bundle(bundle, receipt):
    bundle = validate_bundle(bundle)
    return {
        "schema_version": SCHEMA,
        "query_id": bundle["query_id"],
        "bundle_id": bundle["bundle_id"],
        "source_plan": bundle["source_plan"],
        "coverage": bundle["coverage"],
        "outcomes": [
            {
                k: o[k]
                for k in (
                    "source",
                    "status",
                    "started_at",
                    "completed_at",
                    "reason_code",
                    "basis",
                )
            }
            for o in bundle["outcomes"]
        ],
        "storage_receipt": clone(receipt),
    }
