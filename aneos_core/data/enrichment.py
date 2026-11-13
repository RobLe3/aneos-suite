"""Finite acquisition snapshots and deterministic ledger projections.

Raw source assertions remain separate; normalization is not source verification.
Only explicit field names/units are mapped. No origin inference is made here.
"""

import hashlib
import json
from datetime import datetime
from dataclasses import fields
from .models import (
    NEOData,
    OrbitalElements,
    PhysicalProperties,
    CloseApproach,
    NonGravitationalParameters,
)
from .reconciliation import reconcile_sources

SOURCE_PRIORITY = ("SBDB", "NEODyS", "MPC", "Horizons", "CAD")
ALIASES = {
    "NASA_SBDB": "SBDB",
    "SBDB": "SBDB",
    "JPL SBDB": "SBDB",
    "NASA_CAD": "CAD",
    "CAD": "CAD",
    "NEODYS": "NEODyS",
    "MPC": "MPC",
    "HORIZONS": "Horizons",
    "SENTRY": "Sentry",
}
MAX_BYTES = 16 * 1024 * 1024


def dumps(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("Acquisition snapshot exceeds bounded storage size")
    return text


def clone(value):
    return json.loads(dumps(value))


def fingerprint(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def identifier(value, label, limit=100):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"Invalid {label}")
    return value.strip()


def canonical_source(value):
    value = identifier(value, "source")
    return ALIASES.get(value, ALIASES.get(value.upper(), value))


def snapshot_neo(payload, source, recorded_at):
    """Adapt explicit unit-bearing fields; legacy unknowns are retained raw only."""
    designation = identifier(payload.get("designation"), "designation")
    if payload.get("snapshot_schema") not in (None, "neo-data-v1"):
        raise ValueError("Unsupported source snapshot schema")
    if payload.get("snapshot_schema") == "neo-data-v1":
        neo = NEOData.from_dict(payload["neo_data"])
        if neo.designation != designation:
            raise ValueError("Object identity mismatch")
        return neo, "typed_snapshot_not_independently_verified"
    orbital = payload.get("orbital_elements")
    oe = None
    if orbital:
        if not isinstance(orbital, dict):
            raise ValueError("Invalid orbital payload")
        aliases = {
            "a": "semi_major_axis",
            "e": "eccentricity",
            "i": "inclination",
            "om": "ra_of_ascending_node",
            "node": "ra_of_ascending_node",
            "w": "arg_of_periapsis",
            "peri": "arg_of_periapsis",
            "ma": "mean_anomaly",
            "M": "mean_anomaly",
        }
        mapped = {}
        for key, value in orbital.items():
            name = aliases.get(key, key)
            if name in mapped and mapped[name] != value:
                raise ValueError("Conflicting orbital aliases")
            if name in OrbitalElements.__dataclass_fields__:
                mapped[name] = value
        if mapped:
            oe = OrbitalElements.from_dict(mapped)
    physical = payload.get("physical_properties")
    if physical is not None and not isinstance(physical, dict):
        raise ValueError("Invalid physical payload")
    if physical:
        for key in (
            "diameter_uncertainty",
            "albedo_uncertainty",
            "rotation_period_uncertainty",
        ):
            if physical.get(key) is not None and (
                type(physical[key]) not in (int, float) or physical[key] < 0
            ):
                raise ValueError("Invalid physical uncertainty")
    if physical:
        for key in (
            "spectral_type",
            "spectral_class",
            "reference_source",
            "measurement_method",
            "data_quality_flag",
        ):
            if physical.get(key) is not None and not isinstance(physical[key], str):
                raise ValueError("Invalid physical label")
    pp = PhysicalProperties.from_dict(physical) if physical else None
    # Validate numeric model fields without converting booleans/strings or inventing units.
    for obj in (oe, pp):
        if obj is None:
            continue
        for item in fields(obj):
            value = getattr(obj, item.name)
            if value is not None and item.name not in (
                "designation",
                "epoch",
                "native_epoch",
                "spectral_type",
                "spectral_class",
                "color_indices",
                "data_quality_flag",
                "measurement_method",
                "reference_source",
            ):
                if type(value) not in (int, float):
                    raise ValueError("Unverified numeric coercion in acquisition")
    encounters = payload.get("close_approaches") or []
    if not isinstance(encounters, list):
        raise ValueError("Invalid encounters")
    metadata = clone(payload.get("evidence_metadata") or {})
    stamp = datetime.fromisoformat(recorded_at)
    neo = NEOData(
        designation,
        orbital_elements=oe,
        physical_properties=pp,
        created_at=stamp,
        updated_at=stamp,
        close_approaches=[CloseApproach.from_dict(v) for v in encounters],
        sources_used=[source],
        evidence_metadata=metadata,
        nongrav=(
            NonGravitationalParameters.from_dict(payload["nongrav"])
            if payload.get("nongrav")
            else None
        ),
    )
    if any(
        c.designation and c.designation != designation for c in neo.close_approaches
    ):
        raise ValueError("Encounter/object identity mismatch")
    return neo, "legacy_unit_named_projection_not_independently_verified"


def normalize_assertion(payload, recorded_at):
    payload = clone(payload)
    if not isinstance(payload, dict):
        raise ValueError("Acquisition must be an object")
    designation = identifier(payload.get("designation"), "designation")
    source = canonical_source(payload.get("data_source", "Unknown"))
    neo, status = snapshot_neo(payload, source, recorded_at)
    return {
        "schema_version": "normalized-source-assertion-v1",
        "designation": designation,
        "source": source,
        "native_payload": payload,
        "normalization_timestamp": recorded_at,
        "normalized_neo": neo.to_dict(),
        "normalization_status": status,
        "origin_status": "not_assessed",
        "calibrated_artificial_probability": None,
        "uninterpreted_or_legacy_fields": [
            key
            for key in payload
            if key
            not in (
                "designation",
                "data_source",
                "orbital_elements",
                "physical_properties",
                "close_approaches",
                "nongrav",
                "evidence_metadata",
                "snapshot_schema",
                "neo_data",
            )
        ],
    }


def project_assertions(records, priority=SOURCE_PRIORITY):
    """Records ordered by append sequence; retain all history, project latest per source."""
    latest = {}
    for record in records:
        if record["kind"] == "source_assertion":
            latest[record["source"]] = record
    if not latest:
        raise ValueError("No usable source assertions")
    neos = {
        source: NEOData.from_dict(record["normalized"]["normalized_neo"])
        for source, record in latest.items()
    }
    neo = reconcile_sources(neos, list(priority))
    from .physical_evidence import verify_neo_physical
    from .epoch_evidence import verify_neo_epoch

    verify_neo_physical(neo)
    verify_neo_epoch(neo)
    return {
        "schema_version": "enriched-object-projection-v1",
        "designation": neo.designation,
        "neo_data": neo.to_dict(),
        "selected_source_assertions": {s: r["assertion_id"] for s, r in latest.items()},
        "source_priority": list(priority),
        "artificial_probability": None,
        "origin_status": "not_assessed",
        "overall_data_completeness": None,
        "source_acquisition_coverage": None,
        "limitations": [
            "Source priority is not scientific source authentication or calibrated model selection.",
            "Latest receipt per source is a current projection, not deletion or averaging of previous measurements.",
            "Full source eligibility, canonical alias resolution and physical fit readiness are not established.",
        ],
    }
