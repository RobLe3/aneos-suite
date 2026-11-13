"""Atomic catalogue assertions; normalization is not measurement verification.

SBDB phys_par sigma is documented as 1-sigma. Unknown units/interval syntax
are retained, not guessed. Container orbit epochs are not physical measurement dates.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math

SCHEMA = "physical-evidence-v1"
# Explicit supported unit conversions; no generic dimensional guessing.
SPECS = {
    "diameter": ("diameter_km", "km", {"km": 1.0, "m": 0.001}, "diameter_uncertainty"),
    "rot_per": (
        "rotation_period_hours",
        "h",
        {"h": 1.0, "hr": 1.0, "hours": 1.0, "min": 1 / 60.0, "s": 1 / 3600.0},
        "rotation_period_uncertainty",
    ),
    "albedo": (
        "albedo",
        "dimensionless",
        {None: 1.0, "": 1.0, "1": 1.0, "dimensionless": 1.0},
        "albedo_uncertainty",
    ),
    "H": ("absolute_magnitude_h", "mag", {None: 1.0, "": 1.0, "mag": 1.0}, None),
    "spec_T": ("spectral_type", "catalogue label", {None: 1.0, "": 1.0}, None),
    "spec_B": ("spectral_class", "catalogue label", {None: 1.0, "": 1.0}, None),
}


def clone(value):
    return json.loads(json.dumps(value, allow_nan=False))


def fingerprint(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Not a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Not a finite number")
    return result


@dataclass(frozen=True)
class PhysicalMeasurement:
    """Owned, replay-verifiable provider assertion, with unmodified native literals."""

    payload: dict

    @classmethod
    def from_sbdb(cls, native, context):
        if not isinstance(native, dict) or not isinstance(native.get("name"), str):
            raise ValueError("Physical parameter must be a named object")
        if not isinstance(context, dict):
            raise ValueError("Invalid physical evidence context")
        native = clone(native)
        context = clone(context)
        name = native["name"]
        issues = []
        normalized = None
        if name not in SPECS:
            issues.append("unsupported_parameter")
        else:
            field, units, conversions, _ = SPECS[name]
            unit = native.get("units")
            if not isinstance(unit, (str, type(None))) or unit not in conversions:
                issues.append("unsupported_units")
            else:
                factor = conversions[unit]
                try:
                    if field in ("spectral_type", "spectral_class"):
                        value = native.get("value")
                        if not isinstance(value, str) or not value.strip():
                            raise ValueError("Invalid label")
                    else:
                        value = _number(native.get("value")) * factor
                        if not math.isfinite(value):
                            raise ValueError("Overflow")
                        if name in ("diameter", "rot_per") and value <= 0:
                            raise ValueError("Nonpositive measurement")
                        # Geometric albedo may exceed one; it is not a probability.
                        if name == "albedo" and value < 0:
                            raise ValueError("Negative albedo")
                    sigma = None
                    kind = "not_available"
                    raw_sigma = native.get("sigma")
                    if raw_sigma is not None:
                        if field in ("spectral_type", "spectral_class"):
                            issues.append("unsupported_label_uncertainty")
                            kind = "unsupported"
                        else:
                            try:
                                sigma = _number(raw_sigma) * factor
                                if not math.isfinite(sigma) or sigma < 0:
                                    raise ValueError("Invalid sigma")
                                kind = "provider_1sigma"
                            except (TypeError, ValueError, OverflowError):
                                issues.append("unparsed_or_invalid_uncertainty")
                                sigma = None
                                kind = "unsupported"
                    normalized = {
                        "field": field,
                        "value": value,
                        "sigma": sigma,
                        "units": units,
                        "uncertainty_kind": kind,
                    }
                except (TypeError, ValueError, OverflowError):
                    issues.append("invalid_value")
        result = {
            "schema_version": "physical-measurement-v1",
            "adapter": "sbdb-phys-par-1.3",
            "source": "SBDB",
            "native": native,
            "context": context,
            "normalized": normalized,
            "measurement_epoch": None,
            "estimation_status": "not_provided",
            "evidence_basis": "catalogued_external_assertion_not_independently_verified",
            "issues": issues,
            "status": "normalized" if normalized else "not_normalized",
        }
        result["assertion_id"] = fingerprint(result)
        return cls(result)

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict):
            raise ValueError("Invalid physical measurement")
        actual = cls.from_sbdb(record["native"], record["context"])
        if actual.to_dict() != record:
            raise ValueError("Physical measurement does not reproduce")
        return actual

    def to_dict(self):
        return clone(self.payload)


def build_sbdb_evidence(records, context):
    if not isinstance(records, list):
        raise ValueError("SBDB phys_par must be an array")
    parsed = [
        PhysicalMeasurement.from_sbdb(item, context).to_dict() for item in records
    ]
    selection = {}
    groups = {}
    duplicates = {}
    duplicate_records = {}
    unique = []
    for i, record in enumerate(parsed):
        # Equal native assertions at distinct positions are not independent evidence.
        identity = fingerprint(record["native"])
        if identity in duplicates:
            duplicate_records[str(i)] = duplicates[identity]
            # Duplicate annotation is external to the atomic assertion payload.
        else:
            duplicates[identity] = i
            unique.append(i)
    for i in unique:
        record = parsed[i]
        name = record["native"]["name"]
        if name in SPECS:
            groups.setdefault(SPECS[name][0], []).append(i)
    for field, indices in groups.items():
        # Even an unnormalized alternative must not silently disappear from ambiguity.
        available = [i for i in indices if parsed[i]["normalized"] is not None]
        status = (
            "ambiguous"
            if len(indices) > 1
            else "selected" if available else "unavailable"
        )
        selection[field] = {
            "status": status,
            "record_indices": indices,
            "selected_index": available[0] if status == "selected" else None,
        }
    evidence = {
        "schema_version": SCHEMA,
        "context": clone(context),
        "records": parsed,
        "selection": selection,
        "duplicate_records": duplicate_records,
        "limitations": [
            "Normalization/replay is not independent source verification.",
            "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
            "Alternative records are not averaged; physical correlation/independence is not established.",
        ],
    }
    evidence["evidence_id"] = fingerprint(evidence)
    return evidence


def verify_physical_evidence(evidence):
    try:
        if not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported physical evidence")
        records = evidence.get("records")
        if not isinstance(records, list):
            raise ValueError("Invalid physical evidence records")
        context = evidence.get("context")
        if not isinstance(context, dict) or any(
            r["context"] != context for r in records
        ):
            raise ValueError("Mixed physical response contexts")
        actual = build_sbdb_evidence([r["native"] for r in records], context)
        if actual != evidence:
            raise ValueError("Physical evidence does not reproduce")
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid physical evidence") from exc
    return clone(actual)


def physical_projection(evidence):
    """Compatibility fields, selected atomically; uncertainty never crosses records."""
    evidence = verify_physical_evidence(evidence)
    projection = {}
    for field, selection in evidence["selection"].items():
        i = selection["selected_index"]
        if i is None:
            continue
        r = evidence["records"][i]
        n = r["normalized"]
        projection[field] = n["value"]
        sigma_field = SPECS[r["native"]["name"]][3]
        if sigma_field and n["sigma"] is not None:
            projection[sigma_field] = n["sigma"]
    return projection


def verify_neo_physical(neo):
    """Check new atomic projections at reconstruction and analysis boundaries.

    Old snapshots without this contract remain legacy, not retroactively verified.
    """
    meta = neo.evidence_metadata
    if not isinstance(meta, dict):
        raise ValueError("Invalid physical metadata")
    pp = neo.physical_properties
    atomic = meta.get("physical_evidence")
    if atomic is not None:
        atomic = verify_physical_evidence(atomic)
        projected = physical_projection(atomic)
        for field in atomic["selection"]:
            if (getattr(pp, field, None) if pp else None) != projected.get(field):
                raise ValueError(
                    "Physical value does not match atomic source assertion"
                )
            sigma_field = next(v[3] for v in SPECS.values() if v[0] == field)
            if sigma_field and (
                getattr(pp, sigma_field, None) if pp else None
            ) != projected.get(sigma_field):
                raise ValueError(
                    "Physical uncertainty does not match atomic source assertion"
                )
    selections = meta.get("physical_selection", {})
    snapshots = meta.get("physical_source_snapshots", {})
    if not isinstance(selections, dict) or not isinstance(snapshots, dict):
        raise ValueError("Invalid physical source selection")
    for field, selected in selections.items():
        source = selected["source"]
        snapshot = snapshots[source]
        original = snapshot["physical_properties"] or {}
        if selected.get("reference_source") != original.get(
            "reference_source"
        ) or selected.get("measurement_method") != original.get("measurement_method"):
            raise ValueError("Physical selected provenance changed")
        if meta.get("field_sources", {}).get(f"physical_properties.{field}") != source:
            raise ValueError("Physical selected source changed")
        if meta.get("source_solutions", {}).get(source) != snapshot["source_metadata"]:
            raise ValueError("Physical source metadata copies disagree")
        value = getattr(pp, field, None) if pp else None
        if value != selected["value"] or value != original.get(field):
            raise ValueError("Physical source/value selection changed")
        sigma_field = selected["uncertainty_field"]
        if sigma_field and (
            getattr(pp, sigma_field, None) != selected["uncertainty"]
            or getattr(pp, sigma_field, None) != original.get(sigma_field)
        ):
            raise ValueError("Physical source/uncertainty selection changed")
        if selected.get("assertion_id"):
            atomic = verify_physical_evidence(
                snapshot["source_metadata"]["physical_evidence"]
            )
            i = atomic["selection"][field]["selected_index"]
            if (
                i is None
                or atomic["records"][i]["assertion_id"] != selected["assertion_id"]
            ):
                raise ValueError("Physical assertion selection changed")
            projected = physical_projection(atomic)
            if value != projected.get(field) or (
                sigma_field
                and getattr(pp, sigma_field, None) != projected.get(sigma_field)
            ):
                raise ValueError("Physical selection differs from normalized assertion")
    return True
