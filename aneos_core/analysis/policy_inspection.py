"""Read-only active ranking policy snapshots; no runtime configuration mutation.

Checksums/replay establish internal consistency, not authenticity or calibration.
Historical snapshots are compared with, never installed into, the running service.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math

from aneos_version import APPLICATION_VERSION
from .screening import ScreeningAnalysisService, ScreeningPolicy

SCHEMA = "screening-policy-inspection-v1"
MAX_BYTES = 65536
CAPABILITIES = {
    "screening_score": "uncalibrated_follow_up_ranking",
    "gaussian_significance": "unavailable",
    "artificial_origin_probability": "unavailable",
    "false_discovery_rate": "not_reliably_estimated",
    "natural_origin_verdict": "not_supported",
    "physical_model_comparison": "not_validated_in_this_screening_service",
    "configuration_mutation": "not_supported_by_this_menu",
    "required_validation": "separate_required_review_contract_not_exercised_by_this_view",
}


def canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Policy snapshot exceeds size limit")
    return encoded


def fingerprint(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def build_policy_snapshot(service=None, *, created_at=None):
    service = service if service is not None else ScreeningAnalysisService()
    document = {
        "schema_version": SCHEMA,
        "application_version": APPLICATION_VERSION,
        "created_at": (
            created_at
            if created_at is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "policy_description": service.describe_policy(),
        "capabilities": dict(CAPABILITIES),
        "purpose": "read_only_inspection_not_importable_runtime_configuration",
    }
    document["snapshot_id"] = fingerprint(document)
    verify_policy_snapshot(document, service=service)
    return document


def verify_policy_snapshot(document, *, service=None):
    """Validate a bounded historical snapshot and compare to actual active policy."""
    # JSON-copy rejects nonfinite values and prevents caller aliases.
    document = json.loads(canonical(document))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "application_version",
        "created_at",
        "policy_description",
        "capabilities",
        "purpose",
        "snapshot_id",
    }:
        raise ValueError("Unsupported policy snapshot fields")
    if (
        document["schema_version"] != SCHEMA
        or document["purpose"]
        != "read_only_inspection_not_importable_runtime_configuration"
    ):
        raise ValueError("Unsupported policy snapshot schema/purpose")
    if document["capabilities"] != CAPABILITIES:
        raise ValueError("Unsupported scientific capability claims")
    version = document["application_version"]
    if not isinstance(version, str) or not version or len(version) > 100:
        raise ValueError("Invalid application identity")
    stamp = document["created_at"]
    if not isinstance(stamp, str) or len(stamp) > 80:
        raise ValueError("Invalid snapshot timestamp")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Snapshot timestamp requires UTC offset")
    expected = fingerprint({k: v for k, v in document.items() if k != "snapshot_id"})
    if document["snapshot_id"] != expected:
        raise ValueError("Policy snapshot fingerprint mismatch")
    description = document["policy_description"]
    if not isinstance(description, dict) or set(description) != {
        "policy",
        "engine_version",
        "reference_parameters",
        "policy_id",
    }:
        raise ValueError("Invalid policy description")
    policy = description["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "version",
        "candidate_threshold",
    }:
        raise ValueError("Invalid ranking policy fields")
    if (
        not isinstance(policy["version"], str)
        or not policy["version"]
        or len(policy["version"]) > 100
    ):
        raise ValueError("Invalid ranking policy version")
    threshold = policy["candidate_threshold"]
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("Invalid ranking threshold")
    ScreeningPolicy(**policy)
    if (
        not isinstance(description["reference_parameters"], dict)
        or not isinstance(description["engine_version"], str)
        or not description["engine_version"]
    ):
        raise ValueError("Invalid ranking engine description")
    if description["policy_id"] != fingerprint(
        {k: v for k, v in description.items() if k != "policy_id"}
    ):
        raise ValueError("Policy identity mismatch")
    active = (
        service if service is not None else ScreeningAnalysisService()
    ).describe_policy()
    return {
        "status": "verified_snapshot_not_applied",
        "comparison": (
            "matches_active" if description == active else "differs_from_active"
        ),
        "snapshot_id": document["snapshot_id"],
        "recorded_policy_id": description["policy_id"],
        "active_policy_id": active["policy_id"],
        "runtime_configuration_changed": False,
        "source_authenticity": "not_established",
    }


def parse_snapshot_json(text, *, service=None):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Policy snapshot exceeds size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    document = json.loads(text, object_pairs_hook=unique)
    return document, verify_policy_snapshot(document, service=service)


def synthetic_controls(service=None):
    """Authorized positive, ordinary and missing controls; no live observations."""
    service = service if service is not None else ScreeningAnalysisService()
    cases = (
        ("positive", {"a": 1.0, "e": 0.2, "i": 80.0}, True),
        ("ordinary", {"a": 1.0, "e": 0.2, "i": 15.0}, False),
        ("missing", {}, False),
    )
    results = []
    for name, orbital, expected in cases:
        assessment = service.analyze_raw("synthetic-" + name, orbital)
        passed = assessment.candidate is expected and (
            assessment.screening_score is None
            if name == "missing"
            else assessment.screening_score is not None
        )
        results.append(
            {
                "control": name,
                "expected_candidate": expected,
                "candidate": assessment.candidate,
                "score": assessment.screening_score,
                "classification": assessment.classification,
                "passed": passed,
                "policy_id": assessment.policy_id,
            }
        )
    return {
        "control_passed": all(r["passed"] for r in results),
        "controls": results,
        "scientific_validation_status": "unavailable",
        "five_sigma_claim_permitted": False,
        "policy_id": service.describe_policy()["policy_id"],
    }
