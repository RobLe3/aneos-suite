"""Frozen inputs for required review, distinct from the ranking-only snapshot."""
from hashlib import sha256
import json


def review_evidence_fingerprint(snapshot):
    return sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def freeze_review_inputs(neo):
    # Exclude derived ranks and wall-clock object construction timestamps.
    # Include every field the canonical prerequisite path reads, plus complete
    # physical/orbit evidence so later fitting cannot inherit a rank-only ID.
    snapshot = {
        'schema_version': 'required-review-inputs-v1', 'designation': neo.designation,
        'orbital_elements': neo.orbital_elements.to_dict() if neo.orbital_elements else None,
        'physical_properties': neo.physical_properties.to_dict() if neo.physical_properties else None,
        'nongrav': neo.nongrav.to_dict() if neo.nongrav else None,
        'close_approaches': [ca.to_dict() for ca in neo.close_approaches],
        'sources_used': neo.sources_used, 'source_contributions': neo.source_contributions,
        'evidence_metadata': neo.evidence_metadata,
        'first_observation': neo.first_observation.isoformat() if neo.first_observation else None,
        'last_observation': neo.last_observation.isoformat() if neo.last_observation else None,
        'fetched_at': neo.fetched_at.isoformat() if neo.fetched_at else None,
    }
    frozen = json.loads(json.dumps(snapshot, allow_nan=False))
    return frozen, review_evidence_fingerprint(frozen)
