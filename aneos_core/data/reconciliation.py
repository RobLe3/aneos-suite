"""Shared deterministic atomic source projection; no source authentication or fit."""

import copy
from dataclasses import fields
from .models import NEOData


def reconcile_sources(results, source_priority):
    order = list(dict.fromkeys(source_priority + sorted(results)))
    order = [name for name in order if name in results]
    if not order:
        raise ValueError("No source snapshots to reconcile")

    def complete_orbit(name):
        oe = results[name].orbital_elements
        return oe is not None and all(
            getattr(oe, k) is not None
            for k in ("semi_major_axis", "eccentricity", "inclination")
        )

    primary = next(
        (name for name in order if complete_orbit(name)),
        next(
            (name for name in order if results[name].orbital_elements is not None),
            order[0],
        ),
    )
    merged = copy.deepcopy(results[primary])
    merged.sources_used = order
    provenance = (
        {"orbital_elements": primary} if merged.orbital_elements is not None else {}
    )
    conflicts = {}
    if merged.nongrav is not None:
        provenance["nongrav"] = primary
    # Value/uncertainty are one selection unit. Missing primary uncertainty
    # must not be filled from a different provider, even if values round equal.
    from .models import PhysicalProperties

    paired = {
        "diameter_km": "diameter_uncertainty",
        "albedo": "albedo_uncertainty",
        "rotation_period_hours": "rotation_period_uncertainty",
    }
    uncertainty_fields = set(paired.values())
    global_metadata = {"reference_source", "measurement_method", "data_quality_flag"}
    selection = {}
    snapshots = {}
    merged.physical_properties = PhysicalProperties()
    for name in order:
        source = results[name]
        if source.designation != merged.designation:
            raise ValueError("Cannot reconcile different object designations")
        if (
            name != primary
            and source.orbital_elements is not None
            and (
                merged.orbital_elements is None
                or source.orbital_elements.to_dict()
                != merged.orbital_elements.to_dict()
            )
        ):
            conflicts[f"orbital_elements:{name}"] = source.orbital_elements.to_dict()
        pp = source.physical_properties
        snapshots[name] = {
            "physical_properties": copy.deepcopy(pp.to_dict()) if pp else None,
            "source_metadata": copy.deepcopy(source.evidence_metadata),
        }
        if pp is None:
            continue
        for item in fields(pp):
            field = item.name
            value = getattr(pp, field)
            if field in uncertainty_fields or field in global_metadata:
                continue
            if value is None or (field == "color_indices" and not value):
                continue
            key = f"physical_properties.{field}"
            if field not in selection:
                setattr(merged.physical_properties, field, copy.deepcopy(value))
                sigma_field = paired.get(field)
                if sigma_field:
                    setattr(
                        merged.physical_properties,
                        sigma_field,
                        getattr(pp, sigma_field),
                    )
                    if getattr(pp, sigma_field) is not None:
                        provenance[f"physical_properties.{sigma_field}"] = name
                selection[field] = {
                    "source": name,
                    "value": copy.deepcopy(value),
                    "uncertainty_field": sigma_field,
                    "uncertainty": getattr(pp, sigma_field) if sigma_field else None,
                    "reference_source": pp.reference_source,
                    "measurement_method": pp.measurement_method,
                    "basis": "source_projection_not_independently_verified",
                }
                atomic = source.evidence_metadata.get("physical_evidence")
                if atomic:
                    from .physical_evidence import (
                        verify_physical_evidence,
                        physical_projection,
                    )

                    atomic = verify_physical_evidence(atomic)
                    projected = physical_projection(atomic)
                    if field in projected:
                        if value != projected[field] or (
                            sigma_field
                            and getattr(pp, sigma_field) != projected.get(sigma_field)
                        ):
                            raise ValueError(
                                "Physical projection differs from atomic measurement"
                            )
                        selected = atomic["selection"][field]["selected_index"]
                        selection[field]["assertion_id"] = atomic["records"][selected][
                            "assertion_id"
                        ]
                        selection[field]["basis"] = "normalized_catalogue_assertion"
                provenance[key] = name
            elif value != getattr(merged.physical_properties, field) or (
                field in paired
                and getattr(pp, paired[field])
                != getattr(merged.physical_properties, paired[field])
            ):
                conflicts[f"{key}:{name}"] = {
                    "value": copy.deepcopy(value),
                    "uncertainty": (
                        getattr(pp, paired[field]) if field in paired else None
                    ),
                    "reference_source": pp.reference_source,
                    "measurement_method": pp.measurement_method,
                }
    # Global reference fields cannot truthfully describe a mixed-source object.
    # Their original values remain in each atomic selection/source snapshot.
    if not selection:
        merged.physical_properties = None
    # Preserve computed encounter alternatives; deduplicate exact records only.
    import json, hashlib

    seen = {}
    encounters = []
    encounter_sources = []
    for name in order:
        for event in results[name].close_approaches:
            raw = json.dumps(
                event.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            key = hashlib.sha256(raw.encode()).hexdigest()
            if key not in seen:
                seen[key] = len(encounters)
                encounters.append(copy.deepcopy(event))
            encounter_sources.append(
                {
                    "source": name,
                    "record_sha256": key,
                    "projection_index": seen[key],
                    "basis": "computed_or_legacy_encounter_not_observation",
                }
            )
    merged.close_approaches = encounters
    merged.evidence_metadata = {
        "schema_version": "source-reconciliation-v2",
        "orbit_selection_policy": "first-priority-complete-screening-orbit-else-first-present",
        "field_sources": provenance,
        "source_solutions": {
            name: copy.deepcopy(results[name].evidence_metadata) for name in order
        },
        "conflicts": conflicts,
        "encounter_source_assertions": encounter_sources,
        "physical_selection": selection,
        "physical_source_snapshots": snapshots,
    }
    from .physical_evidence import verify_neo_physical

    verify_neo_physical(merged)
    from .epoch_evidence import verify_neo_epoch

    verify_neo_epoch(merged)
    return merged
