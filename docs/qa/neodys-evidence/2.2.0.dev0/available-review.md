# Object review: 1998 KY26

## Screening classification
NOT_FLAGGED

## Interpretation
Uncalibrated rank 2.8897253901906135 is below policy threshold 5.0 ranking units. This does not establish a natural origin or a perfectly normal object.

## Scientific status
not independently validated; significance and origin probability unavailable

## Evidence binding
Review ID: 3079c70787b140005bf74aea56745545a2ba109b21063d09376efd7e53f2cb60
Evidence ID: 5a10cffa5b9573e6c15b66fa0eeb85be8b74b636e0cab9425f8129d53592d67f
Policy ID: 7998396ede12e26a30e9f2e4f13daf37c6c794a6b0d1138af1a5a17134518d37

## Data coverage
Basic catalogue field presence: 9/15 (declared 15-field profile only).
Screening orbital fields present: 3/3; domain eligibility is separate.
Overall data completeness: not established. Eligible source coverage: not established.

## Epochs and compatibility
Orbital element epoch: 55869.089613878 MJD TDT (NEODyS).
Covariance epoch: not available in this inventory.
Native times retained; not propagated, converted to UTC or treated as observation dates.

## Recorded data
| Field | Value | Units / status |
| --- | --- | --- |
| Semi-major axis | 1.23 | AU; present |
| Eccentricity | 0.141421356 | dimensionless; present |
| Inclination | 1.62046134 | degrees; present |
| Ascending node | 45 | degrees; present |
| Periapsis argument | 0 | degrees; present |
| Mean anomaly | 335 | degrees; present |
| Orbit epoch | 55869.089613878; native scale not converted to UTC | MJD TDT; present |
| Diameter | not available | km; not_available |
| Absolute magnitude H | 25.594 | mag; present |
| Albedo | not available | dimensionless; not_available |
| Spectral label | not available | catalogue label; not_available |
| Rotation period | not available | h; not_available |
| First observation | not available | catalogue timestamp; source precision retained; not_available |
| Last observation | not available | catalogue timestamp; source precision retained; not_available |
| Retrieved at | 2026-11-13T05:35:23.362198+00:00; offset recorded not time scale conversion | retrieval timestamp; present |

## Atomic physical assertions
NEODyS H (OEF MAG): 25.594 mag; uncertainty not available; ref: not available
Native H/G record: MAG 25.594 0.150. G remains native; no typed phase-law fit or independent H/G evidence is implied.
Catalogue assertions, not independently verified measurements. Physical measurement epoch: not available; retrieval/container orbit dates are not measurement dates.

## Explanation
The rank uses fixed global prototype reference parameters, not a size/survey-conditioned population or calibrated null. Its proxy channels are not independent observations.
References: /ranking_trace/reference_parameters, /assessment/analysis_metadata

A hand-set inclination enhancement was applied to the orbital proxy. This is an algorithm assumption, not origin evidence.
References: /ranking_trace/manual_inclination_enhancement, /assessment/evidence_snapshot/orbital/i

Diameter, absolute magnitude and albedo have physical dependencies. Their availability is not three independent pieces of origin evidence; albedo is not independently scored by the current physical proxy.
References: /assessment/evidence_snapshot/physical, /assessment/evidence_sources

The largest squared-ranking term is orbital_dynamics. This is a decomposition of the heuristic, not the strongest physical evidence or a probability.
References: /ranking_trace/terms/0

The displayed profile counts recorded fields, not complete knowledge or verified source coverage. Missing physical uncertainties, observations and unresolved sources still limit scientific use.
References: /inventory/coverage, /inventory/missing_fields

## Follow-up and limits
- Verify source identity, dated measurements, required uncertainty and compatible solutions.
- Use fitted natural/human-technology/artifact comparisons and withheld observations before physical-origin inference.
- Independent population calibration and search-wide error control are required before significance claims.
- Binding/replay verifies consistency, not source authenticity or measurement accuracy.
- No natural/artificial origin, calibrated probability or five-sigma result is supplied.
- Complete local enrichment and scientific operation readiness remain separate requirements.

## Complete unrounded review
```json
{
  "application_version": "2.2.0.dev0",
  "assessment": {
    "analysis_metadata": {
      "engine_version": "catalogue-ranking-v1",
      "evidence_count_semantics": "analyzed_proxy_channels_not_independent_observations",
      "evidence_id": "5a10cffa5b9573e6c15b66fa0eeb85be8b74b636e0cab9425f8129d53592d67f",
      "five_sigma_claim_permitted": false,
      "human_origin_status": "not_assessed",
      "limitations": [
        "No calibrated null or artificial-origin model is installed.",
        "No search-wide significance claim is available.",
        "Human origin has not been excluded.",
        "Legacy ranking includes dependent size/photometry features and hand-set inclination bonuses; it is retained for comparison, not calibrated inference."
      ],
      "origin_status": "inconclusive",
      "policy_id": "7998396ede12e26a30e9f2e4f13daf37c6c794a6b0d1138af1a5a17134518d37",
      "reference_population_status": "fixed_global_prototype_not_conditioned",
      "schema_version": "screening-assessment-v1",
      "score_basis": "legacy_uncalibrated_dependent_features",
      "significance_status": "unavailable_uncalibrated_model"
    },
    "artificial_probability": null,
    "candidate": false,
    "classification": "NOT_FLAGGED",
    "combined_p_value": null,
    "confidence": null,
    "designation": "1998 KY26",
    "engine_version": "catalogue-ranking-v1",
    "evidence_count": 2,
    "evidence_id": "5a10cffa5b9573e6c15b66fa0eeb85be8b74b636e0cab9425f8129d53592d67f",
    "evidence_snapshot": {
      "additional": {},
      "designation": "1998 KY26",
      "orbital": {
        "a": 1.23,
        "e": 0.1414213562373095,
        "i": 1.6204613440875553
      },
      "physical": {
        "absolute_magnitude": 25.594
      },
      "provenance": {
        "acquisition": {
          "bundle_id": "06ff24737fab2fd95c661e4c9172fedb8ef2e8884801e514cc76a45e6d542dab",
          "coverage": {
            "contributing_fraction": 0.2,
            "contributing_sources": 1,
            "network_request_count": null,
            "planned_sources": 5,
            "profile": "installed-object-catalogue-and-bounded-earth-encounters-v1",
            "scientific_completeness": null,
            "semantics": "fixed planned adapters; empty validated responses differ from data contributions; not scientific readiness or independent evidence",
            "status_counts": {
              "ambiguous": 0,
              "available": 1,
              "invalid": 0,
              "missing": 0,
              "not_supported": 0,
              "unavailable": 4
            },
            "validated_response_fraction": 0.2,
            "validated_response_sources": 1
          },
          "outcomes": [
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359205+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "SBDB",
              "started_at": "2026-11-13T05:35:23.359205+00:00",
              "status": "unavailable"
            },
            {
              "basis": "typed_adapter_projection_not_independently_verified",
              "completed_at": "2026-11-13T05:35:23.362198+00:00",
              "reason_code": null,
              "source": "NEODyS",
              "started_at": "2026-11-13T05:35:23.359299+00:00",
              "status": "available"
            },
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359325+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "MPC",
              "started_at": "2026-11-13T05:35:23.359325+00:00",
              "status": "unavailable"
            },
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359349+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "Horizons",
              "started_at": "2026-11-13T05:35:23.359349+00:00",
              "status": "unavailable"
            },
            {
              "basis": "legacy_adapter_outcome_unknown",
              "completed_at": "2026-11-13T05:35:23.363517+00:00",
              "reason_code": "CAD_SCHEMA_UNVERIFIED",
              "source": "CAD",
              "started_at": "2026-11-13T05:35:23.363499+00:00",
              "status": "unavailable"
            }
          ],
          "query_id": "be993308-ae63-4b45-9f7e-9d5ed76e7651",
          "schema_version": "object-acquisition-v1",
          "source_plan": {
            "context_id": "a095ee87592e86951b4be9c51f6c5c35ad8d1126faf6863f467171303fcde070",
            "identity_policy": "exact_request_designation_native_alias_resolution_not_established",
            "limits": "Installed catalogue adapters plus bounded CAD, not every scientific provider, observation dataset or physical-fit requirement.",
            "plan_id": "1ba13d06b236ceb300987eb6368c43af2dd07f57b795e2bf68337ab54ec23179",
            "profile": "installed-object-catalogue-and-bounded-earth-encounters-v1",
            "roles": {
              "CAD": "computed_earth_encounters_bounded_10_rows",
              "Horizons": "catalogue_orbit_and_available_physical_fields",
              "MPC": "catalogue_orbit_and_available_physical_fields",
              "NEODyS": "catalogue_orbit_and_available_physical_fields",
              "SBDB": "catalogue_orbit_and_available_physical_fields"
            },
            "schema_version": "object-source-plan-v1",
            "source_priority": [
              "SBDB",
              "NEODyS",
              "MPC",
              "Horizons"
            ],
            "sources": [
              "SBDB",
              "NEODyS",
              "MPC",
              "Horizons",
              "CAD"
            ]
          },
          "storage_receipt": {
            "assertion_count": 6,
            "bundle_id": "06ff24737fab2fd95c661e4c9172fedb8ef2e8884801e514cc76a45e6d542dab",
            "commit_status": "committed",
            "committed_at": "2026-11-13T05:35:23.418251+00:00",
            "consistency_status": "replayed_not_authenticated",
            "projection_id": "362f281f3782bf6b2291d0e77ed5134702a7de93618e95d50a8945bc5ae5226c",
            "query_id": "be993308-ae63-4b45-9f7e-9d5ed76e7651",
            "query_sequence": 1,
            "schema_version": "acquisition-storage-receipt-v1"
          }
        },
        "conflicts": {},
        "encounter_source_assertions": [],
        "field_sources": {
          "orbital_elements": "NEODyS",
          "physical_properties.absolute_magnitude_h": "NEODyS"
        },
        "orbit_selection_policy": "first-priority-complete-screening-orbit-else-first-present",
        "physical_selection": {
          "absolute_magnitude_h": {
            "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
            "basis": "normalized_catalogue_assertion",
            "measurement_method": null,
            "reference_source": null,
            "source": "NEODyS",
            "uncertainty": null,
            "uncertainty_field": null,
            "value": 25.594
          }
        },
        "physical_source_snapshots": {
          "NEODyS": {
            "physical_properties": {
              "absolute_magnitude_h": 25.594,
              "albedo": null,
              "albedo_uncertainty": null,
              "color_indices": {},
              "data_quality_flag": null,
              "density_g_cm3": null,
              "diameter_km": null,
              "diameter_uncertainty": null,
              "emissivity": null,
              "mass_kg": null,
              "measurement_method": null,
              "pole_ecliptic_lat": null,
              "pole_ecliptic_lon": null,
              "reference_source": null,
              "rotation_period_hours": null,
              "rotation_period_uncertainty": null,
              "spectral_class": null,
              "spectral_type": null,
              "thermal_inertia": null
            },
            "source_metadata": {
              "identity_assessment": {
                "basis": "exact_oef_native_identifier",
                "native_identifier": "1998KY26",
                "requested_catalogue_identifier": "1998KY26",
                "status": "matched_native_identifier_not_authenticated"
              },
              "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
              "native_oef_header": {
                "format": "OEF2.0",
                "rectype": "ML",
                "refsys": "ECLM J2000"
              },
              "normalization_limits": "EQU to classical elliptic elements only. Native covariance remains native; orbital epoch is not a photometry timestamp. No NG acceleration or origin inference from LSP.",
              "physical_evidence": {
                "adapter": "neodys-oef2-photometry-v1",
                "context": {
                  "epoch_role": "orbit_solution_not_physical_measurement_date",
                  "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                  "native_identifier": "1998KY26",
                  "orbital_epoch": {
                    "conversion": "representation_only_no_time_scale_conversion",
                    "format": "MJD",
                    "julian_day_same_scale": "2455869.589613878",
                    "label": null,
                    "native_value": "55869.089613878",
                    "role": "osculating_elements",
                    "schema_version": "native-epoch-v1",
                    "source": "NEODyS",
                    "time_scale": "TDT",
                    "time_scale_status": "declared_not_independently_verified"
                  },
                  "reference_frame": "ECLM J2000",
                  "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                },
                "duplicate_records": {},
                "evidence_id": "eb30f10debf8425f7037242599b9d8409301723c544018d20472cd9b87bd6cac",
                "limitations": [
                  "Normalization/replay is not independent source verification.",
                  "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
                  "Alternative records are not averaged; physical correlation/independence is not established.",
                  "G is retained in native MAG records; no typed phase-slope model or independent photometry fit is supplied."
                ],
                "records": [
                  {
                    "adapter": "neodys-oef2-photometry-v1",
                    "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
                    "context": {
                      "epoch_role": "orbit_solution_not_physical_measurement_date",
                      "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                      "native_identifier": "1998KY26",
                      "orbital_epoch": {
                        "conversion": "representation_only_no_time_scale_conversion",
                        "format": "MJD",
                        "julian_day_same_scale": "2455869.589613878",
                        "label": null,
                        "native_value": "55869.089613878",
                        "role": "osculating_elements",
                        "schema_version": "native-epoch-v1",
                        "source": "NEODyS",
                        "time_scale": "TDT",
                        "time_scale_status": "declared_not_independently_verified"
                      },
                      "reference_frame": "ECLM J2000",
                      "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                    },
                    "estimation_status": "not_provided",
                    "evidence_basis": "catalogued_external_assertion_not_independently_verified",
                    "issues": [],
                    "measurement_epoch": null,
                    "native": {
                      "line": " MAG 25.594 0.150"
                    },
                    "normalized": {
                      "field": "absolute_magnitude_h",
                      "sigma": null,
                      "uncertainty_kind": "not_available",
                      "units": "mag",
                      "value": 25.594
                    },
                    "schema_version": "physical-measurement-v1",
                    "source": "NEODyS",
                    "status": "normalized"
                  }
                ],
                "schema_version": "physical-evidence-v1",
                "selection": {
                  "absolute_magnitude_h": {
                    "record_indices": [
                      0
                    ],
                    "selected_index": 0,
                    "status": "selected"
                  }
                }
              },
              "record_accounting": [
                {
                  "line_number": 6,
                  "native_line": " EQU 1.23 .1 .1 .01 .01 20",
                  "reason": "named_state_or_native_epoch; no covariance transformation",
                  "status": "normalized",
                  "tag": "EQU"
                },
                {
                  "line_number": 7,
                  "native_line": " MJD 55869.089613878 TDT",
                  "reason": "named_state_or_native_epoch; no covariance transformation",
                  "status": "normalized",
                  "tag": "MJD"
                },
                {
                  "line_number": 8,
                  "native_line": " MAG 25.594 0.150",
                  "reason": "H selected only when valid and unambiguous; G native, no typed phase-law model or fitting-status assertion",
                  "status": "partly_normalized",
                  "tag": "MAG"
                },
                {
                  "line_number": 9,
                  "native_line": " COV 1 2 3",
                  "reason": "native matrix/model record; dimensions, basis, correlations or fit status not inferred",
                  "status": "native_not_normalized",
                  "tag": "COV"
                },
                {
                  "line_number": 10,
                  "native_line": " NEW 42",
                  "reason": "unknown data record retained without guessed semantics",
                  "status": "unsupported_native",
                  "tag": "NEW"
                }
              ]
            }
          }
        },
        "schema_version": "source-reconciliation-v2",
        "source_solutions": {
          "NEODyS": {
            "identity_assessment": {
              "basis": "exact_oef_native_identifier",
              "native_identifier": "1998KY26",
              "requested_catalogue_identifier": "1998KY26",
              "status": "matched_native_identifier_not_authenticated"
            },
            "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
            "native_oef_header": {
              "format": "OEF2.0",
              "rectype": "ML",
              "refsys": "ECLM J2000"
            },
            "normalization_limits": "EQU to classical elliptic elements only. Native covariance remains native; orbital epoch is not a photometry timestamp. No NG acceleration or origin inference from LSP.",
            "physical_evidence": {
              "adapter": "neodys-oef2-photometry-v1",
              "context": {
                "epoch_role": "orbit_solution_not_physical_measurement_date",
                "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                "native_identifier": "1998KY26",
                "orbital_epoch": {
                  "conversion": "representation_only_no_time_scale_conversion",
                  "format": "MJD",
                  "julian_day_same_scale": "2455869.589613878",
                  "label": null,
                  "native_value": "55869.089613878",
                  "role": "osculating_elements",
                  "schema_version": "native-epoch-v1",
                  "source": "NEODyS",
                  "time_scale": "TDT",
                  "time_scale_status": "declared_not_independently_verified"
                },
                "reference_frame": "ECLM J2000",
                "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
              },
              "duplicate_records": {},
              "evidence_id": "eb30f10debf8425f7037242599b9d8409301723c544018d20472cd9b87bd6cac",
              "limitations": [
                "Normalization/replay is not independent source verification.",
                "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
                "Alternative records are not averaged; physical correlation/independence is not established.",
                "G is retained in native MAG records; no typed phase-slope model or independent photometry fit is supplied."
              ],
              "records": [
                {
                  "adapter": "neodys-oef2-photometry-v1",
                  "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
                  "context": {
                    "epoch_role": "orbit_solution_not_physical_measurement_date",
                    "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                    "native_identifier": "1998KY26",
                    "orbital_epoch": {
                      "conversion": "representation_only_no_time_scale_conversion",
                      "format": "MJD",
                      "julian_day_same_scale": "2455869.589613878",
                      "label": null,
                      "native_value": "55869.089613878",
                      "role": "osculating_elements",
                      "schema_version": "native-epoch-v1",
                      "source": "NEODyS",
                      "time_scale": "TDT",
                      "time_scale_status": "declared_not_independently_verified"
                    },
                    "reference_frame": "ECLM J2000",
                    "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                  },
                  "estimation_status": "not_provided",
                  "evidence_basis": "catalogued_external_assertion_not_independently_verified",
                  "issues": [],
                  "measurement_epoch": null,
                  "native": {
                    "line": " MAG 25.594 0.150"
                  },
                  "normalized": {
                    "field": "absolute_magnitude_h",
                    "sigma": null,
                    "uncertainty_kind": "not_available",
                    "units": "mag",
                    "value": 25.594
                  },
                  "schema_version": "physical-measurement-v1",
                  "source": "NEODyS",
                  "status": "normalized"
                }
              ],
              "schema_version": "physical-evidence-v1",
              "selection": {
                "absolute_magnitude_h": {
                  "record_indices": [
                    0
                  ],
                  "selected_index": 0,
                  "status": "selected"
                }
              }
            },
            "record_accounting": [
              {
                "line_number": 6,
                "native_line": " EQU 1.23 .1 .1 .01 .01 20",
                "reason": "named_state_or_native_epoch; no covariance transformation",
                "status": "normalized",
                "tag": "EQU"
              },
              {
                "line_number": 7,
                "native_line": " MJD 55869.089613878 TDT",
                "reason": "named_state_or_native_epoch; no covariance transformation",
                "status": "normalized",
                "tag": "MJD"
              },
              {
                "line_number": 8,
                "native_line": " MAG 25.594 0.150",
                "reason": "H selected only when valid and unambiguous; G native, no typed phase-law model or fitting-status assertion",
                "status": "partly_normalized",
                "tag": "MAG"
              },
              {
                "line_number": 9,
                "native_line": " COV 1 2 3",
                "reason": "native matrix/model record; dimensions, basis, correlations or fit status not inferred",
                "status": "native_not_normalized",
                "tag": "COV"
              },
              {
                "line_number": 10,
                "native_line": " NEW 42",
                "reason": "unknown data record retained without guessed semantics",
                "status": "unsupported_native",
                "tag": "NEW"
              }
            ]
          }
        }
      },
      "schema_version": "screening-evidence-v1"
    },
    "evidence_sources": [
      {
        "analyzed": true,
        "anomaly_score": 2.7636736877715866,
        "confidence_interval": [],
        "data_available": true,
        "dependency_group": "orbital_catalogue",
        "effect_size": 2.7636736877715866,
        "evidence_basis": "heuristic_proxy",
        "evidence_type": "orbital_dynamics",
        "p_value": null,
        "quality_score": null,
        "sample_size": 0,
        "score_semantics": "uncalibrated_ranking_heuristic",
        "type": "orbital_dynamics",
        "unavailable_reason": null
      },
      {
        "analyzed": true,
        "anomaly_score": 1.4621428571428576,
        "confidence_interval": [],
        "data_available": true,
        "dependency_group": "physical_size_photometry",
        "effect_size": 1.4621428571428576,
        "evidence_basis": "heuristic_proxy",
        "evidence_type": "physical_properties",
        "p_value": null,
        "quality_score": null,
        "sample_size": 0,
        "score_semantics": "uncalibrated_ranking_heuristic",
        "type": "physical_properties",
        "unavailable_reason": null
      },
      {
        "analyzed": false,
        "anomaly_score": null,
        "confidence_interval": [],
        "data_available": false,
        "dependency_group": "orbital_catalogue",
        "effect_size": null,
        "evidence_basis": "heuristic_proxy",
        "evidence_type": "course_corrections",
        "p_value": null,
        "quality_score": null,
        "sample_size": 0,
        "score_semantics": "uncalibrated_ranking_heuristic",
        "type": "course_corrections",
        "unavailable_reason": "Requires observed residuals and a validated dynamical comparison; ephemeris changes are not manoeuvres"
      },
      {
        "analyzed": false,
        "anomaly_score": null,
        "confidence_interval": [],
        "data_available": false,
        "dependency_group": "orbital_catalogue",
        "effect_size": null,
        "evidence_basis": "heuristic_proxy",
        "evidence_type": "trajectory_patterns",
        "p_value": null,
        "quality_score": null,
        "sample_size": 0,
        "score_semantics": "uncalibrated_ranking_heuristic",
        "type": "trajectory_patterns",
        "unavailable_reason": "Predicted/repeated encounters are not independent artificial-origin evidence"
      },
      {
        "analyzed": false,
        "anomaly_score": null,
        "confidence_interval": [],
        "data_available": false,
        "dependency_group": "orbital_catalogue",
        "effect_size": null,
        "evidence_basis": "heuristic_proxy",
        "evidence_type": "propulsion_signatures",
        "p_value": null,
        "quality_score": null,
        "sample_size": 0,
        "score_semantics": "uncalibrated_ranking_heuristic",
        "type": "propulsion_signatures",
        "unavailable_reason": "Requires a force model, uncertainty and natural-alternative tests"
      }
    ],
    "false_discovery_rate": null,
    "human_origin_status": "not_assessed",
    "interpretation": "1998 KY26: NOT_FLAGGED; screening score is an uncalibrated ranking heuristic. Statistical significance and origin probability are unavailable.",
    "is_artificial": null,
    "limitations": [
      "No calibrated null or artificial-origin model is installed.",
      "No search-wide significance claim is available.",
      "Human origin has not been excluded.",
      "Legacy ranking includes dependent size/photometry features and hand-set inclination bonuses; it is retained for comparison, not calibrated inference."
    ],
    "non_gravitational_evidence": null,
    "origin_status": "inconclusive",
    "policy_id": "7998396ede12e26a30e9f2e4f13daf37c6c794a6b0d1138af1a5a17134518d37",
    "schema_version": "screening-assessment-v1",
    "screening_score": 2.8897253901906135,
    "sigma_confidence": null,
    "sigma_tier": "UNAVAILABLE",
    "significance_status": "unavailable_uncalibrated_model",
    "spacecraft_veto": false,
    "veto_reason": null
  },
  "decision": {
    "status": "not_flagged_by_policy",
    "text": "Uncalibrated rank 2.8897253901906135 is below policy threshold 5.0 ranking units. This does not establish a natural origin or a perfectly normal object."
  },
  "designation": "1998 KY26",
  "follow_up": [
    "Verify source identity, dated measurements, required uncertainty and compatible solutions.",
    "Use fitted natural/human-technology/artifact comparisons and withheld observations before physical-origin inference.",
    "Independent population calibration and search-wide error control are required before significance claims."
  ],
  "insights": [
    {
      "basis": "heuristic_proxy",
      "key": "reference",
      "references": [
        "/ranking_trace/reference_parameters",
        "/assessment/analysis_metadata"
      ],
      "text": "The rank uses fixed global prototype reference parameters, not a size/survey-conditioned population or calibrated null. Its proxy channels are not independent observations."
    },
    {
      "basis": "heuristic_proxy",
      "key": "inclination",
      "references": [
        "/ranking_trace/manual_inclination_enhancement",
        "/assessment/evidence_snapshot/orbital/i"
      ],
      "text": "A hand-set inclination enhancement was applied to the orbital proxy. This is an algorithm assumption, not origin evidence."
    },
    {
      "basis": "heuristic_proxy",
      "key": "dependencies",
      "references": [
        "/assessment/evidence_snapshot/physical",
        "/assessment/evidence_sources"
      ],
      "text": "Diameter, absolute magnitude and albedo have physical dependencies. Their availability is not three independent pieces of origin evidence; albedo is not independently scored by the current physical proxy."
    },
    {
      "basis": "heuristic_proxy",
      "key": "largest_term",
      "references": [
        "/ranking_trace/terms/0"
      ],
      "text": "The largest squared-ranking term is orbital_dynamics. This is a decomposition of the heuristic, not the strongest physical evidence or a probability."
    },
    {
      "basis": "input_inventory",
      "key": "coverage",
      "references": [
        "/inventory/coverage",
        "/inventory/missing_fields"
      ],
      "text": "The displayed profile counts recorded fields, not complete knowledge or verified source coverage. Missing physical uncertainties, observations and unresolved sources still limit scientific use."
    }
  ],
  "inventory": {
    "catalogue_solutions": [],
    "coverage": {
      "basic_catalogue": {
        "fraction_present": 0.6,
        "present": 9,
        "required": 15
      },
      "eligible_source_coverage": null,
      "meaning": "Field presence in the declared profile only; not full data completeness, source coverage, accuracy or scientific readiness.",
      "overall_data_completeness": null,
      "profile_id": "basic-catalogue-presence-15-v2",
      "screening_orbit": {
        "present": 3,
        "required": 3
      }
    },
    "designation": "1998 KY26",
    "encounters": [],
    "epoch_comparison": {
      "covariance_epoch": null,
      "interpretation": "Epoch equality alone does not establish frame/solution/covariance compatibility; no propagation or time-scale conversion is performed.",
      "same_julian_instant": null
    },
    "measurements": [
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "semi_major_axis",
        "label": "Semi-major axis",
        "references": [
          "/source_snapshot/orbital_elements/semi_major_axis"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "AU",
        "value": 1.23
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "eccentricity",
        "label": "Eccentricity",
        "references": [
          "/source_snapshot/orbital_elements/eccentricity"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "dimensionless",
        "value": 0.1414213562373095
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "inclination",
        "label": "Inclination",
        "references": [
          "/source_snapshot/orbital_elements/inclination"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "degrees",
        "value": 1.6204613440875553
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "ascending_node",
        "label": "Ascending node",
        "references": [
          "/source_snapshot/orbital_elements/ra_of_ascending_node"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "degrees",
        "value": 45.0
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "periapsis_argument",
        "label": "Periapsis argument",
        "references": [
          "/source_snapshot/orbital_elements/arg_of_periapsis"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "degrees",
        "value": 0.0
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "mean_anomaly",
        "label": "Mean anomaly",
        "references": [
          "/source_snapshot/orbital_elements/mean_anomaly"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "degrees",
        "value": 335.0
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "orbit_epoch",
        "label": "Orbit epoch",
        "references": [
          "/source_snapshot/orbital_elements/native_epoch"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "time_status": "native_scale_not_converted_to_UTC",
        "units": "MJD TDT",
        "value": "55869.089613878"
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "diameter",
        "label": "Diameter",
        "references": [
          "/source_snapshot/physical_properties/diameter_km"
        ],
        "source_assertion": null,
        "status": "not_available",
        "units": "km",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "absolute_magnitude",
        "label": "Absolute magnitude H",
        "references": [
          "/source_snapshot/physical_properties/absolute_magnitude_h"
        ],
        "source_assertion": "NEODyS",
        "status": "present",
        "units": "mag",
        "value": 25.594
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "albedo",
        "label": "Albedo",
        "references": [
          "/source_snapshot/physical_properties/albedo"
        ],
        "source_assertion": null,
        "status": "not_available",
        "units": "dimensionless",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "spectral_type",
        "label": "Spectral label",
        "references": [
          "/source_snapshot/physical_properties/spectral_type"
        ],
        "source_assertion": null,
        "status": "not_available",
        "units": "catalogue label",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "rotation_period",
        "label": "Rotation period",
        "references": [
          "/source_snapshot/physical_properties/rotation_period_hours"
        ],
        "source_assertion": null,
        "status": "not_available",
        "units": "h",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "first_observation",
        "label": "First observation",
        "references": [
          "/source_snapshot/first_observation"
        ],
        "source_assertion": null,
        "status": "not_available",
        "time_status": "not_available",
        "units": "catalogue timestamp; source precision retained",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "last_observation",
        "label": "Last observation",
        "references": [
          "/source_snapshot/last_observation"
        ],
        "source_assertion": null,
        "status": "not_available",
        "time_status": "not_available",
        "units": "catalogue timestamp; source precision retained",
        "value": null
      },
      {
        "basis": "catalogue_input_not_independently_verified",
        "key": "fetched_at",
        "label": "Retrieved at",
        "references": [
          "/source_snapshot/fetched_at"
        ],
        "source_assertion": null,
        "status": "present",
        "time_status": "offset_recorded_not_time_scale_conversion",
        "units": "retrieval timestamp",
        "value": "2026-11-13T05:35:23.362198+00:00"
      }
    ],
    "missing_fields": [
      {
        "key": "diameter",
        "status": "not_available"
      },
      {
        "key": "albedo",
        "status": "not_available"
      },
      {
        "key": "spectral_type",
        "status": "not_available"
      },
      {
        "key": "rotation_period",
        "status": "not_available"
      },
      {
        "key": "first_observation",
        "status": "not_available"
      },
      {
        "key": "last_observation",
        "status": "not_available"
      }
    ],
    "non_gravitational": null,
    "orbital_epoch_evidence": {
      "conversion": "representation_only_no_time_scale_conversion",
      "format": "MJD",
      "julian_day_same_scale": "2455869.589613878",
      "label": null,
      "native_value": "55869.089613878",
      "role": "osculating_elements",
      "schema_version": "native-epoch-v1",
      "source": "NEODyS",
      "time_scale": "TDT",
      "time_scale_status": "declared_not_independently_verified"
    },
    "physical_measurement_evidence": [
      {
        "evidence": {
          "adapter": "neodys-oef2-photometry-v1",
          "context": {
            "epoch_role": "orbit_solution_not_physical_measurement_date",
            "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
            "native_identifier": "1998KY26",
            "orbital_epoch": {
              "conversion": "representation_only_no_time_scale_conversion",
              "format": "MJD",
              "julian_day_same_scale": "2455869.589613878",
              "label": null,
              "native_value": "55869.089613878",
              "role": "osculating_elements",
              "schema_version": "native-epoch-v1",
              "source": "NEODyS",
              "time_scale": "TDT",
              "time_scale_status": "declared_not_independently_verified"
            },
            "reference_frame": "ECLM J2000",
            "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
          },
          "duplicate_records": {},
          "evidence_id": "eb30f10debf8425f7037242599b9d8409301723c544018d20472cd9b87bd6cac",
          "limitations": [
            "Normalization/replay is not independent source verification.",
            "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
            "Alternative records are not averaged; physical correlation/independence is not established.",
            "G is retained in native MAG records; no typed phase-slope model or independent photometry fit is supplied."
          ],
          "records": [
            {
              "adapter": "neodys-oef2-photometry-v1",
              "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
              "context": {
                "epoch_role": "orbit_solution_not_physical_measurement_date",
                "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                "native_identifier": "1998KY26",
                "orbital_epoch": {
                  "conversion": "representation_only_no_time_scale_conversion",
                  "format": "MJD",
                  "julian_day_same_scale": "2455869.589613878",
                  "label": null,
                  "native_value": "55869.089613878",
                  "role": "osculating_elements",
                  "schema_version": "native-epoch-v1",
                  "source": "NEODyS",
                  "time_scale": "TDT",
                  "time_scale_status": "declared_not_independently_verified"
                },
                "reference_frame": "ECLM J2000",
                "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
              },
              "estimation_status": "not_provided",
              "evidence_basis": "catalogued_external_assertion_not_independently_verified",
              "issues": [],
              "measurement_epoch": null,
              "native": {
                "line": " MAG 25.594 0.150"
              },
              "normalized": {
                "field": "absolute_magnitude_h",
                "sigma": null,
                "uncertainty_kind": "not_available",
                "units": "mag",
                "value": 25.594
              },
              "schema_version": "physical-measurement-v1",
              "source": "NEODyS",
              "status": "normalized"
            }
          ],
          "schema_version": "physical-evidence-v1",
          "selection": {
            "absolute_magnitude_h": {
              "record_indices": [
                0
              ],
              "selected_index": 0,
              "status": "selected"
            }
          }
        },
        "reference": "/source_snapshot/evidence_metadata/source_solutions/NEODyS/physical_evidence",
        "source": "NEODyS"
      }
    ],
    "physical_selection": {
      "absolute_magnitude_h": {
        "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
        "basis": "normalized_catalogue_assertion",
        "measurement_method": null,
        "reference_source": null,
        "source": "NEODyS",
        "uncertainty": null,
        "uncertainty_field": null,
        "value": 25.594
      }
    },
    "query_coverage": {},
    "raw_physical_assertions": [],
    "schema_version": "object-field-inventory-v1",
    "source_differences": {},
    "source_id": "1616bbb45d6ce22aed2bf3a638fac8da648223644966ed7cedbd0776e7c3d8d5",
    "source_snapshot": {
      "close_approaches": [],
      "completeness": 1.0,
      "created_at": "2026-11-13T05:35:23.362198+00:00",
      "designation": "1998 KY26",
      "dynamic_TAS": null,
      "dynamic_category": null,
      "evidence_metadata": {
        "acquisition": {
          "bundle_id": "06ff24737fab2fd95c661e4c9172fedb8ef2e8884801e514cc76a45e6d542dab",
          "coverage": {
            "contributing_fraction": 0.2,
            "contributing_sources": 1,
            "network_request_count": null,
            "planned_sources": 5,
            "profile": "installed-object-catalogue-and-bounded-earth-encounters-v1",
            "scientific_completeness": null,
            "semantics": "fixed planned adapters; empty validated responses differ from data contributions; not scientific readiness or independent evidence",
            "status_counts": {
              "ambiguous": 0,
              "available": 1,
              "invalid": 0,
              "missing": 0,
              "not_supported": 0,
              "unavailable": 4
            },
            "validated_response_fraction": 0.2,
            "validated_response_sources": 1
          },
          "outcomes": [
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359205+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "SBDB",
              "started_at": "2026-11-13T05:35:23.359205+00:00",
              "status": "unavailable"
            },
            {
              "basis": "typed_adapter_projection_not_independently_verified",
              "completed_at": "2026-11-13T05:35:23.362198+00:00",
              "reason_code": null,
              "source": "NEODyS",
              "started_at": "2026-11-13T05:35:23.359299+00:00",
              "status": "available"
            },
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359325+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "MPC",
              "started_at": "2026-11-13T05:35:23.359325+00:00",
              "status": "unavailable"
            },
            {
              "basis": "source_initialization",
              "completed_at": "2026-11-13T05:35:23.359349+00:00",
              "reason_code": "SOURCE_INITIALIZATION_FAILED",
              "source": "Horizons",
              "started_at": "2026-11-13T05:35:23.359349+00:00",
              "status": "unavailable"
            },
            {
              "basis": "legacy_adapter_outcome_unknown",
              "completed_at": "2026-11-13T05:35:23.363517+00:00",
              "reason_code": "CAD_SCHEMA_UNVERIFIED",
              "source": "CAD",
              "started_at": "2026-11-13T05:35:23.363499+00:00",
              "status": "unavailable"
            }
          ],
          "query_id": "be993308-ae63-4b45-9f7e-9d5ed76e7651",
          "schema_version": "object-acquisition-v1",
          "source_plan": {
            "context_id": "a095ee87592e86951b4be9c51f6c5c35ad8d1126faf6863f467171303fcde070",
            "identity_policy": "exact_request_designation_native_alias_resolution_not_established",
            "limits": "Installed catalogue adapters plus bounded CAD, not every scientific provider, observation dataset or physical-fit requirement.",
            "plan_id": "1ba13d06b236ceb300987eb6368c43af2dd07f57b795e2bf68337ab54ec23179",
            "profile": "installed-object-catalogue-and-bounded-earth-encounters-v1",
            "roles": {
              "CAD": "computed_earth_encounters_bounded_10_rows",
              "Horizons": "catalogue_orbit_and_available_physical_fields",
              "MPC": "catalogue_orbit_and_available_physical_fields",
              "NEODyS": "catalogue_orbit_and_available_physical_fields",
              "SBDB": "catalogue_orbit_and_available_physical_fields"
            },
            "schema_version": "object-source-plan-v1",
            "source_priority": [
              "SBDB",
              "NEODyS",
              "MPC",
              "Horizons"
            ],
            "sources": [
              "SBDB",
              "NEODyS",
              "MPC",
              "Horizons",
              "CAD"
            ]
          },
          "storage_receipt": {
            "assertion_count": 6,
            "bundle_id": "06ff24737fab2fd95c661e4c9172fedb8ef2e8884801e514cc76a45e6d542dab",
            "commit_status": "committed",
            "committed_at": "2026-11-13T05:35:23.418251+00:00",
            "consistency_status": "replayed_not_authenticated",
            "projection_id": "362f281f3782bf6b2291d0e77ed5134702a7de93618e95d50a8945bc5ae5226c",
            "query_id": "be993308-ae63-4b45-9f7e-9d5ed76e7651",
            "query_sequence": 1,
            "schema_version": "acquisition-storage-receipt-v1"
          }
        },
        "conflicts": {},
        "encounter_source_assertions": [],
        "field_sources": {
          "orbital_elements": "NEODyS",
          "physical_properties.absolute_magnitude_h": "NEODyS"
        },
        "orbit_selection_policy": "first-priority-complete-screening-orbit-else-first-present",
        "physical_selection": {
          "absolute_magnitude_h": {
            "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
            "basis": "normalized_catalogue_assertion",
            "measurement_method": null,
            "reference_source": null,
            "source": "NEODyS",
            "uncertainty": null,
            "uncertainty_field": null,
            "value": 25.594
          }
        },
        "physical_source_snapshots": {
          "NEODyS": {
            "physical_properties": {
              "absolute_magnitude_h": 25.594,
              "albedo": null,
              "albedo_uncertainty": null,
              "color_indices": {},
              "data_quality_flag": null,
              "density_g_cm3": null,
              "diameter_km": null,
              "diameter_uncertainty": null,
              "emissivity": null,
              "mass_kg": null,
              "measurement_method": null,
              "pole_ecliptic_lat": null,
              "pole_ecliptic_lon": null,
              "reference_source": null,
              "rotation_period_hours": null,
              "rotation_period_uncertainty": null,
              "spectral_class": null,
              "spectral_type": null,
              "thermal_inertia": null
            },
            "source_metadata": {
              "identity_assessment": {
                "basis": "exact_oef_native_identifier",
                "native_identifier": "1998KY26",
                "requested_catalogue_identifier": "1998KY26",
                "status": "matched_native_identifier_not_authenticated"
              },
              "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
              "native_oef_header": {
                "format": "OEF2.0",
                "rectype": "ML",
                "refsys": "ECLM J2000"
              },
              "normalization_limits": "EQU to classical elliptic elements only. Native covariance remains native; orbital epoch is not a photometry timestamp. No NG acceleration or origin inference from LSP.",
              "physical_evidence": {
                "adapter": "neodys-oef2-photometry-v1",
                "context": {
                  "epoch_role": "orbit_solution_not_physical_measurement_date",
                  "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                  "native_identifier": "1998KY26",
                  "orbital_epoch": {
                    "conversion": "representation_only_no_time_scale_conversion",
                    "format": "MJD",
                    "julian_day_same_scale": "2455869.589613878",
                    "label": null,
                    "native_value": "55869.089613878",
                    "role": "osculating_elements",
                    "schema_version": "native-epoch-v1",
                    "source": "NEODyS",
                    "time_scale": "TDT",
                    "time_scale_status": "declared_not_independently_verified"
                  },
                  "reference_frame": "ECLM J2000",
                  "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                },
                "duplicate_records": {},
                "evidence_id": "eb30f10debf8425f7037242599b9d8409301723c544018d20472cd9b87bd6cac",
                "limitations": [
                  "Normalization/replay is not independent source verification.",
                  "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
                  "Alternative records are not averaged; physical correlation/independence is not established.",
                  "G is retained in native MAG records; no typed phase-slope model or independent photometry fit is supplied."
                ],
                "records": [
                  {
                    "adapter": "neodys-oef2-photometry-v1",
                    "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
                    "context": {
                      "epoch_role": "orbit_solution_not_physical_measurement_date",
                      "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                      "native_identifier": "1998KY26",
                      "orbital_epoch": {
                        "conversion": "representation_only_no_time_scale_conversion",
                        "format": "MJD",
                        "julian_day_same_scale": "2455869.589613878",
                        "label": null,
                        "native_value": "55869.089613878",
                        "role": "osculating_elements",
                        "schema_version": "native-epoch-v1",
                        "source": "NEODyS",
                        "time_scale": "TDT",
                        "time_scale_status": "declared_not_independently_verified"
                      },
                      "reference_frame": "ECLM J2000",
                      "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                    },
                    "estimation_status": "not_provided",
                    "evidence_basis": "catalogued_external_assertion_not_independently_verified",
                    "issues": [],
                    "measurement_epoch": null,
                    "native": {
                      "line": " MAG 25.594 0.150"
                    },
                    "normalized": {
                      "field": "absolute_magnitude_h",
                      "sigma": null,
                      "uncertainty_kind": "not_available",
                      "units": "mag",
                      "value": 25.594
                    },
                    "schema_version": "physical-measurement-v1",
                    "source": "NEODyS",
                    "status": "normalized"
                  }
                ],
                "schema_version": "physical-evidence-v1",
                "selection": {
                  "absolute_magnitude_h": {
                    "record_indices": [
                      0
                    ],
                    "selected_index": 0,
                    "status": "selected"
                  }
                }
              },
              "record_accounting": [
                {
                  "line_number": 6,
                  "native_line": " EQU 1.23 .1 .1 .01 .01 20",
                  "reason": "named_state_or_native_epoch; no covariance transformation",
                  "status": "normalized",
                  "tag": "EQU"
                },
                {
                  "line_number": 7,
                  "native_line": " MJD 55869.089613878 TDT",
                  "reason": "named_state_or_native_epoch; no covariance transformation",
                  "status": "normalized",
                  "tag": "MJD"
                },
                {
                  "line_number": 8,
                  "native_line": " MAG 25.594 0.150",
                  "reason": "H selected only when valid and unambiguous; G native, no typed phase-law model or fitting-status assertion",
                  "status": "partly_normalized",
                  "tag": "MAG"
                },
                {
                  "line_number": 9,
                  "native_line": " COV 1 2 3",
                  "reason": "native matrix/model record; dimensions, basis, correlations or fit status not inferred",
                  "status": "native_not_normalized",
                  "tag": "COV"
                },
                {
                  "line_number": 10,
                  "native_line": " NEW 42",
                  "reason": "unknown data record retained without guessed semantics",
                  "status": "unsupported_native",
                  "tag": "NEW"
                }
              ]
            }
          }
        },
        "schema_version": "source-reconciliation-v2",
        "source_solutions": {
          "NEODyS": {
            "identity_assessment": {
              "basis": "exact_oef_native_identifier",
              "native_identifier": "1998KY26",
              "requested_catalogue_identifier": "1998KY26",
              "status": "matched_native_identifier_not_authenticated"
            },
            "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
            "native_oef_header": {
              "format": "OEF2.0",
              "rectype": "ML",
              "refsys": "ECLM J2000"
            },
            "normalization_limits": "EQU to classical elliptic elements only. Native covariance remains native; orbital epoch is not a photometry timestamp. No NG acceleration or origin inference from LSP.",
            "physical_evidence": {
              "adapter": "neodys-oef2-photometry-v1",
              "context": {
                "epoch_role": "orbit_solution_not_physical_measurement_date",
                "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                "native_identifier": "1998KY26",
                "orbital_epoch": {
                  "conversion": "representation_only_no_time_scale_conversion",
                  "format": "MJD",
                  "julian_day_same_scale": "2455869.589613878",
                  "label": null,
                  "native_value": "55869.089613878",
                  "role": "osculating_elements",
                  "schema_version": "native-epoch-v1",
                  "source": "NEODyS",
                  "time_scale": "TDT",
                  "time_scale_status": "declared_not_independently_verified"
                },
                "reference_frame": "ECLM J2000",
                "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
              },
              "duplicate_records": {},
              "evidence_id": "eb30f10debf8425f7037242599b9d8409301723c544018d20472cd9b87bd6cac",
              "limitations": [
                "Normalization/replay is not independent source verification.",
                "No physical measurement epoch or fitted-versus-derived status is supplied by this adapter.",
                "Alternative records are not averaged; physical correlation/independence is not established.",
                "G is retained in native MAG records; no typed phase-slope model or independent photometry fit is supplied."
              ],
              "records": [
                {
                  "adapter": "neodys-oef2-photometry-v1",
                  "assertion_id": "b0ac547cdfdf60ab8325b899718b3793e5434a9463af26b97af81a9e9e56d425",
                  "context": {
                    "epoch_role": "orbit_solution_not_physical_measurement_date",
                    "native_file_sha256": "ca1a26be5daf417b52a3119bc86de443f1c584e5ec053302a430782b1658054c",
                    "native_identifier": "1998KY26",
                    "orbital_epoch": {
                      "conversion": "representation_only_no_time_scale_conversion",
                      "format": "MJD",
                      "julian_day_same_scale": "2455869.589613878",
                      "label": null,
                      "native_value": "55869.089613878",
                      "role": "osculating_elements",
                      "schema_version": "native-epoch-v1",
                      "source": "NEODyS",
                      "time_scale": "TDT",
                      "time_scale_status": "declared_not_independently_verified"
                    },
                    "reference_frame": "ECLM J2000",
                    "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html"
                  },
                  "estimation_status": "not_provided",
                  "evidence_basis": "catalogued_external_assertion_not_independently_verified",
                  "issues": [],
                  "measurement_epoch": null,
                  "native": {
                    "line": " MAG 25.594 0.150"
                  },
                  "normalized": {
                    "field": "absolute_magnitude_h",
                    "sigma": null,
                    "uncertainty_kind": "not_available",
                    "units": "mag",
                    "value": 25.594
                  },
                  "schema_version": "physical-measurement-v1",
                  "source": "NEODyS",
                  "status": "normalized"
                }
              ],
              "schema_version": "physical-evidence-v1",
              "selection": {
                "absolute_magnitude_h": {
                  "record_indices": [
                    0
                  ],
                  "selected_index": 0,
                  "status": "selected"
                }
              }
            },
            "record_accounting": [
              {
                "line_number": 6,
                "native_line": " EQU 1.23 .1 .1 .01 .01 20",
                "reason": "named_state_or_native_epoch; no covariance transformation",
                "status": "normalized",
                "tag": "EQU"
              },
              {
                "line_number": 7,
                "native_line": " MJD 55869.089613878 TDT",
                "reason": "named_state_or_native_epoch; no covariance transformation",
                "status": "normalized",
                "tag": "MJD"
              },
              {
                "line_number": 8,
                "native_line": " MAG 25.594 0.150",
                "reason": "H selected only when valid and unambiguous; G native, no typed phase-law model or fitting-status assertion",
                "status": "partly_normalized",
                "tag": "MAG"
              },
              {
                "line_number": 9,
                "native_line": " COV 1 2 3",
                "reason": "native matrix/model record; dimensions, basis, correlations or fit status not inferred",
                "status": "native_not_normalized",
                "tag": "COV"
              },
              {
                "line_number": 10,
                "native_line": " NEW 42",
                "reason": "unknown data record retained without guessed semantics",
                "status": "unsupported_native",
                "tag": "NEW"
              }
            ]
          }
        }
      },
      "fetched_at": "2026-11-13T05:35:23.362198+00:00",
      "first_observation": null,
      "last_observation": null,
      "nongrav": null,
      "orbital_elements": {
        "arg_of_periapsis": 0.0,
        "argument_of_perihelion": 0.0,
        "ascending_node": 45.0,
        "eccentricity": 0.1414213562373095,
        "epoch": null,
        "inclination": 1.6204613440875553,
        "mean_anomaly": 335.0,
        "native_epoch": {
          "conversion": "representation_only_no_time_scale_conversion",
          "format": "MJD",
          "julian_day_same_scale": "2455869.589613878",
          "label": null,
          "native_value": "55869.089613878",
          "role": "osculating_elements",
          "schema_version": "native-epoch-v1",
          "source": "NEODyS",
          "time_scale": "TDT",
          "time_scale_status": "declared_not_independently_verified"
        },
        "ra_of_ascending_node": 45.0,
        "semi_major_axis": 1.23
      },
      "physical_properties": {
        "absolute_magnitude_h": 25.594,
        "albedo": null,
        "albedo_uncertainty": null,
        "color_indices": {},
        "data_quality_flag": null,
        "density_g_cm3": null,
        "diameter_km": null,
        "diameter_uncertainty": null,
        "emissivity": null,
        "mass_kg": null,
        "measurement_method": null,
        "pole_ecliptic_lat": null,
        "pole_ecliptic_lon": null,
        "reference_source": null,
        "rotation_period_hours": null,
        "rotation_period_uncertainty": null,
        "spectral_class": null,
        "spectral_type": null,
        "thermal_inertia": null
      },
      "raw_TAS": null,
      "score_components": {},
      "source_contributions": {},
      "sources_used": [
        "NEODyS"
      ],
      "updated_at": "2026-11-13T05:35:23.362198+00:00"
    },
    "sources_recorded": [
      "NEODyS"
    ],
    "typed_physical_uncertainty_fields": {
      "albedo_uncertainty": null,
      "diameter_uncertainty": null,
      "rotation_period_uncertainty": null
    },
    "uncertainty_note": "Scalar-model uncertainty fields and raw assertions are retained; unit/measurement reconciliation is not inferred."
  },
  "limitations": [
    "Binding/replay verifies consistency, not source authenticity or measurement accuracy.",
    "No natural/artificial origin, calibrated probability or five-sigma result is supplied.",
    "Complete local enrichment and scientific operation readiness remain separate requirements."
  ],
  "ranking_trace": {
    "interpretation": "Computation trace and assumed prototype reference parameters, not calibrated probabilities, independent evidence or population validation.",
    "manual_inclination_enhancement": true,
    "overall_rank_available": true,
    "policy": {
      "candidate_threshold": 5.0,
      "version": "follow-up-ranking-v1"
    },
    "reference_parameters": {
      "orbital_elements": {
        "eccentricity": {
          "confidence_interval_95": [
            0.4285,
            0.437
          ],
          "mean": 0.4329,
          "n_samples": 9998,
          "source": "Granvik et al. 2018, observational survey",
          "std": 0.2266
        },
        "inclination": {
          "confidence_interval_95": [
            14.8526,
            15.1699
          ],
          "mean": 15.0114,
          "n_samples": 9998,
          "source": "Granvik et al. 2018, observational survey",
          "std": 7.7786
        },
        "semi_major_axis": {
          "confidence_interval_95": [
            1.6335,
            1.6577
          ],
          "mean": 1.6451,
          "n_samples": 9998,
          "source": "Granvik et al. 2018, observational survey",
          "std": 0.6317
        }
      },
      "physical_properties": {
        "absolute_magnitude": {
          "mean": 21.5,
          "n_samples": 158000,
          "source": "WISE/NEOWISE survey data",
          "std": 2.8
        },
        "albedo": {
          "mean": 0.14,
          "n_samples": 25000,
          "source": "WISE albedo measurements",
          "std": 0.12
        },
        "diameter": {
          "mean": 800.0,
          "n_samples": 50000,
          "source": "WISE diameter measurements",
          "std": 1.2
        }
      }
    },
    "schema_version": "ranking-trace-v1",
    "terms": [
      {
        "basis": "heuristic_proxy",
        "channel": "orbital_dynamics",
        "dependency_group": "orbital_catalogue",
        "effect": 2.7636736877715866,
        "share_of_squared_rank": 0.9146614594003916,
        "weight": 1.0,
        "weighted_square": 7.637892252481001
      },
      {
        "basis": "heuristic_proxy",
        "channel": "physical_properties",
        "dependency_group": "physical_size_photometry",
        "effect": 1.4621428571428576,
        "share_of_squared_rank": 0.08533854059960852,
        "weight": 0.3333333333333333,
        "weighted_square": 0.712620578231293
      }
    ],
    "total_squared_rank": 8.350512830712294
  },
  "review_id": "3079c70787b140005bf74aea56745545a2ba109b21063d09376efd7e53f2cb60",
  "schema_version": "object-evidence-review-v1",
  "scientific_status": "not_independently_validated",
  "scope": "full_object_snapshot",
  "screening_policy": {
    "candidate_threshold": 5.0,
    "version": "follow-up-ranking-v1"
  }
}
```

## Recorded acquisition
Query: `be993308-ae63-4b45-9f7e-9d5ed76e7651`; declared committed local query receipt.
1/5 planned adapters contributed data; 1/5 returned typed data or an explicit empty response.
| Source | Outcome | Recorded at (UTC) |
| --- | --- | --- |
| SBDB | unavailable | 2026-11-13T05:35:23.359205+00:00 |
| NEODyS | available | 2026-11-13T05:35:23.362198+00:00 |
| MPC | unavailable | 2026-11-13T05:35:23.359325+00:00 |
| Horizons | unavailable | 2026-11-13T05:35:23.359349+00:00 |
| CAD | unavailable | 2026-11-13T05:35:23.363517+00:00 |
These are fixed installed-profile response counts, not field completeness, independent observations, source authenticity or scientific readiness. The declared local query record is required for full replay; this summary is not a replay of every native packet.
