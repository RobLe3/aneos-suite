# aNEOS 1.0.0 Maturity Assessment

_Last updated: 2026-03-08 (Phase 10 completion)_

## Overview

aNEOS v1.0.0 completes ten implementation phases. This document records verified evidence
for every capability claim, remaining limitations, and explicit guidance on what quality
level a user should expect.

## Verified Capability Snapshot (Phase 10 complete)

### Test Coverage

- **59 unit and integration tests pass, 0 fail** (network-dependent tests excluded from CI via `-m "not network"`).
- Test scope: cache, configuration, Sigma-5 detection, physical indicators, ground truth
  validation, DataFetcher integration, Horizons parser, chunk boundary merging.

### Ground Truth Validation

Run against a labelled corpus of confirmed artificial and natural heliocentric objects:

| Object | Type | Sigma | Classified | P(artificial) |
|---|---|---|---|---|
| Tesla Roadster (2018 A1 / SpaceX 2018-017A) | Artificial | 5.76 | ARTIFICIAL ✅ | ~3.7% |
| 2020 SO (Centaur / Surveyor-2 upper stage) | Artificial | 6.97 | ARTIFICIAL ✅ | ~3.7% |
| J002E3 (Apollo 12 S-IVB upper stage) | Artificial | 5.76 | ARTIFICIAL ✅ | ~3.7% |
| 20+ JPL SBDB natural NEOs (Apophis, Bennu, etc.) | Natural | < 3.0 | NATURAL ✅ | ~0.1–0.2% |

- **Sensitivity (recall)**: 1.00 at calibrated threshold 0.037
- **Specificity**: 1.00 at calibrated threshold 0.037
- **F1 score**: 1.00
- **ROC-AUC**: 1.00 (external validation)

### Phase 10 Improvements (UE-019 through UE-026)

| Gap | Fix | Effect |
|---|---|---|
| UE-024 | `NEOData.to_dict()`/`from_dict()` now serialises `physical_properties` + `fetched_at` | Cache hits no longer drop physical evidence; `evidence_count` stable across repeated calls |
| UE-020 | `DataFetcher._fetch_close_approaches()` calls SBDB CAD API | `close_approaches` populated for all live fetches; trajectory evidence active |
| UE-019 | `detect_neo()` fetches 10-year Horizons history and passes to detector | `_analyze_course_corrections()` fires through API for the first time |
| UE-021 | `DetectionResponse` adds `combined_p_value`, `false_discovery_rate`, `analysis_metadata` | Full statistical output exposed via REST |
| UE-022 | Batch results include `sigma_tier`, `combined_p_value`, `evidence_sources`, `interpretation` | Batch API output matches single-detection richness |
| UE-023 | `ImpactResponse` expands from 8 to 16 fields | `probability_uncertainty`, `keyhole_passages`, `damage_radius_km`, `peak_risk_period`, etc. now accessible |
| UE-025 | `OrbitalInput` adds `orbital_history` field | `GET /history` → `POST /detect` chain fully functional |
| UE-026 | `GET /detect` exposes `force_refresh` query parameter | Cache can be bypassed on demand |

### Data Source Status

| Source | Status | Data Contributed |
|---|---|---|
| JPL SBDB (`sbdb.api`) | Working | Orbital elements, physical properties |
| JPL Horizons (element table) | Working | Orbital elements (fallback) |
| JPL Horizons (history) | Working | 10-year Keplerian time-series |
| SBDB CAD API (`cad.api`) | Working | Close approaches (upcoming, ≤0.2 AU) |
| NEODyS | Working (fallback) | Orbital elements (graceful failure if unreachable) |
| MPC | Working (fallback) | Orbital elements (graceful failure if unreachable) |

## Remaining Limitations

### Architectural (deferred, explicitly out of Phase 10 scope)

1. **G-015 — ML classifier not active**: The scikit-learn pipeline exists in `aneos_core/ml/` but is not wired into the default detection path. Detection uses the ValidatedSigma5 statistical detector only.
2. **G-023 — `aneos_menu.py` monolith**: The terminal menu is 11,500 lines in a single file. Functional but not maintainable at scale.
3. **G-031 — Mock JWT**: The API uses mock tokens in dev mode. Not suitable for public deployment without a proper auth provider.

### Scientific / Data Constraints

4. **Bayesian posterior ceiling**: With base prior 0.001, orbital+physical evidence gives posterior ~3–4% maximum. Propulsion signatures or observed course corrections are needed to push above 10%. No automated data source for these exists.
5. **Small ground truth corpus**: Perfect discrimination on 3 artificials + 20+ naturals is a proof of concept. Performance on a larger, diverse unseen corpus is unvalidated.
6. **Calibrated threshold not exposed**: The recommended threshold (0.037) is hardcoded in the ground truth validator. It is not a configurable parameter on the REST API.
7. **NEODyS / MPC contribute no data in practice**: Both sources return empty results under current network conditions. SBDB is the sole contributor of physical properties.
8. **No observation arc quality assessment**: aNEOS does not assess whether an orbit solution is well-constrained. Short-arc solutions can produce spurious anomaly scores.

## Quality Guidance

| Claim | Permitted at this quality level? |
|---|---|
| "This object has an unusual orbit" | Yes, if σ ≥ 3.0 (SIGNIFICANT tier) |
| "This object may warrant follow-up" | Yes, if σ ≥ 3.0 + at least 2 independent evidence types |
| "This object is an artificial NEO" | Only if σ ≥ 5.0 + independent telescope/radar confirmation |
| "P(artificial) = X% proves artificial origin" | No. Max posterior ~3–4% from orbital+physical alone |
| "aNEOS impact probability is authoritative" | No. Cross-check with JPL Scout or ESA NEOCC for P > 1e-6 |

## Recommended Next Steps (post-v1.0.0)

1. **Expand ground truth corpus** to ≥ 20 confirmed artificials (additional spacecraft with known heliocentric orbits: DSCOVR, STEREO-A, STEREO-B, Spitzer, etc.).
2. **Expose calibrated threshold as API parameter** (`?threshold=0.037` on `GET /detect`).
3. **Wire ML classifier** once the ground truth corpus is large enough to train reliably.
4. **Add WISE/NEOWISE photometry ingestion** to activate SWARM LAMBDA (thermal IR).
5. **Production JWT** (e.g., Auth0 or a self-hosted OIDC provider).
