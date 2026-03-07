# aNEOS Gap Analysis

**Version**: 2.0
**Date**: 2026-03-07
**Supersedes**: v1.0 (2026-03-06)
**Derived from**: Phase 1–3 implementation audit, live code inspection, ground truth
validation results, test suite state (54 pass / 0 fail).

---

## Gap Severity Legend

| Symbol | Severity | Definition |
|--------|----------|------------|
| P0 | Critical | Blocks the core mission; no workaround exists |
| P1 | High | Produces incorrect or misleading results; deployment-blocking |
| P2 | Medium | Maintainability or quality risk; does not affect correctness today |
| P3 | Low | Future-state or polish; no immediate impact |

---

## Closed Gaps (since v1.0)

The following gaps from GAP_ANALYSIS v1.0 are fully closed.

| ID | Title | Closed | Resolution |
|----|-------|--------|------------|
| G-001 | No ground truth dataset | Phase 2+3 | JPL SBDB Query API + Granvik synthetic fallback; GroundTruthValidator; sens=1.00, spec=1.00 on 3 artificials + 20 naturals |
| G-002 | Physical indicator category missing | Phase 2 | `indicators/physical.py` — DiameterAnomalyIndicator, AlbedoAnomalyIndicator, SpectralAnomalyIndicator; wired into pipeline.py |
| G-003 | Pipeline bypasses canonical detector | Phase 3 | `automatic_review_pipeline.py` now uses `DetectionManager(DetectorType.AUTO)` at line 173; MultiModal import removed |
| G-005 | Docker deployment broken | Phase 1 | `init.sql`, `ssl/README.md`, `docker-compose.override.yml.example`, `Makefile` with `bootstrap` target all created |
| G-006 | Monitoring depends on deferred ML | Phase 1 | `monitoring/alerts.py` ML import wrapped in try/except with `_HAS_ML_ALERTS` guard |
| G-007 | README overstates production readiness | Phase 2+3 | README updated to "v0.7.0 Pre-Production Stabilization Series"; measured validation results stated |
| G-010 | No CI/CD pipeline | Phase 1 | `.github/workflows/ci.yml` — pytest + docker-build jobs on push/PR |

---

## P1 — High Severity Gaps

### G-004 (RESIDUAL): Startup Still Soft-Degrades on Missing Core

**Area**: Data Integrity / Operational Reliability
**Status**: Partially closed. `DataSourceUnavailableError` raised in `data/fetcher.py` ✅;
`IntegrationError` raised in `pipeline_integration.py` ✅; `preflight_check()` exists and is
called from `aneos.py status` ✅ and from menu startup ✅.

**Remaining gap**: `aneos_menu.py` lines 24–68 still define `HAS_ANALYSIS_PIPELINE = False`
as a soft-degradation flag. When core components fail to import, the menu shows
`"🟡 LIMITED CAPABILITY"` and continues rather than halting. A user can invoke
Single NEO Analysis, Batch Analysis, or Interactive Mode while the system is silently
missing its detection stack. The menu startup call to `preflight_check()` (line 115)
stores results in `self._preflight` but does not gate the menu on any failure status.

**Evidence**:
```
aneos_menu.py:25   HAS_ANALYSIS_PIPELINE = False
aneos_menu.py:232  detection_status = "🟢 READY" if HAS_ANEOS_CORE else "🟡 LIMITED CAPABILITY"
aneos_menu.py:1409 if not HAS_ANEOS_CORE:   # shows warning but continues
```

**Fix**: After calling `preflight_check()` at startup, if `pipeline` or `analysis` checks
are `"error"`, display the failure table and prompt the user to confirm before proceeding.
**Do not** auto-exit — that breaks offline research workflows. Instead make the degradation
**visible and acknowledged**, not invisible.

**ADR/DDD ref**: ADR-032, DDD BC-1

---

### G-019: Ground Truth Corpus Too Small to Measure Sigma-5 FPR

**Area**: Scientific Validity / Detection Accuracy
**Reality**: The validated external ground truth set contains exactly 3 confirmed
artificial objects (Tesla Roadster, 2020 SO Centaur, Apollo 12 S-IVB). The original
Implementation Plan (Task 2.5) required ≥ 8 artificial objects.

The stated sigma-5 FPR of 5.7×10⁻⁷ (≈ 1 in 1.74 million) cannot be empirically measured
with 3 positives. With only 3 objects:
- Minimum measurable FPR at sigma-5 is 1/3 = 33% (all 3 classified correctly → 0 false positives,
  but this says nothing about rate on the natural population at scale).
- The internal cross-validation set has 2 artificials (Tesla Roadster, DSCOVR spacecraft) — the
  F1=1.0 result is from 2-positive leave-one-out, not statistically meaningful.

**Impact**: The FPR claim in `docs/engineering/sigma5_success_criteria.md` and the README
cannot be verified against real data. Any peer-review of detection accuracy would reject
this corpus as insufficient.

**What is known**: ROC-AUC=1.00, score separation is good (artificials ~3.7% vs naturals
~0.1%). Qualitative discrimination works. Quantitative FPR cannot be measured until the
corpus grows.

**Fix**:
1. Identify additional confirmed artificial heliocentric objects from NORAD/SPICE (e.g.,
   Pioneer 10, Pioneer 11, Voyager 1/2, New Horizons, OSIRIS-REx, Hayabusa2).
2. Fetch orbital elements from JPL Horizons NAIF IDs (not SBDB, which excludes spacecraft).
3. Add to `ARTIFICIAL_FALLBACK` and `ARTIFICIAL_PHYSICAL_FALLBACK` in
   `ground_truth_dataset_preparation.py`.
4. Rerun GroundTruthValidator; document empirical FPR in `sigma5_success_criteria.md`.

**Target corpus**: ≥ 8 confirmed artificials, ≥ 200 natural NEOs.

**ADR/DDD ref**: ADR-038, DDD BC-10

---

### G-020: Bayesian Posterior Ceiling Undocumented and Unsurfaced

**Area**: Scientific Communication / User Trust
**Reality**: The detection system uses a base prior of 0.001 (0.1% artificial NEO rate).
With the maximum achievable likelihood ratio from orbital + physical evidence alone (~10×),
the Bayesian posterior is mathematically bounded to approximately 4%:

```
P(artificial | evidence) = LR × prior / (LR × prior + (1 − prior))
                         = 10 × 0.001 / (10 × 0.001 + 0.999) ≈ 0.99%–4%
```

A sigma-5 detection flag (e.g., Tesla Roadster: σ=5.76) co-exists with a Bayesian
probability of ~3.7%. Nowhere in the UI, documentation, or API responses is this
constraint explained. Operators who read "sigma-5 detection" and expect it to mean
"extremely likely artificial" will be misled.

**The gap is not in the math** — the math is correct. The gap is in communication.
The README says "99.99994% theoretical certainty" which refers to the probability of
observing the data by chance under the null hypothesis, not the probability the object
is artificial.

**Impact**: Publication-ready output and mission reports may be scientifically misleading.
Any external reviewer will flag this discrepancy immediately.

**Fix**:
1. Add a "Detection Interpretation Guide" to the README and `sigma5_success_criteria.md`
   that explicitly distinguishes:
   - **Statistical significance** (sigma level): how rare is this observation under the
     natural NEO null hypothesis?
   - **Bayesian posterior** (P(artificial)): given the prior, how likely is the object
     actually artificial?
2. In `aneos_menu.py` detection output sections, display both values with a brief
   one-line explanation: `"σ=5.76 means orbital characteristics are extremely rare for
   natural objects (1 in 57M chance); P(artificial)=3.7% incorporates 0.1% base rate."`
3. Update `aneos_api/` detection endpoint responses to include this interpretation in
   the response body.
4. Document that smoking-gun evidence (course corrections, propulsion signatures,
   radar specular reflection) is needed to push the posterior above 10–50%.

**ADR/DDD ref**: ADR-011, ADR-039

---

## P2 — Medium Severity Gaps

### G-008: Seven Detector Files, No Archived Canonical

**Area**: Detection / Maintainability
**Status**: OPEN (unchanged from v1.0)
**Reality**: Seven files coexist in `aneos_core/detection/` with no deprecation markers:
- `sigma5_artificial_neo_detector.py`
- `corrected_sigma5_artificial_neo_detector.py`
- `production_artificial_neo_detector.py`
- `multimodal_sigma5_artificial_neo_detector.py`
- `validated_sigma5_artificial_neo_detector.py` ← canonical
- `sigma5_corrected_statistical_framework.py`
- `artificial_neo_test_suite.py` ← test harness (G-013)

**Fix**: Create `aneos_core/detection/_archive/`; move all non-canonical detector files
into it. Keep only `detection_manager.py`, `validated_sigma5_artificial_neo_detector.py`,
and `_archive/`. Update `detection_manager.py` to not import archived detectors. Run
full test suite to verify no breakage.

**ADR/DDD ref**: ADR-013

---

### G-009: Dual Scoring Systems with EMERGENCY Suppressions

**Area**: Anomaly Scoring / Maintainability
**Status**: OPEN (worsened — EMERGENCY comments identified in Phase 3 audit)
**Reality**: Two independent scoring systems coexist:
- `scoring.py` — Standard 6-category, 4-tier classification
- `advanced_scoring.py` (ATLAS) — continuous [0,1], 6 clue categories

`advanced_scoring.py` contains two `# EMERGENCY` comment suppressions:
- Line 70: `# EMERGENCY: Suppress configuration loading warnings`
- Line 192: `# EMERGENCY: Suppress initialization logging`

These were inserted as hotfixes. The underlying configuration/initialization issues they
mask are undiagnosed. This means the ATLAS scoring system may silently fail to load its
configuration or initialize properly on every run.

No document designates which system is canonical for production use.

**Fix**:
1. Diagnose and resolve the underlying issues that required the EMERGENCY suppressions.
   Remove the suppression comments once root causes are fixed.
2. Run both systems against the ground truth set (once G-019 is addressed). Designate
   the higher-F1 system as canonical in ADR-008.
3. Mark the non-canonical system deprecated with a `# DEPRECATED` comment pointing to
   the canonical.

**ADR/DDD ref**: ADR-008

---

### G-011: `OrbitalElements` Conflates Orbital and Physical Properties

**Area**: Domain Model / Maintainability
**Status**: OPEN (unchanged from v1.0)
**Reality**: `aneos_core/data/models.py:OrbitalElements` carries `diameter` (km), `albedo`,
`rot_per`, `spectral_type` — all physical characterization fields that belong in
`PhysicalProperties` (which also exists in the model at line 430).

The physical indicators (`indicators/physical.py`) access these fields via
`neo_data.orbital_elements.diameter` etc., which reinforces the wrong pattern.

**Fix**: Migrate physical fields from `OrbitalElements` to `PhysicalProperties`.
Update all callsites — primarily `indicators/physical.py`, `SBDBSource._parse_sbdb_response()`,
`DataFetcher`, and any analysis pipeline code. This is a broad change requiring full
test suite validation before merging.

**ADR/DDD ref**: ADR-006, DDD BC-2

---

### G-012: API Layer Has No DTO Separation

**Area**: API / Maintainability
**Status**: OPEN (unchanged from v1.0)
**Reality**: `aneos_api/endpoints/` handlers return internal domain types directly.
No `aneos_api/schemas/` directory exists.

**Fix**: Create Pydantic v2 response schemas in `aneos_api/schemas/`. Minimum:
`AnalysisResponse`, `DetectionResponse`, `ImpactResponse`, `HealthResponse`.
Wire into endpoint `response_model=` parameters.

**ADR/DDD ref**: ADR-034, DDD BC-8

---

### G-013: Test Harness Inside Production Module

**Area**: Code Organisation
**Status**: OPEN (unchanged from v1.0)
**Reality**: `aneos_core/detection/artificial_neo_test_suite.py` is a test harness
inside the production module. Coverage tools conflate production and test lines.

**Fix**: Move to `tests/detection/test_artificial_neo_suite.py`. Update import paths.

**ADR/DDD ref**: DDD BC-5

---

### G-014: No OpenAPI Specification Maintained

**Area**: API Documentation
**Status**: OPEN (unchanged from v1.0)
**Reality**: `docs/api/rest-api.md` is hand-maintained and drifts from 52 FastAPI endpoints.
No `docs/api/openapi.json` committed.

**Fix**: Add `make spec` target; generate and commit `openapi.json`; add CI drift check.

**ADR/DDD ref**: ADR-034

---

### G-021: Multi-Source Enrichment Is SBDB-Only in Practice

**Area**: Data Acquisition / Architecture Claim
**Reality**: The 4-source architecture (SBDB, NEODyS, MPC, Horizons) is advertised
in the README and ADR-001. In practice:

- **NEODyS**: `_make_request` synchronous method is not implemented (lines 112–124 are
  bare `pass` blocks). Every NEODyS fetch silently fails. Zero enrichment contribution.
- **MPC**: Same — `_make_request` not implemented.
- **Horizons**: `HorizonsSource` is async-only. `DataFetcher._fetch_from_source()` wraps
  sources with `asyncio.run()`, but Horizons is never instantiated as a source in the
  DataFetcher's source list. Zero contribution.
- **SBDB**: Working. All real enrichment comes exclusively from SBDB.

**Impact**: Every analysis advertised as "multi-source enriched" is single-source. Close
approach data (JPL CAD API), astrometry (NEODyS), and ephemeris history (Horizons) are
not actually retrieved during normal analysis.

**Fix**:
1. **NEODyS/MPC**: Implement `_make_request` using `requests` (sync) or integrate into the
   existing `asyncio.run()` wrapper pattern. Basic orbital element parsing from each
   source's format is sufficient for Phase 4.
2. **Horizons**: Instantiate `HorizonsSource` in `DataFetcher.__init__()` alongside SBDB.
   Its async `fetch()` method already exists — the source just needs to be added to the
   source list.

**ADR/DDD ref**: ADR-001, ADR-002, DDD BC-1

---

### G-022: `advanced_scoring.py` EMERGENCY Suppressions Mask Unknown Issue

**Area**: Anomaly Scoring / Reliability
**Reality**: Two bare `# EMERGENCY` comments in `advanced_scoring.py` suppress
configuration loading warnings (line 70) and initialization logging (line 192). These
represent unresolved hotfixes from an earlier debugging session. The root cause has
never been documented.

If the ATLAS scorer silently fails to load its weights config, every analysis score from
`AdvancedScoreCalculator` will use default/zero weights — producing results that look
valid but are computed incorrectly.

**Fix**:
1. Remove suppression comments.
2. Run the system and observe what warnings/logs appear.
3. Diagnose and fix the root cause (likely a missing config file path or weight
   normalization issue).
4. Re-add logging at appropriate severity.

**ADR/DDD ref**: ADR-008

---

### G-023: `aneos_menu.py` Monolith Is Untestable

**Area**: Engineering Quality / Maintainability
**Reality**: `aneos_menu.py` is 10,781 lines, 265 functions, 1 class. The CI pipeline
exercises module-level imports and the dedicated test suite, but none of the 265 menu
functions have unit tests. The functions mix UI rendering, data fetching, scientific
computation, and business logic in a single flat namespace.

**Impact**: Any regression in the 265 functions is invisible to CI. Adding functionality
requires scrolling 10K+ lines to understand context. The existing pollers run in a
separate terminal and their results flow into functions that have no coverage.

**Fix**: Phase 4 refactor — extract cohesive functional groups into separate modules:
- `aneos_core/ui/analysis_menu.py` — Single NEO, Batch, Enhanced Validation
- `aneos_core/ui/detection_display.py` — Display/formatting for detection results
- `aneos_core/ui/reporting_menu.py` — Reports, accuracy sub-reports
- `aneos_core/ui/system_menu.py` — Status, database, monitoring, dev tools

Each extracted module can then have its own test file. `aneos_menu.py` becomes
a thin orchestrator that imports from these modules.

This is a significant refactor; plan for 2–3 days and a careful test-before/test-after
protocol.

**ADR/DDD ref**: DDD BC-8

---

### G-024: Temporal Analysis Limited to Single Epoch

**Area**: Scientific Analysis / Feature Completeness
**Reality**: `aneos_menu.py:_generate_orbital_history()` returns a single current-epoch
data point with label `[OBSERVED - current epoch only]`. The temporal anomaly detection
feature (sigma evolution over time, orbital evolution) requires multi-epoch ephemeris
data, which was intended to come from JPL Horizons.

Since Horizons is uninstantiated (G-021), there is no practical path to real temporal
data. The interactive investigation mode labels propulsion-signature steps as `[MODEL]`
correctly, but the underlying orbital history is still point-in-time.

**Fix**: Depends on G-021 (Horizons source implementation). Once Horizons is wired:
implement `_generate_orbital_history()` to call `HorizonsSource.fetch_ephemeris(des,
start, end, step='30d')` and return the time series. Fall back to single epoch with
clear label if Horizons fails.

**ADR/DDD ref**: ADR-002, DDD BC-2

---

## P3 — Low Severity Gaps

### G-015: ML Module Has No Activation Path

**Area**: ML / Future Readiness
**Status**: OPEN (unchanged from v1.0)
**Note**: Now unblocked — ground truth dataset exists (G-001 closed). The labelled
training data needed for `TrainingPipeline.train_from_ground_truth()` is available.

**Fix**: Implement `FeatureVector.from_ground_truth_object()` in `ml/features.py`; run
`TrainingPipeline`; register `MLDetectorWrapper` in `DetectionManager`. Per original
plan, set ML detector priority to 5 (lowest) until performance validated against rule-based.

**ADR/DDD ref**: ADR-033

---

### G-016: Redis Integration Unverified

**Area**: Infrastructure
**Status**: OPEN (unchanged from v1.0)
**Note**: Audit found zero Redis-related code in `aneos_api/`. Redis is declared in
`docker-compose.yml` but has no application-layer integration at all — not even a
health-check client. The Phase 4 tasks (4.1 Redis health check, 4.2 Redis streaming
verify) remain unstarted.

**ADR/DDD ref**: ADR-037

---

### G-017 (RESIDUAL): CacheManager Partial Pickle Usage

**Area**: Security / Data Integrity
**Status**: Partially closed. JSON serialization was added as the primary path. However,
`cache.py:288` still falls back to `pickle.dumps(payload)` for complex object types,
and `cache.py:304` still calls `pickle.loads(cache_path.read_bytes())` on read.

The security risk (deserialization of untrusted data) remains for any cache entry that
triggered the pickle fallback path.

**Fix**: Replace the pickle fallback with a `dataclasses.asdict()` + `json.dumps()`
path for dataclass instances, and a custom JSON encoder for non-serializable types.
Delete residual `.pickle` cache files on first startup after change.

**ADR/DDD ref**: ADR-003

---

### G-018: Chunk Boundary Overlap Not Validated

**Area**: Data Acquisition / Scientific Accuracy
**Status**: OPEN (unchanged from v1.0)

**ADR/DDD ref**: ADR-004

---

### G-025: AI Validation Annotator May Overstate Confidence

**Area**: Scientific Communication
**Reality**: `aneos_core/reporting/ai_validation.py` (`AIValidationAnnotator`) generates
academic-language descriptions of detection results. Given that Bayesian posteriors are
bounded to ~4% (G-020), there is a risk the annotator produces language like "high
confidence artificial detection" for objects with σ=5 but P(artificial)=3.7%.

This was flagged in MEMORY.md ("generates academic language that may overstate confidence")
but never audited or fixed.

**Fix**: Audit `AIValidationAnnotator.generate()` output for any language that implies
the Bayesian posterior is high. Replace with language that correctly frames sigma-level
as rarity, not probability. Add a standard disclaimer to all generated annotations.

**ADR/DDD ref**: ADR-039

---

## Gap Summary Table (v2.0)

| ID | Title | Priority | Status | Area |
|----|-------|----------|--------|------|
| G-001 | No ground truth dataset | ~~P0~~ | **CLOSED** | Detection |
| G-002 | Physical indicator category missing | ~~P1~~ | **CLOSED** | Scoring |
| G-003 | Pipeline bypasses canonical detector | ~~P1~~ | **CLOSED** | Detection |
| G-004 | Silent simulation fallback (startup) | **P1** | RESIDUAL | Data Integrity |
| G-005 | Docker deployment broken | ~~P1~~ | **CLOSED** | Infrastructure |
| G-006 | Monitoring depends on deferred ML | ~~P1~~ | **CLOSED** | Monitoring |
| G-007 | README overstates production readiness | ~~P1~~ | **CLOSED** | Documentation |
| G-008 | Seven detector files, no archive | **P2** | OPEN | Maintainability |
| G-009 | Dual scoring systems + EMERGENCY suppression | **P2** | OPEN | Scoring |
| G-010 | No CI/CD pipeline | ~~P2~~ | **CLOSED** | Engineering |
| G-011 | OrbitalElements conflates domain models | **P2** | OPEN | Domain Model |
| G-012 | API has no DTO layer | **P2** | OPEN | API |
| G-013 | Test harness inside production module | **P2** | OPEN | Code Org |
| G-014 | No maintained OpenAPI spec | **P2** | OPEN | API Docs |
| G-015 | ML has no activation path | **P3** | OPEN (unblocked) | ML |
| G-016 | Redis integration unverified | **P3** | OPEN | Infrastructure |
| G-017 | Pickle cache security risk | **P3** | RESIDUAL | Security |
| G-018 | Chunk boundary overlap unvalidated | **P3** | OPEN | Data Accuracy |
| G-019 | Ground truth corpus too small for FPR measurement | **P1** | NEW | Scientific |
| G-020 | Bayesian posterior ceiling undocumented | **P1** | NEW | Communication |
| G-021 | Multi-source enrichment is SBDB-only | **P2** | NEW | Data Sources |
| G-022 | EMERGENCY suppressions mask unknown scoring issue | **P2** | NEW | Reliability |
| G-023 | `aneos_menu.py` monolith untestable | **P2** | NEW | Engineering |
| G-024 | Temporal analysis limited to single epoch | **P2** | NEW | Analysis |
| G-025 | AI annotator may overstate confidence | **P3** | NEW | Communication |

**Totals**: 0 P0 · 3 P1 (2 new, 1 residual) · 9 P2 (5 open, 3 new, 1 residual) ·
5 P3 (3 open, 1 new, 1 residual) · 7 CLOSED

---

## Priority Order for Phase 4

Dependencies drive sequencing. The two new P1 gaps (G-019, G-020) and the residual
P1 (G-004) should be addressed before the P2 architecture work.

```
Phase 4A — Scientific Accuracy (2–3 days)
  G-019  Expand artificial corpus: Horizons NAIF IDs for Pioneer/Voyager/New Horizons
  G-020  Document Bayesian ceiling; update UI, API, and docs
  G-004  Make aneos_menu preflight failure visible and acknowledged (not silent)

Phase 4B — Data Source Completeness (2–3 days)
  G-021  Implement NEODyS/MPC _make_request; instantiate Horizons in DataFetcher
  G-024  Wire orbital history to Horizons ephemeris once 4B complete

Phase 4C — Architecture Cleanup (3–4 days)
  G-008  Archive superseded detector files
  G-013  Move test harness to tests/
  G-009  Diagnose EMERGENCY suppressions; designate canonical scorer
  G-022  Same session as G-009

Phase 4D — API and Model Quality (2–3 days)
  G-012  Add API DTO/schema layer
  G-014  Generate and commit OpenAPI spec; add CI drift check
  G-011  Separate OrbitalElements from PhysicalProperties (broad change, test carefully)

Phase 4E — Future Capabilities (3–5 days)
  G-015  Train ML classifier on ground truth; register in DetectionManager
  G-016  Add Redis health check; verify streaming endpoint integration
  G-017  Complete pickle → JSON migration in CacheManager
  G-018  Add chunk boundary deduplication test
  G-025  Audit and fix AI annotator language
  G-023  Begin aneos_menu.py decomposition (long-running; 2–3 days)
```

---

*Next step: derive Phase 4 implementation plan from this gap list.*
