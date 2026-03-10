# aNEOS Gap Closure — Implementation Plan

**Version**: 1.0
**Date**: 2026-03-06
**Input**: CLOSURE_PLAN.md v1.0, GAP_ANALYSIS.md v1.0
**Workflow**: C&C + Implementation + Q&A per CLAUDE.md

Each task is atomic. Implementation proceeds top-to-bottom. A phase gate
must pass before the next phase begins. Review and acknowledge this entire
document before work starts.

---

## Phase Gate Definitions

| Gate | Criteria |
|------|----------|
| G1 | All Phase 1 tasks complete; `python aneos.py status` shows no silent fallback; CI green |
| G2 | All Phase 2 tasks complete; blind test results documented; README updated |
| G3 | All Phase 3 tasks complete; one detector, one scorer, clean domain model, DTO layer in place |
| G4 | All Phase 4 tasks complete; 18/18 gaps closed |

---

# PHASE 1 — Correctness Foundation

Fix anything that produces wrong or misleading results. No new features until Gate G1.

---

## Task 1.1 — Create `IntegrationError` and `DataSourceUnavailableError`

**Gap**: G-004
**Files to create**: `aneos_core/utils/errors.py`

Create a new module with two custom exceptions:

```
IntegrationError(RuntimeError)
  - Raised when a required pipeline component is not importable
  - Message must name the missing component and the install command

DataSourceUnavailableError(RuntimeError)
  - Raised when all four external APIs fail during a fetch
  - Message must list which sources were tried and their failure reasons
```

No other changes in this task.

---

## Task 1.2 — Create `preflight_check()` Health Gate

**Gap**: G-004
**Files to create**: `aneos_core/utils/health.py`

Implement `preflight_check() -> dict` that validates each required
integration and returns a status dict. It must check:

1. Import availability of `aneos_core.pipeline.automatic_review_pipeline`
2. Import availability of `aneos_core.analysis.pipeline`
3. Network reachability of each API in `APIConfig.data_sources_priority`
   (SBDB, NEODyS, MPC, Horizons) — one lightweight HEAD/OPTIONS request each
4. Writability of the cache directory (`neo_data/cache/`)
5. Writability of the results directory (`neo_data/pipeline_results/`)

Return format:
```python
{
  "pipeline": {"status": "ok" | "error", "detail": str},
  "analysis": {"status": "ok" | "error", "detail": str},
  "sbdb":     {"status": "ok" | "error", "detail": str},
  "neodys":   {"status": "ok" | "error", "detail": str},
  "mpc":      {"status": "ok" | "error", "detail": str},
  "horizons": {"status": "ok" | "error", "detail": str},
  "cache_dir":{"status": "ok" | "error", "detail": str},
  "results_dir":{"status":"ok" | "error", "detail": str},
}
```

---

## Task 1.3 — Replace Silent Fallback in `pipeline_integration.py`

**Gap**: G-004
**Files to modify**: `aneos_core/integration/pipeline_integration.py`

1. At module top: import `IntegrationError` from `aneos_core.utils.errors`.
2. In the `except ImportError` block that sets `HAS_PIPELINE_COMPONENTS = False`:
   replace with `raise IntegrationError(f"Required component missing: {e}. "
   f"Run: pip install -r requirements.txt")`.
3. Remove the `HAS_PIPELINE_COMPONENTS` flag and all guards that reference it.
4. Remove the `PIPELINE_IMPORT_ERROR` variable.

---

## Task 1.4 — Replace Silent Fallback in `data/fetcher.py`

**Gap**: G-004
**Files to modify**: `aneos_core/data/fetcher.py`

1. Import `DataSourceUnavailableError` from `aneos_core.utils.errors`.
2. In the multi-source fetch loop: after all sources are tried and all fail,
   raise `DataSourceUnavailableError` with the list of tried sources and
   their failure reasons.
3. Do not modify the circuit breaker logic — it should still manage per-source
   state; the error is raised only when the merged result is empty after all
   sources are exhausted.

---

## Task 1.5 — Wire `preflight_check()` into Entry Points

**Gap**: G-004
**Files to modify**: `aneos_menu.py`, `aneos_core/pipeline/automatic_review_pipeline.py`, `aneos.py`

1. In `aneos_menu.py`: at startup, call `preflight_check()`. If any required
   check fails, print the status table and prompt the user to confirm before
   continuing. Do not silently proceed.
2. In `automatic_review_pipeline.py:AutomaticReviewPipeline.__init__()`:
   call `preflight_check()` and raise `IntegrationError` if the pipeline
   or analysis components are unavailable.
3. In `aneos.py`: the `status` command must call `preflight_check()` and
   display the returned dict as a formatted table.

---

## Task 1.6 — Wire Pipeline to `DetectionManager`

**Gap**: G-003
**Files to modify**: `aneos_core/pipeline/automatic_review_pipeline.py`

1. Remove line 52:
   ```python
   from ..detection.multimodal_sigma5_artificial_neo_detector import MultiModalSigma5ArtificialNEODetector
   ```
2. Add at the top of the file:
   ```python
   from ..detection.detection_manager import DetectionManager, DetectorType
   ```
3. In `AutomaticReviewPipeline.__init__()`: instantiate
   `self.detector = DetectionManager(preferred_detector=DetectorType.AUTO)`.
4. In the first-stage review function: replace direct
   `MultiModalSigma5ArtificialNEODetector().analyze_neo(...)` calls with
   `self.detector.detect(orbital_elements, physical_data)`.
5. Ensure the returned `DetectionResult` interface is used — not the
   multimodal-specific `ProductionDetectionResult`.

---

## Task 1.7 — Add Detector-Selection Test

**Gap**: G-003
**Files to modify**: `tests/test_detection_manager.py` (create if absent)

Add one test: instantiate `DetectionManager(DetectorType.AUTO)`, assert that
the selected detector is `ValidatedSigma5ArtificialNEODetector` when all
detectors load successfully.

---

## Task 1.8 — Decouple `monitoring/alerts.py` from ML

**Gap**: G-006
**Files to modify**: `aneos_core/monitoring/alerts.py`

1. Replace the hard import:
   ```python
   from ..ml.prediction import Alert as MLAlert, PredictionResult
   ```
   with:
   ```python
   try:
       from ..ml.prediction import Alert as MLAlert, PredictionResult
       _HAS_ML_ALERTS = True
   except ImportError:
       MLAlert = None
       PredictionResult = None
       _HAS_ML_ALERTS = False
   ```
2. Wrap every usage of `MLAlert` and `PredictionResult` inside the module
   with `if _HAS_ML_ALERTS:` guards.
3. Where an `MLAlert` would have been created but ML is unavailable, log a
   `DEBUG` message and skip — do not raise.

---

## Task 1.9 — Add Monitoring Import Test

**Gap**: G-006
**Files to modify**: `tests/test_monitoring_no_ml.py` (create)

Add one test that imports `aneos_core.monitoring.alerts` via a subprocess
with `PYTHONPATH` set to an environment that lacks scikit-learn, and asserts
exit code 0.

---

## Task 1.10 — Fix Docker Deployment: `init.sql`

**Gap**: G-005
**Files to create**: `init.sql`

Create a minimal PostgreSQL init script at repo root:
```sql
-- aNEOS database schema bootstrap
-- SQLAlchemy migrations will extend this schema at runtime.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

---

## Task 1.11 — Fix Docker Deployment: SSL Stub

**Gap**: G-005
**Files to create**: `ssl/README.md`

Document that self-signed certs must be placed here. Include the exact
`openssl` command to generate them:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/server.key -out ssl/server.crt \
  -subj "/CN=localhost"
```

---

## Task 1.12 — Make Nginx HTTP-Only by Default

**Gap**: G-005
**Files to modify**: `docker-compose.yml`, `nginx.conf`

1. In `docker-compose.yml`: change the Nginx volume mount from
   `./ssl:/etc/nginx/ssl` to an optional override. Remove the `443` port
   mapping from the base `docker-compose.yml`.
2. Create `docker-compose.override.yml.example` that adds HTTPS with the
   ssl volume mount — users copy this to `docker-compose.override.yml` when
   they have real certificates.
3. Update `nginx.conf` to serve HTTP on port 80 by default with a comment
   block showing how to enable HTTPS.

---

## Task 1.13 — Create `Makefile` with `bootstrap` Target

**Gap**: G-005
**Files to create**: `Makefile`

Targets:
```
make bootstrap   — generates init.sql if missing; generates self-signed SSL
                   certs into ssl/; creates .env from .env.example if missing
make up          — runs docker-compose up -d
make test        — runs pytest tests/
make spec        — generates docs/api/openapi.json (Phase 3)
make lint        — runs ruff or flake8
```

---

## Task 1.14 — Create GitHub Actions CI Workflow

**Gap**: G-010 (pulled into Phase 1)
**Files to create**: `.github/workflows/ci.yml`

Two jobs:

**Job: test**
- Trigger: push to `main`, all pull requests
- Runner: `ubuntu-latest`, Python 3.11
- Steps: checkout → `pip install -r requirements-core.txt` →
  `python -m pytest tests/ -v --tb=short`

**Job: docker-build**
- Trigger: same as above
- Steps: checkout → `make bootstrap` → `docker-compose build`

---

## Phase Gate G1 Checklist

Before proceeding to Phase 2, verify all of the following:

- [ ] `python aneos.py status` prints a status table; no check silently passes
- [ ] Running without API access raises `DataSourceUnavailableError`, not
      mock data
- [ ] `from aneos_core.monitoring import alerts` succeeds without scikit-learn
- [ ] `docker-compose build` succeeds from a clean clone after `make bootstrap`
- [ ] GitHub Actions CI is green on `main`
- [ ] `pytest tests/ -v` passes (61+ tests)

---

# PHASE 2 — Scientific Completeness

Close the gap between the concept document's methodology and the implementation.
Begin only after Gate G1.

---

## Task 2.1 — Implement `indicators/physical.py`

**Gap**: G-002
**Files to create**: `aneos_core/analysis/indicators/physical.py`

Implement three concrete `AnomalyIndicator` subclasses:

**`DiameterAnomalyIndicator(NumericRangeIndicator)`**
- Normal range: `(ThresholdConfig.diameter_min, ThresholdConfig.diameter_max)`
  i.e. `(0.1 km, 10.0 km)`
- Score = 0.0 if within range; increases linearly outside range
- Return `IndicatorResult` with `metadata={'diameter_km': value}`

**`AlbedoAnomalyIndicator(NumericRangeIndicator)`**
- Normal range: `(ThresholdConfig.albedo_min, ThresholdConfig.albedo_max)`
  i.e. `(0.05, 0.50)`
- Extreme flag: albedo > `ThresholdConfig.albedo_artificial` (0.60)
  → add `contributing_factors=['extreme_albedo_artificial_signature']`
- Return `IndicatorResult` with `metadata={'albedo': value}`

**`SpectralAnomalyIndicator(AnomalyIndicator)`**
- Wraps `SpectralOutlierAnalyzer.analyze(neo_data)` from
  `aneos_core/validation/spectral_outlier_analysis.py`
- Converts `SpectralOutlierResult.outlier_sigma` to `raw_score` (clamped 0–1)
- Returns `IndicatorResult` with `metadata={'spectral_class': matched_class,
  'outlier_sigma': value}`
- If spectral data absent: return confidence=0.0, raw_score=0.0

---

## Task 2.2 — Register Physical Indicators in Pipeline

**Gap**: G-002
**Files to modify**: `aneos_core/analysis/pipeline.py`

1. Import the three new indicators at the top of the file.
2. In `PipelineConfig.__post_init__()`, add the three indicator names to
   `indicator_configs` with `IndicatorConfig(weight=1.0, enabled=True,
   confidence_threshold=0.5)`.
3. In `AnalysisPipeline._load_indicators()` (or equivalent), instantiate
   and register the three new indicator objects alongside the existing 15.

---

## Task 2.3 — Verify Physical Category Scoring

**Gap**: G-002
**Files to modify**: `aneos_core/analysis/scoring.py`

Confirm (do not change) that `ScoreCalculator.indicator_categories['physical']`
already maps `['diameter_anomalies', 'albedo_anomalies', 'spectral_anomalies']`.
These names must exactly match the `.name` attributes set in Task 2.1.
If they differ, align the indicator names to match the existing mapping.

---

## Task 2.4 — Write Physical Indicator Unit Tests

**Gap**: G-002
**Files to create**: `tests/test_physical_indicators.py`

Write tests covering:
- `DiameterAnomalyIndicator`: normal diameter → score ~0; extreme diameter → score > 0
- `AlbedoAnomalyIndicator`: normal albedo → score ~0; albedo > 0.60 → contributing_factors includes `extreme_albedo_artificial_signature`
- `SpectralAnomalyIndicator`: missing spectral data → confidence = 0.0
- All three: missing `orbital_elements` → returns zero-score result, does not raise

---

## Task 2.5 — Compile Confirmed Artificial Objects Dataset

**Gap**: G-001
**Files to modify**: `aneos_core/datasets/ground_truth_dataset_preparation.py`

Operationalise `GroundTruthDatasetBuilder.compile_verified_artificial_objects()`.
Query JPL Horizons via `astroquery.jplhorizons` for heliocentric orbital elements
of each object. Store as `GroundTruthObject` instances.

Minimum required objects:
| Object | NAIF ID | Notes |
|--------|---------|-------|
| Voyager 1 | -31 | Heliocentric escape trajectory |
| Voyager 2 | -32 | Heliocentric escape trajectory |
| Pioneer 10 | -23 | Heliocentric escape trajectory |
| Pioneer 11 | -24 | Heliocentric escape trajectory |
| New Horizons | -98 | Heliocentric, post-Pluto |
| Tesla Roadster | 2018-017A | Heliocentric, known orbital elements |
| OSIRIS-REx | -101 | Deep space probe |
| Hayabusa2 | -37 | JAXA probe, known elements |

For each: fetch `a, e, i, om, w, M` at a reference epoch. Set
`is_artificial=True`, `source='JPL Horizons'`, and record the
`verification_notes` with the retrieval date.

---

## Task 2.6 — Compile Natural NEO Control Set

**Gap**: G-001
**Files to modify**: `aneos_core/datasets/ground_truth_dataset_preparation.py`

Implement `GroundTruthDatasetBuilder.compile_natural_neo_control_set()`.
Query JPL SBDB for 200 well-characterised natural NEOs:
- Filter: multi-opposition, `H < 22`, orbital uncertainty parameter `U <= 3`
- Fetch: `a, e, i, om, w, M, diameter, albedo` per object
- Store as `GroundTruthObject(is_artificial=False, source='JPL SBDB')`

---

## Task 2.7 — Build Blind Test Split

**Gap**: G-001
**Files to modify**: `aneos_core/datasets/ground_truth_dataset_preparation.py`

Implement `GroundTruthDatasetBuilder.build_blind_test_split(seed=42)`:
1. Merge artificial + natural objects into one list.
2. Shuffle with `random.Random(seed)` for reproducibility.
3. 70% → training set; 30% → blind test set.
4. Anonymise blind test set: replace `object_id` with `blind_<n>`;
   store the mapping in a separate file that is not passed to the detector.
5. Write training set to `development/test-results/ground_truth_train.json`.
6. Write anonymised blind test set to
   `development/test-results/ground_truth_blind.json`.
7. Store label mapping in `development/test-results/ground_truth_labels.json`
   (gitignored during testing; revealed only after detector runs).

---

## Task 2.8 — Run Detectors Against Blind Test Set

**Gap**: G-001
**Files to modify**: `aneos_core/detection/artificial_neo_test_suite.py`
(will be relocated in Phase 3; edit in place for now)

Implement `ArtificialNEOTestSuite.run_blind_test(dataset_path, detector_type)`:
1. Load `ground_truth_blind.json`.
2. For each object, call `DetectionManager(detector_type).detect(orbital_elements)`.
3. Record: `object_id, predicted_is_artificial, predicted_sigma, predicted_confidence`.
4. Write predictions to
   `development/test-results/blind_test_predictions_<detector_type>.json`.

Run for `DetectorType.VALIDATED` and `DetectorType.MULTIMODAL`.

---

## Task 2.9 — Compute and Document Accuracy Metrics

**Gap**: G-001
**Files to create**: `development/test-results/ground_truth_accuracy_report.md`

Reveal blind test labels. For each detector run:
- Compute: TP, TN, FP, FN, precision, recall, F1, FPR at sigma-5 threshold.
- Record empirical FPR; compare to claimed 5.7×10⁻⁷.
- If empirical FPR > 5.7×10⁻⁷, document the gap and proceed to Task 2.10.

---

## Task 2.10 — Calibrate Detection Thresholds (up to 3 iterations)

**Gap**: G-001
**Files to modify**: `aneos_core/config/advanced_scoring_weights.json`,
`aneos_core/config/settings.py:ThresholdConfig`

Per CLAUDE.md: maximum 3 calibration cycles. For each cycle:
1. Adjust `AdvancedScoringConfig` weights in `advanced_scoring_weights.json`
   based on which indicator categories are generating false positives.
2. Re-run Task 2.8 (full blind test) with the updated weights.
3. Re-compute Task 2.9 metrics.
4. Record the before/after FPR in `ground_truth_accuracy_report.md`.
Stop when empirical FPR ≤ 5.7×10⁻⁷ or 3 iterations exhausted.

---

## Task 2.11 — Update Documentation to Reflect Measured Reality

**Gap**: G-007
**Files to modify**: `README.md`, `ROADMAP.md`, `CHANGELOG.md`,
`docs/engineering/maturity_assessment.md`, `docs/engineering/sigma5_success_criteria.md`

1. `README.md`: replace "v1.0.0 - Production Ready" with actual version and
   maturity level. Replace theoretical FPR with measured FPR from Task 2.9,
   or state "empirically measured: X" or "calibration in progress".
2. `docs/engineering/sigma5_success_criteria.md`: add a
   `## Verification Results` section with precision/recall/FPR numbers
   and the date measured.
3. `docs/engineering/maturity_assessment.md`: update "Key Risks" section to
   reflect which risks have been closed.
4. `ROADMAP.md`: mark Phase 1–2 complete.
5. `CHANGELOG.md`: add entries for each gap closure.

---

## Phase Gate G2 Checklist

Before proceeding to Phase 3, verify all of the following:

- [ ] `pytest tests/test_physical_indicators.py` passes
- [ ] Ground truth dataset contains ≥ 8 artificial + ≥ 100 natural objects
- [ ] Blind test predictions exist for VALIDATED and MULTIMODAL detectors
- [ ] `ground_truth_accuracy_report.md` documents precision/recall/FPR
- [ ] README no longer claims "Production Ready" without empirical backing
- [ ] `docs/engineering/sigma5_success_criteria.md` has a Verification Results section
- [ ] All Phase 1 tests still pass

---

# PHASE 3 — Architecture Cleanup

Remove structural debt. Begin only after Gate G2.

---

## Task 3.1 — Archive Superseded Detector Files

**Gap**: G-008
**Files to create/move**:
- Create `aneos_core/detection/_archive/__init__.py` (empty)
- Move into `_archive/`:
  - `sigma5_artificial_neo_detector.py`
  - `corrected_sigma5_artificial_neo_detector.py`
  - `production_artificial_neo_detector.py`
  - `multimodal_sigma5_artificial_neo_detector.py`
  - `sigma5_corrected_statistical_framework.py`

**Files to modify**: `aneos_core/detection/detection_manager.py`

1. Remove `DetectorType.BASIC`, `DetectorType.CORRECTED`,
   `DetectorType.PRODUCTION`, `DetectorType.MULTIMODAL` from the enum
   (or mark as `DEPRECATED` with a comment pointing to `_archive/`).
2. Keep only `DetectorType.VALIDATED` and `DetectorType.AUTO`.
3. Update `_load_available_detectors()` to load only `VALIDATED`.
4. Run `pytest tests/` — fix any broken imports caused by removed detectors.

---

## Task 3.2 — Move Test Harness Out of Production Module

**Gap**: G-013
**Files to move**:
- `aneos_core/detection/artificial_neo_test_suite.py`
  → `tests/detection/test_artificial_neo_suite.py`

Fix import paths. Verify `pytest tests/detection/` passes.

---

## Task 3.3 — Designate Canonical Scoring System

**Gap**: G-009

Using the accuracy metrics from Task 2.9:
- Compare F1 score of Standard scoring (`scoring.py`) vs ATLAS
  (`advanced_scoring.py`) on the ground truth dataset.
- The system with higher F1 is designated canonical.

**If ATLAS wins**:
- In `aneos_core/analysis/pipeline.py`: ensure `EnhancedAnalysisPipeline`
  (which uses ATLAS) is the default pipeline returned by `create_analysis_pipeline()`.
- Mark `ScoreCalculator` in `scoring.py` as `# DEPRECATED: use AdvancedScoreCalculator`.
- Remove the `# EMERGENCY` comment in `advanced_scoring.py`; restore the
  configuration warning it was suppressing.

**If Standard wins**:
- Mark `AdvancedScoreCalculator` in `advanced_scoring.py` as deprecated.
- Update `EnhancedAnalysisPipeline` to use `ScoreCalculator` internally.

In both cases: add an ADR-008 update entry in `docs/architecture/ADR.md`
documenting the decision and the empirical basis.

---

## Task 3.4 — Separate `OrbitalElements` from Physical Properties

**Gap**: G-011
**Files to modify**: `aneos_core/data/models.py`, all callsites

1. Remove from `OrbitalElements`: `diameter`, `albedo`, `rot_per`,
   `spectral_type`.
2. Confirm these fields exist in `PhysicalProperties`; add any that are
   missing.
3. Search the entire codebase for reads of `orbital_elements.diameter`,
   `orbital_elements.albedo`, `orbital_elements.rot_per`,
   `orbital_elements.spectral_type`.
4. For each callsite found: update to read from
   `neo_data.physical_properties.{field}` instead.
5. Run `pytest tests/` — fix all failures.

---

## Task 3.5 — Create API DTO / Schema Layer

**Gap**: G-012
**Files to create**: `aneos_api/schemas/__init__.py`,
`aneos_api/schemas/analysis.py`, `aneos_api/schemas/detection.py`,
`aneos_api/schemas/impact.py`, `aneos_api/schemas/health.py`

Define Pydantic v2 `BaseModel` response schemas for each endpoint group.
Minimum required schemas:

`analysis.py`: `AnalysisResponse(designation, overall_score, classification,
confidence, indicator_scores: dict, risk_factors: list, created_at)`

`detection.py`: `DetectionResponse(designation, is_artificial,
artificial_probability, sigma_level, classification, confidence)`

`impact.py`: `ImpactResponse(designation, collision_probability,
moon_collision_probability, moon_earth_ratio, impact_energy_mt,
crater_diameter_km, risk_level, time_to_impact_years)`

`health.py`: `HealthResponse(status, checks: dict, version, timestamp)`

**Files to modify**: `aneos_api/endpoints/analysis.py`,
`aneos_api/endpoints/enhanced_analysis.py`, `aneos_api/endpoints/monitoring.py`

Update each endpoint's `response_model=` parameter and its return statement
to construct the appropriate schema from the domain result.

---

## Task 3.6 — Generate and Commit OpenAPI Spec

**Gap**: G-014
**Files to create**: `docs/api/openapi.json`

1. Add `make spec` to `Makefile`:
   ```makefile
   spec:
       python -c "from aneos_api.app import app; \
         import json; print(json.dumps(app.openapi(), indent=2))" \
       > docs/api/openapi.json
   ```
2. Run `make spec` and commit `docs/api/openapi.json`.
3. Add a CI step in `.github/workflows/ci.yml`:
   - Run `make spec` and compare output to the committed file.
   - Fail if they differ (spec drift detection).
4. Add a note to `docs/api/rest-api.md` pointing readers to `openapi.json`
   as the authoritative reference. Retire the hand-maintained content.

---

## Phase Gate G3 Checklist

Before proceeding to Phase 4, verify all of the following:

- [ ] `aneos_core/detection/` contains only: `__init__.py`,
      `detection_manager.py`, `validated_sigma5_artificial_neo_detector.py`,
      `_archive/`
- [ ] No test code exists under `aneos_core/`
- [ ] One scoring system is documented as canonical in `ADR.md`
- [ ] `OrbitalElements` has no physical property fields
- [ ] `pytest tests/` passes with zero failures
- [ ] `aneos_api/schemas/` exists with at minimum 4 schema files
- [ ] `docs/api/openapi.json` is committed and CI drift check is active
- [ ] CI still green on `main`

---

# PHASE 4 — Future Capabilities

Activate deferred subsystems. Begin only after Gate G3.

---

## Task 4.1 — Add Redis Health Check to `preflight_check()`

**Gap**: G-016
**Files to modify**: `aneos_core/utils/health.py`

Add a `redis` key to the `preflight_check()` return dict:
- Attempt a `PING` to the configured Redis URL (`ANEOS_REDIS_URL` env var,
  default `redis://localhost:6379/0`).
- Status: `"ok"` if PING returns PONG; `"error"` with detail otherwise.
- If Redis is unavailable, do not raise — mark as `"error"` and continue;
  Redis is optional for local development.

---

## Task 4.2 — Verify Redis Application Integration

**Gap**: G-016
**Files to modify**: `aneos_api/endpoints/streaming.py`

1. Implement a minimal Redis pub/sub message on the `/stream/health` endpoint:
   publish `{"event": "health_ping", "timestamp": <iso>}` to a
   `aneos:events` channel on each request.
2. Add an integration test that starts the streaming endpoint, verifies a
   message arrives in the Redis channel, and asserts the round-trip time
   is < 100 ms.

---

## Task 4.3 — Replace Pickle with JSON in `CacheManager`

**Gap**: G-017
**Files to modify**: `aneos_core/data/cache.py`

1. Replace all `pickle.dumps(value)` with `json.dumps(value, default=str)`.
2. Replace all `pickle.loads(data)` with `json.loads(data)`.
3. For values that include `datetime` objects or dataclass instances, apply
   `dataclasses.asdict()` before serialisation and reconstruct with the
   appropriate constructor after deserialisation.
4. Delete any `.pickle` cache files on first startup after the change
   (add a migration note in the startup log).
5. Update `CacheManager` tests to assert no pickle files are written.

---

## Task 4.4 — Validate Chunk Boundary Overlap

**Gap**: G-018
**Files to create**: `tests/test_chunk_boundaries.py`

Write a test that mocks two adjacent 5-year polling chunks with a 7-day
overlap:
- Object A: close approach at the last day of chunk 1
- Object B: close approach at the first day of chunk 2
- Object C: close approach within the 7-day overlap window

Assert after merging:
- A, B, C all appear in the merged result
- No object appears more than once (deduplication by designation)

**Files to modify**: `aneos_core/polling/historical_chunked_poller.py`

If the test fails: add a `_deduplicate(objects)` step in the chunk-merge
logic that removes duplicate designations, keeping the entry with the more
complete data (non-None field count).

---

## Task 4.5 — Train and Validate ML Classifier

**Gap**: G-015
**Depends on**: Ground truth dataset from Task 2.5–2.7

**Files to modify**: `aneos_core/ml/training.py`, `aneos_core/ml/features.py`,
`aneos_core/ml/prediction.py`

1. In `features.py`: implement `FeatureVector.from_ground_truth_object(obj)`
   that converts a `GroundTruthObject` into a numeric feature array using:
   `[a, e, i, om, w, diameter, albedo]` (7 features minimum).
2. In `training.py`: implement `TrainingPipeline.train_from_ground_truth(path)`:
   - Load `ground_truth_train.json`
   - Extract feature vectors
   - Fit `IsolationForest` (unsupervised) and `RandomForestClassifier`
     (supervised, requires labels) via `sklearn`
   - Persist fitted models to `models/isolation_forest.joblib` and
     `models/random_forest.joblib`
3. In `prediction.py`: implement `RealTimePredictor.predict_from_neo_data(neo_data)`
   that loads the persisted model and returns a prediction dict.

---

## Task 4.6 — Register ML Detector in `DetectionManager`

**Gap**: G-015
**Files to modify**: `aneos_core/detection/detection_manager.py`,
`aneos_core/interfaces/detection.py`

1. Add `DetectorType.ML = "ml"` to the enum.
2. Add a `MLDetectorWrapper` in `detection_manager.py` that calls
   `RealTimePredictor.predict_from_neo_data()` and normalises the output
   to `DetectionResult`.
3. Set ML detector priority to 5 (lowest) — it supplements but does not
   replace rule-based detection until performance is validated.
4. Add a test: `DetectionManager(DetectorType.ML).detect(orbital_elements)`
   returns a valid `DetectionResult` (not an exception).

---

## Task 4.7 — Document ML vs Rule-Based Performance Comparison

**Gap**: G-015
**Files to create**: `development/test-results/ml_vs_rulebased_comparison.md`

Run both `DetectorType.VALIDATED` and `DetectorType.ML` against the blind
test set. Record side-by-side:
- Precision, Recall, F1, FPR
- False positive objects (which natural NEOs were misclassified)
- False negative objects (which artificial objects were missed)
- Recommendation: which detector to use as default, and under what conditions

---

## Phase Gate G4 — Full Closure Checklist

All 18 gaps are closed when all of the following pass:

- [ ] `python aneos.py status` shows all checks PASS; no silent fallback
- [ ] Ground truth blind test documented: precision, recall, FPR for VALIDATED
- [ ] Empirical FPR documented (may or may not have reached 5.7×10⁻⁷)
- [ ] `aneos_core/analysis/indicators/physical.py` exists with 3 indicators
- [ ] `automatic_review_pipeline.py` uses `DetectionManager`, not hardcoded import
- [ ] `aneos_core/detection/` has exactly 2 active files + `_archive/`
- [ ] One canonical scoring system designated in `ADR.md`
- [ ] `OrbitalElements` contains no physical property fields
- [ ] `aneos_api/schemas/` contains DTO schemas; endpoints use them
- [ ] `docs/api/openapi.json` generated from code; CI drift check active
- [ ] No test files under `aneos_core/`
- [ ] `docker-compose up` succeeds after `make bootstrap` from clean clone
- [ ] CI green: `pytest tests/` + `docker-compose build` on every push
- [ ] `monitoring/alerts.py` loads without ML packages
- [ ] No pickle files written by `CacheManager`
- [ ] Chunk boundary test passes (no missed, no duplicate objects)
- [ ] Redis health check in `preflight_check()`; streaming endpoint verified
- [ ] ML classifier trained, registered, and compared to rule-based detector
- [ ] README version and all capability claims backed by measured data

---

## Task Summary

| # | Task | Phase | Gap | Effort |
|---|------|-------|-----|--------|
| 1.1 | Create custom exception types | 1 | G-004 | ½ day |
| 1.2 | Create `preflight_check()` | 1 | G-004 | 1 day |
| 1.3 | Replace fallback in `pipeline_integration.py` | 1 | G-004 | ½ day |
| 1.4 | Replace fallback in `data/fetcher.py` | 1 | G-004 | ½ day |
| 1.5 | Wire `preflight_check()` into entry points | 1 | G-004 | ½ day |
| 1.6 | Wire pipeline to `DetectionManager` | 1 | G-003 | ½ day |
| 1.7 | Add detector-selection test | 1 | G-003 | ¼ day |
| 1.8 | Decouple `monitoring/alerts.py` from ML | 1 | G-006 | ½ day |
| 1.9 | Add monitoring import test | 1 | G-006 | ¼ day |
| 1.10 | Create `init.sql` | 1 | G-005 | ¼ day |
| 1.11 | Create `ssl/README.md` | 1 | G-005 | ¼ day |
| 1.12 | Make Nginx HTTP-only by default | 1 | G-005 | ½ day |
| 1.13 | Create `Makefile` with `bootstrap` | 1 | G-005 | ½ day |
| 1.14 | Create GitHub Actions CI workflow | 1 | G-010 | ½ day |
| 2.1 | Implement `indicators/physical.py` | 2 | G-002 | 2 days |
| 2.2 | Register physical indicators in pipeline | 2 | G-002 | ½ day |
| 2.3 | Verify physical category scoring wiring | 2 | G-002 | ¼ day |
| 2.4 | Write physical indicator unit tests | 2 | G-002 | 1 day |
| 2.5 | Compile confirmed artificial objects | 2 | G-001 | 2 days |
| 2.6 | Compile natural NEO control set | 2 | G-001 | 1 day |
| 2.7 | Build blind test split | 2 | G-001 | ½ day |
| 2.8 | Run detectors against blind test | 2 | G-001 | 1 day |
| 2.9 | Compute accuracy metrics | 2 | G-001 | ½ day |
| 2.10 | Calibrate thresholds (≤ 3 iterations) | 2 | G-001 | 1–3 days |
| 2.11 | Update documentation | 2 | G-007 | 1 day |
| 3.1 | Archive superseded detector files | 3 | G-008 | ½ day |
| 3.2 | Move test harness to `tests/` | 3 | G-013 | ¼ day |
| 3.3 | Designate canonical scoring system | 3 | G-009 | 1 day |
| 3.4 | Separate `OrbitalElements` domain model | 3 | G-011 | 2 days |
| 3.5 | Create API DTO/schema layer | 3 | G-012 | 2 days |
| 3.6 | Generate and commit OpenAPI spec | 3 | G-014 | ½ day |
| 4.1 | Add Redis health check | 4 | G-016 | ½ day |
| 4.2 | Verify Redis application integration | 4 | G-016 | ½ day |
| 4.3 | Replace pickle with JSON in cache | 4 | G-017 | 1 day |
| 4.4 | Validate chunk boundary overlap | 4 | G-018 | ½ day |
| 4.5 | Train and validate ML classifier | 4 | G-015 | 3 days |
| 4.6 | Register ML detector in `DetectionManager` | 4 | G-015 | ½ day |
| 4.7 | Document ML vs rule-based comparison | 4 | G-015 | ½ day |

**Total**: 37 tasks · 18 gaps · ~30–45 working days across 4 phases

---

*Acknowledge this plan to begin implementation. Work proceeds task-by-task
in the order shown, one phase at a time.*
