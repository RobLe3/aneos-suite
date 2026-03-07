# aNEOS Gap Closure Plan

**Version**: 1.0
**Date**: 2026-03-06
**Input**: GAP_ANALYSIS.md v1.0

Work is organized into four phases. Each phase must reach its defined
success criteria before the next phase begins. Dependencies between gaps
are called out explicitly.

---

## Phase 1 — Correctness Foundation
**Goal**: Eliminate anything that produces wrong, misleading, or
undeliverable results. No new features until these pass.

---

### CLOSE-G004: Replace Silent Simulation Fallback with Hard Failures
**Gap ref**: G-004 (P1)
**Effort**: Small (1–2 days)
**No dependencies**

**Tasks**
1. In `aneos_core/integration/pipeline_integration.py`: when
   `HAS_PIPELINE_COMPONENTS = False`, raise `IntegrationError` with a
   clear message listing the missing component instead of silently continuing.
2. In `aneos_core/data/fetcher.py`: when all four API sources fail, raise
   `DataSourceUnavailableError` instead of returning mock data.
3. Add a `preflight_check()` function that validates all required external
   connections before any analysis run starts. Wire it as the first call in
   `automatic_review_pipeline.py` and `aneos_menu.py` entry points.
4. Add `HAS_*` guard replacements in `aneos_api/app.py`: if FastAPI is absent,
   print a clear installation error instead of silently loading a no-op app.

**Success criteria**
- Running with missing components produces an explicit error, not a silent run.
- `python aneos.py status` reports each required integration as PASS/FAIL.

---

### CLOSE-G003: Wire Pipeline to DetectionManager
**Gap ref**: G-003 (P1)
**Effort**: Small (1 day)
**Depends on**: CLOSE-G004 complete (so the fix is testable with real data)

**Tasks**
1. In `automatic_review_pipeline.py:52`, replace the hardcoded import of
   `MultiModalSigma5ArtificialNEODetector` with a call to
   `DetectionManager(preferred_detector=DetectorType.AUTO)`.
2. Pass the `DetectionManager` instance into the first-stage review function.
3. Add a test asserting that `DetectorType.AUTO` resolves to
   `ValidatedSigma5ArtificialNEODetector` when all detectors load successfully.

**Success criteria**
- `DetectorType.AUTO` in the pipeline selects `VALIDATED` by default.
- Existing pipeline integration tests still pass.

---

### CLOSE-G006: Decouple Monitoring from ML Module
**Gap ref**: G-006 (P1)
**Effort**: Small (half a day)
**No dependencies**

**Tasks**
1. In `aneos_core/monitoring/alerts.py`, wrap the ML import in a try/except:
   ```python
   try:
       from ..ml.prediction import Alert as MLAlert, PredictionResult
       HAS_ML_ALERTS = True
   except ImportError:
       HAS_ML_ALERTS = False
   ```
2. Guard every usage of `MLAlert` / `PredictionResult` inside the module with
   `if HAS_ML_ALERTS:`.
3. Add a unit test that imports `monitoring.alerts` in an environment without
   scikit-learn or PyTorch and confirms it loads without error.

**Success criteria**
- `aneos_core/monitoring/alerts` imports cleanly without ML packages installed.

---

### CLOSE-G005: Fix Docker Deployment
**Gap ref**: G-005 (P1)
**Effort**: Small (1 day)
**No dependencies**

**Tasks**
1. Create `init.sql` at repo root with the minimum schema required for
   PostgreSQL startup (empty schema stub is acceptable to unblock boot).
2. Create `ssl/README.md` documenting that self-signed certs must be placed
   here for HTTPS; provide a one-liner `openssl` command to generate them.
3. Update `docker-compose.yml` Nginx service to use HTTP-only by default;
   make HTTPS opt-in via a separate `docker-compose.override.yml`.
4. Add a `Makefile` target `make bootstrap` that generates `init.sql` and
   self-signed SSL certs so `docker-compose up` works from a clean clone.

**Success criteria**
- `git clone && make bootstrap && docker-compose up` reaches healthy state
  for all six services.

---

### CLOSE-G005b: Add CI/CD Container Smoke Test
**Gap ref**: G-010 (P2) — pulled forward because it validates G-005
**Effort**: Small (1 day)
**Depends on**: CLOSE-G005 complete

**Tasks**
1. Add `.github/workflows/ci.yml` with jobs:
   - `test`: run `python -m pytest tests/` on push to main and on PRs.
   - `docker-build`: run `docker-compose build` to verify the image builds.
2. Badge the README with the CI status.

**Success criteria**
- Every push to `main` runs the 61-test suite and verifies the Docker build.

---

## Phase 2 — Scientific Completeness
**Goal**: Close all gaps that cause the implementation to diverge from the
concept document's defined methodology.

---

### CLOSE-G002: Implement Physical Indicator Category
**Gap ref**: G-002 (P1)
**Effort**: Medium (3–5 days)
**Depends on**: Phase 1 complete

**Tasks**
1. Create `aneos_core/analysis/indicators/physical.py` with three concrete
   `AnomalyIndicator` subclasses:
   - `DiameterAnomalyIndicator` — flags objects outside the 0.1–10 km
     natural size distribution (using `ThresholdConfig.diameter_min/max`).
   - `AlbedoAnomalyIndicator` — flags albedo outside 0.05–0.50 natural
     range; extreme values (e.g., > 0.60) score as `albedo_artificial`.
   - `SpectralAnomalyIndicator` — wraps the existing
     `SpectralOutlierAnalyzer` result as an `IndicatorResult`.
2. Register the three indicators in `analysis/pipeline.py`
   `PipelineConfig.indicator_configs` alongside existing indicators.
3. Add `'physical'` category mapping in `scoring.py:ScoreCalculator`
   that routes to the three new indicator names.
4. Write unit tests for each indicator covering normal, anomalous, and
   missing-data cases.

**Success criteria**
- `scoring.py` physical category produces non-zero scores for objects with
  anomalous diameter or albedo.
- All existing 61 regression tests continue to pass.
- Scientific documentation category count (5) matches implementation.

---

### CLOSE-G001: Build and Validate Ground Truth Dataset
**Gap ref**: G-001 (P0)
**Effort**: Large (2–4 weeks, iterative)
**Depends on**: CLOSE-G002, CLOSE-G003, CLOSE-G004 complete

This is the highest-priority scientific work in the project.

**Tasks**

**Step 1 — Compile confirmed artificial objects**
Operationalise `datasets/ground_truth_dataset_preparation.py:GroundTruthDatasetBuilder`:
- Pull heliocentric-orbit spacecraft from JPL Horizons:
  Voyager 1/2, Pioneer 10/11, New Horizons, Cassini (post-disposal),
  Elon Musk's Tesla Roadster (2018-017A).
- Pull confirmed rocket bodies from Space-Track.org TLE archive filtered
  to heliocentric orbits.
- Store as `GroundTruthObject(object_id, is_artificial=True, orbital_elements,
  source, verification_notes)`.

**Step 2 — Compile natural NEO control set**
- Download 500–1,000 well-characterised natural NEOs from JPL SBDB
  (H < 22, multi-opposition, low orbital uncertainty).
- Store as `GroundTruthObject(is_artificial=False, ...)`.

**Step 3 — Build blind test suite**
- Randomly shuffle and anonymize object IDs.
- Split 70% training / 30% blind test; withhold test labels.
- Implement `ArtificialNEOTestSuite.run_blind_test(dataset, detector)`.

**Step 4 — Run detectors and measure accuracy**
- Run `DetectionManager(DetectorType.VALIDATED)` against the blind test set.
- Compute: precision, recall, F1, FPR, FNR at the sigma-5 threshold.
- Record results in `development/test-results/ground_truth_v1.json`.

**Step 5 — Iterate on calibration**
- If FPR > 5.7×10⁻⁷, tune `AdvancedScoringConfig` weights and re-run.
- Document each calibration iteration with before/after metrics.
- Maximum 3 iterations per CLAUDE.md framework rules.

**Success criteria**
- Dataset contains ≥ 10 confirmed artificial + ≥ 100 natural objects.
- Blind test results are documented with precision/recall/FPR numbers.
- `docs/engineering/sigma5_success_criteria.md` is updated with
  empirical validation results.
- Detection claims in README are updated to reflect measured accuracy.

---

### CLOSE-G007: Align Documentation with Measured Reality
**Gap ref**: G-007 (P1)
**Effort**: Small (1 day)
**Depends on**: CLOSE-G001 complete (real numbers available)

**Tasks**
1. Update `README.md`:
   - Replace "v1.0.0 - Production Ready" with the actual version and maturity.
   - Replace theoretical FPR claim with empirically measured value from G-001
     closure, or state "under calibration" if not yet achieved.
2. Update `ROADMAP.md` "Current Status" section to reflect Phase 1 + 2
   completion.
3. Update `CHANGELOG.md` with gap closure entries.
4. Update `docs/engineering/maturity_assessment.md` with current state.

**Success criteria**
- No public-facing document contradicts internal engineering assessments.

---

## Phase 3 — Architecture Cleanup
**Goal**: Remove structural debt that creates ongoing maintenance and
onboarding friction.

---

### CLOSE-G008: Archive Superseded Detectors
**Gap ref**: G-008 (P2)
**Effort**: Small (half a day)
**Depends on**: CLOSE-G001 complete (so we know which detector performs best)

**Tasks**
1. Create `aneos_core/detection/_archive/`.
2. Move all non-canonical detectors into `_archive/`:
   `sigma5_artificial_neo_detector.py`,
   `corrected_sigma5_artificial_neo_detector.py`,
   `production_artificial_neo_detector.py`,
   `multimodal_sigma5_artificial_neo_detector.py`,
   `sigma5_corrected_statistical_framework.py`.
3. Keep only `validated_sigma5_artificial_neo_detector.py` and
   `detection_manager.py` in the active module.
4. Update `DetectionManager` imports; remove archived detector entries
   from the registry (or mark as `DEPRECATED`).
5. Move `artificial_neo_test_suite.py` to `tests/detection/`.

**Success criteria**
- `aneos_core/detection/` contains exactly two active files plus `__init__.py`.
- All 61 regression tests pass.

---

### CLOSE-G009: Document Scoring System Arbitration
**Gap ref**: G-009 (P2)
**Effort**: Small (1 day)
**Depends on**: CLOSE-G001 (empirical data to inform the decision)

**Tasks**
1. Compare ATLAS (`advanced_scoring.py`) vs Standard (`scoring.py`) on the
   ground truth dataset: which produces better F1 at sigma-5?
2. Designate the winning system as canonical in a new ADR entry
   (ADR-008 update).
3. Remove or clearly deprecate the other system, or document it as a
   cross-validation reference tool only.
4. Remove the `# EMERGENCY` comment and fix the suppressed warning
   in `advanced_scoring.py`.

**Success criteria**
- A single scoring system is designated canonical in code and documentation.
- No `# EMERGENCY` suppression comments remain.

---

### CLOSE-G011: Separate OrbitalElements from PhysicalProperties
**Gap ref**: G-011 (P2)
**Effort**: Medium (2–3 days — has wide refactor surface)
**Depends on**: Phase 2 complete

**Tasks**
1. Remove `diameter`, `albedo`, `rot_per`, `spectral_type` from
   `OrbitalElements`.
2. Ensure `PhysicalProperties` contains these fields (it already exists
   in `models.py`).
3. Update all callsites that read physical properties from
   `OrbitalElements` to read from `PhysicalProperties`.
4. Run the full test suite; fix any broken references.

**Success criteria**
- `OrbitalElements` contains only Keplerian orbital mechanics fields.
- `PhysicalProperties` is the single source for physical characterization.

---

### CLOSE-G012: Add API DTO Layer
**Gap ref**: G-012 (P2)
**Effort**: Medium (2–3 days)
**Depends on**: CLOSE-G011 complete

**Tasks**
1. Create `aneos_api/schemas/` with Pydantic v2 response models for each
   endpoint group: `AnalysisResponse`, `DetectionResponse`,
   `ImpactResponse`, `HealthResponse`.
2. Update endpoint handlers to convert domain types → DTO before returning.
3. Export the OpenAPI spec as `docs/api/openapi.json` via a `make spec`
   target.
4. Retire the hand-maintained `docs/api/rest-api.md` in favour of the
   generated spec.

**Success criteria**
- Domain model changes do not break API response shapes.
- `docs/api/openapi.json` is generated from code, not hand-edited.

---

### CLOSE-G013: Move Test Harness to tests/
**Gap ref**: G-013 (P2)
**Effort**: Trivial (< 1 hour)
**No dependencies**

**Tasks**
1. Move `aneos_core/detection/artificial_neo_test_suite.py` to
   `tests/detection/test_artificial_neo_suite.py`.
2. Fix any import paths.

**Success criteria**
- No test code exists under `aneos_core/`.

---

### CLOSE-G014: Generate and Maintain OpenAPI Spec
**Gap ref**: G-014 (P2)
**Effort**: Small (half a day)
**Depends on**: CLOSE-G012 complete

**Tasks**
1. Add `make spec` to `Makefile` that runs the FastAPI app in export mode
   and writes `docs/api/openapi.json`.
2. Add a CI step that runs `make spec` and fails if the output differs
   from the committed file (spec drift detection).

**Success criteria**
- The CI job fails when endpoint signatures change without updating the spec.

---

## Phase 4 — Future Capabilities
**Goal**: Activate deferred subsystems once correctness and architecture
foundations are solid.

---

### CLOSE-G015: Activate ML Classification Pipeline
**Gap ref**: G-015 (P3)
**Effort**: Large (2–4 weeks)
**Depends on**: CLOSE-G001 complete (labelled dataset required)

**Tasks**
1. Use the ground truth dataset (G-001) to train `IsolationForest` and
   `RandomForestClassifier` via `aneos_core/ml/training.py`.
2. Wire `RealTimePredictor` into `DetectionManager` as a new
   `DetectorType.ML` option with lower priority than VALIDATED.
3. Add model persistence to `models/` directory.
4. Validate ML model FPR against the blind test set.
5. Document performance comparison: ML vs sigma-5 rule-based.

**Success criteria**
- ML detector runs end-to-end on real NEO data.
- Measured accuracy is documented alongside the rule-based detector.

---

### CLOSE-G016: Verify and Activate Redis Integration
**Gap ref**: G-016 (P3)
**Effort**: Small (1 day)
**Depends on**: CLOSE-G005 complete

**Tasks**
1. Add a Redis health check to `preflight_check()`.
2. Implement a minimal Redis pub/sub message in `endpoints/streaming.py`
   to confirm the connection is active.
3. Wire `CacheManager` to use Redis as a distributed L2 cache behind the
   in-process L1 LRU cache in production mode.

**Success criteria**
- `docker-compose up` shows Redis actively receiving events from the application.

---

### CLOSE-G017: Replace Pickle Cache with JSON-Only Persistence
**Gap ref**: G-017 (P3)
**Effort**: Small (1 day)
**No dependencies**

**Tasks**
1. In `data/cache.py`, replace pickle serialization with JSON for all
   disk persistence paths.
2. For values that are not JSON-serializable, apply `dataclasses.asdict()`
   or `.__dict__` before writing.
3. Remove all `pickle.dumps` / `pickle.loads` calls.

**Success criteria**
- Cache files on disk are human-readable JSON; no pickle files remain.

---

### CLOSE-G018: Validate Chunk Boundary Overlap
**Gap ref**: G-018 (P3)
**Effort**: Small (1 day)
**No dependencies**

**Tasks**
1. Write a unit test that creates two adjacent 5-year chunks with 7-day
   overlap and asserts: (a) objects at the boundary appear in at least
   one chunk, (b) no object appears more than once in the merged result.
2. Add a deduplication step in `HistoricalChunkedPoller.merge_chunks()`
   that removes duplicates by designation before returning.

**Success criteria**
- Boundary test passes with zero missed and zero duplicate objects.

---

## Execution Summary

| Phase | Focus | Gaps closed | Approx effort |
|-------|-------|-------------|---------------|
| 1 — Correctness Foundation | Eliminate wrong/misleading results | G-003, G-004, G-005, G-006, G-010 | ~1 week |
| 2 — Scientific Completeness | Align implementation with concept doc | G-001, G-002, G-007 | 3–6 weeks |
| 3 — Architecture Cleanup | Remove structural debt | G-008, G-009, G-011, G-012, G-013, G-014 | ~2 weeks |
| 4 — Future Capabilities | Activate deferred subsystems | G-015, G-016, G-017, G-018 | 4–6 weeks |

---

## Dependency Graph

```
CLOSE-G006 ──────────────────────────────────────────────────┐
CLOSE-G013 ──────────────────────────────────────────────────┤
CLOSE-G004 ──▶ CLOSE-G003                                    │
CLOSE-G005 ──▶ CLOSE-G005b (CI)                              │
                                                              │
CLOSE-G004 ──┐                                               │
CLOSE-G003 ──┤                                               │
CLOSE-G005b ─┤                                               │
CLOSE-G002 ──┘──▶ CLOSE-G001 ──▶ CLOSE-G007                 │
                            └──▶ CLOSE-G008                  │
                            └──▶ CLOSE-G009                  │
                            └──▶ CLOSE-G015 (ML)             │
CLOSE-G011 ──▶ CLOSE-G012 ──▶ CLOSE-G014                    │
CLOSE-G005 ──▶ CLOSE-G016                                    │
                                                              │
All above ◀──────────────────────────────────────────────────┘
```

---

## Success Criteria for Full Closure

All 18 gaps are closed when:

1. `python aneos.py status` reports all integrations PASS with no mock data in use.
2. Ground truth blind test produces documented precision / recall / FPR numbers.
3. README version and capability claims match measured, documented reality.
4. `aneos_core/detection/` contains exactly one active detector file.
5. `aneos_core/analysis/indicators/` contains `physical.py` with 3 indicators.
6. `docker-compose up` reaches healthy state from a clean clone with `make bootstrap`.
7. CI passes on every push to `main`.
8. No pickle files used for cache persistence.
9. OpenAPI spec is generated from code and drift-detected in CI.

---

*This plan should be reviewed and adjusted after each phase completion.*
