# aNEOS Gap Analysis

**Version**: 1.0
**Date**: 2026-03-06
**Derived from**: ADR.md v1.0, DDD.md v1.0, README.md, docs/scientific/scientific-documentation.md,
docs/engineering/sigma5_success_criteria.md, Calibration Plan v1.2

---

## Gap Severity Legend

| Symbol | Severity | Definition |
|--------|----------|------------|
| P0 | Critical | Blocks the core mission; no workaround exists |
| P1 | High | Produces incorrect or misleading results; deployment-blocking |
| P2 | Medium | Maintainability or quality risk; does not affect correctness today |
| P3 | Low | Future-state or polish; no immediate impact |

---

## P0 — Critical Gaps

### G-001: No Ground Truth Dataset
**Area**: Detection / Scientific Validity
**Concept claim**: "Validated Detection: Sigma calculation methodology and probability separation." (README)
**Reality**: `aneos_core/datasets/ground_truth_dataset_preparation.py` is a stub. No labelled
dataset of confirmed artificial vs natural objects has been compiled or tested against.
**Impact**: All accuracy, recall, and FPR claims (including the 5.7×10⁻⁷ FPR) are
theoretical. No detection claim can be peer-reviewed or published without this dataset.
**ADR/DDD ref**: ADR-038, DDD BC-10

---

## P1 — High Severity Gaps

### G-002: Physical Indicator Category Not Implemented
**Area**: Anomaly Scoring / Concept Alignment
**Concept claim**: Scientific documentation specifies 5 indicator categories including
Physical Indicators (`diameter_anomalies`, `albedo_anomalies`, `spectral_anomalies`).
**Reality**: `aneos_core/analysis/indicators/physical.py` does not exist. `scoring.py`
maps the `physical` category but no indicators evaluate it — physical anomaly scoring
is silently skipped for every object.
**Impact**: One of five core detection dimensions is absent from every analysis run.
**ADR/DDD ref**: ADR-007, DDD BC-3

### G-003: Production Pipeline Bypasses Canonical Detector
**Area**: Detection / Architecture Consistency
**Concept claim**: `DetectionManager` (ADR-011) establishes `ValidatedSigma5ArtificialNEODetector`
as the highest-priority (priority 0) canonical detector.
**Reality**: `aneos_core/pipeline/automatic_review_pipeline.py:52` hardcodes
`MultiModalSigma5ArtificialNEODetector` directly, bypassing `DetectionManager` entirely.
The VALIDATED detector is never used in the main production flow.
**Impact**: The scientifically validated detector has no effect on any real analysis run.
**ADR/DDD ref**: ADR-011, ADR-027, DDD BC-5

### G-004: Silent Simulation Fallback on API or Component Failure
**Area**: Data Integrity / Operational Reliability
**Concept claim**: "Real Data Integration: NASA/JPL data with comprehensive explanations." (README)
**Reality**: `HAS_PIPELINE_COMPONENTS`, `HAS_ANALYSIS`, `HAS_FASTAPI` guard flags silently
substitute mock/simulated results when real integrations fail. No user warning is issued.
**Impact**: An operator can run a full analysis session on synthetic data believing it
used real NEO catalogs. Any detection produced under this condition is invalid.
**ADR/DDD ref**: ADR-032, DDD BC-1

### G-005: Docker Deployment Broken Out-of-the-Box
**Area**: Infrastructure / Deployment
**Reality**: `docker-compose.yml` references:
- `./init.sql` — does not exist in the repository
- `./ssl/` — does not exist in the repository
**Impact**: `docker-compose up` fails immediately without manual preparation.
No CI/CD pipeline validates the container build on commit.
**ADR/DDD ref**: ADR-036, DDD BC-9

### G-006: Monitoring Alerting Hard-Depends on ML Module
**Area**: Monitoring / Reliability
**Reality**: `aneos_core/monitoring/alerts.py` imports:
```python
from ..ml.prediction import Alert as MLAlert, PredictionResult
```
The ML module is classified as deferred (ADR-033), but this import makes it a
hard runtime dependency for the alerting system.
**Impact**: If ML packages (scikit-learn, PyTorch) are absent, the monitoring
alert system fails to load — silently disabling all operational alerts.
**ADR/DDD ref**: ADR-033, DDD BC-9

### G-007: README Claims "v1.0 Production Ready" — Contradicted by Internal Docs
**Area**: Documentation / User Trust
**Reality**: The README states "Current development: v1.0.0 - Production Ready" and
"System maturity: Advanced level." `docs/engineering/maturity_assessment.md` (August 2025),
`CLAUDE.md`, and `docs/engineering/sigma5_success_criteria.md` all contradict this,
explicitly listing unmet production criteria.
**Impact**: External users and collaborators form incorrect expectations about reliability.
AI-generated validation notes (`reporting/ai_validation.py`) may compound this by producing
overconfident academic language.
**ADR/DDD ref**: ADR-032, ADR-039

---

## P2 — Medium Severity Gaps

### G-008: Five Detector Files — No Archived Canonical
**Area**: Detection / Maintainability
**Reality**: Six files coexist in `aneos_core/detection/` representing successive
calibration iterations with no deprecation markers:
`sigma5_artificial_neo_detector.py`, `corrected_sigma5_artificial_neo_detector.py`,
`production_artificial_neo_detector.py`, `multimodal_sigma5_artificial_neo_detector.py`,
`validated_sigma5_artificial_neo_detector.py`, `sigma5_corrected_statistical_framework.py`
**Impact**: New contributors cannot determine the authoritative implementation. Dead code
may mask regressions in tests that import older detectors.
**ADR/DDD ref**: ADR-013

### G-009: Dual Scoring Systems with No Documented Arbitration
**Area**: Anomaly Scoring / Maintainability
**Reality**: Two independent systems exist — `scoring.py` (Standard, 6 categories, 4-tier
classification) and `advanced_scoring.py` (ATLAS, 6 clue categories, continuous [0,1]).
No document states which is authoritative for production use. `advanced_scoring.py`
contains a comment labelled `# EMERGENCY` suppressing configuration warnings.
**Impact**: Score values from the two systems are not directly comparable. Inconsistent
routing may produce different verdicts for the same object depending on entry point.
**ADR/DDD ref**: ADR-008

### G-010: No CI/CD Pipeline
**Area**: Engineering Quality
**Reality**: No GitHub Actions, CircleCI, or equivalent pipeline exists. The 61-test
regression suite must be run manually.
**Impact**: Regressions can reach `main` undetected. The container build is never
validated automatically.
**ADR/DDD ref**: ADR-036

### G-011: `OrbitalElements` Conflates Orbital and Physical Properties
**Area**: Domain Model / Maintainability
**Reality**: `aneos_core/data/models.py:OrbitalElements` carries `diameter`, `albedo`,
`rot_per`, `spectral_type` — fields that belong in `PhysicalProperties`.
**Impact**: Consumers are unclear which model to query for physical data; the domain
model boundary between orbital mechanics and physical characterization is blurred.
**ADR/DDD ref**: ADR-006, DDD BC-2

### G-012: API Layer Has No DTO Separation
**Area**: API / Maintainability
**Reality**: `aneos_api/endpoints/` handlers use internal domain types directly in
responses. No Pydantic response schemas or DTOs exist between the domain model and the
HTTP contract.
**Impact**: Any domain model change (e.g., renaming a field in `OrbitalElements`) breaks
the public API contract without a compilation error. API versioning is not possible
without modifying domain code.
**ADR/DDD ref**: ADR-034, DDD BC-8

### G-013: `ArtificialNEOTestSuite` Lives Inside Detection Module
**Area**: Code Organisation
**Reality**: `aneos_core/detection/artificial_neo_test_suite.py` is a test harness
inside the production module directory.
**Impact**: Test code is importable as production code; coverage tools may conflate
production and test lines.
**ADR/DDD ref**: DDD BC-5

### G-014: No OpenAPI Specification Maintained
**Area**: API Documentation
**Reality**: `docs/api/rest-api.md` is hand-maintained and may drift from the actual
52 FastAPI endpoints. FastAPI generates a spec automatically but only when the server
is running.
**ADR/DDD ref**: ADR-034

---

## P3 — Low Severity Gaps

### G-015: ML Module Has No Activation Path
**Area**: ML / Future Readiness
**Reality**: `aneos_core/ml/` (features, models, training, prediction) is fully
scaffolded but has no wiring into the detection or validation pipeline. Activation
requires a labelled ground truth dataset (blocked by G-001).
**ADR/DDD ref**: ADR-033

### G-016: Redis Integration Unverified
**Area**: Infrastructure
**Reality**: Redis is declared as a Docker Compose service and intended for streaming
and distributed caching, but no Python code path currently confirms active use of the
Redis connection from the application layer.
**ADR/DDD ref**: ADR-037

### G-017: `CacheManager` Pickle Persistence Security Risk
**Area**: Security / Data Integrity
**Reality**: `data/cache.py` uses `pickle` for disk persistence. If cache files come
from an untrusted source, deserialization could execute arbitrary code.
**Impact**: Low in the current research-tool context; higher if deployed as a shared
service.
**ADR/DDD ref**: ADR-003

### G-018: Chunk Boundary Overlap Not Independently Validated
**Area**: Data Acquisition / Scientific Accuracy
**Reality**: The 7-day overlap in `HistoricalChunkedPoller` is intended to prevent
boundary-straddle misses, but no test verifies that objects at 5-year chunk boundaries
are neither missed nor double-counted.
**ADR/DDD ref**: ADR-004

---

## Gap Summary Table

| ID | Title | Priority | Area |
|----|-------|----------|------|
| G-001 | No ground truth dataset | **P0** | Detection / Scientific |
| G-002 | Physical indicator category missing | **P1** | Scoring |
| G-003 | Pipeline bypasses canonical detector | **P1** | Detection |
| G-004 | Silent simulation fallback | **P1** | Data Integrity |
| G-005 | Docker deployment broken | **P1** | Infrastructure |
| G-006 | Monitoring depends on deferred ML | **P1** | Monitoring |
| G-007 | README overstates production readiness | **P1** | Documentation |
| G-008 | Five detector files, no archived canonical | **P2** | Maintainability |
| G-009 | Dual scoring systems, no arbitration | **P2** | Scoring |
| G-010 | No CI/CD pipeline | **P2** | Engineering Quality |
| G-011 | OrbitalElements conflates domain models | **P2** | Domain Model |
| G-012 | API has no DTO layer | **P2** | API |
| G-013 | Test harness inside production module | **P2** | Code Organisation |
| G-014 | No maintained OpenAPI spec | **P2** | API Documentation |
| G-015 | ML has no activation path | **P3** | ML |
| G-016 | Redis integration unverified | **P3** | Infrastructure |
| G-017 | Pickle cache security risk | **P3** | Security |
| G-018 | Chunk boundary overlap unvalidated | **P3** | Data Accuracy |

**Totals**: 1 P0 · 6 P1 · 7 P2 · 4 P3

---

*Next step: derive a prioritised closure plan from this gap list.*
