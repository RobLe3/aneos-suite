# aNEOS — UELO Gap Analysis (Phase 8 Targets)

**Version**: 1.1
**Date**: 2026-03-07
**Supersedes**: UELO.md §5 (seed table)
**Derived from**: Live code audit of 12 source files post-Phase 7
**Test baseline**: 59 pass / 0 fail
**Phase 8 status**: All 10 gaps (UE-001 – UE-010) CLOSED — see PHASE9_UELO_GAP_ANALYSIS.md for next targets

---

## Purpose

This document is a code-verified, persona-anchored gap analysis. Every gap is traced
to exact file locations, the user experience consequence is stated from the persona's
perspective, and acceptance criteria define what "fixed" means in observable terms.

This is **not** an engineering quality register (see GAP_ANALYSIS.md). Every gap here
represents a broken promise to a specific user — something the system implies it does,
but does not currently deliver through the user-facing surface (CLI menu or REST API).

---

## Severity Legend

| Symbol | Meaning |
|--------|---------|
| P1 | Blocks a primary user journey end-to-end |
| P2 | Degrades quality of a delivered result or makes it misleading |
| P3 | Reduces completeness or convenience; workarounds exist |

---

## Gap Index

| ID | Title | Persona | Priority | Phase 8 Sub |
|----|-------|---------|----------|-------------|
| UE-001 | Evidence sources not exposed in API | P1, P3 | P1 | 8A |
| UE-002 | No impact probability REST endpoint | P2, P3 | P1 | 8B |
| UE-003 | Batch analysis non-functional end-to-end | P1, P3 | P1 | 8C |
| UE-004 | No sigma tier label | P1, P4 | P2 | 8A |
| UE-005 | Data source and freshness not surfaced | P3 | P2 | 8D |
| UE-006 | `/health` endpoint schema not wired | P3 | P2 | 8D |
| UE-007 | Interpretation string is generic | P4 | P2 | 8A |
| UE-008 | Spacecraft veto not surfaced in API | P1, P3 | P2 | 8D |
| UE-009 | Export non-functional | P1, P3 | P3 | 8E |
| UE-010 | Auth blocks research API use | P3 | P3 | 8E |

---

## P1 Gaps

---

### UE-001: Evidence Sources Not Exposed in API

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I ran `GET /detect?designation=2020+SO` and got `is_artificial: true`, but how
> do I know *why*? What indicators fired? Was it the orbital inclination? The mass
> anomaly? I need to cite specific evidence in my research."

The `GET /detect` endpoint exists (added Phase 7). It returns `evidence_count: 5`
but none of the actual evidence — the researcher cannot determine which of the 8
`EvidenceType` categories contributed to the detection, nor their statistical weight.

#### Root Cause

**`aneos_api/schemas/detection.py:20`**:
```python
class DetectionResponse(BaseModel):
    ...
    evidence_count: int = 0      # ← count only; no evidence list
```

**`aneos_api/endpoints/analysis.py:172`**:
```python
return DetectionResponse(
    ...
    evidence_count=len(result.evidence_sources),  # ← discards all detail
)
```

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py:67-76`**:
```python
@dataclass
class EvidenceSource:
    evidence_type: EvidenceType     # ORBITAL_DYNAMICS, PHYSICAL_PROPERTIES, etc.
    anomaly_score: float            # Z-score vs natural population
    confidence_interval: Tuple[float, float]
    sample_size: int
    p_value: float
    effect_size: float
    quality_score: float
```

`Sigma5DetectionResult.evidence_sources: List[EvidenceSource]` is fully populated by the
detector but the API serializes only `len(...)`. The `EvidenceType` enum has 8 values;
the typical detection produces 1–2 (orbital + physical). All detail is discarded at the
API boundary.

#### Consequence Chain

1. Researcher calls `GET /detect` → gets `evidence_count: 2`
2. Cannot tell if detection was driven by orbital anomaly, mass anomaly, or both
3. Cannot reproduce or challenge the result without re-running the entire pipeline
4. Published result cannot cite specific evidence — scientific credibility fails

#### Acceptance Criteria

- `GET /detect` response includes `evidence_sources: List[...]` (not just count)
- Each source includes at minimum: `type`, `anomaly_score`, `p_value`, `quality_score`
- `DetectionResponse` appears in OpenAPI spec with the `evidence_sources` field
- Test: `GET /detect?designation=Apophis` → `evidence_sources[0].type == "orbital_dynamics"`

#### Proposed Fix

1. Add `EvidenceSummary` Pydantic model to `schemas/detection.py`:
   ```python
   class EvidenceSummary(BaseModel):
       type: str
       anomaly_score: float
       p_value: float
       quality_score: float
       effect_size: float
   ```
2. Add `evidence_sources: List[EvidenceSummary] = []` to `DetectionResponse`
3. In `detect_neo()` endpoint: serialize `result.evidence_sources` to `EvidenceSummary` list

---

### UE-002: No Impact Probability REST Endpoint

**Persona**: P2 (Planetary Defense Analyst), P3 (System Integrator)

#### User Expectation

> "I want to query impact probability for a list of newly discovered objects
> programmatically. The `/detect` endpoint works — where's `/impact`? The Python
> class exists in the codebase but there's no API surface for it."

#### Root Cause

**`aneos_core/analysis/impact_probability.py:116`**:
`ImpactProbabilityCalculator` is fully implemented with Earth + Moon probability,
crater estimation, keyhole analysis, temporal evolution, and risk tiers.

**`aneos_api/schemas/impact.py:13-21`**:
```python
class ImpactResponse(BaseModel):
    designation: str
    collision_probability: float
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    risk_level: str
    time_to_impact_years: Optional[float] = None
```
`ImpactResponse` schema exists and is exported from `schemas/__init__.py`.

**`aneos_api/endpoints/analysis.py`**: No `GET /impact` or `POST /impact` endpoint.
The `ImpactResponse` schema is never used as a `response_model` in any endpoint.

The connection (`ImpactProbabilityCalculator` → `ImpactResponse` → `/impact` route) is
entirely missing. The `ImpactResponse` schema is dead code.

#### Consequence Chain

1. Analyst queries `GET /docs` → sees no impact endpoint → uses CLI menu instead
2. No programmatic batch impact assessment is possible
3. The `ImpactResponse` schema wastes spec space as unreachable dead code
4. P3 users cannot build tools on top of impact assessment

#### Acceptance Criteria

- `GET /impact?designation=X` returns a valid `ImpactResponse`
- Fields: `collision_probability`, `moon_collision_probability`, `risk_level` at minimum
- `ImpactResponse` appears in `components/schemas` in `docs/api/openapi.json`
- Test: `GET /impact?designation=99942` → `risk_level in ("negligible", "very_low", "low", "moderate", "high", "extreme")`

#### Proposed Fix

1. Add `GET /impact` to `aneos_api/endpoints/analysis.py` with `response_model=ImpactResponse`:
   - Fetch orbital elements via `DataFetcher`
   - Instantiate `ImpactProbabilityCalculator`
   - Call `calculate_impact_probability(oe, designation)`
   - Map result fields to `ImpactResponse`
2. Update `ImpactResponse` if needed to match `ImpactProbability` dataclass fields
3. Run `make spec`; verify `ImpactResponse` in spec

---

### UE-003: Batch Analysis Non-Functional End-to-End

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I submitted 50 designations to `POST /analyze/batch`. Got `batch_id: batch_20260307`.
> Now I call `GET /batch/batch_20260307/status` and always get `'completed': 0,
> 'results': []`. The batch never runs. This is the only way to analyze more than
> one object at a time."

#### Root Cause

**`aneos_api/endpoints/analysis.py:185-186`** — `analyze_batch` requires `get_analysis_pipeline()`:
```python
pipeline: AnalysisPipeline = Depends(get_analysis_pipeline),
```

`get_analysis_pipeline()` (line 63-67) calls `get_aneos_app()` → raises `HTTPException(503)`
if `aneos_app.analysis_pipeline` is `None` (which it is by default). The batch endpoint
cannot even be reached without a running analysis pipeline.

**`aneos_api/endpoints/analysis.py:185-200`** — even if it runs, `batch_status` is stored
only in a local variable passed to `_process_batch_analysis()`. The background task
modifies this local dict, but `get_batch_status()` (lines 185-200) ignores all batch state:

```python
@router.get("/batch/{batch_id}/status", response_model=Dict[str, Any])
async def get_batch_status(batch_id: str, ...):
    """Get status of a batch analysis job."""
    # Implementation would retrieve from persistent storage
    return {
        'batch_id': batch_id,
        'status': 'completed',   # ← hardcoded, always "completed"
        'progress': 100,
        'completed': 0,          # ← always 0
        'failed': 0,
        'results_available': True
    }
```

**Three independent failures**:
1. Endpoint unreachable if pipeline not initialized (503 before any work)
2. Batch state stored in ephemeral local dict, never persisted
3. Status endpoint returns hardcoded mock regardless of actual state

#### Consequence Chain

1. P1 researcher submits a 100-object batch → 503 error
2. If pipeline happened to be initialized: batch "starts", status endpoint always says `completed: 0`
3. P3 integrators cannot build any batch workflow on top of aNEOS API
4. The only multi-object path is the CLI menu (`[2][8] Bulk Detection`), which is not scriptable

#### Acceptance Criteria

- `POST /analyze/batch` accepts a list of designations and returns a `batch_id`
- `GET /batch/{batch_id}/status` returns actual progress and results when complete
- At least one result object in `results` when analysis completes
- The endpoint does not require `AnalysisPipeline` — it should use `DataFetcher` + `ValidatedSigma5ArtificialNEODetector` directly (same path as `GET /detect`)
- Test: Submit `["Apophis", "Bennu"]`; poll status until `status == "completed"`; verify `completed == 2`

#### Proposed Fix

1. Decouple `analyze_batch` from `AnalysisPipeline` — use `DataFetcher` + detector directly
2. Introduce a module-level `_batch_store: Dict[str, Dict]` (analogous to `_analysis_cache`)
3. Background task populates `_batch_store[batch_id]["results"]` as each object completes
4. `get_batch_status()` looks up `_batch_store[batch_id]` instead of returning hardcoded mock
5. Use `asyncio.gather()` or `ThreadPoolExecutor` for concurrent detection

---

## P2 Gaps

---

### UE-004: No Sigma Tier Label

**Persona**: P1 (Anomaly Researcher), P4 (Casual Explorer)

#### User Expectation

> "The detector says `sigma_confidence: 2.1`. Is that good? Bad? I know 5-sigma is
> the discovery threshold. Is 2.1 worth investigating? The number alone doesn't tell
> me where I am in the scale."

#### Root Cause

**`aneos_api/schemas/detection.py`**: `DetectionResponse` has no `sigma_tier` field.

**`aneos_api/endpoints/analysis.py:164-173`**: The `detect_neo()` endpoint returns
a numeric `sigma_confidence` only, with no categorical label.

The UELO document proposed tiers (§5 UE-004); the code has no implementation.

In the CLI menu (`aneos_menu.py`), sigma tiers appear in some displays but use
inconsistent terminology across functions and are not connected to the detector's
`sigma_confidence` value in a single canonical way.

#### Consequence Chain

1. P4 user sees `sigma_confidence: 2.1` — no context for whether this is noteworthy
2. P1 researcher comparing two objects (`σ=2.1` vs `σ=3.8`) must mentally map to significance
3. The `is_artificial: false` flag below σ=5 gives no gradient — every sub-5σ object
   looks equally "not artificial" even if σ=4.9

#### Tier Scale (Proposed, per UELO)

| σ range | Tier | Plain-English meaning |
|---------|------|-----------------------|
| 0.0–1.0 | ROUTINE | Within normal NEO population range |
| 1.0–2.0 | NOTABLE | Mildly unusual; common in the NEO catalog |
| 2.0–3.0 | INTERESTING | Worth closer inspection; 1-in-20 or rarer |
| 3.0–4.0 | SIGNIFICANT | Strong orbital/physical anomaly; peer-review level |
| 4.0–5.0 | ANOMALOUS | Near-discovery threshold; warrants full investigation |
| ≥5.0 | EXCEPTIONAL | Meets σ=5 discovery standard |

#### Acceptance Criteria

- `GET /detect` response includes `sigma_tier: str`
- `sigma_tier` is one of the 6 values above, determined by `sigma_confidence`
- Test: `sigma_confidence = 2.3` → `sigma_tier == "INTERESTING"`
- Test: `sigma_confidence = 5.2` → `sigma_tier == "EXCEPTIONAL"`

#### Proposed Fix

1. Add `sigma_tier: str = "ROUTINE"` to `DetectionResponse`
2. Add `_sigma_tier(sigma: float) -> str` function in `detection.py` schema or endpoint
3. Populate in `detect_neo()`: `sigma_tier=_sigma_tier(result.sigma_confidence)`

---

### UE-005: Data Source and Freshness Not Surfaced

**Persona**: P3 (System Integrator)

#### User Expectation

> "The detector gave me `artificial_probability: 0.037` for Apophis. Was this based
> on SBDB data? MPC? Horizons? And how old is it — did the fetcher pull fresh data
> today or return a 7-day cached result from last week?"

#### Root Cause

**`aneos_core/data/fetcher.py`**: `DataFetcher.fetch_neo_data()` writes `_source` and
`_fetched_at` into the raw orbital dict, but these keys are filtered out before
`OrbitalElements` construction (only `__dataclass_fields__` keys pass through).

**`aneos_core/data/models.py`**: `NEOData` does not store `data_source` or `fetched_at`
as top-level fields.

**`aneos_api/schemas/detection.py`**: `DetectionResponse` has no `data_source` or
`data_freshness` field.

**`aneos_api/endpoints/analysis.py:165-173`**: `detect_neo()` constructs a `DetectionResponse`
from only `result.*` fields; `neo.*` metadata never reaches the response.

#### Consequence Chain

1. P3 integrator cannot determine which data source was authoritative for a result
2. Cache staleness is invisible — a result based on week-old orbital elements looks
   identical to one based on fresh data
3. Reproducibility is compromised: same designation may return different results
   from different data sources without any indication of which was used

#### Acceptance Criteria

- `GET /detect` response includes `data_source: Optional[str]`
- `data_source` reflects which source (`SBDB`, `NEODyS`, `MPC`, `Horizons`) was authoritative
- `data_freshness: Optional[str]` is the ISO timestamp of when data was fetched
- Test: `GET /detect?designation=Apophis` → `data_source == "SBDB"`

#### Proposed Fix

1. Add `data_source: Optional[str] = None` and `data_freshness: Optional[str] = None`
   to `DetectionResponse`
2. `NEOData` already has `orbital_elements` — add `source_name: Optional[str]` field
   (read from `_source` key in the raw dict before filtering) or read from `_source`
   metadata before it's filtered
3. Pass `neo` metadata to `DetectionResponse` in `detect_neo()` endpoint

**Simplest short-term fix**: Store `_source` in `NEOData.source_name` during `DataFetcher`
construction (currently discarded at line ~204 of fetcher.py where only `__dataclass_fields__`
keys pass through).

---

### UE-006: `/health` Endpoint Schema Not Wired

**Persona**: P3 (System Integrator)

#### User Expectation

> "The OpenAPI spec shows `/health` returns `Dict[str, Any]`. That's useless for
> client generation — I need a typed schema. The `HealthResponse` type is in
> the spec's `components/schemas`... but it's never used by any endpoint."

#### Root Cause

**`aneos_api/app.py:261`**:
```python
@app.get("/health", response_model=Dict[str, Any])   # ← untyped
async def health_check():
```

**`aneos_api/schemas/health.py`**:
```python
class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, CheckResult] = Field(default_factory=dict)
    version: str = "0.7.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

**`aneos_api/app.py:188-200`** — `get_health_status()` returns:
```python
{
    'status': 'healthy' | 'initializing',
    'startup_time': ...,
    'services': {
        'analysis_pipeline': bool, 'ml_predictor': bool,
        'metrics_collector': bool, 'alert_manager': bool
    },
    'version': '2.0.0'
}
```

Shape mismatch: `HealthResponse.checks: Dict[str, CheckResult]` expects `CheckResult`
objects; `get_health_status()` returns `services: Dict[str, bool]` (booleans, not objects).
Neither shape is currently validated — FastAPI skips validation when `response_model=Dict[str, Any]`.

Additionally, the `emergency_care` dict added at line 276 (with Star Trek references)
is in the response body but would be rejected by a strict `HealthResponse` schema:
```python
health_status['emergency_care'] = {
    'services_auto_initialized': bool,
    'intensive_care_active': True,
    'federation_romulan_cooperation': 'active'  # ← debugging artifact
}
```

#### Consequence Chain

1. OpenAPI clients cannot generate typed `HealthResponse` deserializers
2. `HealthResponse` and `CheckResult` appear in `components/schemas` but are never
   used as response_model — they are dead schema entries
3. The Star Trek debugging metadata leaks into production responses

#### Acceptance Criteria

- `/health` uses `response_model=HealthResponse`
- `get_health_status()` returns a dict compatible with `HealthResponse`
- `checks` contains at minimum: `pipeline`, `sbdb`, `neodys` keys with `CheckResult` shape
- Star Trek fields removed from response
- Test: `GET /health` → validates against `HealthResponse` model without errors

#### Proposed Fix

1. Adapt `get_health_status()` to return `HealthResponse`-compatible dict:
   - Map `services: {pipeline: bool}` → `checks: {pipeline: CheckResult(status="ok"|"error", detail="...")}`
   - Remove `emergency_care` block and Star Trek comments
2. Update `@app.get("/health", response_model=HealthResponse)`
3. Import `HealthResponse` in `app.py`

---

### UE-007: Interpretation String is Generic

**Persona**: P4 (Casual Explorer)

#### User Expectation

> "The API said `'interpretation': 'sigma_confidence = rarity under natural NEO null
> hypothesis. artificial_probability incorporates 0.1% base rate prior.'` — but I
> just asked about 2024 YR4. What does this mean *for 2024 YR4*? The same string
> appeared for every object I queried."

#### Root Cause

**`aneos_api/schemas/detection.py:21-24`**:
```python
interpretation: str = (
    "sigma_confidence = rarity under natural NEO null hypothesis. "
    "artificial_probability incorporates 0.1% base rate prior."
)
```

This is a **class-level default** — the same static string for every object, every query.
The endpoint does not override it:

**`aneos_api/endpoints/analysis.py:165-173`**: `DetectionResponse(...)` is constructed
without specifying `interpretation=`, so the class default is used.

#### Consequence Chain

1. P4 user gets a boilerplate technical statement that doesn't explain their object
2. A user querying a σ=4.8 "ANOMALOUS" object and a σ=0.3 "ROUTINE" object see
   identical interpretations — no gradation of concern
3. The interpretation does not mention which evidence drove the result
4. Non-experts cannot use the API as a standalone tool

#### Acceptance Criteria

- `interpretation` field in response is unique per object and per result
- Includes: object name, sigma tier label, top evidence type, base rate context
- Example: `"2024 YR4 shows INTERESTING orbital characteristics (σ=2.3, ORBITAL_DYNAMICS dominant). This is statistically unusual but consistent with natural NEO population variation. artificial_probability 0.1% (base rate limited; propulsion evidence needed to raise above 5%)."`
- Test: Query "Apophis" and "Bennu" — interpretations differ in content (not just name)

#### Proposed Fix

1. Remove class-level default from `DetectionResponse.interpretation`
2. Add `_build_interpretation(designation, sigma_confidence, sigma_tier, top_evidence_type, bayesian_prob)` helper in `endpoints/analysis.py`
3. Populate `interpretation=_build_interpretation(...)` in `detect_neo()`

---

### UE-008: Spacecraft Veto Not Surfaced in API

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I queried Tesla Roadster via `GET /detect` and got `is_artificial: false`
> (σ=1.7, below 5σ threshold). But I know Tesla Roadster is artificial! The
> system knows it too — it's in the THETA SWARM's known spacecraft catalog.
> Why didn't the veto fire through the API?"

#### Root Cause

**`aneos_core/validation/human_hardware_analysis.py`**: The THETA SWARM module
implements TLE cross-referencing, constellation pattern detection, and material
fingerprinting. It can definitively identify known spacecraft.

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py:766-848`**:
`analyze_neo_validated()` does not call any THETA SWARM method. It only runs orbital
and physical anomaly scoring plus optional course correction / trajectory / propulsion
analysis. The known spacecraft catalog check is absent from this code path.

**`aneos_api/schemas/detection.py`**: `DetectionResponse` has no `spacecraft_veto: bool`
or `veto_reason: str` field.

**`aneos_api/endpoints/analysis.py:161-162`**:
```python
detector = ValidatedSigma5ArtificialNEODetector()
result = detector.analyze_neo_validated(orbital_dict)
```
No catalog check before or after `analyze_neo_validated()`.

**Consequence**: A known spacecraft (Tesla Roadster, σ=1.7) is returned as
`is_artificial: false` through the API, which contradicts ground truth and
misleads P1 researchers who rely on the API for classification.

#### Consequence Chain

1. Researcher queries Tesla Roadster via API → `is_artificial: false`
2. CLI menu correctly identifies it via `GroundTruthDatasetBuilder.ARTIFICIAL_OBJECTS`
3. The API and the menu disagree — no user can trust the API result for known spacecraft
4. The GroundTruthDatasetBuilder has a 9-object corpus: if designation matches, veto should fire

#### Acceptance Criteria

- `GET /detect` for known spacecraft returns `spacecraft_veto: true` and `is_artificial: true`
- `veto_reason: str` explains the source (e.g., `"Known spacecraft: Tesla Roadster (2018-017A)"`)
- The veto check precedes statistical detection — if veto fires, skip full analysis
- Test: `GET /detect?designation=2018+A1` → `spacecraft_veto: true, is_artificial: true`
- Test: `GET /detect?designation=Apophis` → `spacecraft_veto: false, is_artificial: false`

#### Proposed Fix

1. Add `spacecraft_veto: bool = False` and `veto_reason: Optional[str] = None` to `DetectionResponse`
2. In `detect_neo()` endpoint: before calling `detector.analyze_neo_validated()`, check
   `GroundTruthDatasetBuilder.KNOWN_ARTIFICIAL` catalog (already in codebase) or a
   simple name→NORAD mapping; if matched, return `is_artificial=True, spacecraft_veto=True`
3. For objects not in catalog: proceed with statistical detection, set `spacecraft_veto=False`

---

## P3 Gaps

---

### UE-009: Export Non-Functional

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I ran 20 analyses through the menu and want to export them to CSV for my paper.
> `POST /export` says it succeeded. `GET /export/{id}/download` returns
> `b'Mock export data'`. This is a deprecated endpoint that returns placeholder bytes."

#### Root Cause

**`aneos_api/endpoints/analysis.py:330`**:
```python
@router.get("/export/{export_id}/download", deprecated=True)
async def download_export(...):
    """Download completed export file. Not yet implemented — returns placeholder data."""
    ...
    return StreamingResponse(
        iter([b"Mock export data"]),
        ...
    )
```

**`aneos_api/endpoints/analysis.py:439-451`** — `_process_export()`:
```python
async def _process_export(export_id: str, request: ExportRequest):
    await asyncio.sleep(2)
    job = _export_jobs[export_id]
    job['status'] = 'completed'
    job['download_url'] = f'/api/v1/analysis/export/{export_id}/download'
    job['file_size_bytes'] = 1024  # Mock size
```

No actual serialization of `_analysis_cache`. The `ExportFormat` enum is defined
in `models.py` (JSON, CSV, XLSX, PDF) but none are implemented.

#### Consequence Chain

1. Users post-analysis have no machine-readable output path through the API
2. The menu has JSON export (`[8] Export Results`) but it writes to local files —
   not accessible to remote API consumers
3. `deprecated=True` in the router signals the endpoint is broken, but the
   `POST /export` endpoint still succeeds, creating false expectation

#### Acceptance Criteria

- `GET /export/{id}/download` returns valid JSON or CSV content for completed exports
- JSON export: all items in `_analysis_cache` serialized as array of `AnalysisResponse`
- CSV export: designation, sigma_confidence, artificial_probability, classification per row
- `file_size_bytes` reflects actual content size
- `deprecated=True` removed from the download endpoint once implemented

#### Proposed Fix

1. In `_process_export()`: iterate `_analysis_cache.values()`; serialize to JSON/CSV
2. Store result in memory (`io.BytesIO`) or temp file; update `job['content']`
3. `download_export()` returns `StreamingResponse(iter([job['content']]), ...)`
4. Remove `deprecated=True` from route decorator
5. Scope: JSON and CSV only (skip XLSX/PDF)

---

### UE-010: Auth Blocks Research API Use

**Persona**: P3 (System Integrator)

#### User Expectation

> "I'm running aNEOS locally for research. `GET /detect?designation=Apophis` returns
> 401 Unauthorized because I'm not sending `Authorization: Bearer mock_admin_token`.
> But in development mode, why do I need to authenticate to read public astronomical data?"

#### Root Cause

**`aneos_api/auth.py`** — `get_current_user()` — accepted mock flow:
```python
if token == "mock_admin_token":
    return MOCK_USERS['admin']
```
Any other token raises `HTTPException(401)`. No anonymous access is allowed even in dev mode.

**`aneos_api/endpoints/analysis.py:135`**:
```python
async def detect_neo(
    designation: str = Query(...),
    current_user: Optional[Dict] = Depends(get_current_user)
):
```

`get_current_user` is required (not optional) — the `Optional[Dict]` type hint is
misleading; the dependency always raises 401 if the mock token is absent.

**`aneos_api/auth.py`** — `_assert_auth_configured()` startup guard: added in Phase 6D
to block non-dev deployments without env keys. This is correct. But in dev mode,
requiring `mock_admin_token` for every read request is unnecessary friction.

#### Consequence Chain

1. P3 researcher starts local server: `uvicorn aneos_api.app:create_app --factory`
2. Calls `GET /detect?designation=Apophis` → 401 Unauthorized
3. Must know the mock token value (`mock_admin_token`) which is buried in source code
4. No documentation in OpenAPI spec about how to authenticate

#### Acceptance Criteria

- In dev mode: read endpoints (`GET /detect`, `GET /health`) are accessible without auth
- Write/sensitive endpoints remain protected
- Authentication scheme documented in OpenAPI `securitySchemes`
- OR: A clear `X-API-Key: dev` bypass documented for local development

#### Proposed Fix

1. Make `current_user` truly optional for read endpoints in dev mode:
   ```python
   current_user: Optional[Dict] = Depends(get_current_user_optional)
   ```
   where `get_current_user_optional` returns `None` (not 401) if no token provided
2. Check `ANEOS_ENV == "development"` in `get_current_user_optional` and skip auth
3. Document `X-API-Key: mock_admin_token` or `Authorization: Bearer mock_admin_token`
   in the OpenAPI spec `securitySchemes`

---

## Cross-Cutting Findings

### Finding C-1: The Critical Path Is Incomplete

The critical path from user input to useful outcome has a gap at the **API surface**:

```
DataFetcher ──→ Indicators ──→ Sigma5Detector ──→ Sigma5DetectionResult
                                                        │
                                              ┌─────────▼─────────┐
                                              │   API serializes:  │
                                              │   sigma_confidence │
                                              │   artificial_prob  │
                                              │   evidence_count   │ ← too thin
                                              │                    │
                                              │   NOT serialized:  │
                                              │   evidence_sources │ (UE-001)
                                              │   spacecraft_veto  │ (UE-008)
                                              │   sigma_tier       │ (UE-004)
                                              │   data_source      │ (UE-005)
                                              └────────────────────┘
```

The detector produces rich, well-structured output (`Sigma5DetectionResult`).
The API discards most of it. Fixing UE-001, UE-004, UE-005, UE-007, UE-008
is essentially: **stop discarding the result**.

### Finding C-2: Impact Assessment Has No API Entry Point

`ImpactProbabilityCalculator`, `ImpactResponse`, and `DataFetcher` are all functional
in isolation. They have never been connected end-to-end through the API. This is the
biggest **functional gap** for persona P2 (Planetary Defense Analyst), who has no
programmatic path to the secondary mission output.

### Finding C-3: Batch Is Architecturally Disconnected

The batch endpoint depends on `AnalysisPipeline` (not `DataFetcher`), creating a
completely different (and currently uninitialized) dependency chain than the working
`GET /detect` endpoint. The fix is to mirror the `GET /detect` implementation across
a batch loop, not to fix `AnalysisPipeline` initialization.

### Finding C-4: Health Endpoint Has Debugging Artifacts in Production

The `emergency_care` block with Star Trek references in `get_health_status()` is a
production code path (not gated by a debug flag). It returns `'federation_romulan_cooperation': 'active'`
to all API consumers. This is not a security issue but violates the principle that
API responses should be intentional and stable.

---

## Outcome Quality Assessment (Audited)

| Expectation | Delivers | Delivers to API | Verification |
|-------------|----------|-----------------|--------------|
| Correct classification | ✅ sens=1.00 spec=1.00 | ✅ `is_artificial` field | Tests pass |
| Evidence breakdown | ✅ `evidence_sources` list | ❌ `evidence_count` only | UE-001 |
| Sigma interpretation | ✅ numeric value | ❌ no tier/label | UE-004 |
| Generic interpretation | ✅ static string | ⚠ same for all objects | UE-007 |
| Known spacecraft veto | ✅ ground truth DB | ❌ not called in API path | UE-008 |
| Impact probability | ✅ ImpactProbabilityCalculator | ❌ no REST endpoint | UE-002 |
| Batch analysis | ⚠ partial (pipeline dep) | ❌ status always mocked | UE-003 |
| Data source | ✅ `_source` in raw dict | ❌ filtered before NEOData | UE-005 |
| Health status | ✅ `get_health_status()` | ⚠ untyped `Dict[str, Any]` | UE-006 |
| Export | ❌ mock bytes only | ❌ `Mock export data` | UE-009 |
| Auth usability | ✅ startup guard | ⚠ mock token undocumented | UE-010 |

---

## Phase 8 Implementation Plan

### 8A — Evidence Exposure (UE-001, UE-004, UE-007) — **Highest ROI**

All three gaps are in the same two files (`schemas/detection.py`, `endpoints/analysis.py`).
No new endpoints needed. Fixes existing `GET /detect` to return richer data.

**Files**: `aneos_api/schemas/detection.py`, `aneos_api/endpoints/analysis.py`
**New fields**: `evidence_sources: List[EvidenceSummary]`, `sigma_tier: str`, dynamic `interpretation`
**Tests**:
- `GET /detect?designation=Apophis` → `evidence_sources` non-empty
- `sigma_tier` matches expected range for known σ values
- `interpretation` contains designation name

**Verification**: `make spec` → `EvidenceSummary` in `components/schemas`

---

### 8B — Impact REST API (UE-002)

Wire `ImpactProbabilityCalculator` to `GET /impact` using `ImpactResponse` schema.

**Files**: `aneos_api/endpoints/analysis.py`, `aneos_api/schemas/impact.py`
**New endpoint**: `GET /impact?designation=X`
**Existing schema**: `ImpactResponse` already defined; verify field alignment with `ImpactProbability` dataclass
**Tests**:
- `GET /impact?designation=99942` → `risk_level` is valid string
- `ImpactResponse` in `docs/api/openapi.json` after `make spec`

---

### 8C — Batch Fix (UE-003)

Replace `AnalysisPipeline` dependency with direct `DataFetcher` + detector path.
Add in-memory `_batch_store` for job state. Real results populated by background task.

**Files**: `aneos_api/endpoints/analysis.py`
**Changes**:
- Remove `pipeline: AnalysisPipeline = Depends(get_analysis_pipeline)` from `analyze_batch`
- Add `_batch_store: Dict[str, Dict]`
- Fix `get_batch_status()` to read from `_batch_store`
**Tests**:
- Submit `["Apophis", "Bennu"]` batch; poll until `completed == 2`; verify results non-empty

---

### 8D — API Surface Completeness (UE-005, UE-006, UE-008)

**UE-005** (data source): Add `source_name` to `NEOData`; propagate to `DetectionResponse`
**UE-006** (health schema): Fix `get_health_status()` shape; add `response_model=HealthResponse`; remove Star Trek artifacts
**UE-008** (spacecraft veto): Catalog pre-check in `detect_neo()`; add `spacecraft_veto` + `veto_reason` to `DetectionResponse`

**Files**: `aneos_core/data/models.py` (NEOData), `aneos_core/data/fetcher.py`, `aneos_api/app.py`, `aneos_api/endpoints/analysis.py`, `aneos_api/schemas/detection.py`
**Tests**:
- `GET /detect?designation=Apophis` → `data_source == "SBDB"`
- `GET /health` → validates against `HealthResponse` model
- `GET /detect?designation=2018+A1` → `spacecraft_veto: true`

---

### 8E — Usability (UE-009, UE-010) — **Lower priority**

**UE-009** (export): Implement JSON + CSV serialization in `_process_export()`
**UE-010** (auth): Make read endpoints optionally unauthenticated in dev mode; document mock token in spec

---

## Gap Summary Table (UELO v1.0 → v1.1 after Phase 8)

| ID | Gap | Priority | Status after Phase 8 |
|----|-----|----------|----------------------|
| UE-001 | Evidence sources not in API | P1 | CLOSED Phase 8A |
| UE-002 | No impact REST endpoint | P1 | CLOSED Phase 8B |
| UE-003 | Batch non-functional | P1 | CLOSED Phase 8C |
| UE-004 | No sigma tier label | P2 | CLOSED Phase 8A |
| UE-005 | Data source not surfaced | P2 | CLOSED Phase 8D |
| UE-006 | /health schema not wired | P2 | CLOSED Phase 8D |
| UE-007 | Generic interpretation | P2 | CLOSED Phase 8A |
| UE-008 | Spacecraft veto not in API | P2 | CLOSED Phase 8D |
| UE-009 | Export non-functional | P3 | CLOSED Phase 8E |
| UE-010 | Auth blocks research use | P3 | CLOSED Phase 8E |

**Projected outcome after Phase 8**: All 4 user journeys (J1–J4) deliver their
expected outcome through the REST API. P4 (casual explorer) can rely on the CLI menu
which already provides most outputs; the API surface catches up to menu fidelity.

---

*This document is the executable input for Phase 8 planning. Each gap has acceptance
criteria that serve as test specifications. The implementation order (8A → 8B → 8C → 8D → 8E)
is ranked by user impact and coupling risk. 8A has zero coupling risk — it only adds
fields to existing responses.*
