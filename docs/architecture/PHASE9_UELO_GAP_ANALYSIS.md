# aNEOS — Phase 9 UELO Gap Analysis

**Version**: 1.0
**Date**: 2026-03-07
**Supersedes**: UELO_GAP_ANALYSIS.md v1.1 (Phase 8 — all UE-001–UE-010 CLOSED)
**Derived from**: Live code audit post-Phase 8 implementation
**Test baseline**: 59 pass / 0 fail (Phase 8 complete)

---

## Purpose

Phase 8 closed all 10 user-expectation gaps identified in UELO_GAP_ANALYSIS.md v1.0.
However, implementation of Phase 8 revealed a second layer of gaps — cases where the
newly-wired surface exposes **thinner outputs than the internal logic can produce**.

The theme shifts from *"the outcome layer is disconnected from the experience layer"*
(Phase 8) to **"the experience layer is connected but undersupplied"** (Phase 9):
the API endpoints exist, but they call their underlying engines with fewer inputs than
those engines can accept, or they omit fields the engines already compute.

---

## Severity Legend

| Symbol | Meaning |
|--------|---------|
| P1 | Blocks a primary user journey or produces misleading/incomplete output |
| P2 | Degrades quality; workarounds possible but results are demonstrably weaker |
| P3 | Reduces completeness or convenience; output is still valid without the fix |

---

## Gap Index

| ID | Title | Persona | Priority | Phase 9 Sub |
|----|-------|---------|----------|-------------|
| UE-011 | Detector called with 3 of 5 available data inputs | P1, P3 | P1 | 9A |
| UE-013 | Impact endpoint omits `PhysicalProperties` → energy/crater often `null` | P2 | P1 | 9A |
| UE-012 | `EvidenceSummary` omits `confidence_interval` and `sample_size` | P1 | P2 | 9B |
| UE-014 | Batch runs sequentially — large batches will timeout | P3 | P2 | 9B |
| UE-015 | `data_freshness` reflects API call time, not data fetch time | P3 | P2 | 9C |
| UE-016 | No `POST /detect` accepting raw orbital elements | P3 | P3 | 9C |
| UE-017 | Typed `ErrorResponse` schema unused — all errors are raw `HTTPException` | P3 | P3 | 9D |
| UE-018 | No orbital history REST endpoint (menu-only, no API path) | P1, P2 | P3 | 9D |

---

## P1 Gaps

---

### UE-011: Detector Called With 3 of 5 Available Data Inputs

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I called `GET /detect?designation=2020+SO` and got `evidence_count: 1` —
> one orbital dynamics evidence source. But I know the system has thermal,
> physical, and trajectory analysis. Why is only one evidence type firing?
> The batch results also only show one evidence source per object."

#### Root Cause

**`aneos_api/endpoints/analysis.py` — `detect_neo()`**:

```python
orbital_dict = {
    "a": oe.semi_major_axis,
    "e": oe.eccentricity,
    "i": oe.inclination,
}
detector = ValidatedSigma5ArtificialNEODetector()
result = detector.analyze_neo_validated(orbital_dict)   # ← only orbital_dict
```

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py:766-770`**:

```python
def analyze_neo_validated(self,
    orbital_elements: Dict[str, float],
    physical_data: Dict[str, Any] = None,        # ← optional, ignored by API
    orbital_history: List[Dict[str, Any]] = None, # ← optional, ignored by API
    close_approach_history: List[Dict[str, Any]] = None,  # ← optional, ignored by API
    observation_data: Dict[str, Any] = None       # ← optional, ignored by API
) -> Sigma5DetectionResult:
```

`NEOData` (the result of `fetcher.fetch_neo_data(designation)`) already contains:
- `neo.physical_properties` → maps to `physical_data`
- `neo.close_approaches` → maps to `close_approach_history`

Both are populated by the `DataFetcher` when SBDB data is available. The API
fetches them but immediately discards them before calling the detector.

**Consequence**: The `GET /detect` endpoint always returns `evidence_count: 1`
(orbital dynamics only), even when SBDB physical data is available. The
detector's physical, trajectory, and propulsion analysis (4 additional evidence
modules) never fires through the API path. `_run_batch_detection()` has the
same narrow call.

#### Consequence Chain

1. P1 researcher queries Apophis → `evidence_count: 1`, `evidence_sources[0].type: "orbital_dynamics"`
2. SBDB has diameter + albedo for Apophis — physical evidence should also fire
3. Physical evidence is computed in the CLI menu path (which uses `DataFetcher` differently)
4. API users get fundamentally weaker analysis than menu users for the same object
5. `sigma_confidence` from 1 evidence source is lower than from 3-4 fused sources

#### Acceptance Criteria

- `GET /detect?designation=Apophis` → `evidence_count >= 2` when physical properties available
- `evidence_sources` contains at least `orbital_dynamics` + `physical_properties` for data-rich objects
- `_run_batch_detection()` also passes available physical/approach data
- No new test failures

#### Proposed Fix

In `detect_neo()`, after `neo = fetcher.fetch_neo_data(designation)`:

```python
# Build physical_data dict from PhysicalProperties if available
physical_data = None
if neo.physical_properties:
    pp = neo.physical_properties
    physical_data = {
        "diameter": pp.diameter_km,
        "albedo": pp.albedo,
        "spectral_type": pp.spectral_type,
    }

# Build close_approach_history from CloseApproach list if available
close_approach_history = None
if neo.close_approaches:
    close_approach_history = [
        {"date": ca.date, "distance_au": ca.distance_au,
         "velocity_kms": ca.relative_velocity_kms}
        for ca in neo.close_approaches
    ]

result = detector.analyze_neo_validated(
    orbital_dict,
    physical_data=physical_data,
    close_approach_history=close_approach_history,
)
```

Apply the same pattern in `_run_batch_detection()`.

---

### UE-013: Impact Endpoint Omits `PhysicalProperties` → Energy/Crater Often `null`

**Persona**: P2 (Planetary Defense Analyst)

#### User Expectation

> "I called `GET /impact?designation=Apophis` and got `impact_energy_mt: null,
> crater_diameter_km: null`. Those are the most important outputs for planetary
> defense planning. Why are they null? The CLI menu shows them."

#### Root Cause

**`aneos_api/endpoints/analysis.py` — `impact_neo()`**:

```python
result = calc.calculate_comprehensive_impact_probability(
    orbital_elements=oe,
    close_approaches=neo.close_approaches or [],
)
```

**`aneos_core/analysis/impact_probability.py:611-619`**:

```python
def _compute_impact_energy(...):
    diameter_km = self._get_diameter_km(orbital_elements, physical_props)
    if not diameter_km:
        return None   # ← returns null if no diameter available
```

`_get_diameter_km()` prefers `physical_props.diameter_km` over a fallback.
But `calculate_comprehensive_impact_probability()` accepts a `physical_properties`
parameter that the `impact_neo()` endpoint never passes. The `DataFetcher`
populates `neo.physical_properties` for Apophis (SBDB has diameter data), but
it is discarded before reaching the calculator.

`crater_diameter_km` and `impact_energy_mt` both depend on the same `_get_diameter_km()`
path. Without `physical_properties`, both are `null` for all objects that don't
have diameter embedded in their `OrbitalElements` (which no longer carry physical
fields after G-011 cleanup in Phase 6).

#### Consequence Chain

1. P2 analyst queries Apophis → `impact_energy_mt: null, crater_diameter_km: null`
2. The two most operationally relevant fields are blank
3. SBDB has Apophis diameter (0.375 km) — the data is present but not wired
4. Users cannot assess impact severity through the API

#### Acceptance Criteria

- `GET /impact?designation=Apophis` → `impact_energy_mt` and `crater_diameter_km` non-null
- The fix passes `neo.physical_properties` to `calculate_comprehensive_impact_probability()`
- Fields remain `null` only for objects with genuinely unknown diameters

#### Proposed Fix

`calculate_comprehensive_impact_probability()` already accepts `physical_properties`
(the `ImpactProbabilityCalculator._get_diameter_km()` helper checks it). Pass it:

```python
result = calc.calculate_comprehensive_impact_probability(
    orbital_elements=oe,
    close_approaches=neo.close_approaches or [],
    physical_properties=neo.physical_properties,  # ← add this
)
```

Verify `calculate_comprehensive_impact_probability()` signature accepts
`physical_properties` keyword argument (add if not present).

---

## P2 Gaps

---

### UE-012: `EvidenceSummary` Omits `confidence_interval` and `sample_size`

**Persona**: P1 (Anomaly Researcher)

#### User Expectation

> "I got `evidence_sources[0].anomaly_score: 1.643`. How precise is that?
> Is it ±0.01 or ±2.0? And how big was the reference population? I need
> uncertainty bounds to cite this result in a paper."

#### Root Cause

**`aneos_api/schemas/detection.py` — `EvidenceSummary`** (Phase 8):

```python
class EvidenceSummary(BaseModel):
    type: str
    anomaly_score: float
    p_value: float
    quality_score: float
    effect_size: float
```

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py:45-50`** — `EvidenceSource`:

```python
@dataclass
class EvidenceSource:
    evidence_type: EvidenceType
    anomaly_score: float
    confidence_interval: Tuple[float, float]   # ← 95% CI — not in API
    sample_size: int                            # ← reference N — not in API
    p_value: float
    effect_size: float
    quality_score: float
```

Phase 8 mapped 5 of 7 fields. `confidence_interval` (95% CI of anomaly score)
and `sample_size` (size of reference population, e.g. N=247 natural NEOs) are
computed by the detector and available in the result but stripped at the schema boundary.

#### Acceptance Criteria

- `evidence_sources[i].confidence_interval` is a two-element list `[lower, upper]`
- `evidence_sources[i].sample_size` is the integer reference population size
- Both fields appear in `EvidenceSummary` schema in OpenAPI spec

#### Proposed Fix

1. Add to `EvidenceSummary`:
   ```python
   confidence_interval: List[float] = []   # [lower_95, upper_95]
   sample_size: int = 0
   ```
2. In `detect_neo()` serialization:
   ```python
   EvidenceSummary(
       ...
       confidence_interval=list(e.confidence_interval),
       sample_size=e.sample_size,
   )
   ```

---

### UE-014: Batch Runs Sequentially — Large Batches Will Timeout

**Persona**: P3 (System Integrator)

#### User Expectation

> "I submitted 50 designations. After 5 minutes it's still `status: processing`.
> The batch is sequential — each designation waits for the previous one to
> finish fetching from SBDB."

#### Root Cause

**`aneos_api/endpoints/analysis.py` — `_run_batch_detection()`** (Phase 8):

```python
for designation in designations:
    try:
        neo = fetcher.fetch_neo_data(designation)   # ← network call, sequential
        ...
```

Each SBDB fetch takes 1-3 seconds. A 50-object batch = 50-150 seconds in a
single async function. The background task runs in the asyncio event loop; a
blocking `for` loop over network calls will starve other requests.

The UELO.md §7 noted: *"Use ThreadPoolExecutor for concurrent fetch + detect"*.

#### Acceptance Criteria

- A 10-object batch completes in under 30 seconds (target: ~10s with parallelism)
- `completed` count increments as results arrive (not all-at-once at the end)
- HTTP requests to the API remain responsive during batch execution

#### Proposed Fix

Replace the sequential loop with `asyncio.gather()` over a coroutine per designation:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

_batch_executor = ThreadPoolExecutor(max_workers=8)

async def _run_batch_detection(batch_id: str, designations: List[str]):
    store = _batch_store[batch_id]
    loop = asyncio.get_event_loop()

    async def _detect_one(designation: str) -> dict:
        # Runs in thread pool to avoid blocking event loop
        return await loop.run_in_executor(
            _batch_executor, _detect_sync, designation
        )

    tasks = [_detect_one(d) for d in designations]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for designation, result in zip(designations, results):
        if isinstance(result, Exception):
            store['failed'] += 1
            store['results'].append({'designation': designation, 'status': 'failed', 'error': str(result)})
        else:
            store['completed'] += 1
            store['results'].append(result)
    store['status'] = 'completed'
    store['finished_at'] = datetime.now().isoformat()
```

Where `_detect_sync` is the current sequential body (DataFetcher + detector call),
moved to a synchronous function safe for `run_in_executor`.

---

### UE-015: `data_freshness` Reflects API Call Time, Not Data Fetch Time

**Persona**: P3 (System Integrator)

#### User Expectation

> "`data_freshness` is always the current time — it's useless for cache staleness
> detection. I can't tell if the orbital elements are 10 minutes old or 7 days
> old. I need the timestamp of when SBDB was actually queried."

#### Root Cause

**`aneos_api/endpoints/analysis.py` — `detect_neo()`** (Phase 8):

```python
data_freshness = datetime.now().isoformat()   # ← time of API call
```

`DataFetcher` stores `_fetched_at` metadata in the raw orbital dict, but it is
filtered out before `OrbitalElements` construction (only `__dataclass_fields__`
keys pass through). `NEOData` has no `fetched_at` field.

The fix requires propagating the fetch timestamp from the raw data dict into
a `NEOData.fetched_at: Optional[datetime]` field.

#### Acceptance Criteria

- `data_freshness` in `DetectionResponse` is the ISO timestamp when SBDB/NEODyS was queried
- For cached results, `data_freshness` reflects the original fetch time (not cache hit time)
- When no timestamp is available from source, `data_freshness` may fall back to current time with a note

#### Proposed Fix

1. Add `fetched_at: Optional[datetime] = None` to `NEOData` (`aneos_core/data/models.py`)
2. In `DataFetcher`, after raw dict construction, write `neo.fetched_at = datetime.now()`
   (or parse `_fetched_at` from the raw dict if present)
3. In `detect_neo()`: `data_freshness = neo.fetched_at.isoformat() if neo.fetched_at else datetime.now().isoformat()`

---

## P3 Gaps

---

### UE-016: No `POST /detect` Accepting Raw Orbital Elements

**Persona**: P3 (System Integrator)

#### User Expectation

> "I have orbital elements from my own telescope reduction — `a=1.12, e=0.24,
> i=6.3`. I don't have a JPL designation yet. I can't use `GET /detect` because
> it requires a designation for DataFetcher lookup. I need to POST the elements
> directly."

#### Root Cause

`GET /detect` requires a designation and uses `DataFetcher` to resolve it.
There is no endpoint that accepts raw `{a, e, i, ...}` and runs the detector
without a lookup step.

The `ValidatedSigma5ArtificialNEODetector.analyze_neo_validated()` accepts a
plain `Dict[str, float]` — it does not require a designation. The missing piece
is only the endpoint.

#### Acceptance Criteria

- `POST /detect` accepts body: `{"a": float, "e": float, "i": float, "designation": optional str}`
- Returns `DetectionResponse` (same schema as `GET /detect`)
- Spacecraft veto skipped if no designation provided; or designation used for veto check if present
- `data_source: "user_provided"` in response

#### Proposed Fix

Add `OrbitalInput` Pydantic model and `POST /detect` route:

```python
class OrbitalInput(BaseModel):
    a: float                        # semi-major axis (AU)
    e: float                        # eccentricity
    i: float                        # inclination (deg)
    designation: Optional[str] = None
    diameter_km: Optional[float] = None
    albedo: Optional[float] = None

@router.post("/detect", response_model=DetectionResponse)
async def detect_neo_raw(request: OrbitalInput, ...):
    ...
```

---

### UE-017: Typed `ErrorResponse` Schema Unused

**Persona**: P3 (System Integrator)

#### User Expectation

> "My OpenAPI client generated a typed `ErrorResponse` class from the spec.
> But every error I receive is `{'detail': '...'}` — FastAPI's default shape —
> not the `ErrorResponse` schema. My error handler crashes because the shapes
> don't match."

#### Root Cause

**`aneos_api/models.py`** defines:

```python
class ErrorResponse(BaseModel):
    error: str
    status_code: int
    timestamp: datetime
```

**`aneos_api/app.py`** — the exception handler:

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )
```

The handler already matches the `ErrorResponse` shape, but no endpoint declares
`responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}`.
FastAPI omits the schema from `components/schemas` unless it is referenced by at
least one endpoint response declaration.

`ErrorResponse` is dead code in the OpenAPI spec and invisible to client generators.

#### Acceptance Criteria

- `ErrorResponse` appears in `components/schemas` in OpenAPI spec
- At least `GET /detect` and `GET /impact` declare `responses={404: ..., 500: ...}`
- Client generators produce typed error deserializers

#### Proposed Fix

Add to `GET /detect` and `GET /impact` route decorators:

```python
@router.get(
    "/detect",
    response_model=DetectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No orbital data for designation"},
        500: {"model": ErrorResponse, "description": "Detection pipeline failure"},
    }
)
```

Import `ErrorResponse` from `..models` in `endpoints/analysis.py`.

---

### UE-018: No Orbital History REST Endpoint

**Persona**: P1 (Anomaly Researcher), P2 (Planetary Defense Analyst)

#### User Expectation

> "The CLI menu has `[2][5] Orbital Element History` which pulls a 10-year
> Keplerian element table from JPL Horizons. I can't access this through
> the API. It's the most useful feature for identifying non-gravitational
> accelerations."

#### Root Cause

**`aneos_menu.py`** — `_generate_orbital_history()` calls the JPL Horizons
ELEMENTS API and returns a time-series of Keplerian elements. This is wired
to the interactive menu only.

There is no `GET /history` or `GET /detect/history` endpoint in `analysis.py`.
The `_parse_horizons_element_table()` helper exists in `aneos_menu.py` but
is not accessible from the API package.

#### Consequence Chain

1. P1 researcher wants to verify orbital stability over 10 years → must use CLI
2. Non-gravitational acceleration detection (course correction evidence) can only
   be triggered from the menu, not the API
3. P3 integrators cannot build orbital-history-based workflows on top of aNEOS

#### Acceptance Criteria

- `GET /history?designation=X&years=10` returns a time-series of orbital elements
- Response is a list of `{epoch, a, e, i, node, peri, M}` dicts
- Endpoint uses the same Horizons ELEMENTS API path as the menu function
- Response model is defined in `schemas/`

#### Proposed Fix

1. Extract `_generate_orbital_history()` and `_parse_horizons_element_table()`
   from `aneos_menu.py` into `aneos_core/data/horizons.py` (or equivalent utility)
2. Add `OrbitalElementPoint` and `OrbitalHistoryResponse` schemas
3. Add `GET /history?designation=X&years=10` to `analysis.py` router

---

## Cross-Cutting Findings

### Finding C-5: The API Is Connected But Undersupplied

Phase 8 closed the "experience layer disconnected from outcome layer" theme.
Phase 9's theme is subtler: the connections exist, but each connection passes
only a subset of the available data:

```
DataFetcher.fetch_neo_data(designation)
    neo.orbital_elements   ──→ detect_neo() ──→ 3-key dict  ──→ detector  (1 evidence)
    neo.physical_properties      ↑ discarded                              (0 evidence)
    neo.close_approaches         ↑ discarded                              (0 evidence)

                                                      Full capability: 4-5 evidence sources
                                                      Actual API output: 1 evidence source
```

UE-011 and UE-013 are both instances of this pattern. The fix in both cases
is trivially small (pass existing variables through); the impact on evidence
richness is substantial.

### Finding C-6: Physical Properties Chain Is Broken at Two Points

The `PhysicalProperties` object flows:

```
SBDB → DataFetcher → NEOData.physical_properties
                           │
                     detect_neo()       ← UE-011: not passed to detector
                     impact_neo()       ← UE-013: not passed to calculator
```

Both API endpoints fetch `neo.physical_properties` from `NEOData` and then
ignore it. This is the single highest-ROI fix in Phase 9 — two endpoints,
two one-line changes, fixes two P1 gaps.

### Finding C-7: Evidence Quality Is Incomplete

`EvidenceSummary` (Phase 8) exposed enough for classification interpretation
(type, score, p_value). It does not expose enough for research citation
(confidence_interval, sample_size). A P1 researcher publishing results needs
uncertainty quantification — UE-012 is directly on the academic credibility path.

---

## Outcome Quality Re-Assessment (Post Phase 8)

| Expectation | Delivers | Delivers to API | Phase 9 Target |
|-------------|----------|-----------------|----------------|
| Evidence breakdown | ✅ full detail | ⚠ orbital only (1/5 inputs used) | UE-011 |
| Evidence uncertainty (CI) | ✅ computed | ❌ not in EvidenceSummary | UE-012 |
| Impact energy + crater | ✅ computed | ⚠ null when physical props absent | UE-013 |
| Batch performance | ✅ functional | ⚠ sequential; slow for large sets | UE-014 |
| Data freshness | ✅ stored internally | ⚠ API call time, not fetch time | UE-015 |
| Raw orbital POST | ✅ detector accepts dict | ❌ no POST /detect endpoint | UE-016 |
| Typed error schema | ✅ defined | ❌ not referenced in spec | UE-017 |
| Orbital history | ✅ menu path | ❌ no REST endpoint | UE-018 |

---

## Phase 9 Sub-Phase Sequence

```
9A (data richness)   → 9B (evidence quality + batch concurrency)
→ 9C (freshness + raw POST) → 9D (error schema + history)
```

`pytest tests/ -m "not network" -q` (59 pass, 0 fail) must hold after every sub-phase.

---

### 9A — Data Richness (UE-011, UE-013) — Highest ROI

Two one-to-two line changes in `aneos_api/endpoints/analysis.py`.
Zero new endpoints. Zero schema changes. Maximum evidence and impact output improvement.

**Files**: `aneos_api/endpoints/analysis.py`

**UE-011 changes** (in `detect_neo()` and `_run_batch_detection()`):
- Build `physical_data` dict from `neo.physical_properties` if present
- Build `close_approach_history` list from `neo.close_approaches` if present
- Pass both to `detector.analyze_neo_validated()`

**UE-013 change** (in `impact_neo()`):
- Pass `physical_properties=neo.physical_properties` to `calculate_comprehensive_impact_probability()`
- Verify calculator signature accepts this kwarg; add if absent

**Verify**: `GET /detect?designation=Apophis` → `evidence_count >= 2`
**Verify**: `GET /impact?designation=Apophis` → `impact_energy_mt` non-null

---

### 9B — Evidence Quality + Batch Concurrency (UE-012, UE-014)

**UE-012**: Add `confidence_interval: List[float]` and `sample_size: int` to `EvidenceSummary`.
Populate from `e.confidence_interval` and `e.sample_size` in `detect_neo()`.

**UE-014**: Replace sequential `for` loop in `_run_batch_detection()` with
`asyncio.gather()` over `loop.run_in_executor(_batch_executor, _detect_sync, d)`.
Add module-level `_batch_executor = ThreadPoolExecutor(max_workers=8)`.

**Files**: `aneos_api/schemas/detection.py`, `aneos_api/endpoints/analysis.py`

**Verify**: `make spec` → `confidence_interval` field in `EvidenceSummary` component

---

### 9C — Freshness + Raw POST (UE-015, UE-016)

**UE-015**:
1. Add `fetched_at: Optional[datetime] = None` to `NEOData` (`aneos_core/data/models.py`)
2. In `DataFetcher.fetch_neo_data()`, set `neo.fetched_at = datetime.now()` after fetch
3. In `detect_neo()`: use `neo.fetched_at.isoformat()` for `data_freshness`

**UE-016**:
1. Add `OrbitalInput` Pydantic model to `schemas/detection.py`
2. Add `POST /detect` to `endpoints/analysis.py` using `OrbitalInput` body
3. Spacecraft veto applied if `designation` provided in body

**Files**: `aneos_core/data/models.py`, `aneos_core/data/fetcher.py`,
           `aneos_api/schemas/detection.py`, `aneos_api/endpoints/analysis.py`

---

### 9D — Error Schema + History (UE-017, UE-018)

**UE-017**:
1. Import `ErrorResponse` from `..models` in `endpoints/analysis.py`
2. Add `responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}`
   to `GET /detect` and `GET /impact` decorators

**UE-018**:
1. Extract `_generate_orbital_history()` logic into `aneos_core/data/horizons.py`
2. Add `OrbitalElementPoint` and `OrbitalHistoryResponse` schemas
3. Add `GET /history?designation=X&years=10` endpoint

**Files**: `aneos_api/endpoints/analysis.py`, `aneos_core/data/horizons.py` (new),
           `aneos_api/schemas/` (new history models)

---

## Gap Summary Table

| ID | Gap | Priority | Target Sub-Phase |
|----|-----|----------|-----------------|
| UE-011 | Detector uses 3 of 5 available inputs | P1 | 9A |
| UE-013 | Impact energy/crater null (physical props not passed) | P1 | 9A |
| UE-012 | EvidenceSummary missing CI + sample_size | P2 | 9B |
| UE-014 | Batch sequential — large batches timeout | P2 | 9B |
| UE-015 | data_freshness = API call time, not fetch time | P2 | 9C |
| UE-016 | No POST /detect for raw orbital elements | P3 | 9C |
| UE-017 | ErrorResponse not referenced in OpenAPI spec | P3 | 9D |
| UE-018 | No orbital history REST endpoint | P3 | 9D |

**Deferred (no Phase 9 work)**:
- G-015: ML classifier activation (independent work stream)
- G-023: `aneos_menu.py` decomposition (separate refactor effort)
- Real JWT production auth (requires infrastructure changes)

---

*This document is the executable input for Phase 9 planning. UE-011 and UE-013 together
represent the highest ROI of any fixes in the entire UELO series: two small changes,
two P1 outcomes, no new endpoints, no schema breaks. Implement 9A first.*
