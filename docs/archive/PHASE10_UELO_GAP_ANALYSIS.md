# aNEOS — Phase 10 UELO Gap Analysis

**Version**: 1.0
**Date**: 2026-03-07
**Supersedes**: PHASE9_UELO_GAP_ANALYSIS.md v1.0 (Phase 9 — UE-011–UE-018 CLOSED)
**Derived from**: Live code audit post-Phase 9 implementation
**Test baseline**: 59 pass / 0 fail (Phase 9 complete)

---

## Purpose

Phase 9 closed the "connected but undersupplied" theme: API endpoints now pass
`physical_data` and `close_approach_history` to the detector, and `physical_properties`
to the impact calculator. However, code audit immediately after Phase 9 reveals a
**third layer of gaps**.

The Phase 9 theme was *"the connections exist but data is not passed through"*.
Phase 10's theme is **"the data is passed, but the sources never produce it, and
the cache throws it away"**:

1. `neo.close_approaches` is wired but always `[]` — no DataFetcher source fetches
   close approach data from the CAD API.
2. `to_dict()` / `from_dict()` (cache round-trip) drops `physical_properties` and
   `fetched_at` — after the first request, every cached result regresses to Phase 8
   quality.
3. The 3rd and 5th inputs to `analyze_neo_validated()` (`orbital_history`,
   `observation_data`) remain unconnected — the two most powerful evidence modules
   (`_analyze_course_corrections`, `_analyze_propulsion_signatures`) never fire
   through the API.

---

## Severity Legend

| Symbol | Meaning |
|--------|---------|
| P1 | Blocks a primary user journey or silently produces incorrect/regressed output |
| P2 | Degrades output quality; workarounds possible but results are demonstrably weaker |
| P3 | Reduces completeness or convenience; output is still valid without the fix |

---

## Gap Index

| ID | Title | Priority | Phase 10 Sub |
|----|-------|----------|-------------|
| UE-019 | `orbital_history` never fetched/passed → course correction evidence never fires | P1 | 10A |
| UE-020 | `close_approaches` always empty → trajectory evidence never fires (Phase 9 partial fix) | P1 | 10A |
| UE-024 | Cache round-trip drops `physical_properties` + `fetched_at` → evidence regresses on cache hit | P1 | 10A |
| UE-021 | `DetectionResponse` omits `combined_p_value`, `false_discovery_rate`, `analysis_metadata` | P2 | 10B |
| UE-022 | Batch results lack `evidence_sources`, `sigma_tier`, `interpretation` | P2 | 10B |
| UE-023 | `ImpactResponse` exposes 7 of 25+ `ImpactProbability` fields | P2 | 10B |
| UE-025 | `OrbitalInput` has no `orbital_history` field → `POST /detect` cannot trigger course correction | P3 | 10C |
| UE-026 | `GET /detect` re-fetches SBDB on every call — no freshness-aware cache bypass | P3 | 10C |

---

## P1 Gaps

---

### UE-019: `orbital_history` Never Fetched — Course Correction Evidence Never Fires

**Persona**: P1 (Anomaly Researcher)

#### User Expectation

> "You said the system detects course corrections — non-gravitational accelerations
> that indicate propulsion. I queried 2020 SO and got `evidence_count: 2` (orbital +
> physical). Where is the course correction evidence? That's the definitive smoking gun
> for an artificial object."

#### Root Cause

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py:784-786`**:

```python
if orbital_history:
    course_correction_evidence = self._analyze_course_corrections(orbital_history)
    evidence_sources.append(course_correction_evidence)
```

`_analyze_course_corrections()` is the detector's most powerful module — it detects
non-gravitational accelerations (course corrections) in an orbital element time-series.
It fires only if `orbital_history` is a non-empty list.

**`aneos_api/endpoints/analysis.py` — `detect_neo()`**:

```python
# Phase 9 passed physical_data and close_approach_history — but NOT orbital_history
result = detector.analyze_neo_validated(
    orbital_dict,
    physical_data=physical_data,
    close_approach_history=close_approach_history,
    # orbital_history=???  ← never constructed, never passed
)
```

**`aneos_core/data/sources/horizons.py`** now has `fetch_orbital_history()` (Phase 9,
UE-018). The `GET /history` endpoint calls it. But `detect_neo()` never calls it and
never passes the result to `analyze_neo_validated()`.

**Consequence**: `_analyze_course_corrections()` never fires through the API. For any
genuinely artificial object, the most powerful evidence — orbital drift signatures —
is silently omitted.

#### Consequence Chain

1. P1 researcher queries 2020 SO → `evidence_count: 2` (orbital + physical)
2. Course correction evidence (`EvidenceType.COURSE_CORRECTIONS`) is never computed
3. JPL Horizons has 10-year orbital element history for 2020 SO — the data exists
4. The menu path calls `_generate_orbital_history()` + feeds it to the detector
5. API path and menu path produce different evidence counts for the same object

#### Acceptance Criteria

- `detect_neo()` calls `HorizonsSource.fetch_orbital_history(designation)` after
  fetching `neo` and before calling `analyze_neo_validated()`
- `orbital_history` is built from the returned list and passed as 3rd argument
- For 2020 SO: `evidence_sources` contains an entry with
  `type == "course_corrections"`
- If Horizons history fetch fails (timeout / no data), fall through gracefully with
  `orbital_history=None` (no error, just skip that evidence module)
- No new test failures

#### Proposed Fix

**`aneos_api/endpoints/analysis.py` — in `detect_neo()` after the `oe` null check**:

```python
# Fetch orbital history from Horizons for course correction analysis
orbital_history = None
try:
    from aneos_core.data.sources.horizons import HorizonsSource
    h_source = HorizonsSource()
    raw_history = h_source.fetch_orbital_history(designation, years=10)
    if raw_history:
        orbital_history = raw_history  # List[Dict[str, Any]] from Horizons
except Exception as _hist_exc:
    logger.debug(f"Horizons history unavailable for {designation}: {_hist_exc}")

result = detector.analyze_neo_validated(
    orbital_dict,
    physical_data=physical_data,
    close_approach_history=close_approach_history,
    orbital_history=orbital_history,   # ← add this
)
```

Note: this adds a live network call to the `/detect` hot path. A 10-year span with
`STEP_SIZE=365d` fetches ~10 data points — typically < 2s. The fallback ensures
service continuity.

**Alternative (non-blocking)**: pre-fetch in `DataFetcher._fetch_from_all_sources()`
via the `HorizonsSource` and store history as `NEOData.orbital_history`.

---

### UE-020: `close_approaches` Always Empty — Trajectory Evidence Never Fires

**Persona**: P1 (Anomaly Researcher), P2 (Planetary Defense Analyst)

#### User Expectation

> "Phase 9 said close approach data is now passed to the detector. But the response
> shows `evidence_count: 2` — orbital + physical. `TRAJECTORY_PATTERNS` evidence never
> appears. Where is the close approach data coming from?"

#### Root Cause

Phase 9 (UE-011) correctly wired the API to pass `neo.close_approaches` to
`analyze_neo_validated()`. But none of the four DataFetcher sources populate
`NEOData.close_approaches`:

| Source | Populates `close_approaches`? |
|--------|-------------------------------|
| `SBDBSource` | ❌ — only orbital elements + physical properties |
| `NEODySSource` | ❌ — only orbital elements |
| `MPCSource` | ❌ — only orbital elements |
| `HorizonsSource` | ❌ — only orbital elements + physical properties |

The JPL SBDB Close Approach Data API (`cad.api`) is referenced in `settings.py`:

```python
cad_url: str = "https://ssd-api.jpl.nasa.gov/cad.api"
```

And in `historical_chunked_poller.py:481` it is called for bulk polling. But it is
not wired into the `DataFetcher` → `NEOData` path for individual designations.

**Consequence in code**:

```python
# detect_neo() — Phase 9 fix
close_approach_history = None
if neo.close_approaches:   # ← always False; list is always []
    close_approach_history = [...]

result = detector.analyze_neo_validated(
    orbital_dict,
    physical_data=physical_data,
    close_approach_history=None,   # ← always None, trajectory evidence never fires
)
```

#### Consequence Chain

1. `_analyze_trajectory_patterns()` never fires through the API
2. For Apophis (known close approach in 2029): CAD data exists but never fetched
3. Phase 9 fix (UE-011) is a correct wire but a dead wire — the source is not
   connected to the other end of the wire
4. Planetary defense use case (close approach velocity + distance anomaly) is blocked

#### Acceptance Criteria

- `DataFetcher.fetch_neo_data()` calls SBDB CAD API for the designation and populates
  `NEOData.close_approaches` with at least the nearest upcoming approach
- `GET /detect?designation=Apophis` → `evidence_sources` contains
  `type == "trajectory_patterns"`
- Empty close approaches (no known approaches) = `close_approach_history=None`
  (detector skips that module, no error)
- No new test failures

#### Proposed Fix

**Option A — `DataFetcher` integration** (preferred):

Add a CAD fetch step in `_fetch_from_all_sources()`:

```python
from .sources.sbdb import SBDBSource
# After primary sources complete, augment with CAD data
try:
    cad_data = self._fetch_close_approaches(designation)
    if cad_data:
        for approach in cad_data:
            neo_data.add_close_approach(approach)
except Exception:
    pass  # Close approach data is supplemental, not critical
```

`_fetch_close_approaches()` calls `{cad_url}?des={designation}&date-min=now&dist-max=0.2&limit=10`
and maps to `CloseApproach` objects.

**Option B — endpoint-level fetch** (faster to ship):

In `detect_neo()`, alongside the Horizons history call, also fetch CAD data directly:

```python
from aneos_core.config.settings import APIConfig
import requests

cad_approaches = []
try:
    cfg = APIConfig()
    resp = requests.get(cfg.cad_url, params={"des": designation,
        "date-min": "now", "dist-max": "0.2", "limit": "10"}, timeout=10)
    if resp.ok:
        cad_json = resp.json()
        for row in cad_json.get("data", []):
            cad_approaches.append({
                "date": row[3], "distance_au": float(row[4]),
                "velocity_kms": float(row[7]) if row[7] else None,
            })
except Exception:
    pass

if cad_approaches:
    close_approach_history = cad_approaches
```

---

### UE-024: Cache Round-Trip Drops `physical_properties` and `fetched_at`

**Persona**: P1 (Anomaly Researcher), P3 (System Integrator)

#### User Expectation

> "I called `GET /detect?designation=Apophis` twice. First call: `evidence_count: 3`.
> Second call (cache hit): `evidence_count: 1`. The API is non-deterministic."

#### Root Cause

**`aneos_core/data/models.py` — `NEOData.to_dict()`** (line 302):

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        "designation": self.designation,
        "orbital_elements": self.orbital_elements.to_dict() if self.orbital_elements else None,
        "close_approaches": [...],
        "sources_used": self.sources_used,
        ...
        # physical_properties NOT serialized
        # fetched_at NOT serialized
    }
```

**`NEOData.from_dict()`** (line 322): also does not reconstruct `physical_properties`
or `fetched_at`.

**Flow**:
1. First call: `DataFetcher._fetch_from_source_internal()` builds `NEOData` with
   `physical_properties` set from `_physical` sub-dict (Phase 4D.3). Sets `fetched_at`.
2. Serialized to cache via `_serialize_neo_data()` → `to_dict()` → loses both fields.
3. Second call: cache hit → `_deserialize_neo_data()` → `from_dict()` → `physical_properties=None`, `fetched_at=None`.
4. `detect_neo()`: `if neo.physical_properties:` → `False` → `physical_data=None` → detector
   skips physical evidence → `evidence_count` drops from 3 to 1.
5. `data_freshness` falls back to `datetime.now()` — not the original fetch time.

This is a **silent regression**: the first call is correct; every subsequent
(cached) call is worse quality with no indication to the user.

#### Consequence Chain

1. Cold cache: `evidence_count: 3` (orbital + physical + trajectory if available)
2. Warm cache: `evidence_count: 1` (orbital only)
3. `data_freshness` from cache hit shows `datetime.now()` — identical to pre-Phase 9
4. Phase 9 fixes for UE-011 and UE-015 effectively disabled after the first request

#### Acceptance Criteria

- `NEOData.to_dict()` includes `physical_properties` as a serialized dict
- `NEOData.from_dict()` reconstructs `physical_properties: Optional[PhysicalProperties]`
- `NEOData.to_dict()` includes `fetched_at` as ISO string
- `NEOData.from_dict()` reconstructs `fetched_at: Optional[datetime]`
- Two consecutive calls to `GET /detect?designation=Apophis` return the same
  `evidence_count`

#### Proposed Fix

**`aneos_core/data/models.py` — `to_dict()`**: add two serialization lines:

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        ...
        "physical_properties": {
            "diameter_km": self.physical_properties.diameter_km,
            "albedo": self.physical_properties.albedo,
            "rotation_period_hours": self.physical_properties.rotation_period_hours,
            "spectral_type": self.physical_properties.spectral_type,
            "absolute_magnitude_h": self.physical_properties.absolute_magnitude_h,
        } if self.physical_properties else None,
        "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        ...
    }
```

**`from_dict()`**: reconstruct both:

```python
pp_data = data.get("physical_properties")
physical_properties = None
if pp_data:
    from .models import PhysicalProperties
    physical_properties = PhysicalProperties(**{k: v for k, v in pp_data.items() if v is not None})

return cls(
    ...
    physical_properties=physical_properties,
    fetched_at=_ensure_utc(parse_datetime(data.get("fetched_at"))),
    ...
)
```

---

## P2 Gaps

---

### UE-021: `DetectionResponse` Omits Statistical Completeness Fields

**Persona**: P1 (Anomaly Researcher)

#### User Expectation

> "I have `evidence_count: 3` and three `confidence_interval` ranges. But what is
> the combined p-value across all three evidence sources? And what is the expected
> false discovery rate at this sigma level? I need these for the Methods section of
> my paper."

#### Root Cause

**`aneos_core/detection/validated_sigma5_artificial_neo_detector.py` — `Sigma5DetectionResult`**:

```python
@dataclass
class Sigma5DetectionResult:
    is_artificial: bool
    sigma_confidence: float
    bayesian_probability: float
    evidence_sources: List[EvidenceSource]
    combined_p_value: float        # ← Fisher's combined probability across all evidence
    false_discovery_rate: float   # ← Expected FDR at this threshold
    validation_metrics: Optional[ValidationResult]  # ← Calibration data
    analysis_metadata: Dict[str, Any]  # ← Version, timestamp, method, population ref
```

**`aneos_api/schemas/detection.py` — `DetectionResponse`**: exposes the first 4 fields
(`is_artificial`, `sigma_confidence`, `bayesian_probability`, `evidence_sources`) but
drops the last 4.

`combined_p_value` (Fisher's method across all evidence) and `false_discovery_rate`
(calibrated from the 9-object ground truth dataset) are directly computable and
already computed. They are dropped at the schema boundary.

#### Acceptance Criteria

- `DetectionResponse` includes `combined_p_value: float` and `false_discovery_rate: float`
- `analysis_metadata: Dict[str, Any]` includes at minimum `detector_version`,
  `statistical_method`, `population_parameters_source`
- All three appear in OpenAPI `components/schemas/DetectionResponse`
- `POST /detect` (raw) and `GET /detect` (by designation) both populate these fields

#### Proposed Fix

Add to `DetectionResponse` in `schemas/detection.py`:

```python
combined_p_value: Optional[float] = None      # Fisher's combined probability
false_discovery_rate: Optional[float] = None  # Expected FDR at current threshold
analysis_metadata: Dict[str, Any] = {}        # Detector version, method, population refs
```

Populate in `detect_neo()` and `detect_neo_raw()`:

```python
return DetectionResponse(
    ...
    combined_p_value=result.combined_p_value,
    false_discovery_rate=result.false_discovery_rate,
    analysis_metadata=result.analysis_metadata,
)
```

---

### UE-022: Batch Results Lack Evidence Detail

**Persona**: P3 (System Integrator)

#### User Expectation

> "My batch of 20 objects completed. Results show `'is_artificial': False,
> 'sigma_confidence': 2.3` for each object. But the single-object `/detect`
> endpoint shows evidence sources, sigma tier, and interpretation. Batch gives me
> a summary table — I can't see which evidence modules fired or why."

#### Root Cause

**`aneos_api/endpoints/analysis.py` — `_run_batch_detection()` result dict**:

```python
store['results'].append({
    'designation': designation,
    'status': 'success',
    'is_artificial': result.is_artificial,
    'sigma_confidence': result.sigma_confidence,
    'artificial_probability': result.bayesian_probability,
    'classification': 'ARTIFICIAL' if result.is_artificial else 'NATURAL',
    # ← no sigma_tier
    # ← no evidence_sources
    # ← no evidence_count
    # ← no interpretation
    # ← no combined_p_value
})
```

A user submitting a batch cannot distinguish a `σ=2.1` INTERESTING from a `σ=1.1`
NOTABLE result, nor understand which evidence drove either classification.

#### Acceptance Criteria

- Batch result entries include: `sigma_tier`, `evidence_count`, `interpretation`,
  `evidence_sources` (summary list), `combined_p_value`
- Schema for batch result entries defined in `schemas/` (e.g., `BatchResultEntry`)
- `GET /batch/{id}/status` response is typed in OpenAPI spec
- Existing `total`, `completed`, `failed`, `status` fields unchanged

#### Proposed Fix

In `_run_batch_detection()` result dict:

```python
tier = _sigma_tier(result.sigma_confidence)
top_type = result.evidence_sources[0].evidence_type.value if result.evidence_sources else "orbital_dynamics"
store['results'].append({
    'designation': designation,
    'status': 'success',
    'is_artificial': result.is_artificial,
    'sigma_confidence': result.sigma_confidence,
    'sigma_tier': tier,
    'artificial_probability': result.bayesian_probability,
    'combined_p_value': result.combined_p_value,
    'false_discovery_rate': result.false_discovery_rate,
    'classification': 'ARTIFICIAL' if result.is_artificial else 'NATURAL',
    'evidence_count': len(result.evidence_sources),
    'evidence_sources': [
        {'type': e.evidence_type.value, 'anomaly_score': e.anomaly_score,
         'p_value': e.p_value, 'quality_score': e.quality_score}
        for e in result.evidence_sources
    ],
    'interpretation': _build_interpretation(
        designation, result.sigma_confidence, tier, top_type, result.bayesian_probability
    ),
})
```

---

### UE-023: `ImpactResponse` Exposes 7 of 25+ `ImpactProbability` Fields

**Persona**: P2 (Planetary Defense Analyst)

#### User Expectation

> "I got `collision_probability: 0.00031` and `impact_energy_mt: 450.3`. But where
> are the uncertainty bounds? Where is the keyhole analysis — I need to know if
> there are resonant return corridors. Where is the damage radius? The CLI menu
> shows all of this."

#### Root Cause

**`aneos_core/analysis/impact_probability.py` — `ImpactProbability` dataclass**: 25+
fields, including:

```python
probability_uncertainty: Tuple[float, float]  # (lower_bound, upper_bound)
calculation_confidence: float                  # 0-1
data_arc_years: float                          # Observation arc
damage_radius_km: Optional[float]             # Damage radius
keyhole_passages: List[Dict[str, Any]]        # Resonant return corridors
primary_risk_factors: List[str]               # Scientific rationale
impact_probability_by_decade: Dict[str, float] # Temporal evolution
peak_risk_period: Optional[Tuple[int, int]]   # Peak danger window
comparative_risk: str                          # Plain-English comparison
```

**`aneos_api/schemas/impact.py` — `ImpactResponse`**: exposes only 7 fields:
`designation`, `collision_probability`, `moon_collision_probability`, `moon_earth_ratio`,
`impact_energy_mt`, `crater_diameter_km`, `risk_level`, `time_to_impact_years`.

Missing fields that directly answer planetary defense planning questions:
- Uncertainty bounds (required for risk assessment under incomplete data)
- Keyhole passages (required for deflection mission targeting)
- Damage radius (required for evacuation zone planning)
- Temporal evolution (required for monitoring priority)

#### Acceptance Criteria

- `ImpactResponse` adds: `probability_uncertainty`, `calculation_confidence`,
  `damage_radius_km`, `keyhole_passages`, `primary_risk_factors`,
  `impact_probability_by_decade`, `peak_risk_period`, `comparative_risk`
- `GET /impact?designation=Apophis` → `damage_radius_km` non-null,
  `keyhole_passages` is a list (may be empty)
- New fields are documented in OpenAPI spec

#### Proposed Fix

Extend `ImpactResponse` in `schemas/impact.py`:

```python
class ImpactResponse(BaseModel):
    designation: str
    collision_probability: float
    probability_uncertainty: List[float] = []   # [lower_bound, upper_bound]
    calculation_confidence: Optional[float] = None
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    damage_radius_km: Optional[float] = None
    risk_level: str
    comparative_risk: Optional[str] = None
    time_to_impact_years: Optional[float] = None
    peak_risk_period: Optional[List[int]] = None     # [start_year, end_year]
    keyhole_passages: List[Dict[str, Any]] = []
    primary_risk_factors: List[str] = []
    impact_probability_by_decade: Dict[str, float] = {}
```

Populate in `impact_neo()`:

```python
return ImpactResponse(
    ...
    probability_uncertainty=list(result.probability_uncertainty),
    calculation_confidence=result.calculation_confidence,
    damage_radius_km=result.damage_radius_km,
    comparative_risk=result.comparative_risk,
    peak_risk_period=list(result.peak_risk_period) if result.peak_risk_period else None,
    keyhole_passages=result.keyhole_passages,
    primary_risk_factors=result.primary_risk_factors,
    impact_probability_by_decade=result.impact_probability_by_decade,
)
```

---

## P3 Gaps

---

### UE-025: `OrbitalInput` Missing `orbital_history` Field

**Persona**: P3 (System Integrator)

#### User Expectation

> "I called `GET /history?designation=2020SO` and got 10 years of Keplerian elements.
> Now I want to run detection with that history to trigger course correction analysis.
> I can't — `POST /detect` body has no field for orbital history."

#### Root Cause

**`aneos_api/schemas/detection.py` — `OrbitalInput`** (Phase 9):

```python
class OrbitalInput(BaseModel):
    a: float
    e: float
    i: float
    designation: Optional[str] = None
    diameter_km: Optional[float] = None
    albedo: Optional[float] = None
    # No orbital_history field
```

`analyze_neo_validated()` accepts `orbital_history: List[Dict[str, Any]]`. The output
of `GET /history` is exactly the right format. But there is no field in `OrbitalInput`
to pass it through.

The user must chain `GET /history` → `POST /detect` to get the full evidence picture,
but the chain is broken at the `POST /detect` input boundary.

#### Acceptance Criteria

- `OrbitalInput` adds `orbital_history: Optional[List[Dict[str, Any]]] = None`
- `detect_neo_raw()` passes `request.orbital_history` as 3rd argument to
  `analyze_neo_validated()`
- OpenAPI spec shows `orbital_history` as optional array in `OrbitalInput`

#### Proposed Fix

Add to `OrbitalInput`:

```python
class OrbitalInput(BaseModel):
    a: float
    e: float
    i: float
    designation: Optional[str] = None
    diameter_km: Optional[float] = None
    albedo: Optional[float] = None
    orbital_history: Optional[List[Dict[str, Any]]] = None  # ← add
```

In `detect_neo_raw()`:

```python
result = detector.analyze_neo_validated(
    orbital_dict,
    physical_data=physical_data,
    orbital_history=request.orbital_history,   # ← add
)
```

---

### UE-026: No Freshness-Aware Cache Control on `GET /detect`

**Persona**: P3 (System Integrator)

#### User Expectation

> "`data_freshness` tells me the data is 6 days old. SBDB updates orbital elements
> daily for recently observed objects. I want to force a refresh. There's no
> `?force_refresh=true` query parameter on `/detect`."

#### Root Cause

`GET /detect?designation=X` always calls `fetcher.fetch_neo_data(designation)` with
`force_refresh=False`. The `DataFetcher.fetch_neo_data()` signature accepts
`force_refresh: bool = False`, but the endpoint never exposes it.

`GET /analyze` (the legacy pipeline endpoint) exposes `force_refresh` in the
`AnalysisRequest` model — but the new `/detect` endpoint has no equivalent.

#### Acceptance Criteria

- `GET /detect?designation=X&force_refresh=true` bypasses cache and re-fetches
  from SBDB
- `data_freshness` in the response reflects the new fetch timestamp
- `force_refresh` defaults to `false` — no behaviour change for existing callers

#### Proposed Fix

Add optional query param to `detect_neo()`:

```python
@router.get("/detect", ...)
async def detect_neo(
    designation: str = Query(...),
    force_refresh: bool = Query(False, description="Bypass cache and re-fetch from source"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    ...
    neo = fetcher.fetch_neo_data(designation, force_refresh=force_refresh)
```

---

## Cross-Cutting Findings

### Finding C-8: Three of Five Detector Inputs Are Dark

After Phase 9, the detector can accept 5 input channels:

| # | Channel | API Status |
|---|---------|------------|
| 1 | `orbital_elements` | ✅ always passed |
| 2 | `physical_data` | ✅ Phase 9 (but lost on cache hit — UE-024) |
| 3 | `orbital_history` | ❌ never fetched, never passed (UE-019) |
| 4 | `close_approach_history` | ❌ wire exists, source empty (UE-020) |
| 5 | `observation_data` | ❌ no source, no API field (beyond Phase 10 scope) |

The maximum evidence count reachable through the API right now is 2 (orbital + physical,
first request only). With Phase 10 fixes, channels 2, 3, and 4 all fire: maximum
evidence count rises to 4, matching the menu path.

### Finding C-9: Cache Is the Single Point of Quality Regression

Every quality improvement that depends on `NEOData.physical_properties` regresses
to zero after the first request due to the `to_dict()` omission (UE-024). This affects:

- UE-011 (Phase 9): physical evidence → drops to 0
- UE-013 (Phase 9): impact energy/crater → returns null again
- UE-015 (Phase 9): real `fetched_at` → reverts to `datetime.now()`

UE-024 is therefore the most urgent fix in Phase 10: it undoes three Phase 9 fixes
for all cached (warm) requests.

### Finding C-10: Statistical Completeness Gap Remains

Phases 8 and 9 surfaced what the detector computes (evidence, scores, tier). They did
not surface how reliable those computations are (combined p-value, FDR, calibration
metadata). A P1 researcher publishing results needs the latter for a Methods section.
`Sigma5DetectionResult` already computes these; the API discards them (UE-021).

---

## Outcome Quality Re-Assessment (Post Phase 9)

| Expectation | Computed Internally | Reaches API (cold cache) | Reaches API (warm cache) | Phase 10 Target |
|-------------|--------------------|--------------------------|--------------------------| ---------------|
| Course correction evidence | ✅ if orbital_history supplied | ❌ never fetched | ❌ never fetched | UE-019 |
| Close approach trajectory evidence | ✅ if data available | ❌ source empty | ❌ source empty | UE-020 |
| Physical evidence | ✅ SBDB | ✅ Phase 9 | ❌ cache drops pp | UE-024 |
| Real data_freshness | ✅ fetched_at set | ✅ Phase 9 | ❌ cache drops | UE-024 |
| Combined p-value | ✅ Fisher's method | ❌ not in schema | ❌ not in schema | UE-021 |
| Batch evidence detail | ✅ result has it | ❌ summary only | ❌ summary only | UE-022 |
| Impact uncertainty bounds | ✅ computed | ❌ not in schema | ❌ not in schema | UE-023 |
| Keyhole analysis | ✅ computed | ❌ not in schema | ❌ not in schema | UE-023 |
| POST /detect with history | ✅ accepts it | ❌ no field | ❌ no field | UE-025 |
| Cache-bypass control | ✅ DataFetcher | ❌ no query param | — | UE-026 |

---

## Phase 10 Sub-Phase Sequence

```
10A (data sources + cache)  →  10B (schema completeness)  →  10C (input surface)
```

`pytest tests/ -m "not network" -q` (59 pass, 0 fail) must hold after every sub-phase.

---

### 10A — Data Sources + Cache Integrity (UE-024, UE-020, UE-019)

**Recommended order**: UE-024 first (cache fix unblocks everything else), then
UE-020 (CAD API fetch), then UE-019 (Horizons history in detect path).

**UE-024 fix** (highest ROI — fixes three Phase 9 regressions):
1. Extend `NEOData.to_dict()` to include `physical_properties` dict + `fetched_at` ISO string
2. Extend `NEOData.from_dict()` to reconstruct `physical_properties` + `fetched_at`
3. Verify: two consecutive `GET /detect?designation=Apophis` return same `evidence_count`

**UE-020 fix**:
1. Add `_fetch_close_approaches(designation)` to `DataFetcher` using `cad.api`
2. Call after primary source fetch in `_fetch_from_all_sources()`
3. Populate `NEOData.close_approaches` list
4. Include `close_approaches` in `to_dict()`/`from_dict()` round-trip (already done)

**UE-019 fix**:
1. In `detect_neo()`: call `HorizonsSource.fetch_orbital_history()` after `fetch_neo_data()`
2. Pass result as `orbital_history=` to `analyze_neo_validated()`
3. Handle network failure gracefully (log + continue with `orbital_history=None`)

**Files**: `aneos_core/data/models.py`, `aneos_core/data/fetcher.py`,
`aneos_api/endpoints/analysis.py`

**Verify**:
```bash
pytest tests/ -m "not network" -q    # 59 pass, 0 fail
```
Manual: `GET /detect?designation=Apophis` (cold) vs (warm) → same evidence_count

---

### 10B — Schema Completeness (UE-021, UE-022, UE-023)

**UE-021**: Add `combined_p_value`, `false_discovery_rate`, `analysis_metadata` to
`DetectionResponse`. Populate in `detect_neo()` and `detect_neo_raw()`.

**UE-022**: Enrich batch result dict with `sigma_tier`, `evidence_count`,
`evidence_sources`, `interpretation`, `combined_p_value`. Consider a typed
`BatchResultEntry` schema.

**UE-023**: Extend `ImpactResponse` with `probability_uncertainty`, `calculation_confidence`,
`damage_radius_km`, `keyhole_passages`, `primary_risk_factors`,
`impact_probability_by_decade`, `peak_risk_period`, `comparative_risk`.
Populate in `impact_neo()`.

**Files**: `aneos_api/schemas/detection.py`, `aneos_api/schemas/impact.py`,
`aneos_api/endpoints/analysis.py`

**Verify**:
```bash
pytest tests/ -m "not network" -q
make spec
grep "combined_p_value\|keyhole_passages\|damage_radius" docs/api/openapi.json
```

---

### 10C — Input Surface (UE-025, UE-026)

**UE-025**: Add `orbital_history: Optional[List[Dict[str, Any]]] = None` to
`OrbitalInput`. Pass to `detect_neo_raw()` → `analyze_neo_validated()`.

**UE-026**: Add `force_refresh: bool = Query(False)` to `detect_neo()`. Pass to
`fetcher.fetch_neo_data(designation, force_refresh=force_refresh)`.

**Files**: `aneos_api/schemas/detection.py`, `aneos_api/endpoints/analysis.py`

---

## Gap Summary Table

| ID | Gap | Priority | Sub-Phase |
|----|-----|----------|-----------|
| UE-024 | Cache drops `physical_properties` + `fetched_at` → Phase 9 regressions | P1 | 10A |
| UE-020 | `close_approaches` always empty — no CAD API source | P1 | 10A |
| UE-019 | `orbital_history` never fetched → course correction evidence never fires | P1 | 10A |
| UE-021 | `DetectionResponse` omits combined p-value, FDR, metadata | P2 | 10B |
| UE-022 | Batch results lack evidence detail | P2 | 10B |
| UE-023 | `ImpactResponse` exposes 7 of 25+ impact fields | P2 | 10B |
| UE-025 | `OrbitalInput` has no `orbital_history` field | P3 | 10C |
| UE-026 | No `force_refresh` query param on `GET /detect` | P3 | 10C |

**Deferred (no Phase 10 work)**:
- G-015: ML classifier activation (independent work stream)
- G-023: `aneos_menu.py` decomposition (separate refactor effort)
- UE-005/observation_data: `non_gravitational_accel` / `brightness_variations` —
  no automated source; requires telescope-specific integration beyond aNEOS scope

---

## Critical Ordering Note

**UE-024 must be fixed before UE-019 and UE-020 are useful.** Without the cache
fix, `physical_properties` is lost on the second request. Adding orbital history
and CAD data as additional evidence while the cache strips `physical_properties`
creates a different kind of non-determinism: first call = 4 sources, second call
= 2 sources (history + orbital, no physical). Fix the cache first.

---

*This document is the executable input for Phase 10 planning. UE-024 is the
highest-ROI fix: one two-method change in `models.py` restores three Phase 9
fixes for all warm-cache requests. Implement 10A (UE-024 → UE-020 → UE-019)
before 10B or 10C.*
