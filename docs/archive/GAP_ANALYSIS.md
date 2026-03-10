# aNEOS Gap Analysis

**Version**: 6.0
**Date**: 2026-03-07
**Supersedes**: v5.0 (2026-03-07)
**Derived from**: Phase 7 implementation audit, live code inspection, test suite state
(59 pass / 0 fail after Phase 7). All 9 non-deferred Phase 7 gaps closed.

---

## Gap Severity Legend

| Symbol | Severity | Definition |
|--------|----------|------------|
| P0 | Critical | Blocks the core mission; no workaround exists |
| P1 | High | Produces incorrect or misleading results; deployment-blocking |
| P2 | Medium | Maintainability or quality risk; does not affect correctness today |
| P3 | Low | Future-state or polish; no immediate impact |

---

## Closed Gaps (through Phase 6)

| ID | Title | Closed | Resolution |
|----|-------|--------|------------|
| G-001 | No ground truth dataset | Phase 2+3 | GroundTruthDatasetBuilder + GroundTruthValidator; sens=1.00, spec=1.00 |
| G-002 | Physical indicator category missing | Phase 2 | `indicators/physical.py` — 3 indicators wired into pipeline |
| G-003 | Pipeline bypasses canonical detector | Phase 3 | `automatic_review_pipeline.py` uses `DetectionManager(DetectorType.AUTO)` |
| G-004 | Startup soft-degrades silently | Phase 4A | Preflight gate + `sys.exit(1)` on decline |
| G-005 | Docker deployment broken | Phase 1 | `init.sql`, `Makefile bootstrap`, `docker-compose.override.yml.example` |
| G-006 | Monitoring depends on deferred ML | Phase 1 | ML import guarded with `_HAS_ML_ALERTS` |
| G-007 | README overstates production readiness | Phase 2+3 | Updated to Pre-Production; measured validation results stated |
| G-008 | Seven detector files, no archive | Phase 4C | `detection/_archive/` created; 5 superseded detectors moved |
| G-009 | Dual scoring + EMERGENCY suppressions (scoring) | Phase 4C | EMERGENCY → `logger.debug` in `advanced_scoring.py`; ADR-008 |
| G-010 | No CI/CD pipeline | Phase 1 | `.github/workflows/ci.yml` — pytest + docker-build |
| G-013 | Test harness inside production module | Phase 4C | Moved to `tests/detection/test_artificial_neo_suite.py` |
| G-014 | No maintained OpenAPI spec | Phase 4D | `docs/api/openapi.json`; `make spec`; CI drift check |
| G-016 | Redis integration unverified | Phase 4E | Redis `PING` in `preflight_check()` |
| G-017 | CacheManager pickle security risk | Phase 4E | JSON-only; `import pickle` removed from `cache.py` |
| G-018 | Chunk boundary overlap unvalidated | Phase 4E | `_merge_chunks()` added; `tests/test_chunk_boundaries.py` passes |
| G-020 | Bayesian posterior ceiling undocumented | Phase 4A | README + sigma5_success_criteria + menu interpretation line |
| G-022 | EMERGENCY suppressions mask scoring | Phase 4C | Merged with G-009 |
| G-024 | Temporal analysis limited to single epoch | Phase 4B | `_generate_orbital_history()` calls Horizons ELEMENTS API |
| G-025 | AI annotator may overstate confidence | Phase 4E | `_INTERPRETATION_DISCLAIMER` added |
| G-026 | Pickle in ML persistence layer | Phase 5 | joblib + JSON; `import pickle` removed at module level |
| G-027 | CI spec drift check workflow-hostile | Phase 5 | `sort_keys=True`; CI message; `CONTRIBUTING.md` |
| G-028 | EMERGENCY suppressions in validation/analysis | Phase 6B | All 13 removed across 6 files; failure paths restored to `logger.warning` |
| G-029 | Hardcoded dev path + archive import | Phase 6B | `sys.path.append` removed; `ValidatedSigma5ArtificialNEODetector` used instead |
| G-030 | `APIConfig.neodys_url` phantom URL | Phase 6B | `settings.py` → `~neodys2/epoch/`; `health.py` → probes `99942.eq0` |
| G-021 | NEODyS URL inconsistency | Phase 6B | `NEODySSource.__init__` now reads `config.neodys_url` |
| G-011 | OrbitalElements conflates domain models | Phase 6C | `diameter/albedo/rot_per/spectral_type` fields removed from dataclass |
| G-031 | API auth mock user DB | Phase 6D | `_assert_auth_configured()` startup guard added; halts non-dev starts |
| G-032 | Export endpoint returns stub data | Phase 6D | Endpoint marked `deprecated=True` in OpenAPI router |
| G-019 | Hyperbolic elements rejected by OrbitalElements | Phase 7B | `_validate()` accepts e≥1 and a<0 when hyperbolic; 4th parser test added |
| G-033 | Network tests broken by Phase 6 config change | Phase 7A | Tests use `APIConfig()` instead of `{}`; both sources instantiate cleanly |
| G-034 | SBDB dead writes to removed OrbitalElements fields | Phase 7A | Dead `orbital_data[...]` dual-writes removed; `_physical` is sole target |
| G-035 | MPC albedo silently lost | Phase 7A | `fetch_orbital_elements()` populates `_physical` sub-dict with albedo |
| G-036 | CI runs network tests — flaky pipeline | Phase 7A | `-m "not network"` added to CI pytest invocation |
| G-037 | `ml/models.py` uses `logger` before definition | Phase 7A | `logger` definition moved above all `try/except ImportError` blocks |
| G-012 | Schemas not in OpenAPI spec; field mismatch | Phase 7C | `DetectionResponse.sigma_confidence` wired; `GET /detect` endpoint added; spec regenerated (4 hits) |
| G-039 | `neodys._make_request()` ignores `self.base_url` | Phase 7C | `neodys_rest_url` added to `APIConfig`; `_make_request` uses `self.rest_url` |

---

## P1 — High Severity Gaps

### G-019 (RESIDUAL — deepened): Hyperbolic Elements Cannot Be Stored in `OrbitalElements`

**Area**: Scientific Validity / Ground Truth Corpus
**Status**: Phase 6A closed the regex parse issue (`-?` prefix added). Newly confirmed:
the fix is **incomplete end-to-end** — parsed negative `A` values are rejected downstream.

**Root cause**:

`OrbitalElements._validate()` enforces two constraints that hyperbolic trajectories violate:

```python
# aneos_core/data/models.py:70-80 (Phase 6 state)
if self.eccentricity is not None:
    if not (0 <= self.eccentricity < 1):
        errors.append(...)       # Rejects e > 1 — hyperbolic

if self.semi_major_axis is not None:
    if self.semi_major_axis <= 0:
        errors.append(...)       # Rejects a < 0 — hyperbolic
```

**Live verification** (post-Phase 6):
```
$ python -c "from aneos_core.data.models import OrbitalElements; \
     OrbitalElements(semi_major_axis=-8.79, eccentricity=3.73, inclination=35.4, ...)"

ValueError: Orbital elements validation failed:
  Eccentricity 3.73 outside valid range [0, 1);
  Semi-major axis -8.79 must be positive
```

**Consequence chain**:
1. `GroundTruthDatasetBuilder._fetch_from_horizons()` returns raw dicts → not affected (uses plain dicts, not `OrbitalElements`).
2. `DataFetcher._fetch_from_source()` calls `OrbitalElements(**oe_kwargs)` — would raise `ValueError` if a hyperbolic object's elements were fetched via `HorizonsSource`.
3. The 3rd unit test (`test_parser_handles_negative_semi_major_axis`) passes only because it tests the raw dict output of `_fetch_from_horizons`, not `OrbitalElements` creation.
4. G-019 claimed as "closed" in Phase 6A is therefore only half-fixed — the parse works, the domain model still rejects the result.

**Fix**:
1. Update `_validate()` to accept `semi_major_axis < 0` when `eccentricity >= 1` (hyperbolic gate).
2. Update `_validate()` to accept `eccentricity >= 1` (hyperbolic/parabolic orbits; reject only negative e).
3. Update `is_complete()` — for hyperbolic objects, `semi_major_axis < 0` is a valid complete state.
4. Add a 4th parser test: construct an `OrbitalElements` from the hyperbolic dict (not just the raw dict).
5. Add `orbit_type: str = "elliptical"` field or derive it from `eccentricity` — helps consumers
   distinguish cases without `isinstance` checks.

**ADR/DDD ref**: ADR-038, ADR-006, DDD BC-2

---

### G-033 (NEW — Phase 6 regression): Network Tests Broken by `config.neodys_url` Change

**Area**: Test Integrity / Data Acquisition
**Status**: NEW — introduced by Phase 6B.3.

`test_data_sources_network.py` instantiates both data sources with empty dict config:

```python
# tests/test_data_sources_network.py:9-10
def test_neodys_apophis():
    from aneos_core.data.sources.neodys import NEODySSource
    r = NEODySSource({})._make_request("99942")   # ← empty dict, not APIConfig
```

Phase 6B.3 changed `NEODySSource.__init__` from `self.base_url = _NEODYS_EPOCH_BASE` (constant)
to `self.base_url = config.neodys_url` (attribute access). With `config = {}` (empty dict),
`config.neodys_url` raises `AttributeError` immediately:

```
AttributeError: 'dict' object has no attribute 'neodys_url'
```

**Live verification**:
```
$ python -c "from aneos_core.data.sources.neodys import NEODySSource; NEODySSource({})"
AttributeError: 'dict' object has no attribute 'neodys_url'
```

The tests are marked `@pytest.mark.network` and excluded from the normal test run
(`-m "not network"`), so the regression is masked in CI. But the tests are now broken
as written, and any attempt to run them will fail before they reach the network.

**Fix**:
1. Update `test_data_sources_network.py` to pass `APIConfig()` instead of `{}`:
   ```python
   from aneos_core.config.settings import APIConfig
   r = NEODySSource(APIConfig())._make_request("99942")
   r = MPCSource(APIConfig())._make_request("99942")
   ```
2. Verify that `MPCSource` has the same pattern issue (it does not — `MPCSource.__init__`
   uses `config.mpc_url` but also does `if config is None: config = APIConfig()` — the `{}`
   would fail attribute access too).

**ADR/DDD ref**: ADR-001

---

## P2 — Medium Severity Gaps

### G-034 (NEW — Phase 6 regression): SBDB Source Has Dead Physical Writes to Removed Fields

**Area**: Data Acquisition / Maintainability
**Status**: NEW — introduced by Phase 6C.1 removing `OrbitalElements` physical fields.

`aneos_core/data/sources/sbdb.py` lines 96-109 still write `diameter`, `albedo`,
`rot_per`, `spectral_type` into `orbital_data` (the dict later filtered by
`fetcher.py` → `OrbitalElements`):

```python
# sbdb.py:96-109 (current state)
physical_data: Dict[str, Any] = {}
for float_field in ("diameter", "albedo", "rot_per"):
    if float_field in phys_par:
        val = float(phys_par[float_field])
        orbital_data[float_field] = val      # ← DEAD: 'albedo' not in OrbitalElements
        physical_data[float_field] = val     # ← correct: reaches _physical
if "spec_T" in phys_par:
    orbital_data["spectral_type"] = str(phys_par["spec_T"])  # ← DEAD
    physical_data["spectral_type"] = str(phys_par["spec_T"]) # ← correct
```

`fetcher.py:204` filters with `valid_fields = OrbitalElements.__dataclass_fields__`. Since
`diameter`, `albedo`, `rot_per`, `spectral_type` are no longer in `__dataclass_fields__`,
the `orbital_data[float_field] = val` lines are dead — their values are silently discarded.
The data correctly reaches `physical_properties` via `_physical`, so no functional loss.
But the comment on line 96 (`# Write physical fields to both orbital_data (backward compat)`)
is now misleading — there is no backward compat path.

**Impact**: Code confusion and potential future writer error. A developer may look at this
and assume the dual-write is necessary and replicate the pattern in other sources.

**Fix**:
1. Remove `orbital_data[float_field] = val` and `orbital_data["spectral_type"] = ...` lines.
2. Retain only the `physical_data[...]` writes (correct path to `_physical`).
3. Update comment to reflect single source of truth: `"# Build physical sub-dict for PhysicalProperties construction"`.

**ADR/DDD ref**: ADR-006, DDD BC-2

---

### G-035 (NEW — Phase 6 regression): MPC Source Albedo Silently Lost

**Area**: Data Acquisition / Domain Model
**Status**: NEW — introduced by Phase 6C.1 removing `OrbitalElements.albedo`.

`aneos_core/data/sources/mpc.py:fetch_orbital_elements()` returns:
```python
return {
    "semi_major_axis": ...,
    "eccentricity": ...,
    ...
    "albedo": _safe_float(row.get("albedo")),   # line 75
}
```

`fetcher.py:204-205` filters this dict to only `OrbitalElements.__dataclass_fields__` keys.
`albedo` is no longer in `__dataclass_fields__` (removed Phase 6C.1). The key is
silently discarded. Unlike SBDB, `mpc.py` does not populate a `_physical` sub-dict, so
the albedo data from MPC **never reaches `PhysicalProperties`** and is completely lost.

**Verification**:
```
OrbitalElements.__dataclass_fields__ does not contain 'albedo'
MPC returns 'albedo' → filtered out by fetcher → not in _physical → lost
```

**Impact**: Objects whose albedo is only available from MPC (not in SBDB `phys_par`)
will have `physical_properties.albedo = None`, causing `AlbedoAnomalyIndicator` to return
`confidence=0.0` (no data) even when MPC holds the albedo value. This silently degrades
physical anomaly scoring for objects where SBDB has no physical parameters.

**Fix**:
1. Update `mpc.py:fetch_orbital_elements()` to populate a `_physical` sub-dict:
   ```python
   result = {"semi_major_axis": ..., ...}  # orbital fields only
   physical = {}
   if row.get("albedo"):
       physical["albedo"] = _safe_float(row.get("albedo"))
   if physical:
       result["_physical"] = physical
   return result
   ```
2. Remove `"albedo"` from the top-level return dict (it no longer maps to `OrbitalElements`).
3. Add a test asserting that `DataFetcher` returns `physical_properties.albedo` when MPC
   is the only source and SBDB returns nothing.

**ADR/DDD ref**: ADR-006, DDD BC-1

---

### G-036 (NEW): CI Runs Network Tests — Flaky Pipeline

**Area**: CI/CD / Test Reliability
**Status**: NEW (Phase 6 audit).

The CI test step in `.github/workflows/ci.yml:24` runs:

```yaml
- name: Run tests
  run: python -m pytest tests/ -v --tb=short
```

No `-m "not network"` exclusion. All `@pytest.mark.network` tests in
`tests/test_data_sources_network.py` are collected and run in CI on `ubuntu-latest`,
which makes outbound HTTP calls to `newton.spacedys.com` and `minorplanetcenter.net`.

**Consequences**:
1. CI fails when external APIs are unavailable (planned maintenance, temporary outage).
2. CI runtime is unpredictable — network calls add 3–15 seconds per test.
3. The existing G-033 bug (`NEODySSource({})`) would cause these tests to fail on
   `AttributeError` even before reaching the network.
4. If additional network tests are added, the CI becomes progressively more flaky.

**Fix**: Add `-m "not network"` to the CI pytest invocation:
```yaml
run: python -m pytest tests/ -m "not network" -v --tb=short
```
Network tests should only run in a dedicated integration test step gated on a manual trigger
or a `[ci-network]` commit tag, with explicit caveats about external API availability.

**ADR/DDD ref**: ADR-010 (CI/CD)

---

### G-037 (NEW): `ml/models.py` Uses `logger` Before Definition

**Area**: Engineering Quality / Error Handling
**Status**: NEW (Phase 6 audit — pre-existing but not previously caught).

```python
# aneos_core/ml/models.py:29-44
try:
    from sklearn.ensemble import IsolationForest, ...
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available, ...")   # ← line 31: logger undefined!

...

logger = logging.getLogger(__name__)   # ← line 44: actual definition
```

If scikit-learn is not installed, the `except ImportError` branch executes `logger.warning`
before `logger` is defined at module scope. This raises `NameError: name 'logger' is not
defined` and prevents the module from loading at all — a harder failure than the intended
graceful degradation.

**Verified**: scikit-learn IS installed in the current dev environment, so this has never
triggered. In a minimal deployment without sklearn, all ML functionality would fail to import.

**Fix**: Move `logger = logging.getLogger(__name__)` to immediately after the `import logging`
statement at the top of the file (before all `try/except ImportError` blocks). This is the
standard Python convention.

**ADR/DDD ref**: ADR-033

---

### G-012 (RESIDUAL — deepened): Schema Wiring Incomplete — Schemas Not in OpenAPI Spec

**Area**: API / Maintainability
**Status**: Partially closed in Phase 5 and 6, but the core criterion is still unmet.

Phase 6C.2 added `import aneos_api.schemas  # noqa: F401` to `app.py` and created a
`schemas/__init__.py` exporting all types. The intent was to make the schemas appear in
the generated OpenAPI spec.

**This does not work as intended.** FastAPI/Pydantic only includes models in the OpenAPI
`components/schemas` section if they are:
- Used as `response_model` in a route decorator, OR
- Referenced as a field type in a model that IS used as `response_model`, OR
- Used in a request body model

Simply importing a Pydantic model at module load time does NOT register it in the OpenAPI
spec. Running `make spec` after the Phase 6 change produces an `openapi.json` whose
`components/schemas` section does not contain `DetectionResponse`, `HealthResponse`,
`CheckResult`, or `ImpactResponse`.

**Evidence**:
```
# aneos_api/schemas/detection.py — DetectionResponse defined
# aneos_api/schemas/health.py    — HealthResponse, CheckResult defined
# aneos_api/schemas/impact.py    — ImpactResponse defined

grep -l "DetectionResponse\|HealthResponse\|ImpactResponse" aneos_api/endpoints/*.py
# → 0 results  (no endpoint uses these as response_model)
```

G-012 criterion "no endpoint imports from schemas/" is now met at package level, but the
actual intent — "schemas appear in the OpenAPI spec and endpoints return typed responses" —
is not met.

**Remaining gap**:
```
aneos_api/schemas/detection.py  — DetectionResponse.sigma_level
                                   does not match Sigma5DetectionResult.sigma_confidence
aneos_api/schemas/health.py     — HealthResponse not used as response_model in /health
aneos_api/schemas/impact.py     — ImpactResponse not used as response_model in impact endpoints
```

Note also: `DetectionResponse` has a `sigma_level: float` field, but the canonical result
from `ValidatedSigma5ArtificialNEODetector.analyze_neo_validated()` returns
`sigma_confidence: float`. This field mismatch means even if the schema were wired, it
would not accurately represent the detection output.

**Fix**:
1. Fix `DetectionResponse.sigma_level` → `sigma_confidence` to match canonical result.
2. Add `response_model=DetectionResponse` to at least one detection endpoint.
3. Add `response_model=HealthResponse` to the `/health` endpoint (adapt the response
   or expand `HealthResponse` to cover the existing rich health dict).
4. After wiring, run `make spec` and verify these types appear in `components/schemas`.
5. Either wire `ImpactResponse` or delete it (dead code has a deletion cost of zero).

**ADR/DDD ref**: ADR-034, DDD BC-8

---

### G-039 (NEW): `neodys.py._make_request()` Ignores `self.base_url`

**Area**: Data Acquisition / Configuration
**Status**: NEW (partial completion of G-030/G-021).

Phase 6B.3 set `self.base_url = config.neodys_url` in `NEODySSource.__init__` to make the
base URL configurable. However, `_make_request()` (the method actually used for REST
queries) hardcodes its own URL:

```python
# aneos_core/data/sources/neodys.py:235
url = f"https://newton.spacedys.com/neodys2/objects/{clean}/orbits/nominal"
```

This URL ignores `self.base_url` entirely. The only method that uses `self.base_url` is
`_fetch_eq0()` (line 118: `url = f"{self.base_url}{number}.eq0"`).

The two methods use different base URLs:
- `_fetch_eq0`: `https://newton.spacedys.com/~neodys2/epoch/{number}.eq0` (reads `self.base_url`)
- `_make_request`: `https://newton.spacedys.com/neodys2/objects/{id}/orbits/nominal` (hardcoded)

The first URL is known to work for numbered asteroids. The second URL is an unconfirmed
REST endpoint. `config.neodys_url` = `"https://newton.spacedys.com/~neodys2/epoch/"` does
not even match the base of the `_make_request` URL. The Phase 6 goal of "single source
of truth via config" was achieved for `_fetch_eq0` but not for `_make_request`.

**Impact**: The `config.neodys_url` field cannot be used to override the REST endpoint
URL in tests or staged deployments. The two methods are silently inconsistent.

**Fix**:
1. Either unify the two methods under one configurable base URL, or add a separate
   `neodys_rest_url` field to `APIConfig`.
2. Alternatively: given that `_make_request`'s REST endpoint is unconfirmed live,
   retire `_make_request` in favor of the confirmed `_fetch_eq0` path, and expose
   `_fetch_eq0` as the primary public method.
3. Update `test_data_sources_network.py` to test `_fetch_eq0` rather than `_make_request`
   (G-033 fix).

**ADR/DDD ref**: ADR-001, DDD BC-1

---

## P3 — Low Severity Gaps

### G-015 (DEFERRED): ML Classifier Has No Activation Path

**Area**: ML / Future Readiness
**Status**: DEFERRED (unchanged from v4.0). 9-object artificial corpus exists.
`FeatureVector` exists as a dataclass but `from_ground_truth_object()` is not implemented.
`TrainingPipeline.train_models()` exists but requires `AnalysisPipeline` and `NEOData`
objects — no bridge to `GroundTruthObject` exists. `MLDetectorWrapper` not registered
in `DetectionManager`.

**Fix**: Implement `FeatureVector.from_ground_truth_object(obj: GroundTruthObject)` in
`ml/features.py`; build a training script that runs `TrainingPipeline.train_models([...])`
using ground truth data; register `MLDetectorWrapper` at priority 5 in `DetectionManager`.

**ADR/DDD ref**: ADR-033

---

### G-023 (DEFERRED): `aneos_menu.py` Monolith Is Untestable

**Area**: Engineering Quality / Maintainability
**Status**: DEFERRED (unchanged from v4.0). ~11,500 lines. 0 unit tests.

**Fix**: Standalone phase — extract `detection_ui.py`, `analysis_ui.py`, `reporting_ui.py`,
`data_ui.py` into `aneos_core/ui/`. Each module gets its own test file.

**ADR/DDD ref**: DDD BC-8

---

### G-031 (PARTIAL RESIDUAL): Auth Mock DB Still in-Memory; JWT Still No-op

**Area**: API Security / Deployment Readiness
**Status**: Phase 6D.1 added a startup guard that raises `RuntimeError` in non-development
deployments when API key environment variables are unconfigured. This is a useful safety net.

**Remaining**: The underlying `MOCK_USERS` in-memory database with fake password hashes
(`'mock_admin_hash'`) still exists. JWT validation is still a comment:
```python
# Mock token authentication (would validate JWT in production)
if token == "mock_admin_token":
    return MOCK_USERS['admin']
```

This is not exploitable in the current CLI-only deployment but remains deployment-blocking
for any publicly accessible API scenario. The startup guard is the correct first step.

**Next step**: Replace the mock with env-var-driven API key validation only (drop JWT
for now, add it when a real user database is introduced). The startup guard already enforces
env var presence.

**ADR/DDD ref**: ADR-034

---

### G-032 (PARTIAL RESIDUAL): Export Endpoint Returns Placeholder Data

**Area**: API Functional Completeness
**Status**: Phase 6D.2 marked the endpoint `deprecated=True` in the router decorator.
This makes the deprecation visible in the OpenAPI spec.

**Remaining**: The endpoint still returns `b"Mock export data"` and the analysis export
feature is entirely non-functional. Marking it deprecated is the correct signal but
does not help callers who need the feature.

**Next step**: Implement export of JSON/CSV from the analysis result cache, or remove the
endpoint entirely from the spec using `include_in_schema=False` until implemented.

**ADR/DDD ref**: ADR-034, DDD BC-8

---

## Phase 6 Residual Verification

The following Phase 6 changes introduced sub-issues tracked in this document:

| Change | Location | Sub-issue | Gap |
|--------|----------|-----------|-----|
| `self.base_url = config.neodys_url` | `neodys.py:41` | Network test passes `{}` as config | G-033 |
| Remove `OrbitalElements.diameter/albedo/...` | `models.py:50-53` | SBDB dead writes to removed fields | G-034 |
| Remove `OrbitalElements.albedo` | `models.py:51` | MPC albedo lost (no `_physical` sub-dict) | G-035 |
| Regex `-?` prefix added | `ground_truth_dataset_preparation.py:237` | OrbitalElements still rejects negative SMA | G-019 (deepened) |
| `import aneos_api.schemas` in app.py | `app.py:78` | FastAPI does not auto-register unrouted models | G-012 (deepened) |

---

## Gap Summary Table (v5.0)

| ID | Title | Priority | Status |
|----|-------|----------|--------|
| G-001 | No ground truth dataset | ~~P0~~ | **CLOSED** |
| G-002 | Physical indicator category missing | ~~P1~~ | **CLOSED** |
| G-003 | Pipeline bypasses canonical detector | ~~P1~~ | **CLOSED** |
| G-004 | Silent startup soft-degrade | ~~P1~~ | **CLOSED** |
| G-005 | Docker deployment broken | ~~P1~~ | **CLOSED** |
| G-006 | Monitoring depends on deferred ML | ~~P1~~ | **CLOSED** |
| G-007 | README overstates production readiness | ~~P1~~ | **CLOSED** |
| G-008 | Seven detector files, no archive | ~~P2~~ | **CLOSED** |
| G-009 | Dual scoring + EMERGENCY suppressions (scoring) | ~~P2~~ | **CLOSED** |
| G-010 | No CI/CD pipeline | ~~P2~~ | **CLOSED** |
| G-011 | OrbitalElements conflates domain models | ~~P2~~ | **CLOSED** (Phase 6C) |
| G-012 | API has no DTO layer | ~~P2~~ | **CLOSED** (Phase 7C) |
| G-013 | Test harness inside production module | ~~P2~~ | **CLOSED** |
| G-014 | No maintained OpenAPI spec | ~~P2~~ | **CLOSED** |
| G-015 | ML has no activation path | **P3** | DEFERRED |
| G-016 | Redis integration unverified | ~~P3~~ | **CLOSED** |
| G-017 | CacheManager pickle security risk | ~~P3~~ | **CLOSED** |
| G-018 | Chunk boundary overlap unvalidated | ~~P3~~ | **CLOSED** |
| G-019 | Horizons spacecraft corpus / parser | ~~P1~~ | **CLOSED** (Phase 7B) |
| G-020 | Bayesian posterior ceiling undocumented | ~~P1~~ | **CLOSED** |
| G-021 | Multi-source enrichment SBDB-only | ~~P2~~ | **CLOSED** (Phase 6B) |
| G-022 | EMERGENCY suppressions mask scoring | ~~P2~~ | **CLOSED** |
| G-023 | `aneos_menu.py` monolith untestable | **P3** | DEFERRED |
| G-024 | Temporal analysis limited to single epoch | ~~P2~~ | **CLOSED** |
| G-025 | AI annotator may overstate confidence | ~~P3~~ | **CLOSED** |
| G-026 | Pickle in ML persistence layer | ~~P2~~ | **CLOSED** |
| G-027 | CI spec drift check workflow-hostile | ~~P2~~ | **CLOSED** |
| G-028 | EMERGENCY suppressions in validation/analysis | ~~P2~~ | **CLOSED** (Phase 6B) |
| G-029 | Hardcoded dev path + archive import | ~~P2~~ | **CLOSED** (Phase 6B) |
| G-030 | `APIConfig.neodys_url` phantom URL | ~~P2~~ | **CLOSED** (Phase 6B) |
| G-031 | API auth mock user DB | **P3** | PARTIAL — startup guard added; JWT still mock |
| G-032 | Export endpoint returns stub data | **P3** | PARTIAL — marked `deprecated=True`; still returns placeholder |
| G-033 | Network tests broken by Phase 6 config change | ~~P1~~ | **CLOSED** (Phase 7A) |
| G-034 | SBDB dead writes to removed OrbitalElements fields | ~~P2~~ | **CLOSED** (Phase 7A) |
| G-035 | MPC albedo silently lost | ~~P2~~ | **CLOSED** (Phase 7A) |
| G-036 | CI runs network tests — flaky pipeline | ~~P2~~ | **CLOSED** (Phase 7A) |
| G-037 | `ml/models.py` uses `logger` before definition | ~~P2~~ | **CLOSED** (Phase 7A) |
| G-039 | `neodys.py._make_request()` ignores `self.base_url` | ~~P3~~ | **CLOSED** (Phase 7C) |

**Totals v5.0**: 0 P0 · 2 P1 · 6 P2 · 5 P3 (2 deferred, 2 partial, 1 new) · 27 CLOSED

**Totals v6.0**: 0 P0 · 0 P1 · 0 P2 · 4 P3 (2 deferred, 2 partial) · 35 CLOSED

---

## Comparison v4.0 → v5.0

| Category | v4.0 | v5.0 | v6.0 | Delta (v5→v6) |
|----------|------|------|------|---------------|
| P0 gaps | 0 | 0 | 0 | — |
| P1 gaps | 1 | 2 | 0 | -2 (G-019, G-033 closed) |
| P2 gaps | 7 | 6 | 0 | -6 (G-012, G-034, G-035, G-036, G-037 closed) |
| P3 gaps | 3 | 5 | 4 | -1 (G-039 closed; G-015, G-023, G-031, G-032 remain) |
| CLOSED | 21 | 27 | 35 | +8 |
| Test suite | 57 pass | 58 pass | 59 pass | +1 test |

Phase 6 closed 6 outright gaps (G-028, G-029, G-030, G-021, G-011, G-019 partially).
Phase 6 opened 4 regressions (G-033, G-034, G-035 from G-011 removal; G-019 deepened).

Phase 7 closed all 9 non-deferred gaps in 3 sub-phases (7A regression fixes, 7B
hyperbolic elements, 7C API schema + NEODyS URL). 0 new regressions introduced.

---

## Phase 7 Execution Summary (COMPLETE)

All 9 non-deferred Phase 7 gaps closed. 0 new regressions.

```
Phase 7A — Regression Fixes (COMPLETE)  ✓ G-033 ✓ G-034 ✓ G-035 ✓ G-036 ✓ G-037
Phase 7B — G-019 Full Closure (COMPLETE)  ✓ G-019
Phase 7C — API Schema + NEODyS URL (COMPLETE)  ✓ G-012 ✓ G-039
```

## Remaining Work (Phase 8 candidates)

```
Phase 8 — Future Capabilities (P3, deferred)
  G-015  ML classifier activation path (FeatureVector.from_ground_truth_object + training pipeline).
  G-023  aneos_menu.py decomposition (extract ui/*.py modules; add unit tests).
  G-031  Replace JWT mock with env-var-only API key auth (startup guard already in place).
  G-032  Implement export or remove with include_in_schema=False.
```

---

*Next step: User Expectation / Logic / Outcome analysis — what should the application deliver
from a user-facing perspective? This informs Phase 8 prioritization.*
